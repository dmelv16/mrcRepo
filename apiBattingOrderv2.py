import requests
import json
import time
import pandas as pd
from datetime import datetime
import multiprocessing as mp
from functools import partial
import logging
import os
from sqlalchemy import create_engine, Column, Integer, String, Date, Float, Boolean, Text, MetaData, Table, ForeignKey, text, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import select, and_, or_, func
import sqlalchemy as sa

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mlb_batting_order_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mlb_batting_order_scraper")

# Constants
BASE_URL = "https://statsapi.mlb.com/api/v1"
DB_CONNECTION_STRING = "mssql+pyodbc://DESKTOP-J9IV3OH/StatcastDB?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
CHUNK_SIZE = 50  # Number of games to process per process
RATE_LIMIT = 0.5  # seconds between API calls to avoid rate limiting

# SQLAlchemy setup
Base = declarative_base()

class GameLineup(Base):
    __tablename__ = 'battingOrder'

    id = Column(Integer, primary_key=True, autoincrement=True)
    game_pk = Column(Integer, index=True)
    game_date = Column(Date)
    team_id = Column(Integer)
    team_name = Column(String(100))
    team_type = Column(String(10))  # home or away
    batting_order = Column(Integer)
    player_id = Column(Integer)
    player_name = Column(String(100))
    position = Column(String(10))
    lineup_spot = Column(Integer)
    is_starting_pitcher = Column(Integer, default=0)
    scraped_at = Column(sa.DateTime, default=func.now())

class ScrapeStatus(Base):
    __tablename__ = 'scrape_status_battingOrder'

    game_pk = Column(Integer, primary_key=True)
    processed = Column(Integer, default=0)
    last_attempt = Column(sa.DateTime)
    error_message = Column(Text)

def create_engine_and_session(connection_string):
    """Create SQLAlchemy engine and session"""
    engine = create_engine(connection_string)
    Session = sessionmaker(bind=engine)
    return engine, Session

def create_tables(engine):
    """Create necessary tables if they don't exist"""
    inspector = inspect(engine)
    if not inspector.has_table('battingOrder'):
        GameLineup.__table__.create(engine)
    
    if not inspector.has_table('scrape_status_battingOrder'):
        ScrapeStatus.__table__.create(engine)

def get_pending_games(session):
    """Get list of games that need batting order data"""
    # Query for games from baseballScrapev2 that have roster data but no batting order data
    sql = text("""
    WITH GameDates AS (
        -- Get unique game dates and teams from baseballScrapev2
        SELECT DISTINCT 
            g.gamePk,
            g.team_id,
            CAST(g.game_date AS DATE) as game_date
        FROM baseballScrapev2 g
        WHERE g.gamePk IS NOT NULL
    ),
    RosterGames AS (
        -- Find games where we have roster data
        SELECT DISTINCT
            gd.gamePk,
            gd.game_date
        FROM GameDates gd
        INNER JOIN roster_data r 
            ON gd.team_id = r.team_id 
            AND gd.game_date = CAST(r.date AS DATE)
    )
    -- Select games that haven't been processed for batting order
    SELECT DISTINCT 
        rg.gamePk, 
        rg.game_date
    FROM RosterGames rg
    LEFT JOIN scrape_status_battingOrder s ON rg.gamePk = s.game_pk
    WHERE (s.processed IS NULL OR s.processed = 0)
    ORDER BY rg.game_date
    """)
    
    result = session.execute(sql)
    return [(row[0], row[1]) for row in result]

def fetch_with_retries(url):
    """Fetch data from API with retries"""
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(RATE_LIMIT)  # Rate limiting
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt+1} failed for {url}: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"Failed to fetch {url} after {MAX_RETRIES} attempts")
                raise

def get_roster_players(session, team_id, game_date):
    """Get roster players for a specific team and date"""
    sql = text("""
    SELECT player_id, full_name, position
    FROM roster_data
    WHERE team_id = :team_id 
    AND CAST(date AS DATE) = CAST(:game_date AS DATE)
    """)
    
    result = session.execute(sql, {'team_id': team_id, 'game_date': game_date})
    return {row[0]: {'name': row[1], 'position': row[2]} for row in result}

def get_game_data(game_pk, game_date, session):
    """Get lineup and starting pitcher data for a specific game"""
    try:
        # Get boxscore data which contains lineup information
        boxscore_url = f"{BASE_URL}/game/{game_pk}/boxscore"
        boxscore_data = fetch_with_retries(boxscore_url)
        
        # Extract team data
        teams_data = boxscore_data.get('teams', {})
        if not teams_data:
            logger.warning(f"No teams data found for game {game_pk}")
            return None
        
        lineup_data = []
        
        # Process each team (home and away)
        for team_type in ['home', 'away']:
            team_info = teams_data.get(team_type, {})
            team_id = team_info.get('team', {}).get('id')
            team_name = team_info.get('team', {}).get('name')
            
            if not team_id:
                logger.warning(f"No team ID found for {team_type} team in game {game_pk}")
                continue
            
            # Get roster players for this team and date
            roster_players = get_roster_players(session, team_id, game_date)
            
            # Get batting order
            batting_order = team_info.get('batters', [])
            
            # Get players info from boxscore
            players = team_info.get('players', {})
            if not players:
                logger.warning(f"No players found for {team_type} team in game {game_pk}")
                continue

            # Filter out substitutes
            batting_order = [
                player_id for player_id in batting_order
                if not players.get(f'ID{player_id}', {}).get('gameStatus', {}).get('isSubstitute', False)
            ]
            
            # Find starting pitcher
            starting_pitcher_id = None
            for player_id, player_data in players.items():
                positions = player_data.get('allPositions', [])
                is_pitcher = any(pos.get('code') == '1' for pos in positions)
                games_started = player_data.get('stats', {}).get('pitching', {}).get('gamesStarted', 0)
                is_sub = player_data.get('gameStatus', {}).get('isSubstitute', True)

                if is_pitcher and not is_sub and games_started == 1:
                    starting_pitcher_id = player_data.get('person', {}).get('id')
                    break

            # Process lineup
            for position, player_id in enumerate(batting_order):
                player_key = f"ID{player_id}" if not str(player_id).startswith("ID") else player_id
                
                if player_key not in players:
                    logger.warning(f"Player {player_key} not found in players data for game {game_pk}")
                    continue
                
                player_data = players[player_key]
                player_id_num = player_data.get('person', {}).get('id')
                
                # Check if player is in roster data
                if player_id_num not in roster_players:
                    logger.warning(f"Player {player_id_num} not found in roster_data for team {team_id} on {game_date}")
                    # Use data from API if not in roster
                    player_name = player_data.get('person', {}).get('fullName')
                    player_position = player_data.get('position', {}).get('abbreviation')
                else:
                    # Use roster data
                    player_name = roster_players[player_id_num]['name']
                    player_position = roster_players[player_id_num]['position']
                
                # Determine if this player is the starting pitcher
                is_starting_pitcher = 1 if player_id_num == starting_pitcher_id else 0
                
                lineup_data.append({
                    'game_pk': game_pk,
                    'game_date': game_date,
                    'team_id': team_id,
                    'team_name': team_name,
                    'team_type': team_type,
                    'batting_order': position + 1,  # 1-indexed batting order
                    'player_id': player_id_num,
                    'player_name': player_name,
                    'position': player_position,
                    'lineup_spot': position + 1,
                    'is_starting_pitcher': is_starting_pitcher
                })
            
            # Add starting pitcher if not in batting order (AL games)
            if starting_pitcher_id and not any(p['player_id'] == starting_pitcher_id for p in lineup_data if p['team_id'] == team_id):
                # Check roster data first
                if starting_pitcher_id in roster_players:
                    player_name = roster_players[starting_pitcher_id]['name']
                else:
                    # Fall back to API data
                    for player_id, player_data in players.items():
                        if player_data.get('person', {}).get('id') == starting_pitcher_id:
                            player_name = player_data.get('person', {}).get('fullName')
                            break
                
                lineup_data.append({
                    'game_pk': game_pk,
                    'game_date': game_date,
                    'team_id': team_id,
                    'team_name': team_name,
                    'team_type': team_type,
                    'batting_order': 0,  # 0 for pitchers not in batting order
                    'player_id': starting_pitcher_id,
                    'player_name': player_name,
                    'position': 'P',
                    'lineup_spot': 0,
                    'is_starting_pitcher': 1
                })
        
        return lineup_data
    
    except Exception as e:
        logger.error(f"Error processing game {game_pk}: {str(e)}")
        return None

def save_lineup_data(session, lineup_data):
    """Save lineup data to database"""
    if not lineup_data:
        return False
    
    try:
        # Check if data already exists
        game_pk = lineup_data[0]['game_pk']
        existing_entries = session.query(GameLineup).filter(GameLineup.game_pk == game_pk).count()
        
        if existing_entries > 0:
            # Delete existing data if necessary
            session.query(GameLineup).filter(GameLineup.game_pk == game_pk).delete()
        
        # Insert all lineup data
        for player in lineup_data:
            lineup_entry = GameLineup(
                game_pk=player['game_pk'],
                game_date=player['game_date'],
                team_id=player['team_id'],
                team_name=player['team_name'],
                team_type=player['team_type'],
                batting_order=player['batting_order'],
                player_id=player['player_id'],
                player_name=player['player_name'],
                position=player['position'],
                lineup_spot=player['lineup_spot'],
                is_starting_pitcher=player['is_starting_pitcher']
            )
            session.add(lineup_entry)
        
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"Database error for game {lineup_data[0]['game_pk']}: {str(e)}")
        return False

def update_status(session, game_pk, processed, error_message=None):
    """Update the status of a game in the status table"""
    try:
        # Check if status entry exists
        status_entry = session.query(ScrapeStatus).filter(ScrapeStatus.game_pk == game_pk).first()
        
        if status_entry:
            # Update existing entry
            status_entry.processed = processed
            status_entry.last_attempt = datetime.now()
            status_entry.error_message = error_message
        else:
            # Create new entry
            status_entry = ScrapeStatus(
                game_pk=game_pk,
                processed=processed,
                last_attempt=datetime.now(),
                error_message=error_message
            )
            session.add(status_entry)
        
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to update status for game {game_pk}: {str(e)}")

def process_game(game_pk, game_date, connection_string):
    """Process a single game - to be run in a separate process"""
    try:
        # Create engine and session for this process
        engine, Session = create_engine_and_session(connection_string)
        session = Session()
        
        # Get and process data
        lineup_data = get_game_data(game_pk, game_date, session)
        
        if lineup_data:
            success = save_lineup_data(session, lineup_data)
            if success:
                update_status(session, game_pk, 1)
                logger.info(f"Successfully processed game {game_pk}")
                session.close()
                return game_pk, True
            else:
                update_status(session, game_pk, 0, "Failed to save lineup data")
                logger.error(f"Failed to save lineup data for game {game_pk}")
                session.close()
                return game_pk, False
        else:
            update_status(session, game_pk, 0, "No lineup data retrieved")
            logger.warning(f"No lineup data retrieved for game {game_pk}")
            session.close()
            return game_pk, False
    except Exception as e:
        logger.error(f"Error processing game {game_pk}: {str(e)}")
        try:
            update_status(session, game_pk, 0, str(e))
            session.close()
        except:
            pass
        return game_pk, False

def process_games_chunk(game_pks, connection_string):
    """Process a chunk of games"""
    results = []
    for game_pk, game_date in game_pks:
        result = process_game(game_pk, game_date, connection_string)
        results.append(result)
    return results

def main():
    logger.info("Starting MLB batting order scraper using roster_data")
    
    # Create SQLAlchemy engine and session
    engine, Session = create_engine_and_session(DB_CONNECTION_STRING)
    session = Session()
    
    try:
        # Create tables if they don't exist
        create_tables(engine)
        
        # Get list of games to process
        pending_games = get_pending_games(session)
        total_games = len(pending_games)
        logger.info(f"Found {total_games} games to process")
        
        if total_games == 0:
            logger.info("No pending games to process")
            return
        
        # Show summary of date range
        if pending_games:
            first_date = min(game[1] for game in pending_games)
            last_date = max(game[1] for game in pending_games)
            logger.info(f"Processing games from {first_date} to {last_date}")
        
        # Split games into chunks for multiprocessing
        chunks = [pending_games[i:i + CHUNK_SIZE] for i in range(0, len(pending_games), CHUNK_SIZE)]
        
        # Determine number of processes
        num_processes = max(1, mp.cpu_count() - 1)
        logger.info(f"Using {num_processes} processes")
        
        # Process chunks in parallel
        with mp.Pool(processes=num_processes) as pool:
            process_func = partial(process_games_chunk, connection_string=DB_CONNECTION_STRING)
            results = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} games)")
                chunk_results = pool.apply_async(process_func, (chunk,))
                results.append(chunk_results)
            
            # Collect results
            all_results = []
            for result in results:
                all_results.extend(result.get())
            
            # Summarize results
            successes = sum(1 for _, success in all_results if success)
            failures = sum(1 for _, success in all_results if not success)
            
            logger.info(f"Completed processing. Successes: {successes}, Failures: {failures}")
    
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
    
    finally:
        session.close()
        logger.info("MLB batting order scraper completed")

if __name__ == "__main__":
    main()