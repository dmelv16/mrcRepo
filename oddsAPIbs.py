import requests
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
import urllib.parse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CONFIG
API_KEY = '17b099f6c67411f3eb2c00fc4e99032f'
SPORT = 'baseball_mlb'
REGIONS = 'us'
MARKETS = 'h2h,totals'
ODDS_FORMAT = 'decimal'
TARGET_SPORTSBOOK = "DraftKings"  # Change this to your preferred sportsbook

def create_db_connection():
    """Create database connection"""
    try:
        params = urllib.parse.quote_plus(
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=DESKTOP-J9IV3OH;"
            "DATABASE=StatcastDB;"
            "UID=mlb_user;"
            "PWD=mlbAdmin;"
            "Encrypt=no;"
            "TrustServerCertificate=yes;"
        )
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
        logger.info("Successfully connected to SQL Server")
        return engine
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

def is_mlb_season(date_obj):
    """Check if date falls within MLB season"""
    month = date_obj.month
    day = date_obj.day
    
    # MLB season: March 15 through October
    if month >= 4 and month <= 10:
        return True
    elif month == 3 and day >= 15:
        return True
    else:
        return False

def check_existing_data(engine, date_str):
    """Check if we already have data for this date and sportsbook"""
    try:
        query = text("""
            SELECT COUNT(*) as count 
            FROM mlb_odds_history 
            WHERE CAST(snapshot_time AS DATE) = :date_param 
            AND bookmaker = :sportsbook
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query, {
                'date_param': date_str, 
                'sportsbook': TARGET_SPORTSBOOK
            })
            count = result.scalar()
            
        if count > 0:
            logger.info(f"📊 Found {count} existing {TARGET_SPORTSBOOK} records for {date_str}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error checking existing data: {e}")
        return False

def fetch_current_odds():
    """Fetch current live odds (not historical)"""
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
    params = {
        'apiKey': API_KEY,
        'regions': REGIONS,
        'markets': MARKETS,
        'oddsFormat': ODDS_FORMAT,
        'dateFormat': 'iso'
    }
    
    try:
        logger.info("Fetching current live odds...")
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Found {len(data)} current games")
            return data
        else:
            logger.warning(f"❌ Error fetching current odds: Status {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"Error fetching current odds: {e}")
        return []

def process_odds_data(data, current_time):
    """Process current odds data into database format"""
    all_data = []
    
    for event in data:
        for bookmaker in event.get("bookmakers", []):
            # Only process our target sportsbook
            if bookmaker.get("title") != TARGET_SPORTSBOOK:
                continue
                
            for market in bookmaker.get("markets", []):
                for outcome in market.get("outcomes", []):
                    row = {
                        'snapshot_time': current_time.strftime("%Y-%m-%dT%H:%M:%S"),
                        'game_id': event.get("id"),
                        'home_team': event.get("home_team"),
                        'away_team': event.get("away_team"),
                        'commence_time': event.get("commence_time"),
                        'bookmaker': bookmaker.get("title"),
                        'market': market.get("key"),
                        'outcome': outcome.get("name"),
                        'odds': outcome.get("price"),
                        'point': outcome.get("point") if 'point' in outcome else None
                    }
                    all_data.append(row)
    
    return all_data

def main():
    """Main function - process today's date only"""
    current_time = datetime.now()
    today = current_time.strftime("%Y-%m-%d")
    
    logger.info(f"🚀 Starting daily MLB odds update for {today}")
    logger.info(f"🎯 Target Sportsbook: {TARGET_SPORTSBOOK}")
    logger.info(f"📊 Markets: {MARKETS}")
    
    # Check if it's MLB season
    if not is_mlb_season(current_time):
        logger.info(f"⚾ {today} is outside MLB season (March 15 - October). Skipping.")
        return
    
    # Connect to database
    engine = create_db_connection()
    if not engine:
        logger.error("Failed to connect to database")
        return
    
    # Check if we already have data for today
    if check_existing_data(engine, today):
        user_input = input(f"Data for {today} already exists. Overwrite? (y/n): ").lower()
        if user_input != 'y':
            logger.info("Skipping update.")
            return
        else:
            # Delete existing data for today
            try:
                delete_query = text("""
                    DELETE FROM mlb_odds_history 
                    WHERE CAST(snapshot_time AS DATE) = :date_param 
                    AND bookmaker = :sportsbook
                """)
                with engine.connect() as conn:
                    result = conn.execute(delete_query, {
                        'date_param': today, 
                        'sportsbook': TARGET_SPORTSBOOK
                    })
                    conn.commit()
                logger.info(f"🗑️ Deleted existing {TARGET_SPORTSBOOK} data for {today}")
            except Exception as e:
                logger.error(f"Error deleting existing data: {e}")
                return
    
    try:
        # Fetch current odds
        data = fetch_current_odds()
        
        if data:
            # Process data
            processed_data = process_odds_data(data, current_time)
            
            if processed_data:
                # Save to database
                df = pd.DataFrame(processed_data)
                df.to_sql('mlb_odds_history', con=engine, if_exists='append', index=False)
                
                logger.info(f"💾 Saved {len(df)} {TARGET_SPORTSBOOK} records for {today}")
                
                # Show summary of games
                games = df.groupby(['home_team', 'away_team']).size().reset_index(name='records')
                logger.info(f"🎮 Games processed:")
                for _, game in games.iterrows():
                    logger.info(f"   {game['away_team']} @ {game['home_team']} ({game['records']} records)")
                    
            else:
                logger.info(f"📊 No {TARGET_SPORTSBOOK} data found for {today}")
        else:
            logger.info(f"❌ No games found for {today}")
    
    except Exception as e:
        logger.error(f"💥 Error during processing: {e}")
    
    logger.info(f"🏁 Daily update complete for {today}")

if __name__ == "__main__":
    main()