import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine
import urllib.parse
import logging
import pytz

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database connection
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

def load_team_mappings(engine):
    """Load team mappings from SQL table - handles duplicates by prioritizing current names"""
    query = """
    WITH RankedTeams AS (
        SELECT 
            abbrev,
            team_id, 
            full_name,
            ROW_NUMBER() OVER (
                PARTITION BY team_id 
                ORDER BY 
                    CASE 
                        WHEN full_name = 'Cleveland Guardians' THEN 1
                        WHEN full_name = 'Oakland Athletics' THEN 1
                        WHEN full_name = 'Cleveland Indians' THEN 2
                        WHEN full_name = 'Athletics' THEN 2
                        ELSE 0
                    END
            ) as rn
        FROM team_abbrev_map
    )
    SELECT abbrev, team_id, full_name
    FROM RankedTeams
    WHERE rn = 1
    """
    
    try:
        df = pd.read_sql(query, engine)
        logger.info(f"📋 Loaded {len(df)} team mappings (duplicates filtered)")
        
        # Check for teams that had duplicates
        dup_check_query = """
        SELECT team_id, COUNT(*) as count, STRING_AGG(full_name, ', ') as all_names
        FROM team_abbrev_map
        GROUP BY team_id
        HAVING COUNT(*) > 1
        """
        dup_df = pd.read_sql(dup_check_query, engine)
        if not dup_df.empty:
            logger.info(f"⚠️  Found {len(dup_df)} teams with multiple entries:")
            for _, row in dup_df.iterrows():
                logger.info(f"   Team {row['team_id']}: {row['all_names']}")
        
        # Create mapping dictionaries
        full_name_to_abbrev = dict(zip(df['full_name'], df['abbrev']))
        team_id_to_abbrev = dict(zip(df['team_id'], df['abbrev']))
        
        return full_name_to_abbrev, team_id_to_abbrev
    except Exception as e:
        logger.error(f"Error loading team mappings: {e}")
        return None, None

def load_odds_data(engine):
    """Load odds data from SQL table"""
    query = """
    SELECT 
        game_id,
        home_team,
        away_team,
        commence_time,
        bookmaker,
        market,
        outcome,
        odds,
        point
    FROM mlb_odds_history
    ORDER BY commence_time, game_id
    """
    
    try:
        df = pd.read_sql(query, engine)
        logger.info(f"📊 Loaded {len(df)} odds records")
        
        # Convert commence_time from UTC to Pacific
        df['commence_time'] = pd.to_datetime(df['commence_time'])
        
        # Create timezone objects
        utc = pytz.UTC
        pacific = pytz.timezone('US/Pacific')
        
        # Convert UTC to Pacific time
        df['commence_time_pacific'] = df['commence_time'].dt.tz_localize(utc).dt.tz_convert(pacific)
        
        # Extract date in Pacific time
        df['game_date'] = df['commence_time_pacific'].dt.date
        
        logger.info(f"🕐 Converted times from UTC to Pacific timezone")
        
        return df
    except Exception as e:
        logger.error(f"Error loading odds data: {e}")
        return None

def load_parquet_data(parquet_path, team_id_to_abbrev):
    """Load master MLB parquet data"""
    try:
        df = pd.read_parquet(parquet_path)
        logger.info(f"📈 Loaded {len(df)} parquet records")
        
        # Debug: Check what identifier columns exist
        id_columns = [col for col in df.columns if 'game' in col.lower() and ('id' in col.lower() or 'pk' in col.lower())]
        logger.info(f"🔍 Found game identifier columns: {id_columns}")
        
        # Check for duplicates based on game_id (or game_pk if that's what exists)
        id_column = 'game_id' if 'game_id' in df.columns else 'game_pk' if 'game_pk' in df.columns else None
        if id_column:
            duplicates = df[df.duplicated(subset=[id_column], keep=False)]
            if len(duplicates) > 0:
                logger.warning(f"⚠️  Found {len(duplicates)} duplicate records based on {id_column}")
                # Remove duplicates, keeping the first occurrence
                df = df.drop_duplicates(subset=[id_column], keep='first')
                logger.info(f"📋 After removing duplicates: {len(df)} records")
        
        # Convert game_date to datetime
        df['game_date'] = pd.to_datetime(df['game_date'])
        
        # Add team abbreviations using the mapping from SQL
        df['home_team_abbr'] = df['home_team_id'].map(team_id_to_abbrev)
        df['away_team_abbr'] = df['away_team_id'].map(team_id_to_abbrev)
        
        # Check for unmapped teams
        unmapped_home = df[df['home_team_abbr'].isna()]['home_team_id'].unique()
        unmapped_away = df[df['away_team_abbr'].isna()]['away_team_id'].unique()
        
        if len(unmapped_home) > 0 or len(unmapped_away) > 0:
            logger.warning(f"⚠️  Unmapped team IDs found: Home={unmapped_home}, Away={unmapped_away}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading parquet data: {e}")
        return None

def normalize_team_name(team_name, full_name_to_abbrev):
    """Convert API team name to standard abbreviation using SQL mapping"""
    return full_name_to_abbrev.get(team_name, team_name)

def match_games(odds_df, parquet_df, full_name_to_abbrev):
    """Match odds data to parquet data - handles doubleheaders"""
    logger.info("🔍 Starting game matching process...")
    
    # Determine which identifier columns exist
    parquet_id_cols = [col for col in parquet_df.columns if col in ['game_id', 'game_pk']]
    odds_id_cols = [col for col in odds_df.columns if col in ['game_id', 'game_pk']]
    
    logger.info(f"📌 Parquet ID columns: {parquet_id_cols}")
    logger.info(f"📌 Odds ID columns: {odds_id_cols}")
    
    # We need to match on teams and dates since the IDs don't match directly
    # The parquet has game_pk/game_id from one system, odds has game_id from another
    
    # Normalize team names in odds data using SQL mappings
    odds_df['home_team_norm'] = odds_df['home_team'].apply(
        lambda x: normalize_team_name(x, full_name_to_abbrev)
    )
    odds_df['away_team_norm'] = odds_df['away_team'].apply(
        lambda x: normalize_team_name(x, full_name_to_abbrev)
    )
    
    # Convert odds game_date to datetime for matching
    odds_df['game_date'] = pd.to_datetime(odds_df['game_date'])
    
    # Convert parquet commence_time to datetime and extract game time
    if 'game_time' in parquet_df.columns:
        parquet_df['commence_time'] = pd.to_datetime(parquet_df['game_time'], format='%H:%M', errors='coerce')
        parquet_df['game_hour'] = parquet_df['commence_time'].dt.hour
    
    # Use Pacific time for odds game hour
    odds_df['game_hour'] = odds_df['commence_time_pacific'].dt.hour
    
    # Check for potential doubleheaders in both datasets
    parquet_doubleheaders = parquet_df.groupby(['home_team_abbr', 'away_team_abbr', 'game_date']).size()
    parquet_doubleheaders = parquet_doubleheaders[parquet_doubleheaders > 1]
    
    odds_doubleheaders = odds_df.groupby(['home_team_norm', 'away_team_norm', 'game_date']).size()
    odds_doubleheaders = odds_doubleheaders[odds_doubleheaders > 1]
    
    if len(parquet_doubleheaders) > 0:
        logger.info(f"🎯 Found {len(parquet_doubleheaders)} potential doubleheaders in parquet data")
    
    if len(odds_doubleheaders) > 0:
        logger.info(f"🎯 Found {len(odds_doubleheaders)} potential doubleheaders in odds data")
    
    # Create match keys based on teams and date
    def create_match_key(row, home_col, away_col):
        return f"{row[home_col]}_{row[away_col]}_{row['game_date'].strftime('%Y-%m-%d')}"
    
    # Create match keys for both datasets
    odds_df['match_key'] = odds_df.apply(lambda row: create_match_key(row, 'home_team_norm', 'away_team_norm'), axis=1)
    parquet_df['match_key'] = parquet_df.apply(lambda row: create_match_key(row, 'home_team_abbr', 'away_team_abbr'), axis=1)
    
    # Perform the match
    matched_df = odds_df.merge(
        parquet_df,
        on='match_key',
        how='inner',
        suffixes=('_odds', '_parquet')
    )
    
    # Show unmapped teams
    unmapped_home = odds_df[~odds_df['home_team_norm'].isin(full_name_to_abbrev.values())]['home_team'].unique()
    unmapped_away = odds_df[~odds_df['away_team_norm'].isin(full_name_to_abbrev.values())]['away_team'].unique()
    
    if len(unmapped_home) > 0 or len(unmapped_away) > 0:
        logger.warning(f"⚠️  Unmapped odds teams: Home={unmapped_home}, Away={unmapped_away}")
    
    # Determine which ID column to use for the parquet data
    # Prefer game_id if it exists, otherwise use game_pk
    id_column = 'game_id' if 'game_id' in parquet_df.columns else 'game_pk'
    
    logger.info(f"✅ Successfully matched {len(matched_df)} total records")
    
    # Count unique games using the parquet's identifier (with suffix)
    id_col_in_matched = f'{id_column}_parquet' if f'{id_column}_parquet' in matched_df.columns else id_column
    logger.info(f"📊 Unique games matched: {matched_df[id_col_in_matched].nunique()}")
    
    # Check for potential doubleheader issues in final results
    final_duplicates = matched_df.groupby([id_col_in_matched]).size()
    final_duplicates = final_duplicates[final_duplicates > matched_df['market'].nunique() * matched_df['outcome'].nunique()]
    
    if len(final_duplicates) > 0:
        logger.warning(f"⚠️  Potential doubleheader matching issues found for {len(final_duplicates)} games")
        logger.warning(f"    Consider manual review of {id_column}s: {final_duplicates.index.tolist()[:10]}")  # Show first 10
    
    return matched_df, id_column

def pivot_odds_data(matched_df, id_column='game_id'):
    """Pivot odds data to one row per game with separate columns for each market"""
    logger.info("🔄 Pivoting odds data to one row per game...")
    
    # Debug: Check what columns we have after merge
    logger.info(f"Available columns after merge: {matched_df.columns.tolist()}")
    
    # The identifier column will have _parquet suffix after merge
    id_col_in_matched = f'{id_column}_parquet' if f'{id_column}_parquet' in matched_df.columns else id_column
    
    # Create a base dataframe with game info (take first record per game)
    game_base = matched_df.groupby(id_col_in_matched).first().reset_index()
    
    # Rename the id column to remove _parquet suffix for cleaner output
    game_base.rename(columns={id_col_in_matched: id_column}, inplace=True)
    
    # Extract the columns we want to keep from the base data
    base_columns = [id_column]
    
    # Add game_id_odds if it exists (for reference)
    if 'game_id_odds' in game_base.columns:
        base_columns.append('game_id_odds')
    
    # Add bookmaker
    if 'bookmaker_odds' in game_base.columns:
        base_columns.append('bookmaker_odds')
    elif 'bookmaker' in game_base.columns:
        base_columns.append('bookmaker')
    
    # Find the actual column names (they might have suffixes)
    for col_pattern in ['game_date', 'venue', 'game_time', 'dayNight', 
                       'temperature', 'wind_speed', 'wind_dir', 'conditions',
                       'home_team', 'away_team', 'home_team_id', 'away_team_id',
                       'home_score', 'away_score']:
        # Find columns that contain this pattern
        matching_cols = [col for col in game_base.columns if col_pattern in col]
        if matching_cols:
            # Prefer _parquet suffix if available
            parquet_cols = [col for col in matching_cols if '_parquet' in col]
            if parquet_cols:
                base_columns.append(parquet_cols[0])
            else:
                base_columns.append(matching_cols[0])
    
    # Keep only columns that exist
    base_columns = [col for col in base_columns if col in game_base.columns]
    game_info = game_base[base_columns].copy()
    
    # Rename columns to remove suffixes for cleaner output
    rename_dict = {}
    for col in game_info.columns:
        if '_parquet' in col and col != id_column:
            rename_dict[col] = col.replace('_parquet', '')
        elif '_odds' in col and col not in ['game_id_odds', 'bookmaker_odds']:
            rename_dict[col] = col.replace('_odds', '')
    if 'bookmaker_odds' in rename_dict:
        rename_dict['bookmaker_odds'] = 'bookmaker'
    game_info.rename(columns=rename_dict, inplace=True)
    
    # Now pivot the odds data
    odds_only = matched_df[[id_col_in_matched, 'market', 'outcome', 'odds', 'point']].copy()
    
    # Create separate dataframes for each market type
    h2h_data = odds_only[odds_only['market'] == 'h2h'].copy()
    totals_data = odds_only[odds_only['market'] == 'totals'].copy()
    
    # Process moneyline (h2h) data
    if not h2h_data.empty:
        # For h2h, we need to determine which team is home/away
        h2h_pivot = h2h_data.pivot_table(
            index=id_col_in_matched,
            columns='outcome',
            values='odds',
            aggfunc='first'
        ).reset_index()
        
        # Rename the index column
        h2h_pivot.rename(columns={id_col_in_matched: id_column}, inplace=True)
        
        # Get the actual team names for this game to map correctly
        # Find the home_team and away_team columns (might have suffixes)
        home_team_col = [col for col in matched_df.columns if 'home_team' in col and 'odds' in col]
        away_team_col = [col for col in matched_df.columns if 'away_team' in col and 'odds' in col]
        
        if home_team_col and away_team_col:
            team_mapping = matched_df.groupby(id_col_in_matched)[[home_team_col[0], away_team_col[0]]].first()
            
            # Create home_ml and away_ml columns
            h2h_final = pd.DataFrame()
            h2h_final[id_column] = h2h_pivot[id_column]
            
            for idx, row in h2h_pivot.iterrows():
                game_id = row[id_column]
                if game_id in team_mapping.index:
                    home_team = team_mapping.loc[game_id, home_team_col[0]]
                    away_team = team_mapping.loc[game_id, away_team_col[0]]
                    
                    # Map the odds to home/away based on team names
                    if home_team in h2h_pivot.columns:
                        h2h_final.loc[idx, 'home_ml'] = row[home_team]
                    if away_team in h2h_pivot.columns:
                        h2h_final.loc[idx, 'away_ml'] = row[away_team]
        else:
            # Fallback: just use the first two teams found
            h2h_final = h2h_pivot.copy()
            # Rename the first two non-id columns to home_ml and away_ml
            odds_cols = [col for col in h2h_pivot.columns if col != id_column]
            if len(odds_cols) >= 2:
                h2h_final.rename(columns={odds_cols[0]: 'home_ml', odds_cols[1]: 'away_ml'}, inplace=True)
                h2h_final = h2h_final[[id_column, 'home_ml', 'away_ml']]
            else:
                h2h_final = pd.DataFrame(columns=[id_column, 'home_ml', 'away_ml'])
    else:
        h2h_final = pd.DataFrame(columns=[id_column, 'home_ml', 'away_ml'])
    
    # Process totals data
    if not totals_data.empty:
        # For totals, we want the line (point) and both over/under odds
        totals_with_point = totals_data.copy()
        
        # Get the line value (should be same for over/under)
        totals_line = totals_with_point.groupby(id_column)['point'].first().reset_index()
        totals_line.columns = [id_column, 'total_line']
        
        # Pivot the over/under odds
        totals_pivot = totals_with_point.pivot_table(
            index=id_column,
            columns='outcome', 
            values='odds',
            aggfunc='first'
        ).reset_index()
        
        # Rename columns to be more descriptive
        totals_final = totals_pivot.copy()
        if 'Over' in totals_final.columns:
            totals_final['over_odds'] = totals_final['Over']
            totals_final.drop('Over', axis=1, inplace=True)
        if 'Under' in totals_final.columns:
            totals_final['under_odds'] = totals_final['Under']
            totals_final.drop('Under', axis=1, inplace=True)
            
        # Add the line
        totals_final = totals_final.merge(totals_line, on=id_column, how='left')
    else:
        totals_final = pd.DataFrame(columns=[id_column, 'over_odds', 'under_odds', 'total_line'])
    
    # Merge everything together
    final_df = game_info.copy()
    
    # Merge moneyline data
    if not h2h_final.empty:
        final_df = final_df.merge(h2h_final, on=id_column, how='left')
    else:
        final_df['home_ml'] = None
        final_df['away_ml'] = None
    
    # Merge totals data  
    if not totals_final.empty:
        final_df = final_df.merge(totals_final, on=id_column, how='left')
    else:
        final_df['over_odds'] = None
        final_df['under_odds'] = None
        final_df['total_line'] = None
    
    logger.info(f"✅ Pivoted to {len(final_df)} game rows")
    logger.info(f"📊 Columns: {final_df.columns.tolist()}")
    
    # Show summary of data availability
    ml_games = final_df[final_df['home_ml'].notna() & final_df['away_ml'].notna()]
    totals_games = final_df[final_df['total_line'].notna()]
    
    logger.info(f"💰 Games with moneyline odds: {len(ml_games)}")
    logger.info(f"🎯 Games with totals: {len(totals_games)}")
    
    return final_df

def append_odds_to_master_parquet(parquet_df, odds_columns_df, parquet_path, id_column='game_id'):
    """Append odds columns to the existing master parquet file"""
    logger.info("📝 Appending odds columns to master parquet file...")
    
    # First, clean up the parquet dataframe by removing any existing odds columns and _x/_y columns
    odds_columns = ['home_ml', 'away_ml', 'total_line', 'over_odds', 'under_odds', 'bookmaker', 'game_id_odds']
    
    # Remove any existing odds columns from parquet
    existing_odds_in_parquet = [col for col in parquet_df.columns if col in odds_columns]
    if existing_odds_in_parquet:
        logger.info(f"🧹 Removing existing odds columns from parquet: {existing_odds_in_parquet}")
        parquet_df = parquet_df.drop(columns=existing_odds_in_parquet)
    
    # Remove any columns with _x or _y suffixes (from previous failed merges)
    xy_columns = [col for col in parquet_df.columns if col.endswith('_x') or col.endswith('_y')]
    if xy_columns:
        logger.info(f"🧹 Removing columns with _x/_y suffixes: {xy_columns}")
        parquet_df = parquet_df.drop(columns=xy_columns)
    
    # Also clean up the odds dataframe
    xy_columns_odds = [col for col in odds_columns_df.columns if col.endswith('_x') or col.endswith('_y')]
    if xy_columns_odds:
        logger.info(f"🧹 Removing _x/_y columns from odds data: {xy_columns_odds}")
        odds_columns_df = odds_columns_df.drop(columns=xy_columns_odds)
    
    # Debug: Check game identifier overlap
    parquet_game_ids = set(parquet_df[id_column].unique())
    odds_game_ids = set(odds_columns_df[id_column].unique())
    common_game_ids = parquet_game_ids.intersection(odds_game_ids)
    
    logger.info(f"🔍 Debug Info:")
    logger.info(f"   Games in master parquet: {len(parquet_game_ids)}")
    logger.info(f"   Games in odds data: {len(odds_game_ids)}")
    logger.info(f"   Common games: {len(common_game_ids)}")
    
    # Show sample of game identifiers from each dataset
    logger.info(f"   Sample parquet {id_column}s: {list(parquet_df[id_column].head(5))}")
    logger.info(f"   Sample odds {id_column}s: {list(odds_columns_df[id_column].head(5))}")
    
    # Check data types
    logger.info(f"   Parquet {id_column} dtype: {parquet_df[id_column].dtype}")
    logger.info(f"   Odds {id_column} dtype: {odds_columns_df[id_column].dtype}")
    
    # Get the bookmaker value (should be consistent across all rows)
    bookmaker_value = None
    if 'bookmaker' in odds_columns_df.columns:
        bookmaker_value = odds_columns_df['bookmaker'].iloc[0] if len(odds_columns_df) > 0 else None
    
    # Check which odds columns actually exist in the pivoted data
    existing_odds_columns = [col for col in odds_columns if col in odds_columns_df.columns]
    
    # Select columns to merge - always include the identifier
    merge_columns = [id_column] + existing_odds_columns
    
    # Only proceed if we have some odds columns to merge
    if len(existing_odds_columns) == 0:
        logger.warning("⚠️  No odds columns found in pivoted data!")
        return parquet_df
    
    # Ensure identifier data types match before merging
    if parquet_df[id_column].dtype != odds_columns_df[id_column].dtype:
        logger.warning(f"⚠️  Data type mismatch for {id_column}. Converting to same type...")
        # Convert both to string to ensure compatibility
        parquet_df[id_column] = parquet_df[id_column].astype(str)
        odds_columns_df[id_column] = odds_columns_df[id_column].astype(str)
    
    # Merge the odds columns with the original parquet data
    # No suffixes needed since we cleaned up duplicate columns
    enhanced_df = parquet_df.merge(
        odds_columns_df[merge_columns], 
        on=id_column, 
        how='left'
    )
    
    # Add missing odds columns as NaN if they don't exist
    for col in ['home_ml', 'away_ml', 'total_line', 'over_odds', 'under_odds']:
        if col not in enhanced_df.columns:
            enhanced_df[col] = np.nan
    
    # If bookmaker wasn't in the columns but we know the value, add it for rows with odds
    if 'bookmaker' not in enhanced_df.columns and bookmaker_value:
        # Only set bookmaker for rows that have any odds data
        has_odds_mask = enhanced_df[['home_ml', 'away_ml', 'total_line', 'over_odds', 'under_odds']].notna().any(axis=1)
        enhanced_df.loc[has_odds_mask, 'bookmaker'] = bookmaker_value
    elif 'bookmaker' not in enhanced_df.columns:
        enhanced_df['bookmaker'] = np.nan
    
    # Check how many games got odds data
    odds_mask = enhanced_df[['home_ml', 'away_ml', 'total_line', 'over_odds', 'under_odds']].notna().any(axis=1)
    games_with_odds = enhanced_df[odds_mask]
    logger.info(f"📊 Enhanced {len(games_with_odds)} games out of {len(enhanced_df)} total games with odds data")
    
    # If no games were enhanced, show more debug info
    if len(games_with_odds) == 0 and len(common_game_ids) > 0:
        logger.warning("⚠️  Found common game identifiers but no data was merged. Checking for issues...")
        # Check a specific common game
        sample_game_id = list(common_game_ids)[0]
        logger.info(f"   Sample {id_column}: {sample_game_id}")
        logger.info(f"   In parquet: {sample_game_id in parquet_df[id_column].values}")
        logger.info(f"   In odds: {sample_game_id in odds_columns_df[id_column].values}")
    
    # Save back to parquet file (with backup)
    backup_path = parquet_path.replace('.parquet', '_backup.parquet')
    try:
        # Create backup of original file
        parquet_df.to_parquet(backup_path, index=False)
        logger.info(f"💾 Created backup at {backup_path}")
        
        # Save enhanced version
        enhanced_df.to_parquet(parquet_path, index=False)
        logger.info(f"✅ Updated master parquet at {parquet_path}")
        logger.info(f"📊 Added columns: bookmaker, home_ml, away_ml, total_line, over_odds, under_odds")
        
        # Show sample of new data
        sample_cols = [id_column, 'home_team_id', 'away_team_id', 'home_ml', 'away_ml', 'total_line', 'bookmaker']
        # Keep only columns that exist
        sample_cols = [col for col in sample_cols if col in enhanced_df.columns]
            
        if len(games_with_odds) > 0:
            odds_sample = enhanced_df[odds_mask][sample_cols].head(3)
            logger.info("📋 Sample of enhanced data:")
            logger.info(f"\n{odds_sample.to_string(index=False)}")
        
        return enhanced_df
        
    except Exception as e:
        logger.error(f"Error updating master parquet: {e}")
        logger.info(f"Original file backup preserved at {backup_path}")
        return None

def create_summary_stats(matched_df):
    logger.info("📈 Creating match summary...")
    
    # Determine which id column we have
    id_column = 'game_id' if 'game_id' in matched_df.columns else 'game_pk'
    
    # Use the correct column names based on what's available after merge
    games_by_date_col = 'game_date_parquet' if 'game_date_parquet' in matched_df.columns else 'game_date'
    bookmaker_col = 'bookmaker_odds' if 'bookmaker_odds' in matched_df.columns else 'bookmaker'
    
    games_by_date = matched_df.groupby(games_by_date_col)[id_column].nunique().reset_index()
    games_by_date.columns = ['game_date', 'unique_games']
    
    # Check if bookmaker column exists
    if bookmaker_col in matched_df.columns:
        market_coverage = matched_df.groupby(['market', bookmaker_col]).size().reset_index(name='records')
    else:
        # If no bookmaker column, just group by market
        market_coverage = matched_df.groupby(['market']).size().reset_index(name='records')
    
    # Use the home_team column that exists (might have suffix)
    home_team_col = 'home_team_norm' if 'home_team_norm' in matched_df.columns else 'home_team'
    if home_team_col not in matched_df.columns:
        # Find column containing 'home_team'
        home_team_cols = [col for col in matched_df.columns if 'home_team' in col and 'norm' in col]
        if home_team_cols:
            home_team_col = home_team_cols[0]
    
    if home_team_col in matched_df.columns:
        team_coverage = matched_df.groupby(home_team_col).agg({
            id_column: 'nunique',
            'odds': 'count'
        }).reset_index()
        team_coverage.columns = ['team', 'games', 'total_odds_records']
    else:
        team_coverage = pd.DataFrame(columns=['team', 'games', 'total_odds_records'])
    
    return {
        'games_by_date': games_by_date,
        'market_coverage': market_coverage, 
        'team_coverage': team_coverage,
        'data_availability': {
            'total_games': matched_df[id_column].nunique(),
            'games_with_moneyline': matched_df[matched_df['market'] == 'h2h'][id_column].nunique(),
            'games_with_totals': matched_df[matched_df['market'] == 'totals'][id_column].nunique(),
            'games_with_both': matched_df.groupby(id_column)['market'].nunique().eq(2).sum()
        }
    }


def main():
    """Main function"""
    logger.info("🚀 Starting MLB odds enhancement of master parquet file...")
    
    # Configuration
    PARQUET_PATH = r"C:\Users\DMelv\Documents\bettingModelBaseball\apiBaseball\pipeline\ml_pipeline_output\master_features_table.parquet" # Update this to your master parquet path
    
    # Connect to database
    # Connect to database
    engine = create_db_connection()
    if not engine:
        return
    
    # Load team mappings from SQL
    logger.info("📋 Loading team mappings from SQL...")
    full_name_to_abbrev, team_id_to_abbrev = load_team_mappings(engine)
    if not full_name_to_abbrev or not team_id_to_abbrev:
        logger.error("Failed to load team mappings")
        return
    
    # Load data
    logger.info("📤 Loading odds data from SQL...")
    odds_df = load_odds_data(engine)
    if odds_df is None or odds_df.empty:
        logger.error("No odds data found")
        return
    
    logger.info("📤 Loading master parquet file...")
    parquet_df = load_parquet_data(PARQUET_PATH, team_id_to_abbrev)
    if parquet_df is None or parquet_df.empty:
        logger.error("No parquet data found")
        return
    
    logger.info(f"📊 Original master parquet has {len(parquet_df)} games")
    
    # Check which identifier column exists in the parquet
    if 'game_id' in parquet_df.columns:
        id_column = 'game_id'
    elif 'game_pk' in parquet_df.columns:
        id_column = 'game_pk'
    else:
        logger.error("Neither game_id nor game_pk found in parquet file!")
        return
    
    logger.info(f"📌 Using '{id_column}' as the game identifier")
    
    # Match the data
    matched_df, id_column_from_match = match_games(odds_df, parquet_df, full_name_to_abbrev)
    if matched_df.empty:
        logger.warning("No matches found!")
        return
    
    # Use the id_column from the parquet file, not from match function
    # The match function might be using a different column
    
    # Pivot the odds data to get clean columns
    odds_pivoted = pivot_odds_data(matched_df, id_column)
    
    # Create summary of what we're adding
    summary = create_summary_stats(matched_df)
    
    # Print summary info
    logger.info(f"📊 ODDS DATA SUMMARY:")
    logger.info(f"   Games with odds data: {summary['data_availability']['total_games']}")
    logger.info(f"   Games with moneyline: {summary['data_availability']['games_with_moneyline']}")
    logger.info(f"   Games with totals: {summary['data_availability']['games_with_totals']}")
    logger.info(f"   Games with both: {summary['data_availability']['games_with_both']}")
    
    # Get bookmaker info from the raw matched data
    if 'bookmaker_odds' in matched_df.columns:
        bookmakers = matched_df['bookmaker_odds'].unique()
    elif 'bookmaker' in matched_df.columns:
        bookmakers = matched_df['bookmaker'].unique()
    else:
        bookmakers = ['Unknown']
    logger.info(f"   Bookmaker(s): {bookmakers}")
    
    # Add bookmaker to pivoted data if it's missing
    if 'bookmaker' not in odds_pivoted.columns and len(bookmakers) > 0:
        odds_pivoted['bookmaker'] = bookmakers[0]
    
    # Append odds columns to master parquet
    enhanced_df = append_odds_to_master_parquet(parquet_df, odds_pivoted, PARQUET_PATH, id_column)
    
    if enhanced_df is not None:
        # Final summary
        total_games = len(enhanced_df)
        games_with_any_odds = len(enhanced_df[
            enhanced_df['home_ml'].notna() | 
            enhanced_df['total_line'].notna()
        ])
        coverage_pct = (games_with_any_odds / total_games) * 100 if total_games > 0 else 0
        
        logger.info(f"🎯 FINAL RESULTS:")
        logger.info(f"   Total games in master parquet: {total_games}")
        logger.info(f"   Games enhanced with odds: {games_with_any_odds}")
        logger.info(f"   Coverage: {coverage_pct:.1f}%")
        logger.info(f"   New columns added: bookmaker, home_ml, away_ml, total_line, over_odds, under_odds")
    
    logger.info("🏁 Master parquet enhancement complete!")

if __name__ == "__main__":
    main()