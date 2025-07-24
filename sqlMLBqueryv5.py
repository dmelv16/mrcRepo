import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import urllib.parse
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
import pytz
from sklearn.preprocessing import StandardScaler
import joblib
import os
import urllib
import pickle
import json

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the MLB data pipeline"""
    start_year: int = 2017
    end_year: int = 2025
    rolling_games_team: int = 10
    rolling_games_pitcher: int = 5
    rolling_days_bullpen: int = 5
    min_games_threshold: int = 5
    
    # Database config
    db_server: str = "DESKTOP-J9IV3OH"
    db_name: str = "StatcastDB"
    db_user: str = "mlb_user"
    db_password: str = "mlbAdmin"
    
    # Output paths
    output_dir: str = r"C:\Users\DMelv\Documents\bettingModelBaseball\apiBaseball\pipeline\ml_pipeline_output"
    scaler_path: str = r"C:\Users\DMelv\Documents\bettingModelBaseball\apiBaseball\pipeline\ml_pipeline_output\scalers.pkl"
    feature_names_path: str = r"C:\Users\DMelv\Documents\bettingModelBaseball\apiBaseball\pipeline\ml_pipeline_output\feature_names.pkl"
    cache_dir: str = r"C:\Users\DMelv\Documents\bettingModelBaseball\apiBaseball\pipeline\mlb_cache" # ✅ Fixed: now a proper dataclass field

class DataValidator:
    """Validates data at various stages of the pipeline"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, name: str, required_cols: List[str] = None) -> bool:
        """Validate dataframe has data and required columns"""
        if df is None or df.empty:
            logger.error(f"{name} dataframe is empty or None")
            return False
            
        logger.info(f"{name} shape: {df.shape}")
        
        if required_cols:
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                logger.error(f"{name} missing columns: {missing_cols}")
                return False
                
        # Check for excessive nulls
        null_percentages = df.isnull().sum() / len(df) * 100
        high_null_cols = null_percentages[null_percentages > 50]
        if not high_null_cols.empty:
            logger.warning(f"{name} columns with >50% nulls: {high_null_cols.to_dict()}")

        # Check for excessive zeros
        zero_percentages = (df == 0).sum() / len(df) * 100
        high_zero_cols = zero_percentages[zero_percentages > 50]
        if not high_zero_cols.empty:
            logger.warning(f"{name} columns with >50% zeros: {high_zero_cols.to_dict()}")
            
        return True
    
    @staticmethod
    def validate_statcast_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate statcast data"""
        logger.info("Validating statcast data...")
        
        # Remove duplicate pitches
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['game_pk', 'at_bat_number', 'pitch_number'])
        logger.info(f"Removed {before_dedup - len(df)} duplicate pitches")
        
        # Filter valid pitch types
        valid_pitch_types = ['FF', 'SI', 'FC', 'SL', 'CH', 'CU', 'KC', 'FS', 'KN', 'SC', 'FO', 'FA', 'EP', 'SV']
        df = df[df['pitch_type'].isin(valid_pitch_types) | df['pitch_type'].isna()]
        
        # Validate numeric columns
        numeric_cols = [
            'release_speed',
            'release_spin_rate',
            'release_pos_x',
            'release_pos_y',
            'release_pos_z',
            'release_extension',
            
            'spin_rate',  # if populated; otherwise, use release_spin_rate
            'spin_axis',
            
            'vx0', 'vy0', 'vz0',    # initial velocities
            'ax', 'ay', 'az',       # accelerations
            'pfx_x', 'pfx_z',       # pitch movement
            'plate_x', 'plate_z',   # location over plate

            'sz_top', 'sz_bot',     # strike zone height
            'zone',                 # zone bucket (numeric)

            'launch_speed',
            'launch_angle',
            'launch_speed_angle',
            'hit_distance_sc',

            'estimated_ba_using_speedangle',
            'estimated_woba_using_speedangle',
            'estimated_slg_using_speedangle',

            'woba_value', 'woba_denom',
            'babip_value', 'iso_value',

            'effective_speed',
            
            'delta_home_win_exp',
            'delta_run_exp',
            'delta_pitcher_run_exp',
            'home_win_exp',
            'bat_win_exp',
            
            'bat_speed',
            'swing_length',
            'arm_angle',
            'attack_angle',
            'attack_direction',
            'swing_path_tilt',
            
            'intercept_ball_minus_batter_pos_x_inches',
            'intercept_ball_minus_batter_pos_y_inches',

            'home_score', 'away_score',
            'bat_score', 'fld_score',
            'post_home_score', 'post_away_score',
            'post_bat_score', 'post_fld_score',
            'home_score_diff', 'bat_score_diff',
            
            'age_pit', 'age_bat',
            'n_thruorder_pitcher',
            'n_priorpa_thisgame_player_at_bat',
            
            'pitcher_days_since_prev_game',
            'batter_days_since_prev_game',
            'pitcher_days_until_next_game',
            'batter_days_until_next_game',
            
            'api_break_z_with_gravity',
            'api_break_x_arm',
            'api_break_x_batter_in',
            
            'hyper_speed'  # custom/experimental, but keep if populated
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df



class DatabaseConnection:
    """Handles database connections and queries with smart caching"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.engine = None
        
        # Define master cache paths
        self.master_statcast_cache = os.path.join(self.config.cache_dir, "master_statcast.parquet")
        self.master_baseball_cache = os.path.join(self.config.cache_dir, "master_baseball_scrape.parquet")
        
        # Ensure cache directory exists
        os.makedirs(self.config.cache_dir, exist_ok=True)

    def connect(self):
        """Create database connection"""
        try:
            params = urllib.parse.quote_plus(
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={self.config.db_server};"
                f"DATABASE={self.config.db_name};"
                f"UID={self.config.db_user};"
                f"PWD={self.config.db_password};"
                f"Encrypt=no;"
                f"TrustServerCertificate=yes;"
            )
            self.engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
            logger.info("✅ Successfully connected to SQL Server")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False

    def load_team_mappings(self) -> Tuple[Dict, Dict]:
        """Load team mappings handling duplicates"""
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
                            ELSE 2
                        END
                ) as rn
            FROM team_abbrev_map
        )
        SELECT abbrev, team_id, full_name
        FROM RankedTeams
        WHERE rn = 1
        """
        df = pd.read_sql(query, self.engine)
        return dict(zip(df['full_name'], df['team_id'])), dict(zip(df['abbrev'], df['team_id']))

    def _update_master_cache(self, table_name: str, master_cache_path: str, 
                           date_column: str = "game_date", game_type_filter: str = "",
                           requested_start: str = None, requested_end: str = None) -> pd.DataFrame:
        """Update master cache with only new data from SQL"""
        
        if os.path.exists(master_cache_path):
            # Load existing cache
            master_df = pd.read_parquet(master_cache_path)
            master_df[date_column] = pd.to_datetime(master_df[date_column])
            
            # Find the latest date in cache
            cache_max_date = master_df[date_column].max()
            logger.info(f"📦 Master cache exists for {table_name}. Last date: {cache_max_date.strftime('%Y-%m-%d')}")
            
            # Query only for new data up to requested end date
            where_clause = f"{date_column} > '{cache_max_date.strftime('%Y-%m-%d')}'"
            if requested_end:
                where_clause += f" AND {date_column} <= '{requested_end}'"
        else:
            # No cache exists, only fetch data needed for requested range
            master_df = pd.DataFrame()
            if requested_start and requested_end:
                where_clause = f"{date_column} >= '{requested_start}' AND {date_column} <= '{requested_end}'"
                logger.info(f"📦 No master cache found for {table_name}. Creating cache starting from {requested_start}")
            else:
                # Fallback: if no dates provided, start from beginning of 2017 season
                where_clause = f"{date_column} >= '2017-03-01'"
                logger.info(f"📦 No master cache found for {table_name}. Creating cache starting from 2017-03-01")

        # Build query for new data
        query = f"""
        SELECT * FROM {table_name}
        WHERE {where_clause}
        """
        
        if game_type_filter:
            query += f" AND game_type = '{game_type_filter}'"
            
        # Add ordering for consistency
        if table_name == "statcast_game_logs":
            query += " ORDER BY game_date, game_pk, at_bat_number, pitch_number"
        else:
            query += " ORDER BY game_date, game_time"

        # Fetch new data
        new_df = pd.read_sql(query, self.engine)
        
        if not new_df.empty:
            new_df[date_column] = pd.to_datetime(new_df[date_column])
            logger.info(f"🔄 Found {len(new_df)} new rows in {table_name}")
            
            # Combine with existing data
            updated_df = pd.concat([master_df, new_df], ignore_index=True)
            
            # Remove any duplicates (in case of overlapping data)
            if table_name == "statcast_game_logs":
                updated_df.drop_duplicates(subset=['game_pk', 'at_bat_number', 'pitch_number'], 
                                         keep='last', inplace=True)
            else:
                updated_df.drop_duplicates(subset=['gamePk', 'side'], keep='last', inplace=True)
            
            # Sort by date for efficient future queries
            updated_df.sort_values(date_column, inplace=True)
            
            # Save updated master cache
            updated_df.to_parquet(master_cache_path, index=False)
            logger.info(f"✅ Master cache updated: {master_cache_path} (Total rows: {len(updated_df)})")
            
            return updated_df
        else:
            logger.info(f"✅ No new rows found for {table_name}")
            return master_df

    def load_statcast_data(self, start_date: str, end_date: str, use_cache: bool = True) -> pd.DataFrame:
        """Load statcast data from master cache"""
        if use_cache:
            # Update master cache with any new data
            master_df = self._update_master_cache(
                "statcast_game_logs", 
                self.master_statcast_cache,
                date_column="game_date",
                game_type_filter="R",
                requested_start=start_date,
                requested_end=end_date
            )
            
            # Filter to requested date range
            if not master_df.empty:
                mask = (master_df['game_date'] >= pd.to_datetime(start_date)) & \
                       (master_df['game_date'] <= pd.to_datetime(end_date))
                filtered_df = master_df[mask].copy()
                logger.info(f"📊 Returning {len(filtered_df)} rows for date range {start_date} to {end_date}")
                return filtered_df
            else:
                return pd.DataFrame()
        else:
            # Direct SQL query (bypasses cache)
            query = f"""
            SELECT * FROM statcast_game_logs
            WHERE game_date >= '{start_date}' AND game_date <= '{end_date}'
            AND game_type = 'R'
            ORDER BY game_date, game_pk, at_bat_number, pitch_number
            """
            df = pd.read_sql(query, self.engine)
            df['game_date'] = pd.to_datetime(df['game_date'])
            return df

    def load_baseball_scrape_data(self, start_date: str, end_date: str, use_cache: bool = True) -> pd.DataFrame:
        """Load baseball scrape data from master cache"""
        if use_cache:
            # Update master cache with any new data
            master_df = self._update_master_cache(
                "baseballScrapev2", 
                self.master_baseball_cache,
                date_column="game_date",
                requested_start=start_date,
                requested_end=end_date
            )
            
            # Filter to requested date range
            if not master_df.empty:
                mask = (master_df['game_date'] >= pd.to_datetime(start_date)) & \
                       (master_df['game_date'] <= pd.to_datetime(end_date))
                filtered_df = master_df[mask].copy()
                logger.info(f"📊 Returning {len(filtered_df)} rows for date range {start_date} to {end_date}")
                return filtered_df
            else:
                return pd.DataFrame()
        else:
            # Direct SQL query (bypasses cache)
            query = f"""
            SELECT * FROM baseballScrapev2
            WHERE game_date >= '{start_date}' AND game_date <= '{end_date}'
            ORDER BY game_date, game_time
            """
            df = pd.read_sql(query, self.engine)
            df['game_date'] = pd.to_datetime(df['game_date'])
            return df
    
    def get_cache_info(self) -> dict:
        """Get information about the current cache state"""
        info = {}
        
        if os.path.exists(self.master_statcast_cache):
            df = pd.read_parquet(self.master_statcast_cache)
            info['statcast'] = {
                'exists': True,
                'rows': len(df),
                'min_date': df['game_date'].min().strftime('%Y-%m-%d'),
                'max_date': df['game_date'].max().strftime('%Y-%m-%d'),
                'size_mb': os.path.getsize(self.master_statcast_cache) / (1024 * 1024)
            }
        else:
            info['statcast'] = {'exists': False}
            
        if os.path.exists(self.master_baseball_cache):
            df = pd.read_parquet(self.master_baseball_cache)
            info['baseball_scrape'] = {
                'exists': True,
                'rows': len(df),
                'min_date': df['game_date'].min().strftime('%Y-%m-%d'),
                'max_date': df['game_date'].max().strftime('%Y-%m-%d'),
                'size_mb': os.path.getsize(self.master_baseball_cache) / (1024 * 1024)
            }
        else:
            info['baseball_scrape'] = {'exists': False}
            
        return info
    
    def clear_cache(self, confirm: bool = False):
        """Clear the master cache files"""
        if not confirm:
            logger.warning("⚠️ Set confirm=True to actually clear the cache")
            return
            
        for cache_file in [self.master_statcast_cache, self.master_baseball_cache]:
            if os.path.exists(cache_file):
                os.remove(cache_file)
                logger.info(f"🗑️ Deleted cache file: {cache_file}")
        
        logger.info("✅ Cache cleared")


class FeatureEngineer:
    """Creates features for the ML pipeline with proper temporal integrity"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.validator = DataValidator()
        
    def create_rolling_team_features(self, df: pd.DataFrame, target_date: str = None) -> pd.DataFrame:
        """Simplified rolling features that aligns with TeamStateArrays approach"""
        logger.info(f"Creating {self.config.rolling_games_team}-game rolling averages...")
        
        # Convert target_date if provided
        target_date_pd = pd.to_datetime(target_date) if target_date else None
        
        # Define columns to roll
        batting_cols = [col for col in df.columns if col.startswith('batting_')]
        pitching_cols = [col for col in df.columns if col.startswith('pitching_')]
        fielding_cols = [col for col in df.columns if col.startswith('fielding_')]
        feature_cols = batting_cols + pitching_cols + fielding_cols
        
        # Clean the data first
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].replace(['.---', '---', '-', 'N/A', 'NA', '', ' '], np.nan)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Get numeric columns
        numeric_feature_cols = [col for col in feature_cols 
                            if col in df.columns 
                            and df[col].dtype in ['float64', 'int64', 'float32', 'int32']
                            and df[col].notna().any()]
        
        # Process each team
        rolled_dfs = []
        
        for team_id in df['team_id'].unique():
            team_df = df[df['team_id'] == team_id].copy()
            team_df = team_df.sort_values('game_date')
            
            if len(team_df) < self.config.min_games_threshold:
                continue
            
            # Convert dates for comparison
            team_df['game_date_pd'] = pd.to_datetime(team_df['game_date'])
            
            # Create rolling features
            rolling_features = pd.DataFrame(index=team_df.index)
            
            for idx in range(len(team_df)):
                current_date = team_df.iloc[idx]['game_date_pd']
                
                # Determine which games to use for rolling calculation
                if target_date_pd and current_date >= target_date_pd:
                    # For inference: use all games before target date
                    valid_games = team_df[team_df['game_date_pd'] < target_date_pd]
                else:
                    # FIX: For training: use all games BEFORE current date (exclude current game)
                    valid_games = team_df.iloc[:idx] if idx > 0 else team_df.iloc[0:0]  # Empty DataFrame for first game
                
                # Calculate rolling stats from the most recent N games
                if len(valid_games) >= self.config.min_games_threshold:
                    recent_games = valid_games.tail(self.config.rolling_games_team)
                    
                    for col in numeric_feature_cols:
                        if col in recent_games.columns:
                            rolling_features.loc[team_df.index[idx], f"{col}_roll{self.config.rolling_games_team}"] = \
                                recent_games[col].mean()
                else:
                    # Not enough history
                    for col in numeric_feature_cols:
                        rolling_features.loc[team_df.index[idx], f"{col}_roll{self.config.rolling_games_team}"] = np.nan
            
            # Add identifiers
            rolling_features['team_id'] = team_id
            rolling_features['game_pk'] = team_df['gamePk'].values
            rolling_features['game_date'] = team_df['game_date'].values
            
            rolled_dfs.append(rolling_features)
        
        if not rolled_dfs:
            logger.warning("No rolling features created - insufficient data")
            return df
        
        # Combine results
        return pd.concat(rolled_dfs, ignore_index=True)
        
    def create_pitcher_features(self, statcast_df: pd.DataFrame, target_date: str = None) -> pd.DataFrame:
        """Create rolling features for starting pitchers"""
        logger.info("Creating pitcher features...")
        
        # Identify starting pitchers (first pitcher for each team in each game)
        starters = statcast_df.sort_values(['game_pk', 'at_bat_number', 'pitch_number']).groupby(['game_pk', 'inning_topbot']).first().reset_index()
        starters = starters[['game_date', 'pitcher', 'game_pk']]
        starters['is_starter'] = True
        
        # Merge back to identify all starter pitches
        statcast_df = statcast_df.merge(
            starters[['game_pk', 'pitcher', 'is_starter']],
            on=['game_pk', 'pitcher'],
            how='left'
        )
        statcast_df['is_starter'] = statcast_df['is_starter'].fillna(False)
        
        # Calculate pitcher metrics by game
        pitcher_game_stats = self.calculate_pitcher_game_stats(
            statcast_df[statcast_df['is_starter']]
        )
        
        # Create rolling averages
        pitcher_rolled = []
        
        for pitcher_id in pitcher_game_stats['pitcher'].unique():
            pitcher_df = pitcher_game_stats[
                pitcher_game_stats['pitcher'] == pitcher_id
            ].sort_values('game_date')
            
            if len(pitcher_df) < 1:
                continue
            
            # Get only numeric columns for rolling calculations
            numeric_cols = pitcher_df.select_dtypes(include=[np.number]).columns.tolist()
            id_cols = ['pitcher', 'game_date', 'game_pk']
            roll_cols = [col for col in numeric_cols if col not in id_cols]
            
            if target_date:
                # Incremental mode - split historical and target
                target_date_pd = pd.to_datetime(target_date)
                historical_mask = pitcher_df['game_date'] < target_date_pd
                historical_df = pitcher_df[historical_mask]
                target_df = pitcher_df[~historical_mask]
                
                rolled_features = pd.DataFrame()
                
                # Process historical games normally
                if not historical_df.empty:
                    historical_rolled = pd.DataFrame(index=historical_df.index)
                    
                    for i in range(len(historical_df)):
                        if i == 0:
                            for col in roll_cols:
                                historical_rolled.loc[historical_df.index[i], f"SP_{col}_roll{self.config.rolling_games_pitcher}"] = np.nan
                        else:
                            # FIX: Use games BEFORE current index
                            start_idx = max(0, i - self.config.rolling_games_pitcher)
                            # Use slice [start_idx:i] which excludes i
                            
                            for col in roll_cols:
                                historical_rolled.loc[historical_df.index[i], f"SP_{col}_roll{self.config.rolling_games_pitcher}"] = \
                                    historical_df[col].iloc[start_idx:i].mean()
                    
                    rolled_features = pd.concat([rolled_features, historical_rolled])
                
                # For target date games, use last N games from historical
                if not target_df.empty and not historical_df.empty:
                    target_rolled = pd.DataFrame(index=target_df.index)
                    
                    # Get the last N games from historical data
                    last_n_games = historical_df.tail(self.config.rolling_games_pitcher)
                    
                    # Calculate rolling stats for target games
                    for col in roll_cols:
                        if len(last_n_games) > 0:
                            target_rolled[f"SP_{col}_roll{self.config.rolling_games_pitcher}"] = last_n_games[col].mean()
                        else:
                            target_rolled[f"SP_{col}_roll{self.config.rolling_games_pitcher}"] = np.nan
                    
                    rolled_features = pd.concat([rolled_features, target_rolled])
                
            else:
                # Batch mode - FIX: exclude current game
                rolled_features = pd.DataFrame(index=pitcher_df.index)
                
                for i in range(len(pitcher_df)):
                    if i == 0:
                        for col in roll_cols:
                            rolled_features.loc[pitcher_df.index[i], f"SP_{col}_roll{self.config.rolling_games_pitcher}"] = np.nan
                    else:
                        start_idx = max(0, i - self.config.rolling_games_pitcher)
                        # FIX: Use slice [start_idx:i] which excludes i
                        
                        for col in roll_cols:
                            rolled_features.loc[pitcher_df.index[i], f"SP_{col}_roll{self.config.rolling_games_pitcher}"] = \
                                pitcher_df[col].iloc[start_idx:i].mean()
            
            rolled_features['pitcher'] = pitcher_id
            rolled_features['game_date'] = pitcher_df['game_date'].values
            rolled_features['game_pk'] = pitcher_df['game_pk'].values
            
            pitcher_rolled.append(rolled_features)
        
        return pd.concat(pitcher_rolled, ignore_index=True) if pitcher_rolled else pd.DataFrame()
        
    def create_bullpen_features(self, statcast_df: pd.DataFrame, target_date: str = None) -> pd.DataFrame:
        """Create rolling features for bullpen"""
        logger.info("Creating bullpen features...")
        
        # First, identify starters vs relievers
        game_first_pitchers = statcast_df.groupby('game_pk').agg({
            'pitcher': lambda x: x.iloc[0],
            'game_date': 'first'
        }).reset_index()
        game_first_pitchers.columns = ['game_pk', 'starting_pitcher', 'game_date']
        
        # Add is_starter column
        statcast_df = statcast_df.merge(
            game_first_pitchers[['game_pk', 'starting_pitcher']],
            on='game_pk',
            how='left'
        )
        statcast_df['is_starter'] = statcast_df['pitcher'] == statcast_df['starting_pitcher']
        
        # Identify bullpen pitchers (non-starters)
        bullpen_df = statcast_df[~statcast_df['is_starter']].copy()
        
        # Map teams to pitchers
        pitcher_teams = bullpen_df.groupby(['pitcher', 'game_pk']).agg({
            'home_team': 'first',
            'away_team': 'first',
            'inning_topbot': 'first'
        }).reset_index()
        
        # Determine pitcher's team
        pitcher_teams['team'] = pitcher_teams.apply(
            lambda x: x['home_team'] if x['inning_topbot'] == 'Bot' else x['away_team'],
            axis=1
        )
        
        # Calculate bullpen stats by team and date
        bullpen_stats = self.calculate_bullpen_daily_stats(bullpen_df, pitcher_teams)
        
        # Create rolling averages
        team_rolled = []
        
        for team in bullpen_stats['team'].unique():
            team_df = bullpen_stats[
                bullpen_stats['team'] == team
            ].sort_values('game_date')
            
            # Only select numeric columns
            numeric_cols = team_df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['game_pk']
            roll_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            if target_date:
                # Incremental mode
                target_date_pd = pd.to_datetime(target_date)
                rolled_features = pd.DataFrame()
                
                # Process each game
                for i in range(len(team_df)):
                    current_date = team_df.iloc[i]['game_date']
                    current_row_features = pd.DataFrame(index=[team_df.index[i]])
                    
                    if current_date < target_date_pd:
                        # Historical game - calculate normally (excluding current date)
                        past_window_start = current_date - pd.Timedelta(days=self.config.rolling_days_bullpen)
                        past_data = team_df[
                            (team_df['game_date'] < current_date) & 
                            (team_df['game_date'] >= past_window_start)
                        ]
                    else:
                        # Target date game - use data up to cutoff
                        cutoff_date = target_date_pd - pd.Timedelta(days=1)
                        past_window_start = cutoff_date - pd.Timedelta(days=self.config.rolling_days_bullpen)
                        past_data = team_df[
                            (team_df['game_date'] <= cutoff_date) & 
                            (team_df['game_date'] > past_window_start)
                        ]
                    
                    if len(past_data) >= 3:  # min_periods
                        for col in roll_cols:
                            current_row_features[f"bullpen_{col}_roll{self.config.rolling_days_bullpen}d"] = past_data[col].mean()
                    else:
                        for col in roll_cols:
                            current_row_features[f"bullpen_{col}_roll{self.config.rolling_days_bullpen}d"] = np.nan
                    
                    rolled_features = pd.concat([rolled_features, current_row_features])
                
            else:
                # Batch mode - existing logic is correct (excludes current date)
                rolled_features = pd.DataFrame(index=team_df.index)
                
                for i in range(len(team_df)):
                    current_date = team_df.iloc[i]['game_date']
                    
                    # Get data from past N days, excluding current date
                    past_data = team_df[
                        (team_df['game_date'] < current_date) & 
                        (team_df['game_date'] >= current_date - pd.Timedelta(days=self.config.rolling_days_bullpen))
                    ]
                    
                    if len(past_data) >= 3:  # min_periods
                        for col in roll_cols:
                            rolled_features.loc[team_df.index[i], f"bullpen_{col}_roll{self.config.rolling_days_bullpen}d"] = \
                                past_data[col].mean()
                    else:
                        for col in roll_cols:
                            rolled_features.loc[team_df.index[i], f"bullpen_{col}_roll{self.config.rolling_days_bullpen}d"] = np.nan
            
            rolled_features['team'] = team
            rolled_features['game_date'] = team_df['game_date'].values
            rolled_features['game_pk'] = team_df['game_pk'].values
            
            team_rolled.append(rolled_features)
        
        # Clean up temporary column
        if 'starting_pitcher' in statcast_df.columns:
            statcast_df.drop('starting_pitcher', axis=1, inplace=True)
        
        return pd.concat(team_rolled, ignore_index=True) if team_rolled else pd.DataFrame()

    def calculate_pitcher_game_stats(self, pitcher_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive per-game stats for pitchers with advanced metrics"""
        
        # Early return if empty
        if pitcher_df.empty:
            logger.warning("Empty pitcher dataframe provided to calculate_pitcher_game_stats")
            return pd.DataFrame()
        
        # Debug logging to identify the issue
        logger.info(f"Input pitcher_df shape: {pitcher_df.shape}")
        logger.info(f"Unique pitchers: {pitcher_df['pitcher'].nunique()}")
        logger.info(f"Unique games: {pitcher_df['game_pk'].nunique()}")
        
        # Ensure game_pk is consistent type (convert to int if it's numeric)
        pitcher_df['game_pk'] = pd.to_numeric(pitcher_df['game_pk'], errors='coerce')
        
        # Ensure game_date is datetime
        pitcher_df['game_date'] = pd.to_datetime(pitcher_df['game_date'])
        
        # Check for unique pitcher-game combinations before processing
        unique_pitcher_games = pitcher_df.groupby(['pitcher', 'game_pk']).ngroups
        logger.info(f"Unique pitcher-game combinations: {unique_pitcher_games}")
        
        # Check for duplicate pitches
        duplicate_check = pitcher_df.groupby(['pitcher', 'game_pk', 'at_bat_number', 'pitch_number']).size()
        if (duplicate_check > 1).any():
            logger.warning(f"Found {(duplicate_check > 1).sum()} duplicate pitch entries!")
            # Remove duplicates
            pitcher_df = pitcher_df.drop_duplicates(subset=['pitcher', 'game_pk', 'at_bat_number', 'pitch_number'])
            logger.info(f"After deduplication: {pitcher_df.shape}")

        # --- Helper functions ---
        def safe_divide(numerator, denominator, default=0):
            """Safe division that handles zero denominators"""
            if isinstance(denominator, pd.Series):
                return numerator.fillna(0) / denominator.replace(0, 1)
            if denominator == 0 or pd.isna(denominator):
                return default
            return numerator / denominator

        def whiff_rate(descriptions):
            swings = descriptions.isin(['swinging_strike', 'swinging_strike_blocked', 
                                        'foul_tip', 'foul', 'hit_into_play', 
                                        'foul_bunt', 'missed_bunt'])
            whiffs = descriptions.isin(['swinging_strike', 'swinging_strike_blocked', 
                                        'missed_bunt', 'foul_tip'])
            return safe_divide(whiffs.sum(), swings.sum(), 0)

        def first_pitch_strike_rate(df):
            first_pitches = df[df['pitch_number'] == 1]
            if len(first_pitches) == 0:
                return 0
            strikes = first_pitches['description'].isin(['called_strike', 'swinging_strike', 
                                                        'foul', 'foul_tip', 'hit_into_play'])
            return strikes.mean()

        def edge_rate(zone_series):
            edge_zones = [1, 2, 3, 4, 6, 7, 8, 9]
            in_zone = zone_series.between(1, 9)
            on_edge = zone_series.isin(edge_zones)
            return safe_divide((on_edge & in_zone).sum(), in_zone.sum(), 0)

        def chase_rate_fn(group):
            if 'zone' not in group.columns or 'description' not in group.columns:
                return np.nan
            outside_zone = group['zone'].between(11, 14)
            swings_outside = outside_zone & group['description'].isin([
                'swinging_strike', 'swinging_strike_blocked',
                'foul', 'foul_tip', 'hit_into_play'
            ])
            return safe_divide(swings_outside.sum(), outside_zone.sum(), 0)

        def pitch_sequence_entropy(pitch_types):
            if len(pitch_types) < 2:
                return 0
            sequences = [f"{pitch_types.iloc[i]}-{pitch_types.iloc[i+1]}" 
                        for i in range(len(pitch_types)-1)]
            seq_counts = pd.Series(sequences).value_counts(normalize=True)
            return -sum(p * np.log2(p) for p in seq_counts if p > 0)

        def release_point_consistency(df):
            consistency_scores = []
            for pitch_type in df['pitch_type'].dropna().unique():
                pitch_data = df[df['pitch_type'] == pitch_type]
                if len(pitch_data) > 2:
                    x_std = pitch_data['release_pos_x'].std()
                    z_std = pitch_data['release_pos_z'].std()
                    consistency_scores.append(np.sqrt(x_std**2 + z_std**2))
            return np.mean(consistency_scores) if consistency_scores else None

        # Check which columns actually exist
        available_cols = pitcher_df.columns.tolist()
        
        # Build comprehensive aggregation dict based on available columns
        agg_dict = {}
        
        # Velocity metrics
        velocity_cols = ['release_speed', 'effective_speed']
        for col in velocity_cols:
            if col in available_cols:
                agg_dict[col] = ['mean', 'std', 'max', 'min']
        
        # Spin metrics
        if 'release_spin_rate' in available_cols:
            agg_dict['release_spin_rate'] = ['mean', 'std', 'max', 'min']
        if 'spin_axis' in available_cols:
            agg_dict['spin_axis'] = ['mean', 'std']
        
        # Movement metrics
        movement_cols = ['pfx_x', 'pfx_z', 'api_break_z_with_gravity', 'api_break_x_arm', 'api_break_x_batter_in']
        for col in movement_cols:
            if col in available_cols:
                agg_dict[col] = ['mean', 'std', 'max', 'min']
        
        # Release point metrics
        release_cols = ['release_extension', 'release_pos_x', 'release_pos_y', 'release_pos_z', 'arm_angle']
        for col in release_cols:
            if col in available_cols:
                if col in ['release_pos_x', 'release_pos_y', 'release_pos_z']:
                    agg_dict[col] = ['mean', 'std']
                elif col == 'release_extension':
                    agg_dict[col] = ['mean', 'std', 'max']
                else:
                    agg_dict[col] = ['mean', 'std']
        
        # Location metrics
        if 'plate_x' in available_cols:
            agg_dict['plate_x'] = ['std', 'mean']
        if 'plate_z' in available_cols:
            agg_dict['plate_z'] = ['std', 'mean']
        
        # Zone metrics
        if 'zone' in available_cols:
            agg_dict['zone'] = [
                lambda x: (x.between(1, 9)).mean(),  # zone_rate
                lambda x: (x.between(11, 14)).mean(),  # out_of_zone_rate
                edge_rate
            ]
        
        # Pitch type and sequencing
        if 'pitch_type' in available_cols:
            agg_dict['pitch_type'] = [
                lambda x: x.nunique(),
                pitch_sequence_entropy
            ]
        
        # Game situation
        if 'inning' in available_cols:
            agg_dict['inning'] = ['max', 'nunique', 'mean']
        
        # Count situations
        if 'outs_when_up' in available_cols:
            agg_dict['outs_when_up'] = [
                lambda x: (x == 0).mean(),
                lambda x: (x == 1).mean(),
                lambda x: (x == 2).mean()
            ]
        
        if 'balls' in available_cols:
            agg_dict['balls'] = [
                lambda x: (x == 0).mean(),
                lambda x: (x == 3).mean(),
                lambda x: (x >= 2).mean()
            ]
        
        if 'strikes' in available_cols:
            agg_dict['strikes'] = [
                lambda x: (x == 0).mean(),
                lambda x: (x == 2).mean(),
                lambda x: (x >= 2).mean()
            ]
        
        # Win expectancy and leverage
        leverage_cols = ['delta_home_win_exp', 'delta_run_exp', 'home_score_diff', 'home_win_exp']
        for col in leverage_cols:
            if col in available_cols:
                if col == 'delta_home_win_exp':
                    agg_dict[col] = ['sum', 'mean', 'std']
                elif col == 'delta_run_exp':
                    agg_dict[col] = ['sum', 'mean']
                elif col == 'home_score_diff':
                    agg_dict[col] = ['mean', 'min', 'max']
                else:
                    agg_dict[col] = ['mean', 'std']
        
        # Baserunners
        for base in ['on_1b', 'on_2b', 'on_3b']:
            if base in available_cols:
                agg_dict[base] = lambda x: x.notna().mean()
        
        # Events
        if 'events' in available_cols:
            agg_dict['events'] = [
                lambda x: (x == 'strikeout').sum(),
                lambda x: (x == 'walk').sum(),
                lambda x: (x == 'hit_by_pitch').sum(),
                lambda x: (x == 'home_run').sum(),
                lambda x: (x.isin(['single', 'double', 'triple', 'home_run'])).sum(),
                lambda x: (x == 'field_out').sum(),
                lambda x: (x == 'grounded_into_double_play').sum(),
                lambda x: (x == 'force_out').sum(),
                lambda x: (x == 'double_play').sum(),
                'count'
            ]
        
        # Pitch outcomes
        if 'description' in available_cols:
            agg_dict['description'] = [
                lambda x: (x.isin(['swinging_strike', 'called_strike', 'foul', 'foul_tip'])).mean(),
                lambda x: (x.isin(['swinging_strike', 'swinging_strike_blocked'])).mean(),
                lambda x: (x == 'called_strike').mean(),
                lambda x: (x == 'ball').mean(),
                lambda x: (x == 'blocked_ball').mean(),
                whiff_rate,
                lambda x: (x == 'hit_into_play').sum(),
                lambda x: (x == 'foul').mean()
            ]
        
        # Contact quality
        contact_cols = ['launch_speed', 'launch_angle', 'hit_distance_sc']
        for col in contact_cols:
            if col in available_cols:
                if col == 'launch_speed':
                    agg_dict[col] = ['mean', 'max', 'std', lambda x: x.quantile(0.9) if len(x) > 0 else None]
                elif col == 'launch_angle':
                    agg_dict[col] = ['mean', 'std', lambda x: x.quantile(0.1) if len(x) > 0 else None]
                else:
                    agg_dict[col] = ['mean', 'max', lambda x: x.quantile(0.9) if len(x) > 0 else None]
        
        # Expected stats
        expected_cols = ['estimated_ba_using_speedangle', 'estimated_woba_using_speedangle', 
                        'estimated_slg_using_speedangle']
        for col in expected_cols:
            if col in available_cols:
                agg_dict[col] = ['mean', 'max']
        
        # Value metrics
        if 'woba_value' in available_cols:
            agg_dict['woba_value'] = ['mean', 'sum']
        if 'woba_denom' in available_cols:
            agg_dict['woba_denom'] = 'sum'
        if 'babip_value' in available_cols:
            agg_dict['babip_value'] = 'mean'
        if 'iso_value' in available_cols:
            agg_dict['iso_value'] = 'mean'
        
        # Pitch count and usage
        if 'pitch_number' in available_cols:
            agg_dict['pitch_number'] = ['max', 'mean', lambda x: (x > 100).any()]
        if 'at_bat_number' in available_cols:
            agg_dict['at_bat_number'] = ['nunique', 'max']
        if 'n_thruorder_pitcher' in available_cols:
            agg_dict['n_thruorder_pitcher'] = ['max', 'mean']
        
        # Rest metrics
        if 'pitcher_days_since_prev_game' in available_cols:
            agg_dict['pitcher_days_since_prev_game'] = 'first'
        if 'pitcher_days_until_next_game' in available_cols:
            agg_dict['pitcher_days_until_next_game'] = 'first'
        
        # Swing metrics
        swing_cols = ['bat_speed', 'swing_length', 'attack_angle']
        for col in swing_cols:
            if col in available_cols:
                agg_dict[col] = ['mean', 'std']
        
        # Physics metrics
        physics_cols = ['ax', 'ay', 'az', 'vx0', 'vy0', 'vz0']
        for col in physics_cols:
            if col in available_cols:
                agg_dict[col] = 'mean'
        
        # Perform aggregation
        logger.info(f"Aggregating {len(agg_dict)} metrics")
        try:
            # CRITICAL: Use only pitcher and game_pk for grouping (not game_date to avoid type issues)
            game_stats = pitcher_df.groupby(['pitcher', 'game_pk']).agg(agg_dict)
            
            # Add game_date back after aggregation
            game_dates = pitcher_df.groupby(['pitcher', 'game_pk'])['game_date'].first()
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            return pd.DataFrame()
        
        # Log the shape after aggregation
        logger.info(f"After aggregation: {game_stats.shape}")
        
        # Reset index
        game_stats = game_stats.reset_index()
        
        # Add game_date back
        game_stats['game_date'] = game_dates.values
        
        # Log final unique combinations
        logger.info(f"Final unique pitcher-game combinations: {len(game_stats)}")
        
        # Flatten column names with proper mapping
        flattened_data = {'pitcher': game_stats['pitcher'].values,
                        'game_pk': game_stats['game_pk'].values,
                        'game_date': game_stats['game_date'].values}
        
        # Map of lambda positions to names
        lambda_mappings = {
            'zone': ['zone_rate', 'out_of_zone_rate', 'edge_rate'],
            'pitch_type': ['pitch_types_count', 'pitch_sequence_entropy'],
            'outs_when_up': ['pct_0_outs', 'pct_1_outs', 'pct_2_outs'],
            'balls': ['pct_0_balls', 'pct_3_balls', 'pct_hitters_counts'],
            'strikes': ['pct_0_strikes', 'pct_2_strikes', 'pct_pitchers_counts'],
            'events': ['strikeouts', 'walks', 'hbp', 'home_runs', 'hits', 
                    'field_outs', 'gidp', 'force_outs', 'double_plays'],
            'description': ['strike_rate', 'swinging_strike_rate', 'called_strike_rate',
                        'ball_rate', 'blocked_ball_rate', 'whiff_rate', 
                        'balls_in_play', 'foul_rate'],
            'pitch_number': ['pitch_count', 'avg_pitch_number', 'threw_100_plus']
        }
        
        # Process each column
        lambda_counters = {key: 0 for key in lambda_mappings}
        
        for col in game_stats.columns[2:]:  # Skip pitcher, game_pk (game_date is added separately)
            if isinstance(col, tuple):
                base_name, agg_type = col
                
                # Handle lambda functions
                if '<lambda' in str(agg_type):
                    if base_name in lambda_mappings:
                        idx = lambda_counters[base_name]
                        if idx < len(lambda_mappings[base_name]):
                            col_name = lambda_mappings[base_name][idx]
                        else:
                            col_name = f"{base_name}_lambda_{idx}"
                        lambda_counters[base_name] += 1
                    else:
                        col_name = f"{base_name}_lambda"
                # Handle special cases
                elif base_name == 'events' and agg_type == 'count':
                    col_name = 'batters_faced'
                elif base_name == 'launch_speed' and 'quantile' in str(agg_type):
                    col_name = 'exit_velo_90th_pct'
                elif base_name == 'launch_angle' and 'quantile' in str(agg_type):
                    col_name = 'launch_angle_10th_pct'
                elif base_name == 'hit_distance_sc' and 'quantile' in str(agg_type):
                    col_name = 'hit_distance_90th_pct'
                elif base_name in ['on_1b', 'on_2b', 'on_3b']:
                    col_name = base_name
                else:
                    col_name = f"{base_name}_{agg_type}"
                
                flattened_data[col_name] = game_stats[col].values
        
        # Create new dataframe with flattened columns
        game_stats = pd.DataFrame(flattened_data)
        
        # Calculate pitch mix if available
        if 'pitch_type' in pitcher_df.columns:
            logger.info("Calculating pitch mix percentages...")
            
            # Use only pitcher and game_pk for grouping
            pitch_type_counts = pitcher_df.groupby(['pitcher', 'game_pk'])['pitch_type'].apply(
                lambda x: x.value_counts(normalize=True).to_dict() if len(x) > 0 else {}
            ).reset_index()
            
            # Add game_date
            pitch_dates = pitcher_df.groupby(['pitcher', 'game_pk'])['game_date'].first().reset_index()
            pitch_type_counts = pitch_type_counts.merge(pitch_dates, on=['pitcher', 'game_pk'])
            
            # Rename the pitch_type column
            pitch_type_counts = pitch_type_counts.rename(columns={'pitch_type': 'pitch_mix_dict'})
            
            # Log sample pitch mix
            if len(pitch_type_counts) > 0:
                sample_mix = pitch_type_counts.iloc[0]['pitch_mix_dict']
                logger.info(f"Sample pitch mix: {sample_mix}")
            
            # Extract specific pitch types
            pitch_types = ['FF', 'SI', 'FC', 'SL', 'CU', 'CH', 'FS', 'KC', 'KN', 'FT', 'EP', 'SC', 'CS', 'PO', 'IN', 'AB', 'FA']
            
            for pitch in pitch_types:
                pitch_type_counts[f'pct_{pitch.lower()}'] = pitch_type_counts['pitch_mix_dict'].apply(
                    lambda x: x.get(pitch, 0) / 100 if isinstance(x, dict) else 0  # Convert to percentage
                )
            
            # Check if any pitch type percentages are non-zero
            pitch_pct_cols = [col for col in pitch_type_counts.columns if col.startswith('pct_')]
            non_zero_pitch_types = []
            for col in pitch_pct_cols:
                if (pitch_type_counts[col] > 0).any():
                    non_zero_pitch_types.append(col)
            
            logger.info(f"Non-zero pitch type columns: {non_zero_pitch_types}")
            
            # Merge
            game_stats = game_stats.merge(
                pitch_type_counts.drop(columns=['pitch_mix_dict']), 
                on=['pitcher', 'game_pk', 'game_date'], 
                how='left'
            )
        else:
            logger.warning("No pitch_type column found in pitcher data!")
        
        # Calculate bb_type percentages if available
        if 'bb_type' in pitcher_df.columns:
            logger.info("Calculating batted ball type percentages...")
            
            # Filter only balls in play
            balls_in_play = pitcher_df[pitcher_df['bb_type'].notna()]
            
            if len(balls_in_play) > 0:
                bb_type_counts = balls_in_play.groupby(['pitcher', 'game_pk'])['bb_type'].apply(
                    lambda x: x.value_counts(normalize=True).to_dict() if len(x) > 0 else {}
                ).reset_index()
                
                # Add game_date
                bb_dates = balls_in_play.groupby(['pitcher', 'game_pk'])['game_date'].first().reset_index()
                bb_type_counts = bb_type_counts.merge(bb_dates, on=['pitcher', 'game_pk'])
                
                # Rename the bb_type column
                bb_type_counts = bb_type_counts.rename(columns={'bb_type': 'bb_type_dict'})
                
                # Log sample bb_type mix
                if len(bb_type_counts) > 0 and bb_type_counts.iloc[0]['bb_type_dict']:
                    logger.info(f"Sample bb_type mix: {bb_type_counts.iloc[0]['bb_type_dict']}")
                
                # Extract bb_types
                bb_types = ['ground_ball', 'line_drive', 'fly_ball', 'popup']
                for bb in bb_types:
                    bb_type_counts[f'pct_{bb}'] = bb_type_counts['bb_type_dict'].apply(
                        lambda x: x.get(bb, 0) / 100 if isinstance(x, dict) else 0  # Convert to percentage
                    )
                
                # Merge
                game_stats = game_stats.merge(
                    bb_type_counts.drop(columns=['bb_type_dict']), 
                    on=['pitcher', 'game_pk', 'game_date'], 
                    how='left'
                )
        else:
            logger.warning("No bb_type column found in pitcher data!")
        
        # Additional calculations with fixed groupby patterns
        
        # Chase rate
        chase_rate_df = pitcher_df.groupby(['pitcher', 'game_pk', 'game_date']).apply(
            chase_rate_fn
        ).reset_index()
        if len(chase_rate_df.columns) == 4:
            chase_rate_df.columns = ['pitcher', 'game_pk', 'game_date', 'chase_rate_induced']
        else:
            chase_rate_df = chase_rate_df.rename(columns={0: 'chase_rate_induced'})
        game_stats = game_stats.merge(chase_rate_df, on=['pitcher', 'game_pk', 'game_date'], how='left')
        
        # First pitch strike rate
        fps_rate = pitcher_df.groupby(['pitcher', 'game_pk', 'game_date']).apply(
            first_pitch_strike_rate
        ).reset_index()
        if len(fps_rate.columns) == 4:
            fps_rate.columns = ['pitcher', 'game_pk', 'game_date', 'first_pitch_strike_rate']
        else:
            fps_rate = fps_rate.rename(columns={0: 'first_pitch_strike_rate'})
        game_stats = game_stats.merge(fps_rate, on=['pitcher', 'game_pk', 'game_date'], how='left')
        
        # Release point consistency with fixed merge
        if all(col in available_cols for col in ['release_pos_x', 'release_pos_z', 'pitch_type']):
            release_consistency = pitcher_df.groupby(['pitcher', 'game_pk', 'game_date']).apply(
                release_point_consistency
            ).reset_index()
            
            # Ensure the columns are named correctly after reset_index
            if len(release_consistency.columns) == 4:
                release_consistency.columns = ['pitcher', 'game_pk', 'game_date', 'release_point_consistency']
            else:
                # Handle case where apply might have created a different structure
                release_consistency = release_consistency.rename(columns={0: 'release_point_consistency'})
            
            # Verify the merge keys exist
            if all(col in release_consistency.columns for col in ['pitcher', 'game_pk', 'game_date']):
                game_stats = game_stats.merge(release_consistency, on=['pitcher', 'game_pk', 'game_date'], how='left')
            else:
                logger.warning(f"Release consistency merge skipped - missing columns. Available: {release_consistency.columns.tolist()}")
        
        # Velocity drop by inning
        if 'inning' in available_cols and 'release_speed' in available_cols:
            velo_by_inning_raw = pitcher_df.groupby(['pitcher', 'game_pk', 'game_date', 'inning'])['release_speed'].mean()
            if len(velo_by_inning_raw) > 0:
                velo_by_inning = velo_by_inning_raw.groupby(['pitcher', 'game_pk', 'game_date']).agg(['first', 'last'])
                velo_by_inning['velocity_drop'] = velo_by_inning['first'] - velo_by_inning['last']
                velo_by_inning = velo_by_inning.reset_index()[['pitcher', 'game_pk', 'game_date', 'velocity_drop']]
                game_stats = game_stats.merge(velo_by_inning, on=['pitcher', 'game_pk', 'game_date'], how='left')
        
        # Two-strike approach
        if 'strikes' in available_cols:
            two_strike_data = pitcher_df[pitcher_df['strikes'] == 2]
            if len(two_strike_data) > 0:
                two_strike_df = two_strike_data.groupby(['pitcher', 'game_pk', 'game_date']).agg({
                    'release_speed': 'mean',
                    'description': lambda x: (x.isin(['swinging_strike', 'swinging_strike_blocked'])).mean()
                })
                two_strike_df.columns = ['two_strike_velo', 'two_strike_whiff_rate']
                two_strike_df = two_strike_df.reset_index()
                
                # Two-strike pitch mix
                if 'pitch_type' in available_cols:
                    two_strike_pitch_df = two_strike_data.groupby(['pitcher', 'game_pk', 'game_date'])['pitch_type'].apply(
                        lambda x: x.value_counts(normalize=True).to_dict() if len(x) > 0 else {}
                    ).reset_index()
                    two_strike_pitch_df = two_strike_pitch_df.rename(columns={two_strike_pitch_df.columns[-1]: 'two_strike_pitch_dict'})
                    
                    for pitch in ['FF', 'SI', 'FC', 'SL', 'CU', 'CH', 'FS', 'KC', 'KN']:
                        two_strike_pitch_df[f'two_strike_pct_{pitch.lower()}'] = two_strike_pitch_df['two_strike_pitch_dict'].apply(
                            lambda x: x.get(pitch, 0) if isinstance(x, dict) else 0
                        )
                    two_strike_pitch_df = two_strike_pitch_df.drop(columns=['two_strike_pitch_dict'])
                    
                    two_strike_df = two_strike_df.merge(two_strike_pitch_df, on=['pitcher', 'game_pk', 'game_date'], how='left')
                
                game_stats = game_stats.merge(two_strike_df, on=['pitcher', 'game_pk', 'game_date'], how='left')
        
        # High leverage performance
        if 'delta_home_win_exp' in available_cols:
            high_leverage = pitcher_df[pitcher_df['delta_home_win_exp'].abs() > 0.05]
            if len(high_leverage) > 0:
                high_lev_stats = high_leverage.groupby(['pitcher', 'game_pk', 'game_date']).agg({
                    'events': lambda x: (x == 'strikeout').mean() if 'events' in high_leverage.columns else 0,
                    'release_speed': 'mean' if 'release_speed' in high_leverage.columns else lambda x: 0,
                    'zone': lambda x: (x.between(1, 9)).mean() if 'zone' in high_leverage.columns else 0
                })
                high_lev_stats.columns = ['high_lev_k_rate', 'high_lev_velo', 'high_lev_zone_rate']
                high_lev_stats = high_lev_stats.reset_index()
                game_stats = game_stats.merge(high_lev_stats, on=['pitcher', 'game_pk', 'game_date'], how='left')
        
        # Calculate derived metrics
        # Basic rate stats
        if 'strikeouts' in game_stats.columns and 'batters_faced' in game_stats.columns:
            game_stats['k_rate'] = safe_divide(game_stats['strikeouts'], game_stats['batters_faced'])
        if 'walks' in game_stats.columns and 'batters_faced' in game_stats.columns:
            game_stats['bb_rate'] = safe_divide(game_stats['walks'], game_stats['batters_faced'])
        if 'strikeouts' in game_stats.columns and 'walks' in game_stats.columns:
            game_stats['k_bb_ratio'] = safe_divide(game_stats['strikeouts'], game_stats['walks'].replace(0, 1))
        if 'home_runs' in game_stats.columns and 'batters_faced' in game_stats.columns:
            game_stats['hr_rate'] = safe_divide(game_stats['home_runs'], game_stats['batters_faced'])
        if all(col in game_stats.columns for col in ['walks', 'hits', 'inning_max']):
            game_stats['whip'] = safe_divide(game_stats['walks'] + game_stats['hits'], game_stats['inning_max'] / 3)
        
        # Efficiency metrics
        if 'pitch_count' in game_stats.columns and 'inning_max' in game_stats.columns:
            game_stats['pitches_per_inning'] = safe_divide(game_stats['pitch_count'], game_stats['inning_max'] / 3)
        if 'pitch_count' in game_stats.columns and 'batters_faced' in game_stats.columns:
            game_stats['pitches_per_batter'] = safe_divide(game_stats['pitch_count'], game_stats['batters_faced'])
        
        # Movement efficiency
        if 'effective_speed_mean' in game_stats.columns and 'release_speed_mean' in game_stats.columns:
            game_stats['velocity_efficiency'] = safe_divide(game_stats['effective_speed_mean'], game_stats['release_speed_mean'])
        if 'pfx_z_mean' in game_stats.columns and 'release_spin_rate_mean' in game_stats.columns:
            game_stats['spin_efficiency'] = safe_divide(game_stats['pfx_z_mean'], game_stats['release_spin_rate_mean'] / 120)
        
        # Command scores
        if 'plate_x_std' in game_stats.columns and 'plate_z_std' in game_stats.columns:
            game_stats['command_score'] = 1 / (game_stats['plate_x_std'] + game_stats['plate_z_std']).replace(0, 1)
            game_stats['location_consistency'] = 1 / np.sqrt(game_stats['plate_x_std']**2 + game_stats['plate_z_std']**2).replace(0, 1)
        
        # Quality start indicator
        if 'inning_max' in game_stats.columns:
            game_stats['quality_start'] = (game_stats['inning_max'] >= 18).astype(int)
        
        # Stuff+ components
        if 'release_speed_mean' in game_stats.columns:
            game_stats['velo_percentile'] = game_stats['release_speed_mean'].rank(pct=True)
        if 'pfx_x_std' in game_stats.columns and 'pfx_z_std' in game_stats.columns:
            game_stats['movement_score'] = np.sqrt(game_stats['pfx_x_std']**2 + game_stats['pfx_z_std']**2)
        
        # Fatigue indicators
        if 'velocity_drop' in game_stats.columns:
            game_stats['late_game_velo_drop'] = game_stats['velocity_drop'].fillna(0)
        if 'pitch_count' in game_stats.columns:
            game_stats['pitch_count_stress'] = game_stats['pitch_count'] / 100
        
        # Deception and tunneling
        if 'release_point_consistency' in game_stats.columns:
            game_stats['release_consistency_score'] = 1 / game_stats['release_point_consistency'].replace(0, 1)
        if 'release_speed_max' in game_stats.columns and 'release_speed_min' in game_stats.columns:
            game_stats['velocity_band_usage'] = game_stats['release_speed_max'] - game_stats['release_speed_min']
        
        # Situational performance
        if 'on_2b' in game_stats.columns and 'on_3b' in game_stats.columns:
            game_stats['risp_rate'] = (game_stats['on_2b'] + game_stats['on_3b']) / 2
        if all(col in game_stats.columns for col in ['walks', 'hits', 'inning_nunique']):
            game_stats['clean_inning_rate'] = 1 - safe_divide(game_stats['walks'] + game_stats['hits'], game_stats['inning_nunique'])
        
        # Contact management
        if 'pct_ground_ball' in game_stats.columns:
            game_stats['gb_rate'] = game_stats['pct_ground_ball']
        
        # Expected ERA components
        if 'estimated_woba_using_speedangle_mean' in game_stats.columns:
            game_stats['xERA_component'] = game_stats['estimated_woba_using_speedangle_mean'] * 10
        
        # CRITICAL: Final check for duplicates
        before_final = len(game_stats)
        game_stats = game_stats.drop_duplicates(subset=['pitcher', 'game_pk'])
        after_final = len(game_stats)
        
        if before_final != after_final:
            logger.warning(f"Removed {before_final - after_final} duplicate rows in final output!")
        
        # Fill NaN values with 0 for numeric columns
        numeric_columns = game_stats.select_dtypes(include=[np.number]).columns
        game_stats[numeric_columns] = game_stats[numeric_columns].fillna(0)
        
        # Log results
        logger.info(f"Calculated stats for {len(game_stats)} pitcher-games")
        logger.info(f"Output columns: {len(game_stats.columns)}")
        return game_stats
        
    import pandas as pd
    import numpy as np

    def calculate_bullpen_daily_stats(self, bullpen_df: pd.DataFrame,
                                    pitcher_teams: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive daily bullpen stats by team with advanced metrics"""
        
        # Validate input data
        if bullpen_df.empty:
            logger.warning("Empty bullpen dataframe provided")
            return pd.DataFrame()
        
        # Debug logging
        logger.info(f"Input bullpen_df shape: {bullpen_df.shape}")
        logger.info(f"Input pitcher_teams shape: {pitcher_teams.shape}")
        
        # Ensure consistent data types BEFORE merge
        bullpen_df['game_pk'] = pd.to_numeric(bullpen_df['game_pk'], errors='coerce')
        bullpen_df['pitcher'] = pd.to_numeric(bullpen_df['pitcher'], errors='coerce')
        pitcher_teams['game_pk'] = pd.to_numeric(pitcher_teams['game_pk'], errors='coerce')
        pitcher_teams['pitcher'] = pd.to_numeric(pitcher_teams['pitcher'], errors='coerce')
        
        # Check for duplicates in pitcher_teams before merge
        pitcher_teams_dedup = pitcher_teams.drop_duplicates(subset=['pitcher', 'game_pk'])
        if len(pitcher_teams_dedup) < len(pitcher_teams):
            logger.warning(f"Removed {len(pitcher_teams) - len(pitcher_teams_dedup)} duplicate pitcher-team mappings")
            pitcher_teams = pitcher_teams_dedup
        
        # Validate required columns
        required_columns = ['pitcher', 'game_pk', 'inning', 'events', 'release_speed']
        missing_columns = set(required_columns) - (set(bullpen_df.columns) | set(pitcher_teams.columns))
        if missing_columns:
            logger.warning(f"Missing columns {missing_columns}")
        
        # Helper functions
        def clean_inning_rate(df):
            """Calculate percentage of appearances with clean innings (no baserunners)"""
            if len(df) == 0:
                return 0
            
            bad_events = ['walk', 'hit_by_pitch', 'single', 'double', 'triple', 'home_run']
            
            # Group by pitcher and inning, check if all events in that inning are clean
            clean_by_inning = df.groupby(['pitcher', 'inning'])['events'].apply(
                lambda x: not x.isin(bad_events).any()
            )
            
            # Return the percentage of clean innings
            return clean_by_inning.mean()
        
        def high_leverage_performance(df):
            """Performance in high leverage situations"""
            if 'delta_home_win_exp' not in df.columns:
                return {}
            high_lev = df[df['delta_home_win_exp'].abs() > 0.05]
            if len(high_lev) == 0:
                return {}
            return {
                'k_rate': (high_lev['events'] == 'strikeout').sum() / len(high_lev),
                'bb_rate': (high_lev['events'] == 'walk').sum() / len(high_lev),
                'whiff_rate': high_lev['description'].isin(['swinging_strike', 'swinging_strike_blocked']).mean() if 'description' in high_lev.columns else 0
            }
        
        # Merge team info - only keep necessary columns from pitcher_teams
        team_info = pitcher_teams[['pitcher', 'game_pk', 'team']].drop_duplicates()
        
        logger.info(f"Before merge - bullpen_df: {len(bullpen_df)}, team_info: {len(team_info)}")
        
        bullpen_df = bullpen_df.merge(
            team_info,
            on=['pitcher', 'game_pk'],
            how='left'
        )
        
        logger.info(f"After merge - bullpen_df: {len(bullpen_df)}")
        
        # Check if merge created duplicates
        duplicate_check = bullpen_df.groupby(['pitcher', 'game_pk', 'at_bat_number', 'pitch_number']).size()
        if (duplicate_check > 1).any():
            logger.warning(f"Merge created {(duplicate_check > 1).sum()} duplicate entries!")
            # Remove duplicates if they exist
            bullpen_df = bullpen_df.drop_duplicates(subset=['pitcher', 'game_pk', 'at_bat_number', 'pitch_number'])
            logger.info(f"After deduplication: {len(bullpen_df)}")
        
        # Log unique team-game combinations
        unique_team_games = bullpen_df.groupby(['team', 'game_pk']).ngroups
        logger.info(f"Unique team-game combinations: {unique_team_games}")
        
        # Add pitcher appearance tracking
        pitcher_appearances = bullpen_df.groupby(['team', 'game_pk', 'pitcher']).agg({
            'inning': ['min', 'max', 'nunique'],
            'pitch_number': 'max',
            'at_bat_number': 'nunique'
        }).reset_index()
        
        pitcher_appearances.columns = ['team', 'game_pk', 'pitcher', 'entry_inning', 
                                    'exit_inning', 'innings_pitched', 'pitches_thrown', 'batters_faced']
        
        # Calculate multi-inning appearances
        pitcher_appearances['multi_inning'] = (pitcher_appearances['innings_pitched'] > 1).astype(int)
        
        # Use named aggregations for clarity and to avoid lambda naming issues
        agg_dict = {
            # Core velocity metrics
            'release_speed_mean': ('release_speed', 'mean'),
            'release_speed_std': ('release_speed', 'std'),
            'release_speed_max': ('release_speed', 'max'),
            'release_speed_min': ('release_speed', 'min'),
        }
        
        # Add optional columns if they exist
        if 'effective_speed' in bullpen_df.columns:
            agg_dict['effective_speed_mean'] = ('effective_speed', 'mean')
            agg_dict['effective_speed_max'] = ('effective_speed', 'max')
        
        if 'release_spin_rate' in bullpen_df.columns:
            agg_dict['release_spin_rate_mean'] = ('release_spin_rate', 'mean')
            agg_dict['release_spin_rate_std'] = ('release_spin_rate', 'std')
            agg_dict['release_spin_rate_max'] = ('release_spin_rate', 'max')
        
        # Movement profiles
        movement_cols = ['pfx_x', 'pfx_z', 'api_break_z_with_gravity', 'api_break_x_arm']
        for col in movement_cols:
            if col in bullpen_df.columns:
                agg_dict[f'{col}_mean'] = (col, 'mean')
                agg_dict[f'{col}_std'] = (col, 'std')
        
        # Release consistency
        release_cols = ['release_extension', 'release_pos_x', 'release_pos_z', 'arm_angle']
        for col in release_cols:
            if col in bullpen_df.columns:
                if col in ['release_pos_x', 'release_pos_z']:
                    agg_dict[f'{col}_std'] = (col, 'std')
                else:
                    agg_dict[f'{col}_mean'] = (col, 'mean')
                    agg_dict[f'{col}_std'] = (col, 'std')
        
        # Command metrics
        if 'plate_x' in bullpen_df.columns:
            agg_dict['plate_x_std'] = ('plate_x', 'std')
        if 'plate_z' in bullpen_df.columns:
            agg_dict['plate_z_std'] = ('plate_z', 'std')
        
        # Zone metrics
        if 'zone' in bullpen_df.columns:
            agg_dict['zone_rate'] = ('zone', lambda x: (x.between(1, 9)).mean())
            agg_dict['chase_rate_induced'] = ('zone', lambda x: (x.between(11, 14)).mean())
            agg_dict['corner_rate'] = ('zone', lambda x: ((x == 1) | (x == 3) | (x == 7) | (x == 9)).mean())
        
        # Pitch mix
        if 'pitch_type' in bullpen_df.columns:
            agg_dict['pitch_type_nunique'] = ('pitch_type', 'nunique')
        
        # Situational usage
        agg_dict['early_usage_rate'] = ('inning', lambda x: (x <= 6).mean())
        agg_dict['late_inning_rate'] = ('inning', lambda x: (x >= 8).mean())
        agg_dict['innings_coverage'] = ('inning', 'nunique')
        
        if 'outs_when_up' in bullpen_df.columns:
            agg_dict['clean_inning_start_rate'] = ('outs_when_up', lambda x: (x == 0).mean())
            agg_dict['mid_inning_entry_rate'] = ('outs_when_up', lambda x: (x > 0).mean())
        
        # Leverage situations
        leverage_cols = ['delta_home_win_exp', 'home_score_diff']
        for col in leverage_cols:
            if col in bullpen_df.columns:
                if col == 'delta_home_win_exp':
                    agg_dict['avg_leverage_index'] = (col, lambda x: x.abs().mean())
                    agg_dict['high_leverage_pct'] = (col, lambda x: (x.abs() > 0.05).mean())
                    agg_dict['max_leverage'] = (col, lambda x: x.abs().max())
                else:
                    agg_dict['close_game_usage'] = (col, lambda x: (x.abs() <= 1).mean())
                    agg_dict['blowout_loss_usage'] = (col, lambda x: (x < -3).mean())
                    agg_dict['blowout_win_usage'] = (col, lambda x: (x > 3).mean())
        
        # Base situation
        for base in ['on_1b', 'on_2b', 'on_3b']:
            if base in bullpen_df.columns:
                agg_dict[f'{base}_presence_rate'] = (base, lambda x: x.notna().mean())
        
        # Outcome metrics
        if 'events' in bullpen_df.columns:
            agg_dict['strikeouts'] = ('events', lambda x: (x == 'strikeout').sum())
            agg_dict['walks'] = ('events', lambda x: (x == 'walk').sum())
            agg_dict['hbp'] = ('events', lambda x: (x == 'hit_by_pitch').sum())
            agg_dict['home_runs'] = ('events', lambda x: (x == 'home_run').sum())
            agg_dict['hits'] = ('events', lambda x: (x.isin(['single', 'double', 'triple', 'home_run'])).sum())
            agg_dict['field_outs'] = ('events', lambda x: (x == 'field_out').sum())
            agg_dict['batters_faced'] = ('events', lambda x: x.notna().sum())
        
        # Pitch-level outcomes
        if 'description' in bullpen_df.columns:
            agg_dict['strike_rate'] = ('description', lambda x: (x.isin(['swinging_strike', 'called_strike', 'foul', 'foul_tip'])).mean())
            agg_dict['swinging_strike_rate'] = ('description', lambda x: (x.isin(['swinging_strike', 'swinging_strike_blocked'])).mean())
            agg_dict['called_strike_rate'] = ('description', lambda x: (x == 'called_strike').mean())
            agg_dict['ball_rate'] = ('description', lambda x: (x == 'ball').mean())
            agg_dict['balls_in_play'] = ('description', lambda x: (x == 'hit_into_play').sum())
        
        # Contact quality
        contact_cols = ['launch_speed', 'launch_angle', 'hit_distance_sc', 
                    'estimated_woba_using_speedangle', 'woba_value']
        for col in contact_cols:
            if col in bullpen_df.columns:
                agg_dict[f'{col}_mean'] = (col, 'mean')
                if col in ['launch_speed', 'hit_distance_sc', 'estimated_woba_using_speedangle']:
                    agg_dict[f'{col}_max'] = (col, 'max')
                if col == 'launch_speed':
                    agg_dict['exit_velo_90th_pct'] = (col, lambda x: x.quantile(0.9) if len(x) > 0 else None)
                if col == 'launch_angle':
                    agg_dict[f'{col}_std'] = (col, 'std')
                if col == 'woba_value':
                    agg_dict[f'{col}_sum'] = (col, 'sum')
        
        # Usage patterns
        agg_dict['pitcher_nunique'] = ('pitcher', 'nunique')
        agg_dict['total_pitches'] = ('pitch_number', 'count')
        agg_dict['avg_pitches_per_appearance'] = ('pitch_number', 'mean')
        
        # Rest and workload
        if 'pitcher_days_since_prev_game' in bullpen_df.columns:
            agg_dict['pitcher_days_since_prev_game_mean'] = ('pitcher_days_since_prev_game', 'mean')
            agg_dict['pitcher_days_since_prev_game_min'] = ('pitcher_days_since_prev_game', 'min')
        
        # At-bats
        if 'at_bat_number' in bullpen_df.columns:
            agg_dict['at_bat_number_nunique'] = ('at_bat_number', 'nunique')
        
        # Perform aggregation
        logger.info(f"Aggregating {len(agg_dict)} metrics for {unique_team_games} team-game combinations")
        
        daily_stats = bullpen_df.groupby(['team', 'game_pk']).agg(**agg_dict).reset_index()
        
        logger.info(f"Daily stats shape after aggregation: {daily_stats.shape}")
        
        # Calculate pitch mix percentages separately
        if 'pitch_type' in bullpen_df.columns:
            pitch_mix = bullpen_df.groupby(['team', 'game_pk'])['pitch_type'].apply(
                lambda x: x.value_counts(normalize=True).to_dict()
            ).reset_index(name='pitch_mix_dict')
            
            # Extract common pitch types
            pitch_types = ['FF', 'SI', 'FC', 'SL', 'CU', 'CH', 'FS']
            for pitch in pitch_types:
                pitch_mix[f'pct_{pitch.lower()}'] = pitch_mix['pitch_mix_dict'].apply(
                    lambda x: x.get(pitch, 0) if isinstance(x, dict) else 0
                )
            pitch_mix = pitch_mix.drop(columns=['pitch_mix_dict'])        
        # Additional pitcher-specific calculations
        
        # Back-to-back usage tracking
        back_to_back = bullpen_df.groupby(['team', 'game_pk', 'pitcher']).agg({
            'pitcher_days_since_prev_game': 'first'
        })
        back_to_back_usage = back_to_back[back_to_back['pitcher_days_since_prev_game'] == 1].groupby(['team', 'game_pk']).size()
        back_to_back_usage.name = 'back_to_back_appearances'
        
        # High velocity consistency
        high_velo_pitches = bullpen_df[bullpen_df['release_speed'] >= bullpen_df['release_speed'].quantile(0.8)]
        if len(high_velo_pitches) > 0:
            velo_consistency = high_velo_pitches.groupby(['team', 'game_pk']).agg({
                'release_speed': ['mean', 'std'],
                'pitcher': 'nunique'
            })
            # Flatten columns properly
            velo_consistency.columns = ['high_velo_mean', 'high_velo_std', 'high_velo_pitchers']
            velo_consistency = velo_consistency.reset_index()
        else:
            velo_consistency = pd.DataFrame()
        
        # Leverage-based performance
        leverage_stats = bullpen_df.groupby(['team', 'game_pk']).apply(
            lambda x: pd.Series(high_leverage_performance(x))
        ).add_prefix('high_leverage_')
        
        # Clean innings calculation
        clean_inning_pct = bullpen_df.groupby(['team', 'game_pk']).apply(clean_inning_rate)
        clean_inning_pct.name = 'clean_inning_rate'
        
        # Multi-inning appearance stats
        multi_inning_stats = pitcher_appearances.groupby(['team', 'game_pk']).agg({
            'multi_inning': 'sum',
            'pitches_thrown': ['mean', 'max'],
            'batters_faced': ['mean', 'max']
        })
        # Flatten columns properly
        multi_inning_stats.columns = ['multi_inning_appearances', 'avg_pitches_per_reliever', 
                                    'max_pitches_reliever', 'avg_batters_per_reliever', 'max_batters_reliever']
        multi_inning_stats = multi_inning_stats.reset_index()
        
        # Ensure consistent data types for merging
        daily_stats['game_pk'] = daily_stats['game_pk'].astype(str)
        pitch_mix['game_pk'] = pitch_mix['game_pk'].astype(str)
        
        # Merge all stats
        daily_stats = daily_stats.merge(pitch_mix, on=['team', 'game_pk'], how='left')
        
        # Handle back-to-back usage
        if len(back_to_back_usage) > 0:
            back_to_back_usage = back_to_back_usage.reset_index()
            back_to_back_usage['game_pk'] = back_to_back_usage['game_pk'].astype(str)
            daily_stats = daily_stats.merge(back_to_back_usage, on=['team', 'game_pk'], how='left')
            daily_stats['back_to_back_appearances'] = daily_stats['back_to_back_appearances'].fillna(0)
        else:
            daily_stats['back_to_back_appearances'] = 0
        
        # Handle velocity consistency
        if len(velo_consistency) > 0:
            velo_consistency['game_pk'] = velo_consistency['game_pk'].astype(str)
            daily_stats = daily_stats.merge(velo_consistency, on=['team', 'game_pk'], how='left')
            # Fill NaN values for merged columns
            for col in ['high_velo_mean', 'high_velo_std', 'high_velo_pitchers']:
                if col in daily_stats.columns:
                    daily_stats[col] = daily_stats[col].fillna(0)
        else:
            daily_stats['high_velo_mean'] = 0
            daily_stats['high_velo_std'] = 0
            daily_stats['high_velo_pitchers'] = 0
        
        # Handle leverage stats
        if len(leverage_stats) > 0:
            leverage_stats = leverage_stats.reset_index()
            leverage_stats['game_pk'] = leverage_stats['game_pk'].astype(str)
            
            # Ensure all column names are strings before merging
            leverage_stats.columns = leverage_stats.columns.astype(str)
            
            daily_stats = daily_stats.merge(leverage_stats, on=['team', 'game_pk'], how='left')
            
            # Fill NaN values for known leverage columns
            for col in ['high_leverage_k_rate', 'high_leverage_bb_rate', 'high_leverage_whiff_rate']:
                if col in daily_stats.columns:
                    daily_stats[col] = daily_stats[col].fillna(0)
                else:
                    daily_stats[col] = 0
        else:
            # Create default leverage columns
            daily_stats['high_leverage_k_rate'] = 0
            daily_stats['high_leverage_bb_rate'] = 0
            daily_stats['high_leverage_whiff_rate'] = 0
        
        # Handle clean inning percentage
        if len(clean_inning_pct) > 0:
            clean_inning_pct = clean_inning_pct.reset_index()
            clean_inning_pct['game_pk'] = clean_inning_pct['game_pk'].astype(str)
            daily_stats = daily_stats.merge(clean_inning_pct, on=['team', 'game_pk'], how='left')
            daily_stats['clean_inning_rate'] = daily_stats['clean_inning_rate'].fillna(0)
        else:
            daily_stats['clean_inning_rate'] = 0
        
        # Handle multi-inning stats
        if len(multi_inning_stats) > 0:
            multi_inning_stats['game_pk'] = multi_inning_stats['game_pk'].astype(str)
            daily_stats = daily_stats.merge(multi_inning_stats, on=['team', 'game_pk'], how='left')
            # Fill NaN values for multi-inning columns
            multi_cols = ['multi_inning_appearances', 'avg_pitches_per_reliever', 
                         'max_pitches_reliever', 'avg_batters_per_reliever', 'max_batters_reliever']
            for col in multi_cols:
                if col in daily_stats.columns:
                    daily_stats[col] = daily_stats[col].fillna(0)
        else:
            daily_stats['multi_inning_appearances'] = 0
            daily_stats['avg_pitches_per_reliever'] = 0
            daily_stats['max_pitches_reliever'] = 0
            daily_stats['avg_batters_per_reliever'] = 0
            daily_stats['max_batters_reliever'] = 0
        
        # Calculate runs scored from events and post_bat_score if available
        if 'post_bat_score' in bullpen_df.columns and 'bat_score' in bullpen_df.columns:
            # Calculate runs scored on each play
            bullpen_df['runs_scored'] = bullpen_df['post_bat_score'] - bullpen_df['bat_score']
        elif 'events' in bullpen_df.columns:
            # Estimate runs from events (simplified)
            run_scoring_events = {
                'home_run': 1,  # Plus runners on base
                'triple': 0.9,   # Often scores
                'double': 0.5,   # Sometimes scores
                'single': 0.3,   # Occasionally scores
                'sac_fly': 1,    # Scores runner from third
                'field_error': 0.3,
                'wild_pitch': 0.2,
                'passed_ball': 0.2
            }
            
            # Create a simple runs estimate
            bullpen_df['runs_scored'] = bullpen_df['events'].map(run_scoring_events).fillna(0)
            
            # Adjust for runners on base
            if all(col in bullpen_df.columns for col in ['on_1b', 'on_2b', 'on_3b']):
                runners_on = (
                    bullpen_df['on_1b'].notna().astype(int) +
                    bullpen_df['on_2b'].notna().astype(int) +
                    bullpen_df['on_3b'].notna().astype(int)
                )
                # Home runs score all runners
                home_run_mask = bullpen_df['events'] == 'home_run'
                if home_run_mask.any():
                    bullpen_df.loc[home_run_mask, 'runs_scored'] += runners_on.loc[home_run_mask]
        
        # Calculate inherited runners strand rate 
        if 'runs_scored' in bullpen_df.columns:
            # Ensure sorted for sequencing
            bullpen_df_sorted = bullpen_df.sort_values(['game_pk', 'inning', 'at_bat_number', 'pitch_number'])
            
            # Identify first pitch per reliever (mid-inning entry w/ runners on)
            first_pitches = bullpen_df_sorted.groupby(['game_pk', 'pitcher']).first().reset_index()
            
            # Filter mid-inning entries with runners on base
            inherited = first_pitches[(first_pitches['outs_when_up'] > 0) & (
                first_pitches[['on_1b', 'on_2b', 'on_3b']].notna().any(axis=1)
            )]
            
            if len(inherited) > 0:
                # Count inherited runners
                inherited['inherited_runners'] = (
                    inherited['on_1b'].notna().astype(int) +
                    inherited['on_2b'].notna().astype(int) +
                    inherited['on_3b'].notna().astype(int)
                )
                
                # Count runs allowed by each pitcher
                runs_by_pitcher = bullpen_df.groupby(['team', 'game_pk', 'pitcher'])['runs_scored'].sum().reset_index()
                runs_by_pitcher.rename(columns={'runs_scored': 'inherited_runners_scored'}, inplace=True)
                
                # Merge into inherited table
                inherited = inherited.merge(runs_by_pitcher, on=['team', 'game_pk', 'pitcher'], how='left')
                inherited['inherited_runners_scored'] = inherited['inherited_runners_scored'].fillna(0)
                
                # Group by team/game
                inherited_summary = inherited.groupby(['team', 'game_pk']).agg({
                    'inherited_runners': 'sum',
                    'inherited_runners_scored': 'sum'
                }).reset_index()
                
                # Calculate IRS%
                inherited_summary['inherited_runners_strand_rate'] = 1 - (
                    inherited_summary['inherited_runners_scored'] /
                    inherited_summary['inherited_runners'].replace(0, np.nan)
                )
                
                # Merge into daily_stats
                inherited_summary['game_pk'] = inherited_summary['game_pk'].astype(str)
                daily_stats = daily_stats.merge(
                    inherited_summary[['team', 'game_pk', 'inherited_runners_strand_rate']], 
                    on=['team', 'game_pk'], 
                    how='left'
                )
                # Fill NaN values (when inherited_runners is 0)
                daily_stats['inherited_runners_strand_rate'] = daily_stats['inherited_runners_strand_rate'].fillna(1.0)
            else:
                # No inherited runners situations
                daily_stats['inherited_runners_strand_rate'] = 1.0
        else:
            # Simple approximation based on runners on base presence
            daily_stats['inherited_runners_strand_rate'] = 1 - (
                daily_stats['on_2b_presence_rate'] + daily_stats['on_3b_presence_rate']
            ) / 2
        
        # Calculate derived bullpen metrics
        
        # Basic performance rates
        daily_stats['k_rate'] = daily_stats['strikeouts'] / daily_stats['batters_faced'].replace(0, 1)
        daily_stats['bb_rate'] = daily_stats['walks'] / daily_stats['batters_faced'].replace(0, 1)
        daily_stats['k_bb_ratio'] = daily_stats['strikeouts'] / daily_stats['walks'].replace(0, 1)
        daily_stats['hr_rate'] = daily_stats['home_runs'] / daily_stats['batters_faced'].replace(0, 1)
        daily_stats['whip'] = (daily_stats['walks'] + daily_stats['hits']) / (daily_stats['innings_coverage'].replace(0, 1) * 3)
       
        # Set relievers_used before using it in other calculations
        daily_stats['relievers_used'] = daily_stats['pitcher_nunique']
       
        # Bullpen efficiency
        daily_stats['pitches_per_out'] = daily_stats['total_pitches'] / (daily_stats['field_outs'] + daily_stats['strikeouts']).replace(0, 1)
       
        # Workload and usage patterns
        daily_stats['avg_rest_days'] = daily_stats['pitcher_days_since_prev_game_mean']
        daily_stats['overworked_appearances'] = daily_stats['back_to_back_appearances']  # Fixed: column guaranteed to exist
        daily_stats['bullpen_stress_index'] = (daily_stats['total_pitches'] / daily_stats['relievers_used'].replace(0, 1)) / 15
       
        # Velocity and stuff metrics
        daily_stats['velo_spread'] = daily_stats['release_speed_max'] - daily_stats['release_speed_min']
        daily_stats['spin_consistency'] = 1 / daily_stats['release_spin_rate_std'].replace(0, 1)
        daily_stats['command_score'] = 1 / (daily_stats['plate_x_std'] + daily_stats['plate_z_std']).replace(0, 1)
       
        # Leverage and clutch performance
        if 'high_leverage_k_rate' in daily_stats.columns and daily_stats['high_leverage_k_rate'].notna().any():
            daily_stats['leverage_weighted_k_rate'] = daily_stats['high_leverage_k_rate']
            daily_stats['clutch_score'] = (
                daily_stats['high_leverage_k_rate'] -
                daily_stats['high_leverage_bb_rate']  # Fixed: column guaranteed to exist
            )
        else:
            daily_stats['leverage_weighted_k_rate'] = daily_stats['k_rate']
            daily_stats['clutch_score'] = 0
       
        # Bullpen depth indicators
        daily_stats['bullpen_depth_score'] = (
            daily_stats['relievers_used'] *
            daily_stats['pitch_type_nunique'] /
            daily_stats['release_speed_std'].replace(0, 1)
        )
       
        # Role clarity (based on usage patterns)
        daily_stats['role_specialization'] = 1 - (daily_stats['innings_coverage'] / daily_stats['relievers_used'].replace(0, 1))
        daily_stats['high_leverage_specialist_rate'] = daily_stats['high_leverage_pct'] * daily_stats['late_inning_rate']
       
        # Contact management
        daily_stats['hard_contact_rate'] = daily_stats['exit_velo_90th_pct'] / 100
        daily_stats['expected_ops_against'] = daily_stats['estimated_woba_using_speedangle_mean'] * 2.5
       
        # Bullpen freshness index
        daily_stats['freshness_index'] = (
            daily_stats['avg_rest_days'] *
            (1 - daily_stats['overworked_appearances'] / daily_stats['relievers_used'].replace(0, 1))
        ).fillna(0)
        
        # Add game_date back for reference
        game_dates = bullpen_df.groupby(['team', 'game_pk'])['game_date'].first().reset_index()
        game_dates['game_pk'] = game_dates['game_pk'].astype(str)
        daily_stats = daily_stats.merge(game_dates, on=['team', 'game_pk'], how='left')
        
        # Final validation
        logger.info(f"Final daily_stats shape: {daily_stats.shape}")
        logger.info(f"Unique team-game combinations in output: {len(daily_stats)}")
        
        # Fill any remaining NaN values with sensible defaults
        numeric_columns = daily_stats.select_dtypes(include=[np.number]).columns
        daily_stats[numeric_columns] = daily_stats[numeric_columns].fillna(0)
        
        return daily_stats

    def _calculate_split_stats(self, split_df: pd.DataFrame, split_type: str) -> dict:
        """Calculate statistics for a specific split"""
        if len(split_df) == 0:
            return {}
        
        stats = {}
        
        # Clean numeric values before calculations
        numeric_cols = ['batting_runs', 'batting_ops', 'pitching_era']
        for col in numeric_cols:
            if col in split_df.columns:
                split_df[col] = pd.to_numeric(split_df[col], errors='coerce')
        
        # Calculate averages
        if 'batting_runs' in split_df.columns:
            stats[f'{split_type}_avg_runs'] = split_df['batting_runs'].mean()
        
        if 'batting_ops' in split_df.columns:
            stats[f'{split_type}_ops'] = split_df['batting_ops'].mean()
        
        if 'pitching_era' in split_df.columns:
            stats[f'{split_type}_era'] = split_df['pitching_era'].mean()
        
        return stats
        
    def create_splits_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create team performance splits using only past games"""
        logger.info("Creating team splits features...")
        
        # Sort by date to ensure proper ordering
        df = df.sort_values(['team_id', 'game_date'])
        
        # We'll create a mapping of game_pk to splits features
        splits_dict = {}
        
        for team_id in df['team_id'].unique():
            team_df = df[df['team_id'] == team_id].copy()
            for idx in range(len(team_df)):
                current_game = team_df.iloc[idx]
                game_pk = current_game['gamePk']
                game_date = current_game['game_date']
                
                # Get only past games for this team (exclude current game)
                past_games = team_df.iloc[:idx]  # Everything before current index
                
                if len(past_games) < 10:  # Minimum games for meaningful splits
                    continue
                
                # Calculate splits from past games only
                day_stats = self._calculate_split_stats(
                    past_games[past_games['dayNight'] == 'day'], 'day'
                )
                night_stats = self._calculate_split_stats(
                    past_games[past_games['dayNight'] == 'night'], 'night'
                )
                home_stats = self._calculate_split_stats(
                    past_games[past_games['side'] == 'home'], 'home'
                )
                away_stats = self._calculate_split_stats(
                    past_games[past_games['side'] == 'away'], 'away'
                )
                
                # Store splits for this game
                splits_dict[game_pk] = {
                    'team_id': team_id,
                    **day_stats,
                    **night_stats,
                    **home_stats,
                    **away_stats
                }
        
        # Convert to DataFrame
        splits_df = pd.DataFrame.from_dict(splits_dict, orient='index')
        splits_df.reset_index(inplace=True)
        splits_df.rename(columns={'index': 'gamePk'}, inplace=True)
        
        return splits_df

    def create_streak_features(self, df: pd.DataFrame, target_date: str = None) -> pd.DataFrame:
        """Create winning and losing streak features for teams"""
        logger.info("Creating winning and losing streak features...")
        
        # Ensure we have necessary columns
        if 'batting_runs' not in df.columns or 'pitching_runs' not in df.columns:
            logger.error("Missing required columns for streak calculation")
            return df
        
        # Create a win/loss indicator
        df['game_result'] = (df['batting_runs'] > df['pitching_runs']).astype(int)
        
        # Sort by team and date
        df = df.sort_values(['team_id', 'game_date'])
        
        def calculate_streaks_no_leakage(group, target_date_pd=None):
            """Calculate streaks using only past games"""
            results = group['game_result'].values
            dates = pd.to_datetime(group['game_date'].values)
            
            # Initialize arrays
            current_streak = np.zeros(len(results))
            streak_type = np.zeros(len(results))
            last_10_wins = np.zeros(len(results))
            last_20_wins = np.zeros(len(results))
            last_5_wins = np.zeros(len(results))
            
            for i in range(len(results)):
                # If we have a target date and this game is on or after it
                if target_date_pd is not None and dates[i] >= target_date_pd:
                    # Use only data up to the day before target date
                    historical_mask = dates < target_date_pd
                    historical_results = results[historical_mask]
                    
                    if len(historical_results) == 0:
                        # No history
                        current_streak[i] = 0
                        streak_type[i] = 0
                        last_5_wins[i] = 0
                        last_10_wins[i] = 0
                        last_20_wins[i] = 0
                    else:
                        # Calculate based on historical data only
                        # Current streak
                        if len(historical_results) == 1:
                            current_streak[i] = 1 if historical_results[-1] == 1 else -1
                            streak_type[i] = 1 if historical_results[-1] == 1 else -1
                        else:
                            # Check if last two historical games had same result
                            if historical_results[-1] == historical_results[-2]:
                                # Find the streak length
                                streak_count = 1
                                for j in range(len(historical_results) - 2, -1, -1):
                                    if historical_results[j] == historical_results[-1]:
                                        streak_count += 1
                                    else:
                                        break
                                current_streak[i] = streak_count if historical_results[-1] == 1 else -streak_count
                                streak_type[i] = 1 if historical_results[-1] == 1 else -1
                            else:
                                current_streak[i] = 1 if historical_results[-1] == 1 else -1
                                streak_type[i] = 1 if historical_results[-1] == 1 else -1
                        
                        # Rolling wins
                        last_5_wins[i] = historical_results[-5:].sum() if len(historical_results) >= 5 else historical_results.sum()
                        last_10_wins[i] = historical_results[-10:].sum() if len(historical_results) >= 10 else historical_results.sum()
                        last_20_wins[i] = historical_results[-20:].sum() if len(historical_results) >= 20 else historical_results.sum()
                else:
                    # Historical game or no target date - calculate normally
                    if i == 0:
                        current_streak[i] = 0
                        streak_type[i] = 0
                        last_5_wins[i] = 0
                        last_10_wins[i] = 0
                        last_20_wins[i] = 0
                    else:
                        # Current streak based on previous games
                        if i == 1:
                            current_streak[i] = 1 if results[0] == 1 else -1
                            streak_type[i] = 1 if results[0] == 1 else -1
                        else:
                            if results[i-1] == results[i-2]:
                                if results[i-1] == 1:
                                    current_streak[i] = abs(current_streak[i-1]) + 1
                                    streak_type[i] = 1
                                else:
                                    current_streak[i] = -(abs(current_streak[i-1]) + 1)
                                    streak_type[i] = -1
                            else:
                                current_streak[i] = 1 if results[i-1] == 1 else -1
                                streak_type[i] = 1 if results[i-1] == 1 else -1
                        
                        # Rolling wins excluding current game
                        start_5 = max(0, i - 5)
                        start_10 = max(0, i - 10)
                        start_20 = max(0, i - 20)
                        
                        last_5_wins[i] = results[start_5:i].sum()
                        last_10_wins[i] = results[start_10:i].sum()
                        last_20_wins[i] = results[start_20:i].sum()
            
            # Create result dataframe
            result_df = pd.DataFrame({
                'current_streak': current_streak,
                'win_streak': np.where(streak_type == 1, current_streak, 0),
                'loss_streak': np.where(streak_type == -1, np.abs(current_streak), 0),
                'is_on_win_streak': (streak_type == 1).astype(int),
                'is_on_loss_streak': (streak_type == -1).astype(int),
                'last_10_wins': last_10_wins,
                'last_20_wins': last_20_wins,
                'last_5_wins': last_5_wins,
                'last_10_win_pct': last_10_wins / np.maximum(1, np.minimum(10, np.arange(1, len(results) + 1))),
                'last_20_win_pct': last_20_wins / np.maximum(1, np.minimum(20, np.arange(1, len(results) + 1))),
                'momentum_5_game': last_5_wins / np.maximum(1, np.minimum(5, np.arange(1, len(results) + 1)))
            })
            
            return result_df
        
        # Apply calculation to each team
        streak_dfs = []
        target_date_pd = pd.to_datetime(target_date) if target_date else None
        
        for team_id in df['team_id'].unique():
            team_df = df[df['team_id'] == team_id].copy()
            
            if len(team_df) < 1:
                continue
            
            # Calculate streaks
            streak_features = calculate_streaks_no_leakage(team_df, target_date_pd)
            
            # Add identifiers
            streak_features['team_id'] = team_id
            streak_features['game_pk'] = team_df['gamePk'].values
            streak_features['game_date'] = team_df['game_date'].values
            
            streak_dfs.append(streak_features)
        
        if not streak_dfs:
            logger.warning("No streak features created - insufficient data")
            return df
            
        # Combine all streak data
        all_streaks = pd.concat(streak_dfs, ignore_index=True)
        
        # Fix column naming consistency
        if 'gamePk' in df.columns and 'game_pk' not in df.columns:
            df['game_pk'] = df['gamePk']
        
        # Merge back with original dataframe
        df = df.merge(
            all_streaks,
            on=['team_id', 'game_pk', 'game_date'],
            how='left'
        )
        
        # Create additional derived features
        logger.info("Creating derived streak features...")
        
        # Streak intensity features
        df['streak_intensity'] = df['current_streak'].abs()
        df['streak_pressure'] = df['streak_intensity'] * df['is_on_loss_streak']
        df['streak_confidence'] = df['streak_intensity'] * df['is_on_win_streak']
        
        # Momentum shift indicators
        df['potential_streak_breaker'] = ((df['is_on_loss_streak'] == 1) & (df['momentum_5_game'] > 0.4)).astype(int)
        df['vulnerable_to_upset'] = ((df['is_on_win_streak'] == 1) & (df['momentum_5_game'] < 0.6) & (df['win_streak'] >= 5)).astype(int)
        
        # Extreme streak indicators
        df['extreme_win_streak'] = (df['win_streak'] >= 7).astype(int)
        df['extreme_loss_streak'] = (df['loss_streak'] >= 5).astype(int)
        
        # Historical context features
        df['above_500_last_20'] = (df['last_20_win_pct'] > 0.5).astype(int)
        df['below_400_last_20'] = (df['last_20_win_pct'] < 0.4).astype(int)
        
        # Consistency metric - standard deviation of results in rolling windows
        consistency_dfs = []
        
        for team_id in df['team_id'].unique():
            team_df = df[df['team_id'] == team_id].sort_values('game_date')
            
            if len(team_df) >= 10:
                # Calculate rolling std of game results (excluding current game)
                consistency_values = []
                for i in range(len(team_df)):
                    if i < 5:
                        consistency_values.append(0.5)  # Default for insufficient data
                    else:
                        window_start = max(0, i - 10)
                        window_data = team_df['game_result'].iloc[window_start:i]  # Exclude current
                        consistency_values.append(1 - window_data.std() if len(window_data) > 1 else 0.5)
                
                consistency_df = pd.DataFrame({
                    'team_id': team_id,
                    'game_pk': team_df['game_pk'].values,
                    'result_consistency': consistency_values
                })
                consistency_dfs.append(consistency_df)
        
        if consistency_dfs:
            all_consistency = pd.concat(consistency_dfs, ignore_index=True)
            df = df.merge(all_consistency, on=['team_id', 'game_pk'], how='left')
            df['result_consistency'] = df['result_consistency'].fillna(0.5)
        else:
            df['result_consistency'] = 0.5
        
        # Team psychological state indicators
        df['team_psychological_state'] = 'neutral'
        df.loc[df['win_streak'] >= 5, 'team_psychological_state'] = 'hot'
        df.loc[df['loss_streak'] >= 4, 'team_psychological_state'] = 'cold'
        df.loc[df['momentum_5_game'] >= 0.8, 'team_psychological_state'] = 'surging'
        df.loc[df['momentum_5_game'] <= 0.2, 'team_psychological_state'] = 'slumping'
        
        # One-hot encode psychological states
        psych_dummies = pd.get_dummies(df['team_psychological_state'], prefix='team_state')
        df = pd.concat([df, psych_dummies], axis=1)
        
        # Clean up temporary columns
        df = df.drop(columns=['game_result', 'team_psychological_state'])
        
        logger.info(f"Successfully created {len(all_streaks.columns)} streak-related features")
        
        return df

    # def create_head_to_head_streak_features(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """Create streak features for head-to-head matchups between teams"""
    #     logger.info("Creating head-to-head streak features...")
        
    #     # Create matchup identifier
    #     df['matchup_id'] = df.apply(
    #         lambda x: '_'.join(sorted([str(x['team_id']), str(x['opponent_team_id'])])),
    #         axis=1
    #     )
        
    #     h2h_dfs = []
        
    #     for matchup in df['matchup_id'].unique():
    #         matchup_df = df[df['matchup_id'] == matchup].sort_values('game_date')
            
    #         if len(matchup_df) < 2:
    #             continue
                
    #         # Calculate head-to-head streaks for each team in the matchup
    #         for team in matchup_df['team_id'].unique():
    #             team_matchup_df = matchup_df[matchup_df['team_id'] == team].copy()
                
    #             if len(team_matchup_df) < 1:
    #                 continue
                    
    #             # Win/loss in this matchup
    #             team_matchup_df['h2h_result'] = (
    #                 team_matchup_df['batting_runs'] > team_matchup_df['pitching_runs']
    #             ).astype(int)
                
    #             # Calculate features for each game excluding current result
    #             h2h_features_list = []
                
    #             for i in range(len(team_matchup_df)):
    #                 if i == 0:
    #                     # First game - no history
    #                     h2h_features = {
    #                         'team_id': team,
    #                         'game_pk': team_matchup_df.iloc[i]['game_pk'],
    #                         'h2h_current_streak': 0,
    #                         'h2h_win_streak': 0,
    #                         'h2h_loss_streak': 0,
    #                         'h2h_wins_last5': 0,
    #                         'h2h_wins_last10': 0,
    #                         'h2h_dominance_score': 0
    #                     }
    #                 else:
    #                     # Use only games before current
    #                     past_results = team_matchup_df['h2h_result'].iloc[:i]
                        
    #                     # Calculate current streak
    #                     if len(past_results) == 1:
    #                         h2h_streak = 1 if past_results.iloc[0] == 1 else -1
    #                     else:
    #                         # Check if last two games had same result
    #                         if past_results.iloc[-1] == past_results.iloc[-2]:
    #                             # Count streak length
    #                             streak_count = 1
    #                             for j in range(len(past_results) - 2, -1, -1):
    #                                 if past_results.iloc[j] == past_results.iloc[-1]:
    #                                     streak_count += 1
    #                                 else:
    #                                     break
    #                             h2h_streak = streak_count if past_results.iloc[-1] == 1 else -streak_count
    #                         else:
    #                             h2h_streak = 1 if past_results.iloc[-1] == 1 else -1
                        
    #                     # Calculate rolling wins
    #                     h2h_wins_last5 = past_results.tail(5).sum() if len(past_results) >= 5 else past_results.sum()
    #                     h2h_wins_last10 = past_results.tail(10).sum() if len(past_results) >= 10 else past_results.sum()
                        
    #                     h2h_features = {
    #                         'team_id': team,
    #                         'game_pk': team_matchup_df.iloc[i]['game_pk'],
    #                         'h2h_current_streak': h2h_streak,
    #                         'h2h_win_streak': h2h_streak if h2h_streak > 0 else 0,
    #                         'h2h_loss_streak': abs(h2h_streak) if h2h_streak < 0 else 0,
    #                         'h2h_wins_last5': h2h_wins_last5,
    #                         'h2h_wins_last10': h2h_wins_last10,
    #                         'h2h_dominance_score': h2h_wins_last10 / min(10, i) if i > 0 else 0
    #                     }
                    
    #                 h2h_features_list.append(h2h_features)
                
    #             if h2h_features_list:
    #                 h2h_dfs.append(pd.DataFrame(h2h_features_list))
        
    #     if h2h_dfs:
    #         all_h2h = pd.concat(h2h_dfs, ignore_index=True)
    #         df = df.merge(all_h2h, on=['team_id', 'game_pk'], how='left')
            
    #         # Fill NaN values for teams with no h2h history
    #         h2h_columns = [col for col in df.columns if col.startswith('h2h_')]
    #         df[h2h_columns] = df[h2h_columns].fillna(0)
        
    #     # Clean up
    #     df = df.drop(columns=['matchup_id'])
        
    #     return df
        

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
import multiprocessing as mp
import joblib
import os
import pickle
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class TeamStateArrays:
    """
    MODIFIED: Stores the FULL history for a team's rolling features using dynamic lists.
    This prevents data from being overwritten or truncated.
    """
    feature_names: List[str]
    # Use field from dataclasses to initialize mutable defaults like lists correctly
    games: List[np.ndarray] = field(default_factory=list)
    timestamps: List[int] = field(default_factory=list)

    def add_game(self, features: np.ndarray, timestamp: pd.Timestamp):
        """Appends a new game's features and timestamp to the history lists."""
        self.games.append(features)
        self.timestamps.append(timestamp.value)

    def get_rolling_stats_before(self, target_date: pd.Timestamp, rolling_window_size: int) -> Dict[str, float]:
        """
        Calculates rolling statistics using ONLY data before the target_date
        from the complete stored history.
        """
        if not self.games:
            return {}

        # Convert lists to numpy arrays for efficient filtering
        all_timestamps = np.array(self.timestamps, dtype=np.int64)
        all_data = np.array(self.games)

        # Filter for games that occurred strictly before the target date
        target_ts = pd.Timestamp(target_date).value
        before_mask = all_timestamps < target_ts
        
        # If there's no data before the target date, return empty
        if not np.any(before_mask):
            return {}

        # Select only the relevant historical data using the mask
        # No need to sort here because we append chronologically
        valid_historical_data = all_data[before_mask]
        
        # Get the most recent N games from the valid historical data
        recent_data = valid_historical_data[-rolling_window_size:]
        
        if recent_data.shape[0] == 0:
            return {}

        # Calculate rolling stats
        stats = {}
        for i, name in enumerate(self.feature_names):
            # Select the i-th column from the recent data
            col_data = recent_data[:, i]
            
            # Filter out NaN values before calculating the mean
            non_nan_mask = ~np.isnan(col_data)
            if np.any(non_nan_mask):
                stats[f"{name}_roll{rolling_window_size}"] = np.mean(col_data[non_nan_mask])
        
        return stats


class CompleteOptimizedFeatureBuilder:
    """Complete feature builder with ALL features + optimizations"""
    
    def __init__(self, config: PipelineConfig, n_workers: int = None):
        self.config = config
        self.db = DatabaseConnection(config)
        self.feature_engineer = FeatureEngineer(config)
        self.validator = DataValidator()
        
        # Load mappings
        if self.db.connect():
            self.full_name_to_id, self.abbrev_to_id = self.db.load_team_mappings()
        else:
            raise ConnectionError("Failed to connect to database")
        
        # Set workers
        self.n_workers = n_workers or min(mp.cpu_count() - 1, 4)
        
        # Cache directories (keeping your structure)
        self.incremental_cache_dir = os.path.join(config.output_dir, 'incremental_features')
        self.state_cache_dir = os.path.join(config.output_dir, 'feature_states')
        os.makedirs(self.incremental_cache_dir, exist_ok=True)
        os.makedirs(self.state_cache_dir, exist_ok=True)
        
        # Master feature table path
        self.master_table_path = os.path.join(config.output_dir, 'master_features_table.parquet')
        
        # Feature state tracking - now with numpy arrays for performance
        self.feature_states = {
            'team_rolling': {},     # team_id -> TeamStateArrays
            'team_streaks': {},     # team_id -> {'results': np.array, 'features': dict}
            'pitcher_rolling': {},  # pitcher_id -> numpy arrays
            'bullpen_rolling': {},  # team -> numpy arrays
            'team_splits': {},      # team_id -> optimized split tracking
            'h2h_streaks': {}       # matchup_key -> numpy arrays
        }
        self.last_processed_date = None
        
        # Define feature columns for numpy arrays
        self.team_features = ['batting_runs', 'batting_hits', 'batting_doubles', 'batting_triples',
                             'batting_homeRuns', 'batting_strikeOuts', 'batting_walks', 'batting_avg',
                             'batting_ops', 'batting_slg', 'batting_obp', 'batting_rbi',
                             'pitching_runs', 'pitching_hits', 'pitching_strikeOuts', 
                             'pitching_walks', 'pitching_homeRuns', 'pitching_earnedRuns',
                             'pitching_era', 'pitching_whip', 'fielding_errors', 'fielding_doublePlays']

    def _get_pitcher_features_before_date(self, pitcher_id: int, target_date: pd.Timestamp) -> dict:
        """Get pitcher rolling features using only games before target date"""
        if pitcher_id not in self.feature_states.get('pitcher_rolling', {}):
            return {}
        
        state = self.feature_states['pitcher_rolling'][pitcher_id]
        
        if 'games' not in state:
            return {}
        
        # Filter games before target date
        valid_games = [
            game for game in state['games'] 
            if pd.to_datetime(game['date']) < target_date
        ]
        
        if len(valid_games) == 0:
            return {}
        
        # Get last N games
        recent_games = valid_games[-self.config.rolling_games_pitcher:]
        
        # Calculate rolling features
        features = {}
        all_cols = set()
        for game in recent_games:
            all_cols.update(game['data'].keys())
        
        for col in all_cols:
            values = [g['data'].get(col, np.nan) for g in recent_games]
            values = [v for v in values if not np.isnan(v)]
            if values:
                features[f"SP_{col}_roll{self.config.rolling_games_pitcher}"] = np.mean(values)
        
        return features

    def _get_bullpen_features_before_date(self, team: str, target_date: pd.Timestamp) -> dict:
        """Get bullpen rolling features using only games before target date"""
        if team not in self.feature_states.get('bullpen_rolling', {}):
            return {}
        
        state = self.feature_states['bullpen_rolling'][team]
        
        if 'daily_stats' not in state:
            return {}
        
        # Calculate window
        window_start = target_date - pd.Timedelta(days=self.config.rolling_days_bullpen)
        
        # Filter stats in rolling window before target date
        valid_stats = [
            stat for stat in state['daily_stats']
            if window_start <= pd.to_datetime(stat['date']) < target_date
        ]
        
        if len(valid_stats) < 3:  # min_periods
            return {}
        
        # Calculate rolling features
        features = {}
        all_cols = set()
        for stat in valid_stats:
            all_cols.update(stat['data'].keys())
        
        for col in all_cols:
            values = [s['data'].get(col, np.nan) for s in valid_stats]
            values = [v for v in values if not np.isnan(v)]
            if values:
                features[f"bullpen_{col}_roll{self.config.rolling_days_bullpen}d"] = np.mean(values)
        
        return features

    # 4. Complete fixed _get_splits_features_before_date
    def _get_splits_features_before_date(self, team_id: int, target_date: pd.Timestamp) -> dict:
        """Get splits features using only games before target date"""
        if team_id not in self.feature_states.get('team_splits', {}):
            return {}
        
        state = self.feature_states['team_splits'][team_id]
        
        if 'games' not in state:
            return {}
        
        # Filter games in 60-day window before target date
        cutoff_date = target_date - pd.Timedelta(days=60)
        valid_games = [
            game for game in state['games']
            if cutoff_date < pd.to_datetime(game['date']) < target_date
        ]
        
        if len(valid_games) < 10:  # minimum games for meaningful splits
            return {}
        
        # Convert to DataFrame for easier calculation
        games_df = pd.DataFrame(valid_games)
        
        features = {}
        
        # Day/Night splits
        for time in ['day', 'night']:
            time_games = games_df[games_df['dayNight'] == time]
            if len(time_games) > 0:
                features[f'{time}_avg_runs'] = float(time_games['batting_runs'].mean())
                features[f'{time}_ops'] = float(time_games['batting_ops'].mean())
                features[f'{time}_era'] = float(time_games['pitching_era'].mean())
        
        # Home/Away splits
        for side in ['home', 'away']:
            side_games = games_df[games_df['side'] == side]
            if len(side_games) > 0:
                features[f'{side}_avg_runs'] = float(side_games['batting_runs'].mean())
                features[f'{side}_ops'] = float(side_games['batting_ops'].mean())
                features[f'{side}_era'] = float(side_games['pitching_era'].mean())
        
        return features

    # def _get_h2h_features_before_date(self, matchup_key: str, team_id: int, target_date: pd.Timestamp) -> dict:
    #     """Get H2H features using only games before target date"""
    #     # Default empty features
    #     empty_features = {
    #         'h2h_current_streak': 0,
    #         'h2h_win_streak': 0,
    #         'h2h_loss_streak': 0,
    #         'h2h_wins_last5': 0,
    #         'h2h_wins_last10': 0,
    #         'h2h_dominance_score': 0
    #     }
        
    #     if matchup_key not in self.feature_states.get('h2h_streaks', {}):
    #         return empty_features
        
    #     if team_id not in self.feature_states['h2h_streaks'][matchup_key]:
    #         return empty_features
        
    #     state = self.feature_states['h2h_streaks'][matchup_key][team_id]
        
    #     if 'results' not in state or 'dates' not in state:
    #         return empty_features
        
    #     # Get results and dates
    #     results = state['results']
    #     dates = state.get('dates', [])
        
    #     if len(dates) != len(results) or len(dates) == 0:
    #         return empty_features
        
    #     try:
    #         # Try to safely convert dates
    #         # Handle various date formats and filter out invalid entries
    #         valid_dates = []
    #         valid_results = []
            
    #         for i, date in enumerate(dates):
    #             try:
    #                 # Convert date to pandas timestamp then to numpy datetime64
    #                 if isinstance(date, np.datetime64):
    #                     valid_date = date
    #                 elif isinstance(date, (pd.Timestamp, datetime.datetime)):
    #                     valid_date = np.datetime64(date, 'D')
    #                 elif isinstance(date, str):
    #                     valid_date = np.datetime64(pd.to_datetime(date), 'D')
    #                 else:
    #                     # Skip invalid dates
    #                     continue
                    
    #                 valid_dates.append(valid_date)
    #                 valid_results.append(results[i])
    #             except:
    #                 # Skip any problematic dates
    #                 continue
            
    #         if len(valid_dates) == 0:
    #             return empty_features
            
    #         # Convert to numpy arrays
    #         dates_array = np.array(valid_dates, dtype='datetime64[D]')
    #         results_array = np.array(valid_results)
            
    #         # Filter by target date
    #         target_date_np = np.datetime64(target_date, 'D')
    #         valid_mask = dates_array < target_date_np
            
    #         filtered_results = results_array[valid_mask]
            
    #         if len(filtered_results) == 0:
    #             return empty_features
            
    #         # Calculate H2H features
    #         # Current streak
    #         h2h_streak = 1 if filtered_results[-1] == 1 else -1
    #         for i in range(len(filtered_results) - 2, -1, -1):
    #             if filtered_results[i] == filtered_results[-1]:
    #                 h2h_streak = h2h_streak + 1 if filtered_results[-1] == 1 else h2h_streak - 1
    #             else:
    #                 break
            
    #         features = {
    #             'h2h_current_streak': int(h2h_streak),
    #             'h2h_win_streak': int(h2h_streak) if h2h_streak > 0 else 0,
    #             'h2h_loss_streak': int(abs(h2h_streak)) if h2h_streak < 0 else 0,
    #             'h2h_wins_last5': int(filtered_results[-5:].sum()) if len(filtered_results) >= 5 else int(filtered_results.sum()),
    #             'h2h_wins_last10': int(filtered_results[-10:].sum()) if len(filtered_results) >= 10 else int(filtered_results.sum()),
    #             'h2h_dominance_score': float(filtered_results.mean())
    #         }
            
    #         return features
            
    #     except Exception as e:
    #         logger.warning(f"Error calculating H2H features for {matchup_key}/{team_id}: {str(e)}")
    #         return empty_features
        
    def build_features_incrementally(self, start_date: str, end_date: str, 
                                   use_cache: bool = True, rebuild: bool = False) -> pd.DataFrame:
        """Build features incrementally with optimized batch processing"""
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Load or initialize master table
        if os.path.exists(self.master_table_path) and not rebuild:
            logger.info(f"Loading existing master feature table from {self.master_table_path}")
            master_df = pd.read_parquet(self.master_table_path)
            
            if not master_df.empty:
                self.last_processed_date = pd.to_datetime(master_df['game_date']).max()
                logger.info(f"Last processed date in master table: {self.last_processed_date}")
                
                # Return existing data if already processed
                if self.last_processed_date >= end_dt:
                    logger.info("All requested data already processed")
                    return master_df[
                        (pd.to_datetime(master_df['game_date']) >= start_dt) & 
                        (pd.to_datetime(master_df['game_date']) <= end_dt)
                    ]
                
                # Adjust start date
                if self.last_processed_date >= start_dt:
                    start_dt = self.last_processed_date + pd.Timedelta(days=1)
                    logger.info(f"Adjusting start date to {start_dt}")
        else:
            master_df = pd.DataFrame()
            logger.info("Starting fresh - no existing master table found")
        
        # Load feature states
        self._load_feature_states_optimized()
        
        # Initialize with warm-up period if needed
        if self.last_processed_date is None or rebuild:
            logger.info("Initializing with warm-up period...")
            self._initialize_warm_up_period_optimized(start_dt)
        
        # OPTIMIZATION: Batch load all data
        logger.info(f"Batch loading data from {start_dt} to {end_dt}...")
        all_baseball = self.db.load_baseball_scrape_data(
            start_dt.strftime('%Y-%m-%d'),
            end_dt.strftime('%Y-%m-%d')
        )
        all_statcast = self.db.load_statcast_data(
            start_dt.strftime('%Y-%m-%d'),
            end_dt.strftime('%Y-%m-%d')
        )
        
        if all_baseball.empty:
            logger.warning("No baseball data found for date range")
            return master_df
        
        # Clean data once
        all_statcast = self.validator.validate_statcast_data(all_statcast) if not all_statcast.empty else all_statcast
        
        # Process chronologically to prevent leakage
        all_baseball['game_date_parsed'] = pd.to_datetime(all_baseball['game_date'])
        all_baseball = all_baseball.sort_values(['game_date_parsed', 'gamePk'])
        
        unique_dates = sorted(all_baseball['game_date_parsed'].unique())
        all_features = []
        
        # Process in chunks for efficiency while maintaining chronological order
        chunk_size = 7  # Process a week at a time
        date_chunks = []
        current_chunk = []
        
        for i, date in enumerate(unique_dates):
            if not self._is_baseball_season(date):
                continue
                
            current_chunk.append(date)
            
            # Create chunk when size reached or last date
            if len(current_chunk) >= chunk_size or i == len(unique_dates) - 1:
                if current_chunk:
                    date_chunks.append(current_chunk)
                    current_chunk = []
        
        # Process each chunk
        for chunk_idx, date_chunk in enumerate(date_chunks):
            logger.info(f"Processing chunk {chunk_idx + 1}/{len(date_chunks)}: "
                       f"{date_chunk[0].strftime('%Y-%m-%d')} to {date_chunk[-1].strftime('%Y-%m-%d')}")
            
            chunk_features = self._process_date_chunk_safe(
                date_chunk, all_baseball, all_statcast
            )
            
            if chunk_features:
                all_features.extend(chunk_features)
            
            # Save states periodically
            if chunk_idx % 4 == 0:  # Every 4 weeks
                self._save_feature_states_optimized()
        
        # Combine and save
        if all_features:
            new_features_df = pd.concat(all_features, ignore_index=True)
            
            # Add betting odds for recent years
            # if start_dt.year >= 2021:
            #     new_features_df = self._add_betting_odds_vectorized(new_features_df)
            
            # Append to master
            if not master_df.empty:
                master_df = pd.concat([master_df, new_features_df], ignore_index=True)
            else:
                master_df = new_features_df
            
            # Save
            master_df = self.save_master_table(master_df)
            self._save_feature_states_optimized()
        
        # Return requested range
        return master_df[
            (pd.to_datetime(master_df['game_date']) >= pd.to_datetime(start_date)) & 
            (pd.to_datetime(master_df['game_date']) <= end_dt)
        ]
    
    def _process_date_chunk_safe(self, date_chunk: List[pd.Timestamp],
                                all_baseball: pd.DataFrame,
                                all_statcast: pd.DataFrame) -> List[pd.DataFrame]:
        """Process a chunk of dates maintaining temporal order"""
        
        chunk_features = []
        
        for current_date in date_chunk:
            date_str = current_date.strftime('%Y-%m-%d')
            logger.info(f"  Processing {date_str}...")
            
            # Get data for this date
            date_baseball = all_baseball[all_baseball['game_date_parsed'] == current_date]
            date_statcast = all_statcast[
                pd.to_datetime(all_statcast['game_date']) == current_date
            ] if not all_statcast.empty else pd.DataFrame()
            
            # Build features using states BEFORE this date
            day_features = self._build_features_for_single_day_optimized(
                date_str, date_baseball, date_statcast
            )
            
            if day_features is not None and not day_features.empty:
                chunk_features.append(day_features)
                
                # Update states AFTER building features
                self._update_feature_states_optimized(date_str, date_baseball, date_statcast)
        
        return chunk_features
    
    def _build_features_for_single_day_optimized(self, target_date: str,
                                                target_baseball: pd.DataFrame,
                                                target_statcast: pd.DataFrame) -> pd.DataFrame:
        """Build features for a single day using vectorized operations"""
        
        if target_baseball.empty:
            return None
        
        target_dt = pd.to_datetime(target_date)
        logger.info(f"  Building features for {len(target_baseball)} records on {target_date}")
        
        # Step 1: Create team features using vectorized operations
        team_features = self._create_team_features_vectorized(target_baseball, target_dt)
        
        # Step 2: Create game-level dataframe (vectorized)
        games_df = self._create_game_level_features_vectorized(target_baseball, team_features)
        
        if games_df.empty:
            return None
        
        # Step 3: Add pitcher features (vectorized)
        if not target_statcast.empty:
            games_df = self._add_pitcher_features_vectorized(games_df, target_statcast, target_dt)
            
            # Step 4: Add bullpen features (vectorized)
            games_df = self._add_bullpen_features_vectorized(games_df, target_dt)
        
        # Step 5: Add splits features (vectorized)
        games_df = self._add_splits_features_vectorized(games_df, target_dt)
        
        # Step 6: Add H2H features (vectorized)
        # ✅ Step 7: Add streak features (no leakage)
        games_df = self._add_streak_features_vectorized(games_df)
        # Step 8: Calculate matchup differentials (fully vectorized)
        games_df = self._calculate_matchup_differentials_vectorized(games_df)
        
        return games_df
    
    def _create_team_features_vectorized(self, target_baseball: pd.DataFrame, 
                                    target_date: pd.Timestamp) -> pd.DataFrame:
        """Create team features using pure vectorization"""
        
        # Extract base features
        features_df = target_baseball[[
            'team_id', 'gamePk', 'game_date', 'side', 'venue', 'team'
        ]].copy()
        features_df.rename(columns={'gamePk': 'game_pk'}, inplace=True)
        
        # Add environmental features
        env_cols = ['dayNight', 'temperature', 'wind_speed', 'wind_dir', 'conditions', 'game_time']
        for col in env_cols:
            if col in target_baseball.columns:
                features_df[col] = target_baseball[col]
            else:
                # Set defaults
                if col == 'dayNight':
                    features_df[col] = 'day'
                elif col in ['temperature', 'wind_speed']:
                    features_df[col] = np.nan
                else:
                    features_df[col] = ''
        
        # Get all team IDs
        team_ids = features_df['team_id'].unique()
        
        # Build state features dataframe for all teams at once
        state_features_list = []
        
        for team_id in team_ids:
            team_features = {'team_id': team_id}
            
            # Get rolling features
            if team_id in self.feature_states.get('team_rolling', {}):
                if isinstance(self.feature_states['team_rolling'][team_id], TeamStateArrays):
                    # NEW METHOD CALL
                    rolling_stats = self.feature_states['team_rolling'][team_id].get_rolling_stats_before(
                        target_date,
                        self.config.rolling_games_team
                    )
                    team_features.update(rolling_stats)
                else:
                    # Legacy format support
                    legacy_features = self.feature_states['team_rolling'][team_id].get('features', {})
                    team_features.update(legacy_features)
            
            # DO NOT GET STREAK FEATURES FROM STATE['FEATURES']
            # DO NOT GET SPLITS FEATURES FROM STATE['FEATURES']
            # These will be calculated on-demand in later steps
            
            state_features_list.append(team_features)
        
        # Create state features dataframe
        if state_features_list:
            states_df = pd.DataFrame(state_features_list)
            # Merge all at once
            features_df = features_df.merge(states_df, on='team_id', how='left')
        
        return features_df
    
    def _create_game_level_features_vectorized(self, target_baseball: pd.DataFrame,
                                             team_features: pd.DataFrame) -> pd.DataFrame:
        """Create game-level features using vectorized operations"""
        
        # Get valid games (both teams present)
        game_counts = target_baseball.groupby('gamePk').size()
        valid_games = game_counts[game_counts == 2].index
        
        if len(valid_games) == 0:
            return pd.DataFrame()
        
        # Filter data
        valid_baseball = target_baseball[target_baseball['gamePk'].isin(valid_games)]
        valid_features = team_features[team_features['game_pk'].isin(valid_games)]
        
        # Separate home/away efficiently
        home_mask = valid_baseball['side'] == 'home'
        home_data = valid_baseball[home_mask].set_index('gamePk')
        away_data = valid_baseball[~home_mask].set_index('gamePk')
        
        home_features = valid_features[valid_features['side'] == 'home'].set_index('game_pk')
        away_features = valid_features[valid_features['side'] == 'away'].set_index('game_pk')
        
        # Create base game dataframe
        games_df = pd.DataFrame(index=home_data.index)
        games_df['game_pk'] = games_df.index
        games_df['game_date'] = home_data['game_date']
        
        # Add venue and environmental features
        games_df['venue'] = home_data['venue'].fillna(home_data['team'] + ' Stadium')
        games_df['dayNight'] = home_data['dayNight'].fillna('day')
        games_df['temperature'] = home_data['temperature']
        games_df['wind_speed'] = home_data['wind_speed']
        games_df['wind_dir'] = home_data['wind_dir'].fillna('')
        games_df['conditions'] = home_data['conditions'].fillna('')
        games_df['game_time'] = home_data['game_time'].fillna('')
        
        # Team identifiers
        games_df['home_team'] = home_data['team']
        games_df['away_team'] = away_data['team']
        games_df['home_team_id'] = home_data['team_id']
        games_df['away_team_id'] = away_data['team_id']
        # ✅ ADD THESE TWO LINES
        games_df['home_score'] = home_data['batting_runs']
        games_df['away_score'] = away_data['batting_runs']        
        # Add abbreviations
        team_id_to_abbrev = {v: k for k, v in self.abbrev_to_id.items()}
        games_df['home_team_abbr'] = games_df['home_team_id'].map(team_id_to_abbrev).fillna(games_df['home_team'])
        games_df['away_team_abbr'] = games_df['away_team_id'].map(team_id_to_abbrev).fillna(games_df['away_team'])
        
        # Get feature columns to add
        exclude_cols = {'team_id', 'game_pk', 'side', 'venue', 'team',
                       'dayNight', 'temperature', 'wind_speed', 'wind_dir', 'conditions', 'game_time'}
        
        # Add home features
        for col in home_features.columns:
            if col not in exclude_cols:
                games_df[f'home_{col}'] = home_features[col]
        
        # Add away features
        for col in away_features.columns:
            if col not in exclude_cols:
                games_df[f'away_{col}'] = away_features[col]
        
        return games_df.reset_index(drop=True)
    
    # 5. Complete fixed _add_splits_features_vectorized
    def _add_splits_features_vectorized(self, games_df: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
        """Add splits features using vectorized operations with temporal filtering"""

        # Get all unique team IDs
        all_team_ids = pd.concat([
            games_df['home_team_id'],
            games_df['away_team_id']
        ]).unique()

        # Build splits dataframe
        splits_data = []
        for team_id in all_team_ids:
            features = self._get_splits_features_before_date(team_id, target_date)
            if features:
                features['team_id'] = team_id
                splits_data.append(features)

        if splits_data:
            splits_df = pd.DataFrame(splits_data)

            # --- HOME ---
            home_splits = splits_df.copy()
            
            # Rename all columns except team_id
            home_cols_to_rename = {col: f'home_{col}' for col in home_splits.columns if col != 'team_id'}
            home_splits = home_splits.rename(columns=home_cols_to_rename)
            
            # Merge home splits
            games_df = games_df.merge(
                home_splits.rename(columns={'team_id': 'home_team_id'}), 
                on='home_team_id', 
                how='left',
                suffixes=('', '_home_drop')  # Prevent _x/_y
            )

            # --- AWAY ---
            away_splits = splits_df.copy()
            
            # Rename all columns except team_id
            away_cols_to_rename = {col: f'away_{col}' for col in away_splits.columns if col != 'team_id'}
            away_splits = away_splits.rename(columns=away_cols_to_rename)
            
            # Merge away splits
            games_df = games_df.merge(
                away_splits.rename(columns={'team_id': 'away_team_id'}), 
                on='away_team_id', 
                how='left',
                suffixes=('', '_away_drop')  # Prevent _x/_y
            )
            
            # Drop any columns ending with _drop
            drop_cols = [col for col in games_df.columns if col.endswith('_drop')]
            if drop_cols:
                games_df = games_df.drop(columns=drop_cols)

        return games_df

    # 2. Complete fixed _add_streak_features_vectorized
    def _add_streak_features_vectorized(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Add team streak features to each game, using history only BEFORE each game date"""

        def get_streak_features(team_id, game_date):
            if team_id not in self.feature_states['team_streaks']:
                return {}

            state = self.feature_states['team_streaks'][team_id]
            results = np.array(state['results'])
            dates = np.array(state['dates'], dtype='datetime64[D]')

            mask = dates < np.datetime64(game_date)
            past_results = results[mask]

            if len(past_results) == 0:
                return {}

            # Compute streak from end
            current_streak = 1 if past_results[-1] == 1 else -1
            for i in range(len(past_results) - 2, -1, -1):
                if past_results[i] == past_results[-1]:
                    current_streak = current_streak + 1 if past_results[-1] == 1 else current_streak - 1
                else:
                    break

            return {
                'current_streak': current_streak,
                'win_streak': current_streak if current_streak > 0 else 0,
                'loss_streak': abs(current_streak) if current_streak < 0 else 0,
                'last_5_wins': int(np.sum(past_results[-5:])) if len(past_results) >= 5 else int(np.sum(past_results)),
                'last_10_wins': int(np.sum(past_results[-10:])) if len(past_results) >= 10 else int(np.sum(past_results)),
                'last_20_wins': int(np.sum(past_results[-20:])) if len(past_results) >= 20 else int(np.sum(past_results)),
                'momentum_5_game': float(np.mean(past_results[-5:])) if len(past_results) >= 5 else float(np.mean(past_results)) if len(past_results) > 0 else 0
            }

        # Apply to each row
        streak_rows = []
        for idx, row in games_df.iterrows():
            row_dict = row.to_dict()
            
            # Get home team streaks
            home_feats = get_streak_features(row['home_team_id'], row['game_date'])
            for k, v in home_feats.items():
                row_dict[f'home_{k}'] = v
                
            # Get away team streaks
            away_feats = get_streak_features(row['away_team_id'], row['game_date'])
            for k, v in away_feats.items():
                row_dict[f'away_{k}'] = v

            streak_rows.append(row_dict)

        return pd.DataFrame(streak_rows)

    # def _add_h2h_features_vectorized(self, games_df: pd.DataFrame,
    #                             target_date: pd.Timestamp) -> pd.DataFrame:
    #     """Add H2H features using vectorized operations with temporal filtering"""
        
    #     # Create matchup keys
    #     games_df['matchup_key'] = games_df.apply(
    #         lambda x: '_'.join(sorted([str(x['home_team_id']), str(x['away_team_id'])])),
    #         axis=1
    #     )
        
    #     # Build H2H features
    #     for idx, game in games_df.iterrows():
    #         matchup_key = game['matchup_key']
    #         home_id = game['home_team_id']
    #         away_id = game['away_team_id']
            
    #         # Get H2H features BEFORE target date
    #         home_h2h = self._get_h2h_features_before_date(matchup_key, home_id, target_date)
    #         away_h2h = self._get_h2h_features_before_date(matchup_key, away_id, target_date)
            
    #         # Add to dataframe
    #         for feat, val in home_h2h.items():
    #             games_df.at[idx, f'home_{feat}'] = val
            
    #         for feat, val in away_h2h.items():
    #             games_df.at[idx, f'away_{feat}'] = val
        
    #     # Drop temporary column
    #     games_df.drop('matchup_key', axis=1, inplace=True)
        
    #     return games_df
    
    def _calculate_matchup_differentials_vectorized(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all differentials using vectorized operations"""
        
        # Find all matching home/away pairs
        home_cols = [c for c in games_df.columns if c.startswith('home_') and 
                    c not in ['home_team', 'home_team_id', 'home_team_abbr']]
        
        for home_col in home_cols:
            feature_name = home_col[5:]  # Remove 'home_' prefix
            away_col = f'away_{feature_name}'
            
            if away_col in games_df.columns:
                # Only calculate for numeric columns
                if pd.api.types.is_numeric_dtype(games_df[home_col]):
                    games_df[f'diff_{feature_name}'] = games_df[home_col] - games_df[away_col]
        
        # Additional composite features
        if 'home_last_10_wins' in games_df and 'away_last_10_wins' in games_df:
            games_df['matchup_strength_differential'] = (
                games_df['home_last_10_wins'] - games_df['away_last_10_wins']
            ) / 10
        
        if ('home_batting_runs_roll10' in games_df and 
            'away_pitching_earnedRuns_roll10' in games_df):
            games_df['expected_home_runs'] = (
                games_df['home_batting_runs_roll10'] + 
                games_df['away_pitching_earnedRuns_roll10']
            ) / 2
        
        if ('away_batting_runs_roll10' in games_df and 
            'home_pitching_earnedRuns_roll10' in games_df):
            games_df['expected_away_runs'] = (
                games_df['away_batting_runs_roll10'] + 
                games_df['home_pitching_earnedRuns_roll10']
            ) / 2
        
        return games_df
    
    def _update_feature_states_optimized(self, date: str, 
                                       baseball_data: pd.DataFrame,
                                       statcast_data: pd.DataFrame):
        """Update all feature states using optimized methods"""
        
        date_ts = pd.to_datetime(date)
        
        if not baseball_data.empty:
            # Update team states with numpy arrays
            self._update_team_rolling_states_numpy(baseball_data, date_ts)
            self._update_team_streak_states_numpy(baseball_data, date_ts)
            self._update_team_splits_states_optimized(baseball_data, date_ts)
            # self._update_h2h_states_optimized(baseball_data, date_ts)
        
        if not statcast_data.empty:
            self._update_pitcher_states_optimized(statcast_data, date_ts)
            self._update_bullpen_states_optimized(statcast_data, date_ts)
    
    def _update_team_rolling_states_numpy(self, new_data: pd.DataFrame, date_ts: pd.Timestamp):
        """Update team rolling states using numpy arrays"""
        
        for team_id in new_data['team_id'].unique():
            team_data = new_data[new_data['team_id'] == team_id]
            
            # NEW, SIMPLER INITIALIZATION
            if team_id not in self.feature_states['team_rolling']:
                self.feature_states['team_rolling'][team_id] = TeamStateArrays(
                    feature_names=self.team_features
                )
            
            state = self.feature_states['team_rolling'][team_id]
            
            # Extract features as numpy array
            for _, game in team_data.iterrows():
                features = np.zeros(len(self.team_features))
                
                for i, col in enumerate(self.team_features):
                    if col in game:
                        val = game[col]
                        # Clean value
                        if isinstance(val, str) and val in ['.---', '---', '-', 'N/A', 'NA', '', ' ']:
                            val = np.nan
                        else:
                            try:
                                val = float(val)
                            except:
                                val = np.nan
                        features[i] = val
                    else:
                        features[i] = np.nan
                
                # Add to state
                state.add_game(features, date_ts)
    
    def _update_team_streak_states_numpy(self, new_data: pd.DataFrame, date_ts: pd.Timestamp):
        """Update team streak states using numpy for efficiency, avoiding data leakage"""

        for team_id in new_data['team_id'].unique():
            team_data = new_data[new_data['team_id'] == team_id]

            # Initialize state if needed
            if team_id not in self.feature_states['team_streaks']:
                self.feature_states['team_streaks'][team_id] = {
                    'results': np.array([], dtype=np.int8),
                    'dates': np.array([], dtype='datetime64[D]'),
                    'features': {}  # Keep this but don't populate it here
                }

            state = self.feature_states['team_streaks'][team_id]

            # Prepare results and dates for appending
            new_results = np.array([
                1 if row['batting_runs'] > row['pitching_runs'] else 0
                for _, row in team_data.iterrows()
            ], dtype=np.int8)
            new_dates = np.array([np.datetime64(date_ts, 'D')] * len(new_results), dtype='datetime64[D]')

            # Append using np.append
            state['results'] = np.append(state['results'], new_results)
            state['dates'] = np.append(state['dates'], new_dates)
    
    def _add_pitcher_features_vectorized(self, games_df: pd.DataFrame,
                                    target_statcast: pd.DataFrame,
                                    target_date: pd.Timestamp) -> pd.DataFrame:
        """Add pitcher features using vectorized operations with temporal filtering"""
        
        if target_statcast.empty:
            return games_df
        
        # Identify starting pitchers efficiently
        first_pitches = target_statcast.sort_values(
            ['game_pk', 'at_bat_number', 'pitch_number']
        ).groupby(['game_pk', 'inning_topbot']).first().reset_index()
        
        # Create pitcher feature dataframe
        pitcher_features = []
        
        for _, starter in first_pitches.iterrows():
            pitcher_id = starter['pitcher']
            game_pk = starter['game_pk']
            is_home = starter['inning_topbot'] == 'Top'
            
            # Get features BEFORE target date
            features = self._get_pitcher_features_before_date(pitcher_id, target_date)
            
            if features:
                pitcher_feat = {
                    'game_pk': game_pk,
                    'is_home': is_home
                }
                pitcher_feat.update(features)
                pitcher_features.append(pitcher_feat)
        
        if pitcher_features:
            pitcher_df = pd.DataFrame(pitcher_features)
            
            # Merge home pitchers
            home_pitcher_df = pitcher_df[pitcher_df['is_home']].drop('is_home', axis=1)
            home_pitcher_df.columns = ['game_pk'] + [f'home_{c}' for c in home_pitcher_df.columns[1:]]
            games_df = games_df.merge(home_pitcher_df, on='game_pk', how='left')
            
            # Merge away pitchers
            away_pitcher_df = pitcher_df[~pitcher_df['is_home']].drop('is_home', axis=1)
            away_pitcher_df.columns = ['game_pk'] + [f'away_{c}' for c in away_pitcher_df.columns[1:]]
            games_df = games_df.merge(away_pitcher_df, on='game_pk', how='left')
        
        return games_df

    def _add_bullpen_features_vectorized(self, games_df: pd.DataFrame,
                                    target_date: pd.Timestamp) -> pd.DataFrame:
        """Add bullpen features using vectorized operations with temporal filtering"""
        
        # Get all teams
        all_teams = pd.concat([
            games_df[['home_team', 'home_team_abbr', 'home_team_id']].rename(
                columns={'home_team': 'team', 'home_team_abbr': 'abbr', 'home_team_id': 'team_id'}
            ),
            games_df[['away_team', 'away_team_abbr', 'away_team_id']].rename(
                columns={'away_team': 'team', 'away_team_abbr': 'abbr', 'away_team_id': 'team_id'}
            )
        ]).drop_duplicates()
        
        # Build bullpen features dataframe
        bullpen_features = []
        
        for _, team_info in all_teams.iterrows():
            # Try different identifiers
            for identifier in [team_info['abbr'], team_info['team'], str(team_info['team_id'])]:
                features = self._get_bullpen_features_before_date(identifier, target_date)
                if features:
                    features['team_id'] = team_info['team_id']
                    bullpen_features.append(features)
                    break
        
        if bullpen_features:
            bullpen_df = pd.DataFrame(bullpen_features)
            
            # Merge for home teams
            home_bullpen = bullpen_df.add_prefix('home_').rename(columns={'home_team_id': 'home_team_id'})
            games_df = games_df.merge(home_bullpen, on='home_team_id', how='left')
            
            # Merge for away teams
            away_bullpen = bullpen_df.add_prefix('away_').rename(columns={'away_team_id': 'away_team_id'})
            games_df = games_df.merge(away_bullpen, on='away_team_id', how='left')
        
        return games_df
    
    # def _add_betting_odds_vectorized(self, games_df: pd.DataFrame) -> pd.DataFrame:
    #     """Add betting odds for games after 2021"""
        
    #     # Filter to games that need odds
    #     games_2021_plus = games_df[pd.to_datetime(games_df['game_date']).dt.year >= 2021]
        
    #     if games_2021_plus.empty:
    #         return games_df
        
    #     try:
    #         pipeline = MLPipeline(self.config)
    #         pipeline.full_name_to_id = self.full_name_to_id
    #         pipeline.abbrev_to_id = self.abbrev_to_id
            
    #         # Process in batches for efficiency
    #         batch_size = 100
    #         odds_dfs = []
            
    #         for i in range(0, len(games_2021_plus), batch_size):
    #             batch = games_2021_plus.iloc[i:i+batch_size]
    #             batch_with_odds = pipeline.add_betting_odds(batch)
    #             odds_dfs.append(batch_with_odds)
            
    #         # Combine
    #         all_odds = pd.concat(odds_dfs, ignore_index=True)
            
    #         # Merge back
    #         odds_cols = [c for c in all_odds.columns if c not in games_df.columns]
    #         if odds_cols:
    #             games_df = games_df.merge(
    #                 all_odds[['game_pk'] + odds_cols],
    #                 on='game_pk',
    #                 how='left'
    #             )
            
    #         logger.info(f"Successfully added betting odds for {len(games_2021_plus)} games")
            
    #     except Exception as e:
    #         logger.error(f"Error adding betting odds: {str(e)}")
        
    #     return games_df
    
    def save_master_table(self, master_df: pd.DataFrame) -> pd.DataFrame:
        """Save master table with proper data types (keeping your original logic)"""
        
        # Identify string columns that should not be converted to numeric
        string_columns = [
            'venue', 'dayNight', 'wind_dir', 'conditions', 'game_time',
            'home_team', 'away_team', 'home_team_abbr', 'away_team_abbr',
            'team_psychological_state', 'bookmaker'
        ]
        
        # Identify date columns
        date_columns = ['game_date']
        
        # Optimize dtypes
        for col in master_df.columns:
            if col in string_columns:
                # Convert to category for efficiency
                master_df[col] = master_df[col].astype('category')
            elif col in date_columns:
                master_df[col] = pd.to_datetime(master_df[col])
            elif master_df[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
                    # Use float32 for most numeric columns
                    if master_df[col].dtype == 'float64':
                        master_df[col] = master_df[col].astype('float32')
                except Exception:
                    continue
        
        # Save with compression
        master_df.to_parquet(self.master_table_path, index=False, compression='snappy')
        logger.info(f"Updated master table saved to {self.master_table_path}")
        
        return master_df
    
    def _save_feature_states_optimized(self):
        """Save feature states using joblib for better performance"""
        
        state_file = os.path.join(self.state_cache_dir, 'feature_states.joblib')
        
        # Convert numpy arrays to serializable format
        save_dict = {}
        
        for state_type, states in self.feature_states.items():
            save_dict[state_type] = {}
            
            for key, state in states.items():
                # NEW, CORRECTED SAVE LOGIC
                if isinstance(state, TeamStateArrays):
                    # This correctly saves the new list attributes
                    save_dict[state_type][key] = {
                        'feature_names': state.feature_names,
                        'games': state.games,
                        'timestamps': state.timestamps
                    }
                else:
                    # Keep as is for other formats
                    save_dict[state_type][key] = state
        
        joblib.dump(save_dict, state_file, compress=3)
        logger.info("Saved optimized feature states")
    
    def _load_feature_states_optimized(self):
        """Load feature states using joblib"""
        
        state_file = os.path.join(self.state_cache_dir, 'feature_states.joblib')
        
        if os.path.exists(state_file):
            try:
                save_dict = joblib.load(state_file)
                
                # Reconstruct states
                for state_type, states in save_dict.items():
                    self.feature_states[state_type] = {}
                    
                    for key, state_data in states.items():
                        # NEW, CORRECTED LOAD LOGIC
                        if state_type == 'team_rolling' and 'games' in state_data:
                            # This correctly reconstructs the object from the new list attributes
                            self.feature_states[state_type][key] = TeamStateArrays(
                                feature_names=state_data['feature_names'],
                                games=state_data['games'],
                                timestamps=state_data['timestamps']
                            )
                        else:
                            self.feature_states[state_type][key] = state_data
                
                logger.info("Loaded optimized feature states")
            except Exception as e:
                logger.warning(f"Could not load joblib states, trying pickle: {e}")
                self._load_feature_states_legacy()
        else:
            # Try loading legacy pickle format
            self._load_feature_states_legacy()
    
    def _load_feature_states_legacy(self):
        """Load feature states from legacy pickle format"""
        
        state_file = os.path.join(self.state_cache_dir, 'feature_states.pkl')
        if os.path.exists(state_file):
            with open(state_file, 'rb') as f:
                self.feature_states = pickle.load(f)
            logger.info("Loaded feature states from legacy pickle format")
    
    def _is_baseball_season(self, date: pd.Timestamp) -> bool:
        """Check if date falls within baseball season"""
        return date.month >= 3 and date.month <= 10
    
    # Include all the other update methods from original with optimizations...
    def _initialize_warm_up_period_optimized(self, start_date: pd.Timestamp):
        """Initialize with warm-up data using optimized loading"""
        
        warm_up_start = max(
            start_date - pd.Timedelta(days=60),
            pd.Timestamp(year=start_date.year - 1, month=3, day=1)
        )
        
        logger.info(f"Loading warm-up data from {warm_up_start} to {start_date - pd.Timedelta(days=1)}")
        
        # Load warm-up data
        warm_up_baseball = self.db.load_baseball_scrape_data(
            warm_up_start.strftime('%Y-%m-%d'),
            (start_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        )
        warm_up_statcast = self.db.load_statcast_data(
            warm_up_start.strftime('%Y-%m-%d'),
            (start_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        )
        
        if not warm_up_baseball.empty:
            # Process chronologically
            warm_up_baseball['game_date_parsed'] = pd.to_datetime(warm_up_baseball['game_date'])
            warm_up_baseball = warm_up_baseball.sort_values(['game_date_parsed', 'gamePk'])
            
            # Process in chunks
            for date in warm_up_baseball['game_date_parsed'].unique():
                date_data = warm_up_baseball[warm_up_baseball['game_date_parsed'] == date]
                date_statcast = warm_up_statcast[
                    pd.to_datetime(warm_up_statcast['game_date']) == date
                ] if not warm_up_statcast.empty else pd.DataFrame()
                
                self._update_feature_states_optimized(
                    date.strftime('%Y-%m-%d'),
                    date_data,
                    date_statcast
                )
    
    # Add remaining update methods with same optimization patterns...
    # 3. Complete fixed _update_team_splits_states_optimized
    def _update_team_splits_states_optimized(self, new_data: pd.DataFrame, date_ts: pd.Timestamp):
        """Update team splits states - keeping your cleaning logic"""
        
        # Your original cleaning logic
        numeric_cols = ['batting_runs', 'batting_ops', 'pitching_era']
        for col in numeric_cols:
            if col in new_data.columns:
                def clean_numeric_value(val):
                    if pd.isna(val):
                        return np.nan
                    if isinstance(val, (int, float)):
                        return val
                    if isinstance(val, str):
                        val = val.strip()
                        if val.count('.') > 1:
                            parts = val.split('.')
                            if len(parts) >= 2:
                                try:
                                    return float(f"{parts[0]}.{parts[1]}")
                                except:
                                    return np.nan
                        try:
                            return float(val)
                        except:
                            return np.nan
                    return np.nan
                
                new_data[col] = new_data[col].apply(clean_numeric_value)
        
        # Continue with optimized processing...
        for team_id in new_data['team_id'].unique():
            team_data = new_data[new_data['team_id'] == team_id]
            
            if team_id not in self.feature_states['team_splits']:
                self.feature_states['team_splits'][team_id] = {
                    'games': [],
                    'features': {}  # Keep empty - calculate on demand
                }
            
            state = self.feature_states['team_splits'][team_id]
            
            # Add new games
            for _, game in team_data.iterrows():
                game_dict = {
                    'date': game['game_date'],
                    'dayNight': game.get('dayNight', 'day'),
                    'side': game['side'],
                    'batting_runs': float(game.get('batting_runs', 0)) if pd.notna(game.get('batting_runs', 0)) else 0.0,
                    'batting_ops': float(game.get('batting_ops', 0)) if pd.notna(game.get('batting_ops', 0)) else 0.0,
                    'pitching_era': float(game.get('pitching_era', 0)) if pd.notna(game.get('pitching_era', 0)) else 0.0
                }
                state['games'].append(game_dict)
            
            # Keep only last 180 days
            cutoff_date = date_ts - pd.Timedelta(days=180)
            state['games'] = [g for g in state['games'] if pd.to_datetime(g['date']) > cutoff_date]
    
    # def _update_h2h_states_optimized(self, new_data: pd.DataFrame, date_ts: pd.Timestamp):
    #     """Update H2H states - efficient and safe update with leakage prevention"""

    #     # Group games by gamePk
    #     games_by_pk = {}
    #     for _, game in new_data.iterrows():
    #         pk = game['gamePk']
    #         games_by_pk.setdefault(pk, []).append(game)

    #     for game_pk, teams in games_by_pk.items():
    #         if len(teams) != 2:
    #             continue  # skip incomplete matchups

    #         team1, team2 = teams[0], teams[1]
    #         matchup_key = '_'.join(sorted([str(team1['team_id']), str(team2['team_id'])]))

    #         # Initialize matchup state if not exists
    #         if matchup_key not in self.feature_states['h2h_streaks']:
    #             self.feature_states['h2h_streaks'][matchup_key] = {}

    #         for team in teams:
    #             team_id = team['team_id']

    #             # Initialize team-specific state within this matchup
    #             if team_id not in self.feature_states['h2h_streaks'][matchup_key]:
    #                 self.feature_states['h2h_streaks'][matchup_key][team_id] = {
    #                     'results': np.array([], dtype=np.int8),
    #                     'dates': np.array([], dtype='datetime64[D]'),
    #                     'features': {}  # Keep empty - don't populate
    #                 }

    #             state = self.feature_states['h2h_streaks'][matchup_key][team_id]

    #             # Add result and date
    #             result = 1 if team['batting_runs'] > team['pitching_runs'] else 0
    #             state['results'] = np.append(state['results'], result)
    #             state['dates'] = np.append(state['dates'], np.datetime64(date_ts, 'D'))
                
                # DO NOT CALCULATE FEATURES HERE
                # Features will be calculated on-demand in _get_h2h_features_before_date
                #   # No valid history yet
    
    def _update_pitcher_states_optimized(self, new_statcast: pd.DataFrame, date_ts: pd.Timestamp):
        """Update pitcher states - keeping your original logic with optimization"""
        
        # Identify starting pitchers
        starters = new_statcast.sort_values(
            ['game_pk', 'at_bat_number', 'pitch_number']
        ).groupby(['game_pk', 'inning_topbot']).first().reset_index()
        
        starter_ids = set(zip(starters['game_pk'], starters['pitcher']))
        
        # Filter to only starter pitches
        starter_mask = new_statcast.apply(
            lambda x: (x['game_pk'], x['pitcher']) in starter_ids, axis=1
        )
        starter_pitches = new_statcast[starter_mask]
        
        if starter_pitches.empty:
            return
        
        # Calculate pitcher game stats
        pitcher_stats = self.feature_engineer.calculate_pitcher_game_stats(starter_pitches)
        
        for pitcher_id in pitcher_stats['pitcher'].unique():
            pitcher_games = pitcher_stats[pitcher_stats['pitcher'] == pitcher_id]
            
            if pitcher_id not in self.feature_states['pitcher_rolling']:
                self.feature_states['pitcher_rolling'][pitcher_id] = {
                    'games': [],
                    'features': {}
                }
            
            state = self.feature_states['pitcher_rolling'][pitcher_id]
            
            # Add new games WITH DATES
            for _, game in pitcher_games.iterrows():
                game_data = {}
                for col in game.index:
                    if col not in ['pitcher', 'game_pk', 'game_date']:
                        if pd.notna(game[col]) and isinstance(game[col], (int, float)):
                            game_data[col] = game[col]
                
                state['games'].append({
                    'date': game['game_date'],  # Make sure date is included!
                    'data': game_data
                })
    
    def _update_bullpen_states_optimized(self, new_statcast: pd.DataFrame, date_ts: pd.Timestamp):
        """Update bullpen states - keeping your original logic"""
        
        logger.info(f"Updating bullpen states for {new_statcast['game_date'].nunique()} dates")
        
        # Identify relievers
        starters = new_statcast.sort_values(
            ['game_pk', 'at_bat_number', 'pitch_number']
        ).groupby(['game_pk', 'inning_topbot']).first().reset_index()
        
        starter_ids = set(zip(starters['game_pk'], starters['pitcher']))
        
        # Filter to only reliever pitches
        reliever_mask = new_statcast.apply(
            lambda x: (x['game_pk'], x['pitcher']) not in starter_ids, axis=1
        )
        bullpen_df = new_statcast[reliever_mask]
        
        logger.info(f"Found {len(bullpen_df)} reliever pitches out of {len(new_statcast)} total")
        
        if bullpen_df.empty:
            logger.warning("No reliever pitches found")
            return
        
        # Map pitchers to teams
        pitcher_teams = bullpen_df.groupby(['pitcher', 'game_pk']).agg({
            'home_team': 'first',
            'away_team': 'first',
            'inning_topbot': 'first'
        }).reset_index()
        
        pitcher_teams['team'] = pitcher_teams.apply(
            lambda x: x['home_team'] if x['inning_topbot'] == 'Top' else x['away_team'],
            axis=1
        )
        
        # Calculate daily bullpen stats
        bullpen_stats = self.feature_engineer.calculate_bullpen_daily_stats(bullpen_df, pitcher_teams)
        
        if bullpen_stats.empty:
            logger.warning("calculate_bullpen_daily_stats returned empty DataFrame")
            return
        
        logger.info(f"Calculated bullpen stats for {len(bullpen_stats)} team-games")
        
        # Process each team's bullpen stats
        for team in bullpen_stats['team'].unique():
            team_stats = bullpen_stats[bullpen_stats['team'] == team]
            
            if team not in self.feature_states['bullpen_rolling']:
                self.feature_states['bullpen_rolling'][team] = {
                    'daily_stats': [],
                    'features': {}
                }
            
            state = self.feature_states['bullpen_rolling'][team]
            
            # Add new daily stats
            for _, day_stats in team_stats.iterrows():
                day_data = {}
                
                for col in day_stats.index:
                    if col not in ['team', 'game_pk', 'game_date']:
                        val = day_stats[col]
                        if pd.notna(val) and isinstance(val, (int, float, np.integer, np.floating)):
                            day_data[col] = float(val)
                
                if day_data and 'game_date' in day_stats.index:
                    state['daily_stats'].append({
                        'date': day_stats['game_date'],
                        'data': day_data
                    })

import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import time
from typing import Optional
import multiprocessing as mp

logger = logging.getLogger(__name__)

def main():
    """Optimized pipeline execution with state management and performance tracking"""
    
    # Initialize configuration
    config = PipelineConfig()
    
    # Performance tracking
    pipeline_start_time = time.time()
    
    # Check if we have an existing master table
    master_table_path = os.path.join(config.output_dir, 'master_features_table.parquet')
    
    # Simple date range specification
    start_date = '2017-01-01'  # Change this to your desired start date
    end_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Check if master table exists
    if os.path.exists(master_table_path):
        existing_df = pd.read_parquet(master_table_path)
        last_date = pd.to_datetime(existing_df['game_date']).max()
        logger.info(f"Found existing master table with data up to {last_date}")
        rebuild = False  # Will append new data
    else:
        logger.info("No existing master table found, starting fresh")
        rebuild = True
    
    # Determine number of workers for parallel processing
    n_workers = min(mp.cpu_count() - 1, 4)  # Leave one CPU free
    logger.info(f"Using {n_workers} CPU cores for parallel processing")
    
    # Create optimized builder with parallel processing
    builder = CompleteOptimizedFeatureBuilder(config, n_workers=n_workers)
    
    # Build features incrementally
    logger.info(f"Building features from {start_date} to {end_date} (rebuild={rebuild})")
    
    try:
        # Track progress
        total_games_processed = 0
        processing_times = []
        
        # Process in chunks for better progress tracking
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Process by month for better memory management and progress tracking
        current_start = start_dt
        all_features = []
        
        while current_start <= end_dt:
            # Process one month at a time
            current_end = min(
                current_start + pd.DateOffset(months=1) - pd.Timedelta(days=1),
                end_dt
            )
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {current_start.strftime('%B %Y')}")
            logger.info(f"{'='*60}")
            
            # Time this chunk
            chunk_start_time = time.time()
            
            # Build features for this chunk
            chunk_features = builder.build_features_incrementally(
                current_start.strftime('%Y-%m-%d'),
                current_end.strftime('%Y-%m-%d'),
                use_cache=True,
                rebuild=rebuild and current_start == start_dt  # Only rebuild on first chunk
            )
            
            chunk_time = time.time() - chunk_start_time
            
            if not chunk_features.empty:
                all_features.append(chunk_features)
                games_in_chunk = len(chunk_features)
                total_games_processed += games_in_chunk
                games_per_second = games_in_chunk / chunk_time if chunk_time > 0 else 0
                
                logger.info(f"✅ Processed {games_in_chunk} games for {current_start.strftime('%B %Y')}")
                logger.info(f"   Time: {chunk_time:.2f}s ({games_per_second:.1f} games/sec)")
                logger.info(f"   Memory usage: {get_memory_usage():.1f} MB")
                
                processing_times.append({
                    'month': current_start.strftime('%B %Y'),
                    'games': games_in_chunk,
                    'time': chunk_time,
                    'games_per_sec': games_per_second
                })
            
            # Move to next month
            current_start = current_end + pd.Timedelta(days=1)
            rebuild = False  # After first chunk, always append
            
            # Optional: Free memory between chunks
            import gc
            gc.collect()
        
        # Combine all features
        if all_features:
            logger.info("\n📊 Combining all features...")
            train_df = pd.concat(all_features, ignore_index=True)
            
            # Ensure proper data types for efficiency
            train_df = optimize_dataframe_dtypes(train_df)
        else:
            raise ValueError("No features generated")
        
        # Calculate total pipeline time
        total_time = time.time() - pipeline_start_time
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 PIPELINE COMPLETE!")
        logger.info(f"{'='*60}")
        logger.info(f"Total games processed: {len(train_df):,}")
        logger.info(f"Date range: {train_df['game_date'].min()} to {train_df['game_date'].max()}")
        logger.info(f"Features per game: {len(train_df.columns)}")
        logger.info(f"Master table saved to: {master_table_path}")
        logger.info(f"Total processing time: {format_time(total_time)}")
        logger.info(f"Average speed: {len(train_df)/total_time:.1f} games/second")
        
        # Performance summary
        if processing_times:
            logger.info("\n📈 Performance Summary:")
            logger.info(f"{'Month':<15} {'Games':<10} {'Time':<10} {'Speed (g/s)':<12}")
            logger.info("-" * 50)
            for pt in processing_times:
                logger.info(f"{pt['month']:<15} {pt['games']:<10} "
                          f"{pt['time']:<10.2f} {pt['games_per_sec']:<12.1f}")
        
        # Save feature names for model training
        feature_cols = [col for col in train_df.columns if col not in [
            'game_pk', 'game_date', 'venue', 'home_team', 'away_team',
            'home_team_id', 'away_team_id', 'home_team_abbr', 'away_team_abbr', 'home_score', 'away_score',
            'features_calculated_date', 'features_calculated_for_date',
            'dayNight', 'temperature', 'wind_speed', 'wind_dir', 'conditions', 
            'game_time', 'team_psychological_state', 'bookmaker'
        ]]
        
        joblib.dump(feature_cols, os.path.join(config.output_dir, 'feature_names.pkl'))
        logger.info(f"\n📝 Saved {len(feature_cols)} feature names")
        
        # Create comprehensive pipeline metadata
        metadata = {
            'creation_date': datetime.now().isoformat(),
            'feature_count': len(feature_cols),
            'total_games': len(train_df),
            'date_range': {
                'start': str(train_df['game_date'].min()),
                'end': str(train_df['game_date'].max())
            },
            'columns': train_df.columns.tolist(),
            'feature_columns': feature_cols,
            'version': '3.0',  # Optimized version with numpy arrays
            'performance': {
                'total_time_seconds': total_time,
                'total_time_formatted': format_time(total_time),
                'games_per_second': len(train_df) / total_time,
                'n_workers': n_workers,
                'monthly_performance': processing_times
            },
            'config': {
                'rolling_games_team': config.rolling_games_team,
                'rolling_games_pitcher': config.rolling_games_pitcher,
                'rolling_days_bullpen': config.rolling_days_bullpen,
                'min_games_threshold': config.min_games_threshold
            },
            'data_quality': {
                'total_features': len(train_df.columns),
                'numeric_features': len(train_df.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(train_df.select_dtypes(include=['category', 'object']).columns),
                'memory_usage_mb': train_df.memory_usage(deep=True).sum() / 1024**2
            }
        }
        
        with open(os.path.join(config.output_dir, 'pipeline_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Data quality check
        logger.info("\n🔍 Data Quality Check:")
        null_counts = train_df.isnull().sum()
        high_null_features = null_counts[null_counts > len(train_df) * 0.5]
        if not high_null_features.empty:
            logger.warning(f"⚠️  Features with >50% missing values: {list(high_null_features.index)}")
        else:
            logger.info("✅ No features with excessive missing values")
        
        # Check for data leakage
        logger.info("\n🔒 Temporal Integrity Check:")
        check_temporal_integrity(train_df)
        
        # Optional: Create validation split with temporal awareness
        if input("\n📊 Create train/validation split? [y/n]: ").lower() == 'y':
            val_start = input("Enter validation start date (YYYY-MM-DD) [2024-01-01]: ") or '2024-01-01'
            
            # Ensure temporal split (no leakage)
            val_mask = pd.to_datetime(train_df['game_date']) >= val_start
            val_df = train_df[val_mask].copy()
            train_df_split = train_df[~val_mask].copy()
            
            # Verify no overlap
            train_max_date = train_df_split['game_date'].max()
            val_min_date = val_df['game_date'].min()
            
            if train_max_date >= val_min_date:
                logger.error(f"⚠️  Temporal leakage detected! Train ends {train_max_date}, Val starts {val_min_date}")
            else:
                logger.info(f"✅ Valid temporal split: {(pd.to_datetime(val_min_date) - pd.to_datetime(train_max_date)).days} days gap")
            
            # Save splits
            train_df_split.to_parquet(os.path.join(config.output_dir, 'train_split.parquet'))
            val_df.to_parquet(os.path.join(config.output_dir, 'val_split.parquet'))
            
            logger.info(f"\n📁 Split Summary:")
            logger.info(f"Train: {len(train_df_split):,} games ({train_df_split['game_date'].min()} to {train_df_split['game_date'].max()})")
            logger.info(f"Val:   {len(val_df):,} games ({val_df['game_date'].min()} to {val_df['game_date'].max()})")
            logger.info(f"Split ratio: {len(train_df_split)/(len(train_df_split)+len(val_df)):.1%} train")
        
        # Feature importance preview (if many features)
        if len(feature_cols) > 100:
            logger.info(f"\n📊 Feature Categories (Total: {len(feature_cols)}):")
            categorize_features(feature_cols)
        
        return train_df
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        logger.error("Full traceback:", exc_info=True)
        
        # Save partial results if possible
        if 'all_features' in locals() and all_features:
            try:
                partial_df = pd.concat(all_features, ignore_index=True)
                partial_path = os.path.join(config.output_dir, 'partial_features.parquet')
                partial_df.to_parquet(partial_path)
                logger.info(f"💾 Saved partial results ({len(partial_df)} games) to {partial_path}")
            except:
                pass
        
        raise


def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage by converting to efficient dtypes"""
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    # Convert float64 to float32 where possible
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        if col not in ['game_pk', 'home_team_id', 'away_team_id']:  # Keep IDs as int
            df[col] = df[col].astype('float32')
    
    # Convert object columns to category where appropriate
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        num_unique = df[col].nunique()
        if num_unique / len(df) < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')
    
    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"💾 Memory optimization: {initial_memory:.1f} MB → {final_memory:.1f} MB "
                f"({(1 - final_memory/initial_memory)*100:.1f}% reduction)")
    
    return df


def get_memory_usage() -> float:
    """Get current process memory usage in MB"""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def check_temporal_integrity(df: pd.DataFrame):
    """Check for temporal ordering issues"""
    df_sorted = df.sort_values('game_date')
    
    # Check if already sorted
    if df['game_date'].equals(df_sorted['game_date']):
        logger.info("✅ Data is chronologically ordered")
    else:
        logger.warning("⚠️  Data is not chronologically ordered - this could indicate issues")
    
    # Check for duplicate games
    duplicates = df.duplicated(subset=['game_pk'])
    if duplicates.any():
        logger.warning(f"⚠️  Found {duplicates.sum()} duplicate games")
    else:
        logger.info("✅ No duplicate games found")


def categorize_features(feature_cols: list):
    """Categorize features by type for better understanding"""
    categories = {
        'batting': [],
        'pitching': [],
        'fielding': [],
        'bullpen': [],
        'streak': [],
        'h2h': [],
        'splits': [],
        'differential': [],
        'environmental': [],
        'other': []
    }
    
    for col in feature_cols:
        if 'batting' in col:
            categories['batting'].append(col)
        elif 'pitching' in col or 'SP_' in col:
            categories['pitching'].append(col)
        elif 'fielding' in col:
            categories['fielding'].append(col)
        elif 'bullpen' in col:
            categories['bullpen'].append(col)
        elif 'streak' in col or 'momentum' in col:
            categories['streak'].append(col)
        elif 'h2h' in col:
            categories['h2h'].append(col)
        elif any(split in col for split in ['day_', 'night_', 'home_', 'away_']):
            categories['splits'].append(col)
        elif 'diff_' in col or 'differential' in col:
            categories['differential'].append(col)
        elif any(env in col for env in ['temperature', 'wind', 'condition']):
            categories['environmental'].append(col)
        else:
            categories['other'].append(col)
    
    for category, features in categories.items():
        if features:
            logger.info(f"  {category.capitalize()}: {len(features)} features")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler()
        ]
    )
    
    # Run pipeline
    try:
        features_df = main()
        logger.info("\n✅ Pipeline completed successfully!")
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        raise