"""
MLB Daily Production Pipeline - TRUE ON-THE-FLY CALCULATION
Processes games by calculating all features from scratch, matching sqlMLBqueryv5.py exactly
No reliance on feature_states.joblib - pure calculation for backtesting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import joblib
import json
from typing import Dict, List, Tuple, Optional
import pytz
import warnings
import time
import requests
import re
from sqlalchemy import create_engine
import urllib.parse
from dataclasses import dataclass, field
from sqlMLBqueryv5 import DatabaseConnection, CompleteOptimizedFeatureBuilder, FeatureEngineer, PipelineConfig

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TeamStateArrays:
    """
    MODIFIED: Stores the FULL history for a team's rolling features using dynamic lists.
    This prevents data from being overwritten or truncated.
    """
    feature_names: List[str]
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

        all_timestamps = np.array(self.timestamps, dtype=np.int64)
        all_data = np.array(self.games)

        target_ts = pd.Timestamp(target_date).value
        before_mask = all_timestamps < target_ts
        
        if not np.any(before_mask):
            return {}

        valid_historical_data = all_data[before_mask]
        recent_data = valid_historical_data[-rolling_window_size:]
        
        if recent_data.shape[0] == 0:
            return {}

        stats = {}
        for i, name in enumerate(self.feature_names):
            col_data = recent_data[:, i]
            non_nan_mask = ~np.isnan(col_data)
            if np.any(non_nan_mask):
                stats[f"{name}_roll{rolling_window_size}"] = np.mean(col_data[non_nan_mask])
        
        return stats
    
class MLBDailyPipeline:
    """Production pipeline that uses cached feature states for exact match with training"""
    
    def __init__(self, config_path: str = None):
        """Initialize pipeline using cached states"""
        
        # Load configuration
        if config_path:
            with open(config_path, 'rb') as f:
                import pickle
                self.config = pickle.load(f)
        else:
            self.config = PipelineConfig()
        
        # Set timezone
        self.eastern_tz = pytz.timezone('US/Eastern')
        self.pacific_tz = pytz.timezone('US/Pacific')
        
        # Initialize database connection
        self.db_connection = DatabaseConnection(self.config)
        if not self.db_connection.connect():
            raise ConnectionError("Failed to connect to database")
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(self.config)
        
        # Load mappings
        self.full_name_to_id, self.abbrev_to_id = self.db_connection.load_team_mappings()
        self.id_to_full_name = {v: k for k, v in self.full_name_to_id.items()}
        self.id_to_abbrev = {v: k for k, v in self.abbrev_to_id.items()}
        
        # Create engine for additional queries
        self._create_db_connection()
        
        # Load feature names for ordering
        if os.path.exists(self.config.feature_names_path):
            self.feature_names = joblib.load(self.config.feature_names_path)
        else:
            self.feature_names = []
        
        # Load team locations for weather
        self._load_team_locations()
        
        # CRITICAL: Load feature states from training
        self.feature_states = self._load_feature_states()
    
    # In testingInference.py -> MLBDailyPipeline class

    def _load_feature_states(self):
        """Load the exact feature states and reconstructs stateful objects."""
        state_file_path = os.path.join(self.config.output_dir, 'feature_states', 'feature_states.joblib')
        
        if not os.path.exists(state_file_path):
            raise FileNotFoundError(f"Feature state file not found at {state_file_path}!")
        
        logger.info(f"Loading feature states from {state_file_path}")
        states = joblib.load(state_file_path)
        
        # Reconstruct TeamStateArrays if it was saved as a dictionary
        if 'team_rolling' in states:
            for team_id, state_data in states['team_rolling'].items():
                if isinstance(state_data, dict):
                    # This correctly initializes the new TeamStateArrays object
                    states['team_rolling'][team_id] = TeamStateArrays(
                        feature_names=state_data.get('feature_names', []),
                        games=state_data.get('games', []),
                        timestamps=state_data.get('timestamps', [])
                    )
        logger.info("Successfully loaded feature states")
        return states
    
    # In testingInference.py, replace the existing process_todays_games function

    def process_todays_games(self, target_date: str = None) -> pd.DataFrame:
        """
        Process games for a given date by fetching schedule, merging odds,
        and building features from cached states.
        """
        if not target_date:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        target_dt = pd.to_datetime(target_date)
        logger.info(f"Processing games for {target_date} using CACHED STATES from training")
        
        # Step 1: Get games scheduled for this date from MLB API
        todays_games = self.get_todays_games(target_date)
        if todays_games.empty:
            logger.info("No games scheduled to process.")
            return pd.DataFrame()
        
        # Step 2: Build features from cached states
        features_df = self._build_features_from_state(todays_games, self.feature_states, target_dt)
        if features_df.empty:
            logger.info("Feature generation from state resulted in an empty DataFrame.")
            return pd.DataFrame()
        
        # --- START: NEW ODDS INTEGRATION BLOCK ---
        # Step 3: Fetch and merge today's odds from the database
        daily_odds_df = self._get_daily_odds(target_date)
        
        if not daily_odds_df.empty:
            # Create match key on the features dataframe for joining
            features_df['match_key'] = features_df.apply(
                lambda r: f"{r['home_team_abbr']}_{r['away_team_abbr']}_{pd.to_datetime(r['game_date']).strftime('%Y-%m-%d')}",
                axis=1
            )
            
            # Perform a left merge to add odds to the games
            features_df = pd.merge(features_df, daily_odds_df, on='match_key', how='left')
            features_df.drop(columns=['match_key'], inplace=True)
            logger.info("Successfully merged daily odds into the feature set.")
        else:
            logger.warning("Proceeding without betting odds for today.")
            # Ensure columns exist but are empty
            for col in ['home_ml', 'away_ml', 'total_line', 'over_odds', 'under_odds']:
                features_df[col] = np.nan
        # --- END: NEW ODDS INTEGRATION BLOCK ---

        # Step 4: Calculate differentials
        features_df = self._calculate_matchup_differentials(features_df)
        
        # Step 5: Prepare for prediction
        features_df = self._prepare_for_prediction(features_df)
        
        # Step 6: Save daily features
        self._save_daily_features(features_df, target_date)
        
        return features_df
    
    # In your testingInference.py, inside the MLBDailyPipeline class
# Modify the _build_features_from_state function like this:

    def _build_features_from_state(self, todays_games: pd.DataFrame, 
                                feature_states: dict, 
                                target_dt: pd.Timestamp) -> pd.DataFrame:
        """
        Build features using the cached states - exactly like queryv5
        """
        games = []
        
        # Log what columns we have in todays_games
        logger.info(f"Columns available in todays_games: {list(todays_games.columns)}")
        
        # Verify categorical columns are present
        categorical_cols = ['venue', 'dayNight', 'temperature', 'wind_speed', 'wind_dir', 'conditions']
        missing_cats = [col for col in categorical_cols if col not in todays_games.columns]
        if missing_cats:
            logger.warning(f"Missing categorical columns in todays_games: {missing_cats}")
        
        # Group by game to get both teams
        for game_pk, game_group in todays_games.groupby('gamePk'):
            if len(game_group) != 2:
                logger.warning(f"Game {game_pk} doesn't have both teams, skipping")
                continue
            
            # Get home and away teams
            home_row = game_group[game_group['side'] == 'home'].iloc[0]
            away_row = game_group[game_group['side'] == 'away'].iloc[0]
            
            # Log what we're getting from home_row
            logger.debug(f"Game {game_pk} home_row columns: {list(home_row.index)}")
            
            game_dict = {
                'game_pk': game_pk,
                'game_date': home_row['game_date'],
                
                # Get categorical features with proper defaults
                'venue': home_row.get('venue', 'Unknown'),
                'dayNight': home_row.get('dayNight', 'day'),
                'temperature': home_row.get('temperature', 72.0),
                'wind_speed': home_row.get('wind_speed', 5.0),
                'wind_dir': home_row.get('wind_dir', 'Varies'),
                'conditions': home_row.get('conditions', 'Clear'),
                'game_time': home_row.get('game_time', '19:00'),
                
                'home_team': home_row['team'],
                'away_team': away_row['team'],
                'home_team_id': home_row['team_id'],
                'away_team_id': away_row['team_id'],
                'home_team_abbr': home_row['team'],
                'away_team_abbr': away_row['team']
            }
            
            # Log what we got for this game
            logger.debug(f"Game {game_pk} categorical features: venue={game_dict['venue']}, "
                        f"dayNight={game_dict['dayNight']}, conditions={game_dict['conditions']}, "
                        f"wind_dir={game_dict['wind_dir']}, temp={game_dict['temperature']}")
            
            # Add features from cached states (rest remains the same)
            self._add_team_features_from_state(game_dict, home_row['team_id'], away_row['team_id'], 
                                            feature_states, target_dt)
            self._add_pitcher_features_from_state(game_dict, home_row, away_row, 
                                                feature_states, target_dt)
            self._add_bullpen_features_from_state(game_dict, home_row['team'], away_row['team'],
                                                feature_states, target_dt)
            self._add_splits_features_from_state(game_dict, home_row['team_id'], away_row['team_id'],
                                            feature_states, target_dt)
            self._add_streak_features_from_state(game_dict, home_row['team_id'], away_row['team_id'],
                                            feature_states, target_dt)
            
            games.append(game_dict)
        
        result_df = pd.DataFrame(games)
        
        # Log what categorical columns made it to the final dataframe
        logger.info("Categorical columns in final features dataframe:")
        for col in categorical_cols:
            if col in result_df.columns:
                logger.info(f"  {col}: {result_df[col].value_counts().to_dict()}")
            else:
                logger.warning(f"  {col}: MISSING!")
        
        return result_df
    
    # In testingInference.py -> MLBDailyPipeline class

    def _add_team_features_from_state(self, game_dict: dict, home_team_id: int, away_team_id: int,
                                    feature_states: dict, target_dt: pd.Timestamp):
        """Get team rolling features from the corrected TeamStateArrays object."""

        # Helper to get features for a single team
        def get_features(team_id):
            if team_id in feature_states.get('team_rolling', {}):
                state = feature_states['team_rolling'][team_id]
                # Check if it's the new TeamStateArrays object
                if hasattr(state, 'get_rolling_stats_before'):
                    # Call the method with the required window size
                    return state.get_rolling_stats_before(target_dt, self.config.rolling_games_team)
            return {}

        # Home team
        home_features = get_features(home_team_id)
        for feat_name, feat_val in home_features.items():
            game_dict[f'home_{feat_name}'] = feat_val
        
        # Away team
        away_features = get_features(away_team_id)
        for feat_name, feat_val in away_features.items():
            game_dict[f'away_{feat_name}'] = feat_val
    
    def _add_pitcher_features_from_state(self, game_dict: dict, home_row: pd.Series, away_row: pd.Series,
                                       feature_states: dict, target_dt: pd.Timestamp):
        def get_pitcher_features(pitcher_id):
            if pitcher_id not in feature_states.get('pitcher_rolling', {}):
                logger.debug(f"Pitcher {pitcher_id} not found in feature states")
                return {}
            
            state = feature_states['pitcher_rolling'][pitcher_id]
            
            if 'games' not in state:
                logger.debug(f"Pitcher {pitcher_id} has no 'games' key")
                return {}
            
            # DEBUG: Log the filtering process
            all_games = state['games']
            logger.debug(f"Pitcher {pitcher_id}: Total games in state: {len(all_games)}")
            
            if all_games:
                first_date = pd.to_datetime(all_games[0]['date'])
                last_date = pd.to_datetime(all_games[-1]['date'])
                logger.debug(f"Pitcher {pitcher_id}: Date range in state: {first_date} to {last_date}")
                logger.debug(f"Pitcher {pitcher_id}: Target date for filtering: {target_dt}")
            
            # Filter games before target date
            valid_games = [
                game for game in state['games'] 
                if pd.to_datetime(game['date']) < target_dt
            ]
            
            logger.debug(f"Pitcher {pitcher_id}: Games after filtering: {len(valid_games)}")
            
            if len(valid_games) == 0:
                logger.debug(f"Pitcher {pitcher_id}: No valid games before {target_dt}")
                return {}
            
            # Get last N games
            recent_games = valid_games[-self.config.rolling_games_pitcher:]
            logger.debug(f"Pitcher {pitcher_id}: Using last {len(recent_games)} games for rolling features")
            
            # Calculate rolling features
            features = {}
            all_cols = set()
            for game in recent_games:
                all_cols.update(game['data'].keys())
            
            logger.debug(f"Pitcher {pitcher_id}: Found {len(all_cols)} unique columns in game data")
            
            for col in all_cols:
                values = [g['data'].get(col, np.nan) for g in recent_games]
                values = [v for v in values if not np.isnan(v)]
                if values:
                    features[f"SP_{col}_roll{self.config.rolling_games_pitcher}"] = np.mean(values)
            
            logger.debug(f"Pitcher {pitcher_id}: Created {len(features)} features")
            return features
        
        # Get pitcher IDs
        home_pitcher_id = home_row.get('pitcher_id') or home_row.get('home_SP_id')
        away_pitcher_id = away_row.get('pitcher_id') or away_row.get('away_SP_id')
        
        # Convert to int if needed
        if home_pitcher_id and pd.notna(home_pitcher_id):
            home_pitcher_id = int(home_pitcher_id)
        if away_pitcher_id and pd.notna(away_pitcher_id):
            away_pitcher_id = int(away_pitcher_id)
        
        # Home pitcher
        if home_pitcher_id:
            for feat_name, feat_val in get_pitcher_features(home_pitcher_id).items():
                game_dict[f'home_{feat_name}'] = feat_val
        
        # Away pitcher
        if away_pitcher_id:
            for feat_name, feat_val in get_pitcher_features(away_pitcher_id).items():
                game_dict[f'away_{feat_name}'] = feat_val
    
    def _add_bullpen_features_from_state(self, game_dict: dict, home_team: str, away_team: str,
                                       feature_states: dict, target_dt: pd.Timestamp):
        """Get bullpen features by calculating from raw daily stats - matching queryv5 logic"""
        
        def get_bullpen_features(team):
            # Try different identifiers
            state = None
            for identifier in [team, self.abbrev_to_id.get(team), self.full_name_to_id.get(team)]:
                if identifier in feature_states.get('bullpen_rolling', {}):
                    state = feature_states['bullpen_rolling'][identifier]
                    break
            
            if not state or 'daily_stats' not in state:
                return {}
            
            # Calculate window
            window_start = target_dt - pd.Timedelta(days=self.config.rolling_days_bullpen)
            
            # Filter stats in rolling window before target date
            valid_stats = [
                stat for stat in state['daily_stats']
                if window_start <= pd.to_datetime(stat['date']) < target_dt
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
        
        # Home bullpen
        for feat_name, feat_val in get_bullpen_features(home_team).items():
            game_dict[f'home_{feat_name}'] = feat_val
        
        # Away bullpen
        for feat_name, feat_val in get_bullpen_features(away_team).items():
            game_dict[f'away_{feat_name}'] = feat_val
    
    def _add_splits_features_from_state(self, game_dict: dict, home_team_id: int, away_team_id: int,
                                      feature_states: dict, target_dt: pd.Timestamp):
        """Get splits features by calculating from raw game history - matching queryv5 logic"""
        
        # In testingInference.py -> MLBDailyPipeline -> _add_splits_features_from_state

        def get_splits_features(team_id):
            if team_id not in feature_states.get('team_splits', {}):
                return {}
            
            state = feature_states['team_splits'][team_id]
            
            if 'games' not in state or not state['games']: # Check if list is not empty
                return {}
            
            # Create a DataFrame from the raw history
            games_df = pd.DataFrame(state['games'])
            games_df['date'] = pd.to_datetime(games_df['date']) # Ensure it's datetime
            
            # Filter games in 60-day window before target date
            cutoff_date = target_dt - pd.Timedelta(days=60)
            valid_games = games_df[
                (games_df['date'] < target_dt) &
                (games_df['date'] >= cutoff_date)
            ]
            
            if len(valid_games) < 10:  # minimum games for meaningful splits
                return {}
            
            features = {}
            
            # Day/Night splits
            for time in ['day', 'night']:
                time_games = valid_games[valid_games['dayNight'] == time]
                if not time_games.empty:
                    features[f'{time}_avg_runs'] = float(time_games['batting_runs'].mean())
                    features[f'{time}_ops'] = float(time_games['batting_ops'].mean())
                    features[f'{time}_era'] = float(time_games['pitching_era'].mean())
            
            # Home/Away splits
            for side in ['home', 'away']:
                side_games = valid_games[valid_games['side'] == side]
                if not side_games.empty:
                    features[f'{side}_avg_runs'] = float(side_games['batting_runs'].mean())
                    features[f'{side}_ops'] = float(side_games['batting_ops'].mean())
                    features[f'{side}_era'] = float(side_games['pitching_era'].mean())
            
            return features
        
        # Home splits
        for feat_name, feat_val in get_splits_features(home_team_id).items():
            game_dict[f'home_{feat_name}'] = feat_val
        
        # Away splits
        for feat_name, feat_val in get_splits_features(away_team_id).items():
            game_dict[f'away_{feat_name}'] = feat_val
    
    def _add_streak_features_from_state(self, game_dict: dict, home_team_id: int, away_team_id: int,
                                      feature_states: dict, target_dt: pd.Timestamp):
        """Get streak features by calculating from raw results - matching queryv5 logic"""
        
        def get_streak_features(team_id):
            if team_id not in feature_states.get('team_streaks', {}):
                return {}
            
            state = feature_states['team_streaks'][team_id]
            results = np.array(state['results'])
            dates = np.array(state['dates'], dtype='datetime64[D]')
            
            mask = dates < np.datetime64(target_dt)
            past_results = results[mask]
            
            if len(past_results) == 0:
                return {}
            
            # Calculate streak
            current_streak = 1 if past_results[-1] == 1 else -1
            for i in range(len(past_results) - 2, -1, -1):
                if past_results[i] == past_results[-1]:
                    current_streak = current_streak + 1 if past_results[-1] == 1 else current_streak - 1
                else:
                    break
            
            features = {
                'current_streak': current_streak,
                'win_streak': current_streak if current_streak > 0 else 0,
                'loss_streak': abs(current_streak) if current_streak < 0 else 0,
                'is_on_win_streak': 1 if current_streak > 0 else 0,
                'is_on_loss_streak': 1 if current_streak < 0 else 0,
                'last_5_wins': int(np.sum(past_results[-5:])) if len(past_results) >= 5 else int(np.sum(past_results)),
                'last_10_wins': int(np.sum(past_results[-10:])) if len(past_results) >= 10 else int(np.sum(past_results)),
                'last_20_wins': int(np.sum(past_results[-20:])) if len(past_results) >= 20 else int(np.sum(past_results)),
                'last_10_win_pct': np.sum(past_results[-10:]) / min(10, len(past_results)),
                'last_20_win_pct': np.sum(past_results[-20:]) / min(20, len(past_results)),
                'momentum_5_game': float(np.mean(past_results[-5:])) if len(past_results) >= 5 else float(np.mean(past_results)),
                'streak_intensity': abs(current_streak),
                'streak_pressure': abs(current_streak) * (1 if current_streak < 0 else 0),
                'streak_confidence': abs(current_streak) * (1 if current_streak > 0 else 0),
                'extreme_win_streak': 1 if current_streak >= 7 else 0,
                'extreme_loss_streak': 1 if abs(current_streak) >= 5 and current_streak < 0 else 0,
                'above_500_last_20': 1 if np.sum(past_results[-20:]) / min(20, len(past_results)) > 0.5 else 0,
                'below_400_last_20': 1 if np.sum(past_results[-20:]) / min(20, len(past_results)) < 0.4 else 0,
                'potential_streak_breaker': 0,  # Will be set based on conditions
                'vulnerable_to_upset': 0,  # Will be set based on conditions
                'result_consistency': 0.5  # Default value
            }
            
            # Set conditional features
            if features['is_on_loss_streak'] == 1 and features['momentum_5_game'] > 0.4:
                features['potential_streak_breaker'] = 1
            if features['is_on_win_streak'] == 1 and features['momentum_5_game'] < 0.6 and features['win_streak'] >= 5:
                features['vulnerable_to_upset'] = 1
            
            return features
        
        # Home streaks
        for feat_name, feat_val in get_streak_features(home_team_id).items():
            game_dict[f'home_{feat_name}'] = feat_val
        
        # Away streaks
        for feat_name, feat_val in get_streak_features(away_team_id).items():
            game_dict[f'away_{feat_name}'] = feat_val
    
    def _calculate_matchup_differentials(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate differential features between teams"""
        
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
    
    def get_todays_games(self, target_date: str = None) -> pd.DataFrame:
        """Get scheduled games from MLB API"""
        import requests
        from datetime import date
        
        # Use provided date or today
        if target_date:
            date_str = target_date if isinstance(target_date, str) else pd.to_datetime(target_date).strftime('%Y-%m-%d')
        else:
            date_str = date.today().strftime('%Y-%m-%d')
        
        logger.info(f"Fetching games for {date_str} from MLB API")
        
        # Get games from MLB API
        games_info = self._get_mlb_games_data(date_str)
        
        if not games_info:
            logger.warning(f"No games found for {date_str}")
            return pd.DataFrame()
        
        # Convert to DataFrame format
        games_data = []
        for game in games_info:
            base_info = {
                'gamePk': game['gamePk'],
                'game_date': game['game_date'],
                'game_time': game['game_time'],
                'venue': game['venue'],
                'dayNight': game['dayNight']
            }
            
            # Home team entry
            home_entry = base_info.copy()
            home_entry.update({
                'team': self._get_team_abbrev(game['home_team_id']),
                'opponent_team': self._get_team_abbrev(game['away_team_id']),
                'side': 'home',
                'team_id': game['home_team_id'],
                'pitcher_id': game.get('home_SP_id'),
                'pitcher_name': game.get('home_SP_name')
            })
            games_data.append(home_entry)
            
            # Away team entry
            away_entry = base_info.copy()
            away_entry.update({
                'team': self._get_team_abbrev(game['away_team_id']),
                'opponent_team': self._get_team_abbrev(game['home_team_id']),
                'side': 'away',
                'team_id': game['away_team_id'],
                'pitcher_id': game.get('away_SP_id'),
                'pitcher_name': game.get('away_SP_name')
            })
            games_data.append(away_entry)
        
        games_df = pd.DataFrame(games_data)
        
        # Log before weather
        logger.info(f"Columns before weather: {list(games_df.columns)}")
        
        # Add weather data
        games_df = self._add_weather_data(games_df)
        
        # Log after weather
        logger.info(f"Columns after weather: {list(games_df.columns)}")
        if not games_df.empty:
            weather_sample = games_df[['venue', 'dayNight', 'temperature', 'wind_speed', 'wind_dir', 'conditions']].head(2)
            logger.info(f"Sample weather data:\n{weather_sample}")
        
        logger.info(f"Found {len(games_df)} team records ({len(games_df)//2} games) for {date_str}")
        return games_df
    
    def _create_db_connection(self):
        """Create database connection for additional queries"""
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
        logger.info("Created database connection")
    
    def _load_team_locations(self):
        """Load team location data for weather API"""
        try:
            self.team_locations = pd.read_sql("SELECT * FROM mlb_stadiums", self.engine)
            logger.info(f"Loaded {len(self.team_locations)} stadium locations")
        except Exception as e:
            logger.warning(f"Could not load team locations: {e}")
            self.team_locations = pd.DataFrame()
    
    def _get_team_abbrev(self, team_id):
        """Get team abbreviation from team_id"""
        return self.id_to_abbrev.get(team_id, f"TEAM{team_id}")
    
    def _get_mlb_games_data(self, game_date: str) -> List[Dict]:
        """Get games data from MLB API with probable pitchers"""
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={game_date}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            schedule_data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching MLB schedule: {e}")
            return []
        
        games_info = []
        for date_info in schedule_data.get('dates', []):
            for game in date_info.get('games', []):
                game_pk = game.get('gamePk')
                
                game_info = {
                    'home_team_id': game.get('teams', {}).get('home', {}).get('team', {}).get('id'),
                    'away_team_id': game.get('teams', {}).get('away', {}).get('team', {}).get('id'),
                    'home_team': game.get('teams', {}).get('home', {}).get('team', {}).get('name'),
                    'away_team': game.get('teams', {}).get('away', {}).get('team', {}).get('name'),
                    'gamePk': game_pk,
                    'game_date': game.get('officialDate'),
                    'game_time': self._extract_game_time(game.get('gameDate', '')),
                    'dayNight': game.get('dayNight'),
                    'venue': game.get('venue', {}).get('name', '')
                }
                
                # Get probable pitchers
                logger.info(f"Fetching probable pitchers for game {game_pk}...")
                game_feed = self._get_game_feed(game_pk)
                probable_pitchers = self._extract_probable_pitchers(
                    game_feed,
                    {'home': game_info['home_team_id'], 'away': game_info['away_team_id']}
                )
                
                game_info['home_SP_id'] = probable_pitchers.get('home', {}).get('id') if probable_pitchers.get('home') else None
                game_info['home_SP_name'] = probable_pitchers.get('home', {}).get('fullName') if probable_pitchers.get('home') else None
                game_info['away_SP_id'] = probable_pitchers.get('away', {}).get('id') if probable_pitchers.get('away') else None
                game_info['away_SP_name'] = probable_pitchers.get('away', {}).get('fullName') if probable_pitchers.get('away') else None
                
                logger.info(f"  Home SP: {game_info['home_SP_name']}, Away SP: {game_info['away_SP_name']}")
                games_info.append(game_info)
                time.sleep(0.5)  # Rate limiting
        
        return games_info
    
    def _get_game_feed(self, game_pk: int) -> Dict:
        """Fetch individual game feed from MLB API"""
        url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching game feed for game {game_pk}: {e}")
            return {}
    
    def _extract_game_time(self, game_date_str: str) -> str:
        """Extract time from game date string"""
        time_match = re.search(r'T(\d{2}:\d{2}):', game_date_str)
        return time_match.group(1) if time_match else None
    
    def _extract_probable_pitchers(self, game_feed: Dict, team_ids: Dict) -> Dict:
        """Extract probable pitcher info from game feed"""
        if not game_feed or 'gameData' not in game_feed:
            return {'home': None, 'away': None}
        
        pitchers = game_feed.get('gameData', {}).get('probablePitchers', {})
        
        # Extract home pitcher
        home_pitcher = pitchers.get('home')
        if not home_pitcher or not home_pitcher.get('id'):
            # Fallback to database
            query = f"""
                SELECT TOP 1 home_SP_id AS id, 'Unknown SP' as fullName
                FROM starting_pitchers
                WHERE home_team_id = {team_ids['home']}
                ORDER BY game_pk DESC
            """
            try:
                df = pd.read_sql(query, self.engine)
                home_pitcher = {'id': df.iloc[0]['id'], 'fullName': 'Unknown SP'} if not df.empty else None
            except:
                home_pitcher = None
        
        # Extract away pitcher
        away_pitcher = pitchers.get('away')
        if not away_pitcher or not away_pitcher.get('id'):
            query = f"""
                SELECT TOP 1 away_SP_id AS id, 'Unknown SP' as fullName
                FROM starting_pitchers
                WHERE away_team_id = {team_ids['away']}
                ORDER BY game_pk DESC
            """
            try:
                df = pd.read_sql(query, self.engine)
                away_pitcher = {'id': df.iloc[0]['id'], 'fullName': 'Unknown SP'} if not df.empty else None
            except:
                away_pitcher = None
        
        return {
            'home': home_pitcher,
            'away': away_pitcher
        }
    
    def _add_weather_data(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Add weather forecast data to games"""
        
        # CRITICAL FIX: Initialize ALL games with default values FIRST
        # This ensures every game has valid weather data even if API fails
        games_df['temperature'] = 72.0  # Default temp in Fahrenheit
        games_df['wind_speed'] = 5.0    # Default wind speed in mph
        games_df['wind_dir'] = 'Varies' # Default wind direction
        games_df['conditions'] = 'Clear' # Default conditions
        
        if self.team_locations.empty:
            logger.warning("No stadium locations available, using default weather values")
            return games_df
        
        # Process each unique game
        for game_pk in games_df['gamePk'].unique():
            game_mask = games_df['gamePk'] == game_pk
            game = games_df[game_mask].iloc[0]
            
            # Get venue coordinates
            venue_match = self.team_locations[
                self.team_locations['stadium_name'].str.contains(game['venue'], case=False, na=False)
            ]
            
            if venue_match.empty:
                # Try matching by team
                home_team_id = games_df[game_mask & (games_df['side'] == 'home')]['team_id'].iloc[0]
                venue_match = self.team_locations[self.team_locations['team_id'] == home_team_id]
            
            if not venue_match.empty:
                try:
                    lat = venue_match.iloc[0]['latitude']
                    lon = venue_match.iloc[0]['longitude']
                    
                    # Get game time in UTC
                    game_datetime = pd.to_datetime(f"{game['game_date']} {game['game_time']}")
                    utc_datetime = self._round_to_nearest_hour(
                        (game_datetime + timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M")
                    )
                    
                    # Get weather forecast
                    weather = self._get_weather_forecast(lat, lon, utc_datetime)
                    
                    if weather:
                        # Only update if we got valid weather data
                        games_df.loc[game_mask, 'temperature'] = weather['temperature']
                        games_df.loc[game_mask, 'wind_speed'] = weather['wind_speed']
                        games_df.loc[game_mask, 'wind_dir'] = weather['wind_dir']
                        games_df.loc[game_mask, 'conditions'] = weather['conditions']
                        logger.info(f"Updated weather for game {game_pk}: {weather}")
                    else:
                        logger.warning(f"Weather API failed for game {game_pk}, keeping defaults")
                        
                except Exception as e:
                    logger.warning(f"Error getting weather for game {game_pk}: {e}, keeping defaults")
            else:
                logger.warning(f"No location found for venue: {game['venue']}, keeping defaults")
        
        # Log final weather data to verify
        logger.info("Weather data summary after processing:")
        weather_cols = ['temperature', 'wind_speed', 'wind_dir', 'conditions']
        for col in weather_cols:
            logger.info(f"  {col}: {games_df[col].value_counts().to_dict()}")
        
        return games_df
    
    def _round_to_nearest_hour(self, dt_str: str) -> str:
        """Round datetime to nearest hour"""
        dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M")
        if dt.minute >= 30:
            dt += timedelta(hours=1)
        dt = dt.replace(minute=0)
        return dt.strftime("%Y-%m-%dT%H:00")
    
    def _get_weather_forecast(self, lat: float, lon: float, utc_datetime: str) -> Optional[Dict]:
        """Fetch weather forecast from Open-Meteo API"""
        base_url = "https://api.open-meteo.com/v1/forecast"
        date_part = utc_datetime[:10]
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,weathercode",
            "timezone": "UTC",
            "start_date": date_part,
            "end_date": date_part
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if utc_datetime not in data['hourly']['time']:
                return None
            
            idx = data['hourly']['time'].index(utc_datetime)
            temp_c = data['hourly']['temperature_2m'][idx]
            wind_deg = data['hourly']['wind_direction_10m'][idx]
            weather_code = data['hourly']['weathercode'][idx]
            
            return {
                'temperature': round(temp_c * 9/5 + 32, 1),
                'wind_speed': round(data['hourly']['wind_speed_10m'][idx] * 0.621371, 1),
                'wind_dir': self._map_wind_dir(self._degrees_to_compass(wind_deg)),
                'conditions': self._get_weather_condition(weather_code)
            }
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return None

# In testingInference.py, add this new method inside the MLBDailyPipeline class

    # In testingInference.py, inside the MLBDailyPipeline class, replace this entire function

    def _get_daily_odds(self, target_date: str) -> pd.DataFrame:
        """
        Fetches and processes betting odds for a specific date from the database.
        """
        logger.info(f"Fetching betting odds for {target_date} from SQL...")
        
        # We need to query for a date range to account for UTC time differences
        query_date = pd.to_datetime(target_date)
        # Game day in America spans two UTC days. This captures all games for a given US date.
        start_utc = query_date.strftime('%Y-%m-%d 00:00:00')
        end_utc = (query_date + timedelta(days=1)).strftime('%Y-%m-%d 12:00:00')

        query = f"""
        SELECT home_team, away_team, commence_time, market, outcome, odds, point
        FROM mlb_odds_history
        WHERE commence_time BETWEEN '{start_utc}' AND '{end_utc}'
        """
        try:
            odds_df = pd.read_sql(query, self.engine)
            if odds_df.empty:
                logger.warning(f"No odds found in the database for {target_date}.")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch odds from database: {e}")
            return pd.DataFrame()

        # --- Process Odds (logic adapted from addColumnsv2.py) ---
        utc = pytz.UTC
        pacific = pytz.timezone('US/Pacific')
        odds_df['commence_time'] = pd.to_datetime(odds_df['commence_time'])
        odds_df['game_date_pacific'] = odds_df['commence_time'].dt.tz_localize(utc).dt.tz_convert(pacific).dt.date
        
        # Filter to only games on the target date in Pacific time
        odds_df = odds_df[odds_df['game_date_pacific'] == pd.to_datetime(target_date).date()].copy()

        if odds_df.empty:
            logger.warning(f"No odds found for the specific pacific date {target_date} after timezone conversion.")
            return pd.DataFrame()

        # --- START: CORRECTED TEAM NAME MAPPING ---
        # Create the correct mapping from full team name (from odds) to abbreviation (for merging)
        full_name_to_abbrev = {
            full_name: self.id_to_abbrev.get(team_id)
            for full_name, team_id in self.full_name_to_id.items()
        }
        # --- END: CORRECTED TEAM NAME MAPPING ---

        # Normalize team names
        odds_df['home_team_abbr'] = odds_df['home_team'].map(full_name_to_abbrev)
        odds_df['away_team_abbr'] = odds_df['away_team'].map(full_name_to_abbrev)
        
        # Add debugging for any teams that still fail to map
        unmapped_home = odds_df[odds_df['home_team_abbr'].isna()]['home_team'].unique()
        if len(unmapped_home) > 0:
            logger.warning(f"Could not map these home teams from odds table to an abbreviation: {list(unmapped_home)}")

        unmapped_away = odds_df[odds_df['away_team_abbr'].isna()]['away_team'].unique()
        if len(unmapped_away) > 0:
            logger.warning(f"Could not map these away teams from odds table to an abbreviation: {list(unmapped_away)}")

        # Drop rows where we couldn't map a team, as they can't be joined
        odds_df.dropna(subset=['home_team_abbr', 'away_team_abbr'], inplace=True)
        if odds_df.empty:
            logger.warning("No odds remaining after removing unmappable teams.")
            return pd.DataFrame()

        # Create match key for joining
        odds_df['match_key'] = odds_df.apply(
            lambda r: f"{r['home_team_abbr']}_{r['away_team_abbr']}_{pd.to_datetime(r['game_date_pacific']).strftime('%Y-%m-%d')}",
            axis=1
        )
        
        # Pivot h2h (moneyline)
        h2h = odds_df[odds_df['market'] == 'h2h'].pivot_table(
            index='match_key', columns='outcome', values='odds', aggfunc='first'
        ).reset_index()
        
        if not h2h.empty:
            team_map = odds_df[['match_key', 'home_team', 'away_team']].drop_duplicates()
            h2h = h2h.merge(team_map, on='match_key')
            h2h['home_ml'] = h2h.apply(lambda row: row.get(row['home_team']), axis=1)
            h2h['away_ml'] = h2h.apply(lambda row: row.get(row['away_team']), axis=1)
            h2h_final = h2h[['match_key', 'home_ml', 'away_ml']]
        else:
            h2h_final = pd.DataFrame(columns=['match_key', 'home_ml', 'away_ml'])
            
        # Pivot totals
        totals = odds_df[odds_df['market'] == 'totals']
        if not totals.empty:
            totals_pivot = totals.pivot_table(index='match_key', columns='outcome', values='odds', aggfunc='first')
            lines = totals.groupby('match_key')['point'].first()
            totals_pivot['total_line'] = lines
            totals_final = totals_pivot.rename(columns={'Over': 'over_odds', 'Under': 'under_odds'}).reset_index()
        else:
            totals_final = pd.DataFrame(columns=['match_key', 'over_odds', 'under_odds', 'total_line'])

        # Merge and return
        if h2h_final.empty and totals_final.empty:
            return pd.DataFrame()
        elif h2h_final.empty:
            return totals_final
        elif totals_final.empty:
            return h2h_final
        else:
            return pd.merge(h2h_final, totals_final, on='match_key', how='outer')
            
    def _degrees_to_compass(self, deg: float) -> str:
        """Convert degrees to compass direction"""
        dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        return dirs[int((deg / 22.5) + 0.5) % 16]
    
    def _map_wind_dir(self, compass: str) -> str:
        """Map compass direction to baseball-specific wind direction"""
        mapping = {
            'N': 'In From CF', 'NNE': 'In From CF', 'NE': 'In From LF', 'ENE': 'In From LF',
            'E': 'R To L', 'ESE': 'R To L', 'SE': 'Out To RF', 'SSE': 'Out To CF',
            'S': 'Out To CF', 'SSW': 'Out To CF', 'SW': 'Out To LF', 'WSW': 'L To R',
            'W': 'L To R', 'WNW': 'In From RF', 'NW': 'In From RF', 'NNW': 'In From CF'
        }
        return mapping.get(compass, 'Varies')
    
    def _get_weather_condition(self, weather_code: int) -> str:
        """Convert weather code to condition string"""
        weather_code_map = {
            0: "Clear", 1: "Sunny", 2: "Partly Cloudy", 3: "Overcast", 45: "Fog", 48: "Cloudy",
            51: "Drizzle", 53: "Drizzle", 55: "Drizzle", 61: "Rain", 63: "Rain", 65: "Rain",
            71: "Snow", 73: "Snow", 75: "Snow", 80: "Rain", 81: "Rain", 82: "Rain",
            95: "Rain", 96: "Rain", 99: "Rain"
        }
        return weather_code_map.get(weather_code, f"Unknown ({weather_code})")
    
    # In testingInference.py

    def _prepare_for_prediction(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model prediction"""
        # Ensure all expected features are present
        missing_features = set(self.feature_names) - set(features_df.columns)
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features, filling with defaults")
            for feat in missing_features:
                features_df[feat] = 0
        
        # Define ALL columns we want to keep (both metadata and categorical)
        metadata_cols = [
            'game_pk', 'game_date', 'game_time', 'home_team', 'away_team',
            'home_team_id', 'away_team_id', 'home_team_abbr', 'away_team_abbr',
            # CRITICAL: Include categorical columns here
            'venue', 'dayNight', 'temperature', 'wind_speed', 'wind_dir', 'conditions'
        ]
        
        # Filter for columns that actually exist in the DataFrame
        existing_metadata_cols = [col for col in metadata_cols if col in features_df.columns]
        
        # Get the feature columns in the correct order for the model
        # BUT exclude the categorical columns from feature_names since they need to stay raw
        categorical_cols = ['venue', 'dayNight', 'temperature', 'wind_speed', 'wind_dir', 'conditions']
        feature_cols_ordered = [col for col in self.feature_names 
                            if col in features_df.columns and col not in categorical_cols]
        
        # Combine all columns we want to keep
        all_cols_to_keep = list(dict.fromkeys(existing_metadata_cols + feature_cols_ordered))
        
        # Rebuild the DataFrame
        features_df = features_df[all_cols_to_keep]
        
        # Handle any remaining NaN values in numeric columns only
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_cols] = features_df[numeric_cols].fillna(0)
        
        # Log final categorical data
        logger.info("Final categorical data before returning:")
        for col in categorical_cols:
            if col in features_df.columns:
                sample_values = features_df[col].value_counts().head(3).to_dict()
                logger.info(f"  {col}: {sample_values}")
        
        return features_df
    
    def _save_daily_features(self, features_df: pd.DataFrame, target_date: str = None):
        """Save processed features"""
        date_str = pd.to_datetime(target_date).strftime('%Y%m%d') if target_date else datetime.now().strftime('%Y%m%d')
        
        # Create daily directory
        daily_dir = os.path.join(self.config.output_dir, 'daily_features')
        os.makedirs(daily_dir, exist_ok=True)
        
        # Save features
        output_path = os.path.join(daily_dir, f'features_{date_str}.parquet')
        features_df.to_parquet(output_path)
        logger.info(f"Saved features to {output_path}")
        
        # Also save as CSV for inspection
        csv_path = os.path.join(daily_dir, f'features_{date_str}.csv')
        features_df.to_csv(csv_path, index=False)
        
        # Save metadata
        metadata = {
            'process_time': datetime.now().isoformat(),
            'target_date': target_date or datetime.now().strftime('%Y-%m-%d'),
            'games_processed': len(features_df),
            'feature_count': len([col for col in self.feature_names if col in features_df.columns]),
            'games': [
                {
                    'game_pk': row['game_pk'],
                    'home_team': row.get('home_team', 'Unknown'),
                    'away_team': row.get('away_team', 'Unknown'),
                    'game_time': row.get('game_time', 'TBD')
                }
                for _, row in features_df.iterrows()
            ]
        }
        
        metadata_path = os.path.join(daily_dir, f'metadata_{date_str}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def run_morning_pipeline(target_date: str = None):
    """Main function to run the morning pipeline with true on-the-fly calculation"""
    logger.info("="*60)
    logger.info("Starting MLB Daily Pipeline (On-the-fly Calculation)")
    logger.info(f"Process time: {datetime.now()}")
    logger.info("="*60)
    
    try:
        # Initialize pipeline
        pipeline = MLBDailyPipeline()
        
        # Process games
        features_df = pipeline.process_todays_games(target_date)
        
        if not features_df.empty:
            logger.info(f"\n✅ Successfully processed {len(features_df)} games")
            logger.info("\nGames processed:")
            
            for idx, game in features_df.iterrows():
                away_team = game.get('away_team', 'Unknown')
                home_team = game.get('home_team', 'Unknown')
                game_time = game.get('game_time', 'TBD')
                game_pk = game.get('game_pk', 'Unknown')
                
                logger.info(f"  {away_team} @ {home_team} - {game_time} (Game ID: {game_pk})")
            
            # Show feature summary
            logger.info(f"\nFeature summary:")
            logger.info(f"  Total features: {len(features_df.columns)}")
            logger.info(f"  Non-zero features: {(features_df != 0).any().sum()}")
            
            # Check for each feature type
            feature_types = {
                'Rolling': [col for col in features_df.columns if '_roll' in col],
                'Streak': [col for col in features_df.columns if 'streak' in col.lower() or 'momentum' in col],
                'Splits': [col for col in features_df.columns if any(x in col for x in ['day_', 'night_', 'home_avg', 'away_avg'])],
                'Pitcher': [col for col in features_df.columns if 'SP_' in col],
                'Bullpen': [col for col in features_df.columns if 'bullpen_' in col],
                'H2H': [col for col in features_df.columns if 'h2h_' in col],
                'Differentials': [col for col in features_df.columns if 'diff_' in col],
                'Odds': [col for col in features_df.columns if any(x in col for x in ['ml', 'odds', 'vig', 'favorite', 'spread'])]
            }
            
            logger.info("\nFeature breakdown by type:")
            for feat_type, cols in feature_types.items():
                non_zero_cols = [col for col in cols if (features_df[col] != 0).any()]
                logger.info(f"  {feat_type}: {len(non_zero_cols)}/{len(cols)} features with data")
            
            return features_df
        else:
            logger.info("No games to process today")
            return None
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Run for a specific date
    features = run_morning_pipeline(target_date='2017-04-28')