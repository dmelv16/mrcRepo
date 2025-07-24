# minervaPredictor.py
"""
NRFI Daily Prediction Script
Loads today's games, gets recent lineups, and predicts NRFI using the trained model
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import logging
import requests
import time
from sqlalchemy import create_engine
import urllib.parse
import joblib
import os
from typing import Dict, List, Tuple, Optional

# Import necessary components from existing scripts
from mlbPlayerPropv1 import OptimizedMLBPipeline, OptimizedFeatureEngineer
from testingInference import MLBDailyPipeline

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NRFIPredictor:
    """Handles NRFI predictions for today's games"""
    
    def __init__(self, db_config: Dict[str, str] = None):
        """Initialize the NRFI predictor"""
        
        # Default database configuration
        if db_config is None:
            db_config = {
                'server': "DESKTOP-J9IV3OH",
                'database': "StatcastDB", 
                'username': "mlb_user",
                'password': "mlbAdmin",
                'driver': "ODBC Driver 17 for SQL Server"
            }
        
        self.db_config = db_config
        self._create_db_connection()
        
        # Initialize the pipeline for data loading
        self.inference_pipeline = MLBDailyPipeline()
        
        # Will hold the trained model pipeline
        self.model_pipeline = None

                # --- ADD THIS ---
        self.betting_params = {
            'bankroll': 1000,
            'kelly_fraction': 0.5,
            'min_edge': 0.175,
            'max_bet_pct': 0.05,
            'min_bet': 10,
            'max_bet': 500
        }
        # --- END ADD ---
        
        logger.info("NRFI Predictor initialized")

    # --- ADD THIS ENTIRE METHOD ---
    def _calculate_kelly_bet(self, bankroll: float, edge: float, odds: float) -> float:
        if edge <= 0: return 0
        p = edge + (1 / (odds + 1))
        q = 1 - p
        kelly_fraction = (p * odds - q) / odds
        kelly_fraction *= self.betting_params['kelly_fraction']
        bet_size = bankroll * kelly_fraction
        max_bet = min(bankroll * self.betting_params['max_bet_pct'], self.betting_params['max_bet'])
        return round(max(0, min(bet_size, max_bet)), 2)
    # --- END ADD ---
    
    def _create_db_connection(self):
        """Create database connection"""
        params = urllib.parse.quote_plus(
            f"DRIVER={{{self.db_config['driver']}}};"
            f"SERVER={self.db_config['server']};"
            f"DATABASE={self.db_config['database']};"
            f"UID={self.db_config['username']};"
            f"PWD={self.db_config['password']};"
            f"Encrypt=no;"
            f"TrustServerCertificate=yes;"
        )
        self.engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
        logger.info("Created database connection")
    
    def load_nrfi_model(self, model_path: str = "./models/mlb_predictions.pkl"):
        """Load the trained NRFI model"""
        logger.info(f"Loading NRFI model from {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load the trained pipeline
        self.model_pipeline = OptimizedMLBPipeline()
        self.model_pipeline.load_models()
        
        # Load historical data for feature generation
        # We need some historical context for the feature engineer
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        logger.info(f"Loading historical data from {start_date} to {end_date}")
        self.model_pipeline.all_data = self.model_pipeline.db.load_all_data_bulk(start_date, end_date)
        self.model_pipeline.feature_engineer = OptimizedFeatureEngineer(
            self.model_pipeline.all_data, 
            self.model_pipeline.config
        )
        
        logger.info("Model and feature engineer loaded successfully")
    
    def get_todays_games(self, target_date: str = None) -> List[Dict]:
        """Get today's games with starting pitchers"""
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Fetching games for {target_date}")
        
        # Use the inference pipeline's method to get games
        games_df = self.inference_pipeline.get_todays_games(target_date)
        
        if games_df.empty:
            logger.warning("No games found for today")
            return []
        
        # Convert to list of game dictionaries
        games = []
        for game_pk, game_group in games_df.groupby('gamePk'):
            if len(game_group) != 2:
                continue
            
            home_row = game_group[game_group['side'] == 'home'].iloc[0]
            away_row = game_group[game_group['side'] == 'away'].iloc[0]
            
            game_info = {
                'game_pk': int(game_pk),
                'game_date': target_date,
                'venue': home_row.get('venue', 'Unknown'),
                'home_team_id': int(home_row['team_id']),
                'away_team_id': int(away_row['team_id']),
                'home_team': home_row['team'],
                'away_team': away_row['team'],
                'home_pitcher_id': int(home_row['pitcher_id']) if pd.notna(home_row.get('pitcher_id')) else None,
                'away_pitcher_id': int(away_row['pitcher_id']) if pd.notna(away_row.get('pitcher_id')) else None,
                'home_pitcher_name': home_row.get('pitcher_name', 'TBD'),
                'away_pitcher_name': away_row.get('pitcher_name', 'TBD'),
                'game_time': home_row.get('game_time', '19:00')
            }
            games.append(game_info)
        
        logger.info(f"Found {len(games)} games for {target_date}")
        return games
    
    def get_recent_lineups(self, team_id: int, limit: int = 1) -> List[int]:
        """Get the most recent lineup for a team from the database"""
        query = f"""
        SELECT TOP {limit} 
            game_pk, game_date, player_id, batting_order
        FROM battingOrder
        WHERE team_id = {team_id}
            AND batting_order BETWEEN 1 AND 9
            AND game_date = (
                SELECT MAX(game_date) 
                FROM battingOrder 
                WHERE team_id = {team_id}
            )
        ORDER BY batting_order
        """
        
        try:
            lineup_df = pd.read_sql(query, self.engine)
            if lineup_df.empty:
                logger.warning(f"No lineup found for team {team_id}")
                return []
            
            # Return player IDs in batting order
            lineup = lineup_df.sort_values('batting_order')['player_id'].tolist()
            logger.info(f"Found lineup for team {team_id} from {lineup_df['game_date'].iloc[0]}")
            return lineup
            
        except Exception as e:
            logger.error(f"Error fetching lineup for team {team_id}: {e}")
            return []
    
    def prepare_game_for_prediction(self, game: Dict) -> Dict:
        """Prepare a single game for NRFI prediction"""
        # Get lineups
        home_lineup = self.get_recent_lineups(game['home_team_id'])
        away_lineup = self.get_recent_lineups(game['away_team_id'])
        
        if not home_lineup or not away_lineup:
            logger.warning(f"Missing lineup data for game {game['game_pk']}")
            return None
        
        # Prepare game info in the format expected by create_nrfi_features
        game_info = {
            'game_pk': game['game_pk'],
            'game_date': game['game_date'],
            'venue': game['venue'],
            'home_pitcher_id': game['home_pitcher_id'],
            'away_pitcher_id': game['away_pitcher_id']
        }
        
        # Prepare lineups
        lineups = {
            'home_lineup': home_lineup,
            'away_lineup': away_lineup
        }
        
        return game_info, lineups
    
    def predict_nrfi(self, game: Dict) -> Dict:
        """Make NRFI prediction for a single game and calculate bet"""
        game_data = self.prepare_game_for_prediction(game)
        
        if game_data is None:
            return {
                'game_pk': game['game_pk'], 'prediction': None, 'error': 'Missing lineup data',
                'home_team': game['home_team'], 'away_team': game['away_team']
            }
        
        game_info, lineups = game_data
        
        try:
            # Create features and make prediction
            features = self.model_pipeline.feature_engineer.create_nrfi_features(game_info, lineups)
            features_df = pd.DataFrame([features])
            probability = self.model_pipeline.models.predict_nrfi(features_df)[0]
            prediction = "NRFI" if probability > 0.5 else "YRFI"

            # --- Corrected Betting Logic ---
            # Initialize variables to ensure they always exist
            edge = 0
            bet_size = 0
            potential_profit = 0
            bet_percent = 0  # <-- ADD THIS LINE
            
            # Use fixed odds for the calculation
            american_odds = -130
            decimal_odds = 1.77
            implied_prob = 0.565

            # Only calculate edge and bet size for NRFI predictions
            if prediction == "NRFI":
                edge = probability - implied_prob
                
                # Check if the edge meets the minimum threshold
                if edge > self.betting_params['min_edge']:
                    bet_size = self._calculate_kelly_bet(
                        self.betting_params['bankroll'], 
                        edge, 
                        decimal_odds - 1
                    )
                    if bet_size > 0:
                        potential_profit = bet_size * (decimal_odds - 1)
                        bet_percent = (bet_size / self.betting_params['bankroll']) # <-- ADD THIS LINE

            # --- Return dictionary now ALWAYS includes all keys ---
            return {
                'game_pk': game['game_pk'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'home_pitcher': game['home_pitcher_name'],
                'away_pitcher': game['away_pitcher_name'],
                'prediction': prediction,
                'nrfi_probability': float(probability),
                'confidence': abs(probability - 0.5) * 2,
                'game_time': game['game_time'],
                'edge': edge,
                'suggested_bet': bet_size,
                'potential_profit': potential_profit,
                'odds': american_odds,
                'bet_percent_of_bankroll': bet_percent # <-- ADD THIS LINE
            }
            
        except Exception as e:
            logger.error(f"Error predicting game {game['game_pk']}: {e}")
            return {
                'game_pk': game['game_pk'], 'prediction': None, 'error': str(e),
                'home_team': game['home_team'], 'away_team': game['away_team']
            }
    
    def predict_all_games(self, target_date: str = None) -> pd.DataFrame:
        """Make NRFI predictions for all games on the target date"""
        # Get today's games
        games = self.get_todays_games(target_date)
        
        if not games:
            logger.info("No games to predict")
            return pd.DataFrame()
        
        # Make predictions
        predictions = []
        for game in games:
            logger.info(f"Predicting {game['away_team']} @ {game['home_team']}")
            
            # Check if we have pitcher IDs
            if not game['home_pitcher_id'] or not game['away_pitcher_id']:
                logger.warning(f"Missing pitcher data for game {game['game_pk']}, skipping")
                continue
            
            prediction = self.predict_nrfi(game)
            predictions.append(prediction)
            
            # Add a small delay to avoid overwhelming any APIs
            time.sleep(0.1)
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        # Sort by confidence
        if not predictions_df.empty and 'confidence' in predictions_df.columns:
            predictions_df = predictions_df.sort_values('confidence', ascending=False)
        
        return predictions_df
    
    def display_predictions(self, predictions_df: pd.DataFrame):
        """Display predictions in a formatted way"""
        if predictions_df.empty:
            print("\nNo predictions available")
            return
        
        print("\n" + "="*80)
        print("NRFI PREDICTIONS")
        print("="*80)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Total Games: {len(predictions_df)}")
        print("="*80)
        
        for idx, row in predictions_df.iterrows():
            if pd.notna(row.get('nrfi_probability')):
                print(f"\n{row['away_team']} @ {row['home_team']} - {row['game_time']}")
                print(f"  Prediction: {row['prediction']} (Prob: {row['nrfi_probability']:.3f})")

                if row.get('suggested_bet', 0) > 0:
                    print(f"  Odds: {int(row['odds'])}")
                    print(f"  Edge: {row['edge']:.1%}")
                    print(f"  Suggested Bet: ${row['suggested_bet']:.2f} to win ${row['potential_profit']:.2f}")
                    # --- ADD THIS LINE ---
                    print(f"  Bet Percentage: {row['bet_percent_of_bankroll']:.2%}")
            else:
                print(f"\n{row.get('away_team', 'Unknown Team')} @ {row.get('home_team', 'Unknown Team')}")
                print(f"  Error: {row.get('error', 'Unknown error')}")
        
        # Summary statistics
        valid_predictions = predictions_df[predictions_df['prediction'].notna()]
        if not valid_predictions.empty:
            print("\n" + "="*80)
            print("SUMMARY")
            print("="*80)
            print(f"Games with predictions: {len(valid_predictions)}")
            print(f"NRFI predictions: {len(valid_predictions[valid_predictions['prediction'] == 'NRFI'])}")
            print(f"YRFI predictions: {len(valid_predictions[valid_predictions['prediction'] == 'YRFI'])}")
            print(f"Average NRFI probability: {valid_predictions['nrfi_probability'].mean():.3f}")
            
            # High confidence picks
            high_conf = valid_predictions[valid_predictions['confidence'] > 0.7]
            if not high_conf.empty:
                print("\nHIGH CONFIDENCE PICKS (>70%):")
                for _, row in high_conf.iterrows():
                    print(f"  {row['away_team']} @ {row['home_team']}: {row['prediction']} ({row['confidence']:.1%})")
    
    def save_predictions(self, predictions_df: pd.DataFrame, output_dir: str = "./predictions"):
        """Save predictions to file"""
        if predictions_df.empty:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as CSV
        date_str = datetime.now().strftime('%Y%m%d')
        csv_path = os.path.join(output_dir, f'nrfi_predictions_{date_str}.csv')
        predictions_df.to_csv(csv_path, index=False)
        logger.info(f"Saved predictions to {csv_path}")
        
        # Also save as JSON for API consumption
        json_path = os.path.join(output_dir, f'nrfi_predictions_{date_str}.json')
        predictions_df.to_json(json_path, orient='records', indent=2)


def main():
    """Main function to run NRFI predictions"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NRFI Prediction Tool')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--model-path', type=str, default='./models/mlb_predictions.pkl',
                       help='Path to trained model file')
    parser.add_argument('--save', action='store_true', help='Save predictions to file')
    parser.add_argument('--output-dir', type=str, default='./predictions',
                       help='Directory to save predictions')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = NRFIPredictor()
        
        # Load model
        predictor.load_nrfi_model(args.model_path)
        
        # Make predictions
        predictions = predictor.predict_all_games(args.date)
        
        # Display results
        predictor.display_predictions(predictions)
        
        # Save if requested
        if args.save:
            predictor.save_predictions(predictions, args.output_dir)
        
        return predictions
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    predictions = main()