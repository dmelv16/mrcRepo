# File: generate_predictions_cache.py

import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import warnings
import logging

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import the main model class and config from your V3 script
from bettingModelv3 import MLBBettingModelV2, BettingConfig, EnhancedProbabilityCalibrator
from pipelineTrainv5 import (
    MLBNeuralNetV2, MLBNeuralNetV3, MLBNeuralNetWithUncertainty,
    MLBHybridModel, MLBGraphNeuralNetwork, ModelConfig, 
    ImprovedFeatureSelector, StackingEnsemble, TeamGraphAttention,
    TemporalTeamEncoder, PerceiverBlock, AdvancedFeatureEngineer, BaseModelWrapper
)
# In generate_predictions_cache.py

def generate_prediction_cache(start_date: str, end_date: str, output_path: str = "model_predictions.pkl"):
    """
    Uses your trained supervised model (V3) to generate a cache of historical predictions
    in the format required by your RL agent (V4).
    """
    logging.info("Initializing the supervised betting model (V3)...")
    config = BettingConfig()

    # --- THIS IS THE CRITICAL FIX ---
    # Override the default config to prevent pre-filtering.
    # We want the RL agent to see ALL opportunities, good and bad.
    logging.info("Temporarily overriding config to capture all betting opportunities...")
    config.min_edge_moneyline = -1.0  # Capture all ML bets, even very bad ones
    config.min_edge_totals = -1.0     # Capture all total bets
    config.min_confidence_moneyline = 0.0 # Capture all confidences
    config.min_confidence_over = 0.0
    config.min_confidence_under = 0.0
    # --- END FIX ---
    
    # Initialize the model WITH the overridden config
    model = MLBBettingModelV2(config)
    logging.info("Model initialized with unfiltered settings.")

    all_game_predictions = []
    
    date_range = pd.date_range(start=start_date, end=end_date)
    
    for current_date in tqdm(date_range, desc="Generating Predictions"):
        date_str = current_date.strftime('%Y-%m-%d')
        
        try:
            raw_games_data, games_data_combined, odds_data = model.feature_pipeline.fetch_and_process_day(current_date)
            if raw_games_data.empty:
                continue

            ml_bets = model.analyzer.analyze_moneyline(games_data_combined, odds_data)
            total_bets = model.analyzer.analyze_totals(games_data_combined, odds_data)
            all_opportunities_today = ml_bets + total_bets

            game_data_map = defaultdict(dict)
            for bet in all_opportunities_today:
                game_id = str(bet.game_id)
                selection = bet.selection

                if 'game_id' not in game_data_map[game_id]:
                    game_data_map[game_id]['game_id'] = game_id
                    game_data_map[game_id]['date'] = date_str

                game_data_map[game_id][f"{selection}_edge"] = bet.edge
                game_data_map[game_id][f"{selection}_confidence"] = bet.confidence
                game_data_map[game_id][f"{selection}_uncertainty"] = bet.uncertainty
                game_data_map[game_id][f"{selection}_kelly"] = bet.kelly_stake
                
                if bet.bet_type == 'moneyline':
                    game_data_map[game_id][f"{selection}_win_prob"] = bet.probability
                    game_data_map[game_id][f"{selection}_ml_odds"] = bet.odds
                else: 
                    game_data_map[game_id][f"{selection}_prob"] = bet.probability
                    game_data_map[game_id][f"{selection}_odds"] = bet.odds

            for _, game_row in raw_games_data.iterrows():
                game_id = str(game_row['game_id'])
                if game_id in game_data_map:
                    game_data_map[game_id]['home_score'] = game_row['home_score']
                    game_data_map[game_id]['away_score'] = game_row['away_score']
                    game_data_map[game_id]['total_line'] = game_row.get('total_line')

            all_game_predictions.extend(game_data_map.values())

        except Exception as e:
            logging.error(f"Error processing {date_str}: {e}", exc_info=True)
            
    final_df = pd.DataFrame(all_game_predictions)
    final_df = final_df.fillna(0)
    
    final_df.to_pickle(output_path)
    logging.info(f"\nSuccessfully generated and saved {len(final_df)} game predictions to '{output_path}'")
    
    return final_df

if __name__ == "__main__":
    # --- Configuration ---
    # Define the historical period you want to generate predictions for.
    # This range should cover the train, validation, and test sets for your RL agent.
    START_DATE = "2022-04-01"
    END_DATE = "2025-07-14"  # Use data up to yesterday for a complete set

    # --- Run the generation process ---
    generate_prediction_cache(start_date=START_DATE, end_date=END_DATE)