# mlb_backtesting_system.py
"""
Comprehensive MLB Model Backtesting System
Analyzes model performance from 2024 to present with advanced statistics
"""

# ===================================================================
# IMPORTS
# ===================================================================
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from tqdm import tqdm
import joblib
from pathlib import Path
import warnings
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, roc_auc_score, 
    brier_score_loss, log_loss, confusion_matrix, classification_report,
    roc_curve
)
import json
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import xlsxwriter
# Import from your main model file
from mlbPlayerPropv1 import OptimizedMLBPipeline, OptimizedFeatureEngineer

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===================================================================
# MAIN BACKTESTER CLASS
# ===================================================================
class MLBBacktester:
    """Complete backtesting system for MLB predictions"""
    
    def __init__(self, pipeline, start_date: str = '2024-01-01', end_date: str = None):
        """
        Initialize backtester
        
        Args:
            pipeline: The trained OptimizedMLBPipeline instance
            start_date: Start date for backtesting
            end_date: End date for backtesting (None = today)
        """
        self.pipeline = pipeline
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()
        
        # Storage for results
        self.predictions = defaultdict(list)
        self.actuals = defaultdict(list)
        self.metadata = defaultdict(list)
        
        # Betting simulation parameters
        self.betting_params = {
            'bankroll': 10000,  # Starting bankroll
            'kelly_fraction': 0.25,  # Fraction of Kelly criterion to use
            'min_edge': 0.05,  # Minimum edge required to bet
            'max_bet_pct': 0.05,  # Max bet as % of bankroll
            'min_bet': 10,  # Minimum bet size
            'max_bet': 500  # Maximum bet size
        }
        
        # Results storage
        self.performance_metrics = {}
        self.betting_results = {}
        
    def run_full_backtest(self):
        """Run complete backtesting analysis"""
        print("\n" + "="*80)
        print("MLB MODEL BACKTESTING SYSTEM")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print("="*80 + "\n")
        
        # 1. Generate predictions for the backtest period
        print("1. GENERATING PREDICTIONS...")
        self._generate_historical_predictions()
        
        # 2. Calculate performance metrics
        print("\n2. CALCULATING PERFORMANCE METRICS...")
        self._calculate_performance_metrics()
        
        # 3. Run betting simulations
        print("\n3. RUNNING BETTING SIMULATIONS...")
        self._run_betting_simulations()
        
        # 4. Generate visualizations
        print("\n4. GENERATING VISUALIZATIONS...")
        self._generate_visualizations()
        
        # 5. Generate detailed report
        print("\n5. GENERATING DETAILED REPORT...")
        self._generate_report()
        
        print("\n" + "="*80)
        print("BACKTESTING COMPLETE!")
        print("="*80)
        
        return self.performance_metrics, self.betting_results
    
    # ===================================================================
    # PREDICTION GENERATION METHODS
    # ===================================================================
    def _generate_historical_predictions(self):
        """Generate predictions for each day in the backtest period"""
        # Group games by date for batch processing
        daily_games = defaultdict(list)
        
        # Get all games in the period
        all_games = self.pipeline.feature_engineer.at_bat_results[
            (self.pipeline.feature_engineer.at_bat_results['game_date'] >= self.start_date) &
            (self.pipeline.feature_engineer.at_bat_results['game_date'] <= self.end_date)
        ]
        
        unique_games = all_games[['game_pk', 'game_date']].drop_duplicates()
        
        for _, game in unique_games.iterrows():
            daily_games[game['game_date'].date()].append(game['game_pk'])
        
        # Process each day
        for date, game_pks in tqdm(sorted(daily_games.items()), desc="Processing dates"):
            self._process_date_predictions(date, game_pks)
    
    def _process_date_predictions(self, date: datetime.date, game_pks: List[int]):
        """Process predictions for a specific date"""
        # 1. Hits predictions
        self._process_hits_predictions(date, game_pks)
        
        # 2. Home run predictions
        self._process_home_run_predictions(date, game_pks)
        
        # 3. Strikeout predictions
        # self._process_strikeout_predictions(date, game_pks)
        
        # 4. NRFI predictions
        self._process_nrfi_predictions(date, game_pks)
    
    def _process_hits_predictions(self, date: datetime.date, game_pks: List[int]):
        """Process hits predictions for games on a specific date"""
        # Get player games for this date
        player_games = self.pipeline.feature_engineer.at_bat_results[
            (self.pipeline.feature_engineer.at_bat_results['game_pk'].isin(game_pks))
        ].groupby(['game_pk', 'batter']).agg({
            'is_hit': 'sum',
            'is_at_bat': 'sum',
            'pitcher': 'first',
            'home_team': 'first',
            'away_team': 'first'
        }).reset_index()

        player_games = player_games[player_games['is_at_bat'] >= 1]
        
        # Get all batting orders for the given games to avoid repeated lookups
        all_lineups = self.pipeline.all_data['batting_orders'][
            self.pipeline.all_data['batting_orders']['game_pk'].isin(game_pks)
        ]

        for _, row in player_games.iterrows():
            try:
                # Create batter features
                batter_features = self.pipeline.feature_engineer.create_batter_features(
                    int(row['batter']),
                    str(date),
                    int(row['pitcher'])
                )

                # Get game features
                game_info = self._get_game_info(row['game_pk'])
                game_info['game_date'] = date
                game_features = self.pipeline.feature_engineer.create_game_features(game_info)

                # Get lineup features
                player_lineup_info = all_lineups[all_lineups['player_id'] == row['batter']]
                if not player_lineup_info.empty:
                    batting_order = player_lineup_info.iloc[0]['batting_order']
                    team_type = player_lineup_info.iloc[0]['team_type']
                    
                    game_lineup_ids = all_lineups[
                        all_lineups['team_type'] == team_type
                    ]['player_id'].tolist()
                    
                    # Fixed: Added the missing 'date' argument
                    lineup_features = self.pipeline.feature_engineer.create_lineup_context_features(
                        int(row['batter']), 
                        game_lineup_ids, 
                        int(batting_order), 
                        str(date)  # Added date parameter
                    )
                else:
                    # Create default lineup features if player info is missing
                    lineup_features = {
                        'batting_order': 9, 
                        'is_leadoff': 0, 
                        'is_cleanup': 0, 
                        'lineup_protection': 0.4, 
                        'table_setter_obp': 0.3
                    }

                # Fixed: Added the missing call to create volatility features
                volatility_features = self.pipeline.feature_engineer.create_player_volatility_features(
                    int(row['batter']), 
                    'batter', 
                    str(date)
                )

                # Combine all features
                all_features = {
                    **batter_features, 
                    **game_features, 
                    **lineup_features,
                    **volatility_features  # Ensure volatility features are included
                }

                # Make prediction
                features_df = pd.DataFrame([all_features])
                
                # Check if all required features are present
                required_features = self.pipeline.models.scalers['hits'].feature_names_in_
                missing_features = set(required_features) - set(features_df.columns)
                
                if missing_features:
                    # Create missing features with default values
                    for feature in missing_features:
                        if 'volatility' in feature or 'cv' in feature or 'consistency' in feature:
                            features_df[feature] = 0.1  # Default volatility value
                        elif 'home_factor' in feature:
                            features_df[feature] = 1.0
                        elif 'clutch' in feature:
                            features_df[feature] = 0.0
                        elif 'streak' in feature:
                            features_df[feature] = 0.0
                        else:
                            features_df[feature] = 0.0
                
                # Ensure columns match the training order
                features_df = features_df[required_features]
                
                prediction = self.pipeline.models.predict_hits(features_df)[0]

                # Store results
                self.predictions['hits'].append(prediction)
                self.actuals['hits'].append(row['is_hit'])
                self.metadata['hits'].append({
                    'date': date,
                    'game_pk': row['game_pk'],
                    'batter': row['batter'],
                    'pitcher': row['pitcher'],
                    'at_bats': row['is_at_bat']
                })

            except Exception as e:
                # More specific error logging
                logger.error(f"ERROR processing batter {row['batter']} for hits on {date}: {str(e)}")


    def _process_home_run_predictions(self, date: datetime.date, game_pks: List[int]):
        """Process home run predictions"""
        at_bats = self.pipeline.feature_engineer.at_bat_results[
            (self.pipeline.feature_engineer.at_bat_results['game_pk'].isin(game_pks)) &
            (self.pipeline.feature_engineer.at_bat_results['is_at_bat'])
        ]
        
        if len(at_bats) > 1000:
            at_bats = at_bats.sample(n=1000, random_state=42)
        
        all_lineups = self.pipeline.all_data['batting_orders'][
            self.pipeline.all_data['batting_orders']['game_pk'].isin(game_pks)
        ]

        for _, ab in at_bats.iterrows():
            try:
                # Create batter features
                batter_features = self.pipeline.feature_engineer.create_batter_features(
                    int(ab['batter']),
                    str(date),
                    int(ab['pitcher'])
                )

                game_info = self._get_game_info(ab['game_pk'])
                game_info['game_date'] = date
                game_features = self.pipeline.feature_engineer.create_game_features(game_info)

                # Get lineup features
                player_lineup_info = all_lineups[all_lineups['player_id'] == ab['batter']]
                if not player_lineup_info.empty:
                    batting_order = player_lineup_info.iloc[0]['batting_order']
                    team_type = player_lineup_info.iloc[0]['team_type']
                    
                    game_lineup_ids = all_lineups[
                        all_lineups['team_type'] == team_type
                    ]['player_id'].tolist()
                    
                    # Fixed: Added the missing 'date' argument
                    lineup_features = self.pipeline.feature_engineer.create_lineup_context_features(
                        int(ab['batter']), 
                        game_lineup_ids, 
                        int(batting_order),
                        str(date)  # Added date parameter
                    )
                else:
                    lineup_features = {
                        'batting_order': 9, 
                        'is_leadoff': 0, 
                        'is_cleanup': 0, 
                        'lineup_protection': 0.4, 
                        'table_setter_obp': 0.3
                    }

                # Combine all features
                all_features = {**batter_features, **game_features, **lineup_features}

                # Make prediction
                features_df = pd.DataFrame([all_features])
                
                # Check if all required features are present
                required_features = self.pipeline.models.scalers['home_run'].feature_names_in_
                missing_features = set(required_features) - set(features_df.columns)
                
                if missing_features:
                    # Create missing features with default values
                    for feature in missing_features:
                        features_df[feature] = 0.0
                
                # Ensure columns match the training order
                features_df = features_df[required_features]
                
                prob = self.pipeline.models.predict_home_run_probability(features_df)[0]

                # Store results
                self.predictions['home_run'].append(prob)
                self.actuals['home_run'].append(int(ab['is_home_run']))
                self.metadata['home_run'].append({
                    'date': date,
                    'game_pk': ab['game_pk'],
                    'batter': ab['batter'],
                    'pitcher': ab['pitcher']
                })

            except Exception as e:
                logger.error(f"ERROR processing batter {ab['batter']} for home run on {date}: {str(e)}")
    
    # def _process_strikeout_predictions(self, date: datetime.date, game_pks: List[int]):
    #     """Process strikeout predictions"""
    #     # Get pitcher games
    #     pitcher_games = self.pipeline.feature_engineer.at_bat_results[
    #         self.pipeline.feature_engineer.at_bat_results['game_pk'].isin(game_pks)
    #     ].groupby(['game_pk', 'pitcher']).agg({
    #         'is_strikeout': 'sum',
    #         'batter': 'count'
    #     }).reset_index()
        
    #     # Filter to starting pitchers with enough batters faced
    #     pitcher_games = pitcher_games[pitcher_games['batter'] >= 15]
        
    #     for _, row in pitcher_games.iterrows():
    #         try:
    #             # Create features
    #             features = self.pipeline.feature_engineer.create_pitcher_features(
    #                 int(row['pitcher']), 
    #                 str(date)
    #             )
                
    #             game_info = self._get_game_info(row['game_pk'])
    #             game_info['game_date'] = date # <-- ADD THIS LINE
    #             game_features = self.pipeline.feature_engineer.create_game_features(game_info)
                
    #             all_features = {**features, **game_features}
                
    #             # Make prediction
    #             features_df = pd.DataFrame([all_features])
    #             result = self.pipeline.models.predict_strikeouts(features_df)
                
    #             if isinstance(result, dict):
    #                 prediction = result['prediction'][0] if isinstance(result['prediction'], np.ndarray) else result['prediction']
    #                 probabilities = result.get('probabilities', {})
    #             else:
    #                 prediction = result[0] if isinstance(result, np.ndarray) else result
    #                 probabilities = {}
                
    #             # Store results
    #             self.predictions['strikeouts'].append(prediction)
    #             self.actuals['strikeouts'].append(row['is_strikeout'])
                
    #             metadata = {
    #                 'date': date,
    #                 'game_pk': row['game_pk'],
    #                 'pitcher': row['pitcher'],
    #                 'batters_faced': row['batter']
    #             }
                
    #             # Add probability predictions for different thresholds
    #             for threshold, prob in probabilities.items():
    #                 metadata[f'prob_{threshold}'] = prob[0] if isinstance(prob, np.ndarray) else prob
                
    #             self.metadata['strikeouts'].append(metadata)
                
    #         except Exception as e:
    #             print(f"ERROR processing pitcher {row['pitcher']} on date {date}: {e}")
    
    def _process_nrfi_predictions(self, date: datetime.date, game_pks: List[int]):
        """Process NRFI predictions"""
        for game_pk in game_pks:
            try:
                # Get first inning data
                first_inning = self.pipeline.feature_engineer.at_bat_results[
                    (self.pipeline.feature_engineer.at_bat_results['game_pk'] == game_pk) &
                    (self.pipeline.feature_engineer.at_bat_results['inning'] == 1)
                ]
                
                if len(first_inning) == 0:
                    continue
                
                # Check if run scored in first inning
                run_scored = (first_inning['post_bat_score'] > first_inning['bat_score']).any()
                
                # Get game info and lineups
                game_info = self._get_game_info(game_pk)
                game_info['game_date'] = date
                
                # Get starting pitchers and lineups (simplified)
                starting_pitchers = self._get_starting_pitchers(game_pk)
                if not starting_pitchers:
                    continue
                
                game_info.update(starting_pitchers)
                
                lineups = self._get_lineups(game_pk)
                
                # Create features
                features = self.pipeline.feature_engineer.create_nrfi_features(game_info, lineups)
                
                # Make prediction
                features_df = pd.DataFrame([features])
                prob = self.pipeline.models.predict_nrfi(features_df)[0]
                
                # Store results
                self.predictions['nrfi'].append(prob)
                self.actuals['nrfi'].append(int(not run_scored))
                self.metadata['nrfi'].append({
                    'date': date,
                    'game_pk': game_pk,
                    'home_pitcher': game_info.get('home_pitcher_id'),
                    'away_pitcher': game_info.get('away_pitcher_id')
                })
                
            except Exception as e:
                logger.debug(f"Error processing NRFI for game {game_pk}: {e}")
    
    # ===================================================================
    # PERFORMANCE CALCULATION METHODS
    # ===================================================================
    def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics for each model"""
        
        # 1. Hits Model Metrics
        if self.predictions['hits']:
            self._calculate_regression_metrics('hits')
            self._calculate_hits_threshold_metrics()
        
        # 2. Home Run Model Metrics
        if self.predictions['home_run']:
            self._calculate_classification_metrics('home_run')
            self._calculate_probability_calibration('home_run')
        
        # 3. Strikeout Model Metrics
        if self.predictions['strikeouts']:
            self._calculate_regression_metrics('strikeouts')
            self._calculate_strikeout_threshold_metrics()
        
        # 4. NRFI Model Metrics
        if self.predictions['nrfi']:
            self._calculate_classification_metrics('nrfi')
            self._calculate_probability_calibration('nrfi')
    
    def _calculate_regression_metrics(self, model_type: str):
        """Calculate regression metrics"""
        predictions = np.array(self.predictions[model_type])
        actuals = np.array(self.actuals[model_type])
        
        metrics = {
            'mae': mean_absolute_error(actuals, predictions),
            'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
            'mape': np.mean(np.abs((actuals - predictions) / (actuals + 0.001))) * 100,
            'r2': 1 - (np.sum((actuals - predictions)**2) / np.sum((actuals - actuals.mean())**2)),
            'correlation': np.corrcoef(predictions, actuals)[0, 1],
            'bias': np.mean(predictions - actuals),
            'sample_size': len(predictions)
        }
        
        # Residual analysis
        residuals = actuals - predictions
        metrics['residual_std'] = np.std(residuals)
        metrics['residual_skew'] = stats.skew(residuals)
        metrics['residual_kurtosis'] = stats.kurtosis(residuals)
        
        # Directional accuracy (for hits/strikeouts over/under)
        if model_type == 'hits':
            for threshold in [0.5, 1.5, 2.5]:
                pred_over = predictions > threshold
                actual_over = actuals > threshold
                metrics[f'directional_accuracy_{threshold}'] = np.mean(pred_over == actual_over)
        
        self.performance_metrics[model_type] = metrics
    
    def _calculate_classification_metrics(self, model_type: str):
        """Calculate classification metrics"""
        probabilities = np.array(self.predictions[model_type])
        actuals = np.array(self.actuals[model_type])
        
        # Find optimal threshold using Youden's J statistic
        thresholds = np.linspace(0, 1, 100)
        j_scores = []
        
        for threshold in thresholds:
            predictions = (probabilities > threshold).astype(int)
            if len(np.unique(actuals)) > 1 and len(np.unique(predictions)) > 1:
                tn, fp, fn, tp = confusion_matrix(actuals, predictions).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                j_scores.append(sensitivity + specificity - 1)
            else:
                j_scores.append(0)
        
        optimal_threshold = thresholds[np.argmax(j_scores)]
        predictions = (probabilities > optimal_threshold).astype(int)
        
        # Calculate precision and recall safely
        tp = np.sum((predictions == 1) & (actuals == 1))
        fp = np.sum((predictions == 1) & (actuals == 0))
        fn = np.sum((predictions == 0) & (actuals == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'auc': roc_auc_score(actuals, probabilities) if len(np.unique(actuals)) > 1 else 0.5,
            'brier_score': brier_score_loss(actuals, probabilities),
            'log_loss': log_loss(actuals, probabilities),
            'optimal_threshold': optimal_threshold,
            'accuracy': np.mean(predictions == actuals),
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'sample_size': len(probabilities),
            'positive_rate': np.mean(actuals)
        }
        
        self.performance_metrics[model_type] = metrics
    
    def _calculate_hits_threshold_metrics(self):
        """Calculate performance for different hit thresholds"""
        predictions = np.array(self.predictions['hits'])
        actuals = np.array(self.actuals['hits'])
        
        threshold_metrics = {}
        
        for threshold in [0.5, 1.5, 2.5, 3.5]:
            # Binary classification for over/under
            pred_over = predictions > threshold
            actual_over = actuals > threshold
            
            tp = np.sum((pred_over == True) & (actual_over == True))
            tn = np.sum((pred_over == False) & (actual_over == False))
            fp = np.sum((pred_over == True) & (actual_over == False))
            fn = np.sum((pred_over == False) & (actual_over == True))
            
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            threshold_metrics[f'over_{threshold}'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'samples': len(predictions),
                'actual_over_rate': np.mean(actual_over)
            }
        
        self.performance_metrics['hits_thresholds'] = threshold_metrics
    
    # def _calculate_strikeout_threshold_metrics(self):
    #     """Calculate performance for different strikeout thresholds"""
    #     predictions = np.array(self.predictions['strikeouts'])
    #     actuals = np.array(self.actuals['strikeouts'])
        
    #     threshold_metrics = {}
        
    #     for threshold in [4.5, 5.5, 6.5, 7.5, 8.5, 9.5]:
    #         pred_over = predictions > threshold
    #         actual_over = actuals > threshold
            
    #         if len(np.unique(actual_over)) > 1:
    #             # Get probability predictions if available
    #             probs = []
    #             for i, meta in enumerate(self.metadata['strikeouts']):
    #                 prob_key = f'prob_over_{threshold}'
    #                 if prob_key in meta:
    #                     probs.append(meta[prob_key])
    #                 else:
    #                     # Estimate probability from prediction
    #                     probs.append(1 / (1 + np.exp(-(predictions[i] - threshold))))
                
    #             if probs:
    #                 auc = roc_auc_score(actual_over, probs)
    #                 brier = brier_score_loss(actual_over, probs)
    #             else:
    #                 auc = 0.5
    #                 brier = 0.25
    #         else:
    #             auc = 0.5
    #             brier = 0.25
            
    #         accuracy = np.mean(pred_over == actual_over)
            
    #         threshold_metrics[f'over_{threshold}'] = {
    #             'accuracy': accuracy,
    #             'auc': auc,
    #             'brier_score': brier,
    #             'samples': len(predictions),
    #             'actual_over_rate': np.mean(actual_over)
    #         }
        
    #     self.performance_metrics['strikeout_thresholds'] = threshold_metrics
    
    def _calculate_probability_calibration(self, model_type: str):
        """Calculate probability calibration metrics"""
        probabilities = np.array(self.predictions[model_type])
        actuals = np.array(self.actuals[model_type])
        
        # Create calibration bins
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        calibration_data = []
        
        for i in range(n_bins):
            bin_mask = (probabilities >= bin_edges[i]) & (probabilities < bin_edges[i + 1])
            if np.sum(bin_mask) > 0:
                bin_prob = np.mean(probabilities[bin_mask])
                bin_actual = np.mean(actuals[bin_mask])
                bin_count = np.sum(bin_mask)
                
                calibration_data.append({
                    'bin_center': bin_centers[i],
                    'predicted_prob': bin_prob,
                    'actual_prob': bin_actual,
                    'count': bin_count
                })
        
        # Calculate ECE (Expected Calibration Error)
        ece = 0
        total_samples = len(probabilities)
        
        for cal_data in calibration_data:
            bin_weight = cal_data['count'] / total_samples
            bin_error = abs(cal_data['predicted_prob'] - cal_data['actual_prob'])
            ece += bin_weight * bin_error
        
        self.performance_metrics[f'{model_type}_calibration'] = {
            'calibration_data': calibration_data,
            'ece': ece
        }
    
    # ===================================================================
    # BETTING SIMULATION METHODS
    # ===================================================================
    def _run_betting_simulations(self):
        """Run betting simulations for each model"""
        print("\nRunning betting simulations...")
        
        # 1. Hits betting simulation
        if self.predictions['hits']:
            self._simulate_hits_betting()
        
        # 2. Home run betting simulation
        if self.predictions['home_run']:
            self._simulate_prop_betting('home_run', 'Home Run')
        
        # # 3. Strikeout betting simulation
        # if self.predictions['strikeouts']:
        #     self._simulate_strikeout_betting()
        
        # 4. NRFI betting simulation
        if self.predictions['nrfi']:
            self._simulate_prop_betting('nrfi', 'NRFI')
    
    def _simulate_hits_betting(self):
        """Simulate betting on hits over/under"""
        results = {
            'daily_results': [],
            'bets': []
        }
        
        bankroll = self.betting_params['bankroll']
        
        # Group by date
        daily_data = defaultdict(list)
        for i, meta in enumerate(self.metadata['hits']):
            daily_data[meta['date']].append({
                'prediction': self.predictions['hits'][i],
                'actual': self.actuals['hits'][i],
                'metadata': meta
            })
        
        for date in sorted(daily_data.keys()):
            day_bets = []
            day_profit = 0
            
            for data in daily_data[date]:
                # MODIFIED: Simulate over/under 0.5 hits (player to get a hit)
                threshold = 0.5
                # MODIFIED: Odds set to -200
                implied_prob = 0.6667  # Implied probability for -200 odds
                payout = 0.5           # Payout for a winning -200 bet
                
                # Our probability estimate
                our_prob_over = self._estimate_hits_probability(data['prediction'], threshold)
                
                # Calculate edge
                edge_over = our_prob_over - implied_prob
                # Note: The 'under' bet odds would typically be different, but we simulate them
                # at the same implied probability for simplicity in this backtest.
                edge_under = (1 - our_prob_over) - implied_prob
                
                # Determine bet
                if edge_over > self.betting_params['min_edge']:
                    bet_size = self._calculate_kelly_bet(bankroll, edge_over, payout)
                    if bet_size >= self.betting_params['min_bet']:
                        # Place over bet
                        won = data['actual'] > threshold
                        profit = bet_size * payout if won else -bet_size
                        
                        day_bets.append({
                            'type': f'Over {threshold}',
                            'bet_size': bet_size,
                            'won': won,
                            'profit': profit,
                            'edge': edge_over,
                            'our_prob': our_prob_over
                        })
                        
                        day_profit += profit
                
                elif edge_under > self.betting_params['min_edge']:
                    bet_size = self._calculate_kelly_bet(bankroll, edge_under, payout)
                    if bet_size >= self.betting_params['min_bet']:
                        # Place under bet
                        won = data['actual'] <= threshold
                        profit = bet_size * payout if won else -bet_size
                        
                        day_bets.append({
                            'type': f'Under {threshold}',
                            'bet_size': bet_size,
                            'won': won,
                            'profit': profit,
                            'edge': edge_under,
                            'our_prob': 1 - our_prob_over
                        })
                        
                        day_profit += profit
            
            if day_bets:
                bankroll += day_profit
                results['daily_results'].append({
                    'date': date,
                    'num_bets': len(day_bets),
                    'profit': day_profit,
                    'bankroll': bankroll,
                    'roi': day_profit / sum(bet['bet_size'] for bet in day_bets) if day_bets else 0
                })
                results['bets'].extend(day_bets)
        
        # Calculate overall stats
        if results['bets']:
            total_bet = sum(bet['bet_size'] for bet in results['bets'])
            total_profit = sum(bet['profit'] for bet in results['bets'])
            wins = sum(1 for bet in results['bets'] if bet['won'])
            
            results['summary'] = {
                'total_bets': len(results['bets']),
                'total_wagered': total_bet,
                'total_profit': total_profit,
                'roi': total_profit / total_bet if total_bet > 0 else 0,
                'win_rate': wins / len(results['bets']),
                'final_bankroll': bankroll,
                'bankroll_growth': (bankroll - self.betting_params['bankroll']) / self.betting_params['bankroll'],
                'avg_edge': np.mean([bet['edge'] for bet in results['bets']]),
                'sharpe_ratio': self._calculate_sharpe_ratio(results['daily_results'])
            }
        
        self.betting_results['hits'] = results
    
    # def _simulate_strikeout_betting(self):
    #     """Simulate betting on strikeout over/unders"""
    #     results = {
    #         'daily_results': [],
    #         'bets': []
    #     }
        
    #     bankroll = self.betting_params['bankroll']
        
    #     # Group by date
    #     daily_data = defaultdict(list)
    #     for i, meta in enumerate(self.metadata['strikeouts']):
    #         daily_data[meta['date']].append({
    #             'prediction': self.predictions['strikeouts'][i],
    #             'actual': self.actuals['strikeouts'][i],
    #             'metadata': meta
    #         })
        
    #     for date in sorted(daily_data.keys()):
    #         day_bets = []
    #         day_profit = 0
            
    #         for data in daily_data[date]:
    #             # Check multiple thresholds
    #             for threshold in [5.5, 6.5, 7.5, 8.5]:
    #                 implied_prob = 0.5  # Assuming -110 odds
                    
    #                 # Get our probability if available
    #                 prob_key = f'prob_over_{threshold}'
    #                 if prob_key in data['metadata']:
    #                     our_prob_over = data['metadata'][prob_key]
    #                 else:
    #                     # Estimate from prediction
    #                     our_prob_over = self._estimate_strikeout_probability(
    #                         data['prediction'], threshold
    #                     )
                    
    #                 # Calculate edges
    #                 edge_over = our_prob_over - implied_prob
    #                 edge_under = (1 - our_prob_over) - implied_prob
                    
    #                 # Only bet on strongest edge per game
    #                 if edge_over > self.betting_params['min_edge'] and edge_over > edge_under:
    #                     bet_size = self._calculate_kelly_bet(bankroll, edge_over, 0.91)
    #                     if bet_size >= self.betting_params['min_bet']:
    #                         won = data['actual'] > threshold
    #                         profit = bet_size * 0.91 if won else -bet_size
                            
    #                         day_bets.append({
    #                             'type': f'Over {threshold}',
    #                             'bet_size': bet_size,
    #                             'won': won,
    #                             'profit': profit,
    #                             'edge': edge_over,
    #                             'our_prob': our_prob_over,
    #                             'threshold': threshold
    #                         })
                            
    #                         day_profit += profit
    #                         break  # Only one bet per pitcher
                    
    #                 elif edge_under > self.betting_params['min_edge']:
    #                     bet_size = self._calculate_kelly_bet(bankroll, edge_under, 0.91)
    #                     if bet_size >= self.betting_params['min_bet']:
    #                         won = data['actual'] <= threshold
    #                         profit = bet_size * 0.91 if won else -bet_size
                            
    #                         day_bets.append({
    #                             'type': f'Under {threshold}',
    #                             'bet_size': bet_size,
    #                             'won': won,
    #                             'profit': profit,
    #                             'edge': edge_under,
    #                             'our_prob': 1 - our_prob_over,
    #                             'threshold': threshold
    #                         })
                            
    #                         day_profit += profit
    #                         break
            
    #         if day_bets:
    #             bankroll += day_profit
    #             results['daily_results'].append({
    #                 'date': date,
    #                 'num_bets': len(day_bets),
    #                 'profit': day_profit,
    #                 'bankroll': bankroll,
    #                 'roi': day_profit / sum(bet['bet_size'] for bet in day_bets)
    #             })
    #             results['bets'].extend(day_bets)
        
    #     # Calculate summary
    #     if results['bets']:
    #         total_bet = sum(bet['bet_size'] for bet in results['bets'])
    #         total_profit = sum(bet['profit'] for bet in results['bets'])
    #         wins = sum(1 for bet in results['bets'] if bet['won'])
            
    #         # Breakdown by threshold
    #         threshold_stats = defaultdict(lambda: {'bets': 0, 'wins': 0, 'profit': 0})
    #         for bet in results['bets']:
    #             threshold = bet['threshold']
    #             threshold_stats[threshold]['bets'] += 1
    #             threshold_stats[threshold]['wins'] += int(bet['won'])
    #             threshold_stats[threshold]['profit'] += bet['profit']
            
    #         results['summary'] = {
    #             'total_bets': len(results['bets']),
    #             'total_wagered': total_bet,
    #             'total_profit': total_profit,
    #             'roi': total_profit / total_bet if total_bet > 0 else 0,
    #             'win_rate': wins / len(results['bets']),
    #             'final_bankroll': bankroll,
    #             'bankroll_growth': (bankroll - self.betting_params['bankroll']) / self.betting_params['bankroll'],
    #             'threshold_breakdown': dict(threshold_stats),
    #             'sharpe_ratio': self._calculate_sharpe_ratio(results['daily_results'])
    #         }
        
    #     self.betting_results['strikeouts'] = results
    
    def _simulate_prop_betting(self, model_type: str, bet_name: str):
        """Simulate prop betting (HR, NRFI)"""
        results = {
            'daily_results': [],
            'bets': []
        }
        
        bankroll = self.betting_params['bankroll']
        
        # Group by date
        daily_data = defaultdict(list)
        for i, meta in enumerate(self.metadata[model_type]):
            daily_data[meta['date']].append({
                'probability': self.predictions[model_type][i],
                'actual': self.actuals[model_type][i],
                'metadata': meta
            })
        
        for date in sorted(daily_data.keys()):
            day_bets = []
            day_profit = 0
            
            for data in daily_data[date]:
                # Simulate different odds scenarios
                if model_type == 'home_run':
                    # HR odds typically range from +200 to +800
                    implied_prob = 0.20  # Assuming +400 odds (20% implied)
                    decimal_odds = 5.0
                else:  # NRFI
                    # NRFI typically around -130 to +110
                    implied_prob = 0.565  # Assuming -130 odds
                    decimal_odds = 1.77
                
                our_prob = data['probability']
                edge = our_prob - implied_prob
                
                if edge > self.betting_params['min_edge']:
                    bet_size = self._calculate_kelly_bet(bankroll, edge, decimal_odds - 1)
                    if bet_size >= self.betting_params['min_bet']:
                        won = bool(data['actual'])
                        profit = bet_size * (decimal_odds - 1) if won else -bet_size
                        
                        day_bets.append({
                            'type': bet_name,
                            'bet_size': bet_size,
                            'won': won,
                            'profit': profit,
                            'edge': edge,
                            'our_prob': our_prob,
                            'implied_prob': implied_prob,
                            'odds': decimal_odds
                        })
                        
                        day_profit += profit
            
            if day_bets:
                bankroll += day_profit
                results['daily_results'].append({
                    'date': date,
                    'num_bets': len(day_bets),
                    'profit': day_profit,
                    'bankroll': bankroll,
                    'roi': day_profit / sum(bet['bet_size'] for bet in day_bets)
                })
                results['bets'].extend(day_bets)
        
        # Calculate summary
        if results['bets']:
            total_bet = sum(bet['bet_size'] for bet in results['bets'])
            total_profit = sum(bet['profit'] for bet in results['bets'])
            wins = sum(1 for bet in results['bets'] if bet['won'])
            
            results['summary'] = {
                'total_bets': len(results['bets']),
                'total_wagered': total_bet,
                'total_profit': total_profit,
                'roi': total_profit / total_bet if total_bet > 0 else 0,
                'win_rate': wins / len(results['bets']),
                'final_bankroll': bankroll,
                'bankroll_growth': (bankroll - self.betting_params['bankroll']) / self.betting_params['bankroll'],
                'avg_edge': np.mean([bet['edge'] for bet in results['bets']]),
                'avg_odds': np.mean([bet['odds'] for bet in results['bets']]),
                'sharpe_ratio': self._calculate_sharpe_ratio(results['daily_results'])
            }
        
        self.betting_results[model_type] = results
    
    def _calculate_kelly_bet(self, bankroll: float, edge: float, odds: float) -> float:
        """Calculate bet size using Kelly Criterion"""
        # Kelly formula: f = (p*b - q) / b
        # where p = probability of winning, b = decimal odds - 1, q = 1 - p
        
        if edge <= 0:
            return 0
        
        # Implied probability from edge
        p = edge + (1 / (odds + 1))
        q = 1 - p
        
        kelly_fraction = (p * odds - q) / odds
        
        # Apply Kelly fraction multiplier for safety
        kelly_fraction *= self.betting_params['kelly_fraction']
        
        # Calculate bet size
        bet_size = bankroll * kelly_fraction
        
        # Apply constraints
        max_bet = min(
            bankroll * self.betting_params['max_bet_pct'],
            self.betting_params['max_bet']
        )
        
        bet_size = max(0, min(bet_size, max_bet))
        
        return round(bet_size, 2)
    
    def _estimate_hits_probability(self, prediction: float, threshold: float) -> float:
        """Estimate probability of hits over threshold"""
        # Use Poisson approximation
        from scipy.stats import poisson
        return 1 - poisson.cdf(threshold, prediction)
    
    # def _estimate_strikeout_probability(self, prediction: float, threshold: float) -> float:
    #     """Estimate probability of strikeouts over threshold"""
    #     # Use normal approximation with estimated std
    #     std = 2.0  # Approximate std for strikeouts
    #     z_score = (prediction - threshold - 0.5) / std  # Continuity correction
    #     return 1 - stats.norm.cdf(z_score)
    
    def _calculate_sharpe_ratio(self, daily_results: List[Dict]) -> float:
        """Calculate Sharpe ratio from daily results"""
        if len(daily_results) < 2:
            return 0
        
        daily_returns = []
        for i in range(1, len(daily_results)):
            prev_bankroll = daily_results[i-1]['bankroll']
            curr_bankroll = daily_results[i]['bankroll']
            daily_return = (curr_bankroll - prev_bankroll) / prev_bankroll
            daily_returns.append(daily_return)
        
        if not daily_returns:
            return 0
        
        avg_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        
        # Annualize (assuming 180 betting days per year)
        sharpe = (avg_return * 180) / (std_return * np.sqrt(180)) if std_return > 0 else 0
        
        return sharpe
    
    # ===================================================================
    # VISUALIZATION METHODS
    # ===================================================================
    def _generate_visualizations(self):
        """Generate comprehensive visualizations"""
        # Create output directory
        output_dir = Path('backtest_results')
        output_dir.mkdir(exist_ok=True)
        
        # 1. Performance overview dashboard
        self._create_performance_dashboard(output_dir)
        
        # 2. Betting results visualization
        self._create_betting_charts(output_dir)
        
        # 3. Model calibration plots
        self._create_calibration_plots(output_dir)
        
        # 4. Feature importance analysis
        self._create_feature_analysis(output_dir)
    
    def _create_performance_dashboard(self, output_dir: Path):
        """Create overall performance dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Hits Model', 'Home Run Model', 'Strikeout Model', 'NRFI Model'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        # 1. Hits Model Performance
        if 'hits' in self.predictions and self.predictions['hits']:
            predictions = np.array(self.predictions['hits'])
            actuals = np.array(self.actuals['hits'])
            
            # Scatter plot of predictions vs actuals
            fig.add_trace(
                go.Scatter(
                    x=actuals,
                    y=predictions,
                    mode='markers',
                    name='Predictions',
                    marker=dict(size=4, opacity=0.5),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Add perfect prediction line
            max_val = max(max(predictions), max(actuals))
            fig.add_trace(
                go.Scatter(
                    x=[0, max_val],
                    y=[0, max_val],
                    mode='lines',
                    name='Perfect',
                    line=dict(dash='dash', color='red'),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Add residual histogram on secondary y-axis
            residuals = actuals - predictions
            fig.add_trace(
                go.Histogram(
                    y=residuals,
                    name='Residuals',
                    orientation='h',
                    marker_color='lightblue',
                    showlegend=False
                ),
                row=1, col=1, secondary_y=True
            )
        
        # 2. Home Run Model ROC Curve
        if 'home_run' in self.predictions and self.predictions['home_run']:
            fpr, tpr, _ = roc_curve(self.actuals['home_run'], self.predictions['home_run'])
            
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f'ROC (AUC={self.performance_metrics.get("home_run", {}).get("auc", 0):.3f})',
                    showlegend=True
                ),
                row=1, col=2
            )
            
            # Add diagonal line
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    line=dict(dash='dash', color='gray'),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Strikeout Model Performance
        if 'strikeouts' in self.predictions and self.predictions['strikeouts']:
            predictions = np.array(self.predictions['strikeouts'])
            actuals = np.array(self.actuals['strikeouts'])
            
            # Time series of predictions vs actuals
            dates = [m['date'] for m in self.metadata['strikeouts']]
            
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=predictions,
                    mode='lines',
                    name='Predicted',
                    line=dict(color='blue')
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=actuals,
                    mode='lines',
                    name='Actual',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
        
        # 4. NRFI Model Calibration
        if 'nrfi_calibration' in self.performance_metrics:
            cal_data = self.performance_metrics['nrfi_calibration']['calibration_data']
            df = pd.DataFrame(cal_data)
            
            fig.add_trace(
                go.Scatter(
                    x=df['predicted_prob'],
                    y=df['actual_prob'],
                    mode='markers+lines',
                    name='Calibration',
                    marker=dict(size=10)
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    line=dict(dash='dash', color='gray'),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_xaxes(title_text="Actual Hits", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Hits", row=1, col=1)
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Strikeouts", row=2, col=1)
        fig.update_xaxes(title_text="Predicted Probability", row=2, col=2)
        fig.update_yaxes(title_text="Actual Probability", row=2, col=2)
        
        fig.update_layout(height=800, showlegend=True, title_text="Model Performance Dashboard")
        fig.write_html(output_dir / 'performance_dashboard.html')
    
    def _create_betting_charts(self, output_dir: Path):
        """Create betting performance visualizations"""
        # 1. Combined bankroll evolution chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['All Models Bankroll Evolution', 'Daily P&L Distribution', 
                          'Cumulative ROI by Model', 'Win Rate Analysis'],
            specs=[[{"rowspan": 2}, {}], [None, {}]]
        )
        
        # Bankroll evolution for all models
        colors = {'hits': 'blue', 'home_run': 'red', 'strikeouts': 'green', 'nrfi': 'purple'}
        
        for model_type, results in self.betting_results.items():
            if 'daily_results' not in results or not results['daily_results']:
                continue
            
            daily_df = pd.DataFrame(results['daily_results'])
            
            fig.add_trace(
                go.Scatter(
                    x=daily_df['date'],
                    y=daily_df['bankroll'],
                    mode='lines',
                    name=model_type.title(),
                    line=dict(width=2, color=colors.get(model_type))
                ),
                row=1, col=1
            )
        
        # Add starting bankroll line
        fig.add_hline(
            y=self.betting_params['bankroll'],
            line_dash="dash",
            line_color="gray",
            annotation_text="Starting Bankroll",
            row=1, col=1
        )
        
        # 2. Daily P&L Distribution
        all_daily_pnl = []
        for model_type, results in self.betting_results.items():
            if 'daily_results' in results:
                for day in results['daily_results']:
                    all_daily_pnl.append(day['profit'])
        
        if all_daily_pnl:
            fig.add_trace(
                go.Histogram(
                    x=all_daily_pnl,
                    nbinsx=30,
                    name='Daily P&L',
                    marker_color='lightblue'
                ),
                row=1, col=2
            )
            
            # Add vertical line at 0
            fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=2)
        
        # 3. Cumulative ROI
        for model_type, results in self.betting_results.items():
            if 'daily_results' not in results or not results['daily_results']:
                continue
            
            daily_df = pd.DataFrame(results['daily_results'])
            daily_df['cumulative_wagered'] = daily_df['num_bets'].cumsum() * 100  # Approximate
            daily_df['cumulative_profit'] = daily_df['profit'].cumsum()
            daily_df['cumulative_roi'] = daily_df['cumulative_profit'] / daily_df['cumulative_wagered']
            
            fig.add_trace(
                go.Scatter(
                    x=daily_df['date'],
                    y=daily_df['cumulative_roi'] * 100,
                    mode='lines',
                    name=f'{model_type} ROI',
                    line=dict(width=2, color=colors.get(model_type))
                ),
                row=2, col=2
            )
        
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Bankroll ($)", row=1, col=1)
        fig.update_xaxes(title_text="Daily P&L ($)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="ROI (%)", row=2, col=2)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Comprehensive Betting Performance Analysis"
        )
        
        fig.write_html(output_dir / 'betting_performance.html')
        
        # Create individual model deep dives
        for model_type, results in self.betting_results.items():
            if 'daily_results' in results and results['daily_results']:
                self._create_model_specific_analysis(model_type, results, output_dir)
    
    def _create_model_specific_analysis(self, model_type: str, results: Dict, output_dir: Path):
        """Create detailed analysis for each model"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Bet Size Distribution', 'Edge vs Outcome',
                'P&L by Bet Type', 'Kelly Criterion Analysis',
                'Drawdown Analysis', 'Monthly Performance'
            ]
        )
        
        bets_df = pd.DataFrame(results['bets'])
        daily_df = pd.DataFrame(results['daily_results'])
        
        # 1. Bet Size Distribution
        fig.add_trace(
            go.Histogram(
                x=bets_df['bet_size'],
                nbinsx=30,
                name='Bet Sizes'
            ),
            row=1, col=1
        )
        
        # 2. Edge vs Outcome
        won_bets = bets_df[bets_df['won']]
        lost_bets = bets_df[~bets_df['won']]
        
        fig.add_trace(
            go.Scatter(
                x=won_bets['edge'],
                y=won_bets['profit'],
                mode='markers',
                name='Won',
                marker=dict(color='green', size=5)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=lost_bets['edge'],
                y=lost_bets['profit'],
                mode='markers',
                name='Lost',
                marker=dict(color='red', size=5)
            ),
            row=1, col=2
        )
        
        # 3. P&L by Bet Type
        if 'type' in bets_df.columns:
            type_performance = bets_df.groupby('type').agg({
                'profit': ['sum', 'count'],
                'won': 'mean'
            }).round(2)
            
            fig.add_trace(
                go.Bar(
                    x=type_performance.index,
                    y=type_performance[('profit', 'sum')],
                    name='Total Profit by Type'
                ),
                row=2, col=1
            )
        
        # 4. Kelly Analysis
        if 'our_prob' in bets_df.columns and 'odds' in bets_df.columns:
            bets_df['kelly_fraction'] = bets_df.apply(
                lambda x: (x['our_prob'] * (x['odds'] - 1) - (1 - x['our_prob'])) / (x['odds'] - 1),
                axis=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=bets_df['kelly_fraction'],
                    y=bets_df['bet_size'] / 10000,  # As fraction of bankroll
                    mode='markers',
                    name='Kelly vs Actual'
                ),
                row=2, col=2
            )
        
        # 5. Drawdown Analysis
        daily_df['peak'] = daily_df['bankroll'].cummax()
        daily_df['drawdown'] = (daily_df['bankroll'] - daily_df['peak']) / daily_df['peak'] * 100
        
        fig.add_trace(
            go.Scatter(
                x=daily_df['date'],
                y=daily_df['drawdown'],
                mode='lines',
                fill='tozeroy',
                name='Drawdown %',
                line=dict(color='red')
            ),
            row=3, col=1
        )
        
        # 6. Monthly Performance
        daily_df['month'] = pd.to_datetime(daily_df['date']).dt.to_period('M')
        monthly_perf = daily_df.groupby('month').agg({
            'profit': 'sum',
            'num_bets': 'sum'
        })
        
        fig.add_trace(
            go.Bar(
                x=monthly_perf.index.astype(str),
                y=monthly_perf['profit'],
                name='Monthly P&L'
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text=f"{model_type.title()} Model - Detailed Betting Analysis"
        )
        
        fig.write_html(output_dir / f'{model_type}_detailed_analysis.html')
    
    def _create_calibration_plots(self, output_dir: Path):
        """Create probability calibration plots"""
        for model_type in ['home_run', 'nrfi']:
            if f'{model_type}_calibration' in self.performance_metrics:
                cal_data = self.performance_metrics[f'{model_type}_calibration']['calibration_data']
                
                if not cal_data:
                    continue
                
                df = pd.DataFrame(cal_data)
                
                fig = go.Figure()
                
                # Perfect calibration line
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    name='Perfect Calibration',
                    line=dict(dash='dash', color='gray')
                ))
                
                # Actual calibration
                fig.add_trace(go.Scatter(
                    x=df['predicted_prob'],
                    y=df['actual_prob'],
                    mode='markers+lines',
                    name='Model Calibration',
                    marker=dict(size=df['count']/df['count'].max()*20 + 5)
                ))
                
                fig.update_layout(
                    title=f'{model_type.title()} Probability Calibration',
                    xaxis_title='Predicted Probability',
                    yaxis_title='Actual Probability',
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1])
                )
                
                fig.write_html(output_dir / f'{model_type}_calibration.html')
    
    def _create_feature_analysis(self, output_dir: Path):
        """Analyze feature importance and performance"""
        # Feature importance analysis
        feature_importance_data = {}
        
        # For tree-based models, we can extract feature importances
        for model_type in ['hits', 'home_run', 'strikeouts']:
            if model_type in self.pipeline.models.models:
                model = self.pipeline.models.models[model_type]
                
                # Handle different model types
                if hasattr(model, 'feature_importances_'):
                    # Direct XGBoost model
                    importances = model.feature_importances_
                    feature_names = model.get_booster().feature_names
                elif hasattr(model, 'estimator') and hasattr(model.estimator, 'feature_importances_'):
                    # Calibrated classifier
                    importances = model.estimator.feature_importances_
                    feature_names = model.estimator.get_booster().feature_names
                elif model_type == 'hits' and isinstance(model, dict) and 'base_models' in model:
                    # Ensemble model - get from first base model
                    base_model = model['base_models'][0][1]
                    if hasattr(base_model, 'feature_importances_'):
                        importances = base_model.feature_importances_
                        feature_names = list(self.pipeline.models.scalers[model_type].feature_names_in_)
                    else:
                        continue
                else:
                    continue
                
                feature_importance_data[model_type] = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False).head(20)
        
        # Create feature importance visualization
        if feature_importance_data:
            fig = make_subplots(
                rows=len(feature_importance_data), 
                cols=1,
                subplot_titles=list(feature_importance_data.keys()),
                vertical_spacing=0.1
            )
            
            for i, (model_type, df) in enumerate(feature_importance_data.items(), 1):
                fig.add_trace(
                    go.Bar(
                        x=df['importance'],
                        y=df['feature'],
                        orientation='h',
                        name=model_type
                    ),
                    row=i, col=1
                )
            
            fig.update_layout(
                height=300 * len(feature_importance_data),
                showlegend=False,
                title_text="Top 20 Feature Importances by Model"
            )
            fig.write_html(output_dir / 'feature_importance.html')
        
        # Analyze feature performance over time
        self._analyze_feature_drift(output_dir)
        
        # Analyze prediction patterns
        self._analyze_prediction_patterns(output_dir)
    
    def _analyze_feature_drift(self, output_dir: Path):
        """Analyze how feature distributions change over time"""
        # Group predictions by month
        monthly_performance = defaultdict(lambda: defaultdict(list))
        
        for model_type in ['hits', 'strikeouts']:
            if model_type not in self.predictions:
                continue
            
            for i, meta in enumerate(self.metadata[model_type]):
                month = pd.to_datetime(meta['date']).to_period('M')
                pred = self.predictions[model_type][i]
                actual = self.actuals[model_type][i]
                
                monthly_performance[model_type][month].append({
                    'prediction': pred,
                    'actual': actual,
                    'error': abs(pred - actual)
                })
        
        # Create drift visualization
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Hits Model Monthly Performance', 'Strikeouts Model Monthly Performance']
        )
        
        for i, model_type in enumerate(['hits', 'strikeouts'], 1):
            if model_type not in monthly_performance:
                continue
            
            months = sorted(monthly_performance[model_type].keys())
            monthly_mae = []
            monthly_bias = []
            
            for month in months:
                data = monthly_performance[model_type][month]
                preds = [d['prediction'] for d in data]
                actuals = [d['actual'] for d in data]
                
                mae = np.mean([d['error'] for d in data])
                bias = np.mean([p - a for p, a in zip(preds, actuals)])
                
                monthly_mae.append(mae)
                monthly_bias.append(bias)
            
            fig.add_trace(
                go.Scatter(
                    x=[str(m) for m in months],
                    y=monthly_mae,
                    mode='lines+markers',
                    name='MAE',
                    line=dict(color='red')
                ),
                row=i, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[str(m) for m in months],
                    y=monthly_bias,
                    mode='lines+markers',
                    name='Bias',
                    line=dict(color='blue'),
                    yaxis='y2'
                ),
                row=i, col=1
            )
        
        fig.update_layout(height=600, title_text="Model Performance Drift Analysis")
        fig.write_html(output_dir / 'performance_drift.html')
    
    def _analyze_prediction_patterns(self, output_dir: Path):
        """Analyze patterns in predictions vs actuals"""
        # Analyze by day of week
        day_performance = defaultdict(lambda: defaultdict(list))
        
        for model_type in self.predictions:
            for i, meta in enumerate(self.metadata[model_type]):
                dow = pd.to_datetime(meta['date']).dayofweek
                day_name = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][dow]
                
                if model_type in ['hits', 'strikeouts']:
                    error = abs(self.predictions[model_type][i] - self.actuals[model_type][i])
                    day_performance[model_type][day_name].append(error)
                else:
                    # Classification models
                    correct = (self.predictions[model_type][i] > 0.5) == self.actuals[model_type][i]
                    day_performance[model_type][day_name].append(int(correct))
        
        # Create visualization
        fig = go.Figure()
        
        for model_type in day_performance:
            if model_type in ['hits', 'strikeouts']:
                # Average error by day
                days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                avg_errors = [np.mean(day_performance[model_type][day]) for day in days]
                
                fig.add_trace(go.Bar(
                    x=days,
                    y=avg_errors,
                    name=f'{model_type} MAE'
                ))
        
        fig.update_layout(
            title="Model Performance by Day of Week",
            xaxis_title="Day of Week",
            yaxis_title="Average Error",
            barmode='group'
        )
        fig.write_html(output_dir / 'day_of_week_analysis.html')
    
    # ===================================================================
    # REPORT GENERATION
    # ===================================================================
    def _generate_report(self):
        """Generate comprehensive text report"""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("MLB MODEL BACKTEST REPORT")
        report_lines.append(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        report_lines.append("="*80)
        
        # 1. Model Performance Summary
        report_lines.append("\n1. MODEL PERFORMANCE SUMMARY")
        report_lines.append("-"*40)
        
        for model_type, metrics in self.performance_metrics.items():
            if model_type.endswith('_calibration') or model_type.endswith('_thresholds'):
                continue
            
            report_lines.append(f"\n{model_type.upper()} Model:")
            
            if model_type in ['hits', 'strikeouts']:
                report_lines.append(f"  MAE: {metrics.get('mae', 0):.4f}")
                report_lines.append(f"  RMSE: {metrics.get('rmse', 0):.4f}")
                report_lines.append(f"  Correlation: {metrics.get('correlation', 0):.4f}")
                report_lines.append(f"  Bias: {metrics.get('bias', 0):.4f}")
                report_lines.append(f"  R²: {metrics.get('r2', 0):.4f}")
                report_lines.append(f"  MAPE: {metrics.get('mape', 0):.2f}%")
                
                # Residual analysis
                report_lines.append(f"  Residual Std Dev: {metrics.get('residual_std', 0):.4f}")
                report_lines.append(f"  Residual Skew: {metrics.get('residual_skew', 0):.4f}")
            else:
                report_lines.append(f"  AUC: {metrics.get('auc', 0):.4f}")
                report_lines.append(f"  Brier Score: {metrics.get('brier_score', 0):.4f}")
                report_lines.append(f"  Log Loss: {metrics.get('log_loss', 0):.4f}")
                report_lines.append(f"  Optimal Threshold: {metrics.get('optimal_threshold', 0):.3f}")
                report_lines.append(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
                report_lines.append(f"  Precision: {metrics.get('precision', 0):.4f}")
                report_lines.append(f"  Recall: {metrics.get('recall', 0):.4f}")
                report_lines.append(f"  F1 Score: {metrics.get('f1_score', 0):.4f}")
                report_lines.append(f"  Positive Rate: {metrics.get('positive_rate', 0):.3f}")
            
            report_lines.append(f"  Sample Size: {metrics.get('sample_size', 0):,}")
        
        # 2. Threshold Performance
        report_lines.append("\n2. THRESHOLD PERFORMANCE")
        report_lines.append("-"*40)
        
        if 'hits_thresholds' in self.performance_metrics:
            report_lines.append("\nHits Over/Under:")
            for threshold, perf in self.performance_metrics['hits_thresholds'].items():
                report_lines.append(f"\n  {threshold}:")
                report_lines.append(f"    Accuracy: {perf['accuracy']:.3f}")
                report_lines.append(f"    Precision: {perf['precision']:.3f}")
                report_lines.append(f"    Recall: {perf['recall']:.3f}")
                report_lines.append(f"    Actual Over Rate: {perf['actual_over_rate']:.3f}")
        
        if 'strikeout_thresholds' in self.performance_metrics:
            report_lines.append("\nStrikeouts Over/Under:")
            for threshold, perf in self.performance_metrics['strikeout_thresholds'].items():
                report_lines.append(f"\n  {threshold}:")
                report_lines.append(f"    Accuracy: {perf['accuracy']:.3f}")
                report_lines.append(f"    AUC: {perf.get('auc', 0):.3f}")
                report_lines.append(f"    Brier Score: {perf.get('brier_score', 0):.3f}")
                report_lines.append(f"    Actual Over Rate: {perf['actual_over_rate']:.3f}")
        
        # 3. Betting Simulation Results
        report_lines.append("\n3. BETTING SIMULATION RESULTS")
        report_lines.append("-"*40)
        
        total_profit = 0
        total_wagered = 0
        
        for model_type, results in self.betting_results.items():
            if 'summary' not in results:
                continue
            
            summary = results['summary']
            total_profit += summary['total_profit']
            total_wagered += summary['total_wagered']
            
            report_lines.append(f"\n{model_type.upper()} Betting:")
            report_lines.append(f"  Total Bets: {summary['total_bets']:,}")
            report_lines.append(f"  Total Wagered: ${summary['total_wagered']:,.2f}")
            report_lines.append(f"  Total Profit: ${summary['total_profit']:,.2f}")
            report_lines.append(f"  ROI: {summary['roi']*100:.2f}%")
            report_lines.append(f"  Win Rate: {summary['win_rate']*100:.2f}%")
            report_lines.append(f"  Final Bankroll: ${summary['final_bankroll']:,.2f}")
            report_lines.append(f"  Bankroll Growth: {summary['bankroll_growth']*100:.2f}%")
            report_lines.append(f"  Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}")
            
            if 'avg_edge' in summary:
                report_lines.append(f"  Average Edge: {summary['avg_edge']*100:.2f}%")
            
            # Threshold breakdown for strikeouts
            if 'threshold_breakdown' in summary:
                report_lines.append("\n  Breakdown by Threshold:")
                for threshold, stats in summary['threshold_breakdown'].items():
                    win_rate = stats['wins'] / stats['bets'] if stats['bets'] > 0 else 0
                    report_lines.append(f"    {threshold}: {stats['bets']} bets, "
                                      f"{win_rate*100:.1f}% win rate, "
                                      f"${stats['profit']:.2f} profit")
        
        # Overall betting summary
        if total_wagered > 0:
            report_lines.append(f"\nOVERALL BETTING SUMMARY:")
            report_lines.append(f"  Combined Wagered: ${total_wagered:,.2f}")
            report_lines.append(f"  Combined Profit: ${total_profit:,.2f}")
            report_lines.append(f"  Combined ROI: {(total_profit/total_wagered)*100:.2f}%")
        
        # 4. Key Insights
        report_lines.append("\n4. KEY INSIGHTS AND RECOMMENDATIONS")
        report_lines.append("-"*40)
        
        # Best performing model
        best_roi = -float('inf')
        best_model = None
        most_profitable = None
        highest_profit = -float('inf')
        
        for model, results in self.betting_results.items():
            if 'summary' in results:
                if results['summary']['roi'] > best_roi:
                    best_roi = results['summary']['roi']
                    best_model = model
                if results['summary']['total_profit'] > highest_profit:
                    highest_profit = results['summary']['total_profit']
                    most_profitable = model
        
        if best_model:
            report_lines.append(f"\n- Best ROI: {best_model.upper()} ({best_roi*100:.2f}%)")
        if most_profitable:
            report_lines.append(f"- Most Profitable: {most_profitable.upper()} (${highest_profit:,.2f})")
        
        # Model-specific insights
        report_lines.append("\nMODEL-SPECIFIC INSIGHTS:")
        
        # Hits insights
        if 'hits' in self.performance_metrics:
            hits_metrics = self.performance_metrics['hits']
            report_lines.append(f"\nHits Model:")
            report_lines.append(f"  - Average bias: {hits_metrics.get('bias', 0):.3f} hits")
            if hits_metrics.get('bias', 0) > 0.1:
                report_lines.append("  - Model tends to overpredict hits")
            elif hits_metrics.get('bias', 0) < -0.1:
                report_lines.append("  - Model tends to underpredict hits")
            
            # Best threshold for betting
            if 'hits_thresholds' in self.performance_metrics:
                best_threshold = None
                best_accuracy = 0
                for threshold, perf in self.performance_metrics['hits_thresholds'].items():
                    if perf['accuracy'] > best_accuracy:
                        best_accuracy = perf['accuracy']
                        best_threshold = threshold
                if best_threshold:
                    report_lines.append(f"  - Best threshold for betting: {best_threshold} "
                                      f"(accuracy: {best_accuracy:.3f})")
        
        # Home run insights
        if 'home_run' in self.performance_metrics:
            hr_metrics = self.performance_metrics['home_run']
            report_lines.append(f"\nHome Run Model:")
            report_lines.append(f"  - Model achieves {hr_metrics.get('auc', 0):.3f} AUC")
            if hr_metrics.get('precision', 0) > hr_metrics.get('recall', 0):
                report_lines.append("  - Model is conservative (high precision, lower recall)")
            else:
                report_lines.append("  - Model is aggressive (higher recall, lower precision)")
        
        # Strikeout insights
        if 'strikeout_thresholds' in self.performance_metrics:
            report_lines.append(f"\nStrikeout Model:")
            best_k_threshold = None
            best_k_auc = 0
            for threshold, perf in self.performance_metrics['strikeout_thresholds'].items():
                if perf.get('auc', 0) > best_k_auc:
                    best_k_auc = perf['auc']
                    best_k_threshold = threshold
            if best_k_threshold:
                report_lines.append(f"  - Best threshold: {best_k_threshold} (AUC: {best_k_auc:.3f})")
        
        # Risk analysis
        report_lines.append("\n5. RISK ANALYSIS")
        report_lines.append("-"*40)
        
        for model_type, results in self.betting_results.items():
            if 'daily_results' not in results or not results['daily_results']:
                continue
            
            daily_df = pd.DataFrame(results['daily_results'])
            
            # Calculate max drawdown
            daily_df['peak'] = daily_df['bankroll'].cummax()
            daily_df['drawdown'] = (daily_df['bankroll'] - daily_df['peak']) / daily_df['peak']
            max_drawdown = daily_df['drawdown'].min()
            
            # Calculate longest losing streak
            daily_df['lost_day'] = daily_df['profit'] < 0
            losing_streaks = []
            current_streak = 0
            for lost in daily_df['lost_day']:
                if lost:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        losing_streaks.append(current_streak)
                    current_streak = 0
            
            max_losing_streak = max(losing_streaks) if losing_streaks else 0
            
            # Volatility
            daily_returns = daily_df['profit'].pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0
            
            report_lines.append(f"\n{model_type.upper()} Risk Metrics:")
            report_lines.append(f"  Max Drawdown: {max_drawdown*100:.2f}%")
            report_lines.append(f"  Max Losing Streak: {max_losing_streak} days")
            report_lines.append(f"  Annualized Volatility: {volatility*100:.2f}%")
        
        # 6. Recommendations
        report_lines.append("\n6. RECOMMENDATIONS")
        report_lines.append("-"*40)
        
        recommendations = []
        
        # Check if any models are unprofitable
        unprofitable_models = []
        for model, results in self.betting_results.items():
            if 'summary' in results and results['summary']['total_profit'] < 0:
                unprofitable_models.append(model)
        
        if unprofitable_models:
            recommendations.append(f"- Consider excluding {', '.join(unprofitable_models)} from betting")
        
        # Check for high-performing thresholds
        if 'hits_thresholds' in self.performance_metrics:
            for threshold, perf in self.performance_metrics['hits_thresholds'].items():
                if perf['accuracy'] > 0.6:
                    recommendations.append(f"- Focus on hits {threshold} bets (accuracy: {perf['accuracy']:.3f})")
        
        # Kelly fraction optimization
        for model, results in self.betting_results.items():
            if 'summary' in results and results['summary'].get('sharpe_ratio', 0) > 1.5:
                recommendations.append(f"- Consider increasing Kelly fraction for {model} (high Sharpe ratio)")
        
        # Model calibration
        for model in ['home_run', 'nrfi']:
            if f'{model}_calibration' in self.performance_metrics:
                ece = self.performance_metrics[f'{model}_calibration']['ece']
                if ece > 0.1:
                    recommendations.append(f"- {model} model needs better calibration (ECE: {ece:.3f})")
        
        for rec in recommendations:
            report_lines.append(rec)
        
        # Save report
        report_path = Path('backtest_results') / 'backtest_report.txt'
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Also save a JSON version of key metrics
        json_summary = {
            'period': {
                'start': str(self.start_date.date()),
                'end': str(self.end_date.date())
            },
            'performance_metrics': self.performance_metrics,
            'betting_results': {
                model: {
                    'summary': results.get('summary', {}),
                    'num_days': len(results.get('daily_results', []))
                }
                for model, results in self.betting_results.items()
            }
        }
        
        with open(Path('backtest_results') / 'backtest_summary.json', 'w') as f:
            json.dump(json_summary, f, indent=2, default=str)
        
        # Print to console
        print('\n'.join(report_lines))
    
    # ===================================================================
    # HELPER METHODS
    # ===================================================================
    def _get_game_info(self, game_pk: int) -> Dict:
        """Get game information"""
        game_meta = self.pipeline.all_data['game_metadata']
        game_row = game_meta[game_meta['game_pk'] == game_pk]
        
        if len(game_row) == 0:
            return {
                'temperature': 72,
                'wind_speed': 5,
                'wind_direction': 'In',
                'venue': 'Unknown',
                'start_hour': 19
            }
        
        row = game_row.iloc[0]
        return {
            'temperature': row['temperature'] if pd.notna(row['temperature']) else 72,
            'wind_speed': row['wind_speed'] if pd.notna(row['wind_speed']) else 5,
            'wind_direction': row['wind_direction'] if pd.notna(row['wind_direction']) else 'In',
            'venue': row['venue'] if pd.notna(row['venue']) else 'Unknown',
            'start_hour': 19  # Simplified
        }
    
    def _get_starting_pitchers(self, game_pk: int) -> Dict:
        """Get starting pitchers for a game"""
        starting_pitchers = self.pipeline.all_data['batting_orders'][
            (self.pipeline.all_data['batting_orders']['game_pk'] == game_pk) &
            (self.pipeline.all_data['batting_orders']['is_starting_pitcher'] == 1)
        ]
        
        if len(starting_pitchers) < 2:
            return None
        
        home_pitcher = starting_pitchers[
            starting_pitchers['team_type'] == 'home'
        ].iloc[0]['player_id']
        
        away_pitcher = starting_pitchers[
            starting_pitchers['team_type'] == 'away'
        ].iloc[0]['player_id']
        
        return {
            'home_pitcher_id': home_pitcher,
            'away_pitcher_id': away_pitcher
        }
    
    def _get_lineups(self, game_pk: int) -> Dict:
        """Get lineups for a game"""
        batting_orders = self.pipeline.all_data['batting_orders']
        
        lineups = {}
        for team in ['home', 'away']:
            lineup = batting_orders[
                (batting_orders['game_pk'] == game_pk) &
                (batting_orders['team_type'] == team)
            ]['player_id'].tolist()
            
            lineups[f'{team}_lineup'] = lineup
        
        return lineups
    
    # ===================================================================
    # ADVANCED ANALYSIS METHODS
    # ===================================================================
    def analyze_model_degradation(self):
        """Analyze how model performance degrades over time"""
        degradation_analysis = {}
        
        for model_type in ['hits', 'strikeouts']:
            if model_type not in self.predictions:
                continue
            
            # Group predictions by month
            monthly_metrics = defaultdict(lambda: {'predictions': [], 'actuals': []})
            
            for i, meta in enumerate(self.metadata[model_type]):
                month = pd.to_datetime(meta['date']).to_period('M')
                monthly_metrics[month]['predictions'].append(self.predictions[model_type][i])
                monthly_metrics[month]['actuals'].append(self.actuals[model_type][i])
            
            # Calculate monthly performance
            monthly_performance = []
            for month in sorted(monthly_metrics.keys()):
                preds = np.array(monthly_metrics[month]['predictions'])
                acts = np.array(monthly_metrics[month]['actuals'])
                
                if len(preds) > 10:
                    mae = mean_absolute_error(acts, preds)
                    correlation = np.corrcoef(preds, acts)[0, 1]
                    
                    monthly_performance.append({
                        'month': str(month),
                        'mae': mae,
                        'correlation': correlation,
                        'sample_size': len(preds)
                    })
            
            degradation_analysis[model_type] = pd.DataFrame(monthly_performance)
        
        return degradation_analysis
    
    def analyze_edge_decay(self):
        """Analyze how betting edge changes over time"""
        edge_analysis = {}
        
        for model_type, results in self.betting_results.items():
            if 'bets' not in results:
                continue
            
            bets_df = pd.DataFrame(results['bets'])
            
            # Add dates if not present
            if 'date' not in bets_df.columns:
                dates = []
                bet_counter = 0
                for day_result in results['daily_results']:
                    for _ in range(day_result['num_bets']):
                        dates.append(day_result['date'])
                        bet_counter += 1
                        if bet_counter >= len(bets_df):
                            break
                    if bet_counter >= len(bets_df):
                        break
                bets_df['date'] = dates[:len(bets_df)]
            
            # Calculate rolling edge and win rate
            bets_df = bets_df.sort_values('date')
            bets_df['rolling_edge'] = bets_df['edge'].rolling(50, min_periods=10).mean()
            bets_df['rolling_win_rate'] = bets_df['won'].rolling(50, min_periods=10).mean()
            bets_df['rolling_roi'] = bets_df['profit'].rolling(50, min_periods=10).sum() / \
                                     bets_df['bet_size'].rolling(50, min_periods=10).sum()
            
            edge_analysis[model_type] = bets_df[['date', 'rolling_edge', 'rolling_win_rate', 'rolling_roi']]
        
        return edge_analysis
    
    def analyze_situational_performance(self):
        """Analyze performance in different game situations"""
        situational_analysis = {}
        
        # Analyze by game characteristics
        for model_type in ['hits', 'home_run']:
            if model_type not in self.predictions:
                continue
            
            situational_data = []
            
            for i, meta in enumerate(self.metadata[model_type]):
                game_info = self._get_game_info(meta['game_pk'])
                
                # Day vs Night games
                is_day = game_info.get('start_hour', 19) < 17
                
                # Temperature buckets
                temp = game_info.get('temperature', 72)
                temp_bucket = 'cold' if temp < 60 else 'moderate' if temp < 80 else 'hot'
                
                # Wind conditions
                wind_speed = game_info.get('wind_speed', 5)
                wind_condition = 'calm' if wind_speed < 5 else 'moderate' if wind_speed < 15 else 'strong'
                
                error = abs(self.predictions[model_type][i] - self.actuals[model_type][i])
                
                situational_data.append({
                    'day_night': 'day' if is_day else 'night',
                    'temp_bucket': temp_bucket,
                    'wind_condition': wind_condition,
                    'error': error,
                    'prediction': self.predictions[model_type][i],
                    'actual': self.actuals[model_type][i]
                })
            
            situational_df = pd.DataFrame(situational_data)
            
            # Aggregate by situation
            situation_summary = {}
            
            for column in ['day_night', 'temp_bucket', 'wind_condition']:
                grouped = situational_df.groupby(column).agg({
                    'error': ['mean', 'std', 'count'],
                    'prediction': 'mean',
                    'actual': 'mean'
                }).round(3)
                situation_summary[column] = grouped
            
            situational_analysis[model_type] = situation_summary
        
        return situational_analysis
    
    def generate_advanced_visualizations(self, output_dir: Path):
        """Generate additional advanced visualizations"""
        # 1. Model Degradation Analysis
        degradation = self.analyze_model_degradation()
        
        if degradation:
            fig, axes = plt.subplots(len(degradation), 2, figsize=(15, 5*len(degradation)))
            if len(degradation) == 1:
                axes = axes.reshape(1, -1)
            
            for idx, (model_type, df) in enumerate(degradation.items()):
                if not df.empty:
                    # MAE over time
                    axes[idx, 0].plot(df['month'], df['mae'], marker='o')
                    axes[idx, 0].set_title(f'{model_type.title()} - MAE Over Time')
                    axes[idx, 0].set_xlabel('Month')
                    axes[idx, 0].set_ylabel('MAE')
                    axes[idx, 0].tick_params(axis='x', rotation=45)
                    
                    # Correlation over time
                    axes[idx, 1].plot(df['month'], df['correlation'], marker='o', color='green')
                    axes[idx, 1].set_title(f'{model_type.title()} - Correlation Over Time')
                    axes[idx, 1].set_xlabel('Month')
                    axes[idx, 1].set_ylabel('Correlation')
                    axes[idx, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'model_degradation.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Edge Decay Analysis
        edge_decay = self.analyze_edge_decay()
        
        if edge_decay:
            fig = plt.figure(figsize=(15, 10))
            gs = GridSpec(len(edge_decay), 3, figure=fig)
            
            for idx, (model_type, df) in enumerate(edge_decay.items()):
                # Edge over time
                ax1 = fig.add_subplot(gs[idx, 0])
                ax1.plot(df['date'], df['rolling_edge'], label='Rolling Edge')
                ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax1.set_title(f'{model_type.title()} - Edge Evolution')
                ax1.set_ylabel('Edge')
                
                # Win rate over time
                ax2 = fig.add_subplot(gs[idx, 1])
                ax2.plot(df['date'], df['rolling_win_rate'], label='Rolling Win Rate', color='green')
                ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
                ax2.set_title(f'{model_type.title()} - Win Rate Evolution')
                ax2.set_ylabel('Win Rate')
                
                # ROI over time
                ax3 = fig.add_subplot(gs[idx, 2])
                ax3.plot(df['date'], df['rolling_roi'], label='Rolling ROI', color='purple')
                ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax3.set_title(f'{model_type.title()} - ROI Evolution')
                ax3.set_ylabel('ROI')
                
                # Format x-axis
                for ax in [ax1, ax2, ax3]:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'edge_decay_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Bankroll Risk Analysis
        self._create_risk_analysis_charts(output_dir)
    
    def _create_risk_analysis_charts(self, output_dir: Path):
        """Create comprehensive risk analysis visualizations"""
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig)
        
        # Combined bankroll with confidence bands
        ax1 = fig.add_subplot(gs[0, :])
        
        for model_type, results in self.betting_results.items():
            if 'daily_results' not in results:
                continue
            
            daily_df = pd.DataFrame(results['daily_results'])
            
            # Calculate rolling statistics
            window = min(30, len(daily_df) // 3)
            daily_df['rolling_mean'] = daily_df['bankroll'].rolling(window).mean()
            daily_df['rolling_std'] = daily_df['bankroll'].rolling(window).std()
            
            # Plot with confidence bands
            ax1.plot(daily_df['date'], daily_df['bankroll'], label=model_type.title(), alpha=0.8)
            ax1.fill_between(
                daily_df['date'],
                daily_df['rolling_mean'] - 2 * daily_df['rolling_std'],
                daily_df['rolling_mean'] + 2 * daily_df['rolling_std'],
                alpha=0.2
            )
        
        ax1.axhline(y=self.betting_params['bankroll'], color='black', linestyle='--', alpha=0.5)
        ax1.set_title('Bankroll Evolution with Confidence Bands')
        ax1.set_ylabel('Bankroll ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Value at Risk (VaR) analysis
        ax2 = fig.add_subplot(gs[1, 0])
        
        all_daily_returns = []
        for results in self.betting_results.values():
            if 'daily_results' in results:
                daily_df = pd.DataFrame(results['daily_results'])
                daily_returns = daily_df['profit'].values
                all_daily_returns.extend(daily_returns)
        
        if all_daily_returns:
            # Calculate VaR at different confidence levels
            var_95 = np.percentile(all_daily_returns, 5)
            var_99 = np.percentile(all_daily_returns, 1)
            
            ax2.hist(all_daily_returns, bins=50, alpha=0.7, edgecolor='black')
            ax2.axvline(x=var_95, color='orange', linestyle='--', label=f'VaR 95%: ${var_95:.2f}')
            ax2.axvline(x=var_99, color='red', linestyle='--', label=f'VaR 99%: ${var_99:.2f}')
            ax2.set_title('Daily P&L Distribution with VaR')
            ax2.set_xlabel('Daily P&L ($)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
        
        # Maximum Drawdown periods
        ax3 = fig.add_subplot(gs[1, 1:])
        
        for model_type, results in self.betting_results.items():
            if 'daily_results' not in results:
                continue
            
            daily_df = pd.DataFrame(results['daily_results'])
            daily_df['peak'] = daily_df['bankroll'].cummax()
            daily_df['drawdown'] = (daily_df['bankroll'] - daily_df['peak']) / daily_df['peak'] * 100
            
            ax3.fill_between(
                daily_df['date'],
                0,
                daily_df['drawdown'],
                alpha=0.5,
                label=f'{model_type.title()} DD'
            )
        
        ax3.set_title('Drawdown Periods by Model')
        ax3.set_ylabel('Drawdown (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Kelly Criterion Analysis
        ax4 = fig.add_subplot(gs[2, :])
        
        kelly_analysis = []
        for model_type, results in self.betting_results.items():
            if 'bets' not in results:
                continue
            
            bets_df = pd.DataFrame(results['bets'])
            if 'edge' in bets_df.columns and 'bet_size' in bets_df.columns:
                # Group by edge buckets
                bets_df['edge_bucket'] = pd.cut(bets_df['edge'], bins=10)
                edge_summary = bets_df.groupby('edge_bucket').agg({
                    'won': 'mean',
                    'profit': 'sum',
                    'bet_size': 'mean'
                })
                
                kelly_analysis.append({
                    'model': model_type,
                    'data': edge_summary
                })
        
        if kelly_analysis:
            for analysis in kelly_analysis:
                edge_centers = [interval.mid for interval in analysis['data'].index]
                ax4.scatter(
                    edge_centers,
                    analysis['data']['won'],
                    s=analysis['data']['bet_size'],
                    alpha=0.6,
                    label=analysis['model']
                )
            
            ax4.set_title('Win Rate vs Edge (Bubble size = Bet Size)')
            ax4.set_xlabel('Edge')
            ax4.set_ylabel('Win Rate')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'risk_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def perform_monte_carlo_simulation(self, n_simulations: int = 1000):
        """Perform Monte Carlo simulation for future performance estimation"""
        monte_carlo_results = {}
        
        for model_type, results in self.betting_results.items():
            if 'bets' not in results or not results['bets']:
                continue
            
            bets_df = pd.DataFrame(results['bets'])
            
            # Get empirical distributions
            win_rate = bets_df['won'].mean()
            avg_win = bets_df[bets_df['won']]['profit'].mean()
            avg_loss = bets_df[~bets_df['won']]['profit'].mean()
            avg_bet_size = bets_df['bet_size'].mean()
            bets_per_day = len(bets_df) / ((self.end_date - self.start_date).days + 1)
            
            # Run simulations
            simulation_results = []
            
            for _ in range(n_simulations):
                bankroll = self.betting_params['bankroll']
                daily_results = []
                
                # Simulate 180 days (one season)
                for day in range(180):
                    # Simulate number of bets for the day
                    n_bets = np.random.poisson(bets_per_day)
                    
                    day_profit = 0
                    for _ in range(n_bets):
                        # Determine win/loss
                        if np.random.random() < win_rate:
                            # Win - sample from win distribution
                            profit = np.random.normal(avg_win, abs(avg_win) * 0.2)
                        else:
                            # Loss - sample from loss distribution
                            profit = np.random.normal(avg_loss, abs(avg_loss) * 0.1)
                        
                        day_profit += profit
                    
                    bankroll += day_profit
                    daily_results.append(bankroll)
                
                simulation_results.append({
                    'final_bankroll': bankroll,
                    'total_return': (bankroll - self.betting_params['bankroll']) / self.betting_params['bankroll'],
                    'min_bankroll': min(daily_results),
                    'max_drawdown': 1 - (min(daily_results) / self.betting_params['bankroll'])
                })
            
            # Calculate statistics
            sim_df = pd.DataFrame(simulation_results)
            
            monte_carlo_results[model_type] = {
                'expected_return': sim_df['total_return'].mean(),
                'return_std': sim_df['total_return'].std(),
                'return_percentiles': {
                    '5th': sim_df['total_return'].quantile(0.05),
                    '25th': sim_df['total_return'].quantile(0.25),
                    '50th': sim_df['total_return'].quantile(0.50),
                    '75th': sim_df['total_return'].quantile(0.75),
                    '95th': sim_df['total_return'].quantile(0.95)
                },
                'probability_profit': (sim_df['total_return'] > 0).mean(),
                'probability_double': (sim_df['total_return'] > 1).mean(),
                'expected_max_drawdown': sim_df['max_drawdown'].mean(),
                'var_95': sim_df['total_return'].quantile(0.05),
                'cvar_95': sim_df[sim_df['total_return'] <= sim_df['total_return'].quantile(0.05)]['total_return'].mean()
            }
        
        return monte_carlo_results
    
    def analyze_correlation_between_models(self):
        """Analyze correlation between different model predictions and outcomes"""
        correlation_analysis = {}
        
        # Get daily profits for each model
        daily_profits = {}
        
        for model_type, results in self.betting_results.items():
            if 'daily_results' not in results:
                continue
            
            daily_df = pd.DataFrame(results['daily_results'])
            daily_df.set_index('date', inplace=True)
            daily_profits[model_type] = daily_df[['profit']]
        
        if len(daily_profits) > 1:
            # Merge all daily profits
            combined_df = pd.concat(
                [df.rename(columns={'profit': model}) for model, df in daily_profits.items()],
                axis=1
            ).fillna(0)
            
            # Calculate correlation matrix
            correlation_matrix = combined_df.corr()
            
            # Calculate optimal portfolio weights (simplified Markowitz)
            returns = combined_df.mean()
            cov_matrix = combined_df.cov()
            
            # Equal weight portfolio as baseline
            n_models = len(returns)
            equal_weights = np.ones(n_models) / n_models
            
            # Calculate portfolio statistics
            portfolio_return = np.dot(equal_weights, returns)
            portfolio_std = np.sqrt(np.dot(equal_weights, np.dot(cov_matrix, equal_weights)))
            
            correlation_analysis = {
                'correlation_matrix': correlation_matrix,
                'daily_returns': combined_df,
                'portfolio_stats': {
                    'equal_weight_return': portfolio_return * 252,  # Annualized
                    'equal_weight_std': portfolio_std * np.sqrt(252),
                    'equal_weight_sharpe': (portfolio_return * 252) / (portfolio_std * np.sqrt(252))
                }
            }
        
        return correlation_analysis
    
    def generate_executive_summary(self):
        """Generate a concise executive summary of backtest results"""
        summary = {
            'overview': {
                'backtest_period': f"{self.start_date.date()} to {self.end_date.date()}",
                'days_analyzed': (self.end_date - self.start_date).days,
                'total_predictions': sum(len(preds) for preds in self.predictions.values())
            },
            'model_performance': {},
            'betting_performance': {},
            'key_findings': [],
            'recommendations': []
        }
        
        # Model performance summary
        for model_type in ['hits', 'home_run', 'strikeouts', 'nrfi']:
            if model_type in self.performance_metrics:
                metrics = self.performance_metrics[model_type]
                
                if model_type in ['hits', 'strikeouts']:
                    summary['model_performance'][model_type] = {
                        'mae': round(metrics.get('mae', 0), 4),
                        'correlation': round(metrics.get('correlation', 0), 4),
                        'sample_size': metrics.get('sample_size', 0)
                    }
                else:
                    summary['model_performance'][model_type] = {
                        'auc': round(metrics.get('auc', 0), 4),
                        'accuracy': round(metrics.get('accuracy', 0), 4),
                        'sample_size': metrics.get('sample_size', 0)
                    }
        
        # Betting performance summary
        total_profit = 0
        total_wagered = 0
        
        for model_type, results in self.betting_results.items():
            if 'summary' in results:
                s = results['summary']
                total_profit += s['total_profit']
                total_wagered += s['total_wagered']
                
                summary['betting_performance'][model_type] = {
                    'roi': f"{s['roi']*100:.2f}%",
                    'profit': f"${s['total_profit']:,.2f}",
                    'win_rate': f"{s['win_rate']*100:.2f}%",
                    'sharpe_ratio': round(s.get('sharpe_ratio', 0), 2)
                }
        
        # Overall statistics
        if total_wagered > 0:
            summary['betting_performance']['overall'] = {
                'total_profit': f"${total_profit:,.2f}",
                'total_roi': f"{(total_profit/total_wagered)*100:.2f}%",
                'bankroll_final': f"${self.betting_params['bankroll'] + total_profit:,.2f}"
            }
        
        # Key findings
        # Find best and worst performing models
        best_roi_model = max(
            self.betting_results.items(),
            key=lambda x: x[1].get('summary', {}).get('roi', -float('inf'))
        )[0] if self.betting_results else None
        
        if best_roi_model:
            best_roi = self.betting_results[best_roi_model]['summary']['roi'] * 100
            summary['key_findings'].append(
                f"{best_roi_model.upper()} model shows best ROI at {best_roi:.2f}%"
            )
        
        # Check for model degradation
        degradation = self.analyze_model_degradation()
        for model_type, df in degradation.items():
            if len(df) > 1:
                mae_trend = np.polyfit(range(len(df)), df['mae'].values, 1)[0]
                if mae_trend > 0.01:
                    summary['key_findings'].append(
                        f"{model_type.upper()} model shows performance degradation over time"
                    )
        
        # Add recommendations based on analysis
        if total_profit > 0:
            summary['recommendations'].append(
                "Continue using the current betting strategy with regular model retraining"
            )
        else:
            summary['recommendations'].append(
                "Consider adjusting minimum edge requirements or Kelly fraction"
            )
        
        # Model-specific recommendations
        for model_type, metrics in self.performance_metrics.items():
            if model_type.endswith('_thresholds'):
                continue
            
            if model_type == 'nrfi' and metrics.get('auc', 0) < 0.55:
                summary['recommendations'].append(
                    "NRFI model shows poor predictive power - consider excluding from betting"
                )
            
            if model_type in ['hits', 'strikeouts'] and metrics.get('bias', 0) > 0.2:
                summary['recommendations'].append(
                    f"{model_type.upper()} model has significant bias - consider recalibration"
                )
        
        return summary
    
    def export_results_to_excel(self, filename: str = 'backtest_results.xlsx'):
        """Export all results to Excel for further analysis"""
        output_path = Path('backtest_results') / filename
        
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            # Executive Summary
            exec_summary = self.generate_executive_summary()
            summary_df = pd.DataFrame([
                ['Backtest Period', exec_summary['overview']['backtest_period']],
                ['Days Analyzed', exec_summary['overview']['days_analyzed']],
                ['Total Predictions', exec_summary['overview']['total_predictions']],
                ['', ''],
                ['Model Performance', ''],
            ])
            
            for model, perf in exec_summary['model_performance'].items():
                for metric, value in perf.items():
                    summary_df = pd.concat([
                        summary_df,
                        pd.DataFrame([[f'{model} - {metric}', value]])
                    ])
            
            summary_df.columns = ['Metric', 'Value']
            summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
            
            # Model Performance Details
            for model_type in ['hits', 'home_run', 'strikeouts', 'nrfi']:
                if model_type not in self.predictions:
                    continue
                
                # Create detailed results dataframe
                results_data = []
                for i in range(len(self.predictions[model_type])):
                    row = {
                        'date': self.metadata[model_type][i]['date'],
                        'prediction': self.predictions[model_type][i],
                        'actual': self.actuals[model_type][i],
                        'error': abs(self.predictions[model_type][i] - self.actuals[model_type][i])
                    }
                    
                    # Add metadata
                    for key, value in self.metadata[model_type][i].items():
                        if key != 'date':
                            row[key] = value
                    
                    results_data.append(row)
                
                if results_data:
                    pd.DataFrame(results_data).to_excel(
                        writer,
                        sheet_name=f'{model_type}_predictions',
                        index=False
                    )
            
            # Betting Results
            for model_type, results in self.betting_results.items():
                if 'bets' in results and results['bets']:
                    bets_df = pd.DataFrame(results['bets'])
                    bets_df.to_excel(
                        writer,
                        sheet_name=f'{model_type}_bets',
                        index=False
                    )
            
            # Daily P&L Summary
            all_daily_pnl = []
            for model_type, results in self.betting_results.items():
                if 'daily_results' in results:
                    daily_df = pd.DataFrame(results['daily_results'])
                    daily_df['model'] = model_type
                    all_daily_pnl.append(daily_df)
            
            if all_daily_pnl:
                pd.concat(all_daily_pnl).to_excel(
                    writer,
                    sheet_name='Daily_PnL',
                    index=False
                )
        
        print(f"\nResults exported to: {output_path}")
    
    def create_live_tracking_template(self):
        """Create a template for tracking live predictions going forward"""
        template = {
            'predictions_log': [],
            'betting_log': [],
            'daily_summary': [],
            'model_versions': {
                'trained_date': str(self.start_date.date()),
                'models': list(self.pipeline.models.models.keys())
            }
        }
        
        # Save template
        template_path = Path('backtest_results') / 'live_tracking_template.json'
        with open(template_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        # Create CSV templates
        predictions_template = pd.DataFrame(columns=[
            'date', 'model_type', 'player_id', 'player_name', 'prediction',
            'confidence', 'bet_placed', 'bet_size', 'odds', 'threshold'
        ])
        predictions_template.to_csv(
            Path('backtest_results') / 'predictions_tracking_template.csv',
            index=False
        )
        
        results_template = pd.DataFrame(columns=[
            'date', 'model_type', 'player_id', 'player_name', 'prediction',
            'actual', 'bet_result', 'profit_loss'
        ])
        results_template.to_csv(
            Path('backtest_results') / 'results_tracking_template.csv',
            index=False
        )
        
        print("\nLive tracking templates created in backtest_results folder")


# ===================================================================
# MAIN FUNCTIONS
# ===================================================================
def run_backtest(start_date: str = '2024-01-01', end_date: str = None):
    """Main function to run backtesting"""
    
    # Load the trained pipeline
    print("Loading trained models and data...")
    pipeline = OptimizedMLBPipeline()
    
    # Load recent data for feature generation
    print("Loading historical data for backtesting...")
    data_start = (pd.to_datetime(start_date) - timedelta(days=365)).strftime('%Y-%m-%d')
    data_end = end_date or datetime.now().strftime('%Y-%m-%d')
    
    pipeline.all_data = pipeline.db.load_all_data_bulk(data_start, data_end)
    pipeline.feature_engineer = OptimizedFeatureEngineer(pipeline.all_data, pipeline.config)
    
    # Load the trained models
    print("Loading trained models...")
    pipeline.load_models()
    
    # Initialize and run backtester
    backtester = MLBBacktester(pipeline, start_date, end_date)
    
    # Customize betting parameters if desired
    backtester.betting_params.update({
        'bankroll': 10000,
        'kelly_fraction': 0.25,
        'min_edge': 0.05,
        'max_bet_pct': 0.05
    })
    
    # Run the backtest
    performance_metrics, betting_results = backtester.run_full_backtest()
    
    return backtester, performance_metrics, betting_results


def run_comprehensive_backtest(start_date: str = '2024-01-01', end_date: str = None,
                               monte_carlo_sims: int = 1000, export_excel: bool = True):
    """
    Run a comprehensive backtest with all advanced analytics
    
    Args:
        start_date: Start date for backtesting
        end_date: End date (None = today)
        monte_carlo_sims: Number of Monte Carlo simulations
        export_excel: Whether to export results to Excel
    
    Returns:
        Comprehensive results dictionary
    """
    print("\n" + "="*80)
    print("MLB COMPREHENSIVE BACKTESTING SYSTEM")
    print("="*80 + "\n")
    
    # Initialize
    backtester, performance_metrics, betting_results = run_backtest(start_date, end_date)
    
    print("\n7. RUNNING ADVANCED ANALYTICS...")
    
    # Advanced analytics
    advanced_results = {}
    
    # 1. Model degradation analysis
    print("  - Analyzing model degradation over time...")
    advanced_results['degradation'] = backtester.analyze_model_degradation()
    
    # 2. Edge decay analysis
    print("  - Analyzing betting edge decay...")
    advanced_results['edge_decay'] = backtester.analyze_edge_decay()
    
    # 3. Situational performance
    print("  - Analyzing situational performance...")
    advanced_results['situational'] = backtester.analyze_situational_performance()
    
    # 4. Monte Carlo simulation
    print(f"  - Running {monte_carlo_sims} Monte Carlo simulations...")
    advanced_results['monte_carlo'] = backtester.perform_monte_carlo_simulation(monte_carlo_sims)
    
    # 5. Model correlation analysis
    print("  - Analyzing correlations between models...")
    advanced_results['correlations'] = backtester.analyze_correlation_between_models()
    
    # 6. Generate advanced visualizations
    print("  - Creating advanced visualizations...")
    output_dir = Path('backtest_results')
    backtester.generate_advanced_visualizations(output_dir)
    
    # 7. Executive summary
    print("  - Generating executive summary...")
    executive_summary = backtester.generate_executive_summary()
    
    # Print executive summary
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY")
    print("="*80)
    
    print(f"\nBacktest Period: {executive_summary['overview']['backtest_period']}")
    print(f"Total Predictions Made: {executive_summary['overview']['total_predictions']:,}")
    
    print("\nMODEL PERFORMANCE:")
    for model, metrics in executive_summary['model_performance'].items():
        print(f"\n{model.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
    print("\nBETTING PERFORMANCE:")
    for model, metrics in executive_summary['betting_performance'].items():
        print(f"\n{model.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
    print("\nKEY FINDINGS:")
    for finding in executive_summary['key_findings']:
        print(f"  - {finding}")
    
    print("\nRECOMMENDATIONS:")
    for rec in executive_summary['recommendations']:
        print(f"  - {rec}")
    
    # Monte Carlo results
    print("\n" + "="*80)
    print("MONTE CARLO SIMULATION RESULTS (180-day projection)")
    print("="*80)
    
    for model, mc_results in advanced_results['monte_carlo'].items():
        print(f"\n{model.upper()} Model:")
        print(f"  Expected Return: {mc_results['expected_return']*100:.2f}%")
        print(f"  Return Std Dev: {mc_results['return_std']*100:.2f}%")
        print(f"  Probability of Profit: {mc_results['probability_profit']*100:.1f}%")
        print(f"  Probability of Doubling: {mc_results['probability_double']*100:.1f}%")
        print(f"  Expected Max Drawdown: {mc_results['expected_max_drawdown']*100:.1f}%")
        print(f"  95% VaR: {mc_results['var_95']*100:.2f}%")
        print(f"  95% CVaR: {mc_results['cvar_95']*100:.2f}%")
        print("\n  Return Percentiles:")
        for pct, value in mc_results['return_percentiles'].items():
            print(f"    {pct}: {value*100:.2f}%")
    
    # Export to Excel if requested
    if export_excel:
        print("\n8. EXPORTING RESULTS TO EXCEL...")
        backtester.export_results_to_excel()
        
    # Create live tracking templates
    print("\n9. CREATING LIVE TRACKING TEMPLATES...")
    backtester.create_live_tracking_template()
    
    print("\n" + "="*80)
    print("COMPREHENSIVE BACKTEST COMPLETE!")
    print("="*80)
    print("\nCheck the 'backtest_results' folder for:")
    print("  - Detailed reports and visualizations")
    print("  - Excel export with all data")
    print("  - Live tracking templates")
    print("  - Risk analysis charts")
    print("  - Model degradation analysis")
    
    # Return comprehensive results
    return {
        'backtester': backtester,
        'performance_metrics': performance_metrics,
        'betting_results': betting_results,
        'advanced_results': advanced_results,
        'executive_summary': executive_summary
    }


# Specialized analysis functions
def analyze_specific_threshold(backtester: MLBBacktester, model_type: str, threshold: float):
    """Deep dive analysis for a specific threshold"""
    if model_type == 'hits':
        predictions = np.array(backtester.predictions['hits'])
        actuals = np.array(backtester.actuals['hits'])
        
        pred_over = predictions > threshold
        actual_over = actuals > threshold
        
        # Confusion matrix
        tp = np.sum((pred_over == True) & (actual_over == True))
        tn = np.sum((pred_over == False) & (actual_over == False))
        fp = np.sum((pred_over == True) & (actual_over == False))
        fn = np.sum((pred_over == False) & (actual_over == True))
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate expected value at typical odds
        over_odds = 1.91  # -110
        under_odds = 1.91  # -110
        
        # Expected value calculations
        ev_over = (precision * (over_odds - 1)) - ((1 - precision) * 1)
        ev_under = (((tn / (tn + fn)) if (tn + fn) > 0 else 0) * (under_odds - 1)) - \
                   (((fn / (tn + fn)) if (tn + fn) > 0 else 0) * 1)
        
        return {
            'threshold': threshold,
            'confusion_matrix': {
                'true_positive': tp,
                'true_negative': tn,
                'false_positive': fp,
                'false_negative': fn
            },
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            },
            'betting_analysis': {
                'over_ev': ev_over,
                'under_ev': ev_under,
                'recommended_bet': 'over' if ev_over > ev_under and ev_over > 0 else 
                                  'under' if ev_under > 0 else 'no bet'
            },
            'sample_size': len(predictions)
        }
    
    # Similar analysis for strikeouts...
    return {}


def optimize_betting_parameters(backtester: MLBBacktester):
    """Find optimal betting parameters through grid search"""
    param_grid = {
        'kelly_fraction': [0.1, 0.15, 0.2, 0.25, 0.3],
        'min_edge': [0.03, 0.05, 0.07, 0.10],
        'max_bet_pct': [0.03, 0.05, 0.07, 0.10]
    }
    
    results = []
    
    for kf in param_grid['kelly_fraction']:
        for me in param_grid['min_edge']:
            for mb in param_grid['max_bet_pct']:
                # Update parameters
                backtester.betting_params.update({
                    'kelly_fraction': kf,
                    'min_edge': me,
                    'max_bet_pct': mb
                })
                
                # Re-run betting simulation
                backtester.betting_results = {}
                backtester._run_betting_simulations()
                
                # Calculate total results
                total_profit = sum(
                    r.get('summary', {}).get('total_profit', 0)
                    for r in backtester.betting_results.values()
                )
                
                total_roi = sum(
                    r.get('summary', {}).get('roi', 0)
                    for r in backtester.betting_results.values()
                )
                
                max_dd = min(
                    r.get('summary', {}).get('max_drawdown', 0)
                    for r in backtester.betting_results.values()
                    if 'summary' in r
                ) if backtester.betting_results else 0
                
                results.append({
                    'kelly_fraction': kf,
                    'min_edge': me,
                    'max_bet_pct': mb,
                    'total_profit': total_profit,
                    'total_roi': total_roi,
                    'max_drawdown': max_dd,
                    'sharpe': total_roi / (abs(max_dd) + 0.01)  # Simplified Sharpe proxy
                })
    
    # Find optimal parameters
    results_df = pd.DataFrame(results)
    optimal_idx = results_df['sharpe'].idxmax()
    optimal_params = results_df.iloc[optimal_idx]
    
    return {
        'optimal_parameters': optimal_params.to_dict(),
        'all_results': results_df,
        'improvement': {
            'profit': optimal_params['total_profit'] - results_df.iloc[0]['total_profit'],
            'roi': optimal_params['total_roi'] - results_df.iloc[0]['total_roi']
        }
    }


# ===================================================================
# MAIN EXECUTION
# ===================================================================
if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == 'full':
            # Run comprehensive backtest
            results = run_comprehensive_backtest(
                start_date=sys.argv[2] if len(sys.argv) > 2 else '2024-01-01',
                end_date=sys.argv[3] if len(sys.argv) > 3 else None,
                monte_carlo_sims=1000,
                export_excel=True
            )
        elif sys.argv[1] == 'quick':
            # Quick backtest without Monte Carlo
            backtester, metrics, betting = run_backtest(
                start_date=sys.argv[2] if len(sys.argv) > 2 else '2024-01-01',
                end_date=sys.argv[3] if len(sys.argv) > 3 else None
            )
        elif sys.argv[1] == 'optimize':
            # Optimize betting parameters
            backtester, _, _ = run_backtest(
                start_date=sys.argv[2] if len(sys.argv) > 2 else '2024-01-01',
                end_date=sys.argv[3] if len(sys.argv) > 3 else None
            )
            print("\nOptimizing betting parameters...")
            optimization_results = optimize_betting_parameters(backtester)
            
            print("\nOptimal Parameters Found:")
            for param, value in optimization_results['optimal_parameters'].items():
                if param in ['kelly_fraction', 'min_edge', 'max_bet_pct']:
                    print(f"  {param}: {value}")
            
            print(f"\nImprovement over baseline:")
            print(f"  Profit increase: ${optimization_results['improvement']['profit']:,.2f}")
            print(f"  ROI increase: {optimization_results['improvement']['roi']*100:.2f}%")
            
            # Export optimization results
            optimization_results['all_results'].to_csv(
                Path('backtest_results') / 'parameter_optimization.csv',
                index=False
            )
        else:
            print("Usage: python mlb_backtesting_system.py [full|quick|optimize] [start_date] [end_date]")
    else:
        # Default: Run full comprehensive backtest
        results = run_comprehensive_backtest()