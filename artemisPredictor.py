#!/usr/bin/env python3
"""
MLB Daily Predictor
This script orchestrates the entire prediction pipeline for a given day:
1. Fetches daily games and calculates up-to-date features using MLBDailyPipeline.
2. Loads all trained regression models from the 'artemis' phase.
3. Predicts scores and win probabilities for each model.
4. Creates an ensemble prediction.
5. Engineers meta-features based on model consensus and market disagreement.
6. Loads the final profitability model from 'artemisAnalyzer'.
7. Generates a final betting recommendation for each game.
"""

import pandas as pd
import numpy as np
import warnings
import joblib
import pickle
from datetime import datetime
import os
import argparse
from typing import Dict, List, Any
import sys
from artemisAnalyzer import EnhancedPredictionFeatureEngineer
import json

# Import the necessary class from your pipeline script
from testingInference import MLBDailyPipeline

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def decimal_to_prob(odds):
    if pd.isna(odds) or odds <= 1:  # Note: analyzer checks <= 1, not == 0
        return 0.5  # analyzer returns 0.5, not np.nan
    return 1 / odds

class DailyPredictor:
    """Orchestrates the daily MLB prediction process."""

    def __init__(self, models_dir: str):
        """
        Initialize the predictor.

        Args:
            models_dir (str): The directory where trained models, scalers,
                              and feature lists are stored.
        """
        self.models_dir = models_dir
        if not os.path.exists(self.models_dir):
            raise FileNotFoundError(f"Models directory not found at: {self.models_dir}")

        # Paths to essential files
        self.scaler_path = os.path.join(self.models_dir, 'feature_scaler.pkl')
        self.betting_model_path = os.path.join(self.models_dir, 'betting_model_enhanced.pkl')
        self.classifier_path = os.path.join(self.models_dir, 'win_probability_classifier.pkl')

        # Load essential components
        self.scaler = self._load_model(self.scaler_path)
        self.betting_model_bundle = self._load_model(self.betting_model_path)
        self.output_dir = r"C:\Users\DMelv\Documents\bettingModelBaseball\apiBaseball\pipeline\artemis\json"

        print("Daily Predictor initialized successfully.")

    # ################# START: REPLACEMENT _load_model FUNCTION #################
    def _load_model(self, path: str) -> Any:
        """Helper to load a joblib or pickle file with robust error handling."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cannot find model file at the specified path: {path}")

        with open(path, 'rb') as f:
            try:
                # Prioritize joblib as it's used for saving sklearn models
                return joblib.load(f)
            except Exception as e_joblib:
                # If joblib fails, reset file pointer and try pickle as a fallback
                f.seek(0)
                try:
                    return pickle.load(f)
                except Exception as e_pickle:
                    # If both fail, raise an informative error
                    raise IOError(
                        f"Could not load file '{os.path.basename(path)}'. "
                        f"It might be corrupted or in an unknown format.\n"
                        f"  - Joblib Error: {e_joblib}\n"
                        f"  - Pickle Error: {e_pickle}"
                    )


    # In artemisPredictor.py, replace this function
    def save_predictions_to_json(self, final_df: pd.DataFrame, target_date: str):
        """Save predictions to JSON for the Streamlit dashboard."""
        print("\n[Saving to JSON] Creating JSON output for dashboard...")
        
        # Prepare the predictions data
        predictions = []
        
        for _, game in final_df.iterrows():
            # Calculate expected value for this game
            if game['bet_signal'] == 1:
                if game['ensemble_value_score'] > 0:
                    win_prob = game['ensemble_home_win_prob']
                    odds = game['home_ml']
                    team_bet = game['home_team_abbr']
                else:
                    win_prob = 1 - game['ensemble_home_win_prob']
                    odds = game['away_ml']
                    team_bet = game['away_team_abbr']
                
                ev_pct = ((win_prob * odds) - 1) * 100
                kelly_pct = game['kelly_bet_fraction'] * 100
                
                # Determine bet strength
                if kelly_pct >= 4.0:
                    bet_strength = "STRONG"
                elif kelly_pct >= 2.5:
                    bet_strength = "MEDIUM"
                else:
                    bet_strength = "SMALL"
                
                predictions.append({
                    'game_id': str(game['game_pk']),
                    'matchup': f"{game['away_team_abbr']} @ {game['home_team_abbr']}",
                    'bet_team': team_bet,
                    'bet_type': 'ML',  # Moneyline
                    'odds': round(odds, 2),
                    'stake_pct': round(kelly_pct, 2),
                    'bet_strength': bet_strength,
                    'model_win_prob': round(win_prob * 100, 1),
                    'value_edge': round(abs(game['ensemble_value_score']) * 100, 1),
                    'expected_value': round(ev_pct, 1),
                    'profitability_score': round(game['profitability_score'] * 100, 1),
                    'home_score_pred': round(game['ensemble_pred_home'], 1),
                    'away_score_pred': round(game['ensemble_pred_away'], 1),
                    'total_pred': round(game['ensemble_pred_home'] + game['ensemble_pred_away'], 1),
                    'model_consensus': round(game['model_win_consensus_pct'] * 100, 0),
                    'high_variance_flag': bool(game.get('high_variance_flag', 0)),
                    'pitchers': {
                        'home': game.get('home_pitcher_name', 'TBD'),
                        'away': game.get('away_pitcher_name', 'TBD')
                    }
                })
        
        # Create summary statistics
        total_bets = len(predictions)
        total_stake = sum(p['stake_pct'] for p in predictions)
        avg_stake = total_stake / total_bets if total_bets > 0 else 0
        total_ev = sum(p['expected_value'] * p['stake_pct'] / 100 for p in predictions)
        
        # Count by strength
        strong_bets = len([p for p in predictions if p['bet_strength'] == 'STRONG'])
        medium_bets = len([p for p in predictions if p['bet_strength'] == 'MEDIUM'])
        small_bets = len([p for p in predictions if p['bet_strength'] == 'SMALL'])
        
        # Create the output JSON
        output = {
            'date': target_date,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_games': len(final_df),
                'total_bets': total_bets,
                'total_stake_pct': round(total_stake, 1),
                'avg_stake_pct': round(avg_stake, 1),
                'expected_return_pct': round(total_ev, 1),
                'bets_by_strength': {
                    'strong': strong_bets,
                    'medium': medium_bets,
                    'small': small_bets
                }
            },
            'predictions': predictions,
            'model_info': {
                'threshold': round(self.betting_model_bundle['threshold'], 3),
                'models_used': len([col for col in final_df.columns if col.endswith('_pred_home') and 'ensemble' not in col])
            }
        }
        
        # Save to file
        output_path = os.path.join(self.output_dir, 'daily_predictions.json')
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"  Saved predictions to: {output_path}")
        print(f"  Total bets saved: {total_bets}")
        
        # Also save a detailed version for debugging
        detailed_output_path = os.path.join(self.output_dir, f'detailed_predictions_{target_date}.json')
        detailed_data = final_df.to_dict(orient='records')
        
        # Convert numpy types to Python types for JSON serialization
        for record in detailed_data:
            for key, value in record.items():
                if isinstance(value, (np.integer, np.int64)):
                    record[key] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    record[key] = float(value)
                elif isinstance(value, np.ndarray):
                    record[key] = value.tolist()
                elif pd.isna(value):
                    record[key] = None
        
        with open(detailed_output_path, 'w') as f:
            json.dump({
                'date': target_date,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'games': detailed_data
            }, f, indent=2)
        
        print(f"  Saved detailed predictions to: {detailed_output_path}")

    def get_features_for_date(self, target_date: str) -> pd.DataFrame:
        """
        Step 1: Get the features for all games on a specific date.
        """
        print(f"\n[Step 1/5] Fetching features for {target_date}...")
        try:
            pipeline = MLBDailyPipeline()
            features_df = pipeline.process_todays_games(target_date)
            
            if features_df is None or features_df.empty:
                print("No games found or processed for this date.")
                return None
            
            print(f"Successfully generated features for {len(features_df)} games.")

            # Check the content of the categorical columns
            print("\nContent of key categorical columns (first 5 rows):")
            categorical_check_cols = ['venue', 'dayNight', 'conditions', 'wind_dir', 'game_time']
            existing_check_cols = [col for col in categorical_check_cols if col in features_df.columns]
            if existing_check_cols:
                print(features_df[existing_check_cols].head().to_string())
            else:
                print("ERROR: None of the key categorical columns were found in the generated data.")
            print("="*68 + "\n")
            # --- END: NEW DEBUGGING BLOCK ---
            
            return features_df
        except Exception as e:
            print(f"An error occurred during feature generation: {e}")
            import traceback
            traceback.print_exc()
            return None

    # In artemisPredictor.py, replace this entire function

    # Add this debugging version of generate_level_one_predictions to artemisPredictor.py

    def generate_level_one_predictions(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Load all regression models and generate score predictions.
        Robust version that handles extreme predictions better.
        """
        print("\n[Step 2/5] Generating score predictions from all models...")
        
        base_cols = ['game_pk', 'game_date', 'home_team_abbr', 'away_team_abbr']
        odds_cols = ['home_ml', 'away_ml', 'total_line']
        predictions_df = features_df[base_cols].copy()
        for col in odds_cols:
            predictions_df[col] = features_df.get(col, np.nan)

        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('_home.pkl')]
        model_names = [f.replace('_home.pkl', '') for f in model_files]
        print(f"Found {len(model_names)} models to use for prediction.")

        # Track models with extreme predictions
        problematic_models = set()
        model_extreme_counts = {}

        for model_name in model_names:
            print(f"  Predicting with {model_name}...")
            extreme_count = 0
            
            try:
                model_home = self._load_model(os.path.join(self.models_dir, f'{model_name}_home.pkl'))
                model_away = self._load_model(os.path.join(self.models_dir, f'{model_name}_away.pkl'))
                feature_list = self._load_model(os.path.join(self.models_dir, f'{model_name}_features.pkl'))

                X = features_df.copy()
                
                # One-hot encoding
                categorical_cols_to_encode = [
                    'venue', 'dayNight', 'wind_dir', 'conditions', 'game_time'
                ]
                existing_categorical_cols = [col for col in categorical_cols_to_encode if col in X.columns]
                
                if existing_categorical_cols:
                    X = pd.get_dummies(X, columns=existing_categorical_cols, dtype=float)

                # Align columns
                for col in feature_list:
                    if col not in X.columns:
                        X[col] = 0
                X = X[feature_list]
                
                # Scale features
                X_scaled = self.scaler.transform(X)
                
                # Make predictions
                pred_home = model_home.predict(X_scaled)
                pred_away = model_away.predict(X_scaled)

                # Check for extreme predictions before clipping
                extreme_mask = (np.abs(pred_home) > 20) | (np.abs(pred_away) > 20)
                extreme_count = np.sum(extreme_mask)
                
                if extreme_count > 0:
                    model_extreme_counts[model_name] = extreme_count
                    # If more than 20% of predictions are extreme, mark model as problematic
                    if extreme_count / len(pred_home) > 0.2:
                        problematic_models.add(model_name)
                        print(f"    WARNING: {extreme_count} extreme predictions ({extreme_count/len(pred_home)*100:.1f}% of games)")

                # Clip predictions to reasonable range
                pred_home_clipped = np.clip(pred_home, 0, 20)
                pred_away_clipped = np.clip(pred_away, 0, 20)
                
                predictions_df[f'{model_name}_pred_home'] = pred_home_clipped
                predictions_df[f'{model_name}_pred_away'] = pred_away_clipped
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"    Failed to predict with {model_name}: {e}")
                continue
        
        # Print summary
        if model_extreme_counts:
            print("\n  Models with extreme predictions:")
            for model, count in sorted(model_extreme_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"    {model}: {count} games")
        
        # Calculate robust ensemble
        home_cols = [col for col in predictions_df.columns if col.endswith('_pred_home')]
        away_cols = [col for col in predictions_df.columns if col.endswith('_pred_away')]
        
        # Exclude problematic models from ensemble
        if problematic_models:
            print(f"\n  Excluding problematic models from ensemble: {problematic_models}")
            home_cols = [col for col in home_cols if not any(prob in col for prob in problematic_models)]
            away_cols = [col for col in away_cols if not any(prob in col for prob in problematic_models)]
        
        if not home_cols:
            raise RuntimeError("No model predictions were successfully generated.")
        
        # Use median for robustness (less sensitive to outliers than mean)
        predictions_df['ensemble_pred_home'] = predictions_df[home_cols].median(axis=1)
        predictions_df['ensemble_pred_away'] = predictions_df[away_cols].median(axis=1)
        
        # Also calculate mean for comparison
        predictions_df['ensemble_mean_home'] = predictions_df[home_cols].mean(axis=1)
        predictions_df['ensemble_mean_away'] = predictions_df[away_cols].mean(axis=1)
        
        # Calculate standard deviation to identify high-variance predictions
        predictions_df['ensemble_std_home'] = predictions_df[home_cols].std(axis=1)
        predictions_df['ensemble_std_away'] = predictions_df[away_cols].std(axis=1)
        
        # Flag games with high prediction variance
        high_variance_threshold = 2.0  # Standard deviation > 2 runs
        predictions_df['high_variance_flag'] = (
            (predictions_df['ensemble_std_home'] > high_variance_threshold) | 
            (predictions_df['ensemble_std_away'] > high_variance_threshold)
        ).astype(int)
        
        # Calculate win probability
        score_diff = predictions_df['ensemble_pred_home'] - predictions_df['ensemble_pred_away']
        predictions_df['ensemble_home_win_prob'] = 1 / (1 + np.exp(-score_diff / 3.0))

        # Classifier predictions (rest remains the same)
        try:
            classifier = self._load_model(self.classifier_path)
            clf_feature_list_path = self.classifier_path.replace('.pkl', '_features.pkl')
            if os.path.exists(clf_feature_list_path):
                clf_feature_list = self._load_model(clf_feature_list_path)
            else:
                clf_feature_list = self._load_model(os.path.join(self.models_dir, 'RandomForestRegressor_features.pkl'))
            
            X_clf = features_df.copy()
            if existing_categorical_cols:
                X_clf = pd.get_dummies(X_clf, columns=existing_categorical_cols, dtype=float)

            for col in clf_feature_list:
                if col not in X_clf.columns:
                    X_clf[col] = 0
            X_clf = X_clf[clf_feature_list]
            X_clf_scaled = self.scaler.transform(X_clf)
            
            predictions_df['classifier_home_win_prob'] = classifier.predict_proba(X_clf_scaled)[:, 1]
        except Exception as e:
            print(f"  Error using classifier: {e}")
            predictions_df['classifier_home_win_prob'] = predictions_df['ensemble_home_win_prob']

        # Print games with high variance as a warning
        high_var_games = predictions_df[predictions_df['high_variance_flag'] == 1]
        if len(high_var_games) > 0:
            print("\n  Games with high prediction variance (less reliable):")
            for _, game in high_var_games.iterrows():
                print(f"    {game['away_team_abbr']} @ {game['home_team_abbr']} - "
                    f"Home std: {game['ensemble_std_home']:.2f}, Away std: {game['ensemble_std_away']:.2f}")

        return predictions_df


    # Also add a helper function to analyze which features might be causing issues
    def analyze_problem_games(self, features_df: pd.DataFrame, problem_games: List[str]):
        """
        Analyze features for games with extreme predictions
        """
        print("\n[ANALYSIS] Investigating games with extreme predictions...")
        
        for game in problem_games:
            # Find the game in features_df
            mask = features_df['home_team_abbr'] + ' vs ' + features_df['away_team_abbr'] == game.replace(' @ ', ' vs ')
            if not mask.any():
                continue
            
            game_features = features_df[mask].iloc[0]
            print(f"\n  Game: {game}")
            
            # Check for unusual feature values
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            
            # Compare this game's features to the mean/std of all games
            unusual_features = []
            for col in numeric_cols:
                if col in ['game_pk', 'game_hour', 'home_team_id', 'away_team_id']:
                    continue
                
                all_values = features_df[col]
                mean_val = all_values.mean()
                std_val = all_values.std()
                game_val = game_features[col]
                
                if std_val > 0:
                    z_score = abs((game_val - mean_val) / std_val)
                    if z_score > 3:  # More than 3 standard deviations
                        unusual_features.append({
                            'feature': col,
                            'value': game_val,
                            'mean': mean_val,
                            'z_score': z_score
                        })
            
            if unusual_features:
                print("    Unusual feature values (>3 std devs from mean):")
                # Sort by z-score and show top 10
                unusual_features.sort(key=lambda x: x['z_score'], reverse=True)
                for feat in unusual_features[:10]:
                    print(f"      {feat['feature']}: {feat['value']:.2f} (mean: {feat['mean']:.2f}, z-score: {feat['z_score']:.1f})")
            else:
                print("    No obviously unusual feature values found")

    def engineer_meta_features(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer ALL meta-features that match the artemisAnalyzer training process exactly
        """
        print("\n[Step 3/5] Engineering meta-features from predictions...")
        df = predictions_df.copy()
        
        # Load optimal thresholds from analyzer
        try:
            import json
            thresholds_path = os.path.join(self.models_dir, 'optimal_thresholds.json')
            with open(thresholds_path, 'r') as f:
                optimal_thresholds = json.load(f)
            value_threshold = optimal_thresholds['value']
            certainty_threshold = optimal_thresholds['certainty']
            print(f"  Loaded optimal thresholds: value={value_threshold:.3f}, certainty={certainty_threshold:.3f}")
        except:
            # Fallback to defaults if file not found
            value_threshold = 0.05
            certainty_threshold = 0.15
            print(f"  Using default thresholds: value={value_threshold:.3f}, certainty={certainty_threshold:.3f}")
        
        # Get model names from columns
        pred_home_cols = [col for col in df.columns if col.endswith('_pred_home') and 'ensemble' not in col]
        pred_away_cols = [col for col in df.columns if col.endswith('_pred_away') and 'ensemble' not in col]
        model_names = [col.replace('_pred_home', '') for col in pred_home_cols]
        
        # Calculate win probabilities for each model
        win_prob_cols = []
        win_pred_cols = []
        for model_name in model_names:
            diff = df[f"{model_name}_pred_home"] - df[f"{model_name}_pred_away"]
            df[f"{model_name}_home_win_prob"] = 1 / (1 + np.exp(-diff / 3.0))
            df[f"{model_name}_pred_home_win"] = (df[f"{model_name}_home_win_prob"] > 0.5).astype(int)
            win_prob_cols.append(f"{model_name}_home_win_prob")
            win_pred_cols.append(f"{model_name}_pred_home_win")
        
        # === CONSENSUS FEATURES ===
        df['home_pred_std'] = df[pred_home_cols].std(axis=1)
        df['away_pred_std'] = df[pred_away_cols].std(axis=1)
        df['win_prob_std'] = df[win_prob_cols].std(axis=1)
        
        df['home_pred_iqr'] = df[pred_home_cols].quantile(0.75, axis=1) - df[pred_home_cols].quantile(0.25, axis=1)
        df['win_prob_iqr'] = df[win_prob_cols].quantile(0.75, axis=1) - df[win_prob_cols].quantile(0.25, axis=1)
        
        home_mean = df[pred_home_cols].mean(axis=1)
        away_mean = df[pred_away_cols].mean(axis=1)
        df['home_pred_cv'] = df['home_pred_std'] / (home_mean + 0.001)
        df['away_pred_cv'] = df['away_pred_std'] / (away_mean + 0.001)
        
        df['model_win_consensus_pct'] = df[win_pred_cols].mean(axis=1)
        
        def calculate_entropy(probs):
            probs = np.clip(probs, 1e-10, 1-1e-10)
            return -np.mean(probs * np.log(probs) + (1-probs) * np.log(1-probs))
        
        df['win_prob_entropy'] = df[win_prob_cols].apply(
            lambda row: calculate_entropy(row.values), axis=1
        )
        
        # === MARKET DISAGREEMENT FEATURES ===
        df['home_market_prob'] = df['home_ml'].apply(decimal_to_prob)
        df['away_market_prob'] = df['away_ml'].apply(decimal_to_prob)
        
        df['ensemble_value_score'] = df['ensemble_home_win_prob'] - df['home_market_prob']
        df['ensemble_value_abs'] = abs(df['ensemble_value_score'])
        
        model_values = pd.DataFrame()
        for col in win_prob_cols:
            model_values[col] = df[col] - df['home_market_prob']
        
        df['max_model_value'] = model_values.max(axis=1)
        df['min_model_value'] = model_values.min(axis=1)
        df['value_score_range'] = df['max_model_value'] - df['min_model_value']
        
        df['ensemble_spread'] = df['ensemble_pred_home'] - df['ensemble_pred_away']
        df['ensemble_total'] = df['ensemble_pred_home'] + df['ensemble_pred_away']
        
        if 'total_line' in df.columns:
            df['total_diff_from_market'] = df['ensemble_total'] - df['total_line']
            df['total_diff_pct'] = df['total_diff_from_market'] / (df['total_line'] + 0.001)
        else:
            df['total_diff_from_market'] = 0
            df['total_diff_pct'] = 0
        
        # === CONFIDENCE FEATURES ===
        df['ensemble_pred_magnitude'] = abs(df['ensemble_pred_home'] - df['ensemble_pred_away'])
        df['ensemble_prob_certainty'] = abs(df['ensemble_home_win_prob'] - 0.5)
        
        if 'classifier_home_win_prob' in df.columns:
            df['classifier_prob_certainty'] = abs(df['classifier_home_win_prob'] - 0.5)
            df['ensemble_classifier_agreement'] = 1 - abs(df['ensemble_home_win_prob'] - df['classifier_home_win_prob'])
        else:
            df['classifier_prob_certainty'] = df['ensemble_prob_certainty']
            df['ensemble_classifier_agreement'] = 1.0
        
        df['confidence_component_1'] = 1 - df['win_prob_std']
        df['confidence_component_2'] = df['ensemble_prob_certainty']
        df['confidence_component_3'] = 1 - df['home_pred_cv']
        df['confidence_component_4'] = df['ensemble_classifier_agreement']
        
        # === INTERACTION FEATURES ===
        df['value_uncertainty_interaction'] = df['ensemble_value_abs'] * (1 - df['win_prob_std'])
        df['magnitude_certainty_interaction'] = df['ensemble_pred_magnitude'] * df['ensemble_prob_certainty']
        df['market_model_interaction'] = df['ensemble_value_abs'] * df['model_win_consensus_pct']
        
        # === KELLY CRITERION FEATURES ===
        def kelly_fraction(win_prob, decimal_odds, kelly_multiplier=0.25):
            if decimal_odds <= 1 or win_prob <= 0 or win_prob >= 1:
                return 0
            
            b = decimal_odds - 1
            p = win_prob
            q = 1 - p
            f_full = (p * b - q) / b
            f_safe = f_full * kelly_multiplier
            return min(max(0, f_safe), 0.05)

        df['home_kelly_fraction'] = df.apply(
            lambda row: kelly_fraction(row['ensemble_home_win_prob'], row['home_ml']),
            axis=1
        )

        df['away_kelly_fraction'] = df.apply(
            lambda row: kelly_fraction(1 - row['ensemble_home_win_prob'], row['away_ml']),
            axis=1
        )

        df['kelly_bet_side'] = np.where(
            df['ensemble_value_score'] > 0, 'home',
            np.where(df['ensemble_value_score'] < 0, 'away', 'none')
        )

        df['kelly_bet_fraction'] = np.where(
            df['kelly_bet_side'] == 'home',
            df['home_kelly_fraction'],
            np.where(
                df['kelly_bet_side'] == 'away',
                df['away_kelly_fraction'],
                0
            )
        )

        df['kelly_bet_fraction'] = np.where(
            df['ensemble_value_abs'] > 0.02,
            df['kelly_bet_fraction'],
            0
        )

        # === MULTI-FACTOR BETTING RULES === (ONLY ONCE!)
        # Conservative: High agreement + High value + High certainty + Low variance
        df['conservative_bet_flag'] = (
            (df['ensemble_value_abs'] > value_threshold) & 
            (df['ensemble_prob_certainty'] > certainty_threshold) &
            (df['win_prob_std'] < df['win_prob_std'].quantile(0.25)) &
            (df['model_win_consensus_pct'] > 0.75)
        ).astype(int)

        # Aggressive: Good value with lower requirements
        df['aggressive_bet_flag'] = (
            (df['ensemble_value_abs'] > value_threshold * 0.6) & 
            (df['value_uncertainty_interaction'] > df['value_uncertainty_interaction'].quantile(0.8))
        ).astype(int)

        df['suggested_bet_size'] = np.where(
            df['conservative_bet_flag'] == 1, 'full',
            np.where(df['aggressive_bet_flag'] == 1, 'half', 'none')
        )
        
        # === CATEGORICAL FEATURES ===
        # Use quantile-based binning for categories
        value_quantiles = [0, 0.25, 0.5, 0.75, 1.0]
        df['value_category'] = pd.cut(
            df['ensemble_value_abs'],
            bins=df['ensemble_value_abs'].quantile(value_quantiles).values,
            labels=['minimal', 'small', 'medium', 'large'],
            include_lowest=True,
            duplicates='drop'
        )
        
        df['certainty_category'] = pd.cut(
            df['ensemble_prob_certainty'],
            bins=df['ensemble_prob_certainty'].quantile(value_quantiles).values,
            labels=['very_low', 'low', 'medium', 'high'],
            include_lowest=True,
            duplicates='drop'
        )
        
        df['consensus_category'] = pd.cut(
            df['win_prob_std'],
            bins=df['win_prob_std'].quantile(value_quantiles).values,
            labels=['high', 'medium', 'low', 'very_low'],  # Reversed because low std = high consensus
            include_lowest=True,
            duplicates='drop'
        )
        
        df['strong_bet_flag'] = (
            (df['ensemble_value_abs'] > value_threshold) & 
            (df['ensemble_prob_certainty'] > certainty_threshold)
        ).astype(int)
        
        df['medium_bet_flag'] = (
            (df['ensemble_value_abs'] > value_threshold * 0.7) & 
            (df['ensemble_prob_certainty'] > certainty_threshold * 0.7) &
            (df['strong_bet_flag'] == 0)
        ).astype(int)
        
        df['bet_direction'] = np.where(
            df['ensemble_value_score'] > 0, 'home',
            np.where(df['ensemble_value_score'] < 0, 'away', 'none')
        )
        
        # Convert categorical features to dummy variables (one-hot encoding)
        categorical_cols = ['value_category', 'certainty_category', 'consensus_category', 
                        'bet_direction', 'suggested_bet_size']
        
        for col in categorical_cols:
            if col in df.columns:
                # Get dummies and drop first to avoid multicollinearity
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])  # Drop the original categorical column
        
        print("Meta-features engineered successfully.")
        print(f"  Total features created: {len([col for col in df.columns if col not in predictions_df.columns])}")
        
        # List all numeric features created (for debugging)
        numeric_features = df.select_dtypes(include=[np.number]).columns
        new_features = [col for col in numeric_features if col not in predictions_df.columns]
        print(f"  New numeric features: {len(new_features)}")
        
        return df

    def generate_final_recommendation(self, meta_features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Use the Level 2 model to predict profitability and make a recommendation.
        Enhanced with diagnostics to understand model behavior.
        """
        print("\n[Step 4/5] Generating final betting recommendations...")
        df = meta_features_df.copy()

        # Extract model, features, and threshold from the bundle
        model = self.betting_model_bundle['model']
        feature_list = self.betting_model_bundle['features']
        threshold = self.betting_model_bundle['threshold']
        
        print(f"\n  Model expects {len(feature_list)} features")
        print(f"  Threshold from training: {threshold:.3f}")
        
        # Show first few expected features for debugging
        print(f"  First 10 expected features: {feature_list[:10]}")
        
        # Check which features are missing
        missing_features = [f for f in feature_list if f not in df.columns]
        if missing_features:
            print(f"\n  WARNING: Missing {len(missing_features)} features!")
            print(f"  Missing features: {missing_features[:10]}...")  # Show first 10
        
        # Prepare data for the betting model
        X_betting = df.copy()
        for col in feature_list:
            if col not in X_betting.columns:
                print(f"  Warning: Feature '{col}' not found for betting model. Filling with 0.")
                X_betting[col] = 0
        X_betting = X_betting[feature_list]

        # Check for NaN values
        nan_counts = X_betting.isna().sum()
        if nan_counts.any():
            print(f"\n  WARNING: Found NaN values in {nan_counts[nan_counts > 0].count()} features")
            # Fill NaN with 0 for now
            X_betting = X_betting.fillna(0)

        # Predict profitability score
        df['profitability_score'] = model.predict_proba(X_betting)[:, 1]
        
        # Show distribution of profitability scores
        print(f"\n  Profitability score distribution:")
        print(f"    Min: {df['profitability_score'].min():.3f}")
        print(f"    25%: {df['profitability_score'].quantile(0.25):.3f}")
        print(f"    50%: {df['profitability_score'].quantile(0.50):.3f}")
        print(f"    75%: {df['profitability_score'].quantile(0.75):.3f}")
        print(f"    Max: {df['profitability_score'].max():.3f}")
        
        # Show games near the threshold
        near_threshold = df[
            (df['profitability_score'] > threshold - 0.1) & 
            (df['profitability_score'] < threshold + 0.1)
        ]
        print(f"\n  Games near threshold ({threshold:.3f} ± 0.1): {len(near_threshold)}")
        if len(near_threshold) > 0:
            print("  Examples:")
            for _, game in near_threshold.head(3).iterrows():
                print(f"    {game['away_team_abbr']} @ {game['home_team_abbr']}: "
                    f"score={game['profitability_score']:.3f}, "
                    f"value={game['ensemble_value_abs']:.3f}")
        
        # Original logic with additional diagnostics
        df['bet_signal'] = (
            (df['profitability_score'] > threshold) &
            (df['ensemble_value_abs'] > 0.02)  # Basic value check
        ).astype(int)
        
        # Also create alternative thresholds for comparison
        df['bet_signal_low'] = (
            (df['profitability_score'] > threshold * 0.8) &  # 80% of threshold
            (df['ensemble_value_abs'] > 0.02)
        ).astype(int)
        
        df['bet_signal_medium'] = (
            (df['profitability_score'] > threshold * 0.9) &  # 90% of threshold
            (df['ensemble_value_abs'] > 0.02)
        ).astype(int)
        
        print(f"\n  Betting signals:")
        print(f"    Original threshold ({threshold:.3f}): {df['bet_signal'].sum()} bets")
        print(f"    90% threshold ({threshold*0.9:.3f}): {df['bet_signal_medium'].sum()} bets")
        print(f"    80% threshold ({threshold*0.8:.3f}): {df['bet_signal_low'].sum()} bets")
        
        # Show why some high-value games aren't getting bet
        high_value_no_bet = df[
            (df['ensemble_value_abs'] > 0.05) &  # >5% value
            (df['bet_signal'] == 0)
        ]
        if len(high_value_no_bet) > 0:
            print(f"\n  High value games NOT bet ({len(high_value_no_bet)} games):")
            for _, game in high_value_no_bet.iterrows():
                print(f"    {game['away_team_abbr']} @ {game['home_team_abbr']}: "
                    f"value={game['ensemble_value_abs']:.3f}, "
                    f"prob_score={game['profitability_score']:.3f} < {threshold:.3f}")

        # Determine which side the bet is on
        df['bet_on_team'] = np.where(df['bet_signal'] == 1,
                                    np.where(df['ensemble_value_score'] > 0, df['home_team_abbr'], df['away_team_abbr']),
                                    'No Bet')

        return df

    # In artemisPredictor.py, replace this entire function

    def display_results(self, final_df: pd.DataFrame):
        """
        Step 5: Display the final predictions in a clean, readable format.
        """
        print("\n" + "="*80)
        print(f"======= MLB Predictions for {pd.to_datetime(final_df['game_date'].iloc[0]).strftime('%A, %B %d, %Y')} =======")
        print("="*80)

        if final_df.empty:
            print("No games to display.")
            return

        # Calculate total recommended bets for summary
        total_bets = final_df['bet_signal'].sum()
        if total_bets > 0:
            total_kelly_fraction = final_df[final_df['bet_signal'] == 1]['kelly_bet_fraction'].sum()
            print(f"\nTODAY'S BETTING SUMMARY:")
            print(f"  Total Games: {len(final_df)}")
            print(f"  Recommended Bets: {total_bets}")
            print(f"  Total Bankroll Allocation: {total_kelly_fraction*100:.1f}%")
            print(f"  Average Bet Size: {(total_kelly_fraction/total_bets)*100:.1f}% per bet")
            
            # Add expected value calculation
            bet_games = final_df[final_df['bet_signal'] == 1]
            total_ev = 0
            for _, game in bet_games.iterrows():
                if game['ensemble_value_score'] > 0:
                    win_prob = game['ensemble_home_win_prob']
                    odds = game['home_ml']
                else:
                    win_prob = 1 - game['ensemble_home_win_prob']
                    odds = game['away_ml']
                ev = (win_prob * (odds - 1)) - (1 - win_prob)
                total_ev += ev * game['kelly_bet_fraction']
            
            print(f"  Expected Return: {total_ev*100:.1f}% of bankroll")
            print("="*80)

        for _, game in final_df.iterrows():
            print(f"\nMatchup: {game['away_team_abbr']} @ {game['home_team_abbr']}")
            print("-" * 40)
            
            # Add starting pitchers if available
            if 'home_pitcher_name' in game and pd.notna(game.get('home_pitcher_name')):
                print(f"  Starting Pitchers: {game.get('away_pitcher_name', 'TBD')} vs {game.get('home_pitcher_name', 'TBD')}")
            
            print(f"  Ensemble Prediction: Away {game['ensemble_pred_away']:.2f} - Home {game['ensemble_pred_home']:.2f}")
            print(f"  Home Win Probability: {game['ensemble_home_win_prob']:.1%}")
            
            # Add total runs prediction
            total_runs = game['ensemble_pred_away'] + game['ensemble_pred_home']
            print(f"  Predicted Total Runs: {total_runs:.1f}")
            
            home_odds_valid = pd.notna(game['home_ml']) and game['home_ml'] > 1
            away_odds_valid = pd.notna(game['away_ml']) and game['away_ml'] > 1
            
            if home_odds_valid and away_odds_valid:
                home_implied_prob = 1 / game['home_ml']
                away_implied_prob = 1 / game['away_ml']

                print(f"  Market-Implied Probs: Away {away_implied_prob:.1%} - Home {home_implied_prob:.1%}")
                
                # Add model confidence indicators
                if 'win_prob_std' in game:
                    model_agreement = "High" if game['win_prob_std'] < 0.05 else "Medium" if game['win_prob_std'] < 0.10 else "Low"
                    print(f"  Model Agreement: {model_agreement} (std: {game['win_prob_std']:.3f})")
                
                if pd.notna(game['ensemble_value_score']):
                    value_side = "Home" if game['ensemble_value_score'] > 0 else "Away"
                    print(f"  Model Value Edge: {abs(game['ensemble_value_score']):.1%} on {value_side}")
                    print(f"  Profitability Score: {game['profitability_score']:.1%}")
                    
                    if game['bet_signal'] == 1:
                        # Get the Kelly fraction for this bet
                        kelly_pct = game['kelly_bet_fraction'] * 100
                        
                        # Determine bet sizing category
                        if kelly_pct >= 4.0:
                            size_category = "STRONG"
                        elif kelly_pct >= 2.5:
                            size_category = "MEDIUM"
                        else:
                            size_category = "SMALL"
                        
                        # Calculate expected value
                        if game['ensemble_value_score'] > 0:
                            win_prob = game['ensemble_home_win_prob']
                            odds = game['home_ml']
                        else:
                            win_prob = 1 - game['ensemble_home_win_prob']
                            odds = game['away_ml']
                        
                        ev_pct = ((win_prob * odds) - 1) * 100
                        
                        # Calculate breakeven win rate
                        breakeven = 1 / odds * 100
                        
                        print(f"  RECOMMENDATION: BET on {game['bet_on_team']}")
                        print(f"    - Value Edge: {abs(game['ensemble_value_score']):.1%}")
                        print(f"    - Kelly Bet Size: {kelly_pct:.1f}% of bankroll ({size_category})")
                        print(f"    - Decimal Odds: {odds:.2f}")
                        print(f"    - Expected Value: {ev_pct:.1f}%")
                        print(f"    - Model Win Prob: {win_prob:.1%} (Breakeven: {breakeven:.1%})")
                        
                        # Add warning for high variance games
                        if 'high_variance_flag' in game and game['high_variance_flag'] == 1:
                            print(f"    ⚠️  WARNING: High model variance - less reliable prediction")
                        
                    else:
                        print(f"  RECOMMENDATION: No Bet")
                        
                        # Show why it wasn't bet
                        if abs(game['ensemble_value_score']) > 0.05:
                            print(f"    (Good value {abs(game['ensemble_value_score']):.1%} but low profitability score)")
                            
                else:
                    print("  Could not calculate value score for recommendation.")
            else:
                print("  Market odds are missing or invalid. No betting recommendation possible.")
        
        # Add a summary section at the end
        if total_bets > 0:
            print("\n" + "="*80)
            print("RECOMMENDED BETS SUMMARY:")
            print("="*80)
            
            bet_games = final_df[final_df['bet_signal'] == 1].sort_values('kelly_bet_fraction', ascending=False)
            
            # Group by bet size category
            strong_bets = bet_games[bet_games['kelly_bet_fraction'] >= 0.04]
            medium_bets = bet_games[(bet_games['kelly_bet_fraction'] >= 0.025) & (bet_games['kelly_bet_fraction'] < 0.04)]
            small_bets = bet_games[bet_games['kelly_bet_fraction'] < 0.025]
            
            if len(strong_bets) > 0:
                print("\nSTRONG BETS (4%+ of bankroll):")
                for _, game in strong_bets.iterrows():
                    kelly_pct = game['kelly_bet_fraction'] * 100
                    print(f"  {game['bet_on_team']:>3} vs {game['away_team_abbr'] if game['bet_on_team'] == game['home_team_abbr'] else game['home_team_abbr']:<3}: "
                        f"{kelly_pct:>4.1f}% (Edge: {abs(game['ensemble_value_score']):>4.1%}, "
                        f"Odds: {game['home_ml' if game['bet_on_team'] == game['home_team_abbr'] else 'away_ml']:.2f})")
            
            if len(medium_bets) > 0:
                print("\nMEDIUM BETS (2.5-4% of bankroll):")
                for _, game in medium_bets.iterrows():
                    kelly_pct = game['kelly_bet_fraction'] * 100
                    print(f"  {game['bet_on_team']:>3} vs {game['away_team_abbr'] if game['bet_on_team'] == game['home_team_abbr'] else game['home_team_abbr']:<3}: "
                        f"{kelly_pct:>4.1f}% (Edge: {abs(game['ensemble_value_score']):>4.1%}, "
                        f"Odds: {game['home_ml' if game['bet_on_team'] == game['home_team_abbr'] else 'away_ml']:.2f})")
            
            if len(small_bets) > 0:
                print("\nSMALL BETS (<2.5% of bankroll):")
                for _, game in small_bets.iterrows():
                    kelly_pct = game['kelly_bet_fraction'] * 100
                    print(f"  {game['bet_on_team']:>3} vs {game['away_team_abbr'] if game['bet_on_team'] == game['home_team_abbr'] else game['home_team_abbr']:<3}: "
                        f"{kelly_pct:>4.1f}% (Edge: {abs(game['ensemble_value_score']):>4.1%}, "
                        f"Odds: {game['home_ml' if game['bet_on_team'] == game['home_team_abbr'] else 'away_ml']:.2f})")
            
            print(f"\nTotal Expected Return: +{total_ev*100:.1f}% of bankroll")
            
            # Add risk warnings
            print("\nRISK NOTES:")
            high_var_bets = bet_games[bet_games.get('high_variance_flag', 0) == 1]
            if len(high_var_bets) > 0:
                print(f"  ⚠️  {len(high_var_bets)} bet(s) have high model variance")
            
            large_allocation = total_kelly_fraction > 0.15
            if large_allocation:
                print(f"  ⚠️  Large total allocation ({total_kelly_fraction*100:.1f}%) - consider scaling down")
        
        print("\n" + "="*80)

    def run(self, target_date: str):
        """
        Execute the full daily prediction pipeline.
        
        Args:
            target_date (str): The date to predict in 'YYYY-MM-DD' format.
        """
        # Step 1: Get features
        features_df = self.get_features_for_date(target_date)
        if features_df is None:
            return

        # Step 2: Generate Level 1 predictions
        predictions_df = self.generate_level_one_predictions(features_df)

        # Step 3: Engineer meta-features
        meta_features_df = self.engineer_meta_features(predictions_df)
        
        # Step 4: Generate final recommendations
        final_df = self.generate_final_recommendation(meta_features_df)

        # Step 5: Display results
        self.display_results(final_df)
        # NEW Step 6: Save to JSON
        self.save_predictions_to_json(final_df, target_date)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run MLB Daily Predictions.")
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.today().strftime('%Y-%m-%d'),
        help="The target date for predictions in YYYY-MM-DD format. Defaults to today."
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default=r"C:\Users\DMelv\Documents\bettingModelBaseball\apiBaseball\pipeline\artemis\models",
        help="Directory where trained models are stored."
    )
    args = parser.parse_args()

    # --- Run the pipeline ---
    try:
        predictor = DailyPredictor(models_dir=args.models_dir)
        predictor.run(target_date=args.date)
    except Exception as e:
        print(f"\nAn unrecoverable error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()