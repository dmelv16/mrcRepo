#!/usr/bin/env python3
"""
MLB Score Prediction Script
Complete implementation for predicting MLB game scores using multiple machine learning models
"""

import pandas as pd
import numpy as np
import warnings
import joblib
import pickle
from datetime import datetime
import os
from typing import Dict, List, Tuple, Any

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# All regression models
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge,
    Lars, LassoLars, OrthogonalMatchingPursuit, PassiveAggressiveRegressor,
    HuberRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor
)
from sklearn.svm import SVR, NuSVR
from sklearn.neural_network import MLPRegressor

# For win probability
from sklearn.linear_model import LogisticRegression
from scipy.special import expit  # sigmoid function

warnings.filterwarnings('ignore')


class MLBScorePredictor:
    """Main class for MLB score prediction"""
    
    def __init__(self, data_path: str, output_dir: str = 'mlb_models'):
        """
        Initialize the predictor
        
        Args:
            data_path: Path to the parquet file with MLB data
            output_dir: Directory to save models and results
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.create_output_directory()
        
        # Define columns to exclude from features
        self.exclude_cols = [
            'game_pk', 'gamePk', 'game_date', 'home_team', 'away_team',
            'home_team_id', 'away_team_id', 'home_team_abbr', 'away_team_abbr', 'side',
            'home_game_date', 'away_game_date', 'home_W/L', 'away_W/L', 'bookmaker',
            'time_match_key', 'date_match_key', 'home_score', 'away_score', 'match_key', 'diff_score'
        ]
        
        # Initialize model dictionary
        self.models = self.get_all_models()
        
        # Storage for results
        self.results = []
        self.predictions = {}
        
    def create_output_directory(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
    
    def get_all_models(self) -> Dict[str, Any]:
        """Return dictionary of all models to test"""
        return {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            # 'Lasso': Lasso(alpha=1.0, max_iter=2000),
            # 'ElasticNet': ElasticNet(alpha=1.0, max_iter=2000),
            'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=30),
            'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
            'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'ExtraTreesRegressor': ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'AdaBoostRegressor': AdaBoostRegressor(n_estimators=50, random_state=42),
            'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR_linear': SVR(kernel='linear', C=1.0),
            'SVR_poly': SVR(kernel='poly', C=1.0, degree=3),
            'SVR_rbf': SVR(kernel='rbf', C=1.0),
            'NuSVR': NuSVR(kernel='rbf', C=1.0),
            'MLPRegressor': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
            'BayesianRidge': BayesianRidge(),
            # 'Lars': Lars(),
            'LassoLars': LassoLars(alpha=1.0),
            'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit(),
            'PassiveAggressiveRegressor': PassiveAggressiveRegressor(max_iter=1000, random_state=42),
            'HuberRegressor': HuberRegressor(max_iter=1000),
            'HistGradientBoostingRegressor': HistGradientBoostingRegressor(random_state=42)
        }
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load data and prepare features and targets
        
        Returns:
            X_train, X_test, y_train, y_test DataFrames
        """
        print(f"Loading data from {self.data_path}...")
        df = pd.read_parquet(self.data_path)
        print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Identify feature columns
        feature_cols = [col for col in df.columns if col not in self.exclude_cols]
        print(f"Using {len(feature_cols)} feature columns")
        
        # Create features and targets - use .copy() to avoid warnings
        X = df[feature_cols].copy()
        y = df[['home_score', 'away_score']]
        
        # --- START: NEW FIX ---
        # Explicitly find and remove any datetime columns from the feature set
        datetime_cols = X.select_dtypes(include=['datetime64[ns]', 'datetimetz', 'datetime']).columns
        if not datetime_cols.empty:
            print(f"  Removing {len(datetime_cols)} datetime columns from features: {list(datetime_cols)}")
            X = X.drop(columns=datetime_cols)
        # --- END: NEW FIX ---

        print("Handling missing values and encoding categorical features...")
        
        # Identify column types
        numeric_cols = X.select_dtypes(include=np.number).columns
        categorical_cols = X.select_dtypes(include=['category', 'object']).columns
        
        # 1. Handle missing values separately
        # For numeric columns, fill with the mean
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
        
        # For categorical columns, fill with the mode (most frequent value)
        for col in categorical_cols:
            mode_val = X[col].mode()[0]
            X[col] = X[col].fillna(mode_val)

        # 2. One-hot encode categorical features to convert them to numbers
        if not categorical_cols.empty:
            print(f"  One-hot encoding {len(categorical_cols)} categorical columns...")
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=float)
        
        # 3. Handle any infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Align columns after one-hot encoding, in case a split creates different columns
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        train_cols = X_train.columns
        test_cols = X_test.columns
        
        missing_in_test = set(train_cols) - set(test_cols)
        for c in missing_in_test:
            X_test[c] = 0
        
        missing_in_train = set(test_cols) - set(train_cols)
        for c in missing_in_train:
            X_train[c] = 0
            
        X_test = X_test[train_cols] # Ensure order is the same

        print(f"Training set: {len(X_train)} samples with {len(X_train.columns)} features after encoding")
        print(f"Test set: {len(X_test)} samples")
        
        # Store test data info for later use
        self.X_test_info = df.loc[X_test.index, ['game_pk', 'game_date', 'home_team', 'away_team', 'home_ml', 'away_ml', 'total_line', 'over_odds', 'under_odds']]
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """
        Scale features using StandardScaler
        
        Returns:
            Scaled X_train, X_test arrays and the scaler object
        """
        print("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler for future use
        scaler_path = os.path.join(self.output_dir, 'feature_scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}")
        
        return X_train_scaled, X_test_scaled, scaler
    
    def train_single_model(self, model_name: str, model: Any,
                        X_train: pd.DataFrame, X_test: pd.DataFrame, # Changed to DataFrame
                        y_train: pd.DataFrame, y_test: pd.DataFrame,
                        scaler: StandardScaler) -> Dict[str, Any]: # Added scaler argument
        """
        Train a single model, evaluate its performance, and save the model
        along with its specific feature list.
        """
        print(f"\nTraining {model_name}...")
        start_time = datetime.now()

        # --- NEW: Scale data inside the function ---
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # --- END NEW ---

        # Train models for home and away scores
        model_home = model.__class__(**model.get_params())
        model_away = model.__class__(**model.get_params())

        # Fit models on the scaled data
        model_home.fit(X_train_scaled, y_train['home_score'])
        model_away.fit(X_train_scaled, y_train['away_score'])

        # Make predictions on the scaled data
        pred_home = model_home.predict(X_test_scaled)
        pred_away = model_away.predict(X_test_scaled)

        # ... (all your metric calculation code remains the same) ...
        mae_home = mean_absolute_error(y_test['home_score'], pred_home)
        mse_home = mean_squared_error(y_test['home_score'], pred_home)
        r2_home = r2_score(y_test['home_score'], pred_home)
        mae_away = mean_absolute_error(y_test['away_score'], pred_away)
        mse_away = mean_squared_error(y_test['away_score'], pred_away)
        r2_away = r2_score(y_test['away_score'], pred_away)
        mae_combined = (mae_home + mae_away) / 2
        mse_combined = (mse_home + mse_away) / 2
        r2_combined = (r2_home + r2_away) / 2


        # Save models
        model_home_path = os.path.join(self.output_dir, f'{model_name}_home.pkl')
        model_away_path = os.path.join(self.output_dir, f'{model_name}_away.pkl')
        joblib.dump(model_home, model_home_path)
        joblib.dump(model_away, model_away_path)

        # --- NEW: SAVE THE FEATURE LIST ---
        # Get the column names from the original X_train DataFrame
        feature_list = X_train.columns.tolist()
        features_path = os.path.join(self.output_dir, f'{model_name}_features.pkl')
        with open(features_path, 'wb') as f:
            pickle.dump(feature_list, f)
        # --- END NEW ---

        # Training time
        training_time = (datetime.now() - start_time).total_seconds()

        # Store predictions
        self.predictions[model_name] = {
            'pred_home': pred_home,
            'pred_away': pred_away
        }

        results = {
            'model': model_name,
            'mae_home': mae_home,
            'mse_home': mse_home,
            'r2_home': r2_home,
            'mae_away': mae_away,
            'mse_away': mse_away,
            'r2_away': r2_away,
            'mae_combined': mae_combined,
            'mse_combined': mse_combined,
            'r2_combined': r2_combined,
            'training_time_seconds': training_time
        }

        print(f"  MAE (combined): {mae_combined:.3f}")
        print(f"  R² (combined): {r2_combined:.3f}")
        print(f"  Training time: {training_time:.2f} seconds")

        return results
    
    def calculate_win_probabilities(self, pred_home: np.ndarray, pred_away: np.ndarray) -> np.ndarray:
        """
        Calculate win probabilities using sigmoid function on score differential
        
        Returns:
            Array of home team win probabilities
        """
        score_diff = pred_home - pred_away
        # Apply sigmoid with scaling factor
        # Larger differences should lead to more extreme probabilities
        win_prob = expit(score_diff / 3.0)  # Divide by 3 to scale appropriately
        return win_prob
    
    def train_win_probability_classifier(self, X_train: np.ndarray, X_test: np.ndarray,
                                       y_train: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[np.ndarray, Any]:
        """
        Train a classifier for win probability
        
        Returns:
            Win probabilities and the trained classifier
        """
        print("\nTraining win probability classifier...")
        
        # Create binary target (1 if home team wins)
        y_train_binary = (y_train['home_score'] > y_train['away_score']).astype(int)
        y_test_binary = (y_test['home_score'] > y_test['away_score']).astype(int)
        
        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train_binary)
        
        # Get probabilities
        win_probs = clf.predict_proba(X_test)[:, 1]  # Probability of home team winning
        
        # Save classifier
        clf_path = os.path.join(self.output_dir, 'win_probability_classifier.pkl')
        joblib.dump(clf, clf_path)
        
        # Calculate accuracy
        accuracy = clf.score(X_test, y_test_binary)
        print(f"  Win prediction accuracy: {accuracy:.3f}")
        
        return win_probs, clf
    
    def create_final_predictions_dataframe(self, y_test: pd.DataFrame, 
                                         win_probs_classifier: np.ndarray) -> pd.DataFrame:
        """
        Create comprehensive predictions DataFrame
        """
        print("\nCreating final predictions DataFrame...")
        
        # Start with game info
        predictions_df = self.X_test_info.copy()
        predictions_df['actual_home_score'] = y_test['home_score'].values
        predictions_df['actual_away_score'] = y_test['away_score'].values
        predictions_df['actual_home_win'] = (y_test['home_score'] > y_test['away_score']).astype(int).values
        
        # Add predictions from each model
        for model_name, preds in self.predictions.items():
            predictions_df[f'{model_name}_pred_home'] = preds['pred_home']
            predictions_df[f'{model_name}_pred_away'] = preds['pred_away']
            predictions_df[f'{model_name}_pred_home_win'] = (preds['pred_home'] > preds['pred_away']).astype(int)
            
            # Calculate win probability using sigmoid
            win_probs = self.calculate_win_probabilities(preds['pred_home'], preds['pred_away'])
            predictions_df[f'{model_name}_home_win_prob'] = win_probs
        
        # Add classifier-based win probability
        predictions_df['classifier_home_win_prob'] = win_probs_classifier
        
        # Calculate ensemble predictions (average of all models)
        home_cols = [col for col in predictions_df.columns if col.endswith('_pred_home')]
        away_cols = [col for col in predictions_df.columns if col.endswith('_pred_away')]
        prob_cols = [col for col in predictions_df.columns if col.endswith('_home_win_prob') and 'classifier' not in col]
        
        predictions_df['ensemble_pred_home'] = predictions_df[home_cols].mean(axis=1)
        predictions_df['ensemble_pred_away'] = predictions_df[away_cols].mean(axis=1)
        predictions_df['ensemble_home_win_prob'] = predictions_df[prob_cols].mean(axis=1)
        predictions_df['ensemble_pred_home_win'] = (predictions_df['ensemble_pred_home'] > predictions_df['ensemble_pred_away']).astype(int)
        
        return predictions_df
    
    def save_results(self, results_df: pd.DataFrame, predictions_df: pd.DataFrame):
        """Save all results to files"""
        # Save model performance results
        results_path = os.path.join(self.output_dir, 'model_performance_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\nSaved model performance results to {results_path}")
        
        # Save predictions
        predictions_path = os.path.join(self.output_dir, 'game_predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Saved game predictions to {predictions_path}")
        
        # Also save as parquet for easier loading
        predictions_parquet_path = os.path.join(self.output_dir, 'game_predictions.parquet')
        predictions_df.to_parquet(predictions_parquet_path, index=False)
        print(f"Saved game predictions (parquet) to {predictions_parquet_path}")
    
    def print_summary(self, results_df: pd.DataFrame, predictions_df: pd.DataFrame):
        """Print summary of results"""
        print("\n" + "="*60)
        print("SUMMARY OF RESULTS")
        print("="*60)
        
        # Best models by different metrics
        print("\nBest Models by Metric:")
        print(f"  Lowest MAE: {results_df.loc[results_df['mae_combined'].idxmin(), 'model']} ({results_df['mae_combined'].min():.3f})")
        print(f"  Highest R²: {results_df.loc[results_df['r2_combined'].idxmax(), 'model']} ({results_df['r2_combined'].max():.3f})")
        
        # Top 5 models by MAE
        print("\nTop 5 Models by MAE (Lower is better):")
        top_5 = results_df.nsmallest(5, 'mae_combined')[['model', 'mae_combined', 'r2_combined']]
        for idx, row in top_5.iterrows():
            print(f"  {row['model']}: MAE={row['mae_combined']:.3f}, R²={row['r2_combined']:.3f}")
        
        # Ensemble performance
        ensemble_mae_home = mean_absolute_error(predictions_df['actual_home_score'], predictions_df['ensemble_pred_home'])
        ensemble_mae_away = mean_absolute_error(predictions_df['actual_away_score'], predictions_df['ensemble_pred_away'])
        ensemble_mae = (ensemble_mae_home + ensemble_mae_away) / 2
        
        print(f"\nEnsemble Model Performance:")
        print(f"  MAE (combined): {ensemble_mae:.3f}")
        
        # Win prediction accuracy
        ensemble_win_accuracy = (predictions_df['ensemble_pred_home_win'] == predictions_df['actual_home_win']).mean()
        classifier_win_accuracy = ((predictions_df['classifier_home_win_prob'] > 0.5).astype(int) == predictions_df['actual_home_win']).mean()
        
        print(f"\nWin Prediction Accuracy:")
        print(f"  Ensemble (score-based): {ensemble_win_accuracy:.3f}")
        print(f"  Classifier: {classifier_win_accuracy:.3f}")
    
    def run(self):
        """Main execution method"""
        print("Starting MLB Score Prediction Pipeline")
        print("="*60)
        
        # Step 1: Load and prepare data
        X_train, X_test, y_train, y_test = self.load_and_prepare_data()
        
        # Step 2: Scale features
        X_train_scaled, X_test_scaled, scaler = self.scale_features(X_train, X_test)
        
        # Step 3: Train all models
        print("\nTraining all models...")
        for model_name, model in self.models.items():
            try:
                results = self.train_single_model(
                    model_name, model,
                    X_train, X_test,  # Pass the original DataFrames
                    y_train, y_test,
                    scaler           # Pass the scaler
                )
                self.results.append(results)
            except Exception as e:
                print(f"  Error training {model_name}: {str(e)}")
                continue
        
        # Step 4: Create results DataFrame
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('mae_combined')
        
        # Step 5: Train win probability classifier
        win_probs_classifier, clf = self.train_win_probability_classifier(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        
        # Step 6: Create comprehensive predictions DataFrame
        predictions_df = self.create_final_predictions_dataframe(y_test, win_probs_classifier)
        
        # Step 7: Save all results
        self.save_results(results_df, predictions_df)
        
        # Step 8: Print summary
        self.print_summary(results_df, predictions_df)
        
        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print(f"All models and results saved to: {self.output_dir}")
        print("="*60)


def main():
    """Main function to run the predictor"""
    # Configuration
    DATA_PATH = r"C:\Users\DMelv\Documents\bettingModelBaseball\apiBaseball\pipeline\ml_pipeline_output\master_features_table.parquet"  # Update this to your actual file path
    OUTPUT_DIR = r"C:\Users\DMelv\Documents\bettingModelBaseball\apiBaseball\pipeline\artemis\models"
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        print("Please update the DATA_PATH variable with the correct path to your parquet file.")
        return
    
    # Create and run predictor
    predictor = MLBScorePredictor(DATA_PATH, OUTPUT_DIR)
    predictor.run()


if __name__ == "__main__":
    main()