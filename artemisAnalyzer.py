#!/usr/bin/env python3
"""
MLB Prediction Feature Engineering v2
Enhanced version with multicollinearity handling, data-driven binning, and correlation analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import os
warnings.filterwarnings('ignore')


class EnhancedPredictionFeatureEngineer:
    """Enhanced feature engineering with correlation analysis and adaptive binning"""
    
    def __init__(self, predictions_path: str, output_path: str = r"C:\Users\DMelv\Documents\bettingModelBaseball\apiBaseball\pipeline\artemis\models\game_predictions.parquet",
                 correlation_threshold: float = 0.95):
        """
        Initialize the enhanced feature engineer
        
        Args:
            predictions_path: Path to game_predictions.parquet
            output_path: Path to save engineered features
            correlation_threshold: Threshold for removing highly correlated features
        """
        self.predictions_path = predictions_path
        self.output_path = output_path
        self.correlation_threshold = correlation_threshold
        self.df = None
        self.model_names = []
        self.feature_groups = {}  # Track which features belong to which group
        self.removed_features = []  # Track removed features due to correlation
        
    def load_predictions(self):
        """Load predictions and identify model columns"""
        print("Loading predictions...")
        # Try parquet first, fall back to CSV if needed
        try:
            self.df = pd.read_parquet(self.predictions_path)
            print(f"Loaded {len(self.df)} game predictions from parquet")
        except:
            self.df = pd.read_csv(self.predictions_path)
            print(f"Loaded {len(self.df)} game predictions from CSV")
        
        # Identify all unique model names
        pred_cols = [col for col in self.df.columns if col.endswith('_pred_home') and 'ensemble' not in col]
        self.model_names = [col.replace('_pred_home', '') for col in pred_cols]
        print(f"Found {len(self.model_names)} models")
        
    def calculate_consensus_features(self):
        """Calculate consensus features with reduced redundancy"""
        print("\nCalculating consensus features...")
        
        home_pred_cols = [f"{model}_pred_home" for model in self.model_names]
        away_pred_cols = [f"{model}_pred_away" for model in self.model_names]
        win_prob_cols = [f"{model}_home_win_prob" for model in self.model_names]
        
        # Core dispersion metrics (choose complementary ones)
        self.df['home_pred_std'] = self.df[home_pred_cols].std(axis=1)
        self.df['away_pred_std'] = self.df[away_pred_cols].std(axis=1)
        self.df['win_prob_std'] = self.df[win_prob_cols].std(axis=1)
        
        # IQR as a robust alternative to std
        self.df['home_pred_iqr'] = self.df[home_pred_cols].quantile(0.75, axis=1) - self.df[home_pred_cols].quantile(0.25, axis=1)
        self.df['win_prob_iqr'] = self.df[win_prob_cols].quantile(0.75, axis=1) - self.df[win_prob_cols].quantile(0.25, axis=1)
        
        # Coefficient of variation (normalized dispersion)
        home_mean = self.df[home_pred_cols].mean(axis=1)
        away_mean = self.df[away_pred_cols].mean(axis=1)
        self.df['home_pred_cv'] = self.df['home_pred_std'] / (home_mean + 0.001)
        self.df['away_pred_cv'] = self.df['away_pred_std'] / (away_mean + 0.001)
        
        # Model agreement metrics
        win_pred_cols = [f"{model}_pred_home_win" for model in self.model_names]
        self.df['model_win_consensus_pct'] = self.df[win_pred_cols].mean(axis=1)
        
        # Entropy as a single uncertainty measure
        def calculate_entropy(probs):
            probs = np.clip(probs, 1e-10, 1-1e-10)
            return -np.mean(probs * np.log(probs) + (1-probs) * np.log(1-probs))
        
        self.df['win_prob_entropy'] = self.df[win_prob_cols].apply(
            lambda row: calculate_entropy(row.values), axis=1
        )
        
        # Track feature groups
        self.feature_groups['consensus'] = [
            'home_pred_std', 'away_pred_std', 'win_prob_std',
            'home_pred_iqr', 'win_prob_iqr', 'home_pred_cv', 'away_pred_cv',
            'model_win_consensus_pct', 'win_prob_entropy'
        ]
        
        print(f"  Created {len(self.feature_groups['consensus'])} consensus features")
        
    def calculate_market_disagreement_features(self):
        """Calculate market disagreement with focus on actionable features"""
        print("\nCalculating market disagreement features...")
        
        # Convert decimal odds to implied probabilities
        def decimal_to_prob(odds):
            if pd.isna(odds) or odds == 0:
                return np.nan
            return 1 / odds

        # Market implied probabilities
        self.df['home_market_prob'] = self.df['home_ml'].apply(decimal_to_prob)
        self.df['away_market_prob'] = self.df['away_ml'].apply(decimal_to_prob)
        
        # Key value metrics - focus on ensemble and best/worst
        self.df['ensemble_value_score'] = self.df['ensemble_home_win_prob'] - self.df['home_market_prob']
        self.df['ensemble_value_abs'] = abs(self.df['ensemble_value_score'])
        
        # Get max and min value across models (extremes often matter)
        value_cols = []
        for model in self.model_names:
            prob_col = f"{model}_home_win_prob"
            if prob_col in self.df.columns:
                value_col = f"{model}_value_score"
                self.df[value_col] = self.df[prob_col] - self.df['home_market_prob']
                value_cols.append(value_col)
        
        if value_cols:
            self.df['max_model_value'] = self.df[value_cols].max(axis=1)
            self.df['min_model_value'] = self.df[value_cols].min(axis=1)
            self.df['value_score_range'] = self.df['max_model_value'] - self.df['min_model_value']
            
            # Clean up individual model value scores to reduce features
            self.df.drop(columns=value_cols, inplace=True)
        
        # Spread and total analysis
        self.df['ensemble_spread'] = self.df['ensemble_pred_home'] - self.df['ensemble_pred_away']
        self.df['ensemble_total'] = self.df['ensemble_pred_home'] + self.df['ensemble_pred_away']
        
        if 'total_line' in self.df.columns:
            self.df['total_diff_from_market'] = self.df['ensemble_total'] - self.df['total_line']
            self.df['total_diff_pct'] = self.df['total_diff_from_market'] / (self.df['total_line'] + 0.001)
        
        # Track features
        self.feature_groups['market'] = [
            'home_market_prob', 'away_market_prob', 'ensemble_value_score', 
            'ensemble_value_abs', 'max_model_value', 'min_model_value',
            'value_score_range', 'ensemble_spread', 'ensemble_total'
        ]
        
        if 'total_diff_from_market' in self.df.columns:
            self.feature_groups['market'].extend(['total_diff_from_market', 'total_diff_pct'])
        
        print(f"  Created {len(self.feature_groups['market'])} market features")
        
    def calculate_confidence_features(self):
        """Calculate streamlined confidence features"""
        print("\nCalculating confidence features...")
        
        # Ensemble prediction magnitude
        self.df['ensemble_pred_magnitude'] = abs(self.df['ensemble_pred_home'] - self.df['ensemble_pred_away'])
        
        # Probability certainty (distance from 50/50)
        self.df['ensemble_prob_certainty'] = abs(self.df['ensemble_home_win_prob'] - 0.5)
        self.df['classifier_prob_certainty'] = abs(self.df['classifier_home_win_prob'] - 0.5)
        
        # Agreement between ensemble and classifier
        self.df['ensemble_classifier_agreement'] = 1 - abs(
            self.df['ensemble_home_win_prob'] - self.df['classifier_home_win_prob']
        )
        
        # Instead of hardcoded confidence score, provide components
        # This allows the final model to learn optimal weights
        self.df['confidence_component_1'] = 1 - self.df['win_prob_std']  # Low variance is good
        self.df['confidence_component_2'] = self.df['ensemble_prob_certainty']  # High certainty is good
        self.df['confidence_component_3'] = 1 - self.df['home_pred_cv']  # Low CV is good
        self.df['confidence_component_4'] = self.df['ensemble_classifier_agreement']  # Agreement is good
        
        # Track features
        self.feature_groups['confidence'] = [
            'ensemble_pred_magnitude', 'ensemble_prob_certainty', 
            'classifier_prob_certainty', 'ensemble_classifier_agreement',
            'confidence_component_1', 'confidence_component_2',
            'confidence_component_3', 'confidence_component_4'
        ]
        
        print(f"  Created {len(self.feature_groups['confidence'])} confidence features")
        
    def calculate_interaction_features(self):
        """Calculate key interaction features"""
        print("\nCalculating interaction features...")
        
        # Value × Uncertainty interactions
        self.df['value_uncertainty_interaction'] = (
            self.df['ensemble_value_abs'] * (1 - self.df['win_prob_std'])
        )
        
        # Magnitude × Certainty
        self.df['magnitude_certainty_interaction'] = (
            self.df['ensemble_pred_magnitude'] * self.df['ensemble_prob_certainty']
        )
        
        # Market disagreement × Model consensus
        self.df['market_model_interaction'] = (
            self.df['ensemble_value_abs'] * self.df['model_win_consensus_pct']
        )
        
        # Track features
        self.feature_groups['interaction'] = [
            'value_uncertainty_interaction', 'magnitude_certainty_interaction',
            'market_model_interaction'
        ]
        
        print(f"  Created {len(self.feature_groups['interaction'])} interaction features")

    def calculate_kelly_bet_sizes(self):
        """Calculate optimal bet sizes using Kelly Criterion"""
        print("\nCalculating Kelly Criterion bet sizes...")
        
        # Kelly formula: f = (p*b - q) / b
        # where f = fraction of bankroll to bet
        # p = probability of winning
        # q = probability of losing (1-p)
        # b = odds received on the bet (decimal odds - 1)
        
        def kelly_fraction(win_prob, decimal_odds, kelly_multiplier=0.25):
            """
            Calculate Kelly fraction with safety multiplier
            
            Args:
                win_prob: Probability of winning (0-1)
                decimal_odds: Decimal odds
                kelly_multiplier: Fraction of full Kelly to use (default 0.25 for safety)
            """
            if decimal_odds <= 1 or win_prob <= 0 or win_prob >= 1:
                return 0
            
            # Convert decimal odds to net odds (b)
            b = decimal_odds - 1
            p = win_prob
            q = 1 - p
            
            # Full Kelly
            f_full = (p * b - q) / b
            
            # Apply safety multiplier and cap
            f_safe = f_full * kelly_multiplier
            
            # Cap at 5% of bankroll max
            return min(max(0, f_safe), 0.05)
        
        # Calculate for all games
        self.df['home_kelly_fraction'] = self.df.apply(
            lambda row: kelly_fraction(row['ensemble_home_win_prob'], row['home_ml']),
            axis=1
        )
        
        self.df['away_kelly_fraction'] = self.df.apply(
            lambda row: kelly_fraction(1 - row['ensemble_home_win_prob'], row['away_ml']),
            axis=1
        )
        
        # Determine which side to bet and how much
        self.df['kelly_bet_side'] = np.where(
            self.df['ensemble_value_score'] > 0, 'home',
            np.where(self.df['ensemble_value_score'] < 0, 'away', 'none')
        )
        
        self.df['kelly_bet_fraction'] = np.where(
            self.df['kelly_bet_side'] == 'home',
            self.df['home_kelly_fraction'],
            np.where(
                self.df['kelly_bet_side'] == 'away',
                self.df['away_kelly_fraction'],
                0
            )
        )
        
        # Only bet when we have positive expected value
        self.df['kelly_bet_fraction'] = np.where(
            self.df['ensemble_value_abs'] > 0.02,
            self.df['kelly_bet_fraction'],
            0
        )
        
        # Track features
        self.feature_groups['kelly'] = [
            'home_kelly_fraction', 'away_kelly_fraction', 
            'kelly_bet_side', 'kelly_bet_fraction'
        ]
        
        print(f"  Average Kelly fraction (when betting): {self.df[self.df['kelly_bet_fraction'] > 0]['kelly_bet_fraction'].mean():.3f}")
        print(f"  Max Kelly fraction: {self.df['kelly_bet_fraction'].max():.3f}")

    def create_cumulative_profit_analysis(self):
        """Create cumulative profit charts for all strategies"""
        print("\nCreating cumulative profit analysis...")
        
        # Only use games with results
        backtest_df = self.df[self.df['actual_home_win'].notna()].copy()
        backtest_df = backtest_df.sort_values('game_date')
        
        # Define strategies
        strategies = {
            'Simple Threshold': backtest_df['strong_bet_flag'] == 1,
            'Conservative': backtest_df['conservative_bet_flag'] == 1,
            'ML Enhanced': backtest_df['ml_bet_flag_enhanced'] == 1 if 'ml_bet_flag_enhanced' in backtest_df.columns else None
        }
        
        # Calculate cumulative profits
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        for strategy_name, strategy_mask in strategies.items():
            if strategy_mask is None:
                continue
                
            cumulative_profit = []
            cumulative_roi = []
            total_invested = 0
            current_profit = 0
            
            for idx, row in backtest_df.iterrows():
                if strategy_mask.loc[idx]:
                    # Calculate profit for this bet
                    if row['ensemble_value_score'] > 0:
                        bet_won = row['actual_home_win'] == 1
                        odds = row['home_ml']
                    else:
                        bet_won = row['actual_home_win'] == 0
                        odds = row['away_ml']
                    
                    if bet_won:
                        profit = (100 * odds) - 100
                    else:
                        profit = -100
                    
                    current_profit += profit
                    total_invested += 100
                
                cumulative_profit.append(current_profit)
                roi = (current_profit / total_invested * 100) if total_invested > 0 else 0
                cumulative_roi.append(roi)
            
            # Plot cumulative profit
            ax1.plot(backtest_df['game_date'], cumulative_profit, label=strategy_name, linewidth=2)
            
            # Plot ROI
            ax2.plot(backtest_df['game_date'], cumulative_roi, label=strategy_name, linewidth=2)
        
        # Format plots
        ax1.set_title('Cumulative Profit Over Time', fontsize=14)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Profit ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax2.set_title('Return on Investment (ROI) Over Time', fontsize=14)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('ROI (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        # Save
        chart_path = self.output_path.replace('.parquet', '_cumulative_profit_analysis.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"  Saved cumulative profit chart to {chart_path}")
        plt.close()

    def analyze_strategy_by_segments(self):
        """Analyze strategy performance across different game segments"""
        print("\nAnalyzing strategy performance by segments...")
        
        backtest_df = self.df[self.df['actual_home_win'].notna()].copy()
        
        # Define segments
        segments = {
            'All Games': backtest_df.index,
            'Home Bets': backtest_df[backtest_df['ensemble_value_score'] > 0].index,
            'Away Bets': backtest_df[backtest_df['ensemble_value_score'] < 0].index,
            'Favorites': backtest_df[backtest_df['home_market_prob'] > 0.6].index,
            'Underdogs': backtest_df[backtest_df['home_market_prob'] < 0.4].index,
            'Close Games': backtest_df[backtest_df['home_market_prob'].between(0.4, 0.6)].index,
            'High Totals': backtest_df[backtest_df['ensemble_total'] > backtest_df['ensemble_total'].quantile(0.75)].index,
            'Low Totals': backtest_df[backtest_df['ensemble_total'] < backtest_df['ensemble_total'].quantile(0.25)].index
        }
        
        # Analyze ML model performance by segment
        results = []
        
        for segment_name, segment_idx in segments.items():
            segment_df = backtest_df.loc[segment_idx]
            ml_bets = segment_df[segment_df['ml_bet_flag_enhanced'] == 1]
            
            if len(ml_bets) == 0:
                continue
            
            # Calculate performance
            profits = []
            for idx, row in ml_bets.iterrows():
                if row['ensemble_value_score'] > 0:
                    bet_won = row['actual_home_win'] == 1
                    odds = row['home_ml']
                else:
                    bet_won = row['actual_home_win'] == 0
                    odds = row['away_ml']
                
                profit = (100 * odds - 100) if bet_won else -100
                profits.append(profit)
            
            total_profit = sum(profits)
            num_bets = len(profits)
            win_rate = sum(1 for p in profits if p > 0) / num_bets
            roi = (total_profit / (num_bets * 100)) * 100
            
            results.append({
                'Segment': segment_name,
                'Games_in_Segment': len(segment_df),
                'ML_Bets': num_bets,
                'Bet_Rate': f"{(num_bets/len(segment_df))*100:.1f}%",
                'Win_Rate': f"{win_rate:.1%}",
                'Total_Profit': f"${total_profit:.2f}",
                'ROI': f"{roi:.1f}%"
            })
        
        # Create DataFrame and save
        segment_df = pd.DataFrame(results)
        print("\nML Model Performance by Segment:")
        print(segment_df.to_string(index=False))
        
        # Save results
        segment_path = self.output_path.replace('.parquet', '_segment_analysis.csv')
        segment_df.to_csv(segment_path, index=False)
        print(f"\nSaved segment analysis to {segment_path}")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Convert ROI to numeric for plotting
        segment_df['ROI_numeric'] = segment_df['ROI'].str.rstrip('%').astype(float)
        
        # Bar chart
        colors = ['green' if roi > 0 else 'red' for roi in segment_df['ROI_numeric']]
        bars = ax.bar(segment_df['Segment'], segment_df['ROI_numeric'], color=colors)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height > 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height > 0 else 'top')
        
        ax.set_title('ML Model ROI by Game Segment', fontsize=14)
        ax.set_xlabel('Segment')
        ax.set_ylabel('ROI (%)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = self.output_path.replace('.parquet', '_segment_roi_chart.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"Saved segment ROI chart to {chart_path}")
        plt.close()

    def find_segmented_thresholds(self, bet_amount=100):
        """Find optimal thresholds for different betting segments"""
        print("\nFinding segmented optimal thresholds...")
        
        segments = {
            'home_bets': self.df[self.df['ensemble_value_score'] > 0],
            'away_bets': self.df[self.df['ensemble_value_score'] < 0],
            'favorites': self.df[self.df['home_market_prob'] > 0.65],
            'underdogs': self.df[self.df['home_market_prob'] < 0.35],
            'close_games': self.df[self.df['home_market_prob'].between(0.45, 0.55)],
            'high_totals': self.df[self.df['ensemble_total'] > self.df['ensemble_total'].quantile(0.75)],
            'low_totals': self.df[self.df['ensemble_total'] < self.df['ensemble_total'].quantile(0.25)]
        }
        
        segmented_results = {}
        
        for segment_name, segment_df in segments.items():
            # Only use games with results
            backtest_df = segment_df[segment_df['actual_home_win'].notna()].copy()
            
            if len(backtest_df) < 50:  # Skip if too few games
                print(f"  Skipping {segment_name}: only {len(backtest_df)} games with results")
                continue
                
            # Run same grid search logic as before
            best_roi, best_balanced = self._run_threshold_search(backtest_df, bet_amount)
            
            segmented_results[segment_name] = {
                'value_threshold': best_balanced['value_threshold'],
                'certainty_threshold': best_balanced['certainty_threshold'],
                'expected_roi': best_balanced['roi'],
                'num_games': len(backtest_df),
                'strategy_details': best_balanced
            }
            
            print(f"\n{segment_name.upper()} Optimal Strategy:")
            print(f"  Value: {best_balanced['value_threshold']:.3f}, Certainty: {best_balanced['certainty_threshold']:.3f}")
            print(f"  ROI: {best_balanced['roi']:.1f}%, Bets: {best_balanced['num_bets']}")
        
        return segmented_results
        
    def find_optimal_thresholds(self, bet_amount=100):
        """
        Find optimal value and certainty thresholds through backtesting
        
        Args:
            bet_amount: Amount to bet on each game (default $100)
        
        Returns:
            dict: Optimal thresholds and performance metrics
        """
        print("\nFinding optimal betting thresholds through backtesting...")
        
        # Only use games with actual results
        backtest_df = self.df[self.df['actual_home_win'].notna()].copy()
        print(f"Using {len(backtest_df)} games with known outcomes for backtesting")
        
        # Define search ranges
        value_thresholds = np.linspace(0.01, 0.20, 20)  # 1% to 20% value edge
        certainty_thresholds = np.linspace(0.05, 0.35, 20)  # 5% to 35% certainty
        
        results = []
        
        # Grid search
        for val_threshold in value_thresholds:
            for cert_threshold in certainty_thresholds:
                # Identify bets meeting thresholds
                strong_bets = backtest_df[
                    (backtest_df['ensemble_value_abs'] > val_threshold) & 
                    (backtest_df['ensemble_prob_certainty'] > cert_threshold)
                ].copy()
                
                if len(strong_bets) == 0:
                    continue
                
                # Calculate profit for each bet
                profits = []
                for idx, row in strong_bets.iterrows():
                    # Determine which team we're betting on
                    if row['ensemble_value_score'] > 0:
                        # Bet on home team
                        bet_won = row['actual_home_win'] == 1
                        odds = row['home_ml']
                    else:
                        # Bet on away team
                        bet_won = row['actual_home_win'] == 0
                        odds = row['away_ml']
                    
                    # Calculate profit
                    if bet_won:
                        profit = (bet_amount * odds) - bet_amount
                    else:
                        profit = -bet_amount
                    
                    profits.append(profit)
                
                # Calculate metrics
                total_profit = sum(profits)
                num_bets = len(profits)
                win_rate = sum(1 for p in profits if p > 0) / num_bets
                roi = (total_profit / (num_bets * bet_amount)) * 100
                avg_profit_per_bet = total_profit / num_bets
                
                results.append({
                    'value_threshold': val_threshold,
                    'certainty_threshold': cert_threshold,
                    'num_bets': num_bets,
                    'total_profit': total_profit,
                    'roi': roi,
                    'win_rate': win_rate,
                    'avg_profit_per_bet': avg_profit_per_bet,
                    'bets_per_100_games': (num_bets / len(backtest_df)) * 100
                })
        
        # Convert to DataFrame for analysis
        results_df = pd.DataFrame(results)
        
        # Find best by different criteria
        best_roi = results_df.loc[results_df['roi'].idxmax()]
        best_total_profit = results_df.loc[results_df['total_profit'].idxmax()]
        
        # Find best with minimum bet requirement (at least 50 bets)
        min_bets_df = results_df[results_df['num_bets'] >= 50]
        if len(min_bets_df) > 0:
            best_balanced = min_bets_df.loc[min_bets_df['roi'].idxmax()]
        else:
            best_balanced = best_roi
        
        print("\n" + "="*60)
        print("BACKTESTING RESULTS")
        print("="*60)
        
        print(f"\nBest ROI Strategy:")
        print(f"  Value Threshold: {best_roi['value_threshold']:.3f}")
        print(f"  Certainty Threshold: {best_roi['certainty_threshold']:.3f}")
        print(f"  ROI: {best_roi['roi']:.1f}%")
        print(f"  Win Rate: {best_roi['win_rate']:.1%}")
        print(f"  Number of Bets: {best_roi['num_bets']}")
        print(f"  Total Profit: ${best_roi['total_profit']:.2f}")
        
        print(f"\nBest Total Profit Strategy:")
        print(f"  Value Threshold: {best_total_profit['value_threshold']:.3f}")
        print(f"  Certainty Threshold: {best_total_profit['certainty_threshold']:.3f}")
        print(f"  ROI: {best_total_profit['roi']:.1f}%")
        print(f"  Win Rate: {best_total_profit['win_rate']:.1%}")
        print(f"  Number of Bets: {best_total_profit['num_bets']}")
        print(f"  Total Profit: ${best_total_profit['total_profit']:.2f}")
        
        print(f"\nBest Balanced Strategy (min 50 bets):")
        print(f"  Value Threshold: {best_balanced['value_threshold']:.3f}")
        print(f"  Certainty Threshold: {best_balanced['certainty_threshold']:.3f}")
        print(f"  ROI: {best_balanced['roi']:.1f}%")
        print(f"  Win Rate: {best_balanced['win_rate']:.1%}")
        print(f"  Number of Bets: {best_balanced['num_bets']}")
        print(f"  Total Profit: ${best_balanced['total_profit']:.2f}")
        
        # Create visualization
        self._create_threshold_heatmap(results_df)
        
        # Store optimal thresholds
        self.optimal_thresholds = {
            'value': best_balanced['value_threshold'],
            'certainty': best_balanced['certainty_threshold'],
            'expected_roi': best_balanced['roi'],
            'expected_win_rate': best_balanced['win_rate']
        }
        
        return self.optimal_thresholds

    def _create_threshold_heatmap(self, results_df):
        """Create heatmap visualization of threshold performance"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Pivot for heatmap
        roi_pivot = results_df.pivot(
            index='certainty_threshold',
            columns='value_threshold',
            values='roi'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            roi_pivot,
            cmap='RdYlGn',
            center=0,
            annot=True,
            fmt='.1f',
            cbar_kws={'label': 'ROI (%)'}
        )
        plt.title('ROI Heatmap: Value vs Certainty Thresholds')
        plt.xlabel('Value Threshold')
        plt.ylabel('Certainty Threshold')
        plt.tight_layout()
        
        # Save the heatmap
        heatmap_path = self.output_path.replace('.parquet', '_threshold_optimization.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved threshold optimization heatmap to {heatmap_path}")
        plt.close()

    def create_adaptive_categorical_features_optimized(self):
        """
        Create categorical features using optimized thresholds from backtesting
        This replaces the original create_adaptive_categorical_features method
        """
        print("\nCreating optimized categorical features...")
        
        # First, find optimal thresholds if not already done
        if not hasattr(self, 'optimal_thresholds'):
            self.find_optimal_thresholds()
        
        # Use quantile-based binning for categories (these are still useful)
        value_quantiles = self.df['ensemble_value_abs'].quantile([0, 0.25, 0.5, 0.75, 1.0])
        self.df['value_category'] = pd.cut(
            self.df['ensemble_value_abs'],
            bins=value_quantiles.values,
            labels=['minimal', 'small', 'medium', 'large'],
            include_lowest=True,
            duplicates='drop'
        )
        
        certainty_quantiles = self.df['ensemble_prob_certainty'].quantile([0, 0.25, 0.5, 0.75, 1.0])
        self.df['certainty_category'] = pd.cut(
            self.df['ensemble_prob_certainty'],
            bins=certainty_quantiles.values,
            labels=['very_low', 'low', 'medium', 'high'],
            include_lowest=True,
            duplicates='drop'
        )
        
        std_quantiles = self.df['win_prob_std'].quantile([0, 0.25, 0.5, 0.75, 1.0])
        self.df['consensus_category'] = pd.cut(
            self.df['win_prob_std'],
            bins=std_quantiles.values,
            labels=['high', 'medium', 'low', 'very_low'],
            include_lowest=True,
            duplicates='drop'
        )
        
        # Use OPTIMIZED thresholds for betting flags
        value_threshold = self.optimal_thresholds['value']
        certainty_threshold = self.optimal_thresholds['certainty']
        
        self.df['strong_bet_flag'] = (
            (self.df['ensemble_value_abs'] > value_threshold) & 
            (self.df['ensemble_prob_certainty'] > certainty_threshold)
        ).astype(int)
        
        self.df['bet_direction'] = np.where(
            self.df['ensemble_value_score'] > 0, 'home',
            np.where(self.df['ensemble_value_score'] < 0, 'away', 'none')
        )
        
        # Add additional flags based on optimization insights
        # Medium confidence bets (lower thresholds)
        self.df['medium_bet_flag'] = (
            (self.df['ensemble_value_abs'] > value_threshold * 0.7) & 
            (self.df['ensemble_prob_certainty'] > certainty_threshold * 0.7) &
            (self.df['strong_bet_flag'] == 0)  # Not already a strong bet
        ).astype(int)
        
        # Track features
        self.feature_groups['categorical'] = [
            'value_category', 'certainty_category', 'consensus_category',
            'strong_bet_flag', 'medium_bet_flag', 'bet_direction'
        ]
        
        print(f"  Created {len(self.feature_groups['categorical'])} categorical features")
        print(f"  Optimized value threshold: {value_threshold:.3f}")
        print(f"  Optimized certainty threshold: {certainty_threshold:.3f}")
        print(f"  Expected ROI: {self.optimal_thresholds['expected_roi']:.1f}%")
        print(f"  Expected Win Rate: {self.optimal_thresholds['expected_win_rate']:.1%}")
        
        # Show how many bets qualify
        strong_bets = self.df[self.df['strong_bet_flag'] == 1]
        medium_bets = self.df[self.df['medium_bet_flag'] == 1]
        print(f"  Strong bets identified: {len(strong_bets)} ({len(strong_bets)/len(self.df)*100:.1f}%)")
        print(f"  Medium bets identified: {len(medium_bets)} ({len(medium_bets)/len(self.df)*100:.1f}%)")

    def create_multi_factor_betting_rules(self):
        """Create sophisticated betting rules using multiple features"""
        print("\nCreating multi-factor betting rules...")
        
        # Conservative Strategy: High agreement + High value + High certainty
        self.df['conservative_bet_flag'] = (
            (self.df['ensemble_value_abs'] > self.optimal_thresholds['value']) & 
            (self.df['ensemble_prob_certainty'] > self.optimal_thresholds['certainty']) &
            (self.df['win_prob_std'] < self.df['win_prob_std'].quantile(0.25)) &  # Low variance
            (self.df['model_win_consensus_pct'] > 0.75)  # High agreement
        ).astype(int)
        
        # Aggressive Strategy: Good value with lower requirements
        self.df['aggressive_bet_flag'] = (
            (self.df['ensemble_value_abs'] > self.optimal_thresholds['value'] * 0.6) & 
            (self.df['value_uncertainty_interaction'] > self.df['value_uncertainty_interaction'].quantile(0.8))
        ).astype(int)
        
        # Kelly Criterion-inspired sizing
        self.df['suggested_bet_size'] = np.where(
            self.df['conservative_bet_flag'] == 1, 'full',
            np.where(self.df['aggressive_bet_flag'] == 1, 'half', 'none')
        )
        
        # ADD THE TRACKING HERE:
        # Track these new features
        if 'categorical' not in self.feature_groups:
            self.feature_groups['categorical'] = []
        self.feature_groups['categorical'].extend([
            'conservative_bet_flag', 'aggressive_bet_flag', 'suggested_bet_size'
        ])
        
        # Backtest these new strategies
        strategies = ['conservative_bet_flag', 'aggressive_bet_flag']
        for strategy in strategies:
            self._backtest_strategy(strategy)

    def analyze_feature_correlations(self):
        """Analyze and handle multicollinearity"""
        print("\nAnalyzing feature correlations...")
        
        # Get all numeric features we created
        all_features = []
        for group in ['consensus', 'market', 'confidence', 'interaction']:
            if group in self.feature_groups:
                all_features.extend(self.feature_groups[group])
        
        # Calculate correlation matrix
        numeric_features = [f for f in all_features if f in self.df.columns and self.df[f].dtype in ['float64', 'int64']]
        corr_matrix = self.df[numeric_features].corr()
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(numeric_features)):
            for j in range(i+1, len(numeric_features)):
                if abs(corr_matrix.iloc[i, j]) > self.correlation_threshold:
                    high_corr_pairs.append({
                        'feature1': numeric_features[i],
                        'feature2': numeric_features[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        if high_corr_pairs:
            print(f"\n  Found {len(high_corr_pairs)} highly correlated feature pairs (>{self.correlation_threshold}):")
            
            # Remove features with high correlation
            # Keep the feature that appears first in our feature groups
            features_to_remove = set()
            for pair in high_corr_pairs:
                # Determine which feature to keep based on order in feature groups
                feature1_priority = self._get_feature_priority(pair['feature1'])
                feature2_priority = self._get_feature_priority(pair['feature2'])
                
                if feature1_priority <= feature2_priority:
                    features_to_remove.add(pair['feature2'])
                    print(f"    {pair['feature1']} vs {pair['feature2']} (r={pair['correlation']:.3f}) - keeping {pair['feature1']}")
                else:
                    features_to_remove.add(pair['feature1'])
                    print(f"    {pair['feature1']} vs {pair['feature2']} (r={pair['correlation']:.3f}) - keeping {pair['feature2']}")
            
            # Remove the features
            self.removed_features = list(features_to_remove)
            for feature in features_to_remove:
                if feature in self.df.columns:
                    self.df.drop(columns=[feature], inplace=True)
                # Update feature groups
                for group, features in self.feature_groups.items():
                    if feature in features:
                        self.feature_groups[group].remove(feature)
            
            print(f"\n  Removed {len(features_to_remove)} features due to high correlation")
        else:
            print("  No highly correlated features found")

    def _run_threshold_search(self, backtest_df, bet_amount=100):
        """Helper method to run threshold search on a dataset"""
        value_thresholds = np.linspace(0.01, 0.20, 20)
        certainty_thresholds = np.linspace(0.05, 0.35, 20)
        
        results = []
        
        for val_threshold in value_thresholds:
            for cert_threshold in certainty_thresholds:
                strong_bets = backtest_df[
                    (backtest_df['ensemble_value_abs'] > val_threshold) & 
                    (backtest_df['ensemble_prob_certainty'] > cert_threshold)
                ].copy()
                
                if len(strong_bets) == 0:
                    continue
                
                profits = []
                for idx, row in strong_bets.iterrows():
                    if row['ensemble_value_score'] > 0:
                        bet_won = row['actual_home_win'] == 1
                        odds = row['home_ml']
                    else:
                        bet_won = row['actual_home_win'] == 0
                        odds = row['away_ml']
                    
                    if bet_won:
                        profit = (bet_amount * odds) - bet_amount
                    else:
                        profit = -bet_amount
                    
                    profits.append(profit)
                
                total_profit = sum(profits)
                num_bets = len(profits)
                win_rate = sum(1 for p in profits if p > 0) / num_bets
                roi = (total_profit / (num_bets * bet_amount)) * 100
                
                results.append({
                    'value_threshold': val_threshold,
                    'certainty_threshold': cert_threshold,
                    'num_bets': num_bets,
                    'total_profit': total_profit,
                    'roi': roi,
                    'win_rate': win_rate,
                    'avg_profit_per_bet': total_profit / num_bets
                })
        
        results_df = pd.DataFrame(results)
        
        # Find best strategies
        best_roi = results_df.loc[results_df['roi'].idxmax()]
        
        min_bets_df = results_df[results_df['num_bets'] >= 50]
        if len(min_bets_df) > 0:
            best_balanced = min_bets_df.loc[min_bets_df['roi'].idxmax()]
        else:
            best_balanced = best_roi
        
        return best_roi, best_balanced

    def _backtest_strategy(self, strategy_flag):
        """Backtest a specific betting strategy"""
        backtest_df = self.df[self.df['actual_home_win'].notna()].copy()
        strategy_bets = backtest_df[backtest_df[strategy_flag] == 1]
        
        if len(strategy_bets) == 0:
            print(f"  {strategy_flag}: No bets identified")
            return
        
        profits = []
        for idx, row in strategy_bets.iterrows():
            if row['ensemble_value_score'] > 0:
                bet_won = row['actual_home_win'] == 1
                odds = row['home_ml']
            else:
                bet_won = row['actual_home_win'] == 0
                odds = row['away_ml']
            
            if bet_won:
                profit = (100 * odds) - 100
            else:
                profit = -100
            
            profits.append(profit)
        
        total_profit = sum(profits)
        num_bets = len(profits)
        win_rate = sum(1 for p in profits if p > 0) / num_bets
        roi = (total_profit / (num_bets * 100)) * 100
        
        print(f"  {strategy_flag}: {num_bets} bets, {win_rate:.1%} win rate, {roi:.1f}% ROI")

    def _get_feature_priority(self, feature: str) -> int:
        """Get priority order for a feature (lower number = higher priority)"""
        priority_order = ['consensus', 'market', 'confidence', 'interaction']
        
        for priority, group in enumerate(priority_order):
            if group in self.feature_groups and feature in self.feature_groups[group]:
                return priority * 100 + self.feature_groups[group].index(feature)
        
        return 999  # Unknown features get lowest priority

    def create_betting_profitability_model_enhanced(self):
        """Enhanced model with hyperparameter tuning and better validation"""
        from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import roc_auc_score, make_scorer
        from sklearn.feature_selection import RFECV
        import joblib
        from lightgbm import LGBMClassifier
        
        print("\nTraining enhanced betting profitability model...")
        
        # Create target variable
        train_df = self.df[self.df['actual_home_win'].notna()].copy()
        
        # Calculate profit for each potential bet
        profits = []
        for idx, row in train_df.iterrows():
            if abs(row['ensemble_value_score']) < 0.01:
                profit = 0
            elif row['ensemble_value_score'] > 0:
                if row['actual_home_win'] == 1:
                    profit = (100 * row['home_ml']) - 100
                else:
                    profit = -100
            else:
                if row['actual_home_win'] == 0:
                    profit = (100 * row['away_ml']) - 100
                else:
                    profit = -100
            profits.append(profit)
        
        train_df['bet_profit'] = profits
        train_df['is_profitable'] = (train_df['bet_profit'] > 0).astype(int)
        
        # Expanded feature set - include ALL engineered features
        all_features = []
        for group in ['consensus', 'market', 'confidence', 'interaction']:
            if group in self.feature_groups:
                all_features.extend(self.feature_groups[group])
        
        # Filter to numeric features only
        feature_cols = [f for f in all_features if f in train_df.columns and 
                    train_df[f].dtype in ['float64', 'int64'] and 
                    f not in self.removed_features]
        
        # Filter to games with significant value edge
        model_df = train_df[train_df['ensemble_value_abs'] > 0.02].copy()
        
        X = model_df[feature_cols]
        y = model_df['is_profitable']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"  Training set: {len(X_train)} games ({y_train.mean():.1%} profitable)")
        print(f"  Test set: {len(X_test)} games ({y_test.mean():.1%} profitable)")
        
        # 1. FEATURE SELECTION with RFECV
        print("\n  Performing automated feature selection...")
        base_model = LGBMClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        rfecv = RFECV(
            estimator=base_model,
            step=1,
            cv=StratifiedKFold(5),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        rfecv.fit(X_train, y_train)
        
        # Get selected features
        selected_features = [feature_cols[i] for i in range(len(feature_cols)) if rfecv.support_[i]]
        print(f"  Selected {len(selected_features)} out of {len(feature_cols)} features")
        print(f"  Optimal features: {selected_features[:10]}...")  # Show first 10
        
        # Update training data with selected features
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        # 2. HYPERPARAMETER TUNING
        print("\n  Tuning hyperparameters...")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_child_samples': [20, 50, 100], # Correct parameter name
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Custom scorer that considers profit, not just accuracy
        def profit_scorer(y_true, y_pred_proba):
            # Simple profit-based score
            threshold = 0.5
            y_pred = (y_pred_proba[:, 1] > threshold).astype(int)
            
            # Calculate profit for predictions
            correct_bets = (y_true == 1) & (y_pred == 1)
            incorrect_bets = (y_true == 0) & (y_pred == 1)
            
            # Assume average odds of 2.0 for simplicity in grid search
            profit = correct_bets.sum() * 100 - incorrect_bets.sum() * 100
            return profit
        
        # Use both AUC and profit for evaluation
        scoring = {
            'auc': 'roc_auc',
            'profit': make_scorer(profit_scorer, needs_proba=True)
        }
        
        grid_search = GridSearchCV(
            LGBMClassifier(random_state=42, class_weight='balanced'),
            param_grid,
            cv=StratifiedKFold(5),
            scoring=scoring,
            refit='auc',  # Refit on AUC
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_selected, y_train)
        
        print(f"\n  Best parameters: {grid_search.best_params_}")
        print(f"  Best CV AUC: {grid_search.best_score_:.3f}")
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # 3. FIND OPTIMAL THRESHOLD ON TRAINING SET ONLY
        print("\n  Finding optimal probability threshold (on training set)...")
        
        # Get predictions on training set
        y_train_proba = best_model.predict_proba(X_train_selected)[:, 1]
        
        # Test different thresholds on TRAINING data
        thresholds = np.linspace(0.3, 0.7, 41)
        best_threshold = 0.5
        best_train_profit = -np.inf
        
        train_indices = X_train.index
        for thresh in thresholds:
            train_preds = (y_train_proba > thresh).astype(int)
            
            # Calculate profit on training set
            total_profit = 0
            num_bets = 0
            
            for i, (idx, pred) in enumerate(zip(train_indices, train_preds)):
                if pred == 1:
                    num_bets += 1
                    total_profit += model_df.loc[idx, 'bet_profit']
            
            if num_bets > 0 and total_profit > best_train_profit:
                best_train_profit = total_profit
                best_threshold = thresh
        
        print(f"  Optimal threshold (from training): {best_threshold:.2f}")
        
        # 4. EVALUATE ON TEST SET with fixed threshold
        y_test_proba = best_model.predict_proba(X_test_selected)[:, 1]
        y_test_pred = (y_test_proba > best_threshold).astype(int)
        
        # Calculate test set performance
        test_auc = roc_auc_score(y_test, y_test_proba)
        
        # Calculate actual profit on test set
        test_profit = 0
        test_bets = 0
        test_indices = X_test.index
        
        for i, (idx, pred) in enumerate(zip(test_indices, y_test_pred)):
            if pred == 1:
                test_bets += 1
                test_profit += model_df.loc[idx, 'bet_profit']
        
        test_roi = (test_profit / (test_bets * 100)) * 100 if test_bets > 0 else 0
        
        print(f"\n  TEST SET PERFORMANCE:")
        print(f"    AUC: {test_auc:.3f}")
        print(f"    Number of bets: {test_bets}")
        print(f"    Total profit: ${test_profit:.2f}")
        print(f"    ROI: {test_roi:.1f}%")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n  Top 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.3f}")
        
        # Get the directory where the input predictions are, which is the 'models' dir
        models_dir = os.path.dirname(self.predictions_path)
        
        # Define a consistent name for the betting model
        model_filename = 'betting_model_enhanced.pkl'
        model_path = os.path.join(models_dir, model_filename)

        joblib.dump({
            'model': best_model,
            'features': selected_features,
            'threshold': best_threshold,
            'hyperparameters': grid_search.best_params_,
            'performance': {
                'train_auc': grid_search.best_score_,
                'test_auc': test_auc,
                'test_profit': test_profit,
                'test_roi': test_roi,
                'test_bets': test_bets
            }
        }, model_path)
        
        print(f"\n  Saved enhanced model to {model_path}")
        
        # Add predictions to full dataframe
        self.df['bet_profitability_score_enhanced'] = 0.5
        valid_idx = self.df[selected_features].notna().all(axis=1)
        self.df.loc[valid_idx, 'bet_profitability_score_enhanced'] = best_model.predict_proba(
            self.df.loc[valid_idx, selected_features]
        )[:, 1]
        
        self.df['ml_bet_flag_enhanced'] = (
            (self.df['bet_profitability_score_enhanced'] > best_threshold) &
            (self.df['ensemble_value_abs'] > 0.02)
        ).astype(int)
        
        # Track features
        if 'ml_predictions' not in self.feature_groups:
            self.feature_groups['ml_predictions'] = []
        self.feature_groups['ml_predictions'].extend([
            'bet_profitability_score_enhanced', 'ml_bet_flag_enhanced'
        ])
        
        return best_model, best_threshold, selected_features

    def compare_all_strategies(self):
        """Compare performance of all betting strategies"""
        print("\n" + "="*60)
        print("STRATEGY COMPARISON")
        print("="*60)
        
        # Only use games with results for comparison
        backtest_df = self.df[self.df['actual_home_win'].notna()].copy()
        
        # Define all strategies to compare
        strategies = {
            'Baseline (All Games)': pd.Series([True] * len(backtest_df), index=backtest_df.index),
            'Simple Threshold': backtest_df['strong_bet_flag'] == 1,
            'Medium Confidence': backtest_df['medium_bet_flag'] == 1,
            'Conservative': backtest_df['conservative_bet_flag'] == 1,
            'Aggressive': backtest_df['aggressive_bet_flag'] == 1,
            'ML Model': backtest_df['ml_bet_flag_enhanced'] == 1
        }
        
        # Compare each strategy
        results = []
        for strategy_name, strategy_mask in strategies.items():
            strategy_games = backtest_df[strategy_mask]
            
            if len(strategy_games) == 0:
                continue
            
            # Calculate profits
            profits = []
            for idx, row in strategy_games.iterrows():
                if row['ensemble_value_score'] > 0:
                    bet_won = row['actual_home_win'] == 1
                    odds = row['home_ml']
                else:
                    bet_won = row['actual_home_win'] == 0
                    odds = row['away_ml']
                
                if bet_won:
                    profit = (100 * odds) - 100
                else:
                    profit = -100
                
                profits.append(profit)
            
            # Calculate metrics
            total_profit = sum(profits)
            num_bets = len(profits)
            win_rate = sum(1 for p in profits if p > 0) / num_bets if num_bets > 0 else 0
            roi = (total_profit / (num_bets * 100)) * 100 if num_bets > 0 else 0
            avg_profit = total_profit / num_bets if num_bets > 0 else 0
            
            # Calculate Sharpe ratio (risk-adjusted returns)
            if len(profits) > 1:
                profit_std = np.std(profits)
                sharpe = (avg_profit / profit_std) * np.sqrt(162) if profit_std > 0 else 0  # Annualized for 162 games
            else:
                sharpe = 0
            
            results.append({
                'Strategy': strategy_name,
                'Bets': num_bets,
                'Bets_Pct': f"{(num_bets/len(backtest_df))*100:.1f}%",
                'Win_Rate': f"{win_rate:.1%}",
                'Total_Profit': f"${total_profit:.2f}",
                'ROI': f"{roi:.1f}%",
                'Avg_Profit': f"${avg_profit:.2f}",
                'Sharpe': f"{sharpe:.2f}"
            })
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(results)
        
        # Print comparison table
        print("\nStrategy Performance Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Find best strategies by different metrics
        if len(results) > 1:
            # Convert back to numeric for finding best
            for col in ['ROI', 'Sharpe']:
                comparison_df[f'{col}_numeric'] = comparison_df[col].str.rstrip('%').astype(float)
            
            best_roi_idx = comparison_df['ROI_numeric'].idxmax()
            best_sharpe_idx = comparison_df['Sharpe_numeric'].idxmax()
            
            print("\n" + "-"*60)
            print(f"Best ROI: {comparison_df.iloc[best_roi_idx]['Strategy']} ({comparison_df.iloc[best_roi_idx]['ROI']})")
            print(f"Best Risk-Adjusted: {comparison_df.iloc[best_sharpe_idx]['Strategy']} (Sharpe: {comparison_df.iloc[best_sharpe_idx]['Sharpe']})")
        
        # Save comparison results
        comparison_path = self.output_path.replace('.parquet', '_strategy_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nSaved strategy comparison to {comparison_path}")
        
        # Create visualization
        self._create_strategy_comparison_chart(comparison_df)

    def _create_strategy_comparison_chart(self, comparison_df):
        """Create visualization comparing all strategies"""
        import matplotlib.pyplot as plt
        
        # Prepare data
        comparison_df['ROI_numeric'] = comparison_df['ROI'].str.rstrip('%').astype(float)
        comparison_df['Bets_numeric'] = comparison_df['Bets']
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROI comparison
        colors = ['red' if roi < 0 else 'green' for roi in comparison_df['ROI_numeric']]
        ax1.bar(comparison_df['Strategy'], comparison_df['ROI_numeric'], color=colors)
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('ROI (%)')
        ax1.set_title('Return on Investment by Strategy')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Bet frequency vs ROI scatter
        ax2.scatter(comparison_df['Bets_numeric'], comparison_df['ROI_numeric'], s=100)
        for i, txt in enumerate(comparison_df['Strategy']):
            ax2.annotate(txt, (comparison_df['Bets_numeric'].iloc[i], comparison_df['ROI_numeric'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax2.set_xlabel('Number of Bets')
        ax2.set_ylabel('ROI (%)')
        ax2.set_title('ROI vs Betting Frequency')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the chart
        chart_path = self.output_path.replace('.parquet', '_strategy_comparison_chart.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"Saved strategy comparison chart to {chart_path}")
        plt.close()

    def create_correlation_heatmap(self, save_path: Optional[str] = None):
        """Create a correlation heatmap of engineered features"""
        print("\nCreating correlation heatmap...")
        
        # Get final numeric features
        all_features = []
        for features in self.feature_groups.values():
            all_features.extend(features)
        
        numeric_features = [f for f in all_features if f in self.df.columns and self.df[f].dtype in ['float64', 'int64']]
        
        if len(numeric_features) > 30:
            # Sample features if too many for visualization
            numeric_features = numeric_features[:30]
            print(f"  Showing top 30 features for visualization")
        
        corr_matrix = self.df[numeric_features].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap='RdBu_r', center=0, 
                    cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved heatmap to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    def save_engineered_features(self):
        """Save the engineered features with metadata"""
        print(f"\nSaving engineered features to {self.output_path}...")
        
        # Identify columns to save
        # Try to load original columns from parquet first, then CSV
        try:
            original_cols = pd.read_parquet(self.predictions_path).columns.tolist()
        except:
            original_cols = pd.read_csv(self.predictions_path).columns.tolist()
        
        # Keep key original columns
        cols_to_keep = [
            'game_pk', 'game_date', 'home_team', 'away_team',
            'home_team_id', 'away_team_id', 'home_ml', 'away_ml',
            'total_line', 'over_odds', 'under_odds',
            'actual_home_score', 'actual_away_score', 'actual_home_win',
            'ensemble_pred_home', 'ensemble_pred_away', 'ensemble_home_win_prob',
            'classifier_home_win_prob'
        ]
        
        # Add all engineered features
        engineered_features = []
        for features in self.feature_groups.values():
            engineered_features.extend(features)
        
        # Combine and filter
        columns_to_save = []
        for col in cols_to_keep + engineered_features:
            if col in self.df.columns:
                columns_to_save.append(col)
        
        # Remove duplicates while preserving order
        columns_to_save = list(dict.fromkeys(columns_to_save))
        
        # Save as parquet (primary format)
        self.df[columns_to_save].to_parquet(self.output_path, index=False)
        print(f"  Saved {len(self.df)} rows with {len(columns_to_save)} features to parquet")
        
        # Also save CSV version for compatibility
        csv_path = self.output_path.replace('.parquet', '.csv')
        self.df[columns_to_save].to_csv(csv_path, index=False)
        print(f"  Also saved CSV version: {csv_path}")

        # Save optimal thresholds for the predictor to use
        thresholds_path = os.path.join(os.path.dirname(self.predictions_path), 'optimal_thresholds.json')
        import json
        with open(thresholds_path, 'w') as f:
            json.dump(self.optimal_thresholds, f)
        print(f"  Saved optimal thresholds to {thresholds_path}")

        # Save metadata
        metadata = {
            'feature_groups': self.feature_groups,
            'removed_features': self.removed_features,
            'correlation_threshold': self.correlation_threshold,
            'total_features': len(engineered_features),
            'features_after_correlation_removal': len([f for f in engineered_features if f not in self.removed_features])
        }
        
        metadata_path = self.output_path.replace('.parquet', '_metadata.txt')
        with open(metadata_path, 'w') as f:
            f.write("Feature Engineering Metadata\n")
            f.write("=" * 50 + "\n\n")
            
            for group, features in self.feature_groups.items():
                f.write(f"{group.upper()} FEATURES ({len(features)}):\n")
                for feature in features:
                    status = " [REMOVED]" if feature in self.removed_features else ""
                    f.write(f"  - {feature}{status}\n")
                f.write("\n")
            
            f.write(f"\nTotal engineered features: {metadata['total_features']}\n")
            f.write(f"Features after correlation removal: {metadata['features_after_correlation_removal']}\n")
            f.write(f"Correlation threshold used: {metadata['correlation_threshold']}\n")
        
        print(f"  Saved metadata to {metadata_path}")
        
    def print_summary_statistics(self):
        """Print summary statistics of key features"""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*60)
        
        # Summary by feature group
        print("\nFeature Groups Summary:")
        total_features = 0
        for group, features in self.feature_groups.items():
            active_features = [f for f in features if f not in self.removed_features]
            print(f"  {group.capitalize()}: {len(active_features)} features ({len(features) - len(active_features)} removed)")
            total_features += len(active_features)
        
        print(f"\nTotal Active Features: {total_features}")
        
        # Key statistics
        print("\nKey Feature Statistics:")
        key_features = ['ensemble_value_abs', 'win_prob_std', 'ensemble_prob_certainty']
        for feature in key_features:
            if feature in self.df.columns:
                print(f"\n{feature}:")
                print(f"  Mean: {self.df[feature].mean():.3f}")
                print(f"  Std: {self.df[feature].std():.3f}")
                print(f"  25%: {self.df[feature].quantile(0.25):.3f}")
                print(f"  50%: {self.df[feature].quantile(0.50):.3f}")
                print(f"  75%: {self.df[feature].quantile(0.75):.3f}")
                print(f"  95%: {self.df[feature].quantile(0.95):.3f}")
        
        # Betting opportunities
        if 'strong_bet_flag' in self.df.columns:
            strong_bets = self.df[self.df['strong_bet_flag'] == 1]
            print(f"\nStrong Betting Opportunities: {len(strong_bets)} ({len(strong_bets)/len(self.df)*100:.1f}%)")
            
            if len(strong_bets) > 0:
                print("\nTop 5 Betting Opportunities:")
                # Use the absolute value of 'ensemble_value_score' for sorting
                # This column is guaranteed to exist
                top_bets = strong_bets.sort_values(by='ensemble_value_score', key=abs, ascending=False).head(5)[
                    ['game_date', 'home_team', 'away_team', 'bet_direction', 
                    'ensemble_value_score', 'ensemble_prob_certainty']
                ]
                for idx, row in top_bets.iterrows():
                    print(f"  {row['game_date']}: {row['bet_direction'].upper()} "
                        f"(value={abs(row['ensemble_value_score']):.3f}, certainty={row['ensemble_prob_certainty']:.3f})")
        
    def run(self):
        """Run the enhanced feature engineering pipeline"""
        print("Starting Enhanced MLB Prediction Feature Engineering")
        print("="*60)
        
        # Load data
        self.load_predictions()
        
        # Engineer features
        self.calculate_consensus_features()
        self.calculate_market_disagreement_features()
        self.calculate_confidence_features()
        self.calculate_interaction_features()
        self.create_adaptive_categorical_features_optimized()
        
        # Advanced strategies
        segmented_strategies = self.find_segmented_thresholds()
        self.create_multi_factor_betting_rules()
        
        # ENHANCED ML MODEL
        enhanced_model, enhanced_threshold, selected_features = self.create_betting_profitability_model_enhanced()
        
        # Kelly Criterion sizing
        self.calculate_kelly_bet_sizes()
        
        # Compare all strategies
        self.compare_all_strategies()
        
        # Advanced analysis
        self.create_cumulative_profit_analysis()
        self.analyze_strategy_by_segments()
        
        # Handle multicollinearity
        self.analyze_feature_correlations()
        
        # Create visualizations
        heatmap_path = self.output_path.replace('.parquet', '_correlation_heatmap.png')
        self.create_correlation_heatmap(heatmap_path)
        
        # Save results
        self.save_engineered_features()
        
        # Print summary
        self.print_summary_statistics()
        
        print("\n" + "="*60)
        print("Enhanced feature engineering completed successfully!")
        print("="*60)


def main():
    """Main function to run the enhanced feature engineering"""
    # Configuration
    PREDICTIONS_PATH = r"C:\Users\DMelv\Documents\bettingModelBaseball\apiBaseball\pipeline\artemis\models\game_predictions.parquet" # Using parquet format
    OUTPUT_PATH = r"C:\Users\DMelv\Documents\bettingModelBaseball\apiBaseball\pipeline\artemis\engineered_predictions_v2.parquet"
    CORRELATION_THRESHOLD = 0.95  # Remove features with correlation > 0.95
    
    # Create and run enhanced feature engineer
    engineer = EnhancedPredictionFeatureEngineer(
        predictions_path=PREDICTIONS_PATH,
        output_path=OUTPUT_PATH,
        correlation_threshold=CORRELATION_THRESHOLD
    )
    engineer.run()


if __name__ == "__main__":
    main()