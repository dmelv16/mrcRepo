# mlb_home_run_focused.py
"""
Focused MLB Home Run Model and Backtesting
Imports necessary components from main scripts
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from tqdm import tqdm
from pathlib import Path
import warnings
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, confusion_matrix, roc_curve
import json
from collections import defaultdict

# Import from the main scripts
from mlbPlayerPropv1 import (
    Config, 
    OptimizedDatabaseConnector, 
    OptimizedFeatureEngineer,
    CustomMLBModels
)

from mlbPropbacktestv1 import MLBBacktester

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HomeRunPipeline:
    """Pipeline specifically for home run model training and backtesting"""
    
    def __init__(self, config_path: str = None):
        self.config = Config(config_path)
        self.db = OptimizedDatabaseConnector(self.config)
        self.feature_engineer = None
        self.models = CustomMLBModels(self.config)
        self.all_data = None
        
    def load_data_for_home_runs(self, start_date: str, end_date: str):
        """Load data needed for home run analysis"""
        logger.info(f"Loading data from {start_date} to {end_date}")
        self.all_data = self.db.load_all_data_bulk(start_date, end_date)
        self.feature_engineer = OptimizedFeatureEngineer(self.all_data, self.config)
        
    def create_home_run_dataset(self) -> pd.DataFrame:
        """Create dataset specifically for home run predictions"""
        logger.info("Creating home run dataset...")
        
        # Get all at-bats
        hr_abs = self.feature_engineer.at_bat_results[
            self.feature_engineer.at_bat_results['is_at_bat']
        ].copy()
        
        # Balance dataset - all HRs and sample of non-HRs
        hr_yes = hr_abs[hr_abs['is_home_run']]
        hr_no = hr_abs[~hr_abs['is_home_run']].sample(
            n=min(len(hr_yes) * 10, 100000), 
            random_state=42
        )
        
        balanced_data = pd.concat([hr_yes, hr_no])
        logger.info(f"Dataset: {len(hr_yes):,} HRs, {len(hr_no):,} non-HRs")
        
        # Get game metadata
        game_metadata = {}
        for _, row in self.all_data['game_metadata'].iterrows():
            game_metadata[row['game_pk']] = {
                'temperature': row['temperature'] if pd.notna(row['temperature']) else 72,
                'wind_speed': row['wind_speed'] if pd.notna(row['wind_speed']) else 5,
                'wind_direction': row['wind_direction'] if pd.notna(row['wind_direction']) else 'In',
                'venue': row['venue'] if pd.notna(row['venue']) else 'Unknown',
                'start_hour': 19  # Simplified
            }
        
        features_list = []
        
        for _, ab in tqdm(balanced_data.iterrows(), total=len(balanced_data), desc="Creating HR features"):
            try:
                # Get game metadata
                game_meta = game_metadata.get(ab['game_pk'], {})
                
                # Create batter features
                batter_features = self.feature_engineer.create_batter_features(
                    int(ab['batter']), 
                    str(ab['game_date']), 
                    int(ab['pitcher'])
                )
                
                # Create pitcher features
                pitcher_features = self.feature_engineer.create_pitcher_features(
                    int(ab['pitcher']), 
                    str(ab['game_date'])
                )
                
                # Create game features
                game_features = self.feature_engineer.create_game_features({
                    'game_date': ab['game_date'],
                    **game_meta
                })
                
                # Advanced features
                advanced_pitcher_features = self.feature_engineer.create_advanced_pitcher_features(
                    int(ab['pitcher']), 
                    str(ab['game_date']), 
                    ab['stand']
                )
                
                similarity_features = self.feature_engineer.create_similarity_based_matchup_features(
                    int(ab['batter']), 
                    int(ab['pitcher']), 
                    str(ab['game_date'])
                )
                
                defense_features = self.feature_engineer.create_team_defense_features(
                    ab.get('home_team', 'UNK'), 
                    str(ab['game_date']), 
                    int(ab['batter'])
                )
                
                volatility_features = self.feature_engineer.create_player_volatility_features(
                    int(ab['batter']), 
                    'batter', 
                    str(ab['game_date'])
                )
                
                distribution_features = self.feature_engineer.create_performance_distribution_features(
                    int(ab['batter']), 
                    'batter', 
                    str(ab['game_date'])
                )
                
                # Combine all features
                features = {
                    **batter_features,
                    **game_features,

                }
                
                features['target'] = int(ab['is_home_run'])
                features['game_date'] = ab['game_date']
                features_list.append(features)
                
            except Exception as e:
                logger.debug(f"Error creating features: {e}")
                continue
        
        return pd.DataFrame(features_list)
    
    def train_home_run_model(self, dataset: pd.DataFrame):
        """Train only the home run model"""
        logger.info("Training home run model...")
        
        # Split data
        train_end = pd.to_datetime(self.config['training']['train_end_date'])
        val_end = pd.to_datetime(self.config['training']['val_end_date'])
        
        # Clean data
        dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataset.dropna(subset=['target'], inplace=True)
        dataset.fillna(dataset.median(numeric_only=True), inplace=True)
        
        # Create splits
        train_mask = dataset['game_date'] <= train_end
        val_mask = (dataset['game_date'] > train_end) & (dataset['game_date'] <= val_end)
        
        feature_cols = [col for col in dataset.columns if col not in ['target', 'game_date']]
        
        X_train = dataset.loc[train_mask, feature_cols]
        y_train = dataset.loc[train_mask, 'target']
        X_val = dataset.loc[val_mask, feature_cols]
        y_val = dataset.loc[val_mask, 'target']
        
        logger.info(f"Training samples: {len(X_train):,}, Validation samples: {len(X_val):,}")
        logger.info(f"HR rate - Train: {y_train.mean():.3%}, Val: {y_val.mean():.3%}")
        
        # Train model
        self.models.train_home_run_model(X_train, y_train, X_val, y_val)
        
        # Save model
        save_path = Path(self.config['paths']['models']).parent / 'home_run_model.pkl'
        import joblib
        joblib.dump({
            'models': {'home_run': self.models.models.get('home_run')},
            'scalers': {'home_run': self.models.scalers.get('home_run')},
            'feature_columns': {'home_run': list(X_train.columns)}
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")
        
    def load_home_run_model(self):
        """Load only the home run model"""
        save_path = Path(self.config['paths']['models']).parent / 'home_run_model.pkl'
        import joblib
        saved = joblib.load(save_path)
        
        self.models.models['home_run'] = saved['models']['home_run']
        self.models.scalers['home_run'] = saved['scalers']['home_run']
        
        logger.info("Home run model loaded")


class HomeRunBacktester(MLBBacktester):
    """Specialized backtester for home run predictions only"""
    
    def __init__(self, pipeline: HomeRunPipeline, start_date: str, end_date: str = None):
        # Initialize parent class
        super().__init__(pipeline, start_date, end_date)
        
        # Override betting parameters for home runs
        self.betting_params.update({
            'min_edge': 0.05,
            'kelly_fraction': 0.25,
            'max_bet_pct': 0.05
        })
    
    def run_home_run_backtest(self):
        """Run backtest specifically for home runs"""
        print("\n" + "="*80)
        print("HOME RUN MODEL BACKTESTING")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print("="*80 + "\n")
        
        # 1. Generate predictions
        print("1. GENERATING HOME RUN PREDICTIONS...")
        self._process_home_run_predictions_only()
        
        # 2. Calculate performance metrics
        print("\n2. CALCULATING PERFORMANCE METRICS...")
        self._calculate_home_run_metrics()
        
        # 3. Run betting simulation
        print("\n3. RUNNING BETTING SIMULATION...")
        self._simulate_home_run_betting()
        
        # 4. Generate visualizations
        print("\n4. GENERATING VISUALIZATIONS...")
        self._generate_home_run_visualizations()
        
        # 5. Generate report
        print("\n5. GENERATING REPORT...")
        self._generate_home_run_report()
        
        print("\n" + "="*80)
        print("HOME RUN BACKTESTING COMPLETE!")
        print("="*80)
        
        return self.performance_metrics.get('home_run', {}), self.betting_results.get('home_run', {})
    
    def _process_home_run_predictions_only(self):
        """Process only home run predictions"""
        # Get games in backtest period
        all_games = self.pipeline.feature_engineer.at_bat_results[
            (self.pipeline.feature_engineer.at_bat_results['game_date'] >= self.start_date) &
            (self.pipeline.feature_engineer.at_bat_results['game_date'] <= self.end_date)
        ]
        
        unique_games = all_games[['game_pk', 'game_date']].drop_duplicates()
        
        # Group by date
        daily_games = defaultdict(list)
        for _, game in unique_games.iterrows():
            daily_games[game['game_date'].date()].append(game['game_pk'])
        
        # Process each day
        for date, game_pks in tqdm(sorted(daily_games.items()), desc="Processing dates"):
            self._process_home_run_predictions(date, game_pks)
    
    def _calculate_home_run_metrics(self):
        """Calculate home run specific metrics"""
        if not self.predictions['home_run']:
            logger.warning("No home run predictions to evaluate")
            return
        
        self._calculate_classification_metrics('home_run')
        self._calculate_probability_calibration('home_run')
        
        # Additional home run specific metrics
        predictions = np.array(self.predictions['home_run'])
        actuals = np.array(self.actuals['home_run'])
        
        # Calculate metrics by probability buckets
        prob_buckets = {
            'very_low': (0, 0.02),
            'low': (0.02, 0.05),
            'medium': (0.05, 0.10),
            'high': (0.10, 0.20),
            'very_high': (0.20, 1.0)
        }
        
        bucket_analysis = {}
        for bucket_name, (low, high) in prob_buckets.items():
            mask = (predictions >= low) & (predictions < high)
            if np.sum(mask) > 0:
                bucket_analysis[bucket_name] = {
                    'count': int(np.sum(mask)),
                    'actual_hr_rate': float(np.mean(actuals[mask])),
                    'avg_prediction': float(np.mean(predictions[mask]))
                }
        
        self.performance_metrics['home_run']['bucket_analysis'] = bucket_analysis
    
    def _simulate_home_run_betting(self):
        """Simulate home run betting with realistic odds"""
        results = {
            'daily_results': [],
            'bets': []
        }
        
        bankroll = self.betting_params['bankroll']
        
        # Group by date
        daily_data = defaultdict(list)
        for i, meta in enumerate(self.metadata['home_run']):
            daily_data[meta['date']].append({
                'probability': self.predictions['home_run'][i],
                'actual': self.actuals['home_run'][i],
                'metadata': meta
            })
        
        for date in sorted(daily_data.keys()):
            day_bets = []
            day_profit = 0
            
            for data in daily_data[date]:
                # Home run odds typically range from +200 to +800
                # Use dynamic odds based on probability
                if data['probability'] < 0.05:
                    decimal_odds = 8.0  # +700
                    implied_prob = 0.125
                elif data['probability'] < 0.10:
                    decimal_odds = 6.0  # +500
                    implied_prob = 0.167
                elif data['probability'] < 0.15:
                    decimal_odds = 4.5  # +350
                    implied_prob = 0.222
                else:
                    decimal_odds = 3.0  # +200
                    implied_prob = 0.333
                
                our_prob = data['probability']
                edge = our_prob - implied_prob
                
                if edge > self.betting_params['min_edge']:
                    # Calculate Kelly bet
                    kelly_fraction = (our_prob * (decimal_odds - 1) - (1 - our_prob)) / (decimal_odds - 1)
                    kelly_fraction *= self.betting_params['kelly_fraction']
                    
                    bet_size = bankroll * kelly_fraction
                    bet_size = max(
                        self.betting_params['min_bet'],
                        min(bet_size, self.betting_params['max_bet'], 
                            bankroll * self.betting_params['max_bet_pct'])
                    )
                    
                    won = bool(data['actual'])
                    profit = bet_size * (decimal_odds - 1) if won else -bet_size
                    
                    day_bets.append({
                        'type': 'Home Run',
                        'bet_size': bet_size,
                        'won': won,
                        'profit': profit,
                        'edge': edge,
                        'our_prob': our_prob,
                        'implied_prob': implied_prob,
                        'odds': decimal_odds,
                        'player': data['metadata']['batter']
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
            
            # Breakdown by odds ranges
            odds_breakdown = defaultdict(lambda: {'bets': 0, 'wins': 0, 'profit': 0})
            for bet in results['bets']:
                if bet['odds'] >= 7.0:
                    key = '+600+'
                elif bet['odds'] >= 5.0:
                    key = '+400 to +600'
                elif bet['odds'] >= 3.5:
                    key = '+250 to +400'
                else:
                    key = 'Under +250'
                
                odds_breakdown[key]['bets'] += 1
                odds_breakdown[key]['wins'] += int(bet['won'])
                odds_breakdown[key]['profit'] += bet['profit']
            
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
                'odds_breakdown': dict(odds_breakdown),
                'sharpe_ratio': self._calculate_sharpe_ratio(results['daily_results'])
            }
        
        self.betting_results['home_run'] = results
    
    def _generate_home_run_visualizations(self):
        """Generate visualizations specific to home run analysis"""
        output_dir = Path('backtest_results/home_run')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.predictions['home_run']:
            logger.warning("No predictions to visualize")
            return
        
        # 1. ROC Curve with confidence intervals
        self._create_roc_curve_with_ci(output_dir)
        
        # 2. Calibration plot
        self._create_calibration_plot(output_dir)
        
        # 3. Betting performance analysis
        self._create_betting_analysis(output_dir)
        
        # 4. Player performance analysis
        self._create_player_analysis(output_dir)
        
        # 5. Feature importance
        self._create_feature_importance_plot(output_dir)
    
    def _create_roc_curve_with_ci(self, output_dir: Path):
        """Create ROC curve with confidence intervals"""
        from sklearn.utils import resample
        
        y_true = np.array(self.actuals['home_run'])
        y_scores = np.array(self.predictions['home_run'])
        
        # Calculate main ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        # Bootstrap confidence intervals
        n_bootstraps = 100
        rng_seed = 42
        bootstrapped_aucs = []
        
        for i in range(n_bootstraps):
            indices = resample(range(len(y_true)), random_state=rng_seed + i)
            if len(np.unique(y_true[indices])) < 2:
                continue
            
            auc_boot = roc_auc_score(y_true[indices], y_scores[indices])
            bootstrapped_aucs.append(auc_boot)
        
        auc_lower = np.percentile(bootstrapped_aucs, 2.5)
        auc_upper = np.percentile(bootstrapped_aucs, 97.5)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, 'b-', label=f'ROC curve (AUC = {auc:.3f} [{auc_lower:.3f}-{auc_upper:.3f}])')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.fill_between(fpr, tpr - 0.05, tpr + 0.05, alpha=0.2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Home Run Model ROC Curve with 95% CI')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.savefig(output_dir / 'roc_curve_with_ci.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_calibration_plot(self, output_dir: Path):
        """Create detailed calibration plot"""
        # Check if the correct key exists before proceeding
        if 'home_run_calibration' not in self.performance_metrics:
            logger.warning("Calibration data not found for home run model. Skipping plot.")
            return

        # Access the entire calibration dictionary from the correct top-level key
        calibration_metrics = self.performance_metrics['home_run_calibration']
        cal_data = calibration_metrics.get('calibration_data', [])
        ece_score = calibration_metrics.get('ece', 0.0)
        
        if not cal_data:
            logger.warning("Calibration data is empty. Skipping plot.")
            return

        df = pd.DataFrame(cal_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calibration plot
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax1.scatter(df['predicted_prob'], df['actual_prob'], 
                s=df['count']/df['count'].max()*500 + 50,
                alpha=0.6, edgecolors='black', linewidth=1)
        
        # Add error bars
        for _, row in df.iterrows():
            # Binomial confidence interval
            n = row['count']
            p = row['actual_prob']
            se = np.sqrt(p * (1 - p) / n) if n > 0 else 0
            ax1.plot([row['predicted_prob'], row['predicted_prob']], 
                    [p - 1.96*se, p + 1.96*se], 'gray', alpha=0.5)
        
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        # Use the correctly retrieved ECE score for the title
        ax1.set_title(f'Calibration Plot (ECE = {ece_score:.4f})')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # Histogram of predictions
        ax2.hist(self.predictions['home_run'], bins=50, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Predicted Probabilities')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'calibration_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_betting_analysis(self, output_dir: Path):
        """Create comprehensive betting analysis"""
        if 'home_run' not in self.betting_results or not self.betting_results['home_run']['bets']:
            return
        
        results = self.betting_results['home_run']
        bets_df = pd.DataFrame(results['bets'])
        daily_df = pd.DataFrame(results['daily_results'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Bankroll evolution with drawdowns
        ax1 = axes[0, 0]
        ax1.plot(daily_df['date'], daily_df['bankroll'], 'b-', linewidth=2)
        ax1.axhline(y=self.betting_params['bankroll'], color='red', linestyle='--', alpha=0.5)
        
        # Mark drawdowns
        daily_df['peak'] = daily_df['bankroll'].cummax()
        daily_df['drawdown'] = (daily_df['bankroll'] - daily_df['peak']) / daily_df['peak']
        
        dd_periods = daily_df[daily_df['drawdown'] < -0.05]
        for _, period in dd_periods.iterrows():
            ax1.axvspan(period['date'] - timedelta(days=1), 
                       period['date'] + timedelta(days=1), 
                       alpha=0.3, color='red')
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Bankroll ($)')
        ax1.set_title('Bankroll Evolution with Drawdown Periods')
        ax1.grid(True, alpha=0.3)
        
        # 2. Win rate by edge bucket
        ax2 = axes[0, 1]
        bets_df['edge_bucket'] = pd.cut(bets_df['edge'], bins=10)
        edge_analysis = bets_df.groupby('edge_bucket').agg({
            'won': ['mean', 'count'],
            'profit': 'sum'
        })
        
        edge_centers = [interval.mid for interval in edge_analysis.index]
        ax2.bar(range(len(edge_centers)), edge_analysis[('won', 'mean')], alpha=0.7)
        ax2.set_xticks(range(len(edge_centers)))
        ax2.set_xticklabels([f'{c:.1%}' for c in edge_centers], rotation=45)
        ax2.set_xlabel('Edge')
        ax2.set_ylabel('Win Rate')
        ax2.set_title('Win Rate by Edge Bucket')
        ax2.grid(True, alpha=0.3)
        
        # Add sample size on top of bars
        for i, (_, row) in enumerate(edge_analysis.iterrows()):
            ax2.text(i, row[('won', 'mean')] + 0.01, 
                    f"n={row[('won', 'count')]}", 
                    ha='center', va='bottom', fontsize=8)
        
        # 3. ROI by odds range
        ax3 = axes[1, 0]
        if 'odds_breakdown' in results['summary']:
            odds_data = results['summary']['odds_breakdown']
            odds_ranges = list(odds_data.keys())
            rois = [data['profit'] / (data['bets'] * 100) if data['bets'] > 0 else 0 
                   for data in odds_data.values()]
            
            bars = ax3.bar(odds_ranges, rois, alpha=0.7)
            ax3.set_xlabel('Odds Range')
            ax3.set_ylabel('ROI')
            ax3.set_title('ROI by Odds Range')
            ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax3.grid(True, alpha=0.3)
            
            # Color bars based on positive/negative
            for bar, roi in zip(bars, rois):
                bar.set_color('green' if roi > 0 else 'red')
        
        # 4. Monthly performance
        ax4 = axes[1, 1]
        daily_df['month'] = pd.to_datetime(daily_df['date']).dt.to_period('M')
        monthly_perf = daily_df.groupby('month').agg({
            'profit': 'sum',
            'num_bets': 'sum'
        })
        
        months = monthly_perf.index.astype(str)
        profits = monthly_perf['profit'].values
        
        bars = ax4.bar(months, profits, alpha=0.7)
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Profit ($)')
        ax4.set_title('Monthly Profit/Loss')
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3)
        
        # Color bars
        for bar, profit in zip(bars, profits):
            bar.set_color('green' if profit > 0 else 'red')
        
        # Add bet counts
        for i, (month, row) in enumerate(monthly_perf.iterrows()):
            ax4.text(i, row['profit'] + 50 if row['profit'] > 0 else row['profit'] - 50,
                    f"{row['num_bets']} bets",
                    ha='center', va='bottom' if row['profit'] > 0 else 'top',
                    fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'betting_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_player_analysis(self, output_dir: Path):
        """Analyze performance by player"""
        # Group predictions by player
        player_performance = defaultdict(lambda: {'predictions': [], 'actuals': []})
        
        for i, meta in enumerate(self.metadata['home_run']):
            player_id = meta['batter']
            player_performance[player_id]['predictions'].append(self.predictions['home_run'][i])
            player_performance[player_id]['actuals'].append(self.actuals['home_run'][i])
        
        # Calculate metrics for players with enough data
        player_metrics = []
        for player_id, data in player_performance.items():
            if len(data['predictions']) >= 20:  # Minimum sample size
                predictions = np.array(data['predictions'])
                actuals = np.array(data['actuals'])
                
                player_metrics.append({
                    'player_id': player_id,
                    'n_predictions': len(predictions),
                    'actual_hr_rate': np.mean(actuals),
                    'predicted_hr_rate': np.mean(predictions),
                    'calibration_error': abs(np.mean(predictions) - np.mean(actuals)),
                    'brier_score': brier_score_loss(actuals, predictions)
                })
        
        if not player_metrics:
            return
        
        df = pd.DataFrame(player_metrics)
        df = df.sort_values('n_predictions', ascending=False).head(50)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calibration by player
        ax1 = axes[0]
        ax1.scatter(df['predicted_hr_rate'], df['actual_hr_rate'], 
                   s=df['n_predictions'], alpha=0.6)
        ax1.plot([0, 0.15], [0, 0.15], 'k--', alpha=0.5)
        ax1.set_xlabel('Mean Predicted HR Rate')
        ax1.set_ylabel('Actual HR Rate')
        ax1.set_title('Player-Level Calibration (size = # predictions)')
        ax1.grid(True, alpha=0.3)
        
        # Brier score distribution
        ax2 = axes[1]
        ax2.hist(df['brier_score'], bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(x=df['brier_score'].mean(), color='red', linestyle='--', 
                   label=f'Mean = {df["brier_score"].mean():.4f}')
        ax2.set_xlabel('Brier Score')
        ax2.set_ylabel('Number of Players')
        ax2.set_title('Distribution of Player-Level Brier Scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'player_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_feature_importance_plot(self, output_dir: Path):
        """Create feature importance visualization"""
        model = self.pipeline.models.models.get('home_run')
        if not model or not hasattr(model, 'estimator'):
            return
        
        # Get feature importances
        if hasattr(model.estimator, 'feature_importances_'):
            importances = model.estimator.feature_importances_
            feature_names = list(self.pipeline.models.scalers['home_run'].feature_names_in_)
            
            # Create dataframe and get top features
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(30)
            
            # Create horizontal bar plot
            plt.figure(figsize=(10, 12))
            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.xlabel('Importance')
            plt.title('Top 30 Feature Importances - Home Run Model')
            plt.tight_layout()
            
            plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_home_run_report(self):
        """Generate comprehensive report for home run model"""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("HOME RUN MODEL BACKTEST REPORT")
        report_lines.append(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        report_lines.append("="*80)
        
        if 'home_run' in self.performance_metrics:
            metrics = self.performance_metrics['home_run']
            
            # Safely get the ECE from the correct top-level location
            calibration_metrics = self.performance_metrics.get('home_run_calibration', {})
            ece_score = calibration_metrics.get('ece', 0.0)
            
            report_lines.append("\n1. MODEL PERFORMANCE METRICS")
            report_lines.append("-"*40)
            report_lines.append(f"AUC: {metrics.get('auc', 0):.4f}")
            report_lines.append(f"Brier Score: {metrics.get('brier_score', 0):.4f}")
            report_lines.append(f"Log Loss: {metrics.get('log_loss', 0):.4f}")
            # Use the correctly retrieved ECE score
            report_lines.append(f"ECE (Calibration): {ece_score:.4f}")
            report_lines.append(f"Optimal Threshold: {metrics.get('optimal_threshold', 0):.3f}")
            report_lines.append(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
            report_lines.append(f"Precision: {metrics.get('precision', 0):.4f}")
            report_lines.append(f"Recall: {metrics.get('recall', 0):.4f}")
            report_lines.append(f"F1 Score: {metrics.get('f1_score', 0):.4f}")
            report_lines.append(f"Sample Size: {metrics.get('sample_size', 0):,}")
            report_lines.append(f"Actual HR Rate: {metrics.get('positive_rate', 0):.3%}")
            
            # Probability bucket analysis
            if 'bucket_analysis' in metrics:
                report_lines.append("\nProbability Bucket Analysis:")
                for bucket, stats in metrics['bucket_analysis'].items():
                    report_lines.append(f"  {bucket}: {stats['count']} predictions, "
                                      f"actual HR rate = {stats['actual_hr_rate']:.3%}")
        
        # Betting Performance
        if 'home_run' in self.betting_results and 'summary' in self.betting_results['home_run']:
            summary = self.betting_results['home_run']['summary']
            
            report_lines.append("\n2. BETTING SIMULATION RESULTS")
            report_lines.append("-"*40)
            report_lines.append(f"Total Bets: {summary['total_bets']:,}")
            report_lines.append(f"Total Wagered: ${summary['total_wagered']:,.2f}")
            report_lines.append(f"Total Profit: ${summary['total_profit']:,.2f}")
            report_lines.append(f"ROI: {summary['roi']*100:.2f}%")
            report_lines.append(f"Win Rate: {summary['win_rate']*100:.2f}%")
            report_lines.append(f"Final Bankroll: ${summary['final_bankroll']:,.2f}")
            report_lines.append(f"Bankroll Growth: {summary['bankroll_growth']*100:.2f}%")
            report_lines.append(f"Average Edge: {summary['avg_edge']*100:.2f}%")
            report_lines.append(f"Average Odds: +{int((summary['avg_odds']-1)*100)}")
            report_lines.append(f"Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}")
            
            # Odds breakdown
            if 'odds_breakdown' in summary:
                report_lines.append("\nPerformance by Odds Range:")
                for odds_range, stats in summary['odds_breakdown'].items():
                    win_rate = stats['wins'] / stats['bets'] if stats['bets'] > 0 else 0
                    roi = stats['profit'] / (stats['bets'] * 100) if stats['bets'] > 0 else 0
                    report_lines.append(f"  {odds_range}: {stats['bets']} bets, "
                                      f"{win_rate*100:.1f}% win rate, "
                                      f"${stats['profit']:.2f} profit ({roi*100:.1f}% ROI)")
        
        # Risk Analysis
        if 'home_run' in self.betting_results and 'daily_results' in self.betting_results['home_run']:
            daily_df = pd.DataFrame(self.betting_results['home_run']['daily_results'])
            
            if len(daily_df) > 0:
                # Calculate max drawdown
                daily_df['peak'] = daily_df['bankroll'].cummax()
                daily_df['drawdown'] = (daily_df['bankroll'] - daily_df['peak']) / daily_df['peak']
                max_drawdown = daily_df['drawdown'].min()
                
                # Longest losing streak
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
                
                report_lines.append("\n3. RISK ANALYSIS")
                report_lines.append("-"*40)
                report_lines.append(f"Maximum Drawdown: {max_drawdown*100:.2f}%")
                report_lines.append(f"Longest Losing Streak: {max_losing_streak} days")
                report_lines.append(f"Profitable Days: {(daily_df['profit'] > 0).sum()} / {len(daily_df)} "
                                  f"({(daily_df['profit'] > 0).mean()*100:.1f}%)")
        
        # Key Insights
        report_lines.append("\n4. KEY INSIGHTS")
        report_lines.append("-"*40)
        
        if 'home_run' in self.performance_metrics:
            auc = self.performance_metrics['home_run'].get('auc', 0)
            if auc > 0.7:
                report_lines.append("- Model shows strong predictive power (AUC > 0.7)")
            elif auc > 0.6:
                report_lines.append("- Model shows moderate predictive power (AUC 0.6-0.7)")
            else:
                report_lines.append("- Model shows limited predictive power (AUC < 0.6)")
            
            # --- THIS IS THE FIX ---
            # Use the 'ece_score' variable you already defined, instead of the path that causes the error.
            calibration_metrics = self.performance_metrics.get('home_run_calibration', {})
            ece_score = calibration_metrics.get('ece', 0.0)
            # --- END FIX ---

            if ece_score < 0.05:
                report_lines.append("- Model is well-calibrated (ECE < 0.05)")
            else:
                report_lines.append(f"- Model calibration could be improved (ECE = {ece_score:.3f})")
        
        if 'home_run' in self.betting_results and 'summary' in self.betting_results['home_run']:
            roi = self.betting_results['home_run']['summary']['roi']
            if roi > 0:
                report_lines.append(f"- Betting strategy is profitable with {roi*100:.1f}% ROI")
            else:
                report_lines.append("- Betting strategy is currently unprofitable")
        
        # Save report
        output_dir = Path('backtest_results/home_run')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / 'backtest_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Also save detailed results
        if self.predictions['home_run']:
            results_df = pd.DataFrame({
                'date': [m['date'] for m in self.metadata['home_run']],
                'prediction': self.predictions['home_run'],
                'actual': self.actuals['home_run'],
                'batter': [m['batter'] for m in self.metadata['home_run']],
                'pitcher': [m['pitcher'] for m in self.metadata['home_run']]
            })
            results_df.to_csv(output_dir / 'predictions.csv', index=False)
        
        if 'home_run' in self.betting_results and self.betting_results['home_run']['bets']:
            bets_df = pd.DataFrame(self.betting_results['home_run']['bets'])
            bets_df.to_csv(output_dir / 'betting_log.csv', index=False)
        
        # Print report
        print('\n'.join(report_lines))
        
        # Save summary JSON
        summary_data = {
            'period': {
                'start': str(self.start_date.date()),
                'end': str(self.end_date.date())
            },
            'performance_metrics': self.performance_metrics.get('home_run', {}),
            'betting_summary': self.betting_results.get('home_run', {}).get('summary', {})
        }
        
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)


def main():
    """Main execution function"""
    import sys
    
    # Initialize pipeline
    pipeline = HomeRunPipeline()
    
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        # Training mode
        print("\n" + "="*80)
        print("TRAINING HOME RUN MODEL")
        print("="*80)
        
        # Load data
        pipeline.load_data_for_home_runs('2020-01-01', '2024-12-31')
        
        # Create dataset
        dataset = pipeline.create_home_run_dataset()
        
        # Train model
        pipeline.train_home_run_model(dataset)
        
        print("\nTraining complete!")
        
    else:
        # Backtesting mode
        start_date = sys.argv[1] if len(sys.argv) > 1 else '2024-01-01'
        end_date = sys.argv[2] if len(sys.argv) > 2 else None
        
        print("\n" + "="*80)
        print("HOME RUN MODEL BACKTESTING")
        print("="*80)
        
        # Load data for backtesting
        data_start = (pd.to_datetime(start_date) - timedelta(days=365)).strftime('%Y-%m-%d')
        data_end = end_date or datetime.now().strftime('%Y-%m-%d')
        
        print(f"\nLoading data from {data_start} to {data_end}...")
        pipeline.load_data_for_home_runs(data_start, data_end)
        
        # Load model
        print("Loading trained model...")
        pipeline.load_home_run_model()
        
        # Run backtest
        backtester = HomeRunBacktester(pipeline, start_date, end_date)
        metrics, betting_summary = backtester.run_home_run_backtest()
        
        print("\nBacktest complete! Check backtest_results/home_run/ for detailed results.")


if __name__ == "__main__":
    main()