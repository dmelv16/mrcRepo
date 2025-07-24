# nrfi_backtester.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from tqdm import tqdm
from collections import defaultdict

# Import from your main model file
from mlbPlayerPropv1 import OptimizedMLBPipeline, OptimizedFeatureEngineer

# Import metrics
from sklearn.metrics import (
    roc_auc_score, 
    brier_score_loss, 
    log_loss, 
    confusion_matrix
)

warnings.filterwarnings('ignore')

class NRFIBacktester:
    """A focused backtesting system specifically for NRFI predictions."""
    
    def __init__(self, pipeline, start_date: str = '2024-01-01', end_date: str = None):
        """Initialize the NRFI backtester."""
        self.pipeline = pipeline
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()
        
        # --- SIMPLIFIED: No longer need defaultdict for multiple models ---
        self.predictions = []
        self.actuals = []
        self.metadata = []
        
        # Betting simulation parameters
        self.betting_params = {
            'bankroll': 1000,
            'kelly_fraction': 0.5,
            'min_edge': 0.175,
            'max_bet_pct': 0.05,
            'min_bet': 10,
            'max_bet': 500
        }
        
        # Results storage
        self.performance_metrics = {}
        self.betting_results = {}

    def run_full_backtest(self):
        """Run the complete NRFI backtesting analysis."""
        print("\n" + "="*80)
        print("NRFI MODEL BACKTESTING SYSTEM")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print("="*80 + "\n")
        
        print("1. GENERATING NRFI PREDICTIONS...")
        self._generate_historical_predictions()
        
        print("\n2. CALCULATING NRFI PERFORMANCE METRICS...")
        self._calculate_performance_metrics()
        
        print("\n3. RUNNING NRFI BETTING SIMULATION...")
        self._simulate_nrfi_betting()
        
        print("\n4. GENERATING NRFI REPORT...")
        self._generate_report()
        
        print("\n" + "="*80)
        print("NRFI BACKTEST COMPLETE!")
        print("="*80)
        
        return self.performance_metrics, self.betting_results

    def _generate_historical_predictions(self):
        """Generate NRFI predictions for each day in the backtest period."""
        # --- SIMPLIFIED: Combined logic from original script ---
        all_games = self.pipeline.feature_engineer.at_bat_results[
            (self.pipeline.feature_engineer.at_bat_results['game_date'] >= self.start_date) &
            (self.pipeline.feature_engineer.at_bat_results['game_date'] <= self.end_date)
        ]
        unique_games = all_games[['game_pk', 'game_date']].drop_duplicates()
        
        # Process each game
        for _, game in tqdm(unique_games.iterrows(), total=len(unique_games), desc="Processing Games for NRFI"):
            date = game['game_date'].date()
            game_pk = game['game_pk']
            try:
                # Get first inning data
                first_inning = self.pipeline.feature_engineer.at_bat_results[
                    (self.pipeline.feature_engineer.at_bat_results['game_pk'] == game_pk) &
                    (self.pipeline.feature_engineer.at_bat_results['inning'] == 1)
                ]
                
                if len(first_inning) == 0:
                    continue
                
                run_scored = (first_inning['post_bat_score'] > first_inning['bat_score']).any()
                
                game_info = self._get_game_info(game_pk)
                game_info['game_date'] = date
                
                starting_pitchers = self._get_starting_pitchers(game_pk)
                if not starting_pitchers:
                    continue
                
                game_info.update(starting_pitchers)
                lineups = self._get_lineups(game_pk)
                
                features = self.pipeline.feature_engineer.create_nrfi_features(game_info, lineups)
                
                features_df = pd.DataFrame([features])
                prob = self.pipeline.models.predict_nrfi(features_df)[0]
                
                # Store results
                self.predictions.append(prob)
                self.actuals.append(int(not run_scored)) # 1 if NRFI, 0 if run scored
                self.metadata.append({
                    'date': date,
                    'game_pk': game_pk,
                })
                
            except Exception as e:
                print(f"Error processing NRFI for game {game_pk}: {e}")

    def _calculate_performance_metrics(self):
        """Calculate classification metrics for the NRFI model."""
        # --- SIMPLIFIED: Directly calculates metrics for NRFI ---
        probabilities = np.array(self.predictions)
        actuals = np.array(self.actuals)
        
        # Use a fixed threshold or find an optimal one
        threshold = 0.5 
        predictions = (probabilities > threshold).astype(int)
        
        self.performance_metrics = {
            'auc': roc_auc_score(actuals, probabilities),
            'brier_score': brier_score_loss(actuals, probabilities),
            'log_loss': log_loss(actuals, probabilities),
            'accuracy': np.mean(predictions == actuals),
            'sample_size': len(probabilities),
            'positive_rate': np.mean(actuals) # Actual NRFI rate
        }

    def _simulate_nrfi_betting(self):
        """Simulate betting on NRFI props."""
        # --- SIMPLIFIED: Hardcoded for NRFI betting ---
        results = {'daily_results': [], 'bets': []}
        bankroll = self.betting_params['bankroll']
        
        daily_data = defaultdict(list)
        for i, meta in enumerate(self.metadata):
            daily_data[meta['date']].append({
                'probability': self.predictions[i],
                'actual': self.actuals[i],
            })
        
        for date in sorted(daily_data.keys()):
            day_bets, day_profit = [], 0
            
            for data in daily_data[date]:
                implied_prob = 0.565  # Assumed -130 odds
                decimal_odds = 1.77
                
                our_prob = data['probability']
                edge = our_prob - implied_prob
                
                if edge > self.betting_params['min_edge']:
                    bet_size = self._calculate_kelly_bet(bankroll, edge, decimal_odds - 1)
                    if bet_size >= self.betting_params['min_bet']:
                        won = bool(data['actual'])
                        profit = bet_size * (decimal_odds - 1) if won else -bet_size
                        
                        day_bets.append({'bet_size': bet_size, 'won': won, 'profit': profit, 'edge': edge})
                        day_profit += profit
            
            if day_bets:
                bankroll += day_profit
                results['daily_results'].append({
                    'date': date,
                    'num_bets': len(day_bets),
                    'profit': day_profit,
                    'bankroll': bankroll,
                })
                results['bets'].extend(day_bets)
        
        if results['bets']:
            total_bet = sum(bet['bet_size'] for bet in results['bets'])
            total_profit = sum(bet['profit'] for bet in results['bets'])
            
            results['summary'] = {
                'total_bets': len(results['bets']),
                'total_wagered': total_bet,
                'total_profit': total_profit,
                'roi': total_profit / total_bet if total_bet > 0 else 0,
                'win_rate': sum(1 for bet in results['bets'] if bet['won']) / len(results['bets']),
                'final_bankroll': bankroll,
                'sharpe_ratio': self._calculate_sharpe_ratio(results['daily_results'])
            }
        
        self.betting_results = results

    def _generate_report(self):
        """Generate a simple text report for NRFI results."""
        # --- SIMPLIFIED: NRFI-only report ---
        print("\n1. NRFI MODEL PERFORMANCE")
        print("-"*40)
        for key, val in self.performance_metrics.items():
            print(f"  {key.replace('_', ' ').title()}: {val:.4f}")

        print("\n2. NRFI BETTING SIMULATION RESULTS")
        print("-"*40)
        if 'summary' in self.betting_results:
            summary = self.betting_results['summary']
            print(f"  Total Bets: {summary['total_bets']:,}")
            print(f"  Total Wagered: ${summary['total_wagered']:,.2f}")
            print(f"  Total Profit: ${summary['total_profit']:,.2f}")
            print(f"  ROI: {summary['roi']*100:.2f}%")
            print(f"  Win Rate: {summary['win_rate']*100:.2f}%")
            print(f"  Final Bankroll: ${summary['final_bankroll']:,.2f}")
            print(f"  Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}")

    # --- HELPER METHODS: Copy these directly from your original script ---
    # You need to copy the following methods into this class without any changes:
    # - _get_game_info(self, game_pk: int)
    # - _get_starting_pitchers(self, game_pk: int)
    # - _get_lineups(self, game_pk: int)
    # - _calculate_kelly_bet(self, bankroll: float, edge: float, odds: float)
    # - _calculate_sharpe_ratio(self, daily_results)
    
    def _get_game_info(self, game_pk: int):
        game_meta = self.pipeline.all_data['game_metadata']
        game_row = game_meta[game_meta['game_pk'] == game_pk]
        if len(game_row) == 0: return {'venue': 'Unknown'}
        return {'venue': game_row.iloc[0]['venue']}

    def _get_starting_pitchers(self, game_pk: int):
        sp = self.pipeline.all_data['batting_orders']
        sp = sp[(sp['game_pk'] == game_pk) & (sp['is_starting_pitcher'] == 1)]
        if len(sp) < 2: return None
        home_pitcher = sp[sp['team_type'] == 'home'].iloc[0]['player_id']
        away_pitcher = sp[sp['team_type'] == 'away'].iloc[0]['player_id']
        return {'home_pitcher_id': home_pitcher, 'away_pitcher_id': away_pitcher}

    def _get_lineups(self, game_pk: int):
        bo = self.pipeline.all_data['batting_orders']
        lineups = {}
        for team in ['home', 'away']:
            lineups[f'{team}_lineup'] = bo[(bo['game_pk'] == game_pk) & (bo['team_type'] == team)]['player_id'].tolist()
        return lineups

    def _calculate_kelly_bet(self, bankroll: float, edge: float, odds: float) -> float:
        if edge <= 0: return 0
        p = edge + (1 / (odds + 1))
        q = 1 - p
        kelly_fraction = (p * odds - q) / odds
        kelly_fraction *= self.betting_params['kelly_fraction']
        bet_size = bankroll * kelly_fraction
        max_bet = min(bankroll * self.betting_params['max_bet_pct'], self.betting_params['max_bet'])
        return round(max(0, min(bet_size, max_bet)), 2)
        
    # CORRECTED FUNCTION
    def _calculate_sharpe_ratio(self, daily_results):
        if len(daily_results) < 2: return 0
        
        # Create a DataFrame from the daily results
        daily_df = pd.DataFrame(daily_results)
        
        # Calculate daily returns based on the bankroll, not the profit
        # The return for a given day is (profit / bankroll_at_start_of_day)
        # Bankroll at start = current bankroll - today's profit
        daily_df['daily_return'] = daily_df['profit'] / (daily_df['bankroll'] - daily_df['profit'])

        # Drop non-finite values and handle days with zero bets
        daily_df.replace([np.inf, -np.inf], 0, inplace=True)
        daily_df.fillna(0, inplace=True)

        if daily_df['daily_return'].std() == 0: return 0

        # Annualize the Sharpe Ratio (assuming ~252 betting days in a year)
        sharpe = (daily_df['daily_return'].mean() * np.sqrt(252)) / daily_df['daily_return'].std()
        return sharpe

def run_nrfi_backtest(start_date: str = '2024-01-01', end_date: str = None):
    """Main function to run the NRFI backtest."""

    # Load the trained pipeline
    print("Loading trained models and data...")
    pipeline = OptimizedMLBPipeline()

    # Load historical data needed for feature generation
    data_start = (pd.to_datetime(start_date) - timedelta(days=365)).strftime('%Y-%m-%d')
    data_end = end_date or datetime.now().strftime('%Y-%m-%d')

    pipeline.all_data = pipeline.db.load_all_data_bulk(data_start, data_end)
    pipeline.feature_engineer = OptimizedFeatureEngineer(pipeline.all_data, pipeline.config)

    # Load the trained models
    print("Loading NRFI model...")
    pipeline.load_models() # This will load all models, but we'll only use nrfi

    # Initialize and run the NRFI backtester
    backtester = NRFIBacktester(pipeline, start_date, end_date)
    performance_metrics, betting_results = backtester.run_full_backtest()

    return backtester, performance_metrics, betting_results

if __name__ == "__main__":
    # You can customize the date range here
    run_nrfi_backtest(start_date='2024-01-01', end_date='2025-07-19')
