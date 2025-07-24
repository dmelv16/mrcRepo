"""
MLB Reinforcement Learning Betting Agent V3
Peak performance version with refined rewards and enhanced parlay logic
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import optuna
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class BettingOpportunity:
    """Enhanced data class for a single betting opportunity"""
    game_id: str
    date: str
    bet_type: str  # 'moneyline' or 'total'
    selection: str  # 'home', 'away', 'over', 'under'
    probability: float
    edge: float
    confidence: float
    uncertainty: float
    kelly_stake: float
    odds: float
    won: Optional[bool] = None
    actual_result: Optional[Dict] = None
    
    # Additional metadata
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    
    def __post_init__(self):
        """Calculate derived attributes"""
        # Calculate payout multiplier from odds
        if self.odds > 0:
            self.payout_multiplier = self.odds / 100
        else:
            self.payout_multiplier = 100 / abs(self.odds)
    
    @property
    def expected_value(self) -> float:
        """Calculate expected value of the bet"""
        return self.probability * self.payout_multiplier - (1 - self.probability)


class MLBBettingEnvironmentV3(gym.Env):
    """
    V3 Environment with pure sparse rewards and enhanced parlay validation
    """
    
    def __init__(self, 
                 daily_opportunities: Dict[str, List[BettingOpportunity]], 
                 initial_bankroll: float = 1000,
                 min_bet_size: float = 10,
                 max_bet_fraction: float = 0.3,
                 use_kelly_sizing: bool = True,
                 sparse_rewards_only: bool = True):
        super().__init__()
        
        self.daily_opportunities = daily_opportunities
        self.dates = sorted(self.daily_opportunities.keys())
        self.initial_bankroll = initial_bankroll
        self.min_bet_size = min_bet_size
        self.max_bet_fraction = max_bet_fraction
        self.use_kelly_sizing = use_kelly_sizing
        self.sparse_rewards_only = sparse_rewards_only
        
        # Action space: 0 = No Bet, 1 = Place Bet, 2 = Add to Parlay
        self.action_space = spaces.Discrete(3)
        
        # Enhanced observation space
        # [probability, edge, confidence, uncertainty, kelly_stake, expected_value,
        #  normalized_bankroll, bets_today_normalized, win_rate_recent, 
        #  bet_type_encoded, selection_encoded, parlay_pool_size_normalized]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )
        
        # Tracking variables
        self.episode_history = []
        self.parlay_pool = []
        self.daily_performance = []
        
    def reset(self, seed=None, options=None):
        """Reset environment to start of season"""
        super().reset(seed=seed)
        
        self.bankroll = self.initial_bankroll
        self.current_date_idx = 0
        self.current_bet_idx = 0
        self.daily_pnl = []
        self.bets_placed_today = 0
        self.recent_results = deque(maxlen=20)
        self.parlay_pool = []
        self.episode_history = []
        self.daily_performance = []
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Enhanced observation with additional context"""
        if self.current_date_idx >= len(self.dates):
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        current_date = self.dates[self.current_date_idx]
        opportunities = self.daily_opportunities[current_date]
        
        if self.current_bet_idx >= len(opportunities):
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        bet = opportunities[self.current_bet_idx]
        
        # Calculate recent win rate
        if len(self.recent_results) > 0:
            recent_win_rate = sum(self.recent_results) / len(self.recent_results)
        else:
            recent_win_rate = 0.5
        
        # Encode bet type and selection
        bet_type_encoding = 0.5 if bet.bet_type == 'moneyline' else -0.5
        
        selection_map = {'home': 0.5, 'away': -0.5, 'over': 0.25, 'under': -0.25}
        selection_encoding = selection_map.get(bet.selection, 0)
        
        # Normalize expected value
        normalized_ev = np.clip(bet.expected_value / 0.1, -1, 1)  # Normalize by 10% EV
        
        # Construct observation vector
        obs = np.array([
            (bet.probability - 0.5) * 2,  # Center around 0
            np.clip(bet.edge * 10, -1, 1),  # Scale edge
            (bet.confidence - 0.5) * 2,  # Center confidence
            np.clip(bet.uncertainty, 0, 1),  # Uncertainty is already 0-1
            np.clip(bet.kelly_stake * 20, 0, 1),  # Kelly usually < 0.05
            normalized_ev,  # Expected value
            np.clip((self.bankroll / self.initial_bankroll) - 1, -1, 1),  # Normalized bankroll
            np.clip(self.bets_placed_today / 10, 0, 1),  # Normalized bets today
            (recent_win_rate - 0.5) * 2,  # Recent performance
            bet_type_encoding,  # Bet type indicator
            selection_encoding,  # Selection indicator
            np.clip(len(self.parlay_pool) / 3, 0, 1)  # Parlay pool size
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action):
        """Execute action with sparse rewards only"""
        current_date = self.dates[self.current_date_idx]
        opportunities = self.daily_opportunities[current_date]
        
        if self.current_bet_idx >= len(opportunities):
            # End of day - process any parlays and move to next day
            return self._end_of_day()
        
        bet = opportunities[self.current_bet_idx]
        info = {'action': action, 'bet': bet.__dict__}
        
        # Process action
        if action == 1:  # Place straight bet
            pnl = self._place_bet(bet)
            self.daily_pnl.append(pnl)
            self.bets_placed_today += 1
            info['bet_placed'] = True
            info['pnl'] = pnl
            
        elif action == 2:  # Add to parlay pool
            # Only add if it's a valid parlay candidate
            if self._is_valid_parlay_leg(bet):
                self.parlay_pool.append(bet)
                info['added_to_parlay'] = True
            else:
                info['parlay_rejected'] = True
            info['parlay_pool_size'] = len(self.parlay_pool)
        
        # Move to next opportunity
        self.current_bet_idx += 1
        
        # Check if day is complete
        if self.current_bet_idx >= len(opportunities):
            return self._end_of_day()
        
        # No intermediate rewards in sparse mode
        reward = 0.0
        
        done = False
        truncated = False
        
        return self._get_observation(), reward, done, truncated, info
    
    def _place_bet(self, bet: BettingOpportunity) -> float:
        """Place a bet and return PnL"""
        # Determine bet size
        if self.use_kelly_sizing:
            bet_fraction = bet.kelly_stake
        else:
            # Dynamic sizing based on confidence and edge
            bet_fraction = min(-0.5, max(0.01, bet.edge * bet.confidence))
        
        bet_size = self.bankroll * min(bet_fraction, self.max_bet_fraction)
        
        # Ensure minimum bet size
        if bet_size < self.min_bet_size:
            bet_size = 0
        
        # Ensure we don't bet more than bankroll
        if bet_size > self.bankroll:
            bet_size = 0
        
        if bet_size == 0:
            return 0
        
        # Calculate PnL
        if bet.won:
            pnl = bet_size * bet.payout_multiplier
        else:
            pnl = -bet_size
        
        self.bankroll += pnl
        self.recent_results.append(1 if bet.won else 0)
        
        # Track for analysis
        self.episode_history.append({
            'date': bet.date,
            'game_id': bet.game_id,
            'bet_type': bet.bet_type,
            'selection': bet.selection,
            'probability': bet.probability,
            'edge': bet.edge,
            'confidence': bet.confidence,
            'bet_size': bet_size,
            'pnl': pnl,
            'won': bet.won,
            'bankroll_after': self.bankroll
        })
        
        return pnl
    
    def _is_valid_parlay_leg(self, bet: BettingOpportunity) -> bool:
        """Check if bet is valid for parlay pool"""
        # Don't add if we already have a bet from this game
        game_ids_in_pool = {leg.game_id for leg in self.parlay_pool}
        if bet.game_id in game_ids_in_pool:
            return False
        
        # Don't add if pool is full
        if len(self.parlay_pool) >= 3:
            return False
        
        # Only add bets with decent edge and confidence
        if bet.edge < -0.5 or bet.confidence < 0.35:
            return False
        
        return True
    
    def _process_parlay(self) -> float:
        """Process parlay with enhanced validation"""
        if len(self.parlay_pool) < 2:
            self.parlay_pool = []
            return 0
        
        # Ensure all legs are from different games (double-check)
        unique_game_ids = set()
        valid_legs = []
        
        for leg in self.parlay_pool:
            if leg.game_id not in unique_game_ids:
                valid_legs.append(leg)
                unique_game_ids.add(leg.game_id)
        
        if len(valid_legs) < 2:
            self.parlay_pool = []
            return 0
        
        # Take best 2-3 legs based on expected value
        valid_legs.sort(key=lambda x: x.expected_value, reverse=True)
        parlay_size = min(len(valid_legs), 3)
        parlay_legs = valid_legs[:parlay_size]
        
        # Calculate combined probability and odds
        combined_prob = 1.0
        combined_odds = 1.0
        
        for leg in parlay_legs:
            combined_prob *= leg.probability
            if leg.odds > 0:
                combined_odds *= (1 + leg.odds/100)
            else:
                combined_odds *= (1 + 100/abs(leg.odds))
        
        # Conservative parlay sizing
        parlay_edge = combined_prob * (combined_odds - 1) - (1 - combined_prob)
        parlay_kelly = max(0, parlay_edge / (combined_odds - 1))
        
        # Very conservative multiplier for parlays
        bet_size = self.bankroll * min(parlay_kelly * 0.1, 0.01)
        
        if bet_size < self.min_bet_size or bet_size > self.bankroll:
            self.parlay_pool = []
            return 0
        
        # Check if all legs won
        all_won = all(leg.won for leg in parlay_legs)
        
        if all_won:
            pnl = bet_size * (combined_odds - 1)
        else:
            pnl = -bet_size
        
        self.bankroll += pnl
        
        # Track parlay
        self.episode_history.append({
            'date': parlay_legs[0].date,
            'game_id': f"parlay_{len(self.episode_history)}",
            'bet_type': 'parlay',
            'selection': f"{len(parlay_legs)}_leg",
            'probability': combined_prob,
            'edge': parlay_edge,
            'confidence': np.mean([leg.confidence for leg in parlay_legs]),
            'bet_size': bet_size,
            'pnl': pnl,
            'won': all_won,
            'bankroll_after': self.bankroll,
            'parlay_legs': [leg.game_id for leg in parlay_legs]
        })
        
        self.parlay_pool = []
        return pnl
    
    def _end_of_day(self):
        """Process end of day with a final, stable, clipped Sharpe reward."""
        parlay_pnl = self._process_parlay()
        if parlay_pnl != 0:
            self.daily_pnl.append(parlay_pnl)
        
        daily_return = sum(self.daily_pnl) if self.daily_pnl else 0
        reward = 0.0

        if self.daily_pnl:
            pnl_array = np.array(self.daily_pnl)
            daily_mean = pnl_array.mean()
            daily_std = pnl_array.std()
            
            # --- START: FINAL REWARD STABILIZATION ---
            # Add a small constant (epsilon) to the standard deviation to ensure it's never zero.
            # This is the key to preventing division-by-zero errors.
            epsilon = 1e-6
            sharpe = daily_mean / (daily_std + epsilon)
            
            # Clip the reward to a stable range (e.g., -5 to +5).
            # This prevents the value from exploding and gives the optimizer a clearer signal.
            # A reward of +5 is treated as "excellent," just like a reward of +5 billion would be,
            # making the search more efficient.
            reward = np.clip(sharpe, -5.0, 5.0)
            # --- END: FINAL REWARD STABILIZATION ---

        else: # Penalty for not betting on a day with good opportunities
            current_date = self.dates[self.current_date_idx]
            opportunities = self.daily_opportunities.get(current_date, [])
            good_opportunities = sum(1 for opp in opportunities if opp.edge > 0.03 and opp.confidence > 0.6)
            if good_opportunities > 0:
                reward = -0.1 * min(good_opportunities / 5, 1.0)
        
        self.daily_performance.append({ 'date': self.dates[self.current_date_idx], 'return': daily_return, 'sharpe': reward, 'num_bets': self.bets_placed_today, 'bankroll': self.bankroll })
        self.daily_pnl, self.bets_placed_today, self.current_bet_idx = [], 0, 0
        self.current_date_idx += 1
        terminated = self.current_date_idx >= len(self.dates) or self.bankroll <= 100
        info = {'date': self.dates[self.current_date_idx - 1], 'bankroll': self.bankroll}
        return self._get_observation(), reward, terminated, False, info


class AdvancedCallbacks(BaseCallback):
    """Custom callbacks for monitoring training progress"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_info = []
        
    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get('dones')[0]:
            info = self.locals.get('infos')[0]
            
            # Log episode statistics
            episode_reward = info.get('episode', {}).get('r', 0)
            episode_length = info.get('episode', {}).get('l', 0)
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Log every 10 episodes
            if len(self.episode_rewards) % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                mean_length = np.mean(self.episode_lengths[-10:])
                
                self.logger.record('rollout/mean_episode_reward', mean_reward)
                self.logger.record('rollout/mean_episode_length', mean_length)
                
                if self.verbose > 0:
                    print(f"Episodes: {len(self.episode_rewards)}, "
                          f"Mean Reward: {mean_reward:.4f}, "
                          f"Mean Length: {mean_length:.0f}")
        
        return True


class EnhancedDataPreprocessor:
    """Enhanced preprocessor with additional validation"""
    
    @staticmethod
    def create_opportunities_from_predictions(predictions_df: pd.DataFrame,
                                            min_edge: float = -0.5,
                                            min_confidence: float = 0.3) -> Dict[str, List[BettingOpportunity]]:
        """
        Convert model predictions to betting opportunities with validation
        """
        daily_opportunities = defaultdict(list)
        
        # Validate required columns
        required_columns = ['date', 'game_id', 'home_score', 'away_score']
        missing_columns = [col for col in required_columns if col not in predictions_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        for _, row in predictions_df.iterrows():
            date = row['date']
            game_id = row['game_id']
            
            # Extract teams if available
            home_team = row.get('home_team', 'HOME')
            away_team = row.get('away_team', 'AWAY')
            
            # Actual results
            home_won = row['home_score'] > row['away_score']
            total = row['home_score'] + row['away_score']
            
            # Create moneyline opportunities
            for side in ['home', 'away']:
                edge = row.get(f'{side}_edge', 0)
                confidence = row.get(f'{side}_confidence', 0.5)
                
                # Apply filters
                if edge > min_edge and confidence > min_confidence:
                    opp = BettingOpportunity(
                        game_id=game_id,
                        date=date,
                        bet_type='moneyline',
                        selection=side,
                        probability=row.get(f'{side}_win_prob', 0.5),
                        edge=edge,
                        confidence=confidence,
                        uncertainty=row.get(f'{side}_uncertainty', 1),
                        kelly_stake=row.get(f'{side}_kelly', 0),
                        odds=row.get(f'{side}_ml_odds', -110),
                        won=(home_won if side == 'home' else not home_won),
                        home_team=home_team,
                        away_team=away_team
                    )
                    daily_opportunities[date].append(opp)
            
            # Create totals opportunities
            if pd.notna(row.get('total_line', np.nan)):
                for side in ['over', 'under']:
                    edge = row.get(f'{side}_edge', 0)
                    confidence = row.get(f'{side}_confidence', 0.5)
                    
                    if edge > min_edge and confidence > min_confidence:
                        over_won = total > row['total_line']
                        
                        opp = BettingOpportunity(
                            game_id=game_id,
                            date=date,
                            bet_type='total',
                            selection=side,
                            probability=row.get(f'{side}_prob', 0.5),
                            edge=edge,
                            confidence=confidence,
                            uncertainty=row.get('total_uncertainty', 1),
                            kelly_stake=row.get(f'{side}_kelly', 0),
                            odds=row.get(f'{side}_odds', -110),
                            won=(over_won if side == 'over' else not over_won),
                            home_team=home_team,
                            away_team=away_team
                        )
                        daily_opportunities[date].append(opp)
        
        # Log statistics
        total_opportunities = sum(len(opps) for opps in daily_opportunities.values())
        logging.info(f"Created {total_opportunities} betting opportunities across {len(daily_opportunities)} days")
        
        # Calculate opportunity statistics
        all_edges = [opp.edge for opps in daily_opportunities.values() for opp in opps]
        if all_edges:
            logging.info(f"Edge distribution: min={min(all_edges):.3f}, "
                        f"mean={np.mean(all_edges):.3f}, max={max(all_edges):.3f}")
        
        return dict(daily_opportunities)


class PortfolioAnalyzer:
    """Analyze portfolio composition and risk metrics"""
    
    # In bettingModelv4.py -> PortfolioAnalyzer

    @staticmethod
    def analyze_episode_performance(episode_history: List[Dict]) -> Dict:
        """Comprehensive analysis of a single episode"""
        if not episode_history:
            return {}
        
        df = pd.DataFrame(episode_history)
        
        # Basic metrics
        total_bets = len(df)
        wins = df['won'].sum()
        win_rate = wins / total_bets if total_bets > 0 else 0
        
        # Financial metrics
        total_wagered = df['bet_size'].sum()
        total_pnl = df['pnl'].sum()
        roi = (total_pnl / total_wagered * 100) if total_wagered > 0 else 0
        
        # Risk metrics
        daily_returns = df.groupby('date')['pnl'].sum()
        if len(daily_returns) > 1:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
            max_drawdown = PortfolioAnalyzer._calculate_max_drawdown(df['bankroll_after'].values)
        else:
            sharpe = 0
            max_drawdown = 0
        
        # --- START OF NEW CODE ---
        # Bet composition analysis to track what is being bet on
        
        # Create a combined type_selection column for detailed grouping
        df['type_selection'] = df['bet_type'] + '_' + df['selection']
        
        # Calculate ROI for each bet type
        bet_type_roi = df.groupby('type_selection').apply(
            lambda x: (x['pnl'].sum() / x['bet_size'].sum() * 100) if x['bet_size'].sum() > 0 else 0
        ).to_dict()
        
        # Calculate the count/frequency of each bet type
        bet_type_counts = df['type_selection'].value_counts().to_dict()
        bet_type_freq = df['type_selection'].value_counts(normalize=True).to_dict()
        
        # --- END OF NEW CODE ---
        
        # Edge analysis
        edge_buckets = pd.cut(df['edge'], bins=[-1, 0, 0.03, 0.05, 0.1, 1])
        bets_by_edge = {str(k): v for k, v in df.groupby(edge_buckets, observed=False).size().to_dict().items()}
        roi_by_edge = {str(k): v for k, v in df.groupby(edge_buckets, observed=False).apply(
            lambda x: (x['pnl'].sum() / x['bet_size'].sum() * 100) if x['bet_size'].sum() > 0 else 0
        ).to_dict().items()}
        
        # Confidence analysis
        conf_buckets = pd.cut(df['confidence'], bins=[0, 0.5, 0.6, 0.7, 0.8, 1])
        roi_by_confidence = {str(k): v for k, v in df.groupby(conf_buckets, observed=False).apply(
            lambda x: (x['pnl'].sum() / x['bet_size'].sum() * 100) if x['bet_size'].sum() > 0 else 0
        ).to_dict().items()}
        
        return {
            'total_bets': total_bets,
            'win_rate': win_rate,
            'total_wagered': total_wagered,
            'total_pnl': total_pnl,
            'roi': roi,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            # --- NEW: Add the bet tracking results to the summary ---
            'bet_type_roi': bet_type_roi,
            'bet_type_counts': bet_type_counts,
            'bet_type_frequency': bet_type_freq,
            # --- END OF NEW CODE ---
            'bets_by_edge': bets_by_edge,
            'roi_by_edge': roi_by_edge,
            'roi_by_confidence': roi_by_confidence,
            'avg_bet_size': df['bet_size'].mean(),
            'avg_bet_size_pct': (df['bet_size'] / df['bankroll_after'].shift(1).fillna(10000)).mean() * 100
        }
    
    @staticmethod
    def _calculate_max_drawdown(bankroll_values: np.ndarray) -> float:
        """Calculate maximum drawdown percentage"""
        peak = np.maximum.accumulate(bankroll_values)
        drawdown = (bankroll_values - peak) / peak
        return abs(drawdown.min()) * 100 if len(drawdown) > 0 else 0


# In bettingModelv4.py, replace the entire ModelEvaluator class with this:

class ModelEvaluator:
    """
    Comprehensive model evaluation using bootstrapping to get realistic variance.
    """
    def __init__(self, model, daily_opportunities: Dict[str, List[BettingOpportunity]]):
        self.model = model
        self.all_test_opportunities = daily_opportunities
        self.test_dates = sorted(self.all_test_opportunities.keys())
        
    def evaluate_comprehensive(self, n_episodes: int = 30) -> Dict:
        """Run comprehensive evaluation using bootstrapped seasons."""
        all_results = []
        
        for episode in tqdm(range(n_episodes), desc="Evaluating Agent"):
            # --- THIS IS THE FIX ---
            # Create a bootstrapped (randomly sampled with replacement) season
            bootstrapped_dates = np.random.choice(self.test_dates, size=len(self.test_dates), replace=True)
            
            # Create a temporary opportunities dict for this unique season
            bootstrapped_opportunities = {
                f"day_{i}": self.all_test_opportunities[date]
                for i, date in enumerate(bootstrapped_dates)
            }
            # --- END FIX ---
            
            # Run a single episode on this unique, simulated season
            env = MLBBettingEnvironmentV3(bootstrapped_opportunities)
            obs, _ = env.reset()
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
            
            # Analyze the results of this unique episode
            episode_analysis = PortfolioAnalyzer.analyze_episode_performance(env.episode_history)
            episode_analysis['final_bankroll'] = env.bankroll
            episode_analysis['total_return_pct'] = (env.bankroll - env.initial_bankroll) / env.initial_bankroll * 100
            
            all_results.append(episode_analysis)
        
        # Aggregate results across all unique episodes
        df = pd.DataFrame(all_results)
        
        summary = {
            'episodes': n_episodes,
            'mean_final_bankroll': df['final_bankroll'].mean(),
            'std_final_bankroll': df['final_bankroll'].std(), # This will now be non-zero!
            'mean_total_return_pct': df['total_return_pct'].mean(),
            'mean_roi': df['roi'].mean(),
            'mean_sharpe': df['sharpe_ratio'].mean(),
            'mean_win_rate': df['win_rate'].mean(),
            'mean_max_drawdown': df['max_drawdown'].mean(),
            'mean_bets_per_episode': df['total_bets'].mean(),
            'consistency_score': df['total_return_pct'].mean() / (df['total_return_pct'].std() + 1e-6)
        }
        
        for metric in ['final_bankroll', 'total_return_pct', 'roi']:
            summary[f'{metric}_p25'] = df[metric].quantile(0.25)
            summary[f'{metric}_p50'] = df[metric].quantile(0.50)
            summary[f'{metric}_p75'] = df[metric].quantile(0.75)
        
        return summary, df


def visualize_results(evaluation_df: pd.DataFrame, save_path: str = None):
    """Create comprehensive visualization of results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('RL Betting Agent Performance Analysis', fontsize=16)
    
    # 1. Final bankroll distribution
    axes[0, 0].hist(evaluation_df['final_bankroll'], bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 0].axvline(1000, color='red', linestyle='--', label='Initial Bankroll')
    axes[0, 0].set_xlabel('Final Bankroll ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Final Bankroll Distribution')
    axes[0, 0].legend()
    
    # 2. ROI distribution
    axes[0, 1].hist(evaluation_df['roi'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].axvline(0, color='red', linestyle='--', label='Break Even')
    axes[0, 1].set_xlabel('ROI (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Return on Investment Distribution')
    axes[0, 1].legend()
    
    # 3. Sharpe ratio vs ROI
    axes[0, 2].scatter(evaluation_df['roi'], evaluation_df['sharpe_ratio'], alpha=0.6)
    axes[0, 2].set_xlabel('ROI (%)')
    axes[0, 2].set_ylabel('Sharpe Ratio')
    axes[0, 2].set_title('Risk-Adjusted Returns')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Win rate vs number of bets
    axes[1, 0].scatter(evaluation_df['total_bets'], evaluation_df['win_rate'] * 100, alpha=0.6)
    axes[1, 0].set_xlabel('Total Bets')
    axes[1, 0].set_ylabel('Win Rate (%)')
    axes[1, 0].set_title('Betting Activity vs Success Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Max drawdown distribution
    axes[1, 1].hist(evaluation_df['max_drawdown'], bins=20, alpha=0.7, color='red', edgecolor='black')
    axes[1, 1].set_xlabel('Max Drawdown (%)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Maximum Drawdown Distribution')
    
    # 6. Performance consistency
    returns = evaluation_df['total_return_pct'].values
    axes[1, 2].boxplot(returns, vert=True)
    axes[1, 2].set_ylabel('Total Return (%)')
    axes[1, 2].set_title('Return Consistency')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def main_training_pipeline_v3(predictions_path: str,
                             optimize_hyperparams: bool = True,
                             n_optimization_trials: int = 55,
                             total_training_steps: int = 1000000,
                             n_eval_episodes: int = 100):
    """
    Complete V3 training pipeline with sparse rewards
    """
    
    logging.info("="*60)
    logging.info("MLB RL BETTING AGENT V3 - TRAINING PIPELINE")
    logging.info("="*60)
    
    # Load and preprocess data
    logging.info("Loading model predictions...")
    predictions_df = pd.read_pickle(predictions_path)
    
    # Create opportunities with validation
    preprocessor = EnhancedDataPreprocessor()
    daily_opportunities = preprocessor.create_opportunities_from_predictions(
        predictions_df,
        min_edge=-0.5,  # Allow slightly negative edges for learning
        min_confidence=0.3
    )
    
    # Split data
    all_dates = sorted(daily_opportunities.keys())
    n_dates = len(all_dates)
    train_end = int(n_dates * 0.7)
    val_end = int(n_dates * 0.85)
    
    train_dates = all_dates[:train_end]
    val_dates = all_dates[train_end:val_end]
    test_dates = all_dates[val_end:]
    
    train_opps = {d: daily_opportunities[d] for d in train_dates}
    val_opps = {d: daily_opportunities[d] for d in val_dates}
    test_opps = {d: daily_opportunities[d] for d in test_dates}
    
    logging.info(f"Data split: Train={len(train_dates)} days, Val={len(val_dates)} days, Test={len(test_dates)} days")
    
    # Hyperparameter optimization
# In your main_training_pipeline_v3 function, replace the whole "if" block with this:

    if optimize_hyperparams:
        logging.info("\nRunning hyperparameter optimization...")

        # --- THIS IS THE FIX ---
        # Combine the train and validation sets for a more robust hyperparameter search.
        # This prevents the optimizer from overfitting to the small validation set.
        optimization_opps = {**train_opps, **val_opps}
        logging.info(f"Using {len(optimization_opps)} days of data for hyperparameter search.")
        # --- END FIX ---
        
        def objective(trial):
            # Sample hyperparameters
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
            gamma = trial.suggest_float('gamma', 0.95, 0.999)
            gae_lambda = trial.suggest_float('gae_lambda', 0.9, 0.99)
            ent_coef = trial.suggest_float('ent_coef', 0.0001, 0.05, log=True)
            n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096])
            clip_range = trial.suggest_float('clip_range', 0.1, 0.3)
            
            # --- FIX: Use the combined dataset for the environment ---
            # Now, the agent trains and is evaluated on the same large, diverse dataset
            # during each optimization trial.
            env = Monitor(MLBBettingEnvironmentV3(optimization_opps, sparse_rewards_only=True))
            env = DummyVecEnv([lambda: env])
            
            # Train model
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=64,
                n_epochs=10,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                verbose=0,
                device="auto"
            )
            
            # Use fewer steps per trial to keep optimization fast
            model.learn(total_timesteps=500000) 
            
            # Evaluate on the same large, combined set.
            mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)
            
            return mean_reward
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_optimization_trials)
        
        best_params = study.best_params
        logging.info(f"Best hyperparameters found: {best_params}")
    else:
        best_params = {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'ent_coef': 0.01,
            'n_steps': 2048,
            'clip_range': 0.2
        }
    
    # Train final model
    logging.info("\nTraining final model with best hyperparameters...")
    
    # Combine train and validation for final training
    final_train_opps = {**train_opps, **val_opps}
    
    # Create environment with sparse rewards
    env = MLBBettingEnvironmentV3(final_train_opps, sparse_rewards_only=True)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=best_params.get('learning_rate', 3e-4),
        n_steps=best_params.get('n_steps', 2048),
        batch_size=64,
        n_epochs=10,
        gamma=best_params.get('gamma', 0.99),
        gae_lambda=best_params.get('gae_lambda', 0.95),
        clip_range=best_params.get('clip_range', 0.2),
        ent_coef=best_params.get('ent_coef', 0.01),
        verbose=1,
        tensorboard_log="./betting_agent_v3_tensorboard/",
        device="auto"
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path='./betting_agent_v3_checkpoints/',
        name_prefix='betting_model_v3'
    )
    
    custom_callback = AdvancedCallbacks(verbose=1)
    
    # Train
    model.learn(
        total_timesteps=total_training_steps,
        callback=[checkpoint_callback, custom_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save("trained_betting_agent_v3_final")
    logging.info("Model saved to 'trained_betting_agent_v3_final'")
    
    # Comprehensive evaluation
    logging.info(f"\nEvaluating on test set ({len(test_dates)} days)...")
    evaluator = ModelEvaluator(model, test_opps)
    summary, eval_df = evaluator.evaluate_comprehensive(n_episodes=n_eval_episodes)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS (TEST SET)")
    print("="*60)
    
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Visualize results
    fig = visualize_results(eval_df, save_path='betting_agent_v3_performance.png')
    logging.info("Performance visualization saved to 'betting_agent_v3_performance.png'")
    
# In bettingModelv4.py, in the main_training_pipeline_v3 function,
# replace the entire "Analyze learned strategy" block with this:

    # Analyze learned strategy
    logging.info("\nAnalyzing learned betting strategy...")
    
    # --- THIS IS THE FIX ---
    # Create a single, flat list of ALL opportunities from the test set
    all_test_opps_flat = [
        opp for date in test_opps for opp in test_opps[date]
    ]

    if all_test_opps_flat:
        decision_log = []
        # Create a temporary, minimal environment to get observations
        temp_env = MLBBettingEnvironmentV3({'day_1': all_test_opps_flat})
        obs, _ = temp_env.reset()
        
        # Loop through every single opportunity and record the agent's decision
        for i, opp in enumerate(tqdm(all_test_opps_flat, desc="Analyzing Strategy")):
            # Get the observation for the current opportunity
            temp_env.current_bet_idx = i
            current_obs = temp_env._get_observation()
            
            # Get the agent's action
            action, _ = model.predict(current_obs, deterministic=True)
            
            # --- THIS IS THE FINAL FIX ---
            # Use int() which is more robust than indexing
            decision_log.append({
                'edge': opp.edge,
                'confidence': opp.confidence,
                'action': int(action), # Change action[0] to int(action)
            })
            
        decision_df = pd.DataFrame(decision_log)
        
        print("\n" + "="*60)
        print("LEARNED STRATEGY ANALYSIS (ON FULL TEST SET)")
        print("="*60)
        
        # Betting frequency by edge
        edge_bins = pd.cut(decision_df['edge'], bins=[-1, 0, 0.03, 0.05, 0.1, 1])
        # Add observed=False to silence the FutureWarning
        bet_freq_by_edge = decision_df.groupby(edge_bins, observed=False)['action'].apply(lambda x: (x > 0).mean())
        
        print("\nBetting Frequency by Edge:")
        print(bet_freq_by_edge.to_string(float_format='{:.2%}'.format))
        
        # Betting frequency by confidence
        conf_bins = pd.cut(decision_df['confidence'], bins=[0, 0.5, 0.6, 0.7, 0.8, 1])
        bet_freq_by_conf = decision_df.groupby(conf_bins, observed=False)['action'].apply(lambda x: (x > 0).mean())
        
        print("\nBetting Frequency by Confidence:")
        print(bet_freq_by_conf.to_string(float_format='{:.2%}'.format))
        
        # Overall action distribution
        decision_df['decision'] = decision_df['action'].map({0: 'No Bet', 1: 'Bet', 2: 'Parlay'})
        action_dist = decision_df['decision'].value_counts(normalize=True)
        print("\nOverall Action Distribution:")
        print(action_dist.to_string(float_format='{:.2%}'.format))
    # --- END FIX ---

    return model, summary, eval_df

if __name__ == "__main__":
    # Example usage
    model, results, evaluation_df = main_training_pipeline_v3(
        predictions_path="model_predictions.pkl"
    )