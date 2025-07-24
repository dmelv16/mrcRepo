"""
MLB Reinforcement Learning Betting Agent V4
Enhanced version with Sortino ratio, hybrid action space, and advanced features
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


class MLBBettingEnvironmentV4(gym.Env):
    """
    V4 Environment with Sortino rewards, hybrid action space, and multi-objective optimization
    """
    
    def __init__(self, 
                 daily_opportunities: Dict[str, List[BettingOpportunity]], 
                 initial_bankroll: float = 10000,
                 min_bet_size: float = 10,
                 max_bet_fraction: float = 0.15,
                 use_kelly_sizing: bool = True,
                 use_sortino: bool = True,
                 use_multi_objective: bool = False):
        super().__init__()
        
        self.daily_opportunities = daily_opportunities
        self.dates = sorted(self.daily_opportunities.keys())
        self.initial_bankroll = initial_bankroll
        self.min_bet_size = min_bet_size
        self.max_bet_fraction = max_bet_fraction
        self.use_kelly_sizing = use_kelly_sizing
        self.use_sortino = use_sortino
        self.use_multi_objective = use_multi_objective
        
# [action_choice, size_choice]
        # action_choice has 3 options (0, 1, 2)
        # size_choice has 10 options (0-9 representing 10% to 100%)
        self.action_space = spaces.MultiDiscrete([3, 10])
        
        # Enhanced observation space with additional features
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(17,), dtype=np.float32
        )
        
        # Tracking variables
        self.episode_history = []
        self.parlay_pool = []
        self.daily_performance = []
        self.recent_bets = deque(maxlen=20)
        self.bankroll_history = deque(maxlen=50)
        
    def reset(self, seed=None, options=None):
        """Reset environment to start of season"""
        super().reset(seed=seed)
        
        self.bankroll = self.initial_bankroll
        self.current_date_idx = 0
        self.current_bet_idx = 0
        self.daily_pnl = []
        self.bets_placed_today = 0
        self.recent_results = deque(maxlen=20)
        self.recent_bets = deque(maxlen=20)
        self.bankroll_history = deque(maxlen=50)
        self.parlay_pool = []
        self.episode_history = []
        self.daily_performance = []
        self.parlay_attempts = 0
        self.parlay_wins = 0
        self.bet_type_counts = defaultdict(int)
        
        return self._get_enhanced_observation(), {}
    
    def _get_enhanced_observation(self):
        """Enhanced observation with advanced features"""
        if self.current_date_idx >= len(self.dates):
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        current_date = self.dates[self.current_date_idx]
        opportunities = self.daily_opportunities[current_date]
        
        if self.current_bet_idx >= len(opportunities):
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        bet = opportunities[self.current_bet_idx]
        
        # Base features
        recent_win_rate = sum(self.recent_results) / len(self.recent_results) if self.recent_results else 0.5
        bet_type_encoding = 0.5 if bet.bet_type == 'moneyline' else -0.5
        selection_map = {'home': 0.5, 'away': -0.5, 'over': 0.25, 'under': -0.25}
        selection_encoding = selection_map.get(bet.selection, 0)
        normalized_ev = np.clip(bet.expected_value / 0.1, -1, 1)
        
        # Advanced features
        
        # 1. Edge momentum
        if len(self.recent_bets) >= 3:
            recent_edges = [b.edge for b in list(self.recent_bets)[-10:]]
            if len(recent_edges) > 1:
                edge_momentum = np.polyfit(range(len(recent_edges)), recent_edges, 1)[0]
            else:
                edge_momentum = 0
        else:
            edge_momentum = 0
        
# In bettingModelv5.py -> MLBBettingEnvironmentV4 -> _get_enhanced_observation

        # 2. Conditional win rate
        similar_bets = [b for b in self.episode_history 
                       if abs(b.get('edge', 0) - bet.edge) < 0.01]
        
        # --- START FIX ---
        # Filter for only resolved bets (where 'won' is not None)
        resolved_outcomes = [b['won'] for b in similar_bets if b.get('won') is not None]
        conditional_win_rate = np.mean(resolved_outcomes) if resolved_outcomes else 0.5
        # --- END FIX ---
        
        # 3. Market efficiency
        market_efficiency = 1.0 - abs(bet.probability - 0.5)
        
        # 4. Bankroll trend
        if len(self.bankroll_history) >= 5:
            recent_bankrolls = list(self.bankroll_history)[-5:]
            bankroll_trend = np.polyfit(range(len(recent_bankrolls)), recent_bankrolls, 1)[0]
            normalized_trend = np.tanh(bankroll_trend / 1000)
        else:
            normalized_trend = 0
        
        # 5. Kelly divergence
        kelly_divergence = self._calculate_kelly_divergence(bet)
        
        # Construct observation vector
        obs = np.array([
            # Original features
            (bet.probability - 0.5) * 2,
            np.clip(bet.edge * 10, -1, 1),
            (bet.confidence - 0.5) * 2,
            np.clip(bet.uncertainty, 0, 1),
            np.clip(bet.kelly_stake * 20, 0, 1),
            normalized_ev,
            np.clip((self.bankroll / self.initial_bankroll) - 1, -1, 1),
            np.clip(self.bets_placed_today / 10, 0, 1),
            (recent_win_rate - 0.5) * 2,
            bet_type_encoding,
            selection_encoding,
            np.clip(len(self.parlay_pool) / 3, 0, 1),
            # New advanced features
            np.clip(edge_momentum * 10, -1, 1),
            (conditional_win_rate - 0.5) * 2,
            market_efficiency,
            normalized_trend,
            kelly_divergence
        ], dtype=np.float32)
        
        return obs
    
    def _calculate_kelly_divergence(self, bet: BettingOpportunity) -> float:
        """Calculate how far current sizing is from optimal Kelly"""
        if self.bankroll <= 0:
            return 0
        
        current_fraction = self.min_bet_size / self.bankroll
        optimal_fraction = bet.kelly_stake
        
        if optimal_fraction > 0:
            divergence = abs(current_fraction - optimal_fraction) / optimal_fraction
        else:
            divergence = 0
        
        return np.clip(divergence, -1, 1)
    
    def step(self, action):
        """Execute action with hybrid action space"""
        current_date = self.dates[self.current_date_idx]
        opportunities = self.daily_opportunities[current_date]
        
        if self.current_bet_idx >= len(opportunities):
            return self._end_of_day()
        
        bet = opportunities[self.current_bet_idx]
        # action is now a numpy array like [action_choice, size_choice]
        action_type = action[0]
        size_choice = action[1]

        # Convert the size_choice (0-9) to a multiplier (0.1-1.0)
        size_multiplier = (size_choice + 1) / 10.0
        
        info = {
            'action_type': action_type,
            'size_multiplier': size_multiplier,
            'bet': bet.__dict__
        }
        
        # Process action
        if action_type == 1:  # Place straight bet
            pnl = self._place_bet(bet, size_multiplier)
            if pnl != 0:  # Only count if bet was actually placed
                self.daily_pnl.append(pnl)
                self.bets_placed_today += 1
                self.bet_type_counts[f"{bet.bet_type}_{bet.selection}"] += 1
                info['bet_placed'] = True
                info['pnl'] = pnl
        
        elif action_type == 2:  # Add to parlay pool
            if self._is_valid_parlay_leg(bet):
                self.parlay_pool.append((bet, size_multiplier))
                info['added_to_parlay'] = True
            else:
                info['parlay_rejected'] = True
            info['parlay_pool_size'] = len(self.parlay_pool)
        
        # Track bet for advanced features
        self.recent_bets.append(bet)
        
        # Move to next opportunity
        self.current_bet_idx += 1
        
        # Check if day is complete
        if self.current_bet_idx >= len(opportunities):
            return self._end_of_day()
        
        # No intermediate rewards
        reward = 0.0
        done = False
        truncated = False
        
        return self._get_enhanced_observation(), reward, done, truncated, info
    
    def _place_bet(self, bet: BettingOpportunity, size_multiplier: float = 1.0) -> float:
        """Place a bet with agent-controlled sizing"""
        if self.use_kelly_sizing:
            base_fraction = bet.kelly_stake
        else:
            base_fraction = min(0.03, max(0.01, bet.edge * bet.confidence))
        
        # Apply the agent's size multiplier
        bet_fraction = base_fraction * size_multiplier
        bet_size = self.bankroll * min(bet_fraction, self.max_bet_fraction)
        
        # Ensure minimum bet size
        if bet_size < self.min_bet_size:
            return 0
        
        # Ensure we don't bet more than bankroll
        if bet_size > self.bankroll:
            return 0
        
        # Calculate PnL
        if bet.won:
            pnl = bet_size * bet.payout_multiplier
        else:
            pnl = -bet_size
        
        self.bankroll += pnl
        self.recent_results.append(1 if bet.won else 0)
        self.bankroll_history.append(self.bankroll)
        
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
            'size_multiplier': size_multiplier,
            'pnl': pnl,
            'won': bet.won,
            'bankroll_after': self.bankroll
        })
        
        return pnl
    
    def _is_valid_parlay_leg(self, bet: BettingOpportunity) -> bool:
        """Check if bet is valid for parlay pool"""
        game_ids_in_pool = {leg[0].game_id for leg in self.parlay_pool}
        if bet.game_id in game_ids_in_pool:
            return False
        
        if len(self.parlay_pool) >= 3:
            return False
        
        if bet.edge < 0.02 or bet.confidence < 0.55:
            return False
        
        return True
    
    def _process_parlay(self) -> float:
        """Process parlay with size multipliers"""
        if len(self.parlay_pool) < 2:
            self.parlay_pool = []
            return 0
        
        # Sort by expected value and take best legs
        self.parlay_pool.sort(key=lambda x: x[0].expected_value, reverse=True)
        parlay_legs = self.parlay_pool[:3]
        
        # Calculate combined metrics
        combined_prob = 1.0
        combined_odds = 1.0
        avg_size_multiplier = np.mean([leg[1] for leg in parlay_legs])
        
        for leg, _ in parlay_legs:
            combined_prob *= leg.probability
            if leg.odds > 0:
                combined_odds *= (1 + leg.odds/100)
            else:
                combined_odds *= (1 + 100/abs(leg.odds))
        
        # Calculate parlay sizing
        parlay_edge = combined_prob * (combined_odds - 1) - (1 - combined_prob)
        parlay_kelly = max(0, parlay_edge / (combined_odds - 1))
        
        # Apply size multiplier to parlay
        bet_size = self.bankroll * min(parlay_kelly * 0.1 * avg_size_multiplier, 0.01)
        
        if bet_size < self.min_bet_size or bet_size > self.bankroll:
            self.parlay_pool = []
            return 0
        
        # Check if all legs won
        all_won = all(leg[0].won for leg in parlay_legs)
        self.parlay_attempts += 1
        
        if all_won:
            pnl = bet_size * (combined_odds - 1)
            self.parlay_wins += 1
        else:
            pnl = -bet_size
        
        self.bankroll += pnl
        self.bankroll_history.append(self.bankroll)
        
        # Track parlay
        self.episode_history.append({
            'date': parlay_legs[0][0].date,
            'game_id': f"parlay_{len(self.episode_history)}",
            'bet_type': 'parlay',
            'selection': f"{len(parlay_legs)}_leg",
            'probability': combined_prob,
            'edge': parlay_edge,
            'confidence': np.mean([leg[0].confidence for leg in parlay_legs]),
            'bet_size': bet_size,
            'avg_size_multiplier': avg_size_multiplier,
            'pnl': pnl,
            'won': all_won,
            'bankroll_after': self.bankroll,
            'parlay_legs': [leg[0].game_id for leg in parlay_legs]
        })
        
        self.parlay_pool = []
        return pnl
    
    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio (only penalizes downside volatility)"""
        if len(returns) == 0:
            return 0
        
        mean_return = returns.mean()
        
        # Only consider negative returns for downside deviation
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) > 0:
            downside_std = negative_returns.std()
        else:
            # If no losses, use a small value (this is good!)
            downside_std = 0.01
        
        epsilon = 1e-6
        sortino = mean_return / (downside_std + epsilon)
        
        return sortino
    
    def _calculate_multi_objective_reward(self) -> float:
        """Calculate reward considering multiple objectives"""
        # Get daily returns
        daily_returns = np.array(self.daily_pnl) if self.daily_pnl else np.array([0])
        
        # Objective 1: Risk-adjusted returns (Sortino)
        sortino = self._calculate_sortino(daily_returns)
        
        # Objective 2: Parlay efficiency
        parlay_success_rate = self.parlay_wins / self.parlay_attempts if self.parlay_attempts > 0 else 0
        parlay_reward = parlay_success_rate * 2.0 if parlay_success_rate > 0.3 else -0.5
        
        # Objective 3: Portfolio diversity
        bet_diversity = len(self.bet_type_counts)
        diversity_reward = np.log(1 + bet_diversity) * 0.5
        
        # Weighted combination
        weights = {'sortino': 0.6, 'parlay': 0.25, 'diversity': 0.15}
        
        total_reward = (
            weights['sortino'] * np.clip(sortino, -5, 5) +
            weights['parlay'] * parlay_reward +
            weights['diversity'] * diversity_reward
        )
        
        return total_reward
    
    def _end_of_day(self):
        """Process end of day with configurable reward system"""
        # Process any remaining parlays
        parlay_pnl = self._process_parlay()
        if parlay_pnl != 0:
            self.daily_pnl.append(parlay_pnl)
        
        # Calculate reward based on configuration
        if self.use_multi_objective:
            reward = self._calculate_multi_objective_reward()
        elif self.use_sortino and self.daily_pnl:
            daily_returns = np.array(self.daily_pnl)
            sortino = self._calculate_sortino(daily_returns)
            reward = np.clip(sortino, -5.0, 5.0)
        else:
            # Default sparse reward
            reward = 0.0
            if self.daily_pnl:
                pnl_array = np.array(self.daily_pnl)
                daily_mean = pnl_array.mean()
                daily_std = pnl_array.std()
                epsilon = 1e-6
                sharpe = daily_mean / (daily_std + epsilon)
                reward = np.clip(sharpe, -5.0, 5.0)
        
        # Penalty for not betting when good opportunities exist
        if not self.daily_pnl:
            current_date = self.dates[self.current_date_idx]
            opportunities = self.daily_opportunities.get(current_date, [])
            good_opportunities = sum(1 for opp in opportunities if opp.edge > 0.03 and opp.confidence > 0.6)
            if good_opportunities > 0:
                reward -= 0.1 * min(good_opportunities / 5, 1.0)
        
        # Track daily performance
        self.daily_performance.append({
            'date': self.dates[self.current_date_idx],
            'return': sum(self.daily_pnl) if self.daily_pnl else 0,
            'reward': reward,
            'num_bets': self.bets_placed_today,
            'bankroll': self.bankroll,
            'parlay_attempts': self.parlay_attempts,
            'parlay_wins': self.parlay_wins
        })
        
        # Reset daily variables
        self.daily_pnl = []
        self.bets_placed_today = 0
        self.current_bet_idx = 0
        self.bet_type_counts = defaultdict(int)
        
        # Move to next day
        self.current_date_idx += 1
        
        # Check if episode is complete
        terminated = self.current_date_idx >= len(self.dates) or self.bankroll <= 100
        
        info = {
            'date': self.dates[self.current_date_idx - 1],
            'bankroll': self.bankroll,
            'daily_reward': reward
        }
        
        return self._get_enhanced_observation(), reward, terminated, False, info


class CurriculumBettingTrainer:
    """Progressive training with increasing difficulty"""
    
    def __init__(self, base_opportunities: Dict[str, List[BettingOpportunity]]):
        self.base_opportunities = base_opportunities
        self.difficulty_levels = self._create_curriculum()
    
    def _create_curriculum(self):
        """Create training stages of increasing difficulty"""
        return [
            {
                'name': 'high_edge_only',
                'filter': lambda x: x.edge > 0.05,
                'description': 'Only very favorable bets'
            },
            {
                'name': 'positive_edge',
                'filter': lambda x: x.edge > 0.02,
                'description': 'All positive edge bets'
            },
            {
                'name': 'include_marginal',
                'filter': lambda x: x.edge > -0.01,
                'description': 'Include marginal opportunities'
            },
            {
                'name': 'realistic_market',
                'filter': lambda x: x.edge > -0.03,
                'description': 'Realistic market conditions'
            },
            {
                'name': 'full_difficulty',
                'filter': lambda x: True,
                'description': 'Full market complexity'
            }
        ]
    
    def _filter_opportunities(self, filter_func) -> Dict[str, List[BettingOpportunity]]:
        """Filter opportunities based on difficulty level"""
        filtered = {}
        for date, opps in self.base_opportunities.items():
            filtered_opps = [opp for opp in opps if filter_func(opp)]
            if filtered_opps:
                filtered[date] = filtered_opps
        return filtered
    
    def train_with_curriculum(self, model, total_steps: int, callbacks=None):
        """Train model progressively through curriculum"""
        steps_per_level = total_steps // len(self.difficulty_levels)
        
        for i, level in enumerate(self.difficulty_levels):
            logging.info(f"\nCurriculum Level {i+1}/{len(self.difficulty_levels)}: {level['name']}")
            logging.info(f"Description: {level['description']}")
            
            # Filter opportunities for this difficulty
            filtered_opps = self._filter_opportunities(level['filter'])
            logging.info(f"Training on {sum(len(opps) for opps in filtered_opps.values())} opportunities")
            
            # Create environment with current difficulty
            env = MLBBettingEnvironmentV4(
                filtered_opps,
                use_sortino=True,
                use_multi_objective=(i >= 3)  # Enable multi-objective for harder levels
            )
            env = Monitor(env)
            env = DummyVecEnv([lambda: env])
            
            # Update model's environment
            model.set_env(env)
            
            # Train on this level
            model.learn(
                total_timesteps=steps_per_level,
                callback=callbacks,
                reset_num_timesteps=False  # Continue from previous training
            )
            
            # Evaluate performance on this level
            mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
            logging.info(f"Level {i+1} completed. Mean reward: {mean_reward:.4f}")

def visualize_results(evaluation_df: pd.DataFrame, save_path: str = None):
    """Create comprehensive visualization of results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('RL Betting Agent Performance Analysis', fontsize=16)
    
    # 1. Final bankroll distribution
    axes[0, 0].hist(evaluation_df['final_bankroll'], bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 0].axvline(10000, color='red', linestyle='--', label='Initial Bankroll')
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
            env = MLBBettingEnvironmentV4(bootstrapped_opportunities)
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
    
# Keep all your existing classes (AdvancedCallbacks, EnhancedDataPreprocessor, 
# PortfolioAnalyzer, ModelEvaluator) as they are
from bettingModelv4 import AdvancedCallbacks, EnhancedDataPreprocessor, PortfolioAnalyzer
import json
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)# In bettingModelv5.py

def run_backtest_and_save_history(model, test_opportunities, output_path=r"C:\Users\DMelv\Documents\bettingModelBaseball\apiBaseball\pipeline\artemis\json\backtest_history.json"):
    """
    Runs the trained agent over a set of historical opportunities (a backtest)
    and saves a detailed JSON report of its performance and every bet it made.
    """
    logger.info(f"--- Starting Backtest ---")
    
    # 1. Initialize the environment with the historical test data
    env = MLBBettingEnvironmentV4(test_opportunities)
    obs, _ = env.reset()
    
    # 2. Run the agent through the entire historical period
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        
    logger.info(f"Backtest complete. Agent placed {len(env.episode_history)} bets.")
    
    # 3. Analyze the performance using your existing PortfolioAnalyzer
    # Note: Ensure the PortfolioAnalyzer class is defined in this file.
    final_bankroll = env.bankroll
    initial_bankroll = env.initial_bankroll
    performance_summary = PortfolioAnalyzer.analyze_episode_performance(env.episode_history)
    performance_summary['final_bankroll'] = final_bankroll
    performance_summary['initial_bankroll'] = initial_bankroll
    performance_summary['total_return_pct'] = (final_bankroll - initial_bankroll) / initial_bankroll * 100

    # 4. Create the final report object
    final_report = {
        "backtest_summary": performance_summary,
        "detailed_bets": env.episode_history
    }
    
    # 5. Save the report to a JSON file
    try:
        # A custom converter is needed to handle numpy types in the summary
        def default_converter(o):
            if isinstance(o, (np.int64, np.int32)):
                return int(o)
            if isinstance(o, (np.float64, np.float32)):
                return float(o)
            raise TypeError

        with open(output_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=default_converter)
        logger.info(f"✅ Successfully saved backtest history to '{output_path}'")
        
    except Exception as e:
        logger.error(f"Failed to save JSON report: {e}")

    return final_report

# Update the main training pipeline to use the new features
def main_training_pipeline_v4(predictions_path: str,
                             optimize_hyperparams: bool = True,
                             n_optimization_trials: int = 30,
                             total_training_steps: int = 50000,
                             n_eval_episodes: int = 100,
                             use_curriculum: bool = True):
    """
    Complete V4 training pipeline with advanced features
    """
    
    logging.info("="*60)
    logging.info("MLB RL BETTING AGENT V4 - ADVANCED TRAINING PIPELINE")
    logging.info("="*60)
    
    # Load and preprocess data
    logging.info("Loading model predictions...")
    predictions_df = pd.read_pickle(predictions_path)
    
    # Create opportunities
    preprocessor = EnhancedDataPreprocessor()
    daily_opportunities = preprocessor.create_opportunities_from_predictions(
        predictions_df,
        min_edge=-1.0,
        min_confidence=0.01
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
    
    # Hyperparameter optimization with V4 environment
    if optimize_hyperparams:
        logging.info("\nRunning hyperparameter optimization...")
        optimization_opps = {**train_opps, **val_opps}
        
        def objective(trial):
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
            gamma = trial.suggest_float('gamma', 0.95, 0.999)
            gae_lambda = trial.suggest_float('gae_lambda', 0.9, 0.99)
            ent_coef = trial.suggest_float('ent_coef', 0.0001, 0.05, log=True)
            n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096])
            clip_range = trial.suggest_float('clip_range', 0.1, 0.3)
            
            env = Monitor(MLBBettingEnvironmentV4(optimization_opps, use_sortino=True))
            env = DummyVecEnv([lambda: env])
            # --- ADD THIS LINE ---
            print("--- EXECUTING THE CORRECT, FIXED CODE BLOCK ---")            
            model = PPO(
                            policy="MlpPolicy",  # Explicitly set the policy
                            env=env,
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
                        
            model.learn(total_timesteps=500000)
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
    
    # Create final training environment
    final_train_opps = {**train_opps, **val_opps}
    
    # Initialize model
    if use_curriculum:
        # Start with easiest environment for initial model
        easy_opps = {date: [opp for opp in opps if opp.edge > -1.0] 
                     for date, opps in final_train_opps.items()
                     if any(opp.edge > -1.0 for opp in opps)}
        
        env = MLBBettingEnvironmentV4(easy_opps, use_sortino=True)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
    else:
        env = MLBBettingEnvironmentV4(final_train_opps, use_sortino=True, use_multi_objective=True)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
    
    # Create model with best hyperparameters
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=best_params.get('learning_rate', 3e-4),
        n_steps=best_params.get('n_steps', 2048),
        batch_size=64,
        n_epochs=10,
        gamma=best_params.get('gamma', 0.99),
        gae_lambda=best_params.get('gae_lambda', 0.95),
        clip_range=best_params.get('clip_range', 0.2),
        ent_coef=best_params.get('ent_coef', 0.01),
        verbose=1,
        tensorboard_log="./betting_agent_v4_tensorboard/",
        device="auto"
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='./betting_agent_v4_checkpoints/',
        name_prefix='betting_model_v4'
    )
    
    custom_callback = AdvancedCallbacks(verbose=1)
    
    # Train with curriculum or standard approach
    if use_curriculum:
        logging.info("\nTraining with curriculum learning...")
        trainer = CurriculumBettingTrainer(final_train_opps)
        trainer.train_with_curriculum(
            model, 
            total_training_steps,
            callbacks=[checkpoint_callback, custom_callback]
        )
    else:
        logging.info("\nTraining with standard approach...")
        model.learn(
            total_timesteps=total_training_steps,
            callback=[checkpoint_callback, custom_callback],
            progress_bar=True
        )
    
    # Save final model
    model.save("trained_betting_agent_v4_final")
    logging.info("Model saved to 'trained_betting_agent_v4_final'")
    
    # Evaluate on test set
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
    fig = visualize_results(eval_df, save_path='betting_agent_v4_performance.png')
    logging.info("Performance visualization saved")
    
    return model, summary, eval_df


if __name__ == "__main__":
    # --- 1. Load and Prepare Data ---
    predictions_df = pd.read_pickle("model_predictions.pkl")
    preprocessor = EnhancedDataPreprocessor()
    daily_opportunities = preprocessor.create_opportunities_from_predictions(
        predictions_df, min_edge=-1.0, min_confidence=0.01
    )
    
    # Split data to get the test set
    all_dates = sorted(daily_opportunities.keys())
    val_end = int(len(all_dates) * 0.85)
    test_dates = all_dates[val_end:]
    test_opps = {d: daily_opportunities[d] for d in test_dates}

    # --- 2. Run the Robust Bootstrapped Evaluation ---
    logger.info("--- Running Robust Bootstrapped Evaluation ---")
    try:
        trained_model = PPO.load("trained_betting_agent_v4_final")
        evaluator = ModelEvaluator(trained_model, test_opps)
        summary, eval_df = evaluator.evaluate_comprehensive(n_episodes=100)
        
# In bettingModelv5.py -> if __name__ == "__main__":

        # --- START OF FIX ---
        # Save the aggregated summary and detailed episode data to JSON
        output_path = r"C:\Users\DMelv\Documents\bettingModelBaseball\apiBaseball\pipeline\artemis\json\bootstrapped_evaluation_results.json"
        logger.info(f"Saving detailed evaluation results to '{output_path}'...")
        
        # The 'summary' variable is ALREADY a dictionary, no conversion needed.
        summary_dict = summary 
        
        # Convert the detailed DataFrame to a list of dictionaries
        detailed_results_list = eval_df.to_dict(orient='records')
        
        final_report = {
            "evaluation_summary": summary_dict,
            "detailed_episode_results": detailed_results_list
        }

        with open(output_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        logger.info("✅ Successfully saved evaluation results.")
        # --- END OF FIX ---

        # Print the summary results
        print("\n" + "="*60)
        print("BOOTSTRAPPED EVALUATION SUMMARY (100 SIMULATED SEASONS)")
        print("="*60)
        summary_df = pd.DataFrame.from_dict(summary, orient='index', columns=['Value'])
        print(summary_df)

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}")

# In bettingModelv5.py

# if __name__ == "__main__":
#     # --- 1. Load and Prepare Data ---
#     predictions_df = pd.read_pickle("model_predictions.pkl")
#     preprocessor = EnhancedDataPreprocessor()
#     daily_opportunities = preprocessor.create_opportunities_from_predictions(
#         predictions_df, min_edge=-1.0, min_confidence=0.01
#     )
    
#     # Split data to get the test set
#     all_dates = sorted(daily_opportunities.keys())
#     val_end = int(len(all_dates) * 0.85)
#     test_dates = all_dates[val_end:]
#     test_opps = {d: daily_opportunities[d] for d in test_dates}
    
#     # --- 2. Run the Backtest to Generate the History File ---
#     logger.info("--- Starting Historical Backtest ---")
#     try:
#         trained_model = PPO.load("trained_betting_agent_v4_final")
#         run_backtest_and_save_history(trained_model, test_opps, r"C:\Users\DMelv\Documents\bettingModelBaseball\apiBaseball\pipeline\artemis\json\backtest_history.json")
#     except Exception as e:
#         logger.error(f"An error occurred during backtesting: {e}")