#!/usr/bin/env python3
"""
MLB Reinforcement Learning Betting Agent
A fully autonomous betting system using state-of-the-art deep reinforcement learning
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import random
from typing import Dict, List, Tuple, Any, Optional
import os
import json
from datetime import datetime
import joblib
import logging
from dataclasses import dataclass
import warnings
from scipy.stats import norm
import pickle

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BettingMarket:
    """Represents a single betting market"""
    game_id: str
    market_type: str  # 'moneyline', 'spread', 'total', 'prop'
    team: str
    line: float
    odds: float
    true_probability: float  # From our models
    
    def calculate_edge(self) -> float:
        """Calculate the edge for this bet"""
        implied_prob = self.implied_probability()
        return self.true_probability - implied_prob
    
    def implied_probability(self) -> float:
        """Convert decimal odds to implied probability"""
        return 1.0 / self.odds
    
    def calculate_payout(self, stake: float, won: bool) -> float:
        """Calculate payout for this bet (decimal odds)"""
        if not won:
            return -stake
        return stake * (self.odds - 1)  # Profit = stake * (decimal_odds - 1)


class MLBBettingEnvironment(gym.Env):
    """
    Custom Gym environment for MLB betting
    This simulates the betting market using historical data
    """
    
    def __init__(self, 
                 historical_data_path: str,
                 models_dir: str,
                 initial_bankroll: float = 10000,
                 max_bet_fraction: float = 0.25,
                 episode_length: int = 30,
                 use_terminal_reward: bool = True):  # New parameter
        """
        Initialize the betting environment
        
        Args:
            historical_data_path: Path to historical MLB data
            models_dir: Directory containing trained prediction models
            initial_bankroll: Starting bankroll
            max_bet_fraction: Maximum fraction of bankroll for a single bet
            episode_length: Number of days in an episode
            use_terminal_reward: If True, use Sharpe-based terminal reward
        """
        super().__init__()
        
        self.initial_bankroll = initial_bankroll
        self.max_bet_fraction = max_bet_fraction
        self.episode_length = episode_length
        self.use_terminal_reward = use_terminal_reward
        
        # Load historical data and models
        self.load_data_and_models(historical_data_path, models_dir)
        
        # Define observation and action spaces
        self.setup_spaces()
        
        # Episode tracking
        self.current_episode = 0
        self.episode_history = []
        
        # Initialize metrics tracking
        self.metrics = {
            'total_bets': 0,
            'winning_bets': 0,
            'total_profit': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
    
    def load_data_and_models(self, data_path: str, models_dir: str):
        """Load historical data and prediction models"""
        logger.info("Loading historical data and models...")
        
        # Load historical games data
        self.historical_data = pd.read_parquet(data_path)
        self.historical_data['game_date'] = pd.to_datetime(self.historical_data['game_date'])
        self.historical_data = self.historical_data.sort_values('game_date')
        
        # Filter for games from 2020 onwards (when odds data starts)
        self.historical_data = self.historical_data[self.historical_data['game_date'] >= '2020-01-01']
        logger.info(f"Loaded {len(self.historical_data)} games from 2020 onwards")
        
        # Group by date for easier access
        self.games_by_date = self.historical_data.groupby('game_date')
        self.unique_dates = sorted(self.historical_data['game_date'].unique())
        
        # Load prediction models
        self.models = {}
        # In MLBBettingEnvironment -> load_data_and_models()

        # ... after loading self.scaler ...
        self.scaler = joblib.load(os.path.join(models_dir, 'feature_scaler.pkl'))

        # --- REVISED SECTION START ---
        # Load the feature list for EACH model into a dictionary
        self.model_features = {} 
        all_files = os.listdir(models_dir)
        model_names = set(
            f.removesuffix('_home.pkl').removesuffix('_away.pkl').removesuffix('_features.pkl')
            for f in all_files if f.endswith('.pkl')
        )

        logger.info(f"Discovered the following models: {list(model_names)}")

        for model_name in model_names:
            try:
                # Load the home and away models
                home_model_path = os.path.join(models_dir, f'{model_name}_home.pkl')
                away_model_path = os.path.join(models_dir, f'{model_name}_away.pkl')
                if os.path.exists(home_model_path):
                    self.models[f'{model_name}_home'] = joblib.load(home_model_path)
                if os.path.exists(away_model_path):
                    self.models[f'{model_name}_away'] = joblib.load(away_model_path)

                # Load the feature list for this model
                features_path = os.path.join(models_dir, f'{model_name}_features.pkl')
                if os.path.exists(features_path):
                    with open(features_path, 'rb') as f:
                        feature_list = pickle.load(f)
                        self.model_features[f'{model_name}_home'] = feature_list
                        self.model_features[f'{model_name}_away'] = feature_list
                    logger.info(f"Successfully loaded model pair and features for: {model_name}")

            except Exception as e:
                logger.error(f"An error occurred loading artifacts for {model_name}: {e}")
        # --- REVISED SECTION END ---\
                # --- ADD THIS CALL ---
        self._precompute_all_predictions()
        # --- END ADDITION ---
    
    def setup_spaces(self):
        """Define observation and action spaces"""
        # State space includes:
        # - Current bankroll (normalized)
        # - Days remaining in episode
        # - For each available bet: predicted edge, odds, model confidence, etc.
        
        # We'll use a fixed maximum number of markets per day
        self.max_markets_per_day = 100
        
        # Features per market
        market_features = 7  # edge, odds, confidence, variance, etc.
        
        # Total observation size
        obs_size = 2 + (self.max_markets_per_day * market_features)  # 2 for bankroll and days
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # Action space: fraction of bankroll to bet on each market (0 to max_bet_fraction)
        self.action_space = spaces.Box(
            low=0, 
            high=self.max_bet_fraction, 
            shape=(self.max_markets_per_day,), 
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode"""
        super().reset(seed=seed)
        
        # Reset bankroll and episode tracking
        self.bankroll = self.initial_bankroll
        self.current_step = 0
        self.episode_profit_history = []
        self.episode_returns = []  # For Sharpe calculation
        
        # Randomly select starting date
        max_start_idx = len(self.unique_dates) - self.episode_length
        self.start_date_idx = random.randint(0, max_start_idx)
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def _get_observation(self):
        """Get current state observation"""
        # Current date's games
        current_date = self.unique_dates[self.start_date_idx + self.current_step]
        today_games = self.games_by_date.get_group(current_date)
        
        # Generate betting markets
        markets = self._generate_markets(today_games)
        
        # Build observation vector
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Global features
        obs[0] = self.bankroll / self.initial_bankroll  # Normalized bankroll
        obs[1] = (self.episode_length - self.current_step) / self.episode_length  # Time remaining
        
        # Market features
        market_start_idx = 2
        for i, market in enumerate(markets[:self.max_markets_per_day]):
            idx = market_start_idx + (i * 7)
            obs[idx] = market.calculate_edge()  # Edge
            obs[idx + 1] = market.odds / 10  # Normalized decimal odds (typically 1.5-3.0, so /10 gives 0.15-0.3)
            obs[idx + 2] = market.true_probability  # Model probability
            obs[idx + 3] = market.implied_probability()  # Market probability
            obs[idx + 4] = self._get_model_confidence(market)  # Model confidence
            obs[idx + 5] = self._get_market_type_encoding(market.market_type)
            obs[idx + 6] = self._get_historical_performance(market)
        
        # Store markets for action processing
        self.current_markets = markets
        
        return obs
    
    def _generate_markets(self, games_df: pd.DataFrame) -> List[BettingMarket]:
        """Generate betting markets from games using PRE-COMPUTED predictions."""
        markets = []
        default_ml_odds = 1.909
        
        for _, game in games_df.iterrows():
            # --- THIS IS THE KEY CHANGE ---
            # Get predictions directly from the pre-computed columns
            predictions = {
                'home': game['ai_pred_home'],
                'away': game['ai_pred_away'],
                'home_std': game['ai_pred_home_std'],
                'away_std': game['ai_pred_away_std']
            }
            # --- NO MORE SLOW CALLS TO _extract_features or _get_model_predictions ---

            # Calculate win probabilities from model
            home_win_prob = self._calculate_win_probability(predictions)
            away_win_prob = 1 - home_win_prob
            
            # Get moneyline odds, use defaults if missing
            home_ml = float(game['home_ml']) if pd.notna(game['home_ml']) else default_ml_odds
            away_ml = float(game['away_ml']) if pd.notna(game['away_ml']) else default_ml_odds
            
            # Moneyline markets
            markets.extend([
                BettingMarket(
                    game_id=str(game['game_pk']),
                    market_type='moneyline',
                    team=game['home_team'],
                    line=0,
                    odds=home_ml,  # Decimal odds
                    true_probability=home_win_prob  # Our model's probability
                ),
                BettingMarket(
                    game_id=str(game['game_pk']),
                    market_type='moneyline',
                    team=game['away_team'],
                    line=0,
                    odds=away_ml,  # Decimal odds
                    true_probability=away_win_prob  # Our model's probability
                )
            ])
            
            # Total markets
            # Get total line, default to 8.5 if missing
            total_line = float(game['total_line']) if pd.notna(game.get('total_line')) else 8.5
            
            # Get over/under odds, default to -110 (1.909 decimal) if missing
            over_odds = float(game['over_odds']) if pd.notna(game.get('over_odds')) else default_ml_odds
            under_odds = float(game['under_odds']) if pd.notna(game.get('under_odds')) else default_ml_odds
            
            # Calculate total probabilities from our model
            predicted_total = predictions['home'] + predictions['away']
            
            # Calculate probability of going over/under based on our prediction
            # Using normal distribution assumption with some variance
            total_std = np.sqrt(predictions['home_std']**2 + predictions['away_std']**2) + 1.5  # Add base variance
            over_prob = 1 - norm.cdf(total_line, loc=predicted_total, scale=total_std)
            under_prob = 1 - over_prob
            
            markets.extend([
                BettingMarket(
                    game_id=str(game['game_pk']),
                    market_type='total',
                    team='over',
                    line=total_line,
                    odds=over_odds,  # Decimal odds
                    true_probability=over_prob  # Our model's probability
                ),
                BettingMarket(
                    game_id=str(game['game_pk']),
                    market_type='total',
                    team='under',
                    line=total_line,
                    odds=under_odds,  # Decimal odds
                    true_probability=under_prob  # Our model's probability
                )
            ])
        
        return markets

    def _extract_features(self, game_row: pd.Series) -> pd.DataFrame: #<-- Return type changed
        """
        Extracts and correctly formats features for a single game,
        including one-hot encoding, returning a DataFrame.
        """
        # 1. Convert the game series to a DataFrame with one row.
        features_df = game_row.to_frame().T

        # 2. Drop any raw datetime columns.
        datetime_cols = features_df.select_dtypes(include=['datetime64[ns]', 'datetimetz', 'datetime']).columns
        if not datetime_cols.empty:
            features_df = features_df.drop(columns=datetime_cols)

        # 3. One-hot encode any categorical/text columns.
        categorical_cols = features_df.select_dtypes(include=['category', 'object']).columns
        if not categorical_cols.empty:
            features_df = pd.get_dummies(features_df, columns=categorical_cols, dtype=float)

        # 4. Return the processed DataFrame.
        return features_df
    
    def _get_model_predictions(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """
        Get predictions from all models, aligning features for each one individually
        before scaling and predicting.
        """
        predictions = {'home': [], 'away': []}
        
        for model_name, model in self.models.items():
            try:
                # Get the specific feature list this model was trained on
                specific_feature_list = self.model_features[model_name]
                
                # Align the DataFrame to match what this model expects.
                # This is the key step: it selects the correct columns and adds zeros for missing ones.
                aligned_df = features_df.reindex(columns=specific_feature_list, fill_value=0)
                
                # Scale the ALIGNED data
                scaled_features = self.scaler.transform(aligned_df)
                
                # Now predict using the perfectly aligned and scaled data
                if 'home' in model_name:
                    predictions['home'].append(model.predict(scaled_features)[0])
                else:
                    predictions['away'].append(model.predict(scaled_features)[0])

            except KeyError:
                logger.warning(f"Feature list for model '{model_name}' not found. Skipping.")
            except Exception as e:
                logger.error(f"Error predicting with model '{model_name}': {e}")

        # Return ensemble predictions
        return {
            'home': np.mean(predictions['home']) if predictions['home'] else 0,
            'away': np.mean(predictions['away']) if predictions['away'] else 0,
            'home_std': np.std(predictions['home']) if len(predictions['home']) > 1 else 0,
            'away_std': np.std(predictions['away']) if len(predictions['away']) > 1 else 0
        }
    
    def _calculate_win_probability(self, predictions):
        """Calculate win probability from score predictions"""
        score_diff = predictions['home'] - predictions['away']
        # Use sigmoid with uncertainty adjustment
        uncertainty = (predictions['home_std'] + predictions['away_std']) / 2
        return 1 / (1 + np.exp(-score_diff / (3 + uncertainty)))
    
    def _calculate_total_probability(self, game, total_line, direction):
        """Calculate over/under probability"""
        # Simplified - in practice, you'd have a more sophisticated model
        actual_total = game['home_score'] + game['away_score']
        if direction == 'over':
            return 0.5 + 0.1 * np.tanh((total_line - actual_total) / 5)
        else:
            return 0.5 - 0.1 * np.tanh((total_line - actual_total) / 5)
    
    def _probability_to_odds(self, prob, add_noise=True):
        """Convert probability to American odds with realistic vigorish"""
        if add_noise:
            # Add bookmaker margin (typically 4-8%)
            prob = prob * (1 + random.uniform(-0.04, 0.04))
            prob = np.clip(prob, 0.01, 0.99)
        
        if prob >= 0.5:
            return -100 * prob / (1 - prob)
        else:
            return 100 * (1 - prob) / prob
    
    def _get_model_confidence(self, market):
        """Get model confidence for a market"""
        # Based on edge size and historical accuracy
        edge = abs(market.calculate_edge())
        return np.tanh(edge * 10)  # Normalize to [0, 1]
    
    def _get_market_type_encoding(self, market_type):
        """Encode market type as numeric value"""
        encodings = {'moneyline': 0.0, 'spread': 0.33, 'total': 0.66, 'prop': 1.0}
        return encodings.get(market_type, 0.5)
    
    def _get_historical_performance(self, market):
        """Get historical performance for this type of bet"""
        # Placeholder - would track actual performance
        return random.uniform(-0.1, 0.1)

    # In MLBBettingEnvironment class

    def _precompute_all_predictions(self):
        """
        Pre-computes ensemble predictions for the entire dataset ONCE to avoid
        re-calculation during training, which is the main performance bottleneck.
        """
        logger.info("Starting one-time pre-computation of all model predictions...")
        
        # Lists to store the new data
        all_home_preds = []
        all_away_preds = []
        all_home_stds = []
        all_away_stds = []

        # Use tqdm for a progress bar if you have it installed (pip install tqdm)
        try:
            from tqdm import tqdm
            iterable = tqdm(self.historical_data.iterrows(), total=len(self.historical_data), desc="Pre-computing")
        except ImportError:
            iterable = self.historical_data.iterrows()

        for _, game_row in iterable:
            # This is the slow part we are doing only once now
            features_df = self._extract_features(game_row)
            predictions = self._get_model_predictions(features_df)
            
            all_home_preds.append(predictions['home'])
            all_away_preds.append(predictions['away'])
            all_home_stds.append(predictions['home_std'])
            all_away_stds.append(predictions['away_std'])

        # Add the pre-computed predictions back to the main DataFrame
        self.historical_data['ai_pred_home'] = all_home_preds
        self.historical_data['ai_pred_away'] = all_away_preds
        self.historical_data['ai_pred_home_std'] = all_home_stds
        self.historical_data['ai_pred_away_std'] = all_away_stds
        
        logger.info("Pre-computation complete. Predictions are now cached.")

    def step(self, action):
        """Execute betting action and advance environment"""
        # Process bets
        bets_placed = []
        total_staked = 0
        
        for i, stake_fraction in enumerate(action):
            if i >= len(self.current_markets):
                break
                
            if stake_fraction > 0.001:  # Minimum bet threshold
                market = self.current_markets[i]
                stake = stake_fraction * self.bankroll
                
                # Ensure we don't bet more than we have
                if total_staked + stake <= self.bankroll:
                    bets_placed.append((market, stake))
                    total_staked += stake
        
        # Simulate bet outcomes
        profit = 0
        for market, stake in bets_placed:
            # Determine if bet won (using true outcome from historical data)
            won = self._determine_bet_outcome(market)
            payout = market.calculate_payout(stake, won)
            profit += payout
            
            # Update metrics
            self.metrics['total_bets'] += 1
            if won:
                self.metrics['winning_bets'] += 1
        
        # Update bankroll
        self.bankroll += profit
        self.episode_profit_history.append(profit)
        
        # Store daily return for Sharpe calculation
        daily_return = profit / self.initial_bankroll
        self.episode_returns.append(daily_return)
        
        # Advance time
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.episode_length or self.bankroll <= 0
        
        # Calculate reward
        if self.use_terminal_reward:
            # Terminal reward: only give reward at end of episode
            if done:
                reward = self._calculate_terminal_reward()
            else:
                reward = 0.0  # No intermediate rewards
        else:
            # Step-wise reward (original approach)
            reward = self._calculate_step_reward(profit, total_staked)
        
        # Get next observation
        if not done:
            obs = self._get_observation()
        else:
            obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            self._update_episode_metrics()
        
        info = self._get_info()
        
        return obs, reward, done, False, info
    
    def _determine_bet_outcome(self, market: BettingMarket) -> bool:
        """Determine if a bet won based on historical data"""
        # Find the game in historical data
        game_data = self.historical_data[
            self.historical_data['game_pk'] == int(market.game_id)
        ].iloc[0]
        
        if market.market_type == 'moneyline':
            home_won = game_data['home_score'] > game_data['away_score']
            return home_won if market.team == game_data['home_team'] else not home_won
        
        elif market.market_type == 'total':
            actual_total = game_data['home_score'] + game_data['away_score']
            return actual_total > market.line if market.team == 'over' else actual_total < market.line
        
        return False
    
    def _calculate_step_reward(self, profit: float, total_staked: float) -> float:
        """Calculate step-wise reward (original approach)"""
        if total_staked == 0:
            return -0.001  # Small penalty for not betting
        
        # Simple return
        simple_return = profit / self.initial_bankroll
        
        # Risk penalty for large bets
        risk_penalty = (total_staked / self.bankroll) ** 2
        
        # Combine return with risk adjustment
        reward = simple_return - 0.1 * risk_penalty
        
        return reward
    
    def _calculate_terminal_reward(self) -> float:
        """
        Calculate terminal reward based on episode's Sharpe ratio
        This encourages consistent, risk-adjusted returns over the entire episode
        """
        if len(self.episode_returns) == 0:
            return -1.0  # Penalty for immediate bankruptcy
        
        # Convert to numpy array for calculations
        returns = np.array(self.episode_returns)
        
        # Calculate Sharpe ratio (annualized)
        # Using 252 trading days per year
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return > 0:
            sharpe_ratio = mean_return / std_return * np.sqrt(252 / self.episode_length)
        else:
            # If no volatility (all returns are the same), use mean return
            sharpe_ratio = mean_return * np.sqrt(252 / self.episode_length) * 10
        
        # Additional bonuses/penalties
        final_bankroll = self.bankroll
        
        # Survival bonus (avoid bankruptcy)
        survival_bonus = 0.5 if final_bankroll > 0 else -2.0
        
        # Growth bonus (reward profitable episodes)
        growth_ratio = (final_bankroll - self.initial_bankroll) / self.initial_bankroll
        growth_bonus = np.tanh(growth_ratio * 2)  # Bounded between -1 and 1
        
        # Consistency bonus (reward steady returns)
        if len(returns) > 1:
            # Calculate maximum drawdown
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = np.min((cumulative - running_max) / (running_max + 0.001))
            consistency_bonus = 0.5 * (1 + drawdown)  # Less drawdown = higher bonus
        else:
            consistency_bonus = 0
        
        # Combine all factors
        terminal_reward = (
            sharpe_ratio * 10 +  # Scale Sharpe ratio to meaningful range
            survival_bonus +
            growth_bonus +
            consistency_bonus
        )
        
        # Log the components for debugging
        logger.debug(f"Terminal Reward Components:")
        logger.debug(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
        logger.debug(f"  Survival Bonus: {survival_bonus:.3f}")
        logger.debug(f"  Growth Bonus: {growth_bonus:.3f}")
        logger.debug(f"  Consistency Bonus: {consistency_bonus:.3f}")
        logger.debug(f"  Total Terminal Reward: {terminal_reward:.3f}")
        
        return terminal_reward
    
    def _update_episode_metrics(self):
        """Update metrics at end of episode"""
        # Calculate Sharpe ratio
        if len(self.episode_profit_history) > 1:
            returns = np.array(self.episode_profit_history) / self.initial_bankroll
            self.metrics['sharpe_ratio'] = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        
        # Calculate max drawdown
        cumulative = np.cumsum(self.episode_profit_history)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1)
        self.metrics['max_drawdown'] = np.min(drawdown)
        
        self.metrics['total_profit'] = self.bankroll - self.initial_bankroll
    
    def _get_info(self):
        """Get current environment info"""
        return {
            'bankroll': self.bankroll,
            'total_profit': self.bankroll - self.initial_bankroll,
            'metrics': self.metrics.copy(),
            'current_step': self.current_step,
            'bets_available': len(self.current_markets) if hasattr(self, 'current_markets') else 0
        }


class ActorCriticNetwork(nn.Module):
    """
    Combined Actor-Critic network for PPO
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Sigmoid()  # Output between 0 and 1, will be scaled by max_bet_fraction
        )
        
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, obs):
        """Forward pass for both actor and critic"""
        shared_features = self.shared(obs)
        
        # Actor
        action_mean = self.actor_mean(shared_features)
        action_std = torch.exp(self.actor_log_std)
        
        # Critic
        value = self.critic(shared_features)
        
        return action_mean, action_std, value
    
    def get_action(self, obs, deterministic=False):
        """Sample action from policy"""
        action_mean, action_std, value = self.forward(obs)
        
        if deterministic:
            return action_mean, value
        
        # Sample from normal distribution
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value


class PPOAgent:
    """
    Proximal Policy Optimization agent for betting
    """
    
    def __init__(self, 
                 env: MLBBettingEnvironment,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5):
        
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Initialize network
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.policy = ActorCriticNetwork(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Training tracking
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'value_losses': [],
            'policy_losses': [],
            'bankrolls': [],
            'sharpe_ratios': []
        }
    
    def collect_rollout(self, num_steps: int = 2048):
        """Collect experience by running policy in environment"""
        observations = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        obs, _ = self.env.reset()
        
        for _ in range(num_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(obs_tensor)
            
            # Scale action by max bet fraction
            scaled_action = action.squeeze().numpy() * self.env.max_bet_fraction
            
            # Step environment
            next_obs, reward, done, _, info = self.env.step(scaled_action)
            
            # Store experience
            observations.append(obs)
            actions.append(action.squeeze().numpy())
            rewards.append(reward)
            values.append(value.squeeze().item())
            log_probs.append(log_prob.item())
            dones.append(done)
            
            obs = next_obs
            
            if done:
                # Log episode statistics
                self.training_history['episode_rewards'].append(info['total_profit'])
                self.training_history['bankrolls'].append(info['bankroll'])
                self.training_history['sharpe_ratios'].append(info['metrics']['sharpe_ratio'])
                
                obs, _ = self.env.reset()
        
        return {
            'observations': torch.FloatTensor(observations),
            'actions': torch.FloatTensor(actions),
            'rewards': torch.FloatTensor(rewards),
            'values': torch.FloatTensor(values),
            'log_probs': torch.FloatTensor(log_probs),
            'dones': torch.FloatTensor(dones)
        }
    
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values
        return advantages, returns
    
    def update_policy(self, rollout_data, epochs: int = 10, batch_size: int = 64):
        """Update policy using PPO"""
        # Compute advantages
        advantages, returns = self.compute_gae(
            rollout_data['rewards'],
            rollout_data['values'],
            rollout_data['dones']
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Create dataset
        dataset_size = len(rollout_data['observations'])
        indices = np.arange(dataset_size)
        
        for _ in range(epochs):
            np.random.shuffle(indices)
            
            for start_idx in range(0, dataset_size, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                batch_obs = rollout_data['observations'][batch_indices]
                batch_actions = rollout_data['actions'][batch_indices]
                batch_old_log_probs = rollout_data['log_probs'][batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get current policy outputs
                action_mean, action_std, values = self.policy(batch_obs)
                dist = Normal(action_mean, action_std)
                log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # Calculate ratio for PPO
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Clipped surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Log losses
                self.training_history['policy_losses'].append(policy_loss.item())
                self.training_history['value_losses'].append(value_loss.item())
    
    def train(self, total_timesteps: int = 1_000_000, 
              rollout_steps: int = 2048,
              update_epochs: int = 10,
              save_interval: int = 10):
        """Main training loop"""
        logger.info(f"Starting PPO training for {total_timesteps} timesteps")
        
        num_updates = total_timesteps // rollout_steps
        
        for update in range(num_updates):
            # Collect rollout
            rollout_data = self.collect_rollout(rollout_steps)
            
            # Update policy
            self.update_policy(rollout_data, epochs=update_epochs)
            
            # Log progress
            if update % 10 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-10:])
                avg_bankroll = np.mean(self.training_history['bankrolls'][-10:])
                avg_sharpe = np.mean(self.training_history['sharpe_ratios'][-10:])
                
                logger.info(f"Update {update}/{num_updates}")
                logger.info(f"  Avg Profit: ${avg_reward:.2f}")
                logger.info(f"  Avg Bankroll: ${avg_bankroll:.2f}")
                logger.info(f"  Avg Sharpe: {avg_sharpe:.3f}")
            
            # Save checkpoint
            if update % save_interval == 0:
                self.save_checkpoint(f"ppo_betting_agent_update_{update}.pth")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }
        torch.save(checkpoint, filename)
        logger.info(f"Saved checkpoint: {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filename)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        logger.info(f"Loaded checkpoint: {filename}")
    
    def evaluate(self, num_episodes: int = 100):
        """Evaluate the trained agent"""
        logger.info(f"Evaluating agent over {num_episodes} episodes")

        episode_profits = []
        episode_sharpes = []
        win_rates = []

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False

            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

                with torch.no_grad():
                    # In evaluation, always use the deterministic action
                    action_mean, _, _ = self.policy(obs_tensor)

                # Use action_mean directly for deterministic evaluation
                scaled_action = action_mean.squeeze().numpy() * self.env.max_bet_fraction
                obs, reward, done, _, info = self.env.step(scaled_action)

            episode_profits.append(info['total_profit'])
            episode_sharpes.append(info['metrics']['sharpe_ratio'])
            win_rates.append(info['metrics']['winning_bets'] / max(info['metrics']['total_bets'], 1))

        # --- FIX STARTS HERE ---
        # Convert numpy types to native Python types for JSON serialization
        avg_profit = float(np.mean(episode_profits))
        std_profit = float(np.std(episode_profits))
        avg_sharpe = float(np.mean(episode_sharpes))
        std_sharpe = float(np.std(episode_sharpes))
        avg_win_rate = float(np.mean(win_rates))
        std_win_rate = float(np.std(win_rates))
        profitable_episodes = int(sum(p > 0 for p in episode_profits))

        # Print evaluation results
        logger.info("\nEvaluation Results:")
        logger.info(f"  Average Profit: ${avg_profit:.2f} ± ${std_profit:.2f}")
        logger.info(f"  Average Sharpe Ratio: {avg_sharpe:.3f} ± {std_sharpe:.3f}")
        logger.info(f"  Average Win Rate: {avg_win_rate:.3f} ± {std_win_rate:.3f}")
        logger.info(f"  Profitable Episodes: {profitable_episodes}/{num_episodes}")

        # Return a clean dictionary with native Python types
        return {
            'average_profit': avg_profit,
            'std_dev_profit': std_profit,
            'average_sharpe_ratio': avg_sharpe,
            'std_dev_sharpe_ratio': std_sharpe,
            'average_win_rate': avg_win_rate,
            'std_dev_win_rate': std_win_rate,
            'profitable_episodes': profitable_episodes,
            'total_episodes': num_episodes,
            'raw_profits': episode_profits, # This list of floats is fine
            'raw_sharpe_ratios': episode_sharpes, # This list of floats is fine
            'raw_win_rates': win_rates # This list of floats is fine
        }


def main():
    """Main function to train the RL betting agent"""
    # Configuration
    HISTORICAL_DATA_PATH = r"C:\Users\DMelv\Documents\bettingModelBaseball\apiBaseball\pipeline\ml_pipeline_output\master_features_table.parquet" # Your MLB data
    MODELS_DIR = r"C:\Users\DMelv\Documents\bettingModelBaseball\apiBaseball\pipeline\artemis\models"  # Directory with your trained prediction models
    OUTPUT_DIR = r"C:\Users\DMelv\Documents\bettingModelBaseball\apiBaseball\pipeline\artemis\models\rl_betting_agent"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize environment
    env = MLBBettingEnvironment(
        historical_data_path=HISTORICAL_DATA_PATH,
        models_dir=MODELS_DIR,
        initial_bankroll=10000,
        max_bet_fraction=0.05,
        episode_length=30,  # 30 days per episode
        use_terminal_reward=True  # Use Sharpe-based terminal rewards
    )
    
    # Initialize PPO agent
    agent = PPOAgent(
        env=env,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2
    )
    
    # Train the agent
    agent.train(
        total_timesteps=500_000,  # Adjust based on your compute resources
        rollout_steps=2048,
        update_epochs=10,
        save_interval=50
    )
    
    # Save final model
    agent.save_checkpoint(os.path.join(OUTPUT_DIR, "final_betting_agent.pth"))
    
    # Evaluate the trained agent
    evaluation_results = agent.evaluate(num_episodes=100)
    
    # Save evaluation results
    with open(os.path.join(OUTPUT_DIR, "evaluation_results.json"), "w") as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Save training history
    pd.DataFrame(agent.training_history).to_csv(
        os.path.join(OUTPUT_DIR, "training_history.csv"), 
        index=False
    )
    
    logger.info("\nTraining complete! Check the output directory for results.")


if __name__ == "__main__":
    main()