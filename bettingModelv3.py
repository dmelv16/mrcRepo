"""
MLB ADVANCED BETTING MODEL V2 - ENHANCED PROFESSIONAL EDITION
Complete betting system with uncertainty quantification, backtesting, and advanced analytics
Updated with SQLAlchemy integration and real data sources
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import json
import os
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# Statistical libraries
from scipy import stats
from scipy.stats import poisson, nbinom, norm
from sklearn.preprocessing import StandardScaler, RobustScaler

# For API calls
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# For backtesting
from collections import defaultdict
import pickle

# Database
from sqlalchemy import create_engine, text
import urllib.parse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.impute import KNNImputer
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA

import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import logging
import joblib
import json
import os
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import math
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from pipelineTrainv5 import (
    MLBNeuralNetV2, MLBNeuralNetV3, MLBNeuralNetWithUncertainty,
    MLBHybridModel, MLBGraphNeuralNetwork, ModelConfig, ImprovedFeatureSelector, StackingEnsemble, TeamGraphAttention,
    TemporalTeamEncoder, PerceiverBlock, AdvancedFeatureEngineer, BaseModelWrapper
)
# ============= DATABASE CONNECTION =============
def get_db_engine():
    """Create SQLAlchemy engine with SQL Server connection"""
    params = urllib.parse.quote_plus(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=DESKTOP-J9IV3OH;"
        "DATABASE=StatcastDB;"
        "UID=mlb_user;"
        "PWD=mlbAdmin;"
        "Encrypt=no;"
        "TrustServerCertificate=yes;"
    )
    return create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from scipy.stats import nbinom, norm
from scipy.optimize import minimize

class ProfessionalPortfolioOptimizer:
    """Professional portfolio optimizer with formal optimization and advanced risk management"""
    
    def __init__(self, config):
        self.config = config
        self.min_portfolio_size = 0  # Minimum bets for diversification
        self.max_team_exposure = 0.3  # Max 30% exposure to any single team
        self.max_type_exposure = 0.7  # Max 70% in any bet type
        
    def optimize_portfolio(self, straight_bets: List['EnhancedBet'], 
                          parlays: List['EnhancedParlay'], 
                          bankroll: float,
                          correlation_matrix: Dict = None) -> Dict:
        """Optimize portfolio using formal mathematical optimization"""
        
        if len(straight_bets) == 0:
            return self._empty_portfolio()
        
        # Step 1: Pre-filter candidates
        viable_bets = self._prefilter_bets(straight_bets)
        
        if len(viable_bets) < self.min_portfolio_size:
            logging.warning(f"Only {len(viable_bets)} viable bets found. Need at least {self.min_portfolio_size}")
            return self._empty_portfolio()
        
        # Step 2: Build correlation matrix
        corr_matrix = self._build_correlation_matrix(viable_bets, correlation_matrix)
        
        # Step 3: Run optimization
        optimal_weights = self._optimize_weights(viable_bets, corr_matrix, bankroll)
        
        # Step 4: Apply risk constraints and finalize
        final_portfolio = self._apply_constraints(viable_bets, optimal_weights, bankroll)
        
        # Step 5: Add parlays if appropriate
        final_portfolio = self._add_parlays(final_portfolio, parlays, bankroll)
        
        # Step 6: Calculate advanced metrics
        metrics = self._calculate_portfolio_metrics(final_portfolio, bankroll)
        
        return metrics
    
    def _prefilter_bets(self, bets: List['EnhancedBet']) -> List['EnhancedBet']:
        """
        Pre-filter bets using a unified Quality Score to identify the most promising candidates.
        """
        
        # --- START: THE FIX ---
        
        qualified_bets = []
        for bet in bets:
            # Step 1: Apply the correct minimum edge threshold for the bet type.
            # This fixes the bug of using the moneyline edge for totals.
            min_edge_for_type = 0
            if bet.bet_type == 'moneyline':
                min_edge_for_type = self.config.min_edge_moneyline
            elif bet.bet_type == 'total':
                min_edge_for_type = self.config.min_edge_totals

            # Basic sanity checks to immediately discard impossible bets.
            if bet.edge < min_edge_for_type:
                continue
            if bet.kelly_stake <= 0:
                continue
            if not (0.1 < bet.probability < 0.9):
                continue
            
            # Step 2: Calculate the unified Quality Score.
            # We add a small epsilon to uncertainty to prevent division by zero.
            quality_score = (bet.edge * bet.confidence) / (1 + bet.uncertainty + 1e-9)
            
            # Attach the score to the bet object for sorting.
            bet.quality_score = quality_score
            qualified_bets.append(bet)
            
        # Step 3: Sort all qualified bets by their Quality Score in descending order.
        qualified_bets.sort(key=lambda b: b.quality_score, reverse=True)
        
        # Step 4: Return the top N bets to ensure the optimizer has a strong pool of candidates.
        # This is more flexible than relying on arbitrary confidence/edge multipliers.
        MAX_CANDIDATES = 15 
        return qualified_bets[:MAX_CANDIDATES]

        # --- END: THE FIX ---
    
    def _build_correlation_matrix(self, bets: List['EnhancedBet'], 
                                 historical_corr: Dict = None) -> np.ndarray:
        """Build correlation matrix for portfolio bets"""
        n_bets = len(bets)
        corr_matrix = np.eye(n_bets)  # Start with identity matrix
        
        for i in range(n_bets):
            for j in range(i + 1, n_bets):
                bet_i = bets[i]
                bet_j = bets[j]
                
                # Calculate correlation based on various factors
                correlation = 0.0
                
                # Same game correlation
                if bet_i.game_id == bet_j.game_id:
                    if bet_i.bet_type == bet_j.bet_type:
                        correlation = 0.9  # Same game, same type (very high)
                    else:
                        correlation = 0.5  # Same game, different type
                
                # Use historical correlations if available
                elif historical_corr:
                    key = (f"{bet_i.bet_type}_{bet_i.selection}", 
                           f"{bet_j.bet_type}_{bet_j.selection}")
                    correlation = historical_corr.get(key, 0.1)
                
                # Default small positive correlation
                else:
                    correlation = 0.1
                
                corr_matrix[i, j] = correlation
                corr_matrix[j, i] = correlation
        
        return corr_matrix
    
    def _optimize_weights(self, bets: List['EnhancedBet'], 
                         corr_matrix: np.ndarray, 
                         bankroll: float) -> np.ndarray:
        """Optimize portfolio weights using mean-variance optimization"""
        
        n_bets = len(bets)
        
        # Expected returns (edges)
        expected_returns = np.array([bet.edge for bet in bets])
        
        # Variance estimates (using uncertainty)
        variances = np.array([bet.uncertainty ** 2 for bet in bets])
        
        # Covariance matrix
        std_devs = np.sqrt(variances)
        cov_matrix = np.outer(std_devs, std_devs) * corr_matrix
        
        # Objective function: Maximize Sharpe Ratio
        def sharpe_ratio(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            
            if portfolio_variance <= 0:
                return -np.inf
            
            # Negative because we minimize
            return -portfolio_return / np.sqrt(portfolio_variance)
        
        # Constraints
        constraints = [
            # Weights sum to less than max risk
            {'type': 'ineq', 'fun': lambda w: self.config.daily_risk_limit - np.sum(w)},
            # Individual bet constraints based on Kelly
            *[{'type': 'ineq', 'fun': lambda w, i=i: bets[i].kelly_stake - w[i]} 
              for i in range(n_bets)],
            # Non-negative weights
            *[{'type': 'ineq', 'fun': lambda w, i=i: w[i]} for i in range(n_bets)]
        ]
        
        # Initial guess (proportional to Kelly stakes)
        kelly_stakes = np.array([bet.kelly_stake for bet in bets])
        initial_weights = kelly_stakes / kelly_stakes.sum() * self.config.daily_risk_limit * 0.5
        
        # Bounds
        bounds = [(0, min(bet.kelly_stake, self.config.max_bet_fraction)) 
                  for bet in bets]
        
        # Optimize
        result = minimize(
            sharpe_ratio,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200}
        )
        
        if result.success:
            return result.x
        else:
            logging.warning("Optimization failed, using Kelly weights")
            return kelly_stakes
    
    def _apply_constraints(self, bets: List['EnhancedBet'], 
                          weights: np.ndarray, 
                          bankroll: float) -> Dict:
        """Apply real-world constraints to optimized weights"""
        
        # Calculate bet sizes
        bet_sizes = weights * bankroll
        
        # Apply minimum bet size
        min_bet = 10  # $10 minimum
        bet_sizes[bet_sizes < min_bet] = 0
        
        # Diversification constraints
        portfolio = {
            'bets': [],
            'weights': [],
            'sizes': []
        }
        
        # Track exposures
        team_exposure = {}
        type_exposure = {'moneyline': 0, 'total': 0}
        
        # Sort by weight descending
        sorted_indices = np.argsort(weights)[::-1]
        
        total_allocated = 0
        
        for idx in sorted_indices:
            if bet_sizes[idx] == 0:
                continue
            
            bet = bets[idx]
            bet_size = bet_sizes[idx]
            
            # Check team exposure
            teams = self._extract_teams(bet)
            team_risk = bet_size / bankroll
            
            can_add = True
            for team in teams:
                current_exposure = team_exposure.get(team, 0)
                if current_exposure + team_risk > self.max_team_exposure:
                    can_add = False
                    break
            
            # Check type exposure
            type_risk = type_exposure[bet.bet_type] + team_risk
            if type_risk > self.max_type_exposure:
                can_add = False
            
            # Check total risk
            if total_allocated + bet_size > bankroll * self.config.daily_risk_limit:
                can_add = False
            
            if can_add:
                portfolio['bets'].append(bet)
                portfolio['weights'].append(weights[idx])
                portfolio['sizes'].append(bet_size)
                
                # Update exposures
                for team in teams:
                    team_exposure[team] = team_exposure.get(team, 0) + team_risk
                type_exposure[bet.bet_type] += team_risk
                total_allocated += bet_size
        
        return portfolio
    
    def _add_parlays(self, portfolio: Dict, parlays: List['EnhancedParlay'], 
                    bankroll: float) -> Dict:
        """Add parlays to portfolio with proper sizing"""
        
        if not parlays:
            portfolio['parlays'] = []
            portfolio['parlay_sizes'] = []
            return portfolio
        
        # Current risk
        current_risk = sum(portfolio['sizes'])
        remaining_risk = bankroll * self.config.daily_risk_limit - current_risk
        
        # Allocate max 20% of remaining risk to parlays
        parlay_budget = remaining_risk * 0.2
        
        selected_parlays = []
        parlay_sizes = []
        
        # Sort parlays by EV/risk ratio
        sorted_parlays = sorted(
            parlays,
            key=lambda p: p.expected_value / p.combined_uncertainty,
            reverse=True
        )
        
        for parlay in sorted_parlays[:self.config.max_parlays_per_day]:
            # Size based on Kelly but more conservative
            size = min(
                parlay.kelly_stake * bankroll * 0.3,  # 30% of Kelly for parlays
                parlay_budget / len(sorted_parlays[:self.config.max_parlays_per_day]),
                bankroll * 0.01  # Max 1% per parlay
            )
            
            if size >= 10:  # Minimum bet
                selected_parlays.append(parlay)
                parlay_sizes.append(size)
                parlay_budget -= size
        
        portfolio['parlays'] = selected_parlays
        portfolio['parlay_sizes'] = parlay_sizes
        
        return portfolio
    
    def _calculate_portfolio_metrics(self, portfolio: Dict, bankroll: float) -> Dict:
        """Calculate comprehensive portfolio metrics"""
        
        if not portfolio['bets']:
            return self._empty_portfolio()
        
        bets = portfolio['bets']
        sizes = np.array(portfolio['sizes'])
        
        # Basic metrics
        total_risk = sizes.sum()
        
        # Expected value
        expected_returns = np.array([bet.edge * size for bet, size in zip(bets, sizes)])
        total_ev = expected_returns.sum()
        
        # Portfolio variance (simplified - would use full covariance in production)
        variances = np.array([bet.uncertainty ** 2 * size ** 2 
                             for bet, size in zip(bets, sizes)])
        portfolio_variance = variances.sum()  # Assuming independence for simplicity
        portfolio_std = np.sqrt(portfolio_variance)
        
        # Risk metrics
        sharpe_ratio = total_ev / portfolio_std if portfolio_std > 0 else 0
        
        # Diversification score
        diversification_score = self._calculate_diversification_score(portfolio, bankroll)
        
        # Monte Carlo simulation for advanced risk metrics
        var_95, cvar_95 = self._monte_carlo_risk_analysis(portfolio, n_simulations=10000)
        
        return {
            'straight_bets': bets,
            'parlays': portfolio.get('parlays', []),
            'straight_bet_sizes': {bet.game_id: size for bet, size in zip(bets, sizes)},
            'parlay_bet_sizes': portfolio.get('parlay_sizes', []),
            'total_risk': total_risk,
            'risk_percentage': (total_risk / bankroll) * 100,
            'expected_value': total_ev,
            'expected_roi': (total_ev / total_risk) * 100 if total_risk > 0 else 0,
            'sharpe_ratio': sharpe_ratio,
            'portfolio_std': portfolio_std,
            'diversification_score': diversification_score,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'games_covered': len(set(bet.game_id for bet in bets))
        }
    
    def _calculate_diversification_score(self, portfolio: Dict, bankroll: float) -> float:
        """Calculate portfolio diversification score (0-1, higher is better)"""
        
        bets = portfolio['bets']
        sizes = portfolio['sizes']
        
        if len(bets) == 0:
            return 0
        
        # Factor 1: Number of bets (more is better, with diminishing returns)
        bet_count_score = min(len(bets) / 10, 1.0)  # Optimal at 10+ bets
        
        # Factor 2: Bet type diversity
        ml_weight = sum(s for b, s in zip(bets, sizes) if b.bet_type == 'moneyline')
        total_weight = sum(s for b, s in zip(bets, sizes) if b.bet_type == 'total')
        type_balance = 1 - abs(ml_weight - total_weight) / (ml_weight + total_weight)
        
        # Factor 3: Team diversity (using Herfindahl index)
        team_weights = {}
        for bet, size in zip(bets, sizes):
            teams = self._extract_teams(bet)
            for team in teams:
                team_weights[team] = team_weights.get(team, 0) + size
        
        if team_weights:
            total_weight = sum(team_weights.values())
            herfindahl = sum((w/total_weight)**2 for w in team_weights.values())
            team_diversity = 1 - herfindahl
        else:
            team_diversity = 0
        
        # Factor 4: Probability diversity (avoid all favorites or all dogs)
        probs = [bet.probability for bet in bets]
        prob_std = np.std(probs)
        prob_diversity = min(prob_std / 0.2, 1.0)  # Optimal std around 0.2
        
        # Weighted combination
        diversification_score = (
            0.2 * bet_count_score +
            0.2 * type_balance +
            0.4 * team_diversity +
            0.2 * prob_diversity
        )
        
        return diversification_score
    
    def _monte_carlo_risk_analysis(self, portfolio: Dict, 
                                  n_simulations: int = 10000) -> Tuple[float, float]:
        """Run Monte Carlo simulation to calculate VaR and CVaR"""
        
        bets = portfolio['bets']
        sizes = portfolio['sizes']
        
        # Simulate outcomes
        outcomes = []
        
        for _ in range(n_simulations):
            pnl = 0
            
            # Straight bets
            for bet, size in zip(bets, sizes):
                # Use model probability with some uncertainty
                prob = np.random.normal(bet.probability, bet.uncertainty / 10)
                prob = np.clip(prob, 0, 1)
                
                if np.random.random() < prob:
                    # Win
                    if bet.odds > 0:
                        pnl += size * (bet.odds / 100)
                    else:
                        pnl += size * (100 / abs(bet.odds))
                else:
                    # Loss
                    pnl -= size
            
            # Parlays
            for parlay, size in zip(portfolio.get('parlays', []), 
                                   portfolio.get('parlay_sizes', [])):
                parlay_wins = True
                for leg in parlay.legs:
                    leg_prob = np.random.normal(leg.probability, leg.uncertainty / 10)
                    leg_prob = np.clip(leg_prob, 0, 1)
                    
                    if np.random.random() >= leg_prob:
                        parlay_wins = False
                        break
                
                if parlay_wins:
                    pnl += size * (parlay.combined_odds - 1)
                else:
                    pnl -= size
            
            outcomes.append(pnl)
        
        # Calculate risk metrics
        outcomes = np.array(outcomes)
        var_95 = np.percentile(outcomes, 5)  # 5th percentile (95% VaR)
        cvar_95 = outcomes[outcomes <= var_95].mean()  # Expected loss beyond VaR
        
        return var_95, cvar_95
    
    def _extract_teams(self, bet: 'EnhancedBet') -> List[str]:
        """Extract team identifiers from a bet"""
        # This would need to be implemented based on your game_id format
        # For now, return the game_id as a proxy
        return [bet.game_id]
    
    def _empty_portfolio(self) -> Dict:
        """Return empty portfolio structure"""
        return {
            'straight_bets': [],
            'parlays': [],
            'straight_bet_sizes': {},
            'parlay_bet_sizes': [],
            'total_risk': 0,
            'risk_percentage': 0,
            'expected_value': 0,
            'expected_roi': 0,
            'sharpe_ratio': 0,
            'portfolio_std': 0,
            'diversification_score': 0,
            'var_95': 0,
            'cvar_95': 0,
            'games_covered': 0
        }

from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns

class OptimizedConfidenceCalculator:
    """Data-driven confidence calculator with optimized weights"""
    
    def __init__(self):
        self.weights = None
        self.scaler_params = None
        self.performance_history = []
        
    def optimize_weights(self, historical_data: pd.DataFrame, 
                        n_splits: int = 5) -> Dict:
        """Optimize confidence weights using historical performance"""
        
        logging.info("Optimizing confidence calculator weights...")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        all_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(historical_data)):
            logging.info(f"Processing fold {fold + 1}/{n_splits}")
            
            train_data = historical_data.iloc[train_idx]
            val_data = historical_data.iloc[val_idx]
            
            # Optimize on training data
            optimal_weights = self._optimize_fold(train_data)
            
            # Validate on validation data
            val_metrics = self._evaluate_weights(val_data, optimal_weights)
            
            all_results.append({
                'fold': fold,
                'weights': optimal_weights,
                'val_sharpe': val_metrics['sharpe_ratio'],
                'val_roi': val_metrics['roi']
            })
        
        # Select best weights based on validation Sharpe ratio
        best_fold = max(all_results, key=lambda x: x['val_sharpe'])
        self.weights = best_fold['weights']
        
        logging.info(f"Optimal weights found: {self.weights}")
        
        # Also optimize scaling parameters
        self._optimize_scalers(historical_data)
        
        return {
            'optimal_weights': self.weights,
            'cross_validation_results': all_results,
            'average_sharpe': np.mean([r['val_sharpe'] for r in all_results]),
            'average_roi': np.mean([r['val_roi'] for r in all_results])
        }
    
    def _optimize_fold(self, train_data: pd.DataFrame) -> Dict:
        """Optimize weights for a single fold"""
        
        def objective(weight_array):
            """Objective function: negative Sharpe ratio"""
            weights = {
                'uncertainty': weight_array[0],
                'edge': weight_array[1],
                'agreement': weight_array[2],
                'historical': weight_array[3]
            }
            
            # Calculate confidence for all bets
            confidences = []
            for _, row in train_data.iterrows():
                conf = self._calculate_confidence_with_weights(row, weights)
                confidences.append(conf)
            
            train_data['confidence'] = confidences
            
            # Simulate betting with these confidences
            metrics = self._simulate_betting_performance(train_data)
            
            # Return negative Sharpe (we're minimizing)
            return -metrics['sharpe_ratio']
        
        # Constraints: weights sum to 1, all non-negative
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: w[0]},  # uncertainty >= 0
            {'type': 'ineq', 'fun': lambda w: w[1]},  # edge >= 0
            {'type': 'ineq', 'fun': lambda w: w[2]},  # agreement >= 0
            {'type': 'ineq', 'fun': lambda w: w[3]}   # historical >= 0
        ]
        
        # Initial guess (equal weights)
        initial_weights = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 100}
        )
        
        if result.success:
            return {
                'uncertainty': result.x[0],
                'edge': result.x[1],
                'agreement': result.x[2],
                'historical': result.x[3]
            }
        else:
            logging.warning("Optimization failed, using default weights")
            return {
                'uncertainty': 0.4,
                'edge': 0.3,
                'agreement': 0.2,
                'historical': 0.1
            }
    
    def _calculate_confidence_with_weights(self, bet_data: pd.Series, 
                                         weights: Dict) -> float:
        """Calculate confidence using given weights"""
        
        # Extract components
        uncertainty = bet_data.get('uncertainty', 1.0)
        edge = bet_data.get('edge', 0.0)
        model_std = bet_data.get('model_std', 0.5)
        historical_acc = bet_data.get('historical_accuracy', 0.5)
        
        # Scale components
        scaled_uncertainty = 1 / (1 + uncertainty / self.scaler_params['uncertainty_scale'])
        edge_confidence = min(abs(edge) / self.scaler_params['edge_scale'], 1.0)
        agreement_confidence = 1 / (1 + model_std * self.scaler_params['agreement_scale'])
        
        # Weighted combination
        confidence = (
            weights['uncertainty'] * scaled_uncertainty +
            weights['edge'] * edge_confidence +
            weights['agreement'] * agreement_confidence +
            weights['historical'] * historical_acc
        )
        
        # Apply non-linear transformation
        confidence = self._apply_calibrated_penalty(confidence)
        
        return np.clip(confidence, 0.1, 0.9)
    
    def _optimize_scalers(self, data: pd.DataFrame):
        """Optimize scaling parameters"""
        
        # Calculate optimal scales based on data distribution
        self.scaler_params = {
            'uncertainty_scale': data['uncertainty'].quantile(0.75),
            'edge_scale': data['edge'].abs().quantile(0.9),
            'agreement_scale': 2.0  # Can be optimized further
        }
    
    def _apply_calibrated_penalty(self, raw_confidence: float, 
                                 penalty_strength: float = 0.7) -> float:
        """Apply optimized confidence penalty"""
        
        # Asymmetric penalty: stronger on high confidence
        if raw_confidence > 0.5:
            return 0.5 + (raw_confidence - 0.5) * penalty_strength
        else:
            return 0.5 - (0.5 - raw_confidence) * (penalty_strength + 0.15)
    
    def _simulate_betting_performance(self, data: pd.DataFrame) -> Dict:
        """Simulate betting performance for given confidences"""
        
        # Simple Kelly betting simulation
        returns = []
        
        for _, bet in data.iterrows():
            if bet['confidence'] > 0.5:  # Only bet if confident
                kelly_fraction = 0.25 * bet['confidence']
                stake = kelly_fraction * bet['edge'] / (bet['odds'] - 1)
                
                # Simulate outcome
                if bet.get('won', np.random.random() < bet['probability']):
                    returns.append(stake * (bet['odds'] - 1))
                else:
                    returns.append(-stake)
        
        if not returns:
            return {'sharpe_ratio': 0, 'roi': 0}
        
        returns = np.array(returns)
        
        return {
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252),
            'roi': np.mean(returns) * 100
        }
    
    def _evaluate_weights(self, val_data: pd.DataFrame, weights: Dict) -> Dict:
        """Evaluate weights on validation data"""
        
        # Calculate confidences
        confidences = []
        for _, row in val_data.iterrows():
            conf = self._calculate_confidence_with_weights(row, weights)
            confidences.append(conf)
        
        val_data = val_data.copy()
        val_data['confidence'] = confidences
        
        return self._simulate_betting_performance(val_data)
    
    def save(self, filepath: str):
        """Save optimized parameters"""
        joblib.dump({
            'weights': self.weights,
            'scaler_params': self.scaler_params
        }, filepath)
    
    def load(self, filepath: str):
        """Load optimized parameters"""
        params = joblib.load(filepath)
        self.weights = params['weights']
        self.scaler_params = params['scaler_params']


class OptimizedKellyCalculator:
    """Kelly calculator with data-driven thresholds"""
    
    def __init__(self, base_fraction: float = 0.25):
        self.base_fraction = base_fraction
        self.confidence_thresholds = None
        self.bet_type_adjustments = None
        
    def optimize_thresholds(self, historical_data: pd.DataFrame) -> Dict:
        """Optimize confidence thresholds using historical data"""
        
        logging.info("Optimizing Kelly calculator thresholds...")
        
        # Find optimal thresholds for different confidence levels
        confidence_bins = [0.5, 0.6, 0.7, 0.8, 0.9]
        optimal_multipliers = {}
        
        for i in range(len(confidence_bins) - 1):
            low, high = confidence_bins[i], confidence_bins[i + 1]
            
            # Filter data for this confidence range
            mask = (historical_data['confidence'] >= low) & (historical_data['confidence'] < high)
            bin_data = historical_data[mask]
            
            if len(bin_data) < 50:  # Need sufficient data
                optimal_multipliers[low] = 0.5  # Conservative default
                continue
            
            # Find optimal multiplier for this bin
            multiplier = self._optimize_multiplier_for_bin(bin_data)
            optimal_multipliers[low] = multiplier
        
        self.confidence_thresholds = optimal_multipliers
        
        # Optimize bet type adjustments
        self.bet_type_adjustments = self._optimize_bet_type_adjustments(historical_data)
        
        return {
            'confidence_thresholds': self.confidence_thresholds,
            'bet_type_adjustments': self.bet_type_adjustments
        }
    
    def _optimize_multiplier_for_bin(self, bin_data: pd.DataFrame) -> float:
        """Find optimal Kelly multiplier for a confidence bin"""
        
        def objective(multiplier):
            """Objective: negative geometric mean return"""
            returns = []
            
            for _, bet in bin_data.iterrows():
                # Calculate Kelly stake with this multiplier
                kelly = self._calculate_base_kelly(bet['probability'], bet['odds'])
                stake = kelly * self.base_fraction * multiplier[0]
                stake = np.clip(stake, 0, 0.05)  # Cap at 5%
                
                # Calculate return
                if bet.get('won', np.random.random() < bet['probability']):
                    ret = stake * (bet['odds'] - 1)
                else:
                    ret = -stake
                
                returns.append(1 + ret)
            
            # Geometric mean (for compound growth)
            geo_mean = np.prod(returns) ** (1 / len(returns))
            
            return -(geo_mean - 1)  # Negative for minimization
        
        # Optimize
        result = minimize(
            objective,
            x0=[0.5],  # Initial multiplier
            bounds=[(0.1, 1.5)],  # Reasonable bounds
            method='L-BFGS-B'
        )
        
        return result.x[0] if result.success else 0.5
    
    def _optimize_bet_type_adjustments(self, historical_data: pd.DataFrame) -> Dict:
        """Find optimal adjustments for different bet types"""
        
        adjustments = {}
        
        for bet_type in ['moneyline', 'total']:
            type_data = historical_data[historical_data['bet_type'] == bet_type]
            
            if len(type_data) < 100:
                adjustments[bet_type] = 1.0
                continue
            
            # Find adjustment that maximizes risk-adjusted returns
            best_adj = 1.0
            best_sharpe = -np.inf
            
            for adj in np.linspace(0.5, 1.2, 15):
                sharpe = self._evaluate_adjustment(type_data, adj)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_adj = adj
            
            adjustments[bet_type] = best_adj
        
        return adjustments
    
    def _evaluate_adjustment(self, data: pd.DataFrame, adjustment: float) -> float:
        """Evaluate performance with given adjustment"""
        
        returns = []
        
        for _, bet in data.iterrows():
            kelly = self._calculate_base_kelly(bet['probability'], bet['odds'])
            stake = kelly * self.base_fraction * adjustment
            stake = np.clip(stake, 0, 0.05)
            
            if bet.get('won', np.random.random() < bet['probability']):
                returns.append(stake * (bet['odds'] - 1))
            else:
                returns.append(-stake)
        
        if not returns:
            return -np.inf
        
        return np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
    
    def _calculate_base_kelly(self, probability: float, decimal_odds: float) -> float:
        """Calculate base Kelly stake"""
        q = 1 - probability
        b = decimal_odds - 1
        
        if b <= 0:
            return 0.0
        
        return (probability * b - q) / b
    
    def calculate_optimized_kelly(self, probability: float, decimal_odds: float,
                                confidence: float, bet_type: str) -> float:
        """Calculate Kelly stake with optimized parameters"""
        
        # Base Kelly
        kelly = self._calculate_base_kelly(probability, decimal_odds)
        
        # Get confidence multiplier
        multiplier = 1.0
        if self.confidence_thresholds:
            for threshold, mult in sorted(self.confidence_thresholds.items()):
                if confidence >= threshold:
                    multiplier = mult
        
        # Apply bet type adjustment
        if self.bet_type_adjustments:
            multiplier *= self.bet_type_adjustments.get(bet_type, 1.0)
        
        # Apply all adjustments
        kelly = kelly * self.base_fraction * multiplier
        
        # Dynamic cap based on confidence
        max_bet = 0.03 if confidence < 0.7 else 0.05
        
        return max(0, min(kelly, max_bet))


class CalibrationVisualizer:
    """Visualize calibration quality with reliability diagrams"""
    
    @staticmethod
    def plot_reliability_diagram(predictions: np.ndarray, actuals: np.ndarray,
                               n_bins: int = 10, title: str = "Reliability Diagram"):
        """Plot reliability diagram to assess calibration"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bin predictions
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Calculate observed frequencies
        observed_freq = []
        expected_freq = []
        counts = []
        
        for i in range(n_bins):
            mask = (predictions >= bins[i]) & (predictions < bins[i + 1])
            n_samples = mask.sum()
            
            if n_samples > 0:
                observed = actuals[mask].mean()
                expected = predictions[mask].mean()
            else:
                observed = expected = bin_centers[i]
            
            observed_freq.append(observed)
            expected_freq.append(expected)
            counts.append(n_samples)
        
        # Plot 1: Reliability diagram
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax1.scatter(expected_freq, observed_freq, s=np.array(counts)*5, alpha=0.7)
        ax1.plot(expected_freq, observed_freq, 'b-', label='Model calibration')
        
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Observed Frequency')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # Plot 2: Histogram of predictions
        ax2.hist(predictions, bins=bins, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Predicted Probability')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Predictions')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Calculate ECE
        ece = 0
        for i in range(n_bins):
            if counts[i] > 0:
                weight = counts[i] / len(predictions)
                ece += weight * abs(observed_freq[i] - expected_freq[i])
        
        fig.suptitle(f'Expected Calibration Error (ECE): {ece:.4f}', y=1.02)
        
        return fig
    
    @staticmethod
    def plot_calibration_comparison(raw_predictions: np.ndarray,
                                  calibrated_predictions: np.ndarray,
                                  actuals: np.ndarray):
        """Compare raw vs calibrated predictions"""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot raw calibration
        CalibrationVisualizer._plot_single_reliability(
            axes[0], raw_predictions, actuals, "Raw Model Predictions"
        )
        
        # Plot calibrated
        CalibrationVisualizer._plot_single_reliability(
            axes[1], calibrated_predictions, actuals, "Calibrated Predictions"
        )
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def _plot_single_reliability(ax, predictions, actuals, title):
        """Helper to plot single reliability diagram"""
        
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        
        observed_freq = []
        expected_freq = []
        
        for i in range(n_bins):
            mask = (predictions >= bins[i]) & (predictions < bins[i + 1])
            if mask.sum() > 0:
                observed = actuals[mask].mean()
                expected = predictions[mask].mean()
                observed_freq.append(observed)
                expected_freq.append(expected)
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.7)
        ax.plot(expected_freq, observed_freq, 'bo-', markersize=8)
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Observed Frequency')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

class EnhancedProbabilityCalibrator:
    """Calibrate model probabilities to match reality"""
    
    def __init__(self):
        self.win_calibrator = None
        self.total_calibrator = None
        self.calibration_data = []
        
    def collect_calibration_data(self, predictions: np.ndarray, 
                                actuals: np.ndarray, bet_type: str):
        """Collect prediction/actual pairs for calibration"""
        for pred, actual in zip(predictions, actuals):
            self.calibration_data.append({
                'prediction': pred,
                'actual': actual,
                'bet_type': bet_type
            })
    
    def fit_calibrators(self):
        """Fit isotonic regression calibrators"""
        df = pd.DataFrame(self.calibration_data)
        
        # Separate by bet type
        win_data = df[df['bet_type'] == 'moneyline']
        total_data = df[df['bet_type'] == 'total']
        
        # Fit win probability calibrator
        if len(win_data) > 100:
            self.win_calibrator = IsotonicRegression(out_of_bounds='clip')
            self.win_calibrator.fit(
                win_data['prediction'].values,
                win_data['actual'].values
            )
            logging.info("Win probability calibrator fitted")
        
        # Fit total probability calibrator
        if len(total_data) > 100:
            self.total_calibrator = IsotonicRegression(out_of_bounds='clip')
            self.total_calibrator.fit(
                total_data['prediction'].values,
                total_data['actual'].values
            )
            logging.info("Total probability calibrator fitted")
    
    def calibrate_probability(self, raw_prob: float, bet_type: str) -> float:
        """Apply calibration to raw probability"""
        if bet_type == 'moneyline' and self.win_calibrator:
            return float(self.win_calibrator.predict([raw_prob])[0])
        elif bet_type == 'total' and self.total_calibrator:
            return float(self.total_calibrator.predict([raw_prob])[0])
        else:
            # Conservative adjustment if no calibrator available
            return self._conservative_adjustment(raw_prob)
    
    def _conservative_adjustment(self, raw_prob: float) -> float:
        """Apply conservative adjustment to reduce overconfidence"""
        # Push probabilities toward 0.5 (reduce confidence)
        adjustment_factor = 0.7  # Tune based on backtesting
        return 0.5 + (raw_prob - 0.5) * adjustment_factor


class ImprovedProbabilityCalculator:
    """Use Negative Binomial distribution for better overdispersion modeling"""
    
    def __init__(self, overdispersion_factor: float = 0.5):
        self.overdispersion = overdispersion_factor
        
    def calculate_win_probability_nbinom(self, home_score: float, away_score: float,
                                        home_uncertainty: float, away_uncertainty: float) -> float:
        """Calculate win probability using Negative Binomial distribution"""
        
        # Convert mean and uncertainty to NB parameters
        home_r, home_p = self._mean_var_to_nbinom_params(home_score, home_uncertainty)
        away_r, away_p = self._mean_var_to_nbinom_params(away_score, away_uncertainty)
        
        win_prob = 0.0
        tie_prob = 0.0
        
        # Calculate probability grid
        for h in range(25):  # Extended range for NB's heavier tail
            for a in range(25):
                p_home = nbinom.pmf(h, home_r, home_p)
                p_away = nbinom.pmf(a, away_r, away_p)
                joint_prob = p_home * p_away
                
                if h > a:
                    win_prob += joint_prob
                elif h == a:
                    tie_prob += joint_prob
        
        # Add half of tie probability
        win_prob += tie_prob / 2
        
        return np.clip(win_prob, 0.01, 0.99)
    
    def _mean_var_to_nbinom_params(self, mean: float, uncertainty: float) -> Tuple[float, float]:
        """Convert mean and uncertainty to negative binomial parameters"""
        # FIX 1: Ensure the mean (predicted score) is always positive.
        mean = max(mean, 0.01)

        variance = mean * (1 + self.overdispersion) + uncertainty**2

        # FIX 2: Ensure variance is always greater than mean to prevent p >= 1.
        if variance <= mean:
            variance = mean * 1.1 + 0.1 # Add a small buffer

        p = mean / variance

        # FIX 3: Clip p to a safe range *before* it's used in the next calculation.
        p = np.clip(p, 0.01, 0.99)

        # Now this calculation is safe from division-by-zero errors.
        r = mean * p / (1 - p)
        
        r = max(0.1, r)
        
        return r, p
    
    def calculate_total_probability_nbinom(self, predicted_total: float, line: float,
                                         home_score: float, away_score: float,
                                         uncertainty: float) -> Tuple[float, float]:
        """Calculate over/under probabilities using Negative Binomial"""
        
        home_r, home_p = self._mean_var_to_nbinom_params(home_score, uncertainty/2)
        away_r, away_p = self._mean_var_to_nbinom_params(away_score, uncertainty/2)
        
        over_prob = 0.0
        
        for h in range(25):
            for a in range(25):
                if h + a > line:
                    p_home = nbinom.pmf(h, home_r, home_p)
                    p_away = nbinom.pmf(a, away_r, away_p)
                    over_prob += p_home * p_away
        
        under_prob = 1 - over_prob
        
        return over_prob, under_prob


class DynamicKellyCalculator:
    """Dynamic Kelly fraction based on calibrated confidence"""
    
    def __init__(self, base_fraction: float = 0.25):
        self.base_fraction = base_fraction
        self.confidence_thresholds = {
            0.8: 1.0,    # High confidence: full Kelly fraction
            0.7: 0.7,    # Medium-high: 70% of Kelly
            0.6: 0.4,    # Medium: 40% of Kelly
            0.0: 0.2     # Low: 20% of Kelly
        }
    
    def calculate_dynamic_kelly(self, probability: float, decimal_odds: float,
                              confidence: float, bet_type: str) -> float:
        """Calculate Kelly stake with dynamic fraction based on confidence"""
        
        # Base Kelly calculation
        q = 1 - probability
        b = decimal_odds - 1
        
        if b <= 0:
            return 0.0
        
        kelly = (probability * b - q) / b
        
        # Get confidence multiplier
        multiplier = self._get_confidence_multiplier(confidence)
        
        # Additional penalty for totals (typically harder to predict)
        if bet_type == 'total':
            multiplier *= 0.8
        
        # Apply dynamic fraction
        kelly = kelly * self.base_fraction * multiplier
        
        # Conservative cap based on confidence
        max_bet = 0.03 if confidence < 0.7 else 0.05
        
        return max(0, min(kelly, max_bet))
    
    def _get_confidence_multiplier(self, confidence: float) -> float:
        """Get Kelly multiplier based on confidence level"""
        for threshold, multiplier in sorted(self.confidence_thresholds.items(), reverse=True):
            if confidence >= threshold:
                return multiplier
        return self.confidence_thresholds[0.0]


class ImprovedConfidenceCalculator:
    """Calculate confidence scores with empirically tuned weights"""
    
    def __init__(self):
        # These weights should be tuned on validation data
        self.weights = {
            'uncertainty': 0.5,   # Model uncertainty (most important)
            'edge': 0.2,         # Edge strength
            'agreement': 0.2,    # Model ensemble agreement
            'historical': 0.1    # Historical performance (if available)
        }
    
    def calculate_confidence(self, predictions: Dict, idx: int,
                           probability: float, uncertainty: float,
                           historical_accuracy: float = None) -> float:
        """Calculate refined confidence score"""
        
        # Uncertainty component (inverse relationship)
        # Scale uncertainty to [0, 1] range first
        scaled_uncertainty = 1 / (1 + uncertainty / 3)  # Assuming typical uncertainty 0-3
        
        # Edge component (distance from 50%)
        edge_confidence = abs(probability - 0.5) * 2
        edge_confidence = min(edge_confidence, 1.0)
        
        # Model agreement component
        agreement_confidence = 0.5  # Default if no ensemble
        if 'individual_predictions' in predictions:
            predictions_array = np.array(predictions['individual_predictions'])
            if len(predictions_array) > 0:
                model_std = predictions_array[:, idx].std()
                # Convert std to confidence (lower std = higher confidence)
                agreement_confidence = 1 / (1 + model_std * 2)
        
        # Historical component (if available)
        hist_confidence = historical_accuracy if historical_accuracy else 0.5
        
        # Weighted combination
        confidence = (
            self.weights['uncertainty'] * scaled_uncertainty +
            self.weights['edge'] * edge_confidence +
            self.weights['agreement'] * agreement_confidence +
            self.weights['historical'] * hist_confidence
        )
        
        # Apply non-linear transformation to penalize overconfidence
        # This pushes extreme confidences toward the middle
        confidence = self._apply_confidence_penalty(confidence)
        
        return np.clip(confidence, 0.1, 0.9)
    
    def _apply_confidence_penalty(self, raw_confidence: float) -> float:
        """Apply penalty to reduce overconfidence"""
        # Sigmoid-like transformation that compresses extremes
        if raw_confidence > 0.5:
            # Penalize high confidence more
            return 0.5 + (raw_confidence - 0.5) * 0.7
        else:
            # Less penalty for low confidence
            return 0.5 - (0.5 - raw_confidence) * 0.85
                        
class ImprovedFeatureSelector:
    """Enhanced feature selection with multiple strategies"""
    
    def __init__(self, max_features: int = 250):
        self.max_features = max_features
        self.selected_features = None
        self.feature_importance_df = None
        self.category_importance = {}
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, 
            val_X: pd.DataFrame = None, val_y: pd.DataFrame = None) -> 'ImprovedFeatureSelector':
        """Select features using multiple methods and validation performance"""
        
        logging.info(f"Enhanced feature selection from {X.shape[1]} features...")
        
        # Use 75% of training data for feature selection
        cutoff = int(len(X) * 0.75)
        X_subset = X.iloc[:cutoff]
        y_subset = y.iloc[:cutoff]
        
        # 1. Remove constant features
        variances = X_subset.var()
        low_var_features = variances[variances < 0.001].index.tolist()
        X_filtered = X_subset.drop(columns=low_var_features)
        logging.info(f"Removed {len(low_var_features)} low variance features")
        
        # 2. Remove highly correlated features
        corr_matrix = X_filtered.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = []
        for column in upper_tri.columns:
            if column in to_drop:
                continue
            correlated_features = list(upper_tri.index[upper_tri[column] > 0.90])
            if correlated_features:
                all_features = [column] + correlated_features
                variances = X_filtered[all_features].var()
                keep_feature = variances.idxmax()
                drop_features = [f for f in all_features if f != keep_feature]
                to_drop.extend(drop_features)
        
        X_filtered = X_filtered.drop(columns=to_drop)
        logging.info(f"Removed {len(to_drop)} highly correlated features")
        
        # 3. Mutual information scores
        mi_scores = mutual_info_regression(X_filtered, y_subset['home_score'] - y_subset['away_score'])
        
        # 4. Multiple model importances
        importances = {}
        
        # LightGBM for different targets
        targets = {
            'total_runs': y_subset['home_score'] + y_subset['away_score'],
            'run_diff': y_subset['home_score'] - y_subset['away_score'],
            'home_wins': (y_subset['home_score'] > y_subset['away_score']).astype(int)
        }
        
        for target_name, target_values in targets.items():
            lgb_model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
            lgb_model.fit(X_filtered, target_values)
            importances[f'lgb_{target_name}'] = lgb_model.feature_importances_
        
        # Random Forest for winner prediction
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        rf.fit(X_filtered, (y_subset['home_score'] > y_subset['away_score']).astype(int))
        importances['rf_winner'] = rf.feature_importances_
        
        # 5. Combine importance scores
        importance_df = pd.DataFrame({
            'feature': X_filtered.columns,
            'mi_score': mi_scores
        })
        
        for name, scores in importances.items():
            importance_df[name] = scores
        
        # Weighted combination
        importance_df['combined_score'] = (
            0.2 * importance_df['mi_score'] / importance_df['mi_score'].max() +
            0.3 * importance_df['lgb_total_runs'] / importance_df['lgb_total_runs'].max() +
            0.3 * importance_df['lgb_run_diff'] / importance_df['lgb_run_diff'].max() +
            0.2 * importance_df['rf_winner'] / importance_df['rf_winner'].max()
        )
        
        importance_df = importance_df.sort_values('combined_score', ascending=False)
        self.feature_importance_df = importance_df
        
        # 6. Category-aware selection with better patterns
        selected_features = []
        category_patterns = {
            'batting': ['batting', 'hitting', 'avg', 'obp', 'slg', 'ops', 'rbi', 'hits'],
            'pitching': ['pitching', 'era', 'whip', 'strikeout', 'walk', 'pitch'],
            'SP': ['_sp_', 'starter', 'starting_pitcher'],
            'bullpen': ['bullpen', 'reliever', 'relief'],
            'momentum': ['streak', 'momentum', 'last_', 'recent', 'form'],
            'situational': ['venue', 'park', 'weather', 'temp', 'wind', 'day', 'night'],
            'diff': ['diff_', 'versus', '_vs_', 'matchup']
        }
        
        # Minimum features per category
        category_minimums = {
            'batting': 20,
            'pitching': 15,
            'SP': 25,
            'bullpen': 15,
            'momentum': 10,
            'situational': 15,
            'diff': 20
        }
        
        # First pass: ensure minimum representation
        for category, patterns in category_patterns.items():
            category_features = []
            
            for _, row in importance_df.iterrows():
                if row['feature'] in selected_features:
                    continue
                
                # Check if feature matches any pattern for this category
                feature_lower = row['feature'].lower()
                if any(pattern in feature_lower for pattern in patterns):
                    category_features.append(row['feature'])
                    
                    if len(category_features) >= category_minimums.get(category, 10):
                        break
            
            selected_features.extend(category_features)
            self.category_importance[category] = len(category_features)
            logging.info(f"Selected {len(category_features)} features for {category}")
        
        # Second pass: add top remaining features
        remaining_budget = self.max_features - len(selected_features)
        remaining_features = [f for f in importance_df['feature'] if f not in selected_features]
        selected_features.extend(remaining_features[:remaining_budget])
        
        self.selected_features = selected_features[:self.max_features]
        logging.info(f"Selected {len(self.selected_features)} total features")
        
        # 7. Validation-based refinement if validation set provided
        if val_X is not None and val_y is not None:
            self._refine_with_validation(X_filtered, y_subset, val_X, val_y)
        
        return self
    
    def _refine_with_validation(self, X: pd.DataFrame, y: pd.DataFrame, 
                               val_X: pd.DataFrame, val_y: pd.DataFrame):
        """Refine feature selection based on validation performance"""
        
        # Test incremental feature sets
        feature_sets = [50, 100, 150, 200, 250, 300]
        best_score = -np.inf
        best_n_features = self.max_features
        
        for n_features in feature_sets:
            if n_features > len(self.selected_features):
                break
            
            test_features = self.selected_features[:n_features]
            
            # Quick model to test
            model = lgb.LGBMRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=5,
                verbose=-1
            )
            
            model.fit(X[test_features], y['home_score'] - y['away_score'])
            val_pred = model.predict(val_X[test_features])
            
            # Score based on winner accuracy
            val_true_diff = val_y['home_score'] - val_y['away_score']
            accuracy = ((val_pred > 0) == (val_true_diff > 0)).mean()
            
            if accuracy > best_score:
                best_score = accuracy
                best_n_features = n_features
        
        # Update selected features
        self.selected_features = self.selected_features[:best_n_features]
        logging.info(f"Refined to {best_n_features} features based on validation (accuracy: {best_score:.3%})")
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data to selected features"""
        return X[self.selected_features]
    
# ============= ENHANCED CONFIGURATION =============
@dataclass
class BettingConfig:
    """Enhanced configuration for betting strategy"""
    # Model paths
    model_dir: str = "./mlb_run_prediction_model_v4"
    historical_data_path: str = r"C:\Users\DMelv\Documents\bettingModelBaseball\apiBaseball\pipeline\ml_pipeline_output\master_features_table.parquet" # Path to parquet file
    
    # Betting thresholds
    min_edge_moneyline: float = 0.07352245991140353 # 5% minimum edge for moneyline bets
    min_edge_totals: float = 0.10909197113221557     # 4% minimum edge for totals
    
    # Kelly Criterion parameters
    kelly_fraction: float = 0.10893796712544962    # Conservative Kelly (quarter Kelly)
    max_bet_fraction: float = 0.1   # Max 5% of bankroll per bet

    # --- Parlay Configuration (Revamped) ---
    enable_parlays: bool = True  # Master switch to turn parlays on/off

    # Thresholds for selecting INDIVIDUAL legs
    min_edge_parlay_leg: float = -0.16493987372914232       # Min 3% edge for a bet to be considered as a parlay leg
    min_confidence_parlay_leg: float = 0.5402807215794936 # Min confidence for a bet to be a parlay leg

    # Thresholds for the FINAL combined parlay
    min_parlay_edge: float = -0.7786456496799152           # Min 15% combined edge for the whole parlay to be placed
    
    # Sizing and Risk for parlays
    max_parlay_legs: int = 4                 # Maximum legs in any parlay
    parlay_kelly_multiplier: float = 0.25    # Extra conservatism for parlay sizing (25% of normal Kelly)

    # Risk Management
    daily_risk_limit: float = 0.05           # Max 3% of bankroll at risk per day
    max_parlays_per_day: int = 2             # Maximum number of parlays to place    
    max_straight_bets: int = 15      # Maximum straight bets per day
    
    # flat betting
    flat_bet_unit: float = 0.05  # Bet 1% of bankroll on every flat bet
    # Confidence thresholds
    min_confidence_moneyline: float = 0.5357886892468485
    min_confidence_over: float = 0.5759281613589134
    min_confidence_under: float = 0.5601125162484605
    
    # Distribution parameters
    use_poisson: bool = True          # Use Poisson for run distribution
    poisson_adjustment: float = 1.1   # Adjustment factor for Poisson variance
    
    # Backtesting
    backtest_start_date: str = "2024-04-01"
    backtest_end_date: str = "2025-07-05"
    
    # API Configuration (removed api_key)
    odds_update_frequency: int = 300  # Update odds every 5 minutes
    
    # CLV Tracking
    track_clv: bool = True
    clv_threshold: float = 0.02      # 2% CLV threshold for good bets

@dataclass
class EnhancedBet:
    """Enhanced bet with additional tracking fields"""
    game_id: str
    bet_type: str
    selection: str
    odds: float
    probability: float
    edge: float
    kelly_stake: float
    confidence: float
    model_prediction: Dict
    uncertainty: float
    timestamp: datetime = field(default_factory=datetime.now)
    closing_odds: Optional[float] = None
    clv: Optional[float] = None
    result: Optional[str] = None
    pnl: Optional[float] = None
    bet_size: Optional[float] = None # <-- ADD THIS LINE # <-- ADD THIS LINE

# ============= WEATHER API INTEGRATION =============
def get_weather_forecast(lat, lon, utc_datetime):
    """Fetch weather forecast from Open-Meteo API"""
    base_url = "https://api.open-meteo.com/v1/forecast"
    date_part = utc_datetime[:10]
   
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,weathercode",
        "timezone": "UTC", "start_date": date_part, "end_date": date_part
    }
   
    def degrees_to_compass(deg):
        dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        return dirs[int((deg / 22.5) + 0.5) % 16]
    
    def map_wind_dir(compass_dir):
        """Map compass direction to game format"""
        mapping = {
            'N': 'In From CF', 'S': 'Out To CF',
            'E': 'L To R', 'W': 'R To L',
            'NE': 'In From RF', 'NW': 'In From LF',
            'SE': 'Out To RF', 'SW': 'Out To LF'
        }
        return mapping.get(compass_dir, 'Varies')
   
    weather_code_map = {
        0: "Clear", 1: "Sunny", 2: "Partly Cloudy", 3: "Overcast", 45: "Fog", 48: "Cloudy",
        51: "Drizzle", 53: "Drizzle", 55: "Drizzle", 61: "Rain", 63: "Rain", 65: "Rain",
        71: "Snow", 73: "Snow", 75: "Snow", 80: "Rain", 81: "Rain", 82: "Rain",
        95: "Rain", 96: "Rain", 99: "Rain"
    }
   
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
       
        if utc_datetime not in data['hourly']['time']:
            return None
           
        idx = data['hourly']['time'].index(utc_datetime)
        temp_c = data['hourly']['temperature_2m'][idx]
        wind_deg = data['hourly']['wind_direction_10m'][idx]
        weather_code = data['hourly']['weathercode'][idx]
       
        return {
            'temperature': round(temp_c * 9/5 + 32, 1),
            'wind_speed': round(data['hourly']['wind_speed_10m'][idx] * 0.621371, 1),
            'wind_dir': map_wind_dir(degrees_to_compass(wind_deg)),
            'conditions': weather_code_map.get(weather_code, f"Unknown ({weather_code})")
        }
    except Exception as e:
        print(f"Weather API error: {e}")
        return None

class FeatureEngineeringPipeline:
    """Complete feature engineering pipeline for live games"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = None
        self.feature_names = None
        self.advanced_engineer = None
        self._load_preprocessors()
        self.historical_data = None
        self._load_historical_data()
        
    def _load_preprocessors(self):
        """Load saved preprocessors from training"""
        try:
            import joblib
            import json
            import os
            
            model_dir = self.config.model_dir
            self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
            
            # Load feature configuration
            with open(os.path.join(model_dir, 'config.json'), 'r') as f:
                config_data = json.load(f)
                self.feature_names = config_data.get('selected_features', [])
            
            # Load the fitted feature engineer
            self.advanced_engineer = joblib.load(os.path.join(self.config.model_dir, 'feature_engineer.pkl'))
            
            # Check if it has park factors
            if hasattr(self.advanced_engineer, 'park_factors') and self.advanced_engineer.park_factors is not None:
                logging.info(f"Loaded feature engineer with park factors for {len(self.advanced_engineer.park_factors)} venues")
            else:
                logging.warning("Feature engineer loaded but no park factors found")
                
            logging.info(f"Loaded {len(self.feature_names)} features from training config")
        except Exception as e:
            logging.error(f"Error loading preprocessors: {e}")
            # Initialize a new feature engineer if loading fails
            self.advanced_engineer = AdvancedFeatureEngineer()
    
    def _load_historical_data(self):
        """Load historical data from parquet file"""
        try:
            self.historical_data = pd.read_parquet(self.config.historical_data_path)
            # Fix column name if needed
            if 'game_pk' in self.historical_data.columns and 'game_id' not in self.historical_data.columns:
                self.historical_data['game_id'] = self.historical_data['game_pk'].astype(str)
            logging.info(f"Loaded historical data with {len(self.historical_data)} records")
        except Exception as e:
            logging.error(f"Error loading historical data: {e}")
            
    def fetch_game_data(self, date: str) -> pd.DataFrame:
        """Fetch game data from historical parquet file"""
        if self.historical_data is None:
            logging.error("No historical data loaded")
            return pd.DataFrame()
        
        # Convert date to datetime
        target_date = pd.to_datetime(date).date()
        
        # Filter games for the specific date
        games_data = self.historical_data[
            pd.to_datetime(self.historical_data['game_date']).dt.date == target_date
        ].copy()
        
        if len(games_data) == 0:
            logging.warning(f"No games found for {date}")
            return pd.DataFrame()
        
        # Ensure game_id column exists
        if 'game_pk' in games_data.columns and 'game_id' not in games_data.columns:
            games_data['game_id'] = games_data['game_pk'].astype(str)
            
        return games_data
    
    def engineer_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features matching the training pipeline"""
        features_df = raw_data.copy()
        
        # Fix game_id if needed
        if 'game_pk' in features_df.columns and 'game_id' not in features_df.columns:
            features_df['game_id'] = features_df['game_pk'].astype(str)
        
        # Use the AdvancedFeatureEngineer from training
        # It will use pre-calculated park factors automatically
        features_df = self.advanced_engineer.create_advanced_features(features_df)
        
        # Define feature groups for smart imputation
        feature_groups = {
            'batting': [col for col in features_df.columns if 'batting' in col.lower()],
            'pitching': [col for col in features_df.columns if 'pitching' in col.lower() and 'bullpen' not in col.lower()],
            'bullpen': [col for col in features_df.columns if 'bullpen' in col.lower()],
            'momentum': [col for col in features_df.columns if any(x in col.lower() for x in ['streak', 'momentum', 'last_'])],
            'other': []
        }
        
        # Apply smart imputation
        features_df = self.advanced_engineer.smart_imputation(features_df, feature_groups)
        
        # Create any remaining features that the model expects
        for feature in self.feature_names:
            if feature not in features_df.columns:
                # Intelligent defaults based on feature name
                if 'era' in feature.lower():
                    features_df[feature] = 4.00
                elif 'woba' in feature:
                    features_df[feature] = 0.320
                elif 'ops' in feature:
                    features_df[feature] = 0.740
                elif 'avg' in feature or 'batting' in feature:
                    features_df[feature] = 0.250
                elif 'roll' in feature:
                    features_df[feature] = 0.5
                elif 'park_factor' in feature:
                    features_df[feature] = 1.0  # Neutral park factor
                else:
                    features_df[feature] = 0.0
                    logging.debug(f"Feature '{feature}' not found, using default value 0")
        
        # --- ADD THIS FIX ---
        # Create a list of columns to keep: the features + game_id
        columns_to_keep = self.feature_names[:]  # Create a copy
        if 'game_id' not in columns_to_keep:
            columns_to_keep.append('game_id')
        
        # Return the dataframe with only the necessary columns, ensuring game_id is present
        return features_df[[col for col in columns_to_keep if col in features_df.columns]]
        # --- END FIX ---

    def fetch_and_process_day(self, date: pd.Timestamp):
        date_str = date.strftime('%Y-%m-%d')
        raw_games_for_day = self.historical_data[
            pd.to_datetime(self.historical_data['game_date']).dt.date == date.date()
        ].copy()

        if raw_games_for_day.empty:
            return pd.DataFrame(), pd.DataFrame(), {}

        # --- START FIX ---
        # Define and remove leaky columns BEFORE feature engineering.
        leaky_columns = [
            'game_pk', 'gamePk', 'game_date', 'home_team', 'away_team', 'home_team_id', 
            'away_team_id', 'home_team_abbr', 'away_team_abbr', 'home_game_date', 
            'away_game_date', 'home_W/L', 'away_W/L', 'bookmaker', 'time_match_key', 
            'date_match_key', 'home_ml', 'away_ml', 'total_line', 'over_odds',
            'home_score', 'away_score', 'match_key', 'diff_score'
        ]
        pre_game_data = raw_games_for_day.drop(columns=leaky_columns, errors='ignore')

        # Now, engineer features on the clean pre_game_data.
        games_features = self.engineer_features(pre_game_data)
        games_features.fillna(0, inplace=True)
        # --- END FIX ---

        # Define the columns to bring over from the raw data (odds, scores, etc.)
        # IMPORTANT: Ensure the key 'game_id' is in this list, not 'game_pk'
        columns_to_keep = ['game_id', 'home_ml', 'away_ml', 'total_line', 'over_odds', 'under_odds', 'home_score', 'away_score']

        # Match the features onto the odds and scores using the standardized 'game_id'.
        games_data_combined = pd.merge(
            games_features,
            raw_games_for_day[[col for col in columns_to_keep if col in raw_games_for_day.columns]],
            on='game_id',  # <-- THIS IS THE FIX: Use 'game_id' instead of 'game_pk'
            how='left'
        )

        # Build the odds_data dictionary
        odds_data = {}
        for _, game_row in games_data_combined.iterrows():
            # Use the standardized 'game_id' here as well
            game_id_str = str(game_row['game_id'])
            odds_data[game_id_str] = {
                'home_ml': game_row.get('home_ml'),
                'away_ml': game_row.get('away_ml'),
                'total_line': game_row.get('total_line'),
                'over_odds': game_row.get('over_odds'),
                'under_odds': game_row.get('under_odds'),
            }
        
        return raw_games_for_day, games_data_combined, odds_data
    
@dataclass
class EnhancedParlay:
    """Enhanced parlay with uncertainty tracking"""
    legs: List[EnhancedBet]
    combined_odds: float
    combined_probability: float
    expected_value: float
    kelly_stake: float
    correlation_adjustment: float
    combined_uncertainty: float

# ============= ENHANCED MODEL LOADER WITH UNCERTAINTY =============
import inspect

# ============= ENHANCED MODEL LOADER WITH UNCERTAINTY =============
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import os
import logging
import inspect

# Import all necessary model and configuration classes from the training script
from pipelineTrainv5 import (
    MLBNeuralNetV2, MLBNeuralNetV3, MLBNeuralNetWithUncertainty,
    MLBHybridModel, MLBGraphNeuralNetwork, ModelConfig
)

class EnhancedModelLoader:
    """Fixed model loader that trusts the stacking ensemble"""

    def __init__(self, model_dir: str):
        """
        Constructor for the model loader. This was the missing piece.
        It loads all necessary components from the specified directory.
        """
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        logging.info(f"Loading models from: {model_dir}")
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = []

        # Load configuration
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'r') as f:
            self.training_config = json.load(f)

        # Load preprocessors
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        self.feature_selector = joblib.load(os.path.join(model_dir, 'feature_selector.pkl'))
        self.ensemble_weights = np.load(os.path.join(model_dir, 'ensemble_weights.npy'))
        
        # Load the overall model configuration from the JSON
        model_config_params = self.training_config.get('model_config', {})
        main_model_config = ModelConfig(**model_config_params)

        # Determine the feature dimension from the loaded selector
        feature_dim = len(self.feature_selector.selected_features)

        # In the EnhancedModelLoader class, replace the existing for loop with this one.
        for model_info in self.training_config['models']:
            model_name = model_info['name']
            model_type = model_info['type']
            model_path = os.path.join(self.model_dir, f"{model_name}.pth")

            ModelClass = globals()[model_type]
            
            init_signature = inspect.signature(ModelClass.__init__)
            model_args = {
                param: model_info['params'][param]
                for param in init_signature.parameters
                if param in model_info['params']
            }

            # --- START FIX ---
            # Manually inject missing required arguments.
            if 'config' in init_signature.parameters and 'config' not in model_args:
                model_args['config'] = main_model_config

            if 'feature_dim' in init_signature.parameters and 'feature_dim' not in model_args:
                model_args['feature_dim'] = feature_dim
            elif 'input_dim' in init_signature.parameters and 'input_dim' not in model_args:
                model_args['input_dim'] = feature_dim

            # This block converts the list of features into a count, which is what the model expects.
            if 'feature_groups' in init_signature.parameters and 'feature_groups' not in model_args:
                feature_indices = model_info.get('feature_indices', {})
                if feature_indices:
                    # Convert the dictionary of lists to a dictionary of counts (lengths)
                    feature_groups_counts = {k: len(v) for k, v in feature_indices.items()}
                    model_args['feature_groups'] = feature_groups_counts
                else:
                    model_args['feature_groups'] = {} # Pass an empty dict if no indices
            # --- END FIX ---

            model = ModelClass(**model_args)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()

            self.models.append({
                'name': model_name,
                'model': model,
                'type': model_type,
                'feature_indices': model_info.get('feature_indices')
            })
            logging.info(f"Loaded model: {model_name} ({model_type})")

        # Load the stacking ensemble
        try:
            self.stacking_ensemble = joblib.load(os.path.join(model_dir, 'stacking_ensemble.pkl'))
            logging.info("Stacking ensemble loaded successfully.")
        except FileNotFoundError:
            self.stacking_ensemble = None
            logging.warning("Stacking ensemble not found. Will use weighted average.")

    def predict_with_uncertainty(self, features: pd.DataFrame, n_mc_samples: int = 50) -> dict:
        """Generate predictions with proper uncertainty quantification"""
        features_selected = self.feature_selector.transform(features)
        features_scaled = self.scaler.transform(features_selected)
        X_tensor = torch.FloatTensor(features_scaled).to(self.device)

        all_predictions = []
        all_uncertainties = []

        with torch.no_grad():
            for model_info in self.models:
                model = model_info['model']
                model_type = model_info['type']
                feature_indices = model_info['feature_indices']

                if model_type == 'MLBNeuralNetWithUncertainty':
                    pred_mean, total_uncertainty, _, _ = model.predict_with_uncertainty(X_tensor, n_samples=n_mc_samples)
                    all_predictions.append(pred_mean.cpu().numpy())
                    all_uncertainties.append(total_uncertainty.cpu().numpy())
                else:
                    # Monte Carlo dropout for uncertainty
                    mc_predictions = []
                    model.train()
                    
                    # Keep BatchNorm in eval mode
                    for module in model.modules():
                        if isinstance(module, torch.nn.BatchNorm1d):
                            module.eval()
                    
                    for _ in range(n_mc_samples):
                        if model_type == 'MLBNeuralNetV2':
                            pred = model(X_tensor, feature_indices)
                        elif model_type == 'MLBHybridModel':
                            pred, _ = model(X_tensor)
                        else:
                            pred = model(X_tensor)
                        mc_predictions.append(pred.cpu().numpy())
                    
                    model.eval()
                    mc_predictions = np.array(mc_predictions)
                    all_predictions.append(mc_predictions.mean(axis=0))
                    all_uncertainties.append(mc_predictions.std(axis=0))
        
        # First compute base ensemble prediction for fallback
        ensemble_pred = np.average(all_predictions, axis=0, weights=self.ensemble_weights)
        
        # Calculate uncertainties
        weighted_uncertainty = np.average(all_uncertainties, axis=0, weights=self.ensemble_weights)
        model_disagreement = np.std(all_predictions, axis=0)
        total_uncertainty = np.sqrt(weighted_uncertainty**2 + model_disagreement**2)

        # CRITICAL FIX: Trust the stacking ensemble completely
        final_pred = ensemble_pred  # Default to weighted ensemble
        
        if self.stacking_ensemble is not None:
            try:
                # The stacking ensemble was trained to correct base model errors
                # We should trust it completely, not blend it
                stacking_input = pd.DataFrame(features_scaled, columns=features_selected.columns)
                final_pred = self.stacking_ensemble.predict(stacking_input)
                
                # The stacking model doesn't provide uncertainty, so we'll scale
                # the base uncertainty by how much the stacking model disagrees
                stacking_adjustment = np.abs(final_pred - ensemble_pred).mean()
                total_uncertainty = total_uncertainty * (1 + stacking_adjustment)
                
                logging.debug(f"Stacking ensemble adjustment: {stacking_adjustment:.3f}")
                
            except Exception as e:
                logging.warning(f"Stacking ensemble failed, using base ensemble: {e}")
                # Fall back to base ensemble if stacking fails

        return {
            'predictions': final_pred,
            'aleatory_uncertainty': weighted_uncertainty,
            'epistemic_uncertainty': model_disagreement,
            'total_uncertainty': total_uncertainty,
            'individual_predictions': all_predictions
        }

# ============= PROBABILITY CALCULATORS WITH PROPER DISTRIBUTIONS =============
class ProbabilityCalculator:
    """Calculate probabilities using appropriate statistical distributions"""
    
    def __init__(self, use_poisson: bool = True, poisson_adjustment: float = 1.1):
        self.use_poisson = use_poisson
        self.poisson_adjustment = poisson_adjustment
        
    def calculate_win_probability(self, home_score: float, away_score: float,
                                home_uncertainty: float, away_uncertainty: float) -> float:
        """Calculate win probability using appropriate distribution"""
        
        if self.use_poisson:
            # Use Poisson distribution for discrete scoring
            # Adjust lambda for overdispersion
            home_lambda = home_score * self.poisson_adjustment
            away_lambda = away_score * self.poisson_adjustment
            
            # Calculate probability of home winning
            win_prob = 0.0
            
            # Consider reasonable score range (0-20 runs)
            for h in range(21):
                for a in range(21):
                    if h > a:
                        p_home = poisson.pmf(h, home_lambda)
                        p_away = poisson.pmf(a, away_lambda)
                        win_prob += p_home * p_away
            
            # Add tie probability / 2 (for extra innings)
            tie_prob = sum(poisson.pmf(s, home_lambda) * poisson.pmf(s, away_lambda) 
                          for s in range(21))
            win_prob += tie_prob / 2
            
        else:
            # Use normal distribution approximation
            margin = home_score - away_score
            margin_std = np.sqrt(home_uncertainty**2 + away_uncertainty**2)
            win_prob = norm.cdf(0, loc=-margin, scale=margin_std)
        
        return np.clip(win_prob, 0.01, 0.99)
    
    def calculate_total_probability(self, predicted_total: float, line: float,
                                  home_score: float, away_score: float,
                                  uncertainty: float) -> Tuple[float, float]:
        """Calculate over/under probabilities"""
        
        if self.use_poisson:
            # Use sum of Poisson distributions
            home_lambda = home_score * self.poisson_adjustment
            away_lambda = away_score * self.poisson_adjustment
            
            over_prob = 0.0
            
            # Calculate probability of total > line
            for h in range(21):
                for a in range(21):
                    if h + a > line:
                        p_home = poisson.pmf(h, home_lambda)
                        p_away = poisson.pmf(a, away_lambda)
                        over_prob += p_home * p_away
            
            under_prob = 1 - over_prob
            
        else:
            # Normal approximation
            total_std = uncertainty * np.sqrt(2) + 1.0
            over_prob = 1 - norm.cdf(line + 0.5, loc=predicted_total, scale=total_std)
            under_prob = norm.cdf(line - 0.5, loc=predicted_total, scale=total_std)
        
        return over_prob, under_prob
    
    def calculate_margin_distribution(self, home_score: float, away_score: float,
                                    uncertainty: float) -> Dict[float, float]:
        """Calculate probability distribution of winning margins"""
        
        margin_probs = {}
        
        if self.use_poisson:
            home_lambda = home_score * self.poisson_adjustment
            away_lambda = away_score * self.poisson_adjustment
            
            for margin in range(-10, 11):
                prob = 0.0
                for h in range(max(0, margin), 21):
                    a = h - margin
                    if 0 <= a <= 20:
                        p_home = poisson.pmf(h, home_lambda)
                        p_away = poisson.pmf(a, away_lambda)
                        prob += p_home * p_away
                margin_probs[margin] = prob
        
        return margin_probs

# ============= CORRELATION ANALYZER WITH GENERATOR =============
class CorrelationAnalyzer:
    """Analyze and quantify correlations between different bet types"""
    
    def __init__(self):
        self.correlation_matrix = None
        self.historical_correlations = defaultdict(list)
        
    def load_historical_correlations(self, filepath: str):
        """Load pre-computed correlation data"""
        try:
            with open(filepath, 'rb') as f:
                self.correlation_matrix = pickle.load(f)
            logging.info("Loaded historical correlation data")
        except:
            logging.warning("No historical correlation data found")
            self._initialize_default_correlations()
    
    def _initialize_default_correlations(self):
        """Initialize with reasonable default correlations"""
        self.correlation_matrix = {
            ('ml_home', 'over'): 0.15,    # Slight positive correlation
            ('ml_home', 'under'): -0.15,  # Slight negative correlation
            ('ml_away', 'over'): 0.10,
            ('ml_away', 'under'): -0.10,
            ('ml_favorite', 'under'): 0.20,  # Favorites tend to play lower scoring
            ('ml_underdog', 'over'): 0.15,   # Dogs in shootouts
            ('same_game', 'any'): 0.40,      # High correlation for same game
        }
    
    def generate_correlation_matrix(self, historical_results: pd.DataFrame, 
                                  output_path: str = 'correlations.pkl'):
        """Generate correlation matrix from historical betting results"""
        logging.info("Generating correlation matrix from historical data...")
        
        # Group by game_id to analyze same-game correlations
        correlations = {}
        
        # Analyze different bet type combinations
        bet_types = ['ml_home', 'ml_away', 'over', 'under']
        
        for i, type1 in enumerate(bet_types):
            for type2 in bet_types[i:]:
                # Filter relevant bets
                type1_results = historical_results[
                    (historical_results['bet_type'] == type1.split('_')[0]) & 
                    (historical_results['selection'] == type1.split('_')[1] if '_' in type1 else True)
                ]['result'].map({'win': 1, 'loss': 0})
                
                type2_results = historical_results[
                    (historical_results['bet_type'] == type2.split('_')[0]) & 
                    (historical_results['selection'] == type2.split('_')[1] if '_' in type2 else True)
                ]['result'].map({'win': 1, 'loss': 0})
                
                if len(type1_results) > 0 and len(type2_results) > 0:
                    correlation = np.corrcoef(type1_results, type2_results)[0, 1]
                    correlations[(type1, type2)] = correlation
        
        # Same game correlations
        same_game_corr = historical_results.groupby('game_id').apply(
            lambda x: x['result'].map({'win': 1, 'loss': 0}).corr() if len(x) > 1 else 0
        ).mean()
        correlations[('same_game', 'any')] = same_game_corr
        
        # Save to file
        with open(output_path, 'wb') as f:
            pickle.dump(correlations, f)
        
        self.correlation_matrix = correlations
        logging.info(f"Correlation matrix saved to {output_path}")
        
        return correlations
    
    def calculate_parlay_correlation(self, legs: List[EnhancedBet]) -> float:
        """Calculate correlation adjustment for parlay legs"""
        
        # Start with no correlation
        correlation_factor = 1.0
        
        # Group legs by game
        games = defaultdict(list)
        for leg in legs:
            games[leg.game_id].append(leg)
        
        # Heavy penalty for same-game parlays
        for game_id, game_legs in games.items():
            if len(game_legs) > 1:
                # Multiple legs from same game
                correlation_factor *= 0.5
                
                # Additional penalty for opposite bets (e.g., home ML + under)
                bet_types = [f"{leg.bet_type}_{leg.selection}" for leg in game_legs]
                if self._has_conflicting_bets(bet_types):
                    correlation_factor *= 0.7
        
        # Check for systematic correlations
        if len(set(leg.bet_type for leg in legs)) == 1:
            # All same bet type
            if all(leg.bet_type == 'moneyline' for leg in legs):
                # All favorites or all underdogs?
                if all(leg.probability > 0.6 for leg in legs):
                    correlation_factor *= 0.85  # All favorites
                elif all(leg.probability < 0.4 for leg in legs):
                    correlation_factor *= 0.80  # All underdogs
            
            elif all(leg.bet_type == 'total' for leg in legs):
                # All overs or all unders?
                if len(set(leg.selection for leg in legs)) == 1:
                    correlation_factor *= 0.85
        
        return correlation_factor
    
    def _has_conflicting_bets(self, bet_types: List[str]) -> bool:
        """Check if bet types conflict with each other"""
        conflicts = [
            ('moneyline_home', 'total_under'),
            ('moneyline_away', 'total_under'),
        ]
        
        for bet1, bet2 in conflicts:
            if bet1 in bet_types and bet2 in bet_types:
                return True
        
        return False
    
    def update_correlations(self, results: List[Dict]):
        """Update correlation matrix based on actual results"""
        # This would be called periodically to update correlations
        # based on actual betting outcomes
        pass

# ============= VIG REMOVAL =============
class VigRemover:
    """Remove bookmaker vig to find true probabilities"""
    
    @staticmethod
    def remove_vig_decimal(odds_dict: Dict[str, float]) -> Dict[str, float]:
        """Remove vig from decimal odds to get true probabilities"""
        
        # Convert decimal odds to implied probabilities
        implied_probs = {}
        for outcome, decimal_odds in odds_dict.items():
            implied_probs[outcome] = 1 / decimal_odds
        
        # Calculate total implied probability (overround)
        total_implied = sum(implied_probs.values())
        
        # Normalize to get true probabilities
        true_probs = {}
        for outcome, prob in implied_probs.items():
            true_probs[outcome] = prob / total_implied
        
        return true_probs
    
    @staticmethod
    def calculate_vig_decimal(odds_dict: Dict[str, float]) -> float:
        """Calculate the bookmaker's vig percentage from decimal odds"""
        
        implied_probs = {}
        for outcome, decimal_odds in odds_dict.items():
            implied_probs[outcome] = 1 / decimal_odds
        
        total_implied = sum(implied_probs.values())
        vig = (total_implied - 1) / len(odds_dict)
        
        return vig
    
    @staticmethod
    def decimal_to_american(decimal_odds: float) -> float:
        """Convert decimal odds to American odds"""
        if decimal_odds >= 2.0:
            return (decimal_odds - 1) * 100
        else:
            return -100 / (decimal_odds - 1)

class EnhancedPortfolioOptimizer:
    """Enhanced portfolio optimizer with advanced risk management"""
    
    def __init__(self, config: BettingConfig):
        self.config = config
        
    def optimize_portfolio(self, straight_bets: List[EnhancedBet], 
                          parlays: List[EnhancedParlay], 
                          bankroll: float) -> Dict:
        """Optimize betting portfolio with advanced constraints"""
        
        # Sort by risk-adjusted expected value
        straight_bets.sort(
            key=lambda b: (b.edge * b.confidence) / (1 + b.uncertainty), 
            reverse=True
        )
        
        # Diversification constraints
        games_covered = set()
        bet_types_count = {'moneyline': 0, 'total': 0}
        
        # Select best straight bets with diversification
        selected_straights = []
        total_straight_risk = 0
        
        for bet in straight_bets:
            # Diversification checks
            if bet.game_id in games_covered and len(games_covered) < 5:
                continue  # Skip if we already have a bet on this game (early on)
            
            if bet_types_count[bet.bet_type] >= self.config.max_straight_bets * 0.7:
                continue  # Avoid too many of one type
            
            bet_risk = bet.kelly_stake * bankroll
            
            # Risk limit check
            if (total_straight_risk + bet_risk) / bankroll <= self.config.daily_risk_limit * 0.7:
                selected_straights.append(bet)
                total_straight_risk += bet_risk
                games_covered.add(bet.game_id)
                bet_types_count[bet.bet_type] += 1
                
                if len(selected_straights) >= self.config.max_straight_bets:
                    break
        
        # Select best parlays
        selected_parlays = []
        total_parlay_risk = 0
        
        for parlay in parlays:
            parlay_risk = parlay.kelly_stake * bankroll
            
            # Check overlap with straight bets
            overlap_penalty = self._calculate_overlap_penalty(parlay, selected_straights)
            adjusted_risk = parlay_risk * (1 + overlap_penalty)
            
            if (total_straight_risk + total_parlay_risk + adjusted_risk) / bankroll <= self.config.daily_risk_limit:
                selected_parlays.append(parlay)
                total_parlay_risk += parlay_risk
                
                if len(selected_parlays) >= self.config.max_parlays_per_day:
                    break
        
        # Calculate portfolio metrics
        total_risk = total_straight_risk + total_parlay_risk
        
        # Expected return calculation
        straight_ev = sum(
            bet.edge * bet.kelly_stake * bankroll 
            for bet in selected_straights
        )
        
        parlay_ev = sum(
            parlay.expected_value * parlay.kelly_stake * bankroll 
            for parlay in selected_parlays
        )
        
        total_ev = straight_ev + parlay_ev
        
        # Risk-adjusted metrics
        portfolio_variance = self._calculate_portfolio_variance(
            selected_straights, selected_parlays, bankroll
        )
        
        sharpe_ratio = total_ev / np.sqrt(portfolio_variance) if portfolio_variance > 0 else 0
        
        return {
            'straight_bets': selected_straights,
            'parlays': selected_parlays,
            'total_risk': total_risk,
            'risk_percentage': (total_risk / bankroll) * 100,
            'expected_value': total_ev,
            'expected_roi': (total_ev / total_risk) * 100 if total_risk > 0 else 0,
            'sharpe_ratio': sharpe_ratio,
            'games_covered': len(games_covered),
            'straight_bet_sizes': {
                bet.game_id: bet.kelly_stake * bankroll 
                for bet in selected_straights
            },
            'parlay_bet_sizes': [
                parlay.kelly_stake * bankroll 
                for parlay in selected_parlays
            ]
        }
    
    def _calculate_overlap_penalty(self, parlay: EnhancedParlay, 
                                  straight_bets: List[EnhancedBet]) -> float:
        """Calculate penalty for parlays that overlap with straight bets"""
        
        straight_games = {bet.game_id for bet in straight_bets}
        parlay_games = {leg.game_id for leg in parlay.legs}
        
        # Calculate overlap
        overlap = len(straight_games & parlay_games)
        overlap_ratio = overlap / len(parlay_games) if parlay_games else 0
        
        # Penalty increases with overlap
        return overlap_ratio * 0.3
    
    def _calculate_portfolio_variance(self, straight_bets: List[EnhancedBet],
                                    parlays: List[EnhancedParlay],
                                    bankroll: float) -> float:
        """Calculate portfolio variance considering correlations"""
        
        # Simplified variance calculation
        # In production, would use historical covariance matrix
        
        total_variance = 0
        
        # Straight bet variances
        for bet in straight_bets:
            bet_size = bet.kelly_stake * bankroll
            bet_variance = bet_size**2 * bet.uncertainty**2
            total_variance += bet_variance
        
        # Parlay variances (higher due to multiplication)
        for parlay in parlays:
            parlay_size = parlay.kelly_stake * bankroll
            parlay_variance = parlay_size**2 * parlay.combined_uncertainty**2
            total_variance += parlay_variance
        
        return total_variance
    
# ============= BACKTESTING FRAMEWORK =============
# ============= BACKTESTING FRAMEWORK =============
# In BacktestingEngine class
class BacktestingEngine:
    def __init__(self, config: BettingConfig, analyzer: Any): # Add analyzer here
        self.config = config
        self.results = []
        self.performance_metrics = {}
        self.engine = get_db_engine()
        self.feature_pipeline = FeatureEngineeringPipeline(config)
        self.analyzer = analyzer # Use the analyzer that is passed in
        
    def backtest(self, start_date: str, end_date: str, 
                initial_bankroll: float = 10000) -> Dict:
        """Run full backtest over date range"""
        
        logging.info(f"Running backtest from {start_date} to {end_date}")
        
        # Initialize components
        model_loader = EnhancedModelLoader(self.config.model_dir)
        parlay_gen = EnhancedParlayGenerator(self.config)
        optimizer = EnhancedPortfolioOptimizer(self.config)
        
        historical_data = self.feature_pipeline.historical_data
        
        # Initialize tracking for two separate bankrolls
        kelly_bankroll = initial_bankroll / 2
        flat_bankroll = initial_bankroll / 2
        
        logging.info(f"Starting separate bankrolls: Kelly (${kelly_bankroll:,.2f}), Flat (${flat_bankroll:,.2f})")

        daily_results = []
        all_bets = []
        
        current_date = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        
        while current_date <= end_date_dt:
            date_str = current_date.strftime('%Y-%m-%d')
            raw_games_data, odds_data = self._get_historical_data(current_date, historical_data)
            
            if raw_games_data.empty:
                current_date += timedelta(days=1)
                continue
            
            leaky_columns = [
            'game_pk', 'gamePk', 'game_date', 'home_team', 'away_team', 
            'home_team_id', 'away_team_id', 'home_team_abbr', 'away_team_abbr',
            'home_game_date', 'away_game_date', 'home_W/L', 'away_W/L', 'bookmaker',
            'time_match_key', 'date_match_key', 'home_ml', 'away_ml', 'total_line', 'over_odds',
            'home_score', 'away_score', 'match_key', 'diff_score']  # Target columns

            # Create a "pre-game" version of the data by dropping the leaky columns
            pre_game_data = raw_games_data.drop(columns=leaky_columns, errors='ignore')

            # Pass ONLY the clean, pre-game data to the feature pipeline
            games_features = self.feature_pipeline.engineer_features(pre_game_data)
            
            # # --- START: DIAGNOSTIC CODE TO FIND NULLS ---
            # # Check for any columns that have null values
            # null_counts = games_features.isnull().sum()
            # features_with_nulls = null_counts[null_counts > 0]

            # # If there are features with nulls, log a warning
            # if not features_with_nulls.empty:
            #     logging.warning("Null values detected in the following features before fallback fill:")
            #     logging.warning(features_with_nulls.to_string())
            # # --- END: DIAGNOSTIC CODE ---

            # This is the original fix to prevent crashes
            games_features.fillna(0, inplace=True)

            # This block is inside the backtest method's main while loop

            # --- START: THE CRUCIAL FIX ---
            # Keep only the essential columns from the original data for merging.
            columns_to_keep = [
                'game_id', 'home_ml', 'away_ml', 
                'total_line', 'over_odds', 'under_odds', # Assuming under_odds exists
                'home_score', 'away_score'
            ]
            # This block is inside the backtest method's main while loop
            games_data_combined = games_features.copy()
            for col in columns_to_keep:
                if col in raw_games_data.columns and col not in games_data_combined.columns:
                    # THIS IS THE BUG: .values ignores the index, leading to data misalignment
                    games_data_combined[col] = raw_games_data[col].values
            # --- END: THE CRUCIAL FIX ---
            
            # Run betting model using the analyzer stored in the class instance
            ml_bets = self.analyzer.analyze_moneyline(games_data_combined, odds_data)
            total_bets = self.analyzer.analyze_totals(games_data_combined, odds_data)
            all_straight_bets = ml_bets + total_bets
            
            parlays = parlay_gen.generate_parlays(all_straight_bets)
            
            portfolio = optimizer.optimize_portfolio(
                all_straight_bets, parlays, kelly_bankroll # Optimize based on the Kelly bankroll
            )

            # The simulator now calculates PnL for both strategies in a single call
            day_kelly_pnl, day_flat_pnl, day_bets = self._simulate_day(
                portfolio, raw_games_data, flat_bankroll # Pass the flat bankroll for its calculation
            )

            # Update each bankroll with its respective PnL
            kelly_bankroll += day_kelly_pnl
            flat_bankroll += day_flat_pnl
         
            daily_results.append({
                            'date': current_date,
                            'kelly_starting_bankroll': kelly_bankroll - day_kelly_pnl,
                            'kelly_ending_bankroll': kelly_bankroll,
                            'kelly_pnl': day_kelly_pnl,
                            'flat_starting_bankroll': flat_bankroll - day_flat_pnl,
                            'flat_ending_bankroll': flat_bankroll,
                            'flat_pnl': day_flat_pnl,
                            'num_bets': len(day_bets)
                        })
            
            all_bets.extend(day_bets)
            
            current_date += timedelta(days=1)
        
        self.performance_metrics = self._calculate_performance_metrics(
            daily_results, all_bets, initial_bankroll
        )
        
        return self.performance_metrics

    # (The rest of the BacktestingEngine class remains the same...)
    
    def _get_historical_data(self, date: pd.Timestamp, 
                            historical_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Get historical game data and odds for a specific date"""
        
        # Filter games for the date
        date_str = date.strftime('%Y-%m-%d')
        games_data = historical_df[
            pd.to_datetime(historical_df['game_date']).dt.date == date.date()
        ].copy()
        
        if len(games_data) == 0:
            return pd.DataFrame(), {}
        
        # Build odds_data dict from the parquet data itself
        odds_data = {}
        
        for _, game_row in games_data.iterrows():
            # Get game identifier (could be game_id or game_pk)
            game_id = game_row.get('game_id') or game_row.get('game_pk')
            
            # Create odds dict for this game
            odds_dict = {}
            
            # Get moneyline odds (already in the parquet)
            if 'home_ml' in game_row and pd.notna(game_row['home_ml']):
                odds_dict['home_ml'] = game_row['home_ml']
            if 'away_ml' in game_row and pd.notna(game_row['away_ml']):
                odds_dict['away_ml'] = game_row['away_ml']
            
            # Get total odds
            if 'over_odds' in game_row and pd.notna(game_row['over_odds']):
                odds_dict['over_odds'] = game_row['over_odds']
            if 'under_odds' in game_row and pd.notna(game_row['under_odds']):
                odds_dict['under_odds'] = game_row['under_odds']
            if 'total_line' in game_row and pd.notna(game_row['total_line']):
                odds_dict['total_line'] = game_row['total_line']
            
            # Add actual results from the same row
            if 'home_score' in game_row:
                odds_dict['_home_score'] = game_row['home_score']
            if 'away_score' in game_row:
                odds_dict['_away_score'] = game_row['away_score']
            
            # Only add to odds_data if we have some odds
            if any(key in odds_dict for key in ['home_ml', 'away_ml', 'over_odds', 'under_odds']):
                odds_data[game_id] = odds_dict
        
        logging.info(f"Found {len(odds_data)} games with odds data for {date_str}")
        
        return games_data, odds_data
    
    def _simulate_day(self, portfolio: Dict, daily_historical_data: pd.DataFrame, current_flat_bankroll: float) -> Tuple[float, float, List]:
        """Simulate bet results for a day"""
        day_kelly_pnl = 0
        day_flat_pnl = 0
        day_bets = []

        # Ensure game_id in daily_historical_data is a string for matching
        if 'game_id' in daily_historical_data.columns:
            daily_historical_data.loc[:, 'game_id'] = daily_historical_data['game_id'].astype(str)

        # Process straight bets
        for bet in portfolio.get('straight_bets', []):
            kelly_bet_size = portfolio['straight_bet_sizes'].get(bet.game_id)
            flat_bet_size = current_flat_bankroll * self.config.flat_bet_unit

            if kelly_bet_size is None or flat_bet_size <= 0:
                continue
                
            game_data_rows = daily_historical_data[daily_historical_data['game_id'] == str(bet.game_id)]
            
            if game_data_rows.empty:
                logging.warning(f"Could not find game result for {bet.game_id} in simulation.")
                continue

            game_data = game_data_rows.iloc[0]
            home_score = game_data['home_score']
            away_score = game_data['away_score']
            
            # Determine if bet won
            if bet.bet_type == 'moneyline':
                won = (home_score > away_score) if bet.selection == 'home' else (away_score > home_score)
            else:  # total
                total = home_score + away_score
                won = (total > bet.model_prediction['line']) if bet.selection == 'over' else (total < bet.model_prediction['line'])
            
            # Calculate PnL for Kelly
            if won:
                kelly_pnl = kelly_bet_size * (bet.odds / 100) if bet.odds > 0 else kelly_bet_size * (100 / abs(bet.odds))
            else:
                kelly_pnl = -kelly_bet_size
            day_kelly_pnl += kelly_pnl

            # Calculate PnL for Flat
            if won:
                flat_pnl = flat_bet_size * (bet.odds / 100) if bet.odds > 0 else flat_bet_size * (100 / abs(bet.odds))
            else:
                flat_pnl = -flat_bet_size
            day_flat_pnl += flat_pnl

            # Track bet (pnl stored is for Kelly, but result is the same)
            bet.result = 'win' if won else 'loss'
            bet.pnl = kelly_pnl
            bet.bet_size = kelly_bet_size 
            day_bets.append(bet)
        
        # (The parlay logic would be similarly modified if you were using parlays)

        return day_kelly_pnl, day_flat_pnl, day_bets
    
    def _calculate_performance_metrics(self, daily_results: List[Dict],
                                     all_bets: List[EnhancedBet],
                                     initial_bankroll: float) -> Dict:
        """
        Calculate comprehensive performance metrics, including Kelly vs. Flat
        and a detailed breakdown of ROI and Win Rate by bet type for the Kelly portfolio.
        """
        if not daily_results:
            return {}

        results_df = pd.DataFrame(daily_results)
        final_metrics = {}

        # --- High-Level Kelly vs. Flat Comparison ---
        portfolios = {
            'kelly': {'pnl_col': 'kelly_pnl', 'bankroll_col': 'kelly_ending_bankroll'},
            'flat': {'pnl_col': 'flat_pnl', 'bankroll_col': 'flat_ending_bankroll'}
        }
        initial_portfolio_bankroll = initial_bankroll / 2

        for name, cols in portfolios.items():
            total_pnl = results_df[cols['pnl_col']].sum()
            if name == 'kelly':
                total_wagered = sum(b.bet_size for b in all_bets if b.bet_size is not None)
            else: # flat
                total_wagered = (results_df['flat_starting_bankroll'] * self.config.flat_bet_unit * results_df['num_bets']).sum()

            final_metrics[f'{name}_portfolio_total_pnl'] = total_pnl
            final_metrics[f'{name}_portfolio_roi_pct'] = (total_pnl / total_wagered) * 100 if total_wagered > 0 else 0
            final_metrics[f'{name}_portfolio_max_drawdown_pct'] = self._calculate_max_drawdown(results_df[cols['bankroll_col']])

        # --- Detailed Bet Type Breakdown for Kelly Portfolio ---
        pnl_by_type = defaultdict(float)
        wagered_by_type = defaultdict(float)
        wins_by_type = defaultdict(int)
        bets_by_type = defaultdict(int)

        for bet in all_bets:
            category = bet.bet_type
            if category == 'total':
                category = bet.selection # 'over' or 'under'

            # Use the PnL and bet size from the Kelly simulation
            if bet.pnl is not None and bet.bet_size is not None:
                pnl_by_type[category] += bet.pnl
                wagered_by_type[category] += bet.bet_size
                bets_by_type[category] += 1
                if bet.result == 'win':
                    wins_by_type[category] += 1

        # Calculate and add detailed metrics to the final dictionary
        for bet_type in bets_by_type:
            total_wagered = wagered_by_type[bet_type]
            if total_wagered > 0:
                final_metrics[f"{bet_type}_roi_pct"] = (pnl_by_type[bet_type] / total_wagered) * 100
            else:
                final_metrics[f"{bet_type}_roi_pct"] = 0.0

            total_count = bets_by_type[bet_type]
            if total_count > 0:
                 final_metrics[f"{bet_type}_win_rate"] = (wins_by_type[bet_type] / total_count) * 100
            else:
                 final_metrics[f"{bet_type}_win_rate"] = 0.0

        # --- Add Overall Shared Metrics ---
        final_metrics['overall_win_rate'] = len([b for b in all_bets if b.result == 'win']) / len(all_bets) * 100 if all_bets else 0
        final_metrics['total_bets'] = len(all_bets)

        return final_metrics
    
    def _calculate_max_drawdown(self, bankroll_series: pd.Series) -> float:
        """Calculate maximum drawdown percentage"""
        rolling_max = bankroll_series.expanding().max()
        drawdown = (bankroll_series - rolling_max) / rolling_max
        return abs(drawdown.min()) * 100

class BettingTracker:
    """Track all bets and performance"""
    
    def __init__(self, db_path: str = "betting_history.db"):
        self.db_path = db_path
        self._initialize_db()
        
    def _initialize_db(self):
        """Initialize SQLite database for tracking"""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                game_id TEXT,
                bet_type TEXT,
                selection TEXT,
                odds REAL,
                probability REAL,
                edge REAL,
                bet_size REAL,
                confidence REAL,
                uncertainty REAL,
                result TEXT,
                pnl REAL,
                clv REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS parlays (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                legs TEXT,
                combined_odds REAL,
                probability REAL,
                bet_size REAL,
                result TEXT,
                pnl REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_performance (
                date DATE PRIMARY KEY,
                starting_bankroll REAL,
                ending_bankroll REAL,
                total_bets INTEGER,
                wins INTEGER,
                losses INTEGER,
                total_risk REAL,
                total_pnl REAL,
                roi REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_bet(self, bet: EnhancedBet, bet_size: float):
        """Record a placed bet"""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO bets (
                timestamp, game_id, bet_type, selection, odds, 
                probability, edge, bet_size, confidence, uncertainty
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            bet.timestamp, bet.game_id, bet.bet_type, bet.selection,
            bet.odds, bet.probability, bet.edge, bet_size,
            bet.confidence, bet.uncertainty
        ))
        
        conn.commit()
        conn.close()
    
    def update_bet_result(self, game_id: str, result: str, pnl: float, clv: float):
        """Update bet with result"""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE bets 
            SET result = ?, pnl = ?, clv = ?
            WHERE game_id = ? AND result IS NULL
        ''', (result, pnl, clv, game_id))
        
        conn.commit()
        conn.close()
    
    def get_performance_stats(self, days: int = 30) -> Dict:
        """Get performance statistics"""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent bets
        cursor.execute('''
            SELECT * FROM bets 
            WHERE timestamp > datetime('now', '-{} days')
            AND result IS NOT NULL
        '''.format(days))
        
        bets = cursor.fetchall()
        
        if not bets:
            return {}
        
        # Calculate statistics
        total_bets = len(bets)
        wins = sum(1 for bet in bets if bet[11] == 'win')  # result column
        total_pnl = sum(bet[12] for bet in bets if bet[12])  # pnl column
        total_risk = sum(bet[7] for bet in bets)  # bet_size column
        
        # CLV stats
        clv_bets = [bet for bet in bets if bet[13] and bet[13] > 0]  # clv column
        
        conn.close()
        
        return {
            'total_bets': total_bets,
            'win_rate': wins / total_bets if total_bets > 0 else 0,
            'total_pnl': total_pnl,
            'roi': (total_pnl / total_risk * 100) if total_risk > 0 else 0,
            'clv_rate': len(clv_bets) / total_bets if total_bets > 0 else 0,
            'avg_clv': sum(bet[13] for bet in clv_bets) / len(clv_bets) if clv_bets else 0
        }
    
# ============= ENHANCED PARLAY GENERATOR =============
import itertools

class EnhancedParlayGenerator:
    """
    Revamped parlay generator that is safer, more logical, and configurable.
    It only considers profitable legs and uses a more flexible generation strategy.
    """

    def __init__(self, config: 'BettingConfig'):
        self.config = config
        # The correlation analyzer is a great feature, so we keep it.
        self.correlation_analyzer = CorrelationAnalyzer()
        self.correlation_analyzer.load_historical_correlations('correlations.pkl')

    def generate_parlays(self, bets: List['EnhancedBet']) -> List['EnhancedParlay']:
        """Generate profitable parlays from a list of single bets."""
        
        # If parlays are disabled in the config, return immediately.
        if not self.config.enable_parlays:
            return []

        # 1. Select ONLY high-quality legs with a POSITIVE edge and high confidence.
        #    This is the most critical change to prevent losing money.
        parlay_candidates = [
            bet for bet in bets
            if bet.edge >= self.config.min_edge_parlay_leg and
               bet.confidence >= self.config.min_confidence_parlay_leg
        ]

        if len(parlay_candidates) < 2:
            return []

        all_generated_parlays = []

        # 2. Iterate from 2-leg parlays up to the max configured number of legs.
        for num_legs in range(2, self.config.max_parlay_legs + 1):
            
            # Use itertools.combinations for a clean way to get all unique combinations.
            for leg_combination in itertools.combinations(parlay_candidates, num_legs):
                
                # 3. Apply smart filtering rules to the combination.
                # Rule A: Ensure all legs are from different games.
                if len(set(leg.game_id for leg in leg_combination)) != num_legs:
                    continue
                
                # Rule B: Avoid including too many heavy favorites (e.g., > 70% probability).
                heavy_favorites = sum(1 for leg in leg_combination if leg.probability > 0.70)
                if heavy_favorites > 1:
                    continue

                # 4. If the combination passes the rules, create and validate the parlay.
                parlay = self._create_enhanced_parlay(list(leg_combination))
                if self._is_profitable_parlay(parlay):
                    all_generated_parlays.append(parlay)

        # 5. Sort all generated parlays by their expected value and return the best ones.
        all_generated_parlays.sort(key=lambda p: p.expected_value, reverse=True)
        
        # Return double the daily limit to give the portfolio optimizer more choices.
        return all_generated_parlays[:self.config.max_parlays_per_day * 2]

    def _create_enhanced_parlay(self, legs: List['EnhancedBet']) -> 'EnhancedParlay':
        """Calculates the combined odds, probability, and value of a parlay."""
        combined_decimal_odds = 1.0
        for leg in legs:
            if leg.odds > 0:
                decimal_odds = (leg.odds / 100) + 1
            else:
                decimal_odds = (100 / abs(leg.odds)) + 1
            combined_decimal_odds *= decimal_odds
            
        correlation_adj = self.correlation_analyzer.calculate_parlay_correlation(legs)
        
        combined_prob = 1.0
        for leg in legs:
            combined_prob *= leg.probability
        combined_prob *= correlation_adj
        
        combined_uncertainty = np.sqrt(sum(leg.uncertainty**2 for leg in legs))
        ev = (combined_decimal_odds * combined_prob) - 1
        kelly = self._calculate_parlay_kelly(combined_prob, combined_decimal_odds)
        
        return EnhancedParlay(
            legs=legs, combined_odds=combined_decimal_odds,
            combined_probability=combined_prob, expected_value=ev,
            kelly_stake=kelly, correlation_adjustment=correlation_adj,
            combined_uncertainty=combined_uncertainty
        )

    def _calculate_parlay_kelly(self, probability: float, decimal_odds: float) -> float:
        """Calculates a conservative Kelly stake for parlays."""
        if probability <= 0 or decimal_odds <= 1:
            return 0.0
            
        q = 1 - probability
        b = decimal_odds - 1
        kelly = (probability * b - q) / b
        
        # Use the configurable multiplier for conservatism.
        kelly = kelly * self.config.kelly_fraction * self.config.parlay_kelly_multiplier
        return max(0, min(kelly, self.config.max_bet_fraction))

    def _is_profitable_parlay(self, parlay: 'EnhancedParlay') -> bool:
        """Checks if a generated parlay meets all profitability criteria from the config."""
        return (parlay.expected_value > self.config.min_parlay_edge and
                parlay.kelly_stake > 0.001 and # Ensure a non-zero stake
                parlay.combined_probability > 0.05) # Avoid extreme longshots

# Integration with existing analyzer
class CalibratedBettingAnalyzer:
    """Enhanced analyzer with proper calibration"""
    
    def __init__(self, model_loader, config):
        self.model = model_loader
        self.config = config
        self.prob_calculator = ImprovedProbabilityCalculator()
        self.calibrator = EnhancedProbabilityCalibrator()
        self.kelly_calculator = DynamicKellyCalculator(config.kelly_fraction)
        self.confidence_calculator = ImprovedConfidenceCalculator()
        
        # Load pre-fitted calibrators if available
        self._load_calibrators()
    
    def _load_calibrators(self):
        """Load pre-fitted calibration models"""
        import os
        import joblib
        
        calibrator_path = os.path.join(self.config.model_dir, 'calibrators.pkl')
        if os.path.exists(calibrator_path):
            self.calibrator = joblib.load(calibrator_path)
            logging.info("Loaded pre-fitted calibrators")
    
    def analyze_with_calibration(self, game_features: pd.DataFrame, 
                                odds: Dict) -> List['EnhancedBet']:
        """Analyze bets with proper calibration"""
        bets = []
        
        # Get raw predictions
        predictions = self.model.predict_with_uncertainty(game_features)
        
        for idx, (_, game) in enumerate(game_features.iterrows()):
            game_id = game.get('game_id', game.get('game_pk', f'game_{idx}'))
            
            # Raw model predictions
            pred_home = predictions['predictions'][idx][0]
            pred_away = predictions['predictions'][idx][1]
            uncertainty = predictions['total_uncertainty'][idx].mean()
            
            # Calculate raw probabilities with Negative Binomial
            raw_home_prob = self.prob_calculator.calculate_win_probability_nbinom(
                pred_home, pred_away, uncertainty, uncertainty
            )
            
            # Apply calibration
            calibrated_home_prob = self.calibrator.calibrate_probability(
                raw_home_prob, 'moneyline'
            )
            calibrated_away_prob = 1 - calibrated_home_prob
            
            # Calculate confidence with improved method
            confidence = self.confidence_calculator.calculate_confidence(
                predictions, idx, calibrated_home_prob, uncertainty
            )
            
            # Continue with betting logic using calibrated probabilities...
            # (Rest of the betting analysis logic)
            
        return bets
        
# ============= ENHANCED BETTING ANALYZER =============
class EnhancedBettingAnalyzer:
    """Enhanced analyzer with proper distributions and vig removal"""
    
    def __init__(self, model_loader: EnhancedModelLoader, config: BettingConfig):
        self.model = model_loader
        self.config = config
        self.prob_calculator = ProbabilityCalculator(
            use_poisson=config.use_poisson,
            poisson_adjustment=config.poisson_adjustment
        )
        self.vig_remover = VigRemover()
        self.correlation_analyzer = CorrelationAnalyzer()
        
    def analyze_moneyline(self, game_features: pd.DataFrame, odds: Dict) -> List[EnhancedBet]:
        """Analyze moneyline betting opportunities with enhanced methods"""
        bets = []
        
        # Get model predictions with uncertainty
        predictions = self.model.predict_with_uncertainty(game_features)
        
        for idx, (_, game) in enumerate(game_features.iterrows()):
            # Get game_id - check multiple possible column names
            game_id = game.get('game_id', game.get('game_pk', f'game_{idx}'))
            
            # Model predictions
            pred_home_score = predictions['predictions'][idx][0]
            pred_away_score = predictions['predictions'][idx][1]
            home_uncertainty = predictions['total_uncertainty'][idx][0]
            away_uncertainty = predictions['total_uncertainty'][idx][1]
            
            # Calculate win probabilities using proper distribution
            home_win_prob = self.prob_calculator.calculate_win_probability(
                pred_home_score, pred_away_score,
                home_uncertainty, away_uncertainty
            )
            away_win_prob = 1 - home_win_prob
            
            # Get odds directly from the game data (already in decimal format)
            home_ml_decimal = game.get('home_ml')
            away_ml_decimal = game.get('away_ml')
            
            if pd.notna(home_ml_decimal) and pd.notna(away_ml_decimal):
                # Convert to float
                home_ml_decimal = float(home_ml_decimal)
                away_ml_decimal = float(away_ml_decimal)
                
                # Remove vig to get true market probabilities
                market_probs = self.vig_remover.remove_vig_decimal({
                    'home': home_ml_decimal,
                    'away': away_ml_decimal
                })
                
                # Calculate edges against true market probabilities
                home_edge = home_win_prob - market_probs['home']
                away_edge = away_win_prob - market_probs['away']
                
                # Convert to American odds for display
                home_ml_american = self.vig_remover.decimal_to_american(home_ml_decimal)
                away_ml_american = self.vig_remover.decimal_to_american(away_ml_decimal)
                
                # Check for value on home team
                if home_edge > self.config.min_edge_moneyline:
                    confidence = self._calculate_confidence_score(
                        predictions, idx, home_win_prob, home_uncertainty
                    )
                    
                    if confidence > self.config.min_model_confidence:
                        kelly_stake = self._calculate_kelly_stake_decimal(
                            home_win_prob, home_ml_decimal, confidence
                        )
                        
                        bets.append(EnhancedBet(
                            game_id=str(game_id),
                            bet_type='moneyline',
                            selection='home',
                            odds=home_ml_american,
                            probability=home_win_prob,
                            edge=home_edge,
                            kelly_stake=kelly_stake,
                            confidence=confidence,
                            uncertainty=home_uncertainty,
                            model_prediction={
                                'home_score': pred_home_score,
                                'away_score': pred_away_score,
                                'margin': pred_home_score - pred_away_score,
                                'margin_distribution': self.prob_calculator.calculate_margin_distribution(
                                    pred_home_score, pred_away_score, 
                                    (home_uncertainty + away_uncertainty) / 2
                                )
                            }
                        ))
                
                # Check for value on away team
                if away_edge > self.config.min_edge_moneyline:
                    confidence = self._calculate_confidence_score(
                        predictions, idx, away_win_prob, away_uncertainty
                    )
                    
                    if confidence > self.config.min_model_confidence:
                        kelly_stake = self._calculate_kelly_stake_decimal(
                            away_win_prob, away_ml_decimal, confidence
                        )
                        
                        bets.append(EnhancedBet(
                            game_id=str(game_id),
                            bet_type='moneyline',
                            selection='away',
                            odds=away_ml_american,
                            probability=away_win_prob,
                            edge=away_edge,
                            kelly_stake=kelly_stake,
                            confidence=confidence,
                            uncertainty=away_uncertainty,
                            model_prediction={
                                'home_score': pred_home_score,
                                'away_score': pred_away_score,
                                'margin': pred_home_score - pred_away_score,
                                'margin_distribution': self.prob_calculator.calculate_margin_distribution(
                                    pred_home_score, pred_away_score,
                                    (home_uncertainty + away_uncertainty) / 2
                                )
                            }
                        ))
        
        return bets
    
    def analyze_totals(self, game_features: pd.DataFrame, odds: Dict) -> List[EnhancedBet]:
        """Analyze over/under betting opportunities with enhanced methods"""
        bets = []
        
        # Get model predictions with uncertainty
        predictions = self.model.predict_with_uncertainty(game_features)
        
        for idx, (_, game) in enumerate(game_features.iterrows()):
            game_id = game.get('game_id', game.get('game_pk', f'game_{idx}'))
            
            # Model predictions
            pred_home_score = predictions['predictions'][idx][0]
            pred_away_score = predictions['predictions'][idx][1]
            pred_total = pred_home_score + pred_away_score
            total_uncertainty = predictions['total_uncertainty'][idx].mean()
            
            # Get odds from game data
            total_line = game.get('total_line')
            over_odds_decimal = game.get('over_odds')
            
            if pd.notna(total_line) and pd.notna(over_odds_decimal):
                # Calculate probabilities using proper distribution
                over_prob, under_prob = self.prob_calculator.calculate_total_probability(
                    pred_total, float(total_line),
                    pred_home_score, pred_away_score,
                    total_uncertainty
                )
                
                # Since we don't have under odds, assume standard -110/-110
                under_odds_decimal = 1.909  # -110 in decimal
                
                # Remove vig from totals market
                market_probs = self.vig_remover.remove_vig_decimal({
                    'over': float(over_odds_decimal),
                    'under': under_odds_decimal
                })
                
                # Calculate edges
                over_edge = over_prob - market_probs['over']
                under_edge = under_prob - market_probs['under']
                
                # Convert to American odds for display
                over_odds_american = self.vig_remover.decimal_to_american(float(over_odds_decimal))
                under_odds_american = -110  # Standard
                
                # Check for value on over
                if over_edge > self.config.min_edge_totals:
                    confidence = self._calculate_confidence_score(
                        predictions, idx, over_prob, total_uncertainty
                    )
                    
                    if confidence > self.config.min_model_confidence:
                        kelly_stake = self._calculate_kelly_stake_decimal(
                            over_prob, float(over_odds_decimal), confidence
                        )
                        
                        bets.append(EnhancedBet(
                            game_id=str(game_id),
                            bet_type='total',
                            selection='over',
                            odds=over_odds_american,
                            probability=over_prob,
                            edge=over_edge,
                            kelly_stake=kelly_stake,
                            confidence=confidence,
                            uncertainty=total_uncertainty,
                            model_prediction={
                                'predicted_total': pred_total,
                                'line': float(total_line),
                                'difference': pred_total - float(total_line),
                                'home_score': pred_home_score,
                                'away_score': pred_away_score
                            }
                        ))
                
                # Check for value on under
                if under_edge > self.config.min_edge_totals:
                    confidence = self._calculate_confidence_score(
                        predictions, idx, under_prob, total_uncertainty
                    )
                    
                    if confidence > self.config.min_model_confidence:
                        kelly_stake = self._calculate_kelly_stake_decimal(
                            under_prob, under_odds_decimal, confidence
                        )
                        
                        bets.append(EnhancedBet(
                            game_id=str(game_id),
                            bet_type='total',
                            selection='under',
                            odds=under_odds_american,
                            probability=under_prob,
                            edge=under_edge,
                            kelly_stake=kelly_stake,
                            confidence=confidence,
                            uncertainty=total_uncertainty,
                            model_prediction={
                                'predicted_total': pred_total,
                                'line': float(total_line),
                                'difference': pred_total - float(total_line),
                                'home_score': pred_home_score,
                                'away_score': pred_away_score
                            }
                        ))
        
        return bets
    
    def _calculate_confidence_score(self, predictions: Dict, idx: int, 
                                  probability: float, uncertainty: float) -> float:
        """Calculate refined confidence score based on uncertainty"""
        
        # Base confidence from model uncertainty (lower uncertainty = higher confidence)
        uncertainty_confidence = 1 / (1 + uncertainty)
        
        # Edge confidence (stronger edge = higher confidence)
        edge_confidence = min(abs(probability - 0.5) * 2, 1)
        
        # Model agreement (if using ensemble)
        if 'individual_predictions' in predictions:
            predictions_array = np.array(predictions['individual_predictions'])
            model_std = predictions_array.std(axis=0)[idx].mean()
            agreement_confidence = 1 / (1 + model_std)
        else:
            agreement_confidence = 0.5
        
        # Combined confidence with weights
        confidence = (
            0.4 * uncertainty_confidence +
            0.3 * edge_confidence +
            0.3 * agreement_confidence
        )
        
        return np.clip(confidence, 0, 1)
    
    def _calculate_kelly_stake_decimal(self, probability: float, decimal_odds: float, 
                                     confidence: float) -> float:
        """Calculate Kelly stake with confidence adjustment for decimal odds"""
        
        # Kelly formula for decimal odds
        q = 1 - probability
        b = decimal_odds - 1
        kelly = (probability * b - q) / b
        
        # Adjust by confidence
        kelly = kelly * confidence
        
        # Apply Kelly fraction
        kelly = kelly * self.config.kelly_fraction
        
        # Cap at maximum
        kelly = max(0, min(kelly, self.config.max_bet_fraction))
        
        return kelly

# ============= INTEGRATED CALIBRATED BETTING ANALYZER =============
class CalibratedBettingAnalyzer(EnhancedBettingAnalyzer):
    """Enhanced analyzer with proper calibration and improved probability calculations"""
    
    def __init__(self, model_loader: EnhancedModelLoader, config: BettingConfig):
        super().__init__(model_loader, config)
        # Replace the standard calculator with the improved one
        self.prob_calculator = ImprovedProbabilityCalculator(overdispersion_factor=0.5)
        self.calibrator = EnhancedProbabilityCalibrator()
        self.kelly_calculator = DynamicKellyCalculator(config.kelly_fraction)
        self.confidence_calculator = ImprovedConfidenceCalculator()
        
        # Load pre-fitted calibrators if available
        self._load_calibrators()
    
    def _load_calibrators(self):
        """Load pre-fitted calibration models"""
        import os
        import joblib
        
        calibrator_path = os.path.join(self.config.model_dir, 'calibrators.pkl')
        if os.path.exists(calibrator_path):
            try:
                self.calibrator = joblib.load(calibrator_path)
                logging.info("Loaded pre-fitted calibrators")
            except Exception as e:
                logging.warning(f"Could not load calibrators: {e}")
    
    def analyze_moneyline(self, game_features: pd.DataFrame, odds: Dict) -> List[EnhancedBet]:
        bets = []
        predictions = self.model.predict_with_uncertainty(game_features)

        for idx, (_, game) in enumerate(game_features.iterrows()):
            game_id = str(game.get('game_id', game.get('game_pk', f'game_{idx}')))
            
            pred_home_score = predictions['predictions'][idx][0]
            pred_away_score = predictions['predictions'][idx][1]
            home_uncertainty = predictions['total_uncertainty'][idx][0]
            away_uncertainty = predictions['total_uncertainty'][idx][1]

            # --- START: THE FIX (Sanity Check 1) ---
            # Check for NaN or infinity in the model's raw output.
            if not all(np.isfinite(val) for val in [pred_home_score, pred_away_score, home_uncertainty, away_uncertainty]):
                logging.warning(f"Skipping game {game_id} (Moneyline) due to non-finite model prediction.")
                continue  # Skip to the next game
            # --- END: THE FIX (Sanity Check 1) ---

            raw_home_prob = self.prob_calculator.calculate_win_probability_nbinom(
                pred_home_score, pred_away_score, home_uncertainty, away_uncertainty
            )

            # --- START: THE FIX (Sanity Check 2) ---
            # Check for NaN after the probability calculation.
            if not np.isfinite(raw_home_prob):
                logging.warning(f"Skipping game {game_id} (Moneyline) due to NaN raw probability.")
                continue  # Skip to the next game
            # --- END: THE FIX (Sanity Check 2) ---
            
            # Apply calibration
            home_win_prob = self.calibrator.calibrate_probability(
                raw_home_prob, 'moneyline'
            )
            away_win_prob = 1 - home_win_prob
            
            # Get odds
            home_ml_decimal = game.get('home_ml')
            away_ml_decimal = game.get('away_ml')
            
            if pd.notna(home_ml_decimal) and pd.notna(away_ml_decimal):
                home_ml_decimal = float(home_ml_decimal)
                away_ml_decimal = float(away_ml_decimal)
                
                # Remove vig to get true market probabilities
                market_probs = self.vig_remover.remove_vig_decimal({
                    'home': home_ml_decimal,
                    'away': away_ml_decimal
                })
                
                # Calculate edges against true market probabilities
                home_edge = home_win_prob - market_probs['home']
                away_edge = away_win_prob - market_probs['away']
                
                # Convert to American odds for display
                home_ml_american = self.vig_remover.decimal_to_american(home_ml_decimal)
                away_ml_american = self.vig_remover.decimal_to_american(away_ml_decimal)
                
                # Calculate confidence with improved method
                confidence = self.confidence_calculator.calculate_confidence(
                    predictions, idx, home_win_prob, home_uncertainty
                )
                
                # Check for value on home team
                if home_edge > self.config.min_edge_moneyline and confidence > self.config.min_confidence_moneyline:
                    kelly_stake = self.kelly_calculator.calculate_dynamic_kelly(
                        home_win_prob, home_ml_decimal, confidence, 'moneyline'
                    )
                    
                    # if kelly_stake > 0:
                    bets.append(EnhancedBet(
                        game_id=str(game_id),
                        bet_type='moneyline',
                        selection='home',
                        odds=home_ml_american,
                        probability=home_win_prob,
                        edge=home_edge,
                        kelly_stake=kelly_stake,
                        confidence=confidence,
                        uncertainty=home_uncertainty,
                        model_prediction={
                            'home_score': pred_home_score,
                            'away_score': pred_away_score,
                            'margin': pred_home_score - pred_away_score,
                            'raw_probability': raw_home_prob,
                            'calibrated_probability': home_win_prob
                        }
                    ))
                
                # Check for value on away team
                confidence_away = self.confidence_calculator.calculate_confidence(
                    predictions, idx, away_win_prob, away_uncertainty
                )
                
                if away_edge > self.config.min_edge_moneyline and confidence_away > self.config.min_confidence_moneyline:
                    kelly_stake = self.kelly_calculator.calculate_dynamic_kelly(
                        away_win_prob, away_ml_decimal, confidence_away, 'moneyline'
                    )
                    
                    # if kelly_stake > 0:
                    bets.append(EnhancedBet(
                        game_id=str(game_id),
                        bet_type='moneyline',
                        selection='away',
                        odds=away_ml_american,
                        probability=away_win_prob,
                        edge=away_edge,
                        kelly_stake=kelly_stake,
                        confidence=confidence_away,
                        uncertainty=away_uncertainty,
                        model_prediction={
                            'home_score': pred_home_score,
                            'away_score': pred_away_score,
                            'margin': pred_home_score - pred_away_score,
                            'raw_probability': 1 - raw_home_prob,
                            'calibrated_probability': away_win_prob
                        }
                    ))
        
        return bets
    
    # In the CalibratedBettingAnalyzer class...
    def analyze_totals(self, game_features: pd.DataFrame, odds: Dict) -> List[EnhancedBet]:
        bets = []
        predictions = self.model.predict_with_uncertainty(game_features)

        for idx, (_, game) in enumerate(game_features.iterrows()):
            game_id = str(game.get('game_id', game.get('game_pk', f'game_{idx}')))
            
            pred_home_score = predictions['predictions'][idx][0]
            pred_away_score = predictions['predictions'][idx][1]
            total_uncertainty = predictions['total_uncertainty'][idx].mean()

            # Sanity Check 1: Validate the raw output from the neural network model.
            if not all(np.isfinite(val) for val in [pred_home_score, pred_away_score, total_uncertainty]):
                logging.warning(f"Skipping game {game_id} (Totals) due to non-finite model prediction.")
                continue

            total_line = game.get('total_line')
            over_odds_decimal = game.get('over_odds')

            if pd.notna(total_line) and pd.notna(over_odds_decimal):
                pred_total = pred_home_score + pred_away_score
                raw_over_prob, _ = self.prob_calculator.calculate_total_probability_nbinom(
                    pred_total, float(total_line), pred_home_score, pred_away_score, total_uncertainty
                )

                # Sanity Check 2: Validate the output of the probability calculation.
                if not np.isfinite(raw_over_prob):
                    logging.warning(f"Skipping game {game_id} (Totals) due to NaN raw probability.")
                    continue
                
                # This line fixes the previous NameError bug.
                raw_under_prob = 1 - raw_over_prob

                # Apply calibration to get the final probabilities.
                over_prob = self.calibrator.calibrate_probability(raw_over_prob, 'total')
                under_prob = 1 - over_prob
                
                # --- OVER BET ANALYSIS ---
                confidence_over = self.confidence_calculator.calculate_confidence(
                    predictions, idx, over_prob, total_uncertainty
                )
                market_probs = self.vig_remover.remove_vig_decimal({'over': float(over_odds_decimal), 'under': 1.909})
                over_edge = over_prob - market_probs.get('over', 0.5)

                if over_edge > self.config.min_edge_totals and confidence_over > self.config.min_confidence_over:
                    kelly_stake = self.kelly_calculator.calculate_dynamic_kelly(
                        over_prob, float(over_odds_decimal), confidence_over, 'total'
                    )
                    # if kelly_stake > 0:
                    bets.append(EnhancedBet(
                        game_id=game_id, bet_type='total', selection='over',
                        odds=self.vig_remover.decimal_to_american(float(over_odds_decimal)),
                        probability=over_prob, edge=over_edge, kelly_stake=kelly_stake,
                        confidence=confidence_over, uncertainty=total_uncertainty,
                        model_prediction={'predicted_total': pred_total, 'line': float(total_line), 
                                        'raw_probability': raw_over_prob, 'calibrated_probability': over_prob}
                    ))

                # --- UNDER BET ANALYSIS ---
                confidence_under = self.confidence_calculator.calculate_confidence(
                    predictions, idx, under_prob, total_uncertainty
                )
                under_odds_decimal = 1.909  # Assume standard -110 odds
                under_edge = under_prob - market_probs.get('under', 0.5)

                if under_edge > self.config.min_edge_totals and confidence_under > self.config.min_confidence_under:
                    kelly_stake = self.kelly_calculator.calculate_dynamic_kelly(
                        under_prob, under_odds_decimal, confidence_under, 'total'
                    )
                    # if kelly_stake > 0:
                    bets.append(EnhancedBet(
                        game_id=game_id, bet_type='total', selection='under',
                        odds=-110, probability=under_prob, edge=under_edge,
                        kelly_stake=kelly_stake, confidence=confidence_under, uncertainty=total_uncertainty,
                        model_prediction={'predicted_total': pred_total, 'line': float(total_line),
                                        'raw_probability': raw_under_prob, 'calibrated_probability': under_prob}
                    ))
        
        return bets
    
# ============= MAIN BETTING MODEL V2 =============
class MLBBettingModelV2:
    """Enhanced main betting model with all professional features"""
    
    def __init__(self, config: BettingConfig):
        self.config = config
        self.model_loader = EnhancedModelLoader(config.model_dir)
        self.feature_pipeline = FeatureEngineeringPipeline(config)
        self.analyzer = CalibratedBettingAnalyzer(self.model_loader, config)
        self.parlay_generator = EnhancedParlayGenerator(config)
        
        # Use the new professional optimizer instead
        self.optimizer = ProfessionalPortfolioOptimizer(config)
        
        self.tracker = BettingTracker()
        
        # Load optimized parameters if they exist
        self._load_optimized_parameters()
        
        logging.info("MLB Betting Model V2 initialized successfully")
    
    def _load_optimized_parameters(self):
        """Load optimized confidence and Kelly parameters if available"""
        
        # Load optimized confidence calculator
        conf_path = os.path.join(self.config.model_dir, 'optimized_confidence.pkl')
        if os.path.exists(conf_path):
            optimized_conf = OptimizedConfidenceCalculator()
            optimized_conf.load(conf_path)
            self.analyzer.confidence_calculator = optimized_conf
            logging.info("Loaded optimized confidence calculator")
        
        # Load optimized Kelly calculator
        kelly_path = os.path.join(self.config.model_dir, 'optimized_kelly.pkl')
        if os.path.exists(kelly_path):
            optimized_kelly = joblib.load(kelly_path)
            self.analyzer.kelly_calculator = optimized_kelly
            logging.info("Loaded optimized Kelly calculator")
    
    def optimize_parameters(self, start_date: str = "2022-04-01", 
                          end_date: str = "2024-03-31") -> Dict:
        """Optimize confidence and Kelly parameters on historical data"""
        
        logging.info(f"Optimizing parameters from {start_date} to {end_date}")
        
        # First, we need to generate historical betting data with features
        historical_data = self._generate_historical_betting_data(start_date, end_date)
        
        if historical_data.empty:
            logging.error("No historical data available for optimization")
            return {}
        
        # Optimize confidence calculator
        logging.info("Optimizing confidence calculator...")
        conf_optimizer = OptimizedConfidenceCalculator()
        conf_results = conf_optimizer.optimize_weights(historical_data)
        conf_optimizer.save(os.path.join(self.config.model_dir, 'optimized_confidence.pkl'))
        
        # Optimize Kelly calculator
        logging.info("Optimizing Kelly calculator...")
        kelly_optimizer = OptimizedKellyCalculator(self.config.kelly_fraction)
        kelly_results = kelly_optimizer.optimize_thresholds(historical_data)
        joblib.dump(kelly_optimizer, os.path.join(self.config.model_dir, 'optimized_kelly.pkl'))
        
        # Update the analyzer with optimized components
        self.analyzer.confidence_calculator = conf_optimizer
        self.analyzer.kelly_calculator = kelly_optimizer
        
        logging.info("Parameter optimization complete!")
        
        return {
            'confidence_optimization': conf_results,
            'kelly_optimization': kelly_results
        }
    
    def _generate_historical_betting_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate historical betting data for parameter optimization"""
        
        logging.info("Generating historical betting data for optimization...")
        
        all_bets = []
        
        current_date = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        
        while current_date <= end_date_dt:
            date_str = current_date.strftime('%Y-%m-%d')
            
            try:
                # Fetch historical games
                raw_games = self.feature_pipeline.fetch_game_data(date_str)
                
                if raw_games.empty:
                    current_date += timedelta(days=1)
                    continue
                
                # Prepare features (remove target columns)
                leaky_columns = ['home_score', 'away_score']
                games_for_features = raw_games.drop(columns=leaky_columns, errors='ignore')
                
                # Engineer features
                game_features = self.feature_pipeline.engineer_features(games_for_features)
                game_features.fillna(0, inplace=True)
                
                # Add back necessary columns
                for col in ['game_id', 'home_ml', 'away_ml', 'total_line', 'over_odds']:
                    if col in raw_games.columns:
                        game_features[col] = raw_games[col].values
                
                # Get predictions
                predictions = self.model.predict_with_uncertainty(game_features)
                
                # Analyze each game
                for idx, (_, game) in enumerate(game_features.iterrows()):
                    game_id = game.get('game_id', game.get('game_pk'))
                    
                    # Get actual results
                    actual_game = raw_games[raw_games['game_id'] == game_id].iloc[0]
                    home_score = actual_game['home_score']
                    away_score = actual_game['away_score']
                    
                    # Model predictions
                    pred_home = predictions['predictions'][idx][0]
                    pred_away = predictions['predictions'][idx][1]
                    uncertainty = predictions['total_uncertainty'][idx].mean()
                    model_std = predictions['epistemic_uncertainty'][idx].mean()
                    
                    # Calculate probabilities and edges
                    home_prob = self.analyzer.prob_calculator.calculate_win_probability(
                        pred_home, pred_away, uncertainty, uncertainty
                    )
                    
                    # Market odds
                    home_ml = game.get('home_ml', 2.0)
                    away_ml = game.get('away_ml', 2.0)
                    
                    if pd.notna(home_ml) and home_ml > 0:
                        market_prob = 1 / home_ml
                        edge = home_prob - market_prob
                        
                        # Create bet record
                        bet_record = {
                            'game_id': game_id,
                            'bet_type': 'moneyline',
                            'selection': 'home',
                            'probability': home_prob,
                            'odds': home_ml,
                            'edge': edge,
                            'uncertainty': uncertainty,
                            'model_std': model_std,
                            'won': home_score > away_score,
                            'historical_accuracy': 0.5  # Placeholder
                        }
                        
                        all_bets.append(bet_record)
                    
                    # Also check totals
                    total_line = game.get('total_line')
                    if pd.notna(total_line):
                        pred_total = pred_home + pred_away
                        over_prob, _ = self.analyzer.prob_calculator.calculate_total_probability(
                            pred_total, float(total_line), pred_home, pred_away, uncertainty
                        )
                        
                        over_odds = game.get('over_odds', 1.909)
                        if pd.notna(over_odds) and over_odds > 0:
                            market_prob = 1 / over_odds
                            edge = over_prob - market_prob
                            
                            bet_record = {
                                'game_id': game_id,
                                'bet_type': 'total',
                                'selection': 'over',
                                'probability': over_prob,
                                'odds': over_odds,
                                'edge': edge,
                                'uncertainty': uncertainty,
                                'model_std': model_std,
                                'won': (home_score + away_score) > total_line,
                                'historical_accuracy': 0.5
                            }
                            
                            all_bets.append(bet_record)
            
            except Exception as e:
                logging.warning(f"Error processing {date_str}: {e}")
            
            current_date += timedelta(days=1)
        
        historical_df = pd.DataFrame(all_bets)
        
        # Add confidence scores using current calculator
        if not historical_df.empty:
            confidences = []
            for _, bet in historical_df.iterrows():
                conf = self.analyzer.confidence_calculator._calculate_confidence_score(
                    {'total_uncertainty': [[bet['uncertainty']]], 
                     'epistemic_uncertainty': [[bet['model_std']]]},
                    0, bet['probability'], bet['uncertainty']
                )
                confidences.append(conf)
            
            historical_df['confidence'] = confidences
        
        logging.info(f"Generated {len(historical_df)} historical bet records")
        
        return historical_df
    
    def visualize_calibration(self, start_date: str = "2024-04-01", 
                            end_date: str = "2024-06-30") -> Dict:
        """Generate calibration visualizations"""
        
        logging.info(f"Generating calibration visualizations from {start_date} to {end_date}")
        
        # Collect predictions and actuals
        ml_predictions = []
        ml_actuals = []
        ml_calibrated = []
        
        total_predictions = []
        total_actuals = []
        total_calibrated = []
        
        current_date = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        
        while current_date <= end_date_dt:
            date_str = current_date.strftime('%Y-%m-%d')
            
            try:
                raw_games = self.feature_pipeline.fetch_game_data(date_str)
                
                if raw_games.empty:
                    current_date += timedelta(days=1)
                    continue
                
                # Process games
                leaky_columns = ['home_score', 'away_score']
                games_for_features = raw_games.drop(columns=leaky_columns, errors='ignore')
                game_features = self.feature_pipeline.engineer_features(games_for_features)
                game_features.fillna(0, inplace=True)
                
                # Add odds back
                for col in ['game_id', 'home_ml', 'away_ml', 'total_line', 'over_odds']:
                    if col in raw_games.columns:
                        game_features[col] = raw_games[col].values
                
                # Get predictions
                # THIS IS THE CORRECTED LINE:
                predictions = self.model_loader.predict_with_uncertainty(game_features)
                
                for idx, (_, game) in enumerate(game_features.iterrows()):
                    game_id = game.get('game_id')
                    actual_game = raw_games[raw_games['game_id'] == game_id].iloc[0]
                    home_score = actual_game['home_score']
                    away_score = actual_game['away_score']
                    
                    # Moneyline
                    pred_home = predictions['predictions'][idx][0]
                    pred_away = predictions['predictions'][idx][1]
                    uncertainty = predictions['total_uncertainty'][idx].mean()
                    
                    raw_prob = self.analyzer.prob_calculator.calculate_win_probability_nbinom(
                        pred_home, pred_away, uncertainty, uncertainty
                    )
                    
                    calibrated_prob = self.analyzer.calibrator.calibrate_probability(
                        raw_prob, 'moneyline'
                    )
                    
                    ml_predictions.append(raw_prob)
                    ml_calibrated.append(calibrated_prob)
                    ml_actuals.append(1 if home_score > away_score else 0)
                    
                    # Totals
                    total_line = game.get('total_line')
                    if pd.notna(total_line):
                        pred_total = pred_home + pred_away
                        over_prob, _ = self.analyzer.prob_calculator.calculate_total_probability_nbinom(
                            pred_total, float(total_line), pred_home, pred_away, uncertainty
                        )
                        
                        cal_over_prob = self.analyzer.calibrator.calibrate_probability(
                            over_prob, 'total'
                        )
                        
                        total_predictions.append(over_prob)
                        total_calibrated.append(cal_over_prob)
                        total_actuals.append(1 if (home_score + away_score) > total_line else 0)
            
            except Exception as e:
                logging.warning(f"Error processing {date_str} for visualization: {e}")
            
            current_date += timedelta(days=1)
        
        # Create visualizations
        visualizer = CalibrationVisualizer()
        
        # Moneyline calibration
        if ml_predictions:
            ml_fig = visualizer.plot_reliability_diagram(
                np.array(ml_calibrated), np.array(ml_actuals),
                title="Moneyline Calibration (Calibrated)"
            )
            ml_fig.savefig(os.path.join(self.config.model_dir, 'moneyline_calibration.png'))
            plt.close(ml_fig)
            
            # Comparison plot
            ml_comp_fig = visualizer.plot_calibration_comparison(
                np.array(ml_predictions), np.array(ml_calibrated), np.array(ml_actuals)
            )
            ml_comp_fig.savefig(os.path.join(self.config.model_dir, 'ml_calibration_comparison.png'))
            plt.close(ml_comp_fig)
        
        # Totals calibration
        if total_predictions:
            total_fig = visualizer.plot_reliability_diagram(
                np.array(total_calibrated), np.array(total_actuals),
                title="Totals Calibration (Calibrated)"
            )
            total_fig.savefig(os.path.join(self.config.model_dir, 'totals_calibration.png'))
            plt.close(total_fig)
        
        logging.info("Calibration visualizations saved to model directory")
        
        return {
            'moneyline_samples': len(ml_predictions),
            'totals_samples': len(total_predictions),
            'plots_saved': True
        }

# Add this import at the top of your script
from sklearn.model_selection import train_test_split

class CalibrationTrainer:
    """Train calibration models on historical data"""
    
    def __init__(self, config: BettingConfig):
        self.config = config
        self.calibrator = EnhancedProbabilityCalibrator()
        
    def train_calibrators(self, start_date: str, end_date: str, 
                          validation_split: float = 0.2) -> Dict:
        """Train calibrators on historical betting data with a validation set"""
        
        logging.info(f"Training calibrators from {start_date} to {end_date}")
        
        # This initial data collection loop remains the same
        model_loader = EnhancedModelLoader(self.config.model_dir)
        feature_pipeline = FeatureEngineeringPipeline(self.config)
        analyzer = EnhancedBettingAnalyzer(model_loader, self.config)
        current_date = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        all_predictions, all_actuals, all_bet_types = [], [], []

        while current_date <= end_date_dt:
            # ... (the existing data collection logic) ...
            # This part of your while loop is correct and remains unchanged
            date_str = current_date.strftime('%Y-%m-%d')
            raw_games = feature_pipeline.fetch_game_data(date_str)
            if raw_games.empty:
                current_date += timedelta(days=1)
                continue
            
            # --- (The rest of the data collection logic inside the while loop) ---
            games_with_results = raw_games.copy()
            leaky_columns = ['game_pk', 'game_id', 'home_score', 'away_score', 
                           'home_ml', 'away_ml', 'total_line', 'over_odds']
            pre_game_data = raw_games.drop(columns=[col for col in leaky_columns 
                                                   if col in raw_games.columns])
            game_features = feature_pipeline.engineer_features(pre_game_data)
            game_features.fillna(0, inplace=True)
            for col in ['game_id', 'game_pk', 'home_ml', 'away_ml', 'total_line', 'over_odds']:
                if col in raw_games.columns:
                    game_features[col] = raw_games[col].values
            predictions = model_loader.predict_with_uncertainty(game_features)
            
            for idx, (_, game) in enumerate(game_features.iterrows()):
                game_id = game.get('game_id', game.get('game_pk'))
                result_row = games_with_results[games_with_results['game_id'] == game_id].iloc[0] if 'game_id' in games_with_results.columns else None
                if result_row is None: continue
                
                home_score, away_score = result_row['home_score'], result_row['away_score']
                pred_home, pred_away = predictions['predictions'][idx][0], predictions['predictions'][idx][1]
                uncertainty = predictions['total_uncertainty'][idx].mean()
                
                prob_calc = ProbabilityCalculator(use_poisson=True)
                raw_home_prob = prob_calc.calculate_win_probability(pred_home, pred_away, uncertainty, uncertainty)
                home_won = 1 if home_score > away_score else 0
                
                all_predictions.extend([raw_home_prob, 1 - raw_home_prob])
                all_actuals.extend([home_won, 1 - home_won])
                all_bet_types.extend(['moneyline', 'moneyline'])
                
                total_line = game.get('total_line')
                if pd.notna(total_line):
                    pred_total = pred_home + pred_away
                    over_prob, _ = prob_calc.calculate_total_probability(pred_total, float(total_line), pred_home, pred_away, uncertainty)
                    actual_total = home_score + away_score
                    over_actual = 1 if actual_total > total_line else 0
                    all_predictions.append(over_prob)
                    all_actuals.append(over_actual)
                    all_bet_types.append('total')
            
            current_date += timedelta(days=1)
        
        logging.info(f"Collected {len(all_predictions)} prediction/actual pairs")

        # --- START: THE FIX ---

        # 1. Split data into training and validation sets
        X = pd.DataFrame({'prediction': all_predictions, 'bet_type': all_bet_types})
        y = pd.Series(all_actuals, name='actual')

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=validation_split, 
            random_state=42,
            stratify=X['bet_type']  # Ensures balanced bet types in both sets
        )
        logging.info(f"Split data into {len(X_train)} training samples and {len(X_test)} validation samples.")

        # 2. Train calibrators ONLY on the training data
        train_df = X_train.join(y_train)
        for _, row in train_df.iterrows():
            self.calibrator.collect_calibration_data(np.array([row['prediction']]), np.array([row['actual']]), row['bet_type'])
        
        self.calibrator.fit_calibrators()
        
        # 3. Save the fitted calibrators
        calibrator_path = os.path.join(self.config.model_dir, 'calibrators.pkl')
        joblib.dump(self.calibrator, calibrator_path)
        logging.info(f"Saved calibrators to {calibrator_path}")
        
        # 4. Evaluate calibrators ONLY on the validation data
        logging.info("Evaluating calibrator performance on unseen validation data...")
        results = self._evaluate_calibration(
            X_test['prediction'].tolist(), 
            y_test.tolist(), 
            X_test['bet_type'].tolist()
        )
        
        # --- END: THE FIX ---
        
        return results

    def _evaluate_calibration(self, predictions: List[float], 
                            actuals: List[int], bet_types: List[str]) -> Dict:
        # This method remains unchanged as its logic is correct
        # It will now receive validation data instead of the full dataset
        df = pd.DataFrame({'prediction': predictions, 'actual': actuals, 'bet_type': bet_types})
        results = {}
        for bet_type in ['moneyline', 'total']:
            type_df = df[df['bet_type'] == bet_type]
            if len(type_df) < 100: continue
            
            bins = np.linspace(0, 1, 11)
            type_df['bin'] = pd.cut(type_df['prediction'], bins)
            calibration = type_df.groupby('bin').agg({'prediction': 'mean', 'actual': 'mean', 'bet_type': 'count'}).rename(columns={'bet_type': 'count'})
            
            ece = 0
            for _, row in calibration.iterrows():
                if row['count'] > 0:
                    weight = row['count'] / len(type_df)
                    ece += weight * abs(row['prediction'] - row['actual'])
            
            results[f'{bet_type}_calibration'] = calibration.to_dict()
            results[f'{bet_type}_ece'] = ece
            
            calibrated_preds = [self.calibrator.calibrate_probability(pred, bet_type) for pred in type_df['prediction']]
            type_df['calibrated'] = calibrated_preds
            
            type_df['cal_bin'] = pd.cut(type_df['calibrated'], bins)
            cal_calibration = type_df.groupby('cal_bin').agg({'calibrated': 'mean', 'actual': 'mean'})
            
            cal_ece = 0
            counts = type_df['cal_bin'].value_counts()
            for bin_label, row in cal_calibration.iterrows():
                if pd.isna(row['calibrated']) or pd.isna(row['actual']): continue
                if bin_label in counts:
                    weight = counts[bin_label] / len(type_df)
                    cal_ece += weight * abs(row['calibrated'] - row['actual'])
            
            results[f'{bet_type}_calibrated_ece'] = cal_ece
            logging.info(f"{bet_type.upper()} - Original ECE: {ece:.4f}, Calibrated ECE: {cal_ece:.4f}")
        
        return results

def train_calibration(config: BettingConfig):
    """Main function to train calibration"""
    
    trainer = CalibrationTrainer(config)
    
    # Train on historical data
    results = trainer.train_calibrators(
        start_date="2022-04-01",
        end_date="2024-03-31"  # Use full season for training
    )
    
    print("\nCalibration Training Complete!")
    print("="*50)
    
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")

def optimize_betting_parameters(config: BettingConfig):
    """Standalone function to optimize all betting parameters"""
    
    print("\n" + "="*60)
    print("OPTIMIZING BETTING PARAMETERS")
    print("="*60)
    
    # Create model instance
    betting_model = MLBBettingModelV2(config)
    
    # Run parameter optimization
    optimization_results = betting_model.optimize_parameters(
        start_date="2022-04-01",  # Use 2 years of data
        end_date="2024-03-31"
    )
    
    print("\nOptimization Results:")
    print("-" * 40)
    
    # Display confidence optimization results
    if 'confidence_optimization' in optimization_results:
        conf_results = optimization_results['confidence_optimization']
        print(f"\nOptimal Confidence Weights:")
        for key, value in conf_results['optimal_weights'].items():
            print(f"  {key}: {value:.3f}")
        print(f"\nAverage Validation Sharpe: {conf_results['average_sharpe']:.3f}")
        print(f"Average Validation ROI: {conf_results['average_roi']:.1f}%")
    
    # Display Kelly optimization results
    if 'kelly_optimization' in optimization_results:
        kelly_results = optimization_results['kelly_optimization']
        print(f"\nOptimal Kelly Thresholds:")
        for conf, mult in sorted(kelly_results['confidence_thresholds'].items()):
            print(f"  Confidence >= {conf:.1f}: {mult:.2f}x")
        print(f"\nBet Type Adjustments:")
        for bet_type, adj in kelly_results['bet_type_adjustments'].items():
            print(f"  {bet_type}: {adj:.2f}x")
    
    print("\nParameter optimization complete! Files saved to model directory.")


def generate_calibration_report(config: BettingConfig):
    """Generate comprehensive calibration report with visualizations"""
    
    print("\n" + "="*60)
    print("GENERATING CALIBRATION REPORT")
    print("="*60)
    
    # Create model instance
    betting_model = MLBBettingModelV2(config)
    
    # Generate visualizations
    viz_results = betting_model.visualize_calibration(
        start_date="2024-04-01",
        end_date="2024-06-30"
    )
    
    print(f"\nCalibration Report Generated:")
    print(f"  Moneyline samples analyzed: {viz_results['moneyline_samples']}")
    print(f"  Totals samples analyzed: {viz_results['totals_samples']}")
    print(f"\nVisualization files saved to: {config.model_dir}")
    print("  - moneyline_calibration.png")
    print("  - ml_calibration_comparison.png")
    print("  - totals_calibration.png")

def print_detailed_results(results: dict):
    """Prints a detailed list of all calculated performance metrics."""

    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    
    # Define the desired order of keys for printing
    key_order = [
        'kelly_portfolio_total_pnl', 'kelly_portfolio_roi_pct', 'kelly_portfolio_max_drawdown_pct',
        'flat_portfolio_total_pnl', 'flat_portfolio_roi_pct', 'flat_portfolio_max_drawdown_pct',
        'total_bets', 'overall_win_rate',
        'moneyline_roi_pct', 'moneyline_win_rate',
        'over_roi_pct', 'over_win_rate',
        'under_roi_pct', 'under_win_rate',
        'parlay_roi_pct', 'parlay_win_rate'
    ]

    for key in key_order:
        if key in results:
            value = results[key]
            # Add a '%' sign to percentage values for clarity
            if 'pct' in key or 'win_rate' in key:
                print(f"{key}: {value:.2f}%")
            # Format PnL as currency
            elif 'pnl' in key:
                print(f"{key}: ${value:,.2f}")
            else:
                print(f"{key}: {value}")
                
    print("="*60 + "\n")
# ============= UPDATE YOUR MAIN FUNCTION =============

def generate_caches(config: BettingConfig, cache_path="bet_cache.pkl") -> Tuple[Dict, Dict]:
    """
    Loops through the historical data ONCE to generate and cache all potential bets and game results.
    """
    if os.path.exists(cache_path):
        logging.info(f"Loading pre-computed caches from {cache_path}...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    logging.info(f"Cache not found. Generating new caches from {config.backtest_start_date} to {config.backtest_end_date}...")
    
    betting_model = MLBBettingModelV2(config)
    
    all_potential_bets = defaultdict(list)
    game_results_by_date = defaultdict(dict)
    
    current_date = pd.to_datetime(config.backtest_start_date)
    end_date_dt = pd.to_datetime(config.backtest_end_date)
    
    while current_date <= end_date_dt:
        date_str = current_date.strftime('%Y-%m-%d')
        
        # --- THIS IS THE FIX ---
        # Use the new, centralized method to get all data for the day
        raw_games_data, games_data_combined, odds_data = betting_model.feature_pipeline.fetch_and_process_day(current_date)
        
        if raw_games_data.empty:
            current_date += timedelta(days=1)
            continue
        
        logging.info(f"Processing data for {date_str}...")

        # Generate all potential bets for the day (BEFORE filtering)
        ml_bets = betting_model.analyzer.analyze_moneyline(games_data_combined, odds_data)
        total_bets = betting_model.analyzer.analyze_totals(games_data_combined, odds_data)
        all_potential_bets[date_str].extend(ml_bets + total_bets)
        
        # Store the game results needed for the fast simulation
        for _, game in raw_games_data.iterrows():
            game_id_str = str(game['game_id'])
            game_results_by_date[date_str][game_id_str] = {
                'home_score': game['home_score'],
                'away_score': game['away_score'],
                'total_line': game.get('total_line')
            }
            
        current_date += timedelta(days=1)

    caches = (dict(all_potential_bets), dict(game_results_by_date))
    with open(cache_path, 'wb') as f:
        pickle.dump(caches, f)
    logging.info(f"Bet and result caches saved to {cache_path}")
    
    return caches

def _simulate_fast_day(portfolio: Dict, day_game_results: Dict, current_flat_bankroll: float, config: BettingConfig) -> Tuple[float, float, List]:
    """
    (Helper Function) Simulates a single day's PnL for both straight and parlay bets.
    """
    day_kelly_pnl = 0
    day_flat_pnl = 0
    day_bets = []

    # --- Process Straight Bets (Same as before) ---
    for bet in portfolio.get('straight_bets', []):
        kelly_bet_size = portfolio['straight_bet_sizes'].get(bet.game_id)
        flat_bet_size = current_flat_bankroll * config.flat_bet_unit
        if kelly_bet_size is None or flat_bet_size <= 0: continue
        
        game_result = day_game_results.get(str(bet.game_id))
        if game_result is None: continue

        home_score, away_score = game_result['home_score'], game_result['away_score']
        
        if bet.bet_type == 'moneyline':
            won = (home_score > away_score) if bet.selection == 'home' else (away_score > home_score)
        else: # total
            total = home_score + away_score
            line = game_result.get('total_line')
            if line is None: continue
            won = (total > line) if bet.selection == 'over' else (total < line)
        
        payout_mult = (bet.odds / 100) if bet.odds > 0 else (100 / abs(bet.odds))
        day_kelly_pnl += kelly_bet_size * payout_mult if won else -kelly_bet_size
        day_flat_pnl += flat_bet_size * payout_mult if won else -flat_bet_size
        
        bet.result = 'win' if won else 'loss'; bet.pnl = day_kelly_pnl; bet.bet_size = kelly_bet_size; day_bets.append(bet)

    # --- START: NEW PARLAY SIMULATION LOGIC ---
    for i, parlay in enumerate(portfolio.get('parlays', [])):
        kelly_bet_size = portfolio['parlay_bet_sizes'][i]
        flat_bet_size = current_flat_bankroll * config.flat_bet_unit # Can use a multiplier for parlays if desired

        all_legs_won = True
        for leg in parlay.legs:
            leg_result = day_game_results.get(str(leg.game_id))
            if leg_result is None:
                all_legs_won = False; break
            
            home_score, away_score = leg_result['home_score'], leg_result['away_score']
            
            if leg.bet_type == 'moneyline':
                leg_won = (home_score > away_score) if leg.selection == 'home' else (away_score > home_score)
            else: # total
                total = home_score + away_score
                line = leg_result.get('total_line')
                if line is None: leg_won = False; break
                leg_won = (total > line) if leg.selection == 'over' else (total < line)
            
            if not leg_won:
                all_legs_won = False; break

        # Parlay odds are already in decimal format
        payout_mult = parlay.combined_odds - 1
        day_kelly_pnl += kelly_bet_size * payout_mult if all_legs_won else -kelly_bet_size
        day_flat_pnl += flat_bet_size * payout_mult if all_legs_won else -flat_bet_size

        # Create a representative bet object for tracking
        parlay_bet_obj = EnhancedBet(game_id=f"parlay_{i}", bet_type='parlay', selection='parlay', odds=parlay.combined_odds,
                                     result='win' if all_legs_won else 'loss', pnl=day_kelly_pnl, bet_size=kelly_bet_size)
        day_bets.append(parlay_bet_obj)
    # --- END: NEW PARLAY SIMULATION LOGIC ---

    return day_kelly_pnl, day_flat_pnl, day_bets

def run_fast_simulation(config: BettingConfig, all_potential_bets: Dict, game_results_by_date: Dict) -> Dict:
    """
    The main fast simulation wrapper. It initializes bankrolls and calls the
    day simulator in a loop.
    """
    optimizer = ProfessionalPortfolioOptimizer(config)
    parlay_gen = EnhancedParlayGenerator(config)

    kelly_bankroll = 5000.0
    flat_bankroll = 5000.0
    
    daily_results = []
    all_bets_placed = []
    
    sorted_dates = sorted(all_potential_bets.keys())

    for date_str in sorted_dates:
        potential_bets_today = all_potential_bets[date_str]
        
        # Filter the day's potential bets using the trial's config
        bets_for_day = [
            bet for bet in potential_bets_today
            if (bet.bet_type == 'moneyline' and bet.edge >= config.min_edge_moneyline and bet.confidence >= config.min_confidence_moneyline) or
               (bet.bet_type == 'total' and bet.selection == 'over' and bet.edge >= config.min_edge_totals and bet.confidence >= config.min_confidence_over) or
               (bet.bet_type == 'total' and bet.selection == 'under' and bet.edge >= config.min_edge_totals and bet.confidence >= config.min_confidence_under)
        ]
        
        if not bets_for_day:
            continue

        parlays = parlay_gen.generate_parlays(bets_for_day)
        portfolio = optimizer.optimize_portfolio(bets_for_day, parlays, kelly_bankroll)
        
        # Simulate the day using the cached results
        day_kelly_pnl, day_flat_pnl, day_bets = _simulate_fast_day(
            portfolio, game_results_by_date[date_str], flat_bankroll, config
        )
        
        # Log daily results for performance calculation
        daily_results.append({
            'date': date_str,
            'kelly_starting_bankroll': kelly_bankroll,
            'kelly_pnl': day_kelly_pnl,
            'kelly_ending_bankroll': kelly_bankroll + day_kelly_pnl,
            'flat_starting_bankroll': flat_bankroll,
            'flat_pnl': day_flat_pnl,
            'flat_ending_bankroll': flat_bankroll + day_flat_pnl,
            'num_bets': len(day_bets)
        })

        kelly_bankroll += day_kelly_pnl
        flat_bankroll += day_flat_pnl
        all_bets_placed.extend(day_bets)

    # Use your existing performance calculator
    # We need a dummy BacktestingEngine instance just to call the method
    dummy_engine = BacktestingEngine(config, None) 
    return dummy_engine._calculate_performance_metrics(daily_results, all_bets_placed, 10000)

import optuna

def objective(trial: optuna.Trial, all_potential_bets: Dict, game_results_by_date: Dict) -> float:
    """
    The Optuna objective function, now with a check for a minimum number of bets.
    """
    # Define the minimum bets required for a trial's ROI to be considered valid.
    # 50 is a good starting point, you can adjust this.
    MIN_BETS_FOR_VALID_TRIAL = 15

    # 1. Define the search space for all hyperparameters (this part is unchanged)
    config = BettingConfig(
        min_edge_moneyline=trial.suggest_float("min_edge_moneyline", -0.1, 0.2),
        min_edge_totals=trial.suggest_float("min_edge_totals", -0.1, 0.4),
        kelly_fraction=trial.suggest_float("kelly_fraction", 0.1, 0.75),
        min_confidence_moneyline=trial.suggest_float("min_confidence_moneyline", 0.50, 0.70),
        min_confidence_over=trial.suggest_float("min_confidence_over", 0.50, 0.70),
        min_confidence_under=trial.suggest_float("min_confidence_under", 0.50, 0.70),
        enable_parlays=trial.suggest_categorical("enable_parlays", [True, False]),
        min_edge_parlay_leg=trial.suggest_float("min_edge_parlay_leg", -0.05, 0.5),
        min_confidence_parlay_leg=trial.suggest_float("min_confidence_parlay_leg", 0.50, 0.75),
        min_parlay_edge=trial.suggest_float("min_parlay_edge", -0.05, 0.5),
        max_parlay_legs=trial.suggest_int("max_parlay_legs", 2, 4),
    )
    
    try:
        results = run_fast_simulation(config, all_potential_bets, game_results_by_date)
        
        # --- START: THE FIX ---
        # Check if the simulation produced results AND met the minimum bet count
        if (results and 
            'kelly_portfolio_roi_pct' in results and 
            results.get('total_bets', 0) >= MIN_BETS_FOR_VALID_TRIAL):
            
            # If valid, report the true ROI
            roi = results['kelly_portfolio_roi_pct']
            trial.report(roi, trial.number)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            return roi
        
        # If there are not enough bets, the trial is not statistically significant.
        # Return 0.0 to signal a non-viable (but not failed) strategy.
        return -100.0
        # --- END: THE FIX ---

    except Exception as e:
        logging.error(f"Trial {trial.number} failed with an exception: {e}")
        return -100.0 # Return a large negative for actual errors/crashes


# In your main script logic...

def run_hyperparameter_optimization(n_trials: int, validation_start: str, validation_end: str) -> dict:
    """
    Sets up and executes the Optuna study on a specific VALIDATION period.
    This function finds the best strategy parameters without touching the final test data.
    """
    logging.info(f"Starting hyperparameter optimization on validation period: {validation_start} to {validation_end}")

    # 1. Generate or load a cache specifically for the VALIDATION period.
    # This keeps the optimization process fast.
    validation_config = BettingConfig(
        backtest_start_date=validation_start,
        backtest_end_date=validation_end
    )
    cache_path = f"validation_cache_{validation_start}_to_{validation_end}.pkl"
    all_potential_bets, game_results_by_date = generate_caches(validation_config, cache_path=cache_path)

    if not all_potential_bets:
        logging.error("No data found for the validation period. Cannot run optimization.")
        return {}

    # 2. Create and run the Optuna study.
    # The 'objective' function will now repeatedly test parameters against this validation data.
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, all_potential_bets, game_results_by_date), n_trials=n_trials)

    # 3. Print and return the best parameters found.
    print("\n" + "="*60)
    print("VALIDATION PARAMETER OPTIMIZATION COMPLETE".center(60))
    print("="*60)
    print(f"Best trial on validation set: {study.best_trial.value:.2f}% ROI")
    print("\nBest Parameters Found:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    print("="*60)

    return study.best_trial.params


def main():
    """
    Main function to run the full, trustworthy backtesting workflow.
    This workflow separates parameter tuning (optimization) from final evaluation.
    """

    # --- CHOOSE YOUR MODE ---
    # Set to True to find the best parameters using the validation dataset.
    # Set to False to run a final backtest using a known set of good parameters.
    run_optimization = True

    best_params = {}

    if run_optimization:
        # --- STEP 1: Find the best strategy parameters on the VALIDATION dataset ---
        # This optimization does NOT see the final holdout/test data.
        best_params = run_hyperparameter_optimization(
            n_trials=1500,  # The number of different parameter combinations to try
            validation_start="2024-04-01",
            validation_end="2024-09-30" # e.g., The entire 2024 season
        )
        if not best_params:
            logging.error("Optimization failed to produce parameters. Exiting.")
            return

    else:
        # If not optimizing, use a pre-determined, trusted set of parameters.
        # These would be the result of a previous optimization run.
        logging.info("Skipping optimization. Using pre-defined parameters for the final backtest.")
        best_params = {
            'min_edge_moneyline': 0.07352245991140353,
            'min_edge_totals': 0.10909197113221557,
            'kelly_fraction': 0.10893796712544962,
            'min_confidence_moneyline': 0.5357886892468485,
            'min_confidence_over': 0.5759281613589134,
            'min_confidence_under': 0.5601125162484605,
            'enable_parlays': False,
            'min_edge_parlay_leg': -0.16493987372914232,
            'min_confidence_parlay_leg': 0.5402807215794936,
            'min_parlay_edge': -0.7786456496799152,
            'max_parlay_legs': 4
        }


    # --- STEP 2: Run the final, trustworthy backtest on the unseen HOLDOUT dataset ---
    logging.info("\nStarting final backtest on HOLDOUT data with optimized parameters...")

    # Configure the backtest with the best parameters found during optimization
    # and set the date range to the unseen holdout period.
    final_config = BettingConfig(
        backtest_start_date="2025-04-01",  # The unseen holdout data
        backtest_end_date="2025-07-05",
        **best_params  # This elegantly overwrites the defaults with our optimized values
    )

    # The analyzer must be initialized with the final config to use the correct thresholds
    final_model_loader = EnhancedModelLoader(final_config.model_dir)
    final_analyzer = CalibratedBettingAnalyzer(final_model_loader, final_config)

    # The backtesting engine uses the final config and analyzer
    backtesting_engine = BacktestingEngine(final_config, final_analyzer)

    # The result of THIS backtest is the one you can trust.
    final_results = backtesting_engine.backtest(
        start_date=final_config.backtest_start_date,
        end_date=final_config.backtest_end_date,
        initial_bankroll=10000
    )

    # Print the final, trustworthy results
    if final_results:
        print("\n\n" + "="*60)
        print("TRUSTWORTHY BACKTEST RESULTS (ON HOLDOUT DATA)".center(60))
        print("="*60)
        print_detailed_results(final_results)
    else:
        print("Final backtest on holdout data did not produce results.")


if __name__ == "__main__":
    # Ensure you have all necessary functions like generate_caches, objective, etc. defined
    # in the script before this point.
    main()
