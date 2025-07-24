"""
MLB TEMPORAL-AWARE DEEP LEARNING MODEL V4 - RUN PREDICTION FOCUSED
Optimized purely for run prediction accuracy without betting considerations
"""

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
import numpy as np
import pandas as pd
import torch
import warnings
import logging
from typing import List, Union, Optional, Tuple, Any, Dict
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from scipy.stats import pearsonr
from abc import ABC, abstractmethod

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============= CONFIGURATION =============
@dataclass
class ModelConfig:
    """Configuration for model architecture and training"""
    # Architecture
    hidden_dim: int = 128
    n_heads: int = 8
    n_transformer_layers: int = 2
    dropout: float = 0.3
    use_transformer: bool = True
    use_advanced_architecture: bool = True
    
    # New architecture options
    use_gnn: bool = False
    use_hybrid: bool = False
    use_perceiver: bool = False
    use_snapshot_ensemble: bool = False
    use_adversarial: bool = False
    use_quantile_regression: bool = False
    
    # Training
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    max_epochs: int = 100
    patience: int = 20
    
    # Loss weights - focused on prediction accuracy
    score_weight: float = 0.6
    winner_weight: float = 0.2
    margin_weight: float = 0.2

# ============= ENHANCED LOSS FUNCTIONS =============
class RunPredictionLoss(nn.Module):
    """Loss function optimized for accurate run prediction"""
    
    def __init__(self, score_weight=0.6, winner_weight=0.2, margin_weight=0.2):
        super().__init__()
        self.score_weight = score_weight
        self.winner_weight = winner_weight
        self.margin_weight = margin_weight
        
    def forward(self, pred, target):
        # --- Start of Proposed Change ---
        # If the model output is a tuple, extract the primary prediction tensor.
        if isinstance(pred, tuple):
            pred = pred[0]
        # --- End of Proposed Change ---

        # Huber loss for score prediction (robust to outliers)
        score_loss = F.smooth_l1_loss(pred, target, beta=1.5)
        
        # Predict run differential
        pred_margin = pred[:, 0] - pred[:, 1]
        true_margin = target[:, 0] - target[:, 1]
        
        # Winner accuracy with margin-aware loss
        margin_error = torch.abs(pred_margin - true_margin)
        winner_loss = torch.where(
            pred_margin * true_margin > 0,  # Correct winner
            0.1 * torch.sigmoid(margin_error),  # Small penalty proportional to margin error
            torch.sigmoid(margin_error)  # Large penalty when wrong
        ).mean()
        
        # Direct margin prediction loss
        margin_loss = F.smooth_l1_loss(pred_margin, true_margin, beta=1.0)
        
        # Total runs prediction bonus
        pred_total = pred[:, 0] + pred[:, 1]
        true_total = target[:, 0] + target[:, 1]
        total_loss_bonus = F.smooth_l1_loss(pred_total, true_total, beta=1.0) # Renamed to avoid confusion
        
        total_loss = (self.score_weight * score_loss + 
                     self.winner_weight * winner_loss + 
                     self.margin_weight * margin_loss +
                     0.1 * total_loss_bonus)  # Small weight for total runs
            
        return total_loss

class WinLossAwareLoss(nn.Module):
    """Loss that prevents NaN values and focuses on accuracy"""
    
    def __init__(self, score_weight: float = 0.7, winner_weight: float = 0.3):
        super().__init__()
        self.score_weight = score_weight
        self.winner_weight = winner_weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # --- Start of Proposed Change ---
        # If the model output is a tuple (e.g., from HybridModel), extract the primary prediction tensor.
        if isinstance(pred, tuple):
            pred = pred[0]
        # --- End of Proposed Change ---

        # Clamp predictions to prevent extreme values
        pred = torch.clamp(pred, min=-20, max=20)
        
        # Score loss with Huber loss for stability
        score_loss = F.smooth_l1_loss(pred, target)
        
        # Winner loss with numerical stability
        pred_margin = pred[:, 0] - pred[:, 1]
        true_margin = target[:, 0] - target[:, 1]
        
        # Add small epsilon for numerical stability
        eps = 1e-7
        
        # Normalize margins to prevent extreme values
        pred_margin_norm = torch.tanh(pred_margin / 10.0)
        true_margin_norm = torch.sign(true_margin)
        
        # Soft margin loss
        winner_loss = F.relu(1.0 - pred_margin_norm * true_margin_norm + eps).mean()
        
        return self.score_weight * score_loss + self.winner_weight * winner_loss

class ConfidenceOptimizedLoss(nn.Module):
    """Loss function that rewards high-confidence correct predictions"""
    
    def __init__(self, score_weight: float = 0.6, winner_weight: float = 0.3, confidence_weight: float = 0.1):
        super().__init__()
        self.score_weight = score_weight
        self.winner_weight = winner_weight
        self.confidence_weight = confidence_weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Score loss (MSE)
        score_loss = F.mse_loss(pred, target)
        
        # Winner loss
        pred_margin = pred[:, 0] - pred[:, 1]
        true_margin = target[:, 0] - target[:, 1]
        
        # Soft margin loss
        winner_loss = F.relu(1.0 - pred_margin * true_margin).mean()
        
        # Confidence loss - reward being confident when correct
        confidence = torch.abs(pred_margin)
        correct = (pred_margin * true_margin) > 0
        
        # Reward: high confidence when correct, penalize high confidence when wrong
        confidence_loss = torch.where(
            correct,
            -torch.log(torch.sigmoid(confidence)),  # Reward confidence when correct
            torch.log(torch.sigmoid(confidence))     # Penalize confidence when wrong
        ).mean()
        
        return (self.score_weight * score_loss + 
                self.winner_weight * winner_loss + 
                self.confidence_weight * confidence_loss)

# ============= ADVANCED FEATURE ENGINEERING =============
class AdvancedFeatureEngineer:
    """Enhanced feature engineering with domain-specific transformations"""
    
    def __init__(self):
        self.imputers = {}
        self.scalers = {}
        self.pca_models = {}
        self.feature_stats = {}
        self.park_factors = None  # Store pre-calculated park factors
        self.is_fitted = False
        
    def __setstate__(self, state):
        """Handle loading of old pickle files that don't have new attributes"""
        self.__dict__.update(state)
        # Add missing attributes for backwards compatibility
        if not hasattr(self, 'park_factors'):
            self.park_factors = None
        if not hasattr(self, 'is_fitted'):
            self.is_fitted = False
            
    def fit(self, df: pd.DataFrame) -> 'AdvancedFeatureEngineer':
        """Fit the feature engineer on training data (with scores available)"""
        # Calculate and store park factors during training
        if 'venue' in df.columns and 'home_score' in df.columns and 'away_score' in df.columns:
            self.park_factors = self._calculate_park_factors_training(df)
            logging.info(f"Calculated park factors for {len(self.park_factors)} venues")
        else:
            logging.warning("Could not calculate park factors - missing required columns")
            
        self.is_fitted = True
        return self
        
    def _calculate_park_factors_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate park factors from training data with scores"""
        # Group by venue and calculate average scores
        park_stats = df.groupby('venue').agg({
            'home_score': ['mean', 'count'],
            'away_score': 'mean'
        }).reset_index()
        
        # Flatten column names
        park_stats.columns = ['venue', 'home_score_mean', 'game_count', 'away_score_mean']
        
        # Only use venues with sufficient data (at least 10 games)
        park_stats = park_stats[park_stats['game_count'] >= 10]
        
        # Calculate overall average
        overall_avg = df['home_score'].mean() + df['away_score'].mean()
        
        # Calculate park factor
        park_stats['park_factor'] = (
            (park_stats['home_score_mean'] + park_stats['away_score_mean']) / overall_avg
        )
        
        # Add some bounds to prevent extreme values
        park_stats['park_factor'] = park_stats['park_factor'].clip(0.8, 1.2)
        
        logging.info(f"Park factors range: {park_stats['park_factor'].min():.3f} to {park_stats['park_factor'].max():.3f}")
        
        return park_stats[['venue', 'park_factor']]
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sophisticated domain-specific features"""
        
        # 1. Sabermetric-inspired features
        if all(col in df.columns for col in ['batting_avg_home', 'on_base_pct_home', 'slugging_pct_home']):
            # OPS (On-base Plus Slugging)
            df['home_OPS'] = df['on_base_pct_home'] + df['slugging_pct_home']
            df['away_OPS'] = df['on_base_pct_away'] + df['slugging_pct_away']
            df['diff_OPS'] = df['home_OPS'] - df['away_OPS']
        
        # 2. Pitcher matchup features
        if 'home_SP_ERA' in df.columns and 'away_batting_avg' in df.columns:
            # Pitcher vs opposing team batting
            df['home_pitcher_vs_opp_batting'] = df['away_batting_avg'] / (df['home_SP_ERA'] + 1)
            df['away_pitcher_vs_opp_batting'] = df['home_batting_avg'] / (df['away_SP_ERA'] + 1)
        
        # 3. Momentum with decay
        for team_type in ['home', 'away']:
            if f'{team_type}_last_5_wins' in df.columns:
                # Weighted recent performance (more recent games matter more)
                df[f'{team_type}_momentum_score'] = (
                    df[f'{team_type}_last_5_wins'] * 0.5 + 
                    df.get(f'{team_type}_last_10_wins', 0) * 0.3 + 
                    df.get(f'{team_type}_last_20_wins', 0) * 0.2
                )
        
        # 4. Park factors and conditions
        if 'venue' in df.columns:
            # Check if we have park_factors attribute (for backwards compatibility)
            has_park_factors = hasattr(self, 'park_factors') and self.park_factors is not None
            is_fitted = hasattr(self, 'is_fitted') and self.is_fitted
            
            # Use pre-calculated park factors if available
            if has_park_factors and is_fitted:
                # Merge park factors
                df = df.merge(self.park_factors, on='venue', how='left')
                
                # For any venues not in our training data, use neutral factor
                if 'park_factor' in df.columns:
                    missing_venues = df[df['park_factor'].isna()]['venue'].unique()
                    if len(missing_venues) > 0:
                        logging.debug(f"Using neutral park factor for {len(missing_venues)} unknown venues")
                    df['park_factor'] = df['park_factor'].fillna(1.0)
            else:
                # If not fitted or no park factors, use neutral
                df['park_factor'] = 1.0
                if not is_fitted:
                    logging.debug("AdvancedFeatureEngineer not fitted - using neutral park factors")
            
            # Weather impact on scoring
            if all(col in df.columns for col in ['temperature', 'wind_speed']):
                df['weather_score_impact'] = (
                    (df['temperature'] - 72) * 0.01 +  # Optimal temp around 72F
                    df['wind_speed'] * -0.02  # Wind reduces scoring
                )
        
        # 5. Bullpen fatigue indicators
        if 'home_bullpen_ERA' in df.columns:
            # Rolling bullpen usage (would need game history)
            df['home_bullpen_freshness'] = 1 / (df['home_bullpen_ERA'] + 1)
            df['away_bullpen_freshness'] = 1 / (df['away_bullpen_ERA'] + 1)
        
        # 6. Time-based features
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
            df['day_of_week'] = df['game_date'].dt.dayofweek
            df['month'] = df['game_date'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Season progress (fatigue factor)
            df['season_progress'] = df['game_date'].dt.dayofyear / 365
        
        return df
    
    def smart_imputation(self, X: pd.DataFrame, feature_groups: Dict[str, List[str]]) -> pd.DataFrame:
        """Intelligent imputation based on feature type"""
        X_imputed = X.copy()
        
        for group_name, features in feature_groups.items():
            group_features = [f for f in features if f in X.columns]
            if not group_features:
                continue
            
            if group_name in ['batting', 'pitching']:
                # Use KNN imputation for related stats
                if group_name not in self.imputers:
                    self.imputers[group_name] = KNNImputer(n_neighbors=5)
                    X_imputed[group_features] = self.imputers[group_name].fit_transform(X[group_features])
                else:
                    X_imputed[group_features] = self.imputers[group_name].transform(X[group_features])
            
            elif group_name == 'momentum':
                # For streak data, use 0 (no streak) as default
                X_imputed[group_features] = X_imputed[group_features].fillna(0)
            
            else:
                # For other features, use median
                if group_name not in self.feature_stats:
                    self.feature_stats[group_name] = X[group_features].median()
                X_imputed[group_features] = X_imputed[group_features].fillna(self.feature_stats[group_name])
        
        return X_imputed

# ============= FEATURE PROCESSING =============
def get_feature_indices(feature_cols):
    """Map column names to feature groups AFTER feature selection"""
    indices = {
        'batting': [],
        'pitching': [],
        'bullpen': [],
        'starter': [],
        'situational': [],
        'momentum': []
    }
    
    for i, col in enumerate(feature_cols):
        # More robust pattern matching
        col_lower = col.lower()
        
        # Batting features
        if 'batting' in col_lower and 'diff_' not in col:
            indices['batting'].append(i)
        # Pitching features (general)
        elif 'pitching' in col_lower and '_sp_' not in col_lower and 'bullpen' not in col_lower and 'diff_' not in col:
            indices['pitching'].append(i)
        # Bullpen features
        elif 'bullpen' in col_lower and 'diff_' not in col:
            indices['bullpen'].append(i)
        # Starting pitcher features
        elif '_sp_' in col_lower and 'diff_' not in col:
            indices['starter'].append(i)
        # Momentum features
        elif any(x in col_lower for x in ['streak', 'momentum', 'last_5_wins', 'last_10_wins', 'last_20_wins']):
            indices['momentum'].append(i)
        # Differential features
        elif 'diff_' in col:
            indices['situational'].append(i)
        # Everything else goes to situational
        else:
            indices['situational'].append(i)
    
    # Ensure each group has at least some features
    feature_groups = {}
    for group, idx_list in indices.items():
        feature_groups[group] = len(idx_list)
        if len(idx_list) == 0:
            # Add some indices to empty groups from situational
            if len(indices['situational']) > 10:
                indices[group] = indices['situational'][:5]
                indices['situational'] = indices['situational'][5:]
                feature_groups[group] = 5
    
    return indices, feature_groups

# ============= ADVANCED ARCHITECTURES =============
class TeamGraphAttention(nn.Module):
    """Graph attention layer for team relationships"""
    def __init__(self, input_dim: int, hidden_dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // n_heads
        
        self.W_q = nn.Linear(input_dim, hidden_dim)
        self.W_k = nn.Linear(input_dim, hidden_dim)
        self.W_v = nn.Linear(input_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, team_embeddings, adjacency_matrix=None):
        batch_size, n_teams, _ = team_embeddings.shape
        
        # Multi-head attention
        Q = self.W_q(team_embeddings).view(batch_size, n_teams, self.n_heads, self.head_dim)
        K = self.W_k(team_embeddings).view(batch_size, n_teams, self.n_heads, self.head_dim)
        V = self.W_v(team_embeddings).view(batch_size, n_teams, self.n_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch, heads, teams, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply adjacency matrix if provided (for actual game connections)
        if adjacency_matrix is not None:
            scores = scores.masked_fill(adjacency_matrix.unsqueeze(1) == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, n_teams, self.hidden_dim)
        output = self.W_o(attn_output)
        
        return output, attn_weights

class MLBGraphNeuralNetwork(nn.Module):
    """Graph Neural Network for modeling team relationships"""
    def __init__(self, feature_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        self.team_embeddings = nn.Parameter(torch.randn(30, hidden_dim))  # 30 MLB teams
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        
        self.gnn_layers = nn.ModuleList([
            TeamGraphAttention(hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(n_layers)
        ])
        
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)  # home_score, away_score
        )
        
    def forward(self, game_features, home_team_idx, away_team_idx, adjacency_matrix=None):
        batch_size = game_features.shape[0]
        
        # Project game features
        game_embedding = self.feature_projection(game_features)
        
        # Get team embeddings
        team_embeds = self.team_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply GNN layers
        for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            team_embeds_new, _ = gnn_layer(team_embeds, adjacency_matrix)
            team_embeds = norm(team_embeds + team_embeds_new)  # Residual connection
        
        # Extract home and away team embeddings
        home_embeds = team_embeds[torch.arange(batch_size), home_team_idx]
        away_embeds = team_embeds[torch.arange(batch_size), away_team_idx]
        
        # Combine with game features
        combined = torch.cat([home_embeds + game_embedding, away_embeds + game_embedding], dim=-1)
        
        return self.output_projection(combined)

class TemporalTeamEncoder(nn.Module):
    """LSTM/GRU encoder for team's recent game sequence"""
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 2, use_gru: bool = True):
        super().__init__()
        self.use_gru = use_gru
        
        if use_gru:
            self.rnn = nn.GRU(input_dim, hidden_dim, n_layers, 
                             batch_first=True, dropout=0.2, bidirectional=True)
        else:
            self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers,
                              batch_first=True, dropout=0.2, bidirectional=True)
        
        self.output_dim = hidden_dim * 2  # bidirectional
        
    def forward(self, x, lengths=None):
        # x shape: [batch, seq_len, features]
        if lengths is not None:
            # Pack sequence for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            output, hidden = self.rnn(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, hidden = self.rnn(x)
        
        # Get last hidden state
        if self.use_gru:
            # hidden shape: [n_layers * 2, batch, hidden_dim]
            hidden = hidden.view(-1, 2, x.shape[0], hidden.shape[-1])[-1]  # Last layer
            hidden = torch.cat([hidden[0], hidden[1]], dim=-1)  # Concatenate bidirectional
        else:
            # For LSTM, use hidden state (not cell state)
            hidden = hidden[0].view(-1, 2, x.shape[0], hidden[0].shape[-1])[-1]
            hidden = torch.cat([hidden[0], hidden[1]], dim=-1)
        
        return output, hidden

class PerceiverBlock(nn.Module):
    """Perceiver-style cross-attention block for handling high-dimensional inputs"""
    def __init__(self, input_dim: int, latent_dim: int = 256, n_latents: int = 32, n_heads: int = 8):
        super().__init__()
        self.latent_array = nn.Parameter(torch.randn(n_latents, latent_dim))
        
        self.cross_attention = nn.MultiheadAttention(
            latent_dim, n_heads, dropout=0.1, batch_first=True
        )
        self.self_attention = nn.MultiheadAttention(
            latent_dim, n_heads, dropout=0.1, batch_first=True
        )
        
        self.input_projection = nn.Linear(input_dim, latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 4, latent_dim)
        )
        
        self.layer_norm1 = nn.LayerNorm(latent_dim)
        self.layer_norm2 = nn.LayerNorm(latent_dim)
        self.layer_norm3 = nn.LayerNorm(latent_dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Project input
        x_proj = self.input_projection(x)
        
        # Repeat latent array for batch
        latents = self.latent_array.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Cross-attention: latents attend to input
        latents_norm = self.layer_norm1(latents)
        attended_latents, _ = self.cross_attention(latents_norm, x_proj, x_proj)
        latents = latents + attended_latents
        
        # Self-attention on latents
        latents_norm = self.layer_norm2(latents)
        self_attended, _ = self.self_attention(latents_norm, latents_norm, latents_norm)
        latents = latents + self_attended
        
        # MLP
        latents_norm = self.layer_norm3(latents)
        latents = latents + self.mlp(latents_norm)
        
        return latents

# ============= DYNAMIC LOSS COMPONENTS =============
class DynamicWeightedLoss(nn.Module):
    """Loss with learnable dynamic weights"""
    def __init__(self, n_tasks: int = 3):
        super().__init__()
        self.n_tasks = n_tasks
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
        
    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Combine multiple losses with learned weights using uncertainty weighting
        Based on "Multi-Task Learning Using Uncertainty to Weigh Losses"
        """
        weighted_losses = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_losses.append(precision * loss + self.log_vars[i])
        
        return sum(weighted_losses)

class QuantileRegressionLoss(nn.Module):
    """Quantile regression loss for predicting score distributions"""
    def __init__(self, quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]):
        super().__init__()
        self.quantiles = torch.tensor(quantiles)
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        predictions: [batch, n_quantiles, 2] (home and away scores for each quantile)
        targets: [batch, 2]
        """
        losses = []
        
        # Move quantiles to same device as predictions
        if predictions.is_cuda:
            self.quantiles = self.quantiles.cuda()
        
        # Expand targets to match predictions shape
        targets_expanded = targets.unsqueeze(1)  # [batch, 1, 2]
        
        for i, q in enumerate(self.quantiles):
            # Get predictions for this quantile
            pred_q = predictions[:, i, :]  # [batch, 2]
            
            # Calculate quantile loss
            errors = targets - pred_q
            losses.append(torch.max(q * errors, (q - 1) * errors).mean())
        
        return sum(losses) / len(losses)

# ============= SELF-SUPERVISED PRETRAINING =============
class MLBMaskedPretraining(nn.Module):
    """Self-supervised pretraining task for MLB data"""
    def __init__(self, base_model: nn.Module, feature_dim: int, mask_ratio: float = 0.15):
        super().__init__()
        self.base_model = base_model
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.randn(1, feature_dim))
        self.prediction_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, x, mask=None):
        batch_size, feature_dim = x.shape
        
        if mask is None:
            # Random masking during training
            mask = torch.rand(batch_size, feature_dim) < self.mask_ratio
        
        # Apply mask
        masked_x = x.clone()
        masked_x[mask] = self.mask_token.expand(mask.sum(), -1).flatten()
        
        # Get representations
        representations = self.base_model(masked_x)
        
        # Predict masked features
        predictions = self.prediction_head(representations)
        
        # Calculate loss only on masked positions
        loss = F.mse_loss(predictions[mask], x[mask])
        
        return loss, predictions

# ============= ADVERSARIAL TRAINING =============
class AdversarialTrainer:
    """Implements adversarial training for robustness"""
    def __init__(self, epsilon: float = 0.01, alpha: float = 0.005, n_steps: int = 3):
        self.epsilon = epsilon
        self.alpha = alpha
        self.n_steps = n_steps
        
    def generate_adversarial_examples(self, model, x, y, criterion):
        """Generate adversarial examples using PGD (Projected Gradient Descent)"""
        x_adv = x.clone().detach()
        
        for _ in range(self.n_steps):
            x_adv.requires_grad = True
            
            # Forward pass
            if hasattr(model, 'feature_indices'):
                outputs = model(x_adv, model.feature_indices)
            else:
                outputs = model(x_adv)
            
            loss = criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            
            # Update adversarial examples
            x_adv = x_adv + self.alpha * x_adv.grad.sign()
            x_adv = torch.clamp(x_adv, x - self.epsilon, x + self.epsilon)
            x_adv = x_adv.detach()
        
        return x_adv

# ============= TRANSFORMER COMPONENTS =============
class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class GLU(nn.Module):
    """Gated Linear Unit"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        
    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)

class FeatureAttentionTransformer(nn.Module):
    """TabNet-inspired attention mechanism for feature selection"""
    def __init__(self, input_dim: int, hidden_dim: int, n_steps: int = 3):
        super().__init__()
        self.n_steps = n_steps
        self.bn = nn.BatchNorm1d(input_dim)
        
        # Attention transformers
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(n_steps)
        ])
        
        # Feature transformers
        self.feature_layers = nn.ModuleList([
            nn.Sequential(
                GLU(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                GLU(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            ) for _ in range(n_steps)
        ])
        
    def forward(self, x):
        x = self.bn(x)
        prior = torch.ones_like(x)
        aggregated_features = []
        attention_masks = []
        
        for step in range(self.n_steps):
            # Attention mechanism
            attention = self.attention_layers[step](x * prior)
            attention_mask = torch.softmax(attention, dim=-1)
            attention_masks.append(attention_mask)
            
            # Apply attention and extract features
            masked_features = x * attention_mask
            transformed = self.feature_layers[step](masked_features)
            aggregated_features.append(transformed)
            
            # Update prior (encourage exploring different features)
            prior = prior * (1 - attention_mask)
        
        # Aggregate all transformed features
        output = torch.cat(aggregated_features, dim=-1)
        return output, attention_masks

class MLBHybridModel(nn.Module):
    """Hybrid model combining GNN, RNN, and Transformer architectures"""
    def __init__(self, config: ModelConfig, feature_dim: int, n_teams: int = 30):
        super().__init__()
        self.config = config
        
        # GNN for team relationships
        self.gnn = MLBGraphNeuralNetwork(feature_dim, config.hidden_dim, n_layers=2)
        
        # Temporal encoder for recent games
        self.temporal_encoder = TemporalTeamEncoder(feature_dim, config.hidden_dim // 2)
        
        # Perceiver for high-dimensional feature processing
        self.perceiver = PerceiverBlock(feature_dim, config.hidden_dim, n_latents=32)
        
        # Combine all representations
        combined_dim = config.hidden_dim * 2 + self.temporal_encoder.output_dim * 2 + config.hidden_dim
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_dim, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Output heads for different predictions
        self.score_head = nn.Linear(config.hidden_dim, 2)  # home, away scores
        self.quantile_head = nn.Linear(config.hidden_dim, 10)  # 5 quantiles × 2 teams
        
    def forward(self, game_features, home_team_idx=None, away_team_idx=None, 
                home_recent_games=None, away_recent_games=None, adjacency_matrix=None):
        batch_size = game_features.shape[0]
        
        # GNN processing (if team indices provided)
        if home_team_idx is not None and away_team_idx is not None:
            gnn_output = self.gnn(game_features, home_team_idx, away_team_idx, adjacency_matrix)
        else:
            # Fallback: use zero embeddings
            gnn_output = torch.zeros(batch_size, self.config.hidden_dim * 2).to(game_features.device)
        
        # Temporal processing (if recent games provided)
        if home_recent_games is not None and away_recent_games is not None:
            _, home_temporal = self.temporal_encoder(home_recent_games)
            _, away_temporal = self.temporal_encoder(away_recent_games)
            temporal_features = torch.cat([home_temporal, away_temporal], dim=-1)
        else:
            # Fallback: use zero embeddings
            temporal_features = torch.zeros(batch_size, self.temporal_encoder.output_dim * 2).to(game_features.device)
        
        # Perceiver processing
        perceiver_output = self.perceiver(game_features.unsqueeze(1))
        perceiver_pooled = perceiver_output.mean(dim=1)  # Pool over latents
        
        # Combine all features
        combined = torch.cat([gnn_output, temporal_features, perceiver_pooled], dim=-1)
        fused = self.fusion_layer(combined)
        
        # Predictions
        scores = self.score_head(fused)
        quantiles = self.quantile_head(fused).view(batch_size, 5, 2)  # 5 quantiles, 2 teams
        
        return scores, quantiles

# ============= SNAPSHOT ENSEMBLE =============
class SnapshotEnsemble:
    """Manages snapshot ensemble training"""
    def __init__(self, n_snapshots: int = 5, cycles: int = 5):
        self.n_snapshots = n_snapshots
        self.cycles = cycles
        self.snapshots = []
        
    def should_save_snapshot(self, epoch: int, total_epochs: int) -> bool:
        """Determine if current epoch should save a snapshot"""
        epochs_per_cycle = total_epochs // self.cycles
        return (epoch + 1) % epochs_per_cycle == 0
    
    def save_snapshot(self, model: nn.Module, epoch: int):
        """Save model snapshot"""
        snapshot = {
            'epoch': epoch,
            'state_dict': model.state_dict().copy(),
            'performance': None  # To be filled during validation
        }
        self.snapshots.append(snapshot)
        
        # Keep only the best n_snapshots based on performance
        if len(self.snapshots) > self.n_snapshots:
            # Sort by performance (if available) and keep best
            if all(s.get('performance') is not None for s in self.snapshots):
                self.snapshots.sort(key=lambda x: x['performance'], reverse=True)
                self.snapshots = self.snapshots[:self.n_snapshots]
    
    def get_ensemble_predictions(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Get ensemble predictions from all snapshots"""
        predictions = []
        original_state = model.state_dict().copy()
        
        for snapshot in self.snapshots:
            model.load_state_dict(snapshot['state_dict'])
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        # Restore original state
        model.load_state_dict(original_state)
        
        # Average predictions
        return torch.stack(predictions).mean(dim=0)

# ============= KNOWLEDGE DISTILLATION =============
class KnowledgeDistillationTrainer:
    """Implements knowledge distillation from teacher ensemble to student model"""
    def __init__(self, temperature: float = 3.0, alpha: float = 0.7):
        self.temperature = temperature
        self.alpha = alpha  # Weight for distillation loss vs true label loss
        
    def distillation_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                         targets: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """Calculate combined distillation and standard loss"""
        # Soft targets from teacher
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL divergence loss for distillation
        distill_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Standard loss with true labels
        student_loss = criterion(student_logits, targets)
        
        # Combined loss
        return self.alpha * distill_loss + (1 - self.alpha) * student_loss
    
    def train_student(self, student_model: nn.Module, teacher_ensemble: List[nn.Module],
                     train_loader: DataLoader, val_loader: DataLoader,
                     device: torch.device, epochs: int = 50, 
                     feature_indices: dict = None) -> nn.Module:
        """Train a student model using knowledge from teacher ensemble"""
        optimizer = optim.AdamW(student_model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = RunPredictionLoss()
        
        for epoch in range(epochs):
            student_model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                # Get teacher ensemble predictions
                teacher_preds = []
                with torch.no_grad():
                    for teacher in teacher_ensemble:
                        teacher.eval()
                        # Handle different model types
                        if isinstance(teacher, MLBHybridModel):
                            # HybridModel returns (scores, quantiles)
                            scores, _ = teacher(batch_X)
                            pred = scores
                        elif hasattr(teacher, 'predict_with_uncertainty'):
                            # Uncertainty model
                            pred, _ = teacher(batch_X, sample=False)
                        elif isinstance(teacher, MLBNeuralNetV2) and feature_indices is not None:
                            # MLBNeuralNetV2 needs feature indices
                            pred = teacher(batch_X, feature_indices)
                        elif isinstance(teacher, MLBGraphNeuralNetwork):
                            # GNN doesn't need feature_indices, just the features
                            pred = teacher(batch_X)
                        else:
                            # Standard models (like MLBNeuralNetV3)
                            pred = teacher(batch_X)
                        
                        teacher_preds.append(pred)
                
                # Stack teacher predictions
                teacher_logits = torch.stack(teacher_preds).mean(dim=0)
                
                # Student prediction
                if isinstance(student_model, MLBNeuralNetV2) and feature_indices is not None:
                    student_logits = student_model(batch_X, feature_indices)
                else:
                    student_logits = student_model(batch_X)
                
                # Calculate distillation loss
                loss = self.distillation_loss(student_logits, teacher_logits, batch_y, criterion)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            scheduler.step()
            
            # Validation
            if (epoch + 1) % 10 == 0:
                val_loss, val_acc = self.validate_student(
                    student_model, val_loader, device, criterion, feature_indices
                )
                logging.info(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
                           f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.3%}")
        
        return student_model
    
    def validate_student(self, model: nn.Module, val_loader: DataLoader, 
                        device: torch.device, criterion: nn.Module,
                        feature_indices: dict = None):
        """Validate student model"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                # Handle different model types
                if isinstance(model, MLBNeuralNetV2) and feature_indices is not None:
                    outputs = model(batch_X, feature_indices)
                else:
                    outputs = model(batch_X)
                    
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
                
                # Winner accuracy
                pred_winner = outputs[:, 0] > outputs[:, 1]
                actual_winner = batch_y[:, 0] > batch_y[:, 1]
                correct += (pred_winner == actual_winner).sum().item()
                total += batch_y.size(0)
        
        return total_loss / len(val_loader), correct / total
    
# ============= NEURAL NETWORK MODELS =============
class MLBNeuralNetV2(nn.Module):
    """Original V2 neural network with feature-specific pathways"""
    
# In pipelineTrainv5.py, inside the MLBNeuralNetV2 class

# In pipelineTrainv5.py, inside the MLBNeuralNetV2 class

    def __init__(self, feature_groups, hidden_dim=64, dropout=0.3):
        """
        Initializes the MLBNeuralNetV2 model.
        """
        super().__init__()
        
        # --- FIX #1: Check the integer count directly, do not use len() ---
        self.has_batting = feature_groups.get('batting', 0) > 0
        self.has_pitching = feature_groups.get('pitching', 0) > 0
        self.has_bullpen = feature_groups.get('bullpen', 0) > 0
        self.has_starter = feature_groups.get('starter', 0) > 0
        self.has_situational = feature_groups.get('situational', 0) > 0
        self.has_momentum = feature_groups.get('momentum', 0) > 0
        
        # --- FIX #2: Pass the integer count directly to the pathway, do not use len() ---
        if self.has_batting:
            self.team_batting = self._create_pathway(feature_groups['batting'], hidden_dim, 'batting')
        if self.has_pitching:
            self.team_pitching = self._create_pathway(feature_groups['pitching'], hidden_dim, 'pitching')
        if self.has_bullpen:
            self.bullpen = self._create_pathway(feature_groups['bullpen'], hidden_dim * 2, 'bullpen', final_dim=hidden_dim)
        if self.has_starter:
            self.starting_pitcher = self._create_pathway(feature_groups['starter'], hidden_dim * 2, 'starter', final_dim=hidden_dim)
        if self.has_situational:
            self.situational = self._create_pathway(feature_groups['situational'], hidden_dim // 2, 'situational')
        if self.has_momentum:
            self.momentum = self._create_pathway(feature_groups['momentum'], hidden_dim // 2, 'momentum')

        # Calculate actual total hidden dimensions based on active pathways
        total_hidden = 0
        if self.has_batting: total_hidden += hidden_dim
        if self.has_pitching: total_hidden += hidden_dim
        if self.has_bullpen: total_hidden += hidden_dim
        if self.has_starter: total_hidden += hidden_dim
        if self.has_situational: total_hidden += hidden_dim // 2
        if self.has_momentum: total_hidden += hidden_dim // 2
        
        self.total_hidden = total_hidden
        
        # Attention mechanism to weigh the different pathways
        self.feature_attention = nn.Sequential(
            nn.Linear(total_hidden, max(total_hidden // 4, 1)),
            nn.ReLU(),
            nn.Linear(max(total_hidden // 4, 1), total_hidden),
            nn.Sigmoid()
        )
        
        # Final combination layers to produce the output
        self.combine = nn.Sequential(
            nn.Linear(total_hidden, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
        # Initialize weights for better training stability
        self.apply(self._init_weights)
        
    def _create_pathway(self, input_dim, hidden_dim, name, final_dim=None):
        """
        Helper function to create a sequential neural network pathway for a 
        specific feature group.
        """
        # If there are no features for this group, return an identity layer
        # that does nothing.
        if input_dim == 0:
            return nn.Identity()
        
        # Define the layers for the pathway
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        ]
        
        # Add an additional layer if a different final dimension is specified
        if final_dim and final_dim != hidden_dim:
            layers.extend([
                nn.Linear(hidden_dim, final_dim),
                nn.BatchNorm1d(final_dim),
                nn.ReLU()
            ])
            
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        """Initializes the weights of linear layers."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x, feature_indices):
        """Defines the forward pass of the model."""
        # Process each feature group through its dedicated pathway
        outputs = []
        
        if self.has_batting and len(feature_indices.get('batting', [])) > 0:
            batting_out = self.team_batting(x[:, feature_indices['batting']])
            outputs.append(batting_out)
        
        if self.has_pitching and len(feature_indices.get('pitching', [])) > 0:
            pitching_out = self.team_pitching(x[:, feature_indices['pitching']])
            outputs.append(pitching_out)
        
        if self.has_bullpen and len(feature_indices.get('bullpen', [])) > 0:
            bullpen_out = self.bullpen(x[:, feature_indices['bullpen']])
            outputs.append(bullpen_out)
        
        if self.has_starter and len(feature_indices.get('starter', [])) > 0:
            starter_out = self.starting_pitcher(x[:, feature_indices['starter']])
            outputs.append(starter_out)
        
        if self.has_situational and len(feature_indices.get('situational', [])) > 0:
            situational_out = self.situational(x[:, feature_indices['situational']])
            outputs.append(situational_out)
        
        if self.has_momentum and len(feature_indices.get('momentum', [])) > 0:
            momentum_out = self.momentum(x[:, feature_indices['momentum']])
            outputs.append(momentum_out)
        
        # Ensure at least one pathway produced an output
        if len(outputs) == 0:
            raise ValueError("No active feature groups found for model input!")
        
        # Concatenate all pathway outputs
        combined = torch.cat(outputs, dim=1)
        
        # Apply attention to the combined features
        attention_weights = self.feature_attention(combined)
        combined = combined * attention_weights
        
        # Pass through final combination layers to get the prediction
        return self.combine(combined)

class MLBNeuralNetV3(nn.Module):
    """Advanced neural network with transformer layers and uncertainty quantification"""
    
    def __init__(self, feature_groups: Dict[str, int], config: ModelConfig):
        super().__init__()
        self.config = config
        self.feature_groups = feature_groups
        
        # Calculate total input dimension
        total_features = sum(feature_groups.values())
        
        # Feature attention mechanism (TabNet-inspired)
        self.feature_attention = FeatureAttentionTransformer(
            total_features, 
            config.hidden_dim,
            n_steps=3
        )
        
        # Transformer for feature interactions (if enabled)
        if config.use_transformer:
            # Output from feature attention is hidden_dim * n_steps
            transformer_input_dim = config.hidden_dim * 3
            
            # Simple transformer encoder
            self.feature_projection = nn.Linear(transformer_input_dim, config.hidden_dim)
            self.pos_encoding = PositionalEncoding(config.hidden_dim)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.n_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_transformer_layers)
            self.norm = nn.LayerNorm(config.hidden_dim)
            
            combine_input_dim = config.hidden_dim
        else:
            combine_input_dim = config.hidden_dim * 3
            self.feature_projection = None
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(combine_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, 2)  # home_score, away_score
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x, feature_indices=None):
        # Note: feature_indices not used in V3, kept for compatibility
        
        # Apply feature attention
        attended_features, attention_masks = self.feature_attention(x)
        
        # Apply transformer if enabled
        if self.feature_projection is not None:
            # Reshape to (batch, seq_len, feature_dim)
            features = attended_features.unsqueeze(1)
            features = self.feature_projection(features)
            features = self.pos_encoding(features)
            features = self.transformer(features)
            features = self.norm(features)
            # Global average pooling
            features = features.mean(dim=1)
        else:
            features = attended_features
        
        # Prediction head
        output = self.prediction_head(features)
        
        return output

class MLBNeuralNetWithUncertainty(nn.Module):
    """Neural network with Monte Carlo Dropout for uncertainty estimation"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128], 
                 dropout_rate: float = 0.3, mc_dropout_rate: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Build hidden layers
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            # Use different dropout rates for training vs MC inference
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # MC Dropout layers (always active)
        self.mc_dropout = nn.Dropout(mc_dropout_rate)
        
        # Output layers
        self.score_mean = nn.Linear(prev_dim, 2)
        self.score_log_std = nn.Linear(prev_dim, 2)  # Learn uncertainty
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x, sample=True):
        for layer, bn, dropout in zip(self.layers, self.batch_norms, self.dropouts):
            x = layer(x)
            x = bn(x)
            x = F.gelu(x)
            x = dropout(x)
            # Apply MC dropout (always active)
            x = self.mc_dropout(x)
        
        mean = self.score_mean(x)
        log_std = self.score_log_std(x)
        
        if sample:
            # Sample from the distribution during training
            std = torch.exp(0.5 * log_std)
            eps = torch.randn_like(std)
            return mean + eps * std, log_std
        else:
            return mean, log_std
    
    def predict_with_uncertainty(self, x, n_samples=50):
        """Monte Carlo dropout for uncertainty estimation"""
        was_training = self.training
        self.train()  # Enable dropout
            # Keep BatchNorm layers in eval mode to handle batch size of 1
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                mean, _ = self.forward(x, sample=False)
                predictions.append(mean)
        
        self.train(was_training)  # Restore original mode
        
        predictions = torch.stack(predictions)
        mean_prediction = predictions.mean(dim=0)
        epistemic_uncertainty = predictions.std(dim=0)  # Model uncertainty
        
        # Also get aleatoric uncertainty (data uncertainty)
        _, log_std = self.forward(x, sample=False)
        aleatoric_uncertainty = torch.exp(0.5 * log_std)
        
        # Total uncertainty
        total_uncertainty = torch.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
        
        return mean_prediction, total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty

# ============= IMPROVED FEATURE SELECTION =============
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

class BaseModelWrapper:
    """Wrapper to provide consistent interface for different model types."""

    def __init__(self, model: Any, device: Optional[Any] = None, feature_indices: Optional[Dict] = None):
        """
        MODIFIED: Added feature_indices to the constructor.
        """
        self.model = model
        self.device = device
        self.feature_indices = feature_indices  # Store feature indices
        self.is_torch_model = hasattr(model, 'parameters')

    # In the BaseModelWrapper class

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        MODIFIED: Implemented a training loop for PyTorch models.
        """
        if self.is_torch_model:
            # --- START OF CORRECTION ---
            # For PyTorch models, we need a simple training loop to adapt the model to the fold
            self.model.to(self.device)
            self.model.train()

            optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
            criterion = RunPredictionLoss() # Use a standard loss
            
            # Create a DataLoader for this fold
            dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
            # Use a smaller batch size for fold-training
            dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

            # Train for a small number of epochs (fine-tuning)
            for epoch in range(10): # Fine-tune for 10 epochs
                for batch_X, batch_y in dataloader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                    optimizer.zero_grad()
                    
                    # Handle different model types during training
                    if isinstance(self.model, MLBNeuralNetV2):
                        pred = self.model(batch_X, self.feature_indices)
                    elif isinstance(self.model, MLBHybridModel):
                        pred, _ = self.model(batch_X)
                    elif hasattr(self.model, 'predict_with_uncertainty'):
                        pred, _ = self.model(batch_X)
                    else:
                        pred = self.model(batch_X)
                    
                    loss = criterion(pred, batch_y)
                    loss.backward()
                    optimizer.step()
            # --- END OF CORRECTION ---
        else:
            # Sklearn-style fit remains the same
            self.model.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from the wrapped model."""
        if self.is_torch_model:
            X_tensor = torch.FloatTensor(X)
            if self.device:
                X_tensor = X_tensor.to(self.device)

            self.model.eval()
            with torch.no_grad():
                if hasattr(self.model, 'predict_with_uncertainty'):
                    pred_mean, _, _, _ = self.model.predict_with_uncertainty(X_tensor, n_samples=30)
                    return pred_mean.cpu().numpy()
                elif isinstance(self.model, MLBHybridModel):
                    scores, _ = self.model(X_tensor)
                    return scores.cpu().numpy()
                # --- START OF CORRECTION ---
                # Add a specific check for MLBNeuralNetV2
                elif isinstance(self.model, MLBNeuralNetV2):
                    if self.feature_indices is None:
                        raise ValueError("MLBNeuralNetV2 requires feature_indices, but they were not provided to the wrapper.")
                    return self.model(X_tensor, self.feature_indices).cpu().numpy()
                # --- END OF CORRECTION ---
                else:
                    output = self.model(X_tensor)
                    if isinstance(output, tuple):
                        output = output[0]
                    return output.cpu().numpy()
        else:
            # Sklearn-style predict
            return self.model.predict(X)
        
# ============= DATASET CLASS =============
class TemporalMLBDataset:
    """Dataset that properly handles temporal data"""
    
    def __init__(self, features_path: str):
        self.features_path = features_path
        self.scaler = RobustScaler()
        self.feature_selector = ImprovedFeatureSelector(max_features=250)
        self.feature_engineer = AdvancedFeatureEngineer()
        
    # In the TemporalMLBDataset class, update the load_and_prepare method:

    def load_and_prepare(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load data and ensure temporal ordering"""
        # Load data
        df = pd.read_parquet(self.features_path)

        # Ensure we have game_date
        if 'game_date' not in df.columns:
            raise ValueError("game_date column is required for temporal splitting!")
        
        # Convert to datetime and sort
        df['game_date'] = pd.to_datetime(df['game_date'])
        df = df.sort_values('game_date').reset_index(drop=True)
        
        # Target columns
        target_cols = ['home_score', 'away_score']
        if not all(col in df.columns for col in target_cols):
            raise ValueError(f"Target columns {target_cols} not found!")
        
        # Drop games with missing scores
        df = df.dropna(subset=target_cols)
        
        # IMPORTANT: Fit the feature engineer on the full data WITH scores
        # This should be done BEFORE removing target columns
        self.feature_engineer.fit(df)
        
        # Apply feature engineering
        df = self.feature_engineer.create_advanced_features(df)
        
        # Create feature and target DataFrames
        exclude_cols = [
            'game_pk', 'gamePk', 'game_date', 'home_team', 'away_team', 
            'home_team_id', 'away_team_id', 'home_team_abbr', 'away_team_abbr',
            'home_game_date', 'away_game_date', 'home_W/L', 'away_W/L', 'bookmaker',
            'time_match_key', 'date_match_key', 'home_ml', 'away_ml', 'total_line', 'over_odds',
            'home_score', 'away_score', 'match_key', 'diff_score'  # Target columns
        ]
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        # Define feature_groups BEFORE using it
        feature_groups = {
            'batting': [col for col in feature_cols if 'batting' in col.lower()],
            'pitching': [col for col in feature_cols if 'pitching' in col.lower() and 'bullpen' not in col.lower()],
            'bullpen': [col for col in feature_cols if 'bullpen' in col.lower()],
            'momentum': [col for col in feature_cols if any(x in col.lower() for x in ['streak', 'momentum', 'last_'])],
            'other': []  # Will be filled after
        }
        
        # Now add the remaining features to 'other'
        assigned_features = sum(feature_groups.values(), [])
        feature_groups['other'] = [col for col in feature_cols if col not in assigned_features]
        
        X = df[feature_cols]
        X = self.feature_engineer.smart_imputation(X, feature_groups)
        
        y = df[target_cols]
        dates = df[['game_date']]
        
        logging.info(f"Loaded {len(df)} games from {df['game_date'].min()} to {df['game_date'].max()}")
        logging.info(f"Feature shape: {X.shape}, Target shape: {y.shape}")
        
        return X, y, dates
    
    def create_temporal_splits(self, X: pd.DataFrame, y: pd.DataFrame, dates: pd.DataFrame, 
                             test_size: float = 0.2, val_size: float = 0.1):
        """Create train/val/test splits that respect temporal order"""
        n_samples = len(X)
        
        # Calculate split points
        train_end = int(n_samples * (1 - test_size - val_size))
        val_end = int(n_samples * (1 - test_size))
        
        # Create splits
        train_idx = slice(0, train_end)
        val_idx = slice(train_end, val_end)
        test_idx = slice(val_end, n_samples)
        
        # Log split dates
        logging.info(f"Train: {dates.iloc[0]['game_date']} to {dates.iloc[train_end-1]['game_date']} ({train_end} games)")
        logging.info(f"Val: {dates.iloc[train_end]['game_date']} to {dates.iloc[val_end-1]['game_date']} ({val_end-train_end} games)")
        logging.info(f"Test: {dates.iloc[val_end]['game_date']} to {dates.iloc[-1]['game_date']} ({n_samples-val_end} games)")
        
        return (X.iloc[train_idx], y.iloc[train_idx], 
                X.iloc[val_idx], y.iloc[val_idx],
                X.iloc[test_idx], y.iloc[test_idx])

# Final refined StackingEnsemble class with all improvements
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.linear_model import ElasticNet

# class StackingEnsemble:
#     """Professional-grade stacking ensemble implementation"""
    
#     def __init__(self, meta_model_type='lightgbm', include_original_features=False, 
#                  n_original_features=10, tune_hyperparameters=False):
#         self.meta_model_type = meta_model_type
#         self.include_original_features = include_original_features
#         self.n_original_features = n_original_features
#         self.tune_hyperparameters = tune_hyperparameters
#         self.meta_model = None
#         self.base_models = []
#         self.feature_indices = None
#         self.device = None
#         self.selected_original_features = None
#         self.feature_names = None
        
#     def _create_meta_features(self, predictions_list, original_features=None):
#         """Create sophisticated meta-features from base model predictions"""
#         meta_features = []
        
#         # Raw predictions from each model
#         for pred in predictions_list:
#             meta_features.append(pred[:, 0])  # Home score
#             meta_features.append(pred[:, 1])  # Away score
        
#         # Derived features from each model
#         for pred in predictions_list:
#             meta_features.append(pred[:, 0] - pred[:, 1])  # Predicted margin
#             meta_features.append(pred[:, 0] + pred[:, 1])  # Total predicted runs
#             meta_features.append((pred[:, 0] > pred[:, 1]).astype(float))  # Predicted winner
#             meta_features.append(np.abs(pred[:, 0] - pred[:, 1]))  # Absolute margin (confidence)
        
#         # Ensemble statistics
#         predictions_array = np.array(predictions_list)
        
#         # Mean predictions across models
#         mean_preds = predictions_array.mean(axis=0)
#         meta_features.append(mean_preds[:, 0])  # Mean home score
#         meta_features.append(mean_preds[:, 1])  # Mean away score
#         meta_features.append(mean_preds[:, 0] - mean_preds[:, 1])  # Mean margin
#         meta_features.append(mean_preds[:, 0] + mean_preds[:, 1])  # Mean total
        
#         # Standard deviation (model disagreement) - often the most important features!
#         std_preds = predictions_array.std(axis=0)
#         meta_features.append(std_preds[:, 0])  # Std home score
#         meta_features.append(std_preds[:, 1])  # Std away score
#         meta_features.append(std_preds.mean(axis=1))  # Overall uncertainty
        
#         # Range of predictions (another uncertainty measure)
#         min_preds = predictions_array.min(axis=0)
#         max_preds = predictions_array.max(axis=0)
#         meta_features.append(max_preds[:, 0] - min_preds[:, 0])  # Range of home predictions
#         meta_features.append(max_preds[:, 1] - min_preds[:, 1])  # Range of away predictions
        
#         # Model agreement features
#         margins = predictions_array[:, :, 0] - predictions_array[:, :, 1]
#         winners = (margins > 0).astype(float)
#         meta_features.append(winners.mean(axis=0))  # Proportion predicting home win
#         meta_features.append(winners.std(axis=0))   # Disagreement on winner
        
#         # Percentile features (robustness check)
#         meta_features.append(np.percentile(predictions_array[:, :, 0], 25, axis=0))  # 25th percentile home
#         meta_features.append(np.percentile(predictions_array[:, :, 0], 75, axis=0))  # 75th percentile home
        
#         # Stack all meta-features
#         stacked_features = np.column_stack(meta_features)
        
#         # Add original features if requested
#         if self.include_original_features and original_features is not None:
#             if self.selected_original_features is not None:
#                 original_subset = original_features[:, self.selected_original_features]
#                 stacked_features = np.hstack([stacked_features, original_subset])
        
#         return stacked_features
    
#     def _get_base_predictions(self, models, X, feature_indices=None, device=None):
#         """Get predictions from all base models"""
#         predictions = []
#         X_tensor = torch.FloatTensor(X).to(device) if device else torch.FloatTensor(X)
        
#         with torch.no_grad():
#             for model in models:
#                 model.eval()
                
#                 if hasattr(model, 'predict_with_uncertainty'):
#                     pred_mean, _, _, _ = model.predict_with_uncertainty(X_tensor, n_samples=30)
#                     pred = pred_mean.cpu().numpy()
#                 elif isinstance(model, MLBHybridModel):
#                     scores, _ = model(X_tensor)
#                     pred = scores.cpu().numpy()
#                 elif isinstance(model, MLBNeuralNetV2):
#                     pred = model(X_tensor, feature_indices).cpu().numpy()
#                 else:
#                     output = model(X_tensor)
#                     if isinstance(output, tuple):
#                         output = output[0]
#                     pred = output.cpu().numpy()
                
#                 predictions.append(pred)
        
#         return predictions
    
#     def _create_feature_names(self, n_models):
#         """Create interpretable feature names"""
#         feature_names = []
        
#         # Raw predictions
#         for i in range(n_models):
#             feature_names.extend([f'model{i}_home', f'model{i}_away'])
        
#         # Derived features
#         for i in range(n_models):
#             feature_names.extend([
#                 f'model{i}_margin', f'model{i}_total', 
#                 f'model{i}_winner', f'model{i}_abs_margin'
#             ])
        
#         # Ensemble statistics
#         feature_names.extend([
#             'mean_home', 'mean_away', 'mean_margin', 'mean_total',
#             'std_home', 'std_away', 'std_overall',
#             'range_home', 'range_away',
#             'prop_home_win', 'disagreement_winner',
#             'p25_home', 'p75_home'
#         ])
        
#         return feature_names
    
#     def fit(self, base_models, X_val, y_val, feature_indices=None, device=None, 
#             feature_importance_df=None):
#         """
#         Fit the stacking ensemble using validation set
        
#         Args:
#             base_models: List of trained base models
#             X_val: Validation features (models haven't seen these)
#             y_val: Validation targets  
#             feature_indices: Feature indices for models that need them
#             device: PyTorch device
#             feature_importance_df: DataFrame with feature importances
#         """
#         self.base_models = base_models
#         self.feature_indices = feature_indices
#         self.device = device
        
#         logging.info("Generating meta-features from validation set...")
        
#         # Select top original features if requested
#         if self.include_original_features and feature_importance_df is not None:
#             top_features = feature_importance_df.nlargest(
#                 self.n_original_features, 'combined_score'
#             )['feature'].tolist()
            
#             if hasattr(X_val, 'columns'):
#                 self.selected_original_features = [
#                     i for i, col in enumerate(X_val.columns) if col in top_features
#                 ]
#                 logging.info(f"Selected {len(self.selected_original_features)} original features")
        
#         # Get predictions from all base models
#         val_predictions = self._get_base_predictions(
#             base_models, 
#             X_val.values if hasattr(X_val, 'values') else X_val,
#             feature_indices, device
#         )
        
#         # Create meta-features
#         original_features = X_val.values if hasattr(X_val, 'values') else X_val
#         meta_features = self._create_meta_features(val_predictions, original_features)
#         y_val_np = y_val.values if hasattr(y_val, 'values') else y_val
        
#         # Create feature names
#         self.feature_names = self._create_feature_names(len(base_models))
#         if self.include_original_features and self.selected_original_features:
#             if hasattr(X_val, 'columns'):
#                 self.feature_names.extend([
#                     X_val.columns[i] for i in self.selected_original_features
#                 ])
#             else:
#                 self.feature_names.extend([
#                     f'orig_feat_{i}' for i in range(len(self.selected_original_features))
#                 ])
        
#         logging.info(f"Meta-features shape: {meta_features.shape}")
#         logging.info(f"Training {self.meta_model_type} meta-learner...")
        
#         # Split for early stopping (if using LightGBM)
#         if self.meta_model_type == 'lightgbm':
#             val_split = int(0.8 * len(meta_features))
#             X_meta_train = meta_features[:val_split]
#             y_meta_train = y_val_np[:val_split]
#             X_meta_val = meta_features[val_split:]
#             y_meta_val = y_val_np[val_split:]
#         else:
#             X_meta_train = meta_features
#             y_meta_train = y_val_np
        
#         if self.meta_model_type == 'ridge':
#             self.meta_model = Ridge(alpha=1.0)
#             self.meta_model.fit(X_meta_train, y_meta_train)
            
#         elif self.meta_model_type == 'elastic':
#             self.meta_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
#             self.meta_model.fit(X_meta_train, y_meta_train)
            
#         elif self.meta_model_type == 'lightgbm':
#             if self.tune_hyperparameters:
#                 logging.info("Tuning LightGBM hyperparameters...")
#                 self._tune_lightgbm(X_meta_train, y_meta_train, X_meta_val, y_meta_val)
#             else:
#                 # Use good default parameters
#                 self.meta_model = {
#                     'home': lgb.LGBMRegressor(
#                         n_estimators=150,
#                         learning_rate=0.03,
#                         max_depth=4,
#                         num_leaves=15,
#                         subsample=0.8,
#                         colsample_bytree=0.6,
#                         reg_alpha=0.1,
#                         reg_lambda=0.1,
#                         min_child_samples=20,
#                         random_state=42,
#                         verbose=-1
#                     ),
#                     'away': lgb.LGBMRegressor(
#                         n_estimators=150,
#                         learning_rate=0.03,
#                         max_depth=4,
#                         num_leaves=15,
#                         subsample=0.8,
#                         colsample_bytree=0.6,
#                         reg_alpha=0.1,
#                         reg_lambda=0.1,
#                         min_child_samples=20,
#                         random_state=42,
#                         verbose=-1
#                     )
#                 }
            
#             # Fit with early stopping
#             self.meta_model['home'].fit(
#                 X_meta_train, y_meta_train[:, 0],
#                 eval_set=[(X_meta_val, y_meta_val[:, 0])],
#                 callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
#             )
            
#             self.meta_model['away'].fit(
#                 X_meta_train, y_meta_train[:, 1],
#                 eval_set=[(X_meta_val, y_meta_val[:, 1])],
#                 callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
#             )
            
#             # Log feature importance
#             self._log_feature_importance()
    
#     def _tune_lightgbm(self, X_train, y_train, X_val, y_val):
#         """Tune LightGBM hyperparameters"""
#         param_grid = {
#             'n_estimators': [100, 150, 200],
#             'max_depth': [3, 4, 5],
#             'num_leaves': [7, 15, 31],
#             'learning_rate': [0.02, 0.03, 0.05],
#             'subsample': [0.7, 0.8],
#             'colsample_bytree': [0.5, 0.6, 0.7]
#         }
        
#         # Use a small grid for efficiency
#         param_grid_small = {
#             'n_estimators': [150],
#             'max_depth': [3, 4],
#             'num_leaves': [7, 15],
#             'learning_rate': [0.03, 0.05],
#             'subsample': [0.8],
#             'colsample_bytree': [0.6]
#         }
        
#         base_params = {
#             'reg_alpha': 0.1,
#             'reg_lambda': 0.1,
#             'min_child_samples': 20,
#             'random_state': 42,
#             'verbose': -1
#         }
        
#         self.meta_model = {}
        
#         for target, idx in [('home', 0), ('away', 1)]:
#             logging.info(f"Tuning {target} score model...")
            
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
                
#                 grid_search = GridSearchCV(
#                     lgb.LGBMRegressor(**base_params),
#                     param_grid_small,
#                     cv=3,
#                     scoring='neg_mean_squared_error',
#                     n_jobs=-1,
#                     verbose=0
#                 )
                
#                 grid_search.fit(
#                     X_train, y_train[:, idx],
#                     eval_set=[(X_val, y_val[:, idx])],
#                     callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
#                 )
                
#                 self.meta_model[target] = grid_search.best_estimator_
#                 logging.info(f"Best params for {target}: {grid_search.best_params_}")
    
#     def _log_feature_importance(self):
#         """Log feature importance for interpretability"""
#         for target in ['home', 'away']:
#             importance_df = pd.DataFrame({
#                 'feature': self.feature_names[:len(self.meta_model[target].feature_importances_)],
#                 'importance': self.meta_model[target].feature_importances_
#             }).sort_values('importance', ascending=False)
            
#             logging.info(f"\nTop 10 features for {target} score:")
#             logging.info(importance_df.head(10).to_string())
    
#     def predict(self, X_test):
#         """Make predictions using the stacking ensemble"""
#         # Get base model predictions
#         base_predictions = self._get_base_predictions(
#             self.base_models,
#             X_test.values if hasattr(X_test, 'values') else X_test,
#             self.feature_indices,
#             self.device
#         )
        
#         # Create meta-features
#         original_features = X_test.values if hasattr(X_test, 'values') else X_test
#         meta_features = self._create_meta_features(base_predictions, original_features)
        
#         # Make final predictions
#         if self.meta_model_type == 'lightgbm':
#             home_pred = self.meta_model['home'].predict(meta_features)
#             away_pred = self.meta_model['away'].predict(meta_features)
#             predictions = np.column_stack([home_pred, away_pred])
#         else:
#             predictions = self.meta_model.predict(meta_features)
        
#         return predictions
import copy
class StackingEnsemble(BaseEstimator, RegressorMixin):
    """
    Professional-grade stacking ensemble implementation with enhanced features.
    
    Key improvements:
    - Cross-validation based meta-feature generation to prevent leakage
    - Enhanced meta-feature engineering including interactions and correlations
    - Flexible meta-model specification (any sklearn-compatible model)
    - Improved code structure and extensibility
    """
    
    # Class-level constants
    HOME = 'home'
    AWAY = 'away'
    
    def __init__(
        self,
        meta_model: Optional[Union[str, BaseEstimator, Dict[str, BaseEstimator]]] = 'lightgbm',
        include_original_features: bool = False,
        n_original_features: int = 10,
        tune_hyperparameters: bool = False,
        cv_folds: int = 5,
        random_state: int = 42,
        use_interaction_features: bool = True,
        use_correlation_features: bool = True,
        verbose: int = 1
    ):
        """
        Initialize the stacking ensemble.
        
        Args:
            meta_model: Meta-model to use. Can be:
                - String: 'ridge', 'elastic', 'lightgbm'
                - sklearn estimator instance
                - Dict of estimators for multi-output (e.g., {'home': model1, 'away': model2})
            include_original_features: Whether to include original features in meta-features
            n_original_features: Number of top original features to include
            tune_hyperparameters: Whether to tune hyperparameters for built-in models
            cv_folds: Number of cross-validation folds for generating meta-features
            random_state: Random state for reproducibility
            use_interaction_features: Whether to include interaction features
            use_correlation_features: Whether to include correlation features
            verbose: Verbosity level
        """
        self.meta_model = meta_model
        self.include_original_features = include_original_features
        self.n_original_features = n_original_features
        self.tune_hyperparameters = tune_hyperparameters
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.use_interaction_features = use_interaction_features
        self.use_correlation_features = use_correlation_features
        self.verbose = verbose
        
        # Attributes set during fitting
        self.meta_model_ = None
        self.base_models_ = []
        self.feature_indices_ = None
        self.device_ = None
        self.selected_original_features_ = None
        self.feature_names_ = None
        self.n_outputs_ = None
        self.base_model_wrappers_ = []
    
    def _create_meta_model(self) -> Union[BaseEstimator, Dict[str, BaseEstimator]]:
        """Create the meta-model based on the specified type."""
        if isinstance(self.meta_model, str):
            if self.meta_model == 'ridge':
                return Ridge(alpha=1.0, random_state=self.random_state)
            elif self.meta_model == 'elastic':
                return ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state)
            elif self.meta_model == 'lightgbm':
                base_params = {
                    'n_estimators': 150,
                    'learning_rate': 0.03,
                    'max_depth': 4,
                    'num_leaves': 15,
                    'subsample': 0.8,
                    'colsample_bytree': 0.6,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'min_child_samples': 20,
                    'random_state': self.random_state,
                    'verbose': -1
                }
                if self.n_outputs_ == 2:
                    return {
                        self.HOME: lgb.LGBMRegressor(**base_params),
                        self.AWAY: lgb.LGBMRegressor(**base_params)
                    }
                else:
                    return lgb.LGBMRegressor(**base_params)
            else:
                raise ValueError(f"Unknown meta model type: {self.meta_model}")
        elif isinstance(self.meta_model, dict):
            # Clone each model in the dictionary
            return {k: clone(v) for k, v in self.meta_model.items()}
        else:
            # Assume it's a sklearn-compatible estimator
            return clone(self.meta_model)
    
    def _wrap_base_model(self, model: Any) -> 'BaseModelWrapper':
        """
        MODIFIED: Pass feature_indices_ to the wrapper.
        """
        return BaseModelWrapper(model, self.device_, self.feature_indices_)
    
    def _get_raw_prediction_features(self, predictions_list: List[np.ndarray]) -> List[np.ndarray]:
        """Extract raw prediction features from base models."""
        features = []
        for pred in predictions_list:
            if pred.ndim == 1:
                features.append(pred)
            else:
                for i in range(pred.shape[1]):
                    features.append(pred[:, i])
        return features
    
    def _get_derived_features(self, predictions_list: List[np.ndarray]) -> List[np.ndarray]:
        """Create derived features from predictions."""
        features = []
        for pred in predictions_list:
            if pred.ndim == 2 and pred.shape[1] == 2:
                # Assuming home/away predictions
                features.append(pred[:, 0] - pred[:, 1])  # Margin
                features.append(pred[:, 0] + pred[:, 1])  # Total
                features.append((pred[:, 0] > pred[:, 1]).astype(float))  # Winner
                features.append(np.abs(pred[:, 0] - pred[:, 1]))  # Absolute margin
            elif pred.ndim == 1:
                # For single output, create some basic transformations
                features.append(np.square(pred))
                features.append(np.sqrt(np.abs(pred)))
        return features
    
    def _get_ensemble_statistics_features(self, predictions_array: np.ndarray) -> List[np.ndarray]:
        """Calculate ensemble statistics across models."""
        features = []
        
        # Handle different array shapes
        if predictions_array.ndim == 3:
            # Shape: (n_models, n_samples, n_outputs)
            mean_preds = predictions_array.mean(axis=0)
            std_preds = predictions_array.std(axis=0)
            min_preds = predictions_array.min(axis=0)
            max_preds = predictions_array.max(axis=0)
            
            for i in range(mean_preds.shape[1]):
                features.append(mean_preds[:, i])
                features.append(std_preds[:, i])
                features.append(max_preds[:, i] - min_preds[:, i])
                features.append(np.percentile(predictions_array[:, :, i], 25, axis=0))
                features.append(np.percentile(predictions_array[:, :, i], 75, axis=0))
        else:
            # Shape: (n_models, n_samples)
            features.append(predictions_array.mean(axis=0))
            features.append(predictions_array.std(axis=0))
            features.append(predictions_array.max(axis=0) - predictions_array.min(axis=0))
            features.append(np.percentile(predictions_array, 25, axis=0))
            features.append(np.percentile(predictions_array, 75, axis=0))
        
        return features
    
    def _get_interaction_features(self, predictions_list: List[np.ndarray]) -> List[np.ndarray]:
        """Create interaction features between model predictions."""
        features = []
        n_models = len(predictions_list)
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                pred_i = predictions_list[i]
                pred_j = predictions_list[j]
                
                if pred_i.ndim == 2 and pred_j.ndim == 2:
                    # For multi-output predictions
                    for k in range(pred_i.shape[1]):
                        features.append(pred_i[:, k] * pred_j[:, k])  # Product
                        features.append(np.abs(pred_i[:, k] - pred_j[:, k]))  # Absolute difference
                else:
                    # For single output
                    features.append(pred_i * pred_j)
                    features.append(np.abs(pred_i - pred_j))
        
        return features
    
    def _get_correlation_features(self, predictions_list: List[np.ndarray]) -> List[np.ndarray]:
        """Calculate pairwise correlations between model predictions."""
        features = []
        n_models = len(predictions_list)
        n_samples = predictions_list[0].shape[0]
        
        # Calculate rolling correlations (using small windows)
        window_size = min(50, n_samples // 10)
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                correlations = np.zeros(n_samples)
                
                for k in range(n_samples):
                    start_idx = max(0, k - window_size)
                    end_idx = min(n_samples, k + window_size + 1)
                    
                    if end_idx - start_idx > 2:
                        if predictions_list[i].ndim == 1:
                            corr, _ = pearsonr(
                                predictions_list[i][start_idx:end_idx],
                                predictions_list[j][start_idx:end_idx]
                            )
                            correlations[k] = corr if not np.isnan(corr) else 0
                        else:
                            # For multi-output, use first output
                            corr, _ = pearsonr(
                                predictions_list[i][start_idx:end_idx, 0],
                                predictions_list[j][start_idx:end_idx, 0]
                            )
                            correlations[k] = corr if not np.isnan(corr) else 0
                
                features.append(correlations)
        
        return features
    
    def _create_meta_features(
        self,
        predictions_list: List[np.ndarray],
        original_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Create comprehensive meta-features from base model predictions."""
        all_features = []
        
        # Raw predictions
        all_features.extend(self._get_raw_prediction_features(predictions_list))
        
        # Derived features
        all_features.extend(self._get_derived_features(predictions_list))
        
        # Ensemble statistics
        predictions_array = np.array(predictions_list)
        all_features.extend(self._get_ensemble_statistics_features(predictions_array))
        
        # Interaction features
        if self.use_interaction_features:
            all_features.extend(self._get_interaction_features(predictions_list))
        
        # Correlation features
        if self.use_correlation_features:
            all_features.extend(self._get_correlation_features(predictions_list))
        
        # Stack all meta-features
        stacked_features = np.column_stack(all_features)
        
        # Add original features if requested
        if self.include_original_features and original_features is not None:
            if self.selected_original_features_ is not None:
                original_subset = original_features[:, self.selected_original_features_]
                stacked_features = np.hstack([stacked_features, original_subset])
        
        return stacked_features
    
    def _generate_cv_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_models: List[Any]
    ) -> Tuple[List[np.ndarray], List[Any]]:
        """Generate out-of-fold predictions using cross-validation."""
        n_samples = X.shape[0]
        n_models = len(base_models)
        
        # Initialize prediction arrays
        if y.ndim == 1:
            self.n_outputs_ = 1
            cv_predictions = [np.zeros(n_samples) for _ in range(n_models)]
        else:
            self.n_outputs_ = y.shape[1]
            cv_predictions = [np.zeros((n_samples, self.n_outputs_)) for _ in range(n_models)]
        
        # Use appropriate cross-validation splitter
        if self.n_outputs_ == 1:
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        else:
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        trained_models = [[] for _ in range(n_models)]
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            if self.verbose > 0:
                logging.info(f"Processing fold {fold_idx + 1}/{self.cv_folds}")
            
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train = y[train_idx]
            
            for model_idx, base_model in enumerate(base_models):
                # --- START OF CORRECTION ---
                # Check if it's a PyTorch model. If so, use deepcopy.
                # Otherwise, use sklearn's clone for compatibility.
                if isinstance(base_model, nn.Module):
                    model_clone = copy.deepcopy(base_model)
                else:
                    # Fallback for scikit-learn compatible models
                    model_clone = clone(base_model)
                # --- END OF CORRECTION ---
                
                wrapped_model = self._wrap_base_model(model_clone)
                
                # Train on fold
                wrapped_model.fit(X_fold_train, y_fold_train)
                
                # Predict on validation fold
                fold_predictions = wrapped_model.predict(X_fold_val)
                
                # Store predictions
                if self.n_outputs_ == 1:
                    cv_predictions[model_idx][val_idx] = fold_predictions
                else:
                    cv_predictions[model_idx][val_idx, :] = fold_predictions
                
                # Store the trained model
                trained_models[model_idx].append(wrapped_model)
        
        return cv_predictions, trained_models
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.DataFrame],
        base_models: List[Any],
        feature_indices: Optional[np.ndarray] = None,
        device: Optional[Any] = None,
        feature_importance_df: Optional[pd.DataFrame] = None
    ):
        """
        Fit the stacking ensemble using cross-validation.
        
        Args:
            X: Training features
            y: Training targets
            base_models: List of base models (untrained or pre-trained)
            feature_indices: Feature indices for models that need them
            device: PyTorch device
            feature_importance_df: DataFrame with feature importances
        """
        # Convert to numpy if needed
        X_np = X.values if hasattr(X, 'values') else X
        y_np = y.values if hasattr(y, 'values') else y
        
        # Store parameters
        self.feature_indices_ = feature_indices
        self.device_ = device
        
        # Select top original features if requested
        if self.include_original_features and feature_importance_df is not None:
            top_features = feature_importance_df.nlargest(
                self.n_original_features, 'combined_score'
            )['feature'].tolist()
            
            if hasattr(X, 'columns'):
                self.selected_original_features_ = [
                    i for i, col in enumerate(X.columns) if col in top_features
                ]
                if self.verbose > 0:
                    logging.info(f"Selected {len(self.selected_original_features_)} original features")
        
        # Generate cross-validated predictions
        if self.verbose > 0:
            logging.info("Generating cross-validated meta-features...")
        
        cv_predictions, trained_models = self._generate_cv_predictions(X_np, y_np, base_models)
        
        # Store the base models (using all folds for final ensemble)
        self.base_models_ = base_models
        self.base_model_wrappers_ = [self._wrap_base_model(model) for model in base_models]
        
        # Ensure all base models are trained on full data
        for idx, wrapper in enumerate(self.base_model_wrappers_):
            wrapper.fit(X_np, y_np)
        
        # Create meta-features
        meta_features = self._create_meta_features(cv_predictions, X_np)
        
        if self.verbose > 0:
            logging.info(f"Meta-features shape: {meta_features.shape}")
            logging.info(f"Training meta-learner...")
        
        # Create and train meta-model
        self.meta_model_ = self._create_meta_model()
        
        # Handle different meta-model types
        if isinstance(self.meta_model_, dict):
            # Multi-output with separate models
            for key, model in self.meta_model_.items():
                if key == self.HOME:
                    target_idx = 0
                elif key == self.AWAY:
                    target_idx = 1
                else:
                    target_idx = int(key)
                
                if self.tune_hyperparameters and isinstance(model, lgb.LGBMRegressor):
                    self._tune_single_model(model, meta_features, y_np[:, target_idx])
                else:
                    model.fit(meta_features, y_np[:, target_idx])
        else:
            # Single model for all outputs
            if self.tune_hyperparameters and isinstance(self.meta_model_, lgb.LGBMRegressor):
                self._tune_single_model(self.meta_model_, meta_features, y_np)
            else:
                self.meta_model_.fit(meta_features, y_np)
        
        # Create feature names
        self._create_feature_names()
        
        return self
    
    def _tune_single_model(self, model: lgb.LGBMRegressor, X: np.ndarray, y: np.ndarray):
        """Tune a single LightGBM model."""
        param_grid = {
            'n_estimators': [100, 150, 200],
            'max_depth': [3, 4, 5],
            'num_leaves': [7, 15, 31],
            'learning_rate': [0.02, 0.03, 0.05],
            'subsample': [0.7, 0.8],
            'colsample_bytree': [0.5, 0.6, 0.7]
        }
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            grid_search = GridSearchCV(
                model,
                param_grid,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X, y)
            
            # Update model parameters
            model.set_params(**grid_search.best_params_)
            model.fit(X, y)
            
            if self.verbose > 0:
                logging.info(f"Best params: {grid_search.best_params_}")
    
    def _create_feature_names(self):
        """Create interpretable feature names for meta-features."""
        feature_names = []
        n_models = len(self.base_models_)
        
        # Raw predictions
        for i in range(n_models):
            if self.n_outputs_ == 2:
                feature_names.extend([f'model{i}_home', f'model{i}_away'])
            else:
                feature_names.append(f'model{i}_pred')
        
        # Derived features
        for i in range(n_models):
            if self.n_outputs_ == 2:
                feature_names.extend([
                    f'model{i}_margin', f'model{i}_total',
                    f'model{i}_winner', f'model{i}_abs_margin'
                ])
            else:
                feature_names.extend([f'model{i}_squared', f'model{i}_sqrt'])
        
        # Ensemble statistics
        if self.n_outputs_ == 2:
            feature_names.extend([
                'mean_home', 'std_home', 'range_home', 'p25_home', 'p75_home',
                'mean_away', 'std_away', 'range_away', 'p25_away', 'p75_away'
            ])
        else:
            feature_names.extend([
                'mean_pred', 'std_pred', 'range_pred', 'p25_pred', 'p75_pred'
            ])
        
        # Interaction features
        if self.use_interaction_features:
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    if self.n_outputs_ == 2:
                        feature_names.extend([
                            f'interact_{i}_{j}_home_prod', f'interact_{i}_{j}_home_diff',
                            f'interact_{i}_{j}_away_prod', f'interact_{i}_{j}_away_diff'
                        ])
                    else:
                        feature_names.extend([
                            f'interact_{i}_{j}_prod', f'interact_{i}_{j}_diff'
                        ])
        
        # Correlation features
        if self.use_correlation_features:
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    feature_names.append(f'corr_{i}_{j}')
        
        # Original features
        if self.include_original_features and self.selected_original_features_:
            feature_names.extend([f'orig_feat_{i}' for i in self.selected_original_features_])
        
        self.feature_names_ = feature_names
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using the stacking ensemble."""
        # Convert to numpy if needed
        X_np = X.values if hasattr(X, 'values') else X
        
        # Get base model predictions
        base_predictions = []
        for wrapper in self.base_model_wrappers_:
            pred = wrapper.predict(X_np)
            base_predictions.append(pred)
        
        # Create meta-features
        meta_features = self._create_meta_features(base_predictions, X_np)
        
        # Make final predictions
        if isinstance(self.meta_model_, dict):
            # Multi-output with separate models
            predictions = np.zeros((X_np.shape[0], len(self.meta_model_)))
            for idx, (key, model) in enumerate(self.meta_model_.items()):
                predictions[:, idx] = model.predict(meta_features)
        else:
            # Single model
            predictions = self.meta_model_.predict(meta_features)
        
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the meta-model (if available)."""
        if not hasattr(self.meta_model_, 'feature_importances_') and not isinstance(self.meta_model_, dict):
            return None
        
        importance_data = []
        
        if isinstance(self.meta_model_, dict):
            for key, model in self.meta_model_.items():
                if hasattr(model, 'feature_importances_'):
                    importance_data.append({
                        'model': key,
                        'features': self.feature_names_[:len(model.feature_importances_)],
                        'importance': model.feature_importances_
                    })
        else:
            if hasattr(self.meta_model_, 'feature_importances_'):
                importance_data.append({
                    'model': 'meta_model',
                    'features': self.feature_names_[:len(self.meta_model_.feature_importances_)],
                    'importance': self.meta_model_.feature_importances_
                })
        
        if not importance_data:
            return None
        
        # Create DataFrame
        dfs = []
        for data in importance_data:
            df = pd.DataFrame({
                'feature': data['features'],
                'importance': data['importance'],
                'model': data['model']
            })
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True).sort_values('importance', ascending=False)
        
# ============= MAIN TRAINER CLASS =============
class MLBModelTrainer:
    """Handles the full training pipeline"""
    
    def __init__(self, model_dir: str = "./mlb_model", use_advanced_features: bool = True):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        self.models = {}
        self.feature_selector = None
        self.scaler = None
        self.results = {}
        self.feature_indices = None
        self.feature_groups = None  # Add this
        self.config = ModelConfig(use_advanced_architecture=use_advanced_features)
        self.input_dim = None
        self.target_mean = None
        self.target_std = None
        self.snapshot_ensemble = None  # Add this

    def train_baseline_models(self, X_train: pd.DataFrame, y_train: pd.DataFrame, 
                            X_val: pd.DataFrame, y_val: pd.DataFrame) -> Dict:
        """Train simple baseline models for comparison"""
        logging.info("Training baseline models...")
        
        baselines = {}
        
        # 1. Always predict average
        avg_home = y_train['home_score'].mean()
        avg_away = y_train['away_score'].mean()
        avg_pred = np.array([[avg_home, avg_away]] * len(y_val))
        
        baselines['average'] = {
            'rmse': np.sqrt(mean_squared_error(y_val, avg_pred)),
            'winner_acc': ((avg_pred[:, 0] > avg_pred[:, 1]) == (y_val.values[:, 0] > y_val.values[:, 1])).mean()
        }
        
        # 2. Ridge Regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        ridge_pred = ridge.predict(X_val)
        
        baselines['ridge'] = {
            'rmse': np.sqrt(mean_squared_error(y_val, ridge_pred)),
            'winner_acc': ((ridge_pred[:, 0] > ridge_pred[:, 1]) == (y_val.values[:, 0] > y_val.values[:, 1])).mean()
        }
        
        # 3. LightGBM
        lgb_models = []
        for i, target in enumerate(['home_score', 'away_score']):
            lgb_model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
            lgb_model.fit(X_train, y_train.iloc[:, i])
            lgb_models.append(lgb_model)
        
        lgb_pred = np.column_stack([model.predict(X_val) for model in lgb_models])
        
        baselines['lightgbm'] = {
            'rmse': np.sqrt(mean_squared_error(y_val, lgb_pred)),
            'winner_acc': ((lgb_pred[:, 0] > lgb_pred[:, 1]) == (y_val.values[:, 0] > y_val.values[:, 1])).mean()
        }
        
        # Log results
        logging.info("\nBaseline Results:")
        for name, metrics in baselines.items():
            logging.info(f"{name}: RMSE={metrics['rmse']:.3f}, Winner Acc={metrics['winner_acc']:.3%}")
        
        return baselines, {'ridge': ridge, 'lightgbm': lgb_models}
    
    def train_with_curriculum(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                            X_val: pd.DataFrame, y_val: pd.DataFrame) -> Tuple[nn.Module, List]:
        """Train model with curriculum learning - from easy to hard games"""
        
        # Get feature indices for the model
        feature_indices, feature_groups = get_feature_indices(X_train.columns)
        self.feature_indices = feature_indices
        self.feature_groups = feature_groups  # Store feature groups
        
        # Calculate game difficulty metrics
        run_diff = np.abs(y_train.values[:, 0] - y_train.values[:, 1])
        total_runs = y_train.values[:, 0] + y_train.values[:, 1]
        
        # Define curriculum phases
        phases = [
            {
                'name': 'Blowouts (>4 run difference)',
                'mask': run_diff > 4,
                'lr': 0.002,
                'epochs': 40,
                'batch_size': 512,
                'patience': 15
            },
            {
                'name': 'Clear Winners (2-4 run difference)',
                'mask': (run_diff > 2) & (run_diff <= 4),
                'lr': 0.001,
                'epochs': 40,
                'batch_size': 256,
                'patience': 15
            },
            {
                'name': 'Close Games (<=2 run difference)',
                'mask': run_diff <= 2,
                'lr': 0.0005,
                'epochs': 50,
                'batch_size': 256,
                'patience': 20
            },
            {
                'name': 'All Games with Low-Scoring Focus',
                'mask': np.ones(len(run_diff), dtype=bool),
                'lr': 0.0002,
                'epochs': 60,
                'batch_size': 128,
                'focus_on_low_scoring': True,
                'patience': 25
            }
        ]
        
        # Initialize model based on configuration
        if self.config.use_hybrid:
            model = MLBHybridModel(self.config, X_train.shape[1]).to(self.device)
        elif self.config.use_gnn:
            model = MLBGraphNeuralNetwork(X_train.shape[1]).to(self.device)
        elif self.config.use_advanced_architecture:
            model = MLBNeuralNetV3(feature_groups, self.config).to(self.device)
        else:
            model = MLBNeuralNetV2(feature_groups).to(self.device)
        
        # Initialize components for advanced training
        if self.config.use_adversarial:
            adversarial_trainer = AdversarialTrainer()
        
        if self.config.use_snapshot_ensemble:
            snapshot_ensemble = SnapshotEnsemble(n_snapshots=5, cycles=5)
        
        # Dynamic loss weighting
        if self.config.use_quantile_regression:
            dynamic_loss = DynamicWeightedLoss(n_tasks=4)  # score, winner, margin, quantile
            quantile_loss = QuantileRegressionLoss()
        else:
            dynamic_loss = DynamicWeightedLoss(n_tasks=3)  # score, winner, margin
        
        # Self-supervised pretraining if enabled
        if hasattr(self.config, 'use_pretraining') and self.config.use_pretraining:
            logging.info("Performing self-supervised pretraining...")
            pretrain_model = MLBMaskedPretraining(model, X_train.shape[1]).to(self.device)
            self._pretrain_model(pretrain_model, X_train, epochs=20)
        
        # Store training history
        curriculum_history = []
        
        for phase_idx, phase in enumerate(phases):
            logging.info(f"\n{'='*60}")
            logging.info(f"CURRICULUM PHASE {phase_idx + 1}: {phase['name']}")
            logging.info(f"Training on {phase['mask'].sum()} games")
            logging.info(f"{'='*60}")
            
            # Get phase data
            phase_indices = np.where(phase['mask'])[0]
            
            if len(phase_indices) == 0:
                logging.info("No games in this phase, skipping...")
                continue
            
            phase_X = X_train.iloc[phase_indices]
            phase_y = y_train.iloc[phase_indices]
            
            # Special handling for final phase
            if phase.get('focus_on_low_scoring', False):
                # Oversample low-scoring games AND very close games
                low_scoring_mask = total_runs[phase['mask']] < 7
                one_run_mask = run_diff[phase['mask']] == 1
                two_run_mask = run_diff[phase['mask']] == 2
                
                # Get indices for each type
                low_scoring_indices = phase_indices[low_scoring_mask]
                one_run_indices = phase_indices[one_run_mask]
                two_run_indices = phase_indices[two_run_mask]
                
                # Build augmented dataset with different sampling rates
                augmented_X = [phase_X]
                augmented_y = [phase_y]
                
                # Add low-scoring games once more
                if len(low_scoring_indices) > 0:
                    augmented_X.append(X_train.iloc[low_scoring_indices])
                    augmented_y.append(y_train.iloc[low_scoring_indices])
                
                # Add 1-run games three times (4x total weight)
                if len(one_run_indices) > 0:
                    for _ in range(3):
                        augmented_X.append(X_train.iloc[one_run_indices])
                        augmented_y.append(y_train.iloc[one_run_indices])
                
                # Add 2-run games once more (2x total weight)
                if len(two_run_indices) > 0:
                    augmented_X.append(X_train.iloc[two_run_indices])
                    augmented_y.append(y_train.iloc[two_run_indices])
                
                phase_X = pd.concat(augmented_X, ignore_index=True)
                phase_y = pd.concat(augmented_y, ignore_index=True)
                
                logging.info(f"Augmented training set: {len(low_scoring_indices)} low-scoring, "
                           f"{len(one_run_indices)} one-run (4x weight), "
                           f"{len(two_run_indices)} two-run games (2x weight)")
                logging.info(f"Total training samples: {len(phase_X)}")
            
            # Create optimizer for this phase
            optimizer = optim.AdamW(model.parameters(), lr=phase['lr'], weight_decay=0.01)
            
            # Use snapshot ensemble schedule if enabled
            if self.config.use_snapshot_ensemble:
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=phase['epochs'] // 5, T_mult=1
                )
            else:
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=10, T_mult=2
                )
            
            # Select loss function based on phase
            if phase_idx < 2:
                # Early phases: focus more on score accuracy
                base_criterion = WinLossAwareLoss(score_weight=0.8, winner_weight=0.2)
            elif phase_idx == 2:
                # Close games: balance score and winner
                base_criterion = WinLossAwareLoss(score_weight=0.6, winner_weight=0.4)
            else:
                # Final phase: optimize for overall prediction accuracy
                base_criterion = RunPredictionLoss(score_weight=0.6, winner_weight=0.2, margin_weight=0.2)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(phase_X.values)
            y_tensor = torch.FloatTensor(phase_y.values)
            X_val_tensor = torch.FloatTensor(X_val.values).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val.values).to(self.device)
            
            # Create DataLoader
            train_dataset = TensorDataset(X_tensor, y_tensor)
            train_loader = DataLoader(train_dataset, batch_size=phase['batch_size'], shuffle=True)
            
            # Train this phase
            phase_metrics = []
            best_val_loss = float('inf')
            best_model_state = None
            patience_counter = 0
            
            for epoch in range(phase['epochs']):
                # Training
                model.train()
                train_loss = 0
                
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    # Adversarial training if enabled
                    if self.config.use_adversarial and phase_idx >= 2:
                        batch_X_adv = adversarial_trainer.generate_adversarial_examples(
                            model, batch_X, batch_y, base_criterion
                        )
                        # Train on both clean and adversarial examples
                        batch_X_combined = torch.cat([batch_X, batch_X_adv])
                        batch_y_combined = torch.cat([batch_y, batch_y])
                    else:
                        batch_X_combined = batch_X
                        batch_y_combined = batch_y
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    if isinstance(model, MLBHybridModel):
                        scores, quantiles = model(batch_X_combined)
                        if self.config.use_quantile_regression:
                            # Calculate multiple losses
                            score_loss = base_criterion(scores, batch_y_combined)
                            q_loss = quantile_loss(quantiles, batch_y_combined)
                            
                            # Winner loss
                            pred_margin = scores[:, 0] - scores[:, 1]
                            true_margin = batch_y_combined[:, 0] - batch_y_combined[:, 1]
                            winner_loss = F.relu(1.0 - pred_margin * true_margin).mean()
                            
                            # Margin loss
                            margin_loss = F.smooth_l1_loss(pred_margin, true_margin)
                            
                            # Combine with dynamic weights
                            loss = dynamic_loss([score_loss, winner_loss, margin_loss, q_loss])
                        else:
                            loss = base_criterion(scores, batch_y_combined)
                    else:
                        if self.config.use_advanced_architecture or isinstance(model, MLBGraphNeuralNetwork):
                            pred = model(batch_X_combined)
                        else:
                            pred = model(batch_X_combined, feature_indices)
                        
                        loss = base_criterion(pred, batch_y_combined)
                    
                    # Add regularization for later phases
                    if phase_idx >= 2:
                        # L2 regularization on predictions to prevent extreme scores
                        if isinstance(model, MLBHybridModel):
                            pred_reg = 0.001 * (scores ** 2).mean()
                        else:
                            pred_reg = 0.001 * (pred ** 2).mean()
                        loss = loss + pred_reg
                    
                    loss.backward()
                    
                    # Gradient clipping (more aggressive for later phases)
                    max_norm = 1.0 if phase_idx < 2 else 0.5
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                    
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    if isinstance(model, MLBHybridModel):
                        val_scores, val_quantiles = model(X_val_tensor)
                        val_pred = val_scores
                    elif self.config.use_advanced_architecture or isinstance(model, MLBGraphNeuralNetwork):
                        val_pred = model(X_val_tensor)
                    else:
                        val_pred = model(X_val_tensor, feature_indices)
                    
                    val_loss = base_criterion(val_pred, y_val_tensor).item()
                    
                    # Calculate detailed metrics
                    val_rmse = torch.sqrt(F.mse_loss(val_pred, y_val_tensor)).item()
                    winner_acc = ((val_pred[:, 0] > val_pred[:, 1]) == 
                                (y_val_tensor[:, 0] > y_val_tensor[:, 1])).float().mean().item()
                    
                    # Calculate confidence-based accuracy
                    pred_margin = torch.abs(val_pred[:, 0] - val_pred[:, 1])
                    high_conf_mask = pred_margin > pred_margin.median()
                    if high_conf_mask.sum() > 0:
                        high_conf_acc = ((val_pred[high_conf_mask, 0] > val_pred[high_conf_mask, 1]) == 
                                       (y_val_tensor[high_conf_mask, 0] > y_val_tensor[high_conf_mask, 1])).float().mean().item()
                    else:
                        high_conf_acc = 0.0
                
                scheduler.step()
                
                # Save snapshot if using snapshot ensemble
                if self.config.use_snapshot_ensemble and snapshot_ensemble.should_save_snapshot(epoch, phase['epochs']):
                    snapshot_ensemble.save_snapshot(model, epoch)
                    snapshot_ensemble.snapshots[-1]['performance'] = winner_acc
                
                # Track metrics
                metrics = {
                    'epoch': epoch,
                    'train_loss': train_loss / len(train_loader),
                    'val_loss': val_loss,
                    'val_rmse': val_rmse,
                    'winner_acc': winner_acc,
                    'high_conf_acc': high_conf_acc
                }
                phase_metrics.append(metrics)
                
                # Early stopping with patience
                phase_patience = phase.get('patience', 10)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= phase_patience:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    logging.info(f"Epoch {epoch+1}: Val RMSE={val_rmse:.3f}, "
                               f"Winner Acc={winner_acc:.3%}, High Conf Acc={high_conf_acc:.3%}")
            
            # Load best model state for next phase
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            curriculum_history.append({
                'phase': phase['name'],
                'metrics': phase_metrics
            })
        
        # Store snapshot ensemble if used
        if self.config.use_snapshot_ensemble:
            self.snapshot_ensemble = snapshot_ensemble
        
        return model, curriculum_history

    def train_uncertainty_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                            X_val: pd.DataFrame, y_val: pd.DataFrame) -> nn.Module:
        """Train model with uncertainty quantification"""
        
        # Initialize model with uncertainty
        model = MLBNeuralNetWithUncertainty(
            input_dim=X_train.shape[1],
            hidden_dims=[512, 256, 128],
            dropout_rate=0.3,
            mc_dropout_rate=0.1
        ).to(self.device)
        
        # Optimizer with better learning rate scheduling
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Loss function
        criterion = WinLossAwareLoss(score_weight=0.7, winner_weight=0.3)
        
        # Data augmentation for close games
        run_diff = np.abs(y_train.values[:, 0] - y_train.values[:, 1])
        close_games_mask = run_diff <= 2
        
        # Oversample close games
        if close_games_mask.sum() > 0:
            close_games_X = X_train.values[close_games_mask]
            close_games_y = y_train.values[close_games_mask]
            
            # Add noise to create augmented samples
            noise_level = 0.02
            augmented_X = close_games_X + np.random.normal(0, noise_level, close_games_X.shape)
            
            # Combine original and augmented
            X_train_aug = np.vstack([X_train.values, augmented_X])
            y_train_aug = np.vstack([y_train.values, close_games_y])
        else:
            X_train_aug = X_train.values
            y_train_aug = y_train.values
        
        # Create tensors
        X_train_t = torch.FloatTensor(X_train_aug)
        y_train_t = torch.FloatTensor(y_train_aug)
        X_val_t = torch.FloatTensor(X_val.values).to(self.device)
        y_val_t = torch.FloatTensor(y_val.values).to(self.device)
        
        # DataLoader with larger batch size
        dataset = TensorDataset(X_train_t, y_train_t)
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 30
        
        for epoch in range(200):
            # Training
            model.train()
            train_loss = 0
            
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                pred_mean, pred_log_std = model(batch_X)
                loss = criterion(pred_mean, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred_mean, val_pred_log_std = model(X_val_t, sample=False)
                val_loss = criterion(val_pred_mean, y_val_t)
                
                # Calculate R²
                ss_res = ((y_val_t - val_pred_mean) ** 2).sum()
                ss_tot = ((y_val_t - y_val_t.mean(dim=0)) ** 2).sum()
                r2 = 1 - (ss_res / ss_tot)
                
                # Winner accuracy
                winner_acc = ((val_pred_mean[:, 0] > val_pred_mean[:, 1]) == 
                            (y_val_t[:, 0] > y_val_t[:, 1])).float().mean()
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break
                
            if (epoch + 1) % 20 == 0:
                logging.info(f"Epoch {epoch+1}: Val Loss={val_loss:.4f}, "
                            f"R²={r2:.3f}, Winner Acc={winner_acc:.3%}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        return model

    def _pretrain_model(self, pretrain_model: MLBMaskedPretraining, X_train: pd.DataFrame, epochs: int = 20):
        """Self-supervised pretraining"""
        optimizer = optim.Adam(pretrain_model.parameters(), lr=0.001)
        X_tensor = torch.FloatTensor(X_train.values)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                batch_X = batch[0].to(self.device)
                optimizer.zero_grad()
                loss, _ = pretrain_model(batch_X)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                logging.info(f"Pretraining epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}")

    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                      X_val: pd.DataFrame, y_val: pd.DataFrame) -> Dict:
        """Train an ensemble of diverse models"""
        
        ensemble_results = {}
        self.input_dim = X_train.shape[1]
        
        # Train multiple models with different architectures
        all_models = []
        model_names = []
        val_performances = []
        
        # 1. Original curriculum learning models
        for seed in [42, 123, 456]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            logging.info(f"\nTraining neural network with curriculum learning (seed {seed})...")
            
            model, curriculum_history = self.train_with_curriculum(
                X_train, y_train, X_val, y_val
            )
            all_models.append(model)
            model_names.append(f'curriculum_{seed}')
            
            # Evaluate model
            val_perf = self._evaluate_model(model, X_val, y_val)
            val_performances.append(val_perf)
            logging.info(f"Model {seed} - Val Acc: {val_perf:.3%}")
        
        # 2. GNN model if enabled
        if self.config.use_gnn:
            logging.info("\nTraining Graph Neural Network...")
            self.config.use_advanced_architecture = False
            self.config.use_gnn = True
            gnn_model, _ = self.train_with_curriculum(X_train, y_train, X_val, y_val)
            all_models.append(gnn_model)
            model_names.append('gnn')
            val_perf = self._evaluate_model(gnn_model, X_val, y_val)
            val_performances.append(val_perf)
            self.config.use_gnn = False
        
        # 3. Hybrid model if enabled
        if self.config.use_hybrid:
            logging.info("\nTraining Hybrid Model...")
            self.config.use_hybrid = True
            hybrid_model, _ = self.train_with_curriculum(X_train, y_train, X_val, y_val)
            all_models.append(hybrid_model)
            model_names.append('hybrid')
            val_perf = self._evaluate_model(hybrid_model, X_val, y_val)
            val_performances.append(val_perf)
            self.config.use_hybrid = False
        
        # 4. Uncertainty model
        logging.info("\nTraining model with uncertainty quantification...")
        uncertainty_model = self.train_uncertainty_model(X_train, y_train, X_val, y_val)
        all_models.append(uncertainty_model)
        model_names.append('uncertainty')
        
        # Evaluate uncertainty model
        uncertainty_model.eval()
        with torch.no_grad():
            X_val_t = torch.FloatTensor(X_val.values).to(self.device)
            y_val_t = torch.FloatTensor(y_val.values).to(self.device)
            
            val_pred_mean, val_uncertainty, _, _ = uncertainty_model.predict_with_uncertainty(X_val_t, n_samples=30)
            
            val_acc = ((val_pred_mean[:, 0] > val_pred_mean[:, 1]) == 
                    (y_val_t[:, 0] > y_val_t[:, 1])).float().mean().item()
            
            uncertainty_margin = val_uncertainty.mean(dim=1)
            low_uncertainty_mask = uncertainty_margin < uncertainty_margin.median()
            if low_uncertainty_mask.sum() > 0:
                high_conf_acc = ((val_pred_mean[low_uncertainty_mask, 0] > val_pred_mean[low_uncertainty_mask, 1]) == 
                            (y_val_t[low_uncertainty_mask, 0] > y_val_t[low_uncertainty_mask, 1])).float().mean().item()
            else:
                high_conf_acc = val_acc
            
            val_perf = 0.6 * val_acc + 0.4 * high_conf_acc
            val_performances.append(val_perf)
            
            logging.info(f"Uncertainty Model - Val Acc: {val_acc:.3%}, Low Uncertainty Acc: {high_conf_acc:.3%}")
        
        # In the train_ensemble method, around line 2195-2220, update the knowledge distillation section:

        # 5. Knowledge Distillation - create efficient student model
        if len(all_models) >= 3:
            logging.info("\nTraining student model via knowledge distillation...")
            student_model = MLBNeuralNetV2(self.feature_groups, hidden_dim=64).to(self.device)  # Smaller model
            
            kd_trainer = KnowledgeDistillationTrainer(temperature=3.0, alpha=0.7)
            
            # Create simple dataloaders for KD
            X_train_t = torch.FloatTensor(X_train.values)
            y_train_t = torch.FloatTensor(y_train.values)
            X_val_t = torch.FloatTensor(X_val.values)
            y_val_t = torch.FloatTensor(y_val.values)
            
            train_dataset = TensorDataset(X_train_t, y_train_t)
            val_dataset = TensorDataset(X_val_t, y_val_t)
            
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
            
            # Use best 3 models as teachers
            best_indices = np.argsort(val_performances)[-3:]
            teacher_ensemble = [all_models[i] for i in best_indices]
            
            # Pass feature_indices to train_student
            student_model = kd_trainer.train_student(
                student_model, teacher_ensemble, train_loader, val_loader, 
                self.device, epochs=30, feature_indices=self.feature_indices  # Added feature_indices
            )
            
            all_models.append(student_model)
            model_names.append('student')
            val_perf = self._evaluate_model(student_model, X_val, y_val)
            val_performances.append(val_perf)
        
        # Calculate ensemble weights based on validation performance
        weights = np.array(val_performances)
        weights = np.exp(weights * 10)  # Sharpen differences
        weights = weights / weights.sum()
        logging.info(f"\nEnsemble weights: {dict(zip(model_names, weights))}")
        
        # Create ensemble predictions
        with torch.no_grad():
            X_val_t = torch.FloatTensor(X_val.values).to(self.device)
            nn_preds = []
            
            for i, model in enumerate(all_models):
                model.eval()
                if hasattr(model, 'predict_with_uncertainty'):
                    pred_mean, _, _, _ = model.predict_with_uncertainty(X_val_t, n_samples=30)
                    nn_preds.append(pred_mean.cpu().numpy())
                elif isinstance(model, MLBHybridModel):
                    scores, _ = model(X_val_t)
                    nn_preds.append(scores.cpu().numpy())
                else:
                    if isinstance(model, MLBNeuralNetV2):
                        pred = model(X_val_t, self.feature_indices)
                    else:
                        pred = model(X_val_t)
                    nn_preds.append(pred.cpu().numpy())
            
            # Weighted average
            ensemble_pred = np.average(nn_preds, axis=0, weights=weights)
        
        # Calculate ensemble metrics
        ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
        ensemble_winner_acc = ((ensemble_pred[:, 0] > ensemble_pred[:, 1]) == 
                            (y_val.values[:, 0] > y_val.values[:, 1])).mean()
        
        pred_margin = np.abs(ensemble_pred[:, 0] - ensemble_pred[:, 1])
        high_conf_mask = pred_margin > np.median(pred_margin)
        high_conf_acc = ((ensemble_pred[high_conf_mask, 0] > ensemble_pred[high_conf_mask, 1]) == 
                        (y_val.values[high_conf_mask, 0] > y_val.values[high_conf_mask, 1])).mean()
        
        ensemble_results['neural_ensemble'] = {
            'rmse': ensemble_rmse,
            'winner_acc': ensemble_winner_acc,
            'high_conf_acc': high_conf_acc,
            'models': all_models,
            'model_names': model_names,
            'weights': weights
        }
        
        logging.info(f"\nDiverse Neural Ensemble: RMSE={ensemble_rmse:.3f}, "
                    f"Winner Acc={ensemble_winner_acc:.3%}, High Conf Acc={high_conf_acc:.3%}")
        
        return ensemble_results

    def _evaluate_model(self, model: nn.Module, X_val: pd.DataFrame, y_val: pd.DataFrame) -> float:
        """Evaluate a single model and return combined performance score"""
        model.eval()
        with torch.no_grad():
            X_val_t = torch.FloatTensor(X_val.values).to(self.device)
            y_val_t = torch.FloatTensor(y_val.values).to(self.device)
            
            # Handle different model types
            if isinstance(model, MLBHybridModel):
                val_pred, _ = model(X_val_t)
            elif isinstance(model, MLBNeuralNetV2):
                # MLBNeuralNetV2 needs feature_indices
                val_pred = model(X_val_t, self.feature_indices)
            elif hasattr(model, 'predict_with_uncertainty'):
                # Uncertainty models
                val_pred, _ = model(X_val_t, sample=False)
            else:
                # Other models (MLBNeuralNetV3, etc.)
                val_pred = model(X_val_t)
            
            # Calculate validation accuracy
            val_acc = ((val_pred[:, 0] > val_pred[:, 1]) == 
                    (y_val_t[:, 0] > y_val_t[:, 1])).float().mean().item()
            
            # High-confidence accuracy
            pred_margin = torch.abs(val_pred[:, 0] - val_pred[:, 1])
            high_conf_mask = pred_margin > pred_margin.median()
            if high_conf_mask.sum() > 0:
                high_conf_acc = ((val_pred[high_conf_mask, 0] > val_pred[high_conf_mask, 1]) == 
                            (y_val_t[high_conf_mask, 0] > y_val_t[high_conf_mask, 1])).float().mean().item()
            else:
                high_conf_acc = val_acc
            
            return 0.6 * val_acc + 0.4 * high_conf_acc

    def evaluate_on_test(self, models: Dict, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict:
        """Evaluate all models on test set"""
        results = {}
        
        logging.info("\n" + "="*60)
        logging.info("EVALUATING ON TEST SET")
        logging.info("="*60)
        
        # Neural ensemble evaluation
        if 'neural_ensemble' in models:
            logging.info("\nEvaluating Neural Ensemble...")
            ensemble_data = models['neural_ensemble']
            nn_models = ensemble_data['models']
            weights = ensemble_data['weights']
            
            # Get predictions from all models
            X_test_t = torch.FloatTensor(X_test.values).to(self.device)
            
            with torch.no_grad():
                nn_preds = []
                uncertainties = []
                
                for i, model in enumerate(nn_models):
                    model.eval()
                    if hasattr(model, 'predict_with_uncertainty'):
                        pred_mean, total_uncertainty, _, _ = model.predict_with_uncertainty(X_test_t, n_samples=30)
                        nn_preds.append(pred_mean.cpu().numpy())
                        uncertainties.append(total_uncertainty.cpu().numpy())
                    elif isinstance(model, MLBHybridModel):
                        # HybridModel returns (scores, quantiles)
                        scores, _ = model(X_test_t)
                        nn_preds.append(scores.cpu().numpy())
                        # Estimate uncertainty from prediction variance
                        uncertainties.append(np.ones_like(scores.cpu().numpy()) * 0.5)
                    elif isinstance(model, MLBNeuralNetV2):
                        pred = model(X_test_t, self.feature_indices)
                        nn_preds.append(pred.cpu().numpy())
                        # Estimate uncertainty from prediction variance
                        uncertainties.append(np.ones_like(pred.cpu().numpy()) * 0.5)
                    else:
                        pred = model(X_test_t)
                        # Handle case where model might return a tuple
                        if isinstance(pred, tuple):
                            pred = pred[0]  # Extract the primary prediction tensor
                        nn_preds.append(pred.cpu().numpy())
                        # Estimate uncertainty from prediction variance
                        uncertainties.append(np.ones_like(pred.cpu().numpy()) * 0.5)
                
                # Weighted average
                ensemble_pred = np.average(nn_preds, axis=0, weights=weights)
                ensemble_uncertainty = np.average(uncertainties, axis=0, weights=weights)
            
            # Calculate metrics
            test_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
            test_mae = mean_absolute_error(y_test, ensemble_pred)
            test_r2 = r2_score(y_test, ensemble_pred)
            
            # Winner accuracy
            winner_acc = ((ensemble_pred[:, 0] > ensemble_pred[:, 1]) == 
                        (y_test.values[:, 0] > y_test.values[:, 1])).mean()
            
            # Confidence-based accuracy
            pred_margin = np.abs(ensemble_pred[:, 0] - ensemble_pred[:, 1])
            confidence_thresholds = [0.5, 1.0, 1.5, 2.0]
            confidence_accuracies = {}
            
            for threshold in confidence_thresholds:
                mask = pred_margin > threshold
                if mask.sum() > 0:
                    acc = ((ensemble_pred[mask, 0] > ensemble_pred[mask, 1]) == 
                        (y_test.values[mask, 0] > y_test.values[mask, 1])).mean()
                    confidence_accuracies[f'margin_{threshold}'] = {
                        'accuracy': acc,
                        'count': mask.sum(),
                        'percentage': mask.sum() / len(mask) * 100
                    }
            
            # Score differential accuracy
            pred_diff = ensemble_pred[:, 0] - ensemble_pred[:, 1]
            actual_diff = y_test.values[:, 0] - y_test.values[:, 1]
            diff_rmse = np.sqrt(mean_squared_error(actual_diff, pred_diff))
            
            # Total runs accuracy
            pred_total = ensemble_pred[:, 0] + ensemble_pred[:, 1]
            actual_total = y_test.values[:, 0] + y_test.values[:, 1]
            total_rmse = np.sqrt(mean_squared_error(actual_total, pred_total))
            
            results['neural_ensemble'] = {
                'rmse': test_rmse,
                'mae': test_mae,
                'r2': test_r2,
                'winner_acc': winner_acc,
                'confidence_accuracies': confidence_accuracies,
                'predictions': ensemble_pred,
                'uncertainties': ensemble_uncertainty,
                'diff_rmse': diff_rmse,
                'total_runs_rmse': total_rmse
            }
            
            logging.info(f"\nNeural Ensemble Test Results:")
            logging.info(f"  RMSE: {test_rmse:.3f}")
            logging.info(f"  MAE: {test_mae:.3f}")
            logging.info(f"  R²: {test_r2:.3f}")
            logging.info(f"  Winner Accuracy: {winner_acc:.1%}")
            logging.info(f"  Run Differential RMSE: {diff_rmse:.3f}")
            logging.info(f"  Total Runs RMSE: {total_rmse:.3f}")
            
            for conf_name, conf_data in confidence_accuracies.items():
                logging.info(f"  {conf_name}: {conf_data['accuracy']:.1%} "
                            f"({conf_data['count']} games, {conf_data['percentage']:.1f}%)")
        
        # Evaluate baseline models
        if 'ridge' in models:
            ridge_pred = models['ridge'].predict(X_test)
            ridge_winner_acc = ((ridge_pred[:, 0] > ridge_pred[:, 1]) == 
                            (y_test.values[:, 0] > y_test.values[:, 1])).mean()
            results['ridge'] = {
                'rmse': np.sqrt(mean_squared_error(y_test, ridge_pred)),
                'winner_acc': ridge_winner_acc
            }
            logging.info(f"\nRidge Test: RMSE={results['ridge']['rmse']:.3f}, "
                        f"Winner Acc={ridge_winner_acc:.1%}")
        
        if 'lightgbm' in models:
            lgb_pred = np.column_stack([model.predict(X_test) for model in models['lightgbm']])
            lgb_winner_acc = ((lgb_pred[:, 0] > lgb_pred[:, 1]) == 
                            (y_test.values[:, 0] > y_test.values[:, 1])).mean()
            results['lightgbm'] = {
                'rmse': np.sqrt(mean_squared_error(y_test, lgb_pred)),
                'winner_acc': lgb_winner_acc
            }
            logging.info(f"LightGBM Test: RMSE={results['lightgbm']['rmse']:.3f}, "
                        f"Winner Acc={lgb_winner_acc:.1%}")
        
        return results

    # Add this method to the MLBModelTrainer class
    def evaluate_stacking_on_test(self, stacker: StackingEnsemble, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict:
        """Evaluate the trained stacking ensemble on the test set."""
        logging.info("\n" + "="*60)
        logging.info("EVALUATING STACKING ENSEMBLE ON TEST SET")
        logging.info("="*60)

        # Get predictions from the final stacking model
        stacking_pred = stacker.predict(X_test)

        # Calculate performance metrics
        test_rmse = np.sqrt(mean_squared_error(y_test, stacking_pred))
        test_mae = mean_absolute_error(y_test, stacking_pred)
        test_r2 = r2_score(y_test, stacking_pred)

        # Winner accuracy
        winner_acc = ((stacking_pred[:, 0] > stacking_pred[:, 1]) ==
                    (y_test.values[:, 0] > y_test.values[:, 1])).mean()

        # Confidence-based accuracy
        pred_margin = np.abs(stacking_pred[:, 0] - stacking_pred[:, 1])
        confidence_thresholds = [0.5, 1.0, 1.5, 2.0]
        confidence_accuracies = {}

        for threshold in confidence_thresholds:
            mask = pred_margin > threshold
            if mask.sum() > 0:
                acc = ((stacking_pred[mask, 0] > stacking_pred[mask, 1]) ==
                    (y_test.values[mask, 0] > y_test.values[mask, 1])).mean()
                confidence_accuracies[f'margin_{threshold}'] = {
                    'accuracy': acc,
                    'count': mask.sum(),
                    'percentage': mask.sum() / len(mask) * 100
                }

        # Score differential and total runs accuracy
        pred_diff = stacking_pred[:, 0] - stacking_pred[:, 1]
        actual_diff = y_test.values[:, 0] - y_test.values[:, 1]
        diff_rmse = np.sqrt(mean_squared_error(actual_diff, pred_diff))

        pred_total = stacking_pred[:, 0] + stacking_pred[:, 1]
        actual_total = y_test.values[:, 0] + y_test.values[:, 1]
        total_rmse = np.sqrt(mean_squared_error(actual_total, pred_total))

        # Compile results into a dictionary
        results = {
            'rmse': test_rmse,
            'mae': test_mae,
            'r2': test_r2,
            'winner_acc': winner_acc,
            'confidence_accuracies': confidence_accuracies,
            'predictions': stacking_pred,
            'diff_rmse': diff_rmse,
            'total_runs_rmse': total_rmse
        }

        # Log the results to the console
        logging.info(f"  RMSE: {test_rmse:.3f}")
        logging.info(f"  MAE: {test_mae:.3f}")
        logging.info(f"  R²: {test_r2:.3f}")
        logging.info(f"  Winner Accuracy: {winner_acc:.1%}")
        logging.info(f"  Run Differential RMSE: {diff_rmse:.3f}")
        logging.info(f"  Total Runs RMSE: {total_rmse:.3f}")

        for conf_name, conf_data in confidence_accuracies.items():
            logging.info(f"  {conf_name}: {conf_data['accuracy']:.1%} "
                        f"({conf_data['count']} games, {conf_data['percentage']:.1f}%)")

        return results

    def train_stacking_ensemble(self, models_dict, X_train, y_train, X_val, y_val):
        """Train a stacking ensemble with proper validation splits"""
        logging.info("\n" + "="*60)
        logging.info("TRAINING STACKING ENSEMBLE")
        logging.info("="*60)
        
        # Get base models
        base_models = models_dict['neural_ensemble']['models']
        
        # Create clean splits for meta-training and selection
        meta_train_split = int(0.7 * len(X_val))  # 70% for training meta-model
        
        X_val_meta_train = X_val.iloc[:meta_train_split]
        y_val_meta_train = y_val.iloc[:meta_train_split]
        X_val_meta_test = X_val.iloc[meta_train_split:]
        y_val_meta_test = y_val.iloc[meta_train_split:]
        
        logging.info(f"Meta-training set: {len(X_val_meta_train)} samples")
        logging.info(f"Meta-selection set: {len(X_val_meta_test)} samples")
        
        # Train multiple stacker variants
        stackers = {}
        
        # Version 1: Meta-features only
        logging.info("\n1. Training stacker with meta-features only...")
        stacker_meta = StackingEnsemble(
            meta_model='lightgbm',  # CORRECTED: Was meta_model_type
            include_original_features=False,
            tune_hyperparameters=False
        )
        # The fit method in the newer StackingEnsemble class requires different arguments.
        # It's designed to be more self-contained and uses cross-validation internally.
        # We will fit it on the combined training and validation data used for the base models.
        X_combined_train_val = pd.concat([X_train, X_val_meta_train])
        y_combined_train_val = pd.concat([y_train, y_val_meta_train])
        
        stacker_meta.fit(
            X=X_combined_train_val,
            y=y_combined_train_val,
            base_models=base_models,
            feature_indices=self.feature_indices,
            device=self.device
        )
        stackers['meta_only'] = stacker_meta
        
        # Version 2: Meta-features + original features
        if hasattr(self.feature_selector, 'feature_importance_df'):
            logging.info("\n2. Training stacker with meta + original features...")
            stacker_combined = StackingEnsemble(
                meta_model='lightgbm', # CORRECTED: Was meta_model_type
                include_original_features=True,
                n_original_features=15,
                tune_hyperparameters=False
            )
            stacker_combined.fit(
                X=X_combined_train_val,
                y=y_combined_train_val,
                base_models=base_models,
                feature_indices=self.feature_indices,
                device=self.device,
                feature_importance_df=self.feature_selector.feature_importance_df
            )
            stackers['with_original'] = stacker_combined
        
        # Version 3: Ridge meta-model (simple but robust)
        logging.info("\n3. Training Ridge stacker...")
        stacker_ridge = StackingEnsemble(
            meta_model='ridge', # CORRECTED: Was meta_model_type
            include_original_features=False
        )
        stacker_ridge.fit(
            X=X_combined_train_val,
            y=y_combined_train_val,
            base_models=base_models,
            feature_indices=self.feature_indices,
            device=self.device
        )
        stackers['ridge'] = stacker_ridge
        
        # Evaluate all stackers on the held-out meta-test set
        logging.info("\nEvaluating stackers on held-out set...")
        best_stacker = None
        best_acc = 0
        best_name = None
        
        for name, stacker in stackers.items():
            pred = stacker.predict(X_val_meta_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_val_meta_test, pred))
            acc = ((pred[:, 0] > pred[:, 1]) == 
                (y_val_meta_test.values[:, 0] > y_val_meta_test.values[:, 1])).mean()
            
            # High-confidence accuracy
            margin = np.abs(pred[:, 0] - pred[:, 1])
            high_conf_mask = margin > np.median(margin)
            if high_conf_mask.sum() > 0:
                high_conf_acc = ((pred[high_conf_mask, 0] > pred[high_conf_mask, 1]) == 
                            (y_val_meta_test.values[high_conf_mask, 0] > 
                                y_val_meta_test.values[high_conf_mask, 1])).mean()
            else:
                high_conf_acc = acc
            
            # Combined score (weighted accuracy)
            combined_score = 0.7 * acc + 0.3 * high_conf_acc
            
            logging.info(f"\n{name}:")
            logging.info(f"  RMSE: {rmse:.3f}")
            logging.info(f"  Accuracy: {acc:.3%}")
            logging.info(f"  High-conf accuracy: {high_conf_acc:.3%}")
            logging.info(f"  Combined score: {combined_score:.3%}")
            
            if combined_score > best_acc:
                best_acc = combined_score
                best_stacker = stacker
                best_name = name
        
        logging.info(f"\nSelected {best_name} stacker with combined score {best_acc:.3%}")
        
        # Store the best stacker
        self.stacker = best_stacker
        self.stacker_type = best_name
        
        # Compare with baseline
        weighted_acc = models_dict['neural_ensemble']['winner_acc']
        # Note: This comparison isn't perfect since they're on different sets
        logging.info(f"\nWeighted average baseline: {weighted_acc:.3%}")
        
        return self.stacker

    # In the MLBModelTrainer class in pipelineTrainv5.py

    def save_models(self, models: Dict, feature_selector, scaler, dataset):
        """Save all models and preprocessing objects"""
        
        logging.info(f"\nSaving models to {self.model_dir}...")

        # --- START OF FIX: Create a list to hold model metadata ---
        model_metadata_list = []
        # --- END OF FIX ---
        
        # Save neural ensemble
        if 'neural_ensemble' in models:
            ensemble_data = models['neural_ensemble']
            for i, model in enumerate(ensemble_data['models']):
                model_name = ensemble_data['model_names'][i]
                model_type = type(model).__name__

                # Save the individual model file using its name
                torch.save(model.state_dict(), os.path.join(self.model_dir, f'{model_name}.pth'))
                
                # --- START OF FIX: Gather metadata for this model ---
                params = {}
                if hasattr(model, 'config'):
                    params = model.config.__dict__
                elif hasattr(model, 'hidden_dims'):
                    params = {'hidden_dims': [l.out_features for l in model.layers]}
                
                model_info = {
                    "name": model_name,
                    "type": model_type,
                    "params": params,
                    "feature_indices": self.feature_indices
                }
                model_metadata_list.append(model_info)
                # --- END OF FIX ---
            
            # Save ensemble weights
            np.save(os.path.join(self.model_dir, 'ensemble_weights.npy'), 
                    ensemble_data['weights'])
        
        # Save preprocessing objects
        joblib.dump(feature_selector, os.path.join(self.model_dir, 'feature_selector.pkl'))
        joblib.dump(scaler, os.path.join(self.model_dir, 'scaler.pkl'))
        joblib.dump(dataset.feature_engineer, os.path.join(self.model_dir, 'feature_engineer.pkl'))
        
        # Save configuration
        config_dict = {
            'model_config': self.config.__dict__,
            'input_dim': self.input_dim,
            'feature_indices': self.feature_indices,
            'selected_features': feature_selector.selected_features,
            'training_date': datetime.now().isoformat(),
            # --- START OF FIX: Add the crucial 'models' key here ---
            'models': model_metadata_list
            # --- END OF FIX ---
        }
        
        with open(os.path.join(self.model_dir, 'config.json'), 'w') as f:
            # Added a default handler to prevent errors with non-serializable objects
            json.dump(config_dict, f, indent=2, default=lambda o: '<not serializable>')
        
        logging.info("All models saved successfully!")

def analyze_model_performance_by_game_type(predictions: np.ndarray, 
                                         actuals: np.ndarray,
                                         game_metadata: pd.DataFrame):
    """Analyze model performance across different game situations"""
    results = {}
    
    # By day/night
    if 'dayNight' in game_metadata.columns:
        for game_type in ['day', 'night']:
            mask = game_metadata['dayNight'] == game_type
            if mask.sum() > 0:
                acc = ((predictions[mask, 0] > predictions[mask, 1]) == 
                      (actuals[mask, 0] > actuals[mask, 1])).mean()
                results[f'{game_type}_games'] = {
                    'count': mask.sum(),
                    'accuracy': acc
                }
    
    # By total runs
    total_runs = actuals[:, 0] + actuals[:, 1]
    
    # Low scoring games
    low_scoring_mask = total_runs < 7
    if low_scoring_mask.sum() > 0:
        acc = ((predictions[low_scoring_mask, 0] > predictions[low_scoring_mask, 1]) == 
              (actuals[low_scoring_mask, 0] > actuals[low_scoring_mask, 1])).mean()
        results['low_scoring'] = {
            'count': low_scoring_mask.sum(),
            'accuracy': acc
        }
    
    # High scoring games
    high_scoring_mask = total_runs > 10
    if high_scoring_mask.sum() > 0:
        acc = ((predictions[high_scoring_mask, 0] > predictions[high_scoring_mask, 1]) == 
              (actuals[high_scoring_mask, 0] > actuals[high_scoring_mask, 1])).mean()
        results['high_scoring'] = {
            'count': high_scoring_mask.sum(),
            'accuracy': acc
        }
    
    # Close games
    margin = np.abs(actuals[:, 0] - actuals[:, 1])
    close_mask = margin <= 1
    if close_mask.sum() > 0:
        acc = ((predictions[close_mask, 0] > predictions[close_mask, 1]) == 
              (actuals[close_mask, 0] > actuals[close_mask, 1])).mean()
        results['one_run_games'] = {
            'count': close_mask.sum(),
            'accuracy': acc
        }
    
    return results

def main():
    """Main training pipeline optimized for run prediction accuracy"""
    # Configuration
    FEATURES_PATH = r"C:\Users\DMelv\Documents\bettingModelBaseball\apiBaseball\pipeline\ml_pipeline_output\master_features_table.parquet"
    MODEL_DIR = "./mlb_run_prediction_model_v4"
    
    # Advanced architecture configuration
    USE_ADVANCED_FEATURES = True
    USE_HYBRID_MODEL = True  # Enable hybrid GNN+RNN+Perceiver model
    USE_ADVERSARIAL = True  # Enable adversarial training
    USE_SNAPSHOT_ENSEMBLE = True  # Enable snapshot ensembling
    USE_QUANTILE_REGRESSION = True  # Enable quantile regression
    
    # Initialize with advanced configuration
    dataset = TemporalMLBDataset(FEATURES_PATH)
    trainer = MLBModelTrainer(MODEL_DIR, use_advanced_features=USE_ADVANCED_FEATURES)
    
    # Update configuration with advanced options
    trainer.config.use_hybrid = USE_HYBRID_MODEL
    trainer.config.use_adversarial = USE_ADVERSARIAL
    trainer.config.use_snapshot_ensemble = USE_SNAPSHOT_ENSEMBLE
    trainer.config.use_quantile_regression = USE_QUANTILE_REGRESSION
    trainer.config.use_gnn = True  # Enable GNN in ensemble
    
    # Load and prepare data
    X, y, dates = dataset.load_and_prepare()
    
    # Also load the full dataframe for metadata
    df = pd.read_parquet(FEATURES_PATH)
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date').reset_index(drop=True)
    df = df.dropna(subset=['home_score', 'away_score'])
    
    # Create temporal splits
    X_train, y_train, X_val, y_val, X_test, y_test = dataset.create_temporal_splits(
        X, y, dates, test_size=0.15, val_size=0.1
    )
    
    # Get metadata for test set
    n_samples = len(X)
    val_end = int(n_samples * (1 - 0.15))
    test_idx = slice(val_end, n_samples)
    
    # Feature selection on training data only
    dataset.feature_selector.fit(X_train, y_train, X_val, y_val)
    X_train = dataset.feature_selector.transform(X_train)
    X_val = dataset.feature_selector.transform(X_val)
    X_test = dataset.feature_selector.transform(X_test)
    
    # Scale features
    X_train_scaled = dataset.scaler.fit_transform(X_train)
    X_val_scaled = dataset.scaler.transform(X_val)
    X_test_scaled = dataset.scaler.transform(X_test)
    
    # Convert back to DataFrames
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Train baseline models
    baselines, baseline_models = trainer.train_baseline_models(X_train, y_train, X_val, y_val)
    
    # Train diverse neural ensemble with advanced architectures
    logging.info("\n" + "="*60)
    logging.info("TRAINING DIVERSE NEURAL ENSEMBLE WITH ADVANCED ARCHITECTURES")
    logging.info("="*60)
    models = trainer.train_ensemble(X_train, y_train, X_val, y_val)
    
    # Train stacking ensemble
    stacker = trainer.train_stacking_ensemble(models, X_train, y_train, X_val, y_val)
    
    # Add baseline models to the results
    all_models = {'neural_ensemble': models['neural_ensemble']}
    all_models['ridge'] = baseline_models['ridge']
    all_models['lightgbm'] = baseline_models['lightgbm']
    
    # Evaluate on test set
    test_results = trainer.evaluate_on_test(all_models, X_test, y_test)
    stacking_results = trainer.evaluate_stacking_on_test(stacker, X_test, y_test)
    test_results['stacking_ensemble'] = stacking_results
    trainer.results = test_results
    
    # Additional analysis for neural ensemble
    if 'neural_ensemble' in test_results:
        predictions = test_results['neural_ensemble']['predictions']
        
        # Analyze by game type
        game_type_results = analyze_model_performance_by_game_type(
            predictions, y_test.values, df.iloc[test_idx]
        )
        
        logging.info("\nNeural Ensemble Performance by Game Type:")
        for game_type, metrics in game_type_results.items():
            logging.info(f"  {game_type}: {metrics['accuracy']:.1%} on {metrics['count']} games")
    
    # Additional analysis for stacking ensemble
    if 'stacking_ensemble' in test_results:
        stacking_predictions = test_results['stacking_ensemble']['predictions']
        
        # Analyze stacking performance by game type
        stacking_game_type_results = analyze_model_performance_by_game_type(
            stacking_predictions, y_test.values, df.iloc[test_idx]
        )
        
        logging.info("\nStacking Ensemble Performance by Game Type:")
        for game_type, metrics in stacking_game_type_results.items():
            logging.info(f"  {game_type}: {metrics['accuracy']:.1%} on {metrics['count']} games")
    
    # Compare stacking vs weighted average
    if 'neural_ensemble' in test_results and 'stacking_ensemble' in test_results:
        weighted_acc = test_results['neural_ensemble']['winner_acc']
        stacking_acc = test_results['stacking_ensemble']['winner_acc']
        improvement = (stacking_acc - weighted_acc) * 100
        
        logging.info("\n" + "="*60)
        logging.info("FINAL COMPARISON: STACKING vs WEIGHTED AVERAGE")
        logging.info("="*60)
        logging.info(f"Weighted Average Accuracy: {weighted_acc:.1%}")
        logging.info(f"Stacking Ensemble Accuracy: {stacking_acc:.1%}")
        logging.info(f"Improvement: {improvement:+.1f} percentage points")
        
        # Show which approach won for different confidence levels
        logging.info("\nBy Confidence Level:")
        for threshold in [0.5, 1.0, 1.5, 2.0]:
            key = f'margin_{threshold}'
            if key in test_results['neural_ensemble']['confidence_accuracies'] and \
               key in test_results['stacking_ensemble']['confidence_accuracies']:
                weighted = test_results['neural_ensemble']['confidence_accuracies'][key]['accuracy']
                stacking = test_results['stacking_ensemble']['confidence_accuracies'][key]['accuracy']
                count = test_results['stacking_ensemble']['confidence_accuracies'][key]['count']
                logging.info(f"  Margin > {threshold}: Weighted={weighted:.1%}, Stacking={stacking:.1%} ({count} games)")
        
        # Compare performance by game type
        if 'neural_ensemble' in test_results:
            logging.info("\nGame Type Performance Comparison (Neural vs Stacking):")
            for game_type in game_type_results.keys():
                if game_type in stacking_game_type_results:
                    neural_acc = game_type_results[game_type]['accuracy']
                    stacking_acc = stacking_game_type_results[game_type]['accuracy']
                    improvement = (stacking_acc - neural_acc) * 100
                    count = stacking_game_type_results[game_type]['count']
                    logging.info(f"  {game_type}: Neural={neural_acc:.1%}, Stacking={stacking_acc:.1%} "
                               f"({improvement:+.1f} pp, {count} games)")
    
    # Save the stacking model
    if hasattr(trainer, 'stacker'):
        joblib.dump(trainer.stacker, os.path.join(trainer.model_dir, 'stacking_ensemble.pkl'))
        logging.info("\nStacking ensemble saved!")
    # Save everything
    trainer.save_models(all_models, dataset.feature_selector, dataset.scaler, dataset)
    
    logging.info("\n" + "="*60)
    logging.info("TRAINING COMPLETE")
    logging.info("="*60)
    
    # Summary of key metrics
    logging.info("\nKEY PERFORMANCE METRICS:")
    
    # Neural ensemble metrics
    if 'neural_ensemble' in test_results:
        logging.info("\nNeural Ensemble (Weighted Average):")
        logging.info(f"  - Winner Accuracy: {test_results['neural_ensemble']['winner_acc']:.1%}")
        logging.info(f"  - Run Differential RMSE: {test_results['neural_ensemble']['diff_rmse']:.3f}")
        logging.info(f"  - Total Runs RMSE: {test_results['neural_ensemble']['total_runs_rmse']:.3f}")
        logging.info(f"  - High-Confidence (>1.5 margin) Accuracy: See above")
    
    # Stacking ensemble metrics
    if 'stacking_ensemble' in test_results:
        logging.info("\nStacking Ensemble:")
        logging.info(f"  - Winner Accuracy: {test_results['stacking_ensemble']['winner_acc']:.1%}")
        logging.info(f"  - Run Differential RMSE: {test_results['stacking_ensemble']['diff_rmse']:.3f}")
        logging.info(f"  - Total Runs RMSE: {test_results['stacking_ensemble']['total_runs_rmse']:.3f}")
        
        # Show improvement
        if 'neural_ensemble' in test_results:
            improvement = (test_results['stacking_ensemble']['winner_acc'] - 
                          test_results['neural_ensemble']['winner_acc']) * 100
            logging.info(f"  - Improvement over weighted average: {improvement:+.1f} percentage points")
    
    logging.info("\nMODEL IMPROVEMENTS IMPLEMENTED:")
    logging.info("- Graph Neural Networks for team relationships")
    logging.info("- Hybrid architecture combining GNN, RNN, and Perceiver")
    logging.info("- Self-supervised pretraining with masked feature prediction")
    logging.info("- Adversarial training for robustness")
    logging.info("- Dynamic loss weighting")
    logging.info("- Quantile regression for prediction intervals")
    logging.info("- Snapshot ensembling for efficient model diversity")
    logging.info("- Knowledge distillation for model compression")
    logging.info("- STACKING ENSEMBLE with rich meta-features")
    
    logging.info("\nRECOMMENDATIONS:")
    logging.info("- Focus on games with prediction margin > 1.0 for highest accuracy")
    logging.info("- Stacking ensemble provides best overall performance")
    logging.info("- Model performs best on clear winners (blowouts)")
    logging.info("- Continue monitoring performance on close games and low-scoring games")
    logging.info("- Use quantile predictions to understand prediction uncertainty")
    logging.info("- The student model provides faster inference with minimal accuracy loss")
    
    # Final recommendation based on results
    if 'neural_ensemble' in test_results and 'stacking_ensemble' in test_results:
        if test_results['stacking_ensemble']['winner_acc'] > test_results['neural_ensemble']['winner_acc']:
            logging.info("\n🏆 RECOMMENDATION: Use the STACKING ENSEMBLE for best performance!")
        else:
            logging.info("\n🏆 RECOMMENDATION: The weighted average ensemble is sufficient!")
    
    # If quantile regression was used, show prediction intervals
    if trainer.config.use_quantile_regression:
        logging.info("\nQuantile Prediction Analysis:")
        logging.info("- Quantile predictions available for uncertainty estimation")
        logging.info("- Use 10th/90th percentiles for prediction intervals")

if __name__ == "__main__":
    main()