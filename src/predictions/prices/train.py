"""
Price Prediction Model Training Script
=====================================

This script trains an LSTM-based model to predict electricity prices for the next 24 hours.

Usage:
------
python train.py [mode] [--scaler SCALER_TYPE]

Parameters:
-----------
mode : {'production', 'evaluation'}
    - production: Trains using all available data for real-world deployment
    - evaluation: Trains using train/validation/test split for model evaluation

--scaler : {'robust', 'minmax', 'standard'}, default='robust'
    Type of scaler to use for feature normalization:
    - robust: RobustScaler (handles outliers better)
    - minmax: MinMaxScaler (preserves feature magnitudes)
    - standard: StandardScaler (normalizes to mean=0, std=1)

Examples:
---------
# Train a production model using all data with robust scaler (default)
python train.py production

# Train an evaluation model with validation/test split using minmax scaler
python train.py evaluation --scaler minmax

# Train a production model with standard scaler
python train.py production --scaler standard

Outputs:
--------
The script will create the following in the models directory:

For production models (models/production/):
    - saved/price_model_production.keras - The trained model
    - saved/price_scaler.save - Price feature scaler
    - saved/grid_scaler.save - Grid feature scaler
    - saved/target_info.json - Information about target column and scaling
    - saved/feature_config.json - Feature configuration
    - logs/ - TensorBoard logs for training visualization
    - production_training_history.png - Plot of training metrics

For evaluation models (models/evaluation/):
    - saved/ - Same files as production
    - test_data/ - Test data for later evaluation
    - quick_evaluation.png - Visual performance summary on test data

Notes:
------
- Production models use all available data with a small validation set
- Evaluation models create a train/val/test split and save test data for later evaluation
- Both modes use the feature configuration from FeatureConfig class
- The window size (past hours used for prediction) is set in the feature config
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import argparse
from datetime import datetime, timedelta
import json
import os
import logging
from pathlib import Path
import sys

# Configure TensorFlow to use available CPU instructions
try:
    tf.config.optimizer.set_jit(True)  # Enable XLA optimization
except:
    logging.warning("Could not enable XLA optimization")

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the new FeatureConfig class
from src.predictions.prices.feature_config import FeatureConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PriceModelTrainer:
    def __init__(self, window_size=None, train_from_all_data=False):
        """Initialize trainer with project paths and default settings
        
        Args:
            window_size (int): Size of the input window
            train_from_all_data (bool): If True, uses all data for training without test split
        """
        # Load feature configuration
        self.feature_config = FeatureConfig()
        
        # Get window size from config if not provided
        training_params = self.feature_config.get_training_params()
        self.window_size = window_size or training_params.get("window_size", 168)
        self.model = None
        self.train_from_all_data = train_from_all_data
        
        # Setup project paths
        self.project_root = Path(__file__).resolve().parents[3]
        
        # Model directory paths
        # Models directory structure:
        # models/
        #   ├── production/  - Production model (trained on all data)
        #   │   ├── saved/   - Model files and scalers
        #   │   └── logs/    - TensorBoard logs
        #   └── evaluation/  - Evaluation model (with test split)
        #       ├── saved/   - Model files and scalers
        #       ├── logs/    - TensorBoard logs
        #       └── test_data/ - Test data for evaluation
        self.models_dir = Path(__file__).resolve().parent / "models"
        
        # Create specific paths based on training mode
        if self.train_from_all_data:
            # Production model paths
            self.model_dir = self.models_dir / "production"
            self.saved_dir = self.model_dir / "saved"
            self.logs_dir = self.model_dir / "logs"
            self.model_name = "price_model_production.keras"
        else:
            # Evaluation model paths
            self.model_dir = self.models_dir / "evaluation"
            self.saved_dir = self.model_dir / "saved"
            self.logs_dir = self.model_dir / "logs"
            self.test_data_dir = self.model_dir / "test_data"
            self.model_name = "price_model_evaluation.keras"
        
        # Create directories
        self.saved_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if not self.train_from_all_data:
            self.test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize scalers
        self._initialize_scalers()
    
    def _initialize_scalers(self):
        """Initialize the appropriate scalers based on config"""
        # Try to get scaling parameters from config if available
        scaling_config = None
        try:
            scaling_config = self.feature_config.get_scaling_params()
        except (AttributeError, KeyError):
            logging.warning("Could not load scaling parameters from config, using defaults")
        
        # Price scaler config
        if scaling_config and 'price_scaler' in scaling_config:
            price_scaler_type = scaling_config['price_scaler'].get('type', 'MinMaxScaler')
            logging.info(f"Using price scaler from config: {price_scaler_type}")
            
            if price_scaler_type == 'MinMaxScaler':
                # Get feature range from config or use default
                feature_range = scaling_config['price_scaler'].get('feature_range', (-1, 1))
                # Ensure feature_range is a tuple (scikit-learn requirement)
                if isinstance(feature_range, list):
                    feature_range = tuple(feature_range)
                    logging.info(f"Converted feature_range from list to tuple: {feature_range}")
                self.price_scaler = MinMaxScaler(feature_range=feature_range)
                logging.info(f"Using MinMaxScaler with feature_range={feature_range} for prices")
            elif price_scaler_type == 'RobustScaler':
                # Get price scaler parameters from config
                price_quantile_range = scaling_config['price_scaler'].get('quantile_range', (0, 100))
                # Ensure quantile_range is a tuple
                if isinstance(price_quantile_range, list):
                    price_quantile_range = tuple(price_quantile_range)
                    logging.info(f"Converted quantile_range from list to tuple: {price_quantile_range}")
                unit_variance = scaling_config['price_scaler'].get('unit_variance', True)
                self.price_scaler = RobustScaler(quantile_range=price_quantile_range, unit_variance=unit_variance)
                logging.info(f"Using RobustScaler with quantile_range={price_quantile_range}, unit_variance={unit_variance} for prices")
            elif price_scaler_type == 'StandardScaler':
                self.price_scaler = StandardScaler()
                logging.info("Using StandardScaler for prices")
            else:
                # Default to MinMaxScaler if the type is not recognized
                self.price_scaler = MinMaxScaler(feature_range=(-1, 1))
                logging.info(f"Unknown price scaler type '{price_scaler_type}', using MinMaxScaler with feature_range=(-1, 1)")
        else:
            # Default to MinMaxScaler if no config is available
            self.price_scaler = MinMaxScaler(feature_range=(-1, 1))
            logging.info("No price scaler config found, using MinMaxScaler with feature_range=(-1, 1)")
            
        # Store additional price scaler settings
        if scaling_config and 'price_scaler' in scaling_config:
            self.clip_negative = scaling_config['price_scaler'].get('clip_negative', False)
            self.max_reasonable_price = scaling_config['price_scaler'].get('max_reasonable_price', 1500)
        else:
            self.clip_negative = False
            self.max_reasonable_price = 1500
            
        # Initialize price transformation settings from config
        self.price_transform_config = {}
        if scaling_config and 'price_transform_config' in scaling_config:
            self.price_transform_config = scaling_config['price_transform_config']
            if self.price_transform_config.get('enable_log_transform', False):
                cols_to_transform = self.price_transform_config.get('apply_to_cols', [])
                transform_offset = self.price_transform_config.get('offset', 1.0)
                logging.info(f"Enabling log transformation for price columns: {cols_to_transform} with offset {transform_offset}")
        
        # Grid scaler config
        if scaling_config and 'grid_scaler' in scaling_config:
            grid_scaler_type = scaling_config['grid_scaler'].get('type', 'RobustScaler')
            logging.info(f"Using grid scaler from config: {grid_scaler_type}")
            
            if grid_scaler_type == 'RobustScaler':
                # Get grid scaler parameters from config or use enhanced defaults
                grid_quantile_range = scaling_config['grid_scaler'].get('quantile_range', (1, 99))
                # Ensure quantile_range is a tuple
                if isinstance(grid_quantile_range, list):
                    grid_quantile_range = tuple(grid_quantile_range)
                    logging.info(f"Converted grid quantile_range from list to tuple: {grid_quantile_range}")
                unit_variance = scaling_config['grid_scaler'].get('unit_variance', False)
                self.grid_scaler = RobustScaler(quantile_range=grid_quantile_range, unit_variance=unit_variance)
                logging.info(f"Using RobustScaler with quantile_range={grid_quantile_range}, unit_variance={unit_variance} for grid features")
            elif grid_scaler_type == 'MinMaxScaler':
                feature_range = scaling_config['grid_scaler'].get('feature_range', (-1, 1))
                # Ensure feature_range is a tuple
                if isinstance(feature_range, list):
                    feature_range = tuple(feature_range)
                    logging.info(f"Converted grid feature_range from list to tuple: {feature_range}")
                self.grid_scaler = MinMaxScaler(feature_range=feature_range)
                logging.info(f"Using MinMaxScaler with feature_range={feature_range} for grid features")
            elif grid_scaler_type == 'StandardScaler':
                self.grid_scaler = StandardScaler()
                logging.info("Using StandardScaler for grid features")
            else:
                # Default to RobustScaler if the type is not recognized
                self.grid_scaler = RobustScaler(quantile_range=(1, 99), unit_variance=False)
                logging.info(f"Unknown grid scaler type '{grid_scaler_type}', using RobustScaler with quantile_range=(1, 99), unit_variance=False")
        else:
            # Default to RobustScaler if no config is available
            self.grid_scaler = RobustScaler(quantile_range=(1, 99), unit_variance=False)
            logging.info("No grid scaler config found, using RobustScaler with quantile_range=(1, 99), unit_variance=False")
                
        # Store additional outlier handling parameters
        if scaling_config and 'grid_scaler' in scaling_config:
            try:
                self.outlier_threshold = scaling_config['grid_scaler'].get('outlier_threshold', 8)
                self.handle_extreme_values = scaling_config['grid_scaler'].get('handle_extreme_values', True)
                self.max_zscore = scaling_config['grid_scaler'].get('max_zscore', 8)
                self.import_export_cols = scaling_config['grid_scaler'].get('import_export_cols', [])
                self.log_transform_large_values = scaling_config['grid_scaler'].get('log_transform_large_values', True)
                self.individual_scaling = scaling_config['grid_scaler'].get('individual_scaling', True)
                logging.info(f"Grid outlier handling: threshold={self.outlier_threshold}, enabled={self.handle_extreme_values}")
                logging.info(f"Grid advanced settings: max_zscore={self.max_zscore}, log_transform={self.log_transform_large_values}, individual_scaling={self.individual_scaling}")
            except Exception as e:
                logging.warning(f"Error loading grid outlier config: {e}")
                self.outlier_threshold = 8
                self.handle_extreme_values = True
                self.max_zscore = 8
                self.import_export_cols = []
                self.log_transform_large_values = True
                self.individual_scaling = True
        else:
            self.outlier_threshold = 8
            self.handle_extreme_values = True
            self.max_zscore = 8
            self.import_export_cols = []
            self.log_transform_large_values = True
            self.individual_scaling = True
    
    def get_required_features(self):
        """List all required features for the model"""
        return self.feature_config.get_ordered_features()
    
    def find_matching_features(self, df, required_features):
        """
        Find available features that match the required features.
        If an exact match doesn't exist, try to find one with the same prefix.
        
        Args:
            df: DataFrame with actual columns
            required_features: List of feature names from config
            
        Returns:
            Dict mapping required feature names to actual column names
        """
        feature_map = {}
        df_columns = df.columns.tolist()
        
        for feature in required_features:
            if feature in df_columns:
                # Exact match
                feature_map[feature] = feature
            else:
                # Try to find close matches
                # 1. Check for prefix match
                prefix_matches = [col for col in df_columns if col.startswith(feature.split('_')[0])]
                # 2. Check for semantic match (e.g., price -> SE3_price_ore)
                semantic_matches = [col for col in df_columns if feature.split('_')[0].lower() in col.lower()]
                
                if prefix_matches:
                    logging.info(f"Using {prefix_matches[0]} as substitute for {feature}")
                    feature_map[feature] = prefix_matches[0]
                elif semantic_matches:
                    logging.info(f"Using {semantic_matches[0]} as substitute for {feature}")
                    feature_map[feature] = semantic_matches[0]
                else:
                    # No match found - this will be handled by the caller
                    feature_map[feature] = None
        
        return feature_map
    
    def load_data(self):
        """Load and prepare the processed data"""
        logging.info("Loading datasets...")
        
        # Load price data first
        price_df = pd.read_csv(
            self.project_root / "data/processed/SE3prices.csv", 
            index_col=0
        )
        price_df.index = pd.to_datetime(price_df.index)
        
        # Load time features
        time_df = pd.read_csv(
            self.project_root / "data/processed/time_features.csv",
            index_col=0
        )
        time_df.index = pd.to_datetime(time_df.index)
        
        # Load holiday data
        holiday_df = pd.read_csv(
            self.project_root / "data/processed/holidays.csv",
            index_col=0
        )
        holiday_df.index = pd.to_datetime(holiday_df.index)
        
        # Load grid features
        grid_df = pd.read_csv(
            self.project_root / "data/processed/SwedenGrid.csv",
            index_col=0
        )
        grid_df.index = pd.to_datetime(grid_df.index)
        
        # Get common date range where all data is available
        start_date = max(
            price_df.index.min(),
            time_df.index.min(),
            holiday_df.index.min(),
            grid_df.index.min()
        )
        
        # For production model, use all available data up to the latest date
        end_date = min(
            price_df.index.max(),
            time_df.index.max(),
            holiday_df.index.max(),
            grid_df.index.max()
        )
        
        logging.info(f"Using data from {start_date} to {end_date}")
        
        # Trim all dataframes to common date range
        price_df = price_df[
            (price_df.index >= start_date)
        ]
        
        time_df = time_df[
            (time_df.index >= start_date)
        ]
        
        holiday_df = holiday_df[
            (holiday_df.index >= start_date)
        ]
        
        grid_df = grid_df[
            (grid_df.index >= start_date)
        ]
        
        # Merge all features
        df = price_df.join(time_df, how='left')
        df = df.join(holiday_df, how='left')
        df = df.join(grid_df, how='left')
        
        # Handle any missing values
        df = df.ffill().bfill()
        
        logging.info(f"Final dataset shape: {df.shape}")
        logging.info(f"Available columns: {sorted(df.columns.tolist())}")
        
        return df

    def create_sequences(self, data):
        """Convert dataframe of features to sequences for LSTM training"""
        X, y = [], []
        values = data.values
        for i in range(len(values) - self.window_size - 24):  # Need 24 more steps for targets
            X.append(values[i:(i + self.window_size)])
            # Get next 24 hours of prices as target
            y.append(values[i + self.window_size:i + self.window_size + 24, 0])  # First column is price
        return np.array(X), np.array(y)
    
    def prepare_data(self, df):
        """
        Prepare the data for training by scaling features and target.
        
        Args:
            df: DataFrame with features and target
        
        Returns:
            X: Scaled feature matrix
            y: Target values
        """
        # Check for missing columns and available price columns
        target_name = self.feature_config.get_target_name()
        if target_name not in df.columns:
            logging.warning(f"Target column '{target_name}' not found in dataframe")
            # Try to find a suitable price column
            price_cols = [col for col in df.columns if 'price' in col.lower()]
            if price_cols:
                new_target = price_cols[0]
                logging.info(f"Using '{new_target}' as target column instead")
                # Update feature_config
                self.feature_config.metadata['target_feature'] = new_target
                target_name = new_target
            else:
                raise ValueError(f"No suitable price column found in dataframe. Available columns: {sorted(df.columns.tolist())}")
        
        # Add specialized features for morning and evening peak hours
        logging.info("Adding specialized features for better spike prediction")
        
        # Create dataframe copy to avoid modifying the original
        df_enhanced = df.copy()
        
        # Add hour-of-day indicator to improve temporal patterns
        df_enhanced['hour'] = df_enhanced.index.hour
        
        # Add peak hour multipliers to better capture morning and evening price spikes
        # Morning peak (7-9am)
        df_enhanced['morning_peak'] = df_enhanced.index.hour.isin([7, 8, 9]).astype(float)
        
        # Evening peak (17-20pm)
        df_enhanced['evening_peak'] = df_enhanced.index.hour.isin([17, 18, 19, 20]).astype(float)
        
        # Calculate price momentum features using robust methods that avoid division by zero
        # For price_momentum_1h, use difference instead of percentage for stability
        prices = df_enhanced[target_name].values
        
        # For 1-hour momentum, use simple difference normalized by recent average price
        # This avoids division by zero when calculating percentages
        diffs_1h = np.zeros_like(prices)
        diffs_1h[1:] = prices[1:] - prices[:-1]
        
        # Get average prices over last week to normalize differences (avoid division by zero)
        avg_prices = df_enhanced[target_name].rolling(168, min_periods=1).mean().bfill().values
        # Use a minimum value to avoid division by very small numbers
        avg_prices = np.maximum(avg_prices, 10.0)  # Minimum average price of 10 öre/kWh 
        
        # Normalize differences by average price
        df_enhanced['price_momentum_1h'] = diffs_1h / avg_prices
        
        # For 24-hour momentum, use the same approach
        diffs_24h = np.zeros_like(prices)
        diffs_24h[24:] = prices[24:] - prices[:-24]
        df_enhanced['price_momentum_24h'] = diffs_24h / avg_prices
        
        # Create spike indicators
        mean_price = df_enhanced[target_name].rolling(168).mean().fillna(df_enhanced[target_name].mean())
        std_price = df_enhanced[target_name].rolling(168).std().fillna(df_enhanced[target_name].std())
        
        # Recent spike indicator (1 if price was >2 std above mean in past 24h, else 0)
        spike_threshold = mean_price + 2 * std_price
        df_enhanced['recent_spike'] = df_enhanced[target_name].rolling(24).max().fillna(0) > spike_threshold
        df_enhanced['recent_spike'] = df_enhanced['recent_spike'].astype(float)
        
        # Add price differential to nearest extremes, using robust methods
        # Use rolling max with fillna to prevent division by zero
        day_high = df_enhanced[target_name].rolling(24).max().fillna(df_enhanced[target_name])
        day_low = df_enhanced[target_name].rolling(24).min().fillna(df_enhanced[target_name])
        
        # Ensure minimum values for denominators
        day_high = np.maximum(day_high.values, 10.0)  # Minimum of 10 öre/kWh
        day_low = np.maximum(day_low.values, 10.0)    # Minimum of 10 öre/kWh
        
        df_enhanced['price_vs_day_high'] = prices / day_high
        df_enhanced['price_vs_day_low'] = prices / day_low
        
        # Replace any potential infinities or NaNs with zeros
        for col in ['price_momentum_1h', 'price_momentum_24h', 'price_vs_day_high', 'price_vs_day_low']:
            df_enhanced[col] = df_enhanced[col].replace([np.inf, -np.inf, np.nan], 0)
        
        # Fill NaN values that might be introduced
        df_enhanced = df_enhanced.fillna(0)
        
        # Make sure the enhanced dataframe has all the original columns plus the new ones
        all_columns = list(df.columns) + [
            'hour', 'morning_peak', 'evening_peak', 
            'price_momentum_1h', 'price_momentum_24h', 
            'recent_spike', 'price_vs_day_high', 'price_vs_day_low'
        ]
        
        # Log the new features
        logging.info(f"Added spike prediction features: {[col for col in all_columns if col not in df.columns]}")
        logging.info("Feature value ranges after calculation:")
        for col in ['price_momentum_1h', 'price_momentum_24h', 'price_vs_day_high', 'price_vs_day_low']:
            stats = df_enhanced[col].describe()
            # Safe way to access describe stats that works with all pandas versions
            mean_val = stats.loc['mean'] if hasattr(stats, 'loc') else stats['mean']
            std_val = stats.loc['std'] if hasattr(stats, 'loc') else stats['std']
            min_val = stats.loc['min'] if hasattr(stats, 'loc') else stats['min']
            max_val = stats.loc['max'] if hasattr(stats, 'loc') else stats['max']
            logging.info(f"  {col}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}, std={std_val:.4f}")
        
        # Get features and target from the enhanced dataframe
        X = df_enhanced[[col for col in all_columns if col != target_name and col in df_enhanced.columns]].copy()
        y = df_enhanced[target_name].values.reshape(-1, 1)
            
        # Log pre-scaling feature statistics
        price_cols = self.feature_config.get_price_cols()
        price_cols = [col for col in price_cols if col in X.columns]  # Filter to only include existing columns
        
        if price_cols:
            logging.info("Price feature statistics before scaling:")
            for col in price_cols:
                if col in X.columns:
                    stats = X[col].describe()
                    # Safe way to access describe stats that works with all pandas versions
                    mean_val = stats.loc['mean'] if hasattr(stats, 'loc') else stats['mean']
                    std_val = stats.loc['std'] if hasattr(stats, 'loc') else stats['std']
                    min_val = stats.loc['min'] if hasattr(stats, 'loc') else stats['min']
                    max_val = stats.loc['max'] if hasattr(stats, 'loc') else stats['max']
                    logging.info(f"{col}: mean={mean_val:.3f}, std={std_val:.3f}, min={min_val:.3f}, max={max_val:.3f}")
        
        grid_cols = self.feature_config.get_grid_cols()
        grid_cols = [col for col in grid_cols if col in X.columns]  # Filter to only include existing columns
        
        if grid_cols:
            logging.info("Grid feature statistics before scaling:")
            for col in grid_cols:
                if col in X.columns:
                    stats = X[col].describe()
                    # Safe way to access describe stats that works with all pandas versions
                    mean_val = stats.loc['mean'] if hasattr(stats, 'loc') else stats['mean']
                    std_val = stats.loc['std'] if hasattr(stats, 'loc') else stats['std']
                    min_val = stats.loc['min'] if hasattr(stats, 'loc') else stats['min']
                    max_val = stats.loc['max'] if hasattr(stats, 'loc') else stats['max']
                    logging.info(f"{col}: mean={mean_val:.3f}, std={std_val:.3f}, min={min_val:.3f}, max={max_val:.3f}")
        
        # Get scaling configuration
        scaling_config = self.feature_config.get_scaling_params()
        grid_scaling_config = scaling_config.get("grid_scaler", {}) if scaling_config else {}
        
        # Check if we should apply log transformation to price features
        if hasattr(self, 'price_transform_config') and self.price_transform_config.get('enable_log_transform', False):
            cols_to_transform = self.price_transform_config.get('apply_to_cols', [])
            cols_to_transform = [col for col in cols_to_transform if col in X.columns]  # Filter to only include existing columns
            transform_offset = self.price_transform_config.get('offset', 1.0)
            
            if cols_to_transform:
                logging.info(f"Applying log transformation to price columns: {cols_to_transform}")
                for col in cols_to_transform:
                    if col in X.columns:
                        # Save original stats for comparison
                        stats = X[col].describe()
                        # Safe way to access describe stats that works with all pandas versions
                        orig_mean = stats.loc['mean'] if hasattr(stats, 'loc') else stats['mean']
                        orig_std = stats.loc['std'] if hasattr(stats, 'loc') else stats['std']
                        orig_min = stats.loc['min'] if hasattr(stats, 'loc') else stats['min']
                        orig_max = stats.loc['max'] if hasattr(stats, 'loc') else stats['max']
                        
                        # Check if column has negative values
                        has_negative = (X[col] < 0).any()
                        
                        if has_negative:
                            # Apply signed log transform: sign(x) * log(|x| + offset)
                            logging.warning(f"Column {col} has negative values (min={orig_min:.3f}). Using signed log transform.")
                            signs = np.sign(X[col])
                            abs_vals = np.abs(X[col])
                            X[col] = signs * np.log1p(abs_vals + transform_offset - 1.0)  # log1p(x) = log(1+x)
                        else:
                            # Standard log transform: log(x + offset)
                            X[col] = np.log1p(X[col] + transform_offset - 1.0)  # Ensure log1p operates on x + offset - 1
                        
                        # Log transformation results
                        new_stats = X[col].describe()
                        # Safe way to access describe stats that works with all pandas versions
                        new_mean = new_stats.loc['mean'] if hasattr(new_stats, 'loc') else new_stats['mean']
                        new_std = new_stats.loc['std'] if hasattr(new_stats, 'loc') else new_stats['std']
                        new_min = new_stats.loc['min'] if hasattr(new_stats, 'loc') else new_stats['min']
                        new_max = new_stats.loc['max'] if hasattr(new_stats, 'loc') else new_stats['max']
                        logging.info(f"Log-transformed {col}: original(mean={orig_mean:.3f}, std={orig_std:.3f}, min={orig_min:.3f}, max={orig_max:.3f}) → "
                                    f"transformed(mean={new_mean:.3f}, std={new_std:.3f}, min={new_min:.3f}, max={new_max:.3f})")

        # Handle grid features first - identify import/export columns which need special handling
        import_export_cols = []
        if hasattr(self, 'import_export_cols'):
            import_export_cols = self.import_export_cols
        else:
            import_export_cols = grid_scaling_config.get("import_export_cols", [])
        
        import_export_cols = [col for col in import_export_cols if col in X.columns]  # Filter to only include existing columns
        
        # Apply log transformation to import/export features if enabled
        should_log_transform = False
        if hasattr(self, 'log_transform_large_values'):
            should_log_transform = self.log_transform_large_values
        else:
            should_log_transform = grid_scaling_config.get("log_transform_large_values", False)
        
        if should_log_transform and import_export_cols:
            logging.info("Applying log transformation to high-magnitude import/export features")
            for col in import_export_cols:
                if col in X.columns:
                    # Check if column has negative values
                    has_negative = (X[col] < 0).any()
                    min_val = X[col].min()
                    
                    # Save original values for logging
                    stats = X[col].describe()
                    # Safe way to access describe stats that works with all pandas versions
                    orig_mean = stats.loc['mean'] if hasattr(stats, 'loc') else stats['mean']
                    orig_std = stats.loc['std'] if hasattr(stats, 'loc') else stats['std']
                    orig_max = stats.loc['max'] if hasattr(stats, 'loc') else stats['max']
                    orig_min = min_val
                    
                    if has_negative:
                        # Safe transformation for data with negative values: sign(x) * log(|x| + 1)
                        logging.warning(f"Column {col} has negative values (min={min_val:.3f}). Using signed log transform.")
                        signs = np.sign(X[col])
                        abs_vals = np.abs(X[col])
                        X[col] = signs * np.log1p(abs_vals)
                    else:
                        # Standard log1p for non-negative data
                        X[col] = np.log1p(X[col])
                    
                    # Log the transformation effect
                    new_stats = X[col].describe()
                    # Safe way to access describe stats that works with all pandas versions
                    new_mean = new_stats.loc['mean'] if hasattr(new_stats, 'loc') else new_stats['mean']
                    new_std = new_stats.loc['std'] if hasattr(new_stats, 'loc') else new_stats['std']
                    new_min = new_stats.loc['min'] if hasattr(new_stats, 'loc') else new_stats['min']
                    new_max = new_stats.loc['max'] if hasattr(new_stats, 'loc') else new_stats['max']
                    logging.info(f"Log-transformed {col}: original(mean={orig_mean:.3f}, std={orig_std:.3f}, "
                                f"min={orig_min:.3f}, max={orig_max:.3f}) → "
                                f"transformed(mean={new_mean:.3f}, std={new_std:.3f}, "
                                f"min={new_min:.3f}, max={new_max:.3f})")
        
        # Apply clipping to negative price values if configured
        if hasattr(self, 'clip_negative') and self.clip_negative and price_cols:
            logging.info("Clipping negative price values to 0")
            for col in price_cols:
                if col in X.columns:
                    neg_count = (X[col] < 0).sum()
                    if neg_count > 0:
                        logging.info(f"Clipping {neg_count} negative values in {col}")
                        X.loc[X[col] < 0, col] = 0.0
        
        # Scale price features (make sure all columns exist)
        existing_price_cols = [col for col in price_cols if col in X.columns]
        if existing_price_cols:
            price_features = X[existing_price_cols].copy()
            # Fix: Convert list to tuple for RobustScaler
            if hasattr(self.price_scaler, 'quantile_range') and isinstance(self.price_scaler.quantile_range, list):
                self.price_scaler.quantile_range = tuple(self.price_scaler.quantile_range)
            X.loc[:, existing_price_cols] = self.price_scaler.fit_transform(price_features)
            
            # Log sample of scaled price values
            logging.info("Sample of scaled price values:")
            sample_idx = min(5, len(X))
            for col in existing_price_cols:
                if col in X.columns:
                    logging.info(f"{col} (scaled): {X[col].iloc[:sample_idx].values}")
        
        # Handle grid features with special care for import/export values
        existing_grid_cols = [col for col in grid_cols if col in X.columns]
        if existing_grid_cols:
            grid_features = X[existing_grid_cols].copy()
            
            # Handle extreme outliers if enabled
            handle_extreme = False
            if hasattr(self, 'handle_extreme_values'):
                handle_extreme = self.handle_extreme_values
            else:
                handle_extreme = grid_scaling_config.get("handle_extreme_values", False)
                
            if handle_extreme:
                # Log maximum values before capping
                extreme_values = []
                max_zscore = self.max_zscore if hasattr(self, 'max_zscore') else grid_scaling_config.get("max_zscore", 8)
                
                for col in existing_grid_cols:
                    if col in grid_features.columns:
                        col_max = grid_features[col].max()
                        col_mean = grid_features[col].mean()
                        col_std = grid_features[col].std()
                        if col_std > 0:
                            z_score_max = (col_max - col_mean) / col_std
                            if z_score_max > max_zscore:
                                extreme_values.append((col, col_max, z_score_max))
                
                if extreme_values:
                    logging.warning(f"Found {len(extreme_values)} grid features with extreme values:")
                    for col, max_val, z_score in extreme_values:
                        logging.warning(f"{col}: max={max_val:.2f}, z-score={z_score:.2f}")
                    
                    # Cap extreme values using winsorization
                    for col in existing_grid_cols:
                        if col in grid_features.columns:
                            mean = grid_features[col].mean()
                            std = grid_features[col].std()
                            if std > 0:  # Avoid division by zero
                                upper_bound = mean + max_zscore * std
                                lower_bound = mean - max_zscore * std
                                if grid_features[col].max() > upper_bound:
                                    count_capped_upper = (grid_features[col] > upper_bound).sum()
                                    if count_capped_upper > 0:
                                        logging.info(f"Capping {count_capped_upper} high extreme values in {col} at {upper_bound:.2f}")
                                        grid_features.loc[grid_features[col] > upper_bound, col] = upper_bound
                                if grid_features[col].min() < lower_bound:
                                    count_capped_lower = (grid_features[col] < lower_bound).sum()
                                    if count_capped_lower > 0:
                                        logging.info(f"Capping {count_capped_lower} low extreme values in {col} at {lower_bound:.2f}")
                                        grid_features.loc[grid_features[col] < lower_bound, col] = lower_bound
            
            # Check if we should use individual scaling for grid columns
            use_individual_scaling = False
            if hasattr(self, 'individual_scaling'):
                use_individual_scaling = self.individual_scaling
            else:
                use_individual_scaling = grid_scaling_config.get("individual_scaling", False)
            
            if use_individual_scaling:
                logging.info("Using individual scaling for grid features")
                for col in existing_grid_cols:
                    if col in grid_features.columns:
                        try:
                            # Create a new scaler for each column
                            quantile_range = tuple(grid_scaling_config.get("quantile_range", (0, 100)))
                            unit_variance = grid_scaling_config.get("unit_variance", True)
                            col_scaler = RobustScaler(quantile_range=quantile_range, unit_variance=unit_variance)
                            X.loc[:, col] = col_scaler.fit_transform(grid_features[col].values.reshape(-1, 1))
                            logging.info(f"Individual scaling applied to {col}")
                        except Exception as col_err:
                            logging.error(f"Could not scale column {col}: {col_err}")
                            # Last resort: standardize manually
                            mean = grid_features[col].mean()
                            std = grid_features[col].std() or 1.0  # Avoid division by zero
                            X.loc[:, col] = (grid_features[col] - mean) / std
                            logging.warning(f"Manual standardization applied to {col}")
            else:
                try:
                    # Fix: Convert list to tuple for RobustScaler
                    if hasattr(self.grid_scaler, 'quantile_range') and isinstance(self.grid_scaler.quantile_range, list):
                        self.grid_scaler.quantile_range = tuple(self.grid_scaler.quantile_range)
                    
                    # Try to scale grid features together
                    scaled_grid = self.grid_scaler.fit_transform(grid_features)
                    X.loc[:, existing_grid_cols] = scaled_grid
                    
                    # Log post-scaling grid feature statistics
                    logging.info("Grid feature statistics after scaling:")
                    for i, col in enumerate(existing_grid_cols):
                        if col in X.columns:
                            stats = X[col].describe()
                            # Safe way to access describe stats that works with all pandas versions
                            mean_val = stats.loc['mean'] if hasattr(stats, 'loc') else stats['mean']
                            std_val = stats.loc['std'] if hasattr(stats, 'loc') else stats['std']
                            min_val = stats.loc['min'] if hasattr(stats, 'loc') else stats['min']
                            max_val = stats.loc['max'] if hasattr(stats, 'loc') else stats['max']
                            logging.info(f"{col}: mean={mean_val:.3f}, std={std_val:.3f}, min={min_val:.3f}, max={max_val:.3f}")
                    
                except (ValueError, RuntimeError) as e:
                    logging.error(f"Error scaling grid features: {e}")
                    logging.warning("Falling back to individual column scaling for grid features")
                    
                    # Fallback: scale each column individually
                    for col in existing_grid_cols:
                        if col in grid_features.columns:
                            try:
                                # Create a new scaler for each column
                                quantile_range = tuple(grid_scaling_config.get("quantile_range", (0, 100)))
                                unit_variance = grid_scaling_config.get("unit_variance", True)
                                col_scaler = RobustScaler(quantile_range=quantile_range, unit_variance=unit_variance)
                                X.loc[:, col] = col_scaler.fit_transform(grid_features[col].values.reshape(-1, 1))
                                logging.info(f"Individual scaling applied to {col}")
                            except Exception as col_err:
                                logging.error(f"Could not scale column {col}: {col_err}")
                                # Last resort: standardize manually
                                mean = grid_features[col].mean()
                                std = grid_features[col].std() or 1.0  # Avoid division by zero
                                X.loc[:, col] = (grid_features[col] - mean) / std
                                logging.warning(f"Manual standardization applied to {col}")
        
        # Scale new features added for spike prediction
        # Handle the momentum features directly with MinMaxScaler, which is more robust to outliers
        # than RobustScaler for these specific features
        momentum_features = ['price_momentum_1h', 'price_momentum_24h']
        momentum_present = [col for col in momentum_features if col in X.columns]
        
        if momentum_present:
            logging.info("Scaling momentum features")
            try:
                # Use a simple MinMaxScaler instead of RobustScaler for more stability with momentum features
                momentum_scaler = MinMaxScaler(feature_range=(-1, 1))
                X.loc[:, momentum_present] = momentum_scaler.fit_transform(X[momentum_present])
                logging.info("Successfully scaled momentum features")
            except Exception as e:
                logging.warning(f"Error scaling momentum features: {e}")
                # Fallback to manual scaling
                for col in momentum_present:
                    values = X[col].values
                    abs_max = max(abs(np.nanmin(values)), abs(np.nanmax(values))) or 1.0
                    X[col] = X[col] / abs_max
                    logging.info(f"Applied manual scaling to {col}")
        
        # Other spike features need less processing
        spike_features = [
            'morning_peak', 'evening_peak', 
            'recent_spike', 'price_vs_day_high', 'price_vs_day_low'
        ]
        
        spike_features_present = [col for col in spike_features if col in X.columns]
        if spike_features_present:
            # These binary/ratio features are already in good ranges
            logging.info("Spike feature statistics after preparation:")
            for col in spike_features_present:
                stats = X[col].describe()
                # Safe way to access describe stats that works with all pandas versions
                mean_val = stats.loc['mean'] if hasattr(stats, 'loc') else stats['mean']
                std_val = stats.loc['std'] if hasattr(stats, 'loc') else stats['std']
                min_val = stats.loc['min'] if hasattr(stats, 'loc') else stats['min']
                max_val = stats.loc['max'] if hasattr(stats, 'loc') else stats['max']
                logging.info(f"{col}: mean={mean_val:.3f}, std={std_val:.3f}, min={min_val:.3f}, max={max_val:.3f}")
        
        # Log overall statistics after scaling
        logging.info("Feature statistics after all scaling:")
        for col in X.columns:
            try:
                stats = X[col].describe()
                # Safe way to access describe stats that works with all pandas versions
                mean_val = stats.loc['mean'] if hasattr(stats, 'loc') else stats['mean']
                std_val = stats.loc['std'] if hasattr(stats, 'loc') else stats['std']
                min_val = stats.loc['min'] if hasattr(stats, 'loc') else stats['min']
                max_val = stats.loc['max'] if hasattr(stats, 'loc') else stats['max']
                logging.info(f"{col}: mean={mean_val:.3f}, std={std_val:.3f}, min={min_val:.3f}, max={max_val:.3f}")
            except Exception as e:
                logging.warning(f"Unable to log statistics for {col}: {e}")
                logging.info(f"{col}: Sample values = {X[col].iloc[:5].values}")
        
        # Further validation - check for NaNs or infinities in numeric columns only
        # Get numeric columns to avoid TypeError with string columns
        numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        
        # Check for NaNs
        nan_cols = X[numeric_cols].columns[X[numeric_cols].isna().any()].tolist()
        if nan_cols:
            logging.error(f"Found NaN values in columns after scaling: {nan_cols}")
            # Try to fix NaNs with fillna
            X = X.fillna(0)
            logging.warning("NaNs have been replaced with zeros")
            
        # Check for inf values - only on numeric columns
        numeric_X = X[numeric_cols]
        inf_mask = np.isinf(numeric_X)
        inf_cols = numeric_X.columns[inf_mask.any()].tolist()
        
        if inf_cols:
            logging.error(f"Found infinite values in columns after scaling: {inf_cols}")
            # Try to replace infinities with large but finite values
            X.loc[:, inf_cols] = X.loc[:, inf_cols].replace([np.inf, -np.inf], [1e6, -1e6])
            logging.warning("Infinite values have been replaced with +/-1e6")
        
        return X, y
    
    def build_model(self, input_shape):
        """Build LSTM model using Keras Functional API for better architecture"""
        # Input layer with explicit shape
        inputs = Input(shape=(input_shape[0], input_shape[1]), name='sequence_input')
        
        # Get architecture config
        architecture = self.feature_config.get_architecture_params()
        lstm_layers = architecture.get("lstm_layers", [
            {"units": 256, "return_sequences": True, "dropout": 0.2},
            {"units": 128, "return_sequences": False, "dropout": 0.2}
        ])
        dense_layers = architecture.get("dense_layers", [
            {"units": 64, "activation": "relu"},
            {"units": 24, "activation": None}
        ])
        
        # Build LSTM layers
        x = inputs
        for i, layer in enumerate(lstm_layers):
            return_sequences = layer.get("return_sequences", False)
            dropout_rate = layer.get("dropout", 0)
            
            # Add LSTM layer
            lstm_layer = LSTM(
                units=layer.get("units", 128),
                return_sequences=return_sequences,
                activation='tanh',
                name=f'lstm_{i+1}'
            )(x)
            
            # Add dropout if specified
            if dropout_rate > 0:
                lstm_layer = Dropout(dropout_rate, name=f'lstm_dropout_{i+1}')(lstm_layer)
            
            x = lstm_layer
        
        # Add Dense layers
        for i, layer in enumerate(dense_layers[:-1]):
            x = Dense(
                units=layer.get("units", 64),
                activation=layer.get("activation", "relu"),
                name=f'dense_{i+1}'
            )(x)
        
        # Add output layer separately to ensure proper shape
        outputs = Dense(
            units=dense_layers[-1].get("units", 24) if dense_layers else 24,
            activation=dense_layers[-1].get("activation") if dense_layers else None,
            name='output'
        )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Get training parameters from config
        training_params = self.feature_config.get_training_params()
        learning_rate = training_params.get("learning_rate", 0.001)
        loss = training_params.get("loss", "huber")
        metrics = training_params.get("metrics", ["mae"])
        
        # Define custom asymmetric loss function that penalizes underprediction more than overprediction
        # This helps the model better predict price spikes which are important for decision making
        def asymmetric_huber_loss(y_true, y_pred, delta=1.0, asymmetry_factor=2.0):
            """Custom asymmetric Huber loss that penalizes underprediction more heavily
            
            Args:
                y_true: Ground truth values
                y_pred: Predicted values
                delta: Threshold for switching between MSE and MAE (as in Huber loss)
                asymmetry_factor: Factor to multiply loss when underpredicting (y_pred < y_true)
            """
            import tensorflow as tf
            error = y_true - y_pred
            is_underpredict = tf.cast(tf.less(y_pred, y_true), tf.float32)
            
            # Calculate absolute error
            abs_error = tf.abs(error)
            
            # Standard Huber loss calculation
            huber_loss = tf.where(
                abs_error <= delta,
                0.5 * tf.square(error),  # MSE for small errors
                delta * (abs_error - 0.5 * delta)  # MAE for large errors
            )
            
            # Apply asymmetry factor to underpredictions
            asymmetric_factor = 1.0 + (asymmetry_factor - 1.0) * is_underpredict
            weighted_loss = huber_loss * asymmetric_factor
            
            return tf.reduce_mean(weighted_loss)
        
        # Use the custom loss function if specified in config
        use_asymmetric_loss = training_params.get("use_asymmetric_loss", True)
        if use_asymmetric_loss and loss == "huber":
            # Custom loss with partial application to set default parameters
            from functools import partial
            asymmetry_factor = training_params.get("asymmetry_factor", 2.0)
            custom_loss = partial(asymmetric_huber_loss, asymmetry_factor=asymmetry_factor)
            custom_loss.__name__ = 'asymmetric_huber_loss'  # Required for Keras
            logging.info(f"Using asymmetric Huber loss with asymmetry factor: {asymmetry_factor}")
            loss = custom_loss
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        model.summary()
        return model
    
    def train(self, epochs=None, batch_size=None):
        """Train the model using config parameters"""
        logging.info("Preparing data...")
        (X_train, y_train, X_val, y_val, X_test, y_test), timestamps = self.prepare_data()
        
        # Get training parameters from config
        training_params = self.feature_config.get_training_params()
        epochs = epochs or training_params["max_epochs"]
        batch_size = batch_size or training_params["batch_size"]
        
        # Get callback parameters
        callback_params = self.feature_config.get_callback_params()
        
        # Save scalers
        joblib.dump(self.price_scaler, self.saved_dir / 'price_scaler.save')
        joblib.dump(self.grid_scaler, self.saved_dir / 'grid_scaler.save')
        
        if not self.train_from_all_data:
            # Save test data for evaluation
            train_split = int(len(timestamps) * self.feature_config.data_split["train_ratio"])
            val_split = int(len(timestamps) * (
                self.feature_config.data_split["train_ratio"] + 
                self.feature_config.data_split["val_ratio"]
            ))
            
            np.save(self.test_data_dir / 'X_test.npy', X_test)
            np.save(self.test_data_dir / 'y_test.npy', y_test)
            np.save(self.test_data_dir / 'test_timestamps.npy', timestamps[val_split:])
            
            # Save validation data for quick evaluation later
            np.save(self.test_data_dir / 'X_val.npy', X_val)
            np.save(self.test_data_dir / 'y_val.npy', y_val)
            np.save(self.test_data_dir / 'val_timestamps.npy', timestamps[train_split:val_split])
        
        logging.info("Building model...")
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Configure callbacks
        model_checkpoint_path = self.saved_dir / self.model_name
        log_dir = self.logs_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
        
        callbacks = [
            EarlyStopping(
                monitor=callback_params["early_stopping"]["monitor"],
                patience=callback_params["early_stopping"]["patience"],
                restore_best_weights=callback_params["early_stopping"]["restore_best_weights"],
                mode='min'
            ),
            ModelCheckpoint(
                model_checkpoint_path,
                monitor=callback_params["early_stopping"]["monitor"],
                save_best_only=True,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor=callback_params["reduce_lr"]["monitor"],
                factor=callback_params["reduce_lr"]["factor"],
                patience=callback_params["reduce_lr"]["patience"],
                min_lr=callback_params["reduce_lr"]["min_lr"],
                mode='min'
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True
            )
        ]
        
        model_type = 'production' if self.train_from_all_data else 'evaluation'
        logging.info(f"Training {model_type} model...")
        logging.info(f"Model will be saved to: {model_checkpoint_path}")
        logging.info(f"TensorBoard logs: {log_dir}")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot training history
        self.plot_training_history(history, model_type)
        
        # Save the history for later reference
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(self.saved_dir / f'training_history.csv')
        
        # If it's an evaluation model, run a quick evaluation
        if not self.train_from_all_data and X_test is not None:
            self.quick_evaluation(X_test, y_test)
        
        return history
    
    def quick_evaluation(self, X_test, y_test):
        """Run a quick evaluation on test data and save results"""
        logging.info("\nRunning quick evaluation on test data...")
        
        # Predict on test data
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        test_loss = self.model.evaluate(X_test, y_test, verbose=0)
        test_mae = test_loss[1] if len(test_loss) > 1 else None
        
        # Set consistent style for plots
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create a 2x2 dashboard with key evaluation insights
        fig = plt.figure(figsize=(16, 14), dpi=100)
        
        # Define a consistent color scheme
        colors = {
            'actual': '#1F77B4',    # Blue
            'predicted': '#FF7F0E', # Orange
            'error': '#D62728',     # Red
            'grid': '#BDBDBD'       # Light gray
        }
        
        # 1. Top left: Representative sample with prediction
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        
        # Get a representative sample from middle of test set
        sample_idx = len(y_test) // 2  # Middle sample is often more representative
        true_vals = y_test[sample_idx]
        pred_vals = y_pred[sample_idx]
        
        # Inverse transform the values for realistic prices
        true_dummy = np.zeros((len(true_vals), len(self.price_scaler.scale_)))
        true_dummy[:, 0] = true_vals
        pred_dummy = np.zeros((len(pred_vals), len(self.price_scaler.scale_)))
        pred_dummy[:, 0] = pred_vals
        
        true_inv = self.price_scaler.inverse_transform(true_dummy)[:, 0]
        pred_inv = self.price_scaler.inverse_transform(pred_dummy)[:, 0]
        
        # Calculate error for this sample
        error = np.abs(true_inv - pred_inv)
        
        # Plot sample with error band
        hours = range(len(true_vals))
        ax1.plot(hours, true_inv, '-', color=colors['actual'], label='Actual', linewidth=2.5)
        ax1.plot(hours, pred_inv, '--', color=colors['predicted'], label='Predicted', linewidth=2.5)
        
        # Add error shading
        ax1.fill_between(hours, pred_inv - error/2, pred_inv + error/2, color=colors['predicted'], alpha=0.2)
        
        # Format plot
        ax1.set_title('Representative 24-Hour Prediction Sample', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Hour of Day', fontsize=12)
        ax1.set_ylabel('Price (öre/kWh)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
        ax1.legend(fontsize=10)
        
        # Add metrics annotation for this sample
        sample_mae = np.mean(error)
        sample_mape = np.mean(np.abs(error / (true_inv + 1e-8))) * 100
        metrics_text = f"Sample MAE: {sample_mae:.2f} öre/kWh\nSample MAPE: {sample_mape:.2f}%"
        ax1.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        # 2. Top right: Error distribution histogram
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        
        # Calculate all errors across test set
        all_errors = []
        for i in range(min(len(y_test), 100)):  # Use up to 100 samples for efficiency
            true_sample = y_test[i]
            pred_sample = y_pred[i]
            
            # Inverse transform
            true_dummy = np.zeros((len(true_sample), len(self.price_scaler.scale_)))
            true_dummy[:, 0] = true_sample
            pred_dummy = np.zeros((len(pred_sample), len(self.price_scaler.scale_)))
            pred_dummy[:, 0] = pred_sample
            
            true_inv = self.price_scaler.inverse_transform(true_dummy)[:, 0]
            pred_inv = self.price_scaler.inverse_transform(pred_dummy)[:, 0]
            
            # Add errors to list
            all_errors.extend(np.abs(true_inv - pred_inv))
        
        # Plot error distribution
        ax2.hist(all_errors, bins=30, color=colors['error'], alpha=0.7)
        ax2.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Absolute Error (öre/kWh)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
        
        # Add distribution statistics
        mean_error = np.mean(all_errors)
        median_error = np.median(all_errors)
        p90_error = np.percentile(all_errors, 90)
        stats_text = f"Mean Error: {mean_error:.2f}\nMedian Error: {median_error:.2f}\n90th Percentile: {p90_error:.2f}"
        ax2.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction', 
                    ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        # 3. Bottom left: Hourly pattern analysis
        ax3 = plt.subplot2grid((2, 2), (1, 0))
        
        # Analyze error patterns by hour of day
        hourly_errors = [[] for _ in range(24)]
        for i in range(min(len(y_test), 100)):  # Use up to 100 samples
            true_sample = y_test[i]
            pred_sample = y_pred[i]
            
            # Inverse transform
            true_dummy = np.zeros((len(true_sample), len(self.price_scaler.scale_)))
            true_dummy[:, 0] = true_sample
            pred_dummy = np.zeros((len(pred_sample), len(self.price_scaler.scale_)))
            pred_dummy[:, 0] = pred_sample
            
            true_inv = self.price_scaler.inverse_transform(true_dummy)[:, 0]
            pred_inv = self.price_scaler.inverse_transform(pred_dummy)[:, 0]
            
            # Group errors by hour
            for h in range(min(24, len(true_inv))):
                hourly_errors[h].append(abs(true_inv[h] - pred_inv[h]))
        
        # Calculate mean and std for each hour
        hourly_means = [np.mean(errors) for errors in hourly_errors if errors]
        hourly_stds = [np.std(errors) for errors in hourly_errors if errors]
        hours = range(len(hourly_means))
        
        # Plot hourly patterns
        ax3.bar(hours, hourly_means, yerr=hourly_stds, color=colors['error'], alpha=0.7, 
               error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
        
        # Highlight peak hours (typically morning and evening)
        peak_hours = [7, 8, 9, 17, 18, 19, 20]
        for hour in peak_hours:
            if hour < len(hourly_means):
                ax3.axvspan(hour-0.4, hour+0.4, color='yellow', alpha=0.2)
        
        ax3.set_title('Error by Hour of Day', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Hour', fontsize=12)
        ax3.set_ylabel('Mean Absolute Error (öre/kWh)', fontsize=12)
        ax3.set_xticks(hours)
        ax3.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
        
        # Annotate peak hours
        ax3.annotate('Morning Peak', xy=(8, ax3.get_ylim()[1]*0.95), 
                    ha='center', fontsize=9, 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.2))
        ax3.annotate('Evening Peak', xy=(18.5, ax3.get_ylim()[1]*0.95), 
                    ha='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.2))
        
        # 4. Bottom right: Overall metrics summary
        ax4 = plt.subplot2grid((2, 2), (1, 1))
        ax4.axis('off')
        
        # Calculate additional metrics
        mean_abs_error = np.mean(all_errors)
        median_abs_error = np.median(all_errors)
        p90_error = np.percentile(all_errors, 90)
        max_error = np.max(all_errors)
        
        # Create metrics summary
        metrics_summary = [
            f"Test Dataset Metrics:",
            f"",
            f"Mean Absolute Error: {mean_abs_error:.2f} öre/kWh",
            f"Median Absolute Error: {median_abs_error:.2f} öre/kWh",
            f"90th Percentile Error: {p90_error:.2f} öre/kWh",
            f"Maximum Error: {max_error:.2f} öre/kWh",
            f"",
            f"Model Loss: {test_loss[0]:.6f}",
        ]
        
        if test_mae is not None:
            metrics_summary.append(f"Model MAE (scaled): {test_mae:.6f}")
        
        # Add error pattern insights
        highest_error_hour = np.argmax(hourly_means)
        lowest_error_hour = np.argmin(hourly_means)
        
        insights = [
            f"",
            f"Key Insights:",
            f"",
            f"• Hardest to predict: Hour {highest_error_hour}:00 (MAE: {hourly_means[highest_error_hour]:.2f})",
            f"• Most accurate: Hour {lowest_error_hour}:00 (MAE: {hourly_means[lowest_error_hour]:.2f})",
        ]
        
        # Determine if morning or evening peaks have higher errors
        morning_peak_error = np.mean([hourly_means[h] for h in [7, 8, 9] if h < len(hourly_means)])
        evening_peak_error = np.mean([hourly_means[h] for h in [17, 18, 19, 20] if h < len(hourly_means)])
        
        if morning_peak_error > evening_peak_error:
            insights.append(f"• Morning peak hours are more challenging to predict")
        else:
            insights.append(f"• Evening peak hours are more challenging to predict")
        
        # Create a text box with all metrics
        ax4.text(0.5, 0.5, '\n'.join(metrics_summary + insights), 
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round,pad=1', facecolor='#f0f0f0', edgecolor='gray', alpha=0.9),
                transform=ax4.transAxes)
        
        # Add overall title
        plt.suptitle('Model Quick Evaluation Dashboard', fontsize=18, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save figure
        plt.savefig(self.saved_dir / 'quick_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics to file
        metrics = {
            'test_loss': test_loss[0],
            'test_mae': test_mae,
            'mean_abs_error': mean_abs_error,
            'median_abs_error': median_abs_error,
            'p90_error': p90_error,
            'max_error': max_error
        }
        
        # Save metrics to file
        with open(self.saved_dir / 'test_metrics.txt', 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        
        logging.info(f"Quick evaluation complete. Saved to quick_evaluation.png")
        logging.info(f"Test Loss: {test_loss[0]:.4f}")
        if test_mae is not None:
            logging.info(f"Test MAE: {test_mae:.4f}")
        logging.info(f"Mean Absolute Error: {mean_abs_error:.2f} öre/kWh")

    def plot_training_history(self, history, model_type=''):
        """Plot and save training history with improved styling"""
        # Create a larger figure with better resolution
        plt.figure(figsize=(16, 6), dpi=100)
        
        # Set a consistent style for better readability
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Plot loss with improved styling
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], linewidth=2.5, label='Training Loss', color='#2471A3')
        plt.plot(history.history['val_loss'], linewidth=2.5, label='Validation Loss', color='#E67E22', linestyle='--')
        plt.title(f'{model_type.capitalize()} Model - Loss', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Loss', fontsize=12, fontweight='bold')
        plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='gray')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Add minor gridlines for better readability
        plt.minorticks_on()
        plt.grid(True, which='minor', linestyle=':', alpha=0.4)
        
        # Plot MAE if available with consistent styling
        if 'mae' in history.history:
            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'], linewidth=2.5, label='Training MAE', color='#2471A3')
            plt.plot(history.history['val_mae'], linewidth=2.5, label='Validation MAE', color='#E67E22', linestyle='--')
            plt.title(f'{model_type.capitalize()} Model - MAE', fontsize=16, fontweight='bold', pad=15)
            plt.xlabel('Epoch', fontsize=12, fontweight='bold')
            plt.ylabel('MAE', fontsize=12, fontweight='bold')
            plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='gray')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add minor gridlines for better readability
            plt.minorticks_on()
            plt.grid(True, which='minor', linestyle=':', alpha=0.4)
        
        # Add a dataset text to show data splits
        if not self.train_from_all_data:
            train_ratio = self.feature_config.data_split["train_ratio"]
            val_ratio = self.feature_config.data_split["val_ratio"]
            test_ratio = self.feature_config.data_split["test_ratio"]
            plt.figtext(0.5, 0.01, 
                       f"Data split: Training {train_ratio*100:.1f}% | Validation {val_ratio*100:.1f}% | Test {test_ratio*100:.1f}%",
                       ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5, "edgecolor":"gray"})
        else:
            plt.figtext(0.5, 0.01, 
                       "Production model: Trained on all available data with small validation sample",
                       ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5, "edgecolor":"gray"})
        
        # Adjust layout and save with higher quality
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make room for the text at the bottom
        plt.savefig(self.saved_dir / f'{model_type}_training_history.png', dpi=300, bbox_inches='tight')
        
        # Additional informative plot: Learning rate if available
        if 'lr' in history.history:
            plt.figure(figsize=(10, 4))
            plt.plot(history.history['lr'], 'o-', color='#16A085', linewidth=2)
            plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Learning Rate', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.yscale('log')  # Log scale makes it easier to see learning rate decay
            plt.savefig(self.saved_dir / f'{model_type}_learning_rate.png', dpi=300, bbox_inches='tight')
            
        # Don't show plots during automated training
        plt.close('all')

    def run_training(self):
        """Run the training process"""
        # Load and prepare data
        df = self.load_data()
        
        # Verify all required features exist
        missing_cols = self.feature_config.missing_columns(df)
        if missing_cols:
            logging.error("Missing features:")
            for col in missing_cols:
                logging.error(f"  - {col}")
            logging.error(f"Available columns: {sorted(df.columns.tolist())}")
            raise ValueError(f"Missing required features: {missing_cols}")
            
        # Prepare data for training (scaling)
        X_scaled, y_scaled = self.prepare_data(df)
        
        # Create sequences with ordered features
        feature_cols = self.feature_config.get_all_feature_names()
        df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
        df_scaled[self.feature_config.get_target_name()] = y_scaled
        
        # Create sequences
        X, y = self.create_sequences(df_scaled)
        
        if self.train_from_all_data:
            logging.info("\nUsing all data for production model training")
            # Use all data for training, keeping a small validation set for monitoring
            val_size = min(int(len(X) * 0.05), 5000)  # Use 5% or max 5000 samples for validation
            
            # Take the validation set from the middle of the dataset to avoid recent data
            mid_point = len(X) // 2
            val_start = mid_point - val_size // 2
            val_end = val_start + val_size
            
            # Create validation set from middle
            X_val = X[val_start:val_end]
            y_val = y[val_start:val_end]
            
            # Create training set from remaining data
            X_train = np.concatenate([X[:val_start], X[val_end:]], axis=0)
            y_train = np.concatenate([y[:val_start], y[val_end:]], axis=0)
            
            logging.info(f"Total samples: {len(X)}")
            logging.info(f"Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
            logging.info(f"Validation samples: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
            
            # Build and train model
            model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            history = self.train_model(model, X_train, y_train, X_val, y_val)
            
            # Save the model and scalers
            self.save_model_and_scalers(model)
            
            return model, history, df.index[self.window_size + 24:]
        else:
            # Get data split ratios from config for evaluation model
            split_ratios = self.feature_config.get_data_split_ratios()
            train_ratio = split_ratios["train_ratio"]
            val_ratio = split_ratios["val_ratio"]
            
            # Calculate split indices
            total_samples = len(X)
            train_split = int(total_samples * train_ratio)
            val_split = int(total_samples * (train_ratio + val_ratio))
            
            # Log split information
            logging.info("\nData Split Information for Evaluation Model:")
            logging.info(f"Total samples: {total_samples}")
            logging.info(f"Training samples: {train_split} ({train_ratio*100:.1f}%)")
            logging.info(f"Validation samples: {val_split - train_split} ({val_ratio*100:.1f}%)")
            logging.info(f"Test samples: {total_samples - val_split} ({split_ratios['test_ratio']*100:.1f}%)")
            
            # Split the data
            X_train = X[:train_split]
            y_train = y[:train_split]
            X_val = X[train_split:val_split]
            y_val = y[train_split:val_split]
            X_test = X[val_split:]
            y_test = y[val_split:]
            
            # Save test data for later evaluation
            self.test_data_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Saving test data to {self.test_data_dir}")
            
            # Save test features and targets
            np.save(self.test_data_dir / "X_test.npy", X_test)
            np.save(self.test_data_dir / "y_test.npy", y_test)
            
            # Save validation data too (needed for the evaluation script)
            np.save(self.test_data_dir / "X_val.npy", X_val)
            np.save(self.test_data_dir / "y_val.npy", y_val)
            
            # Save timestamps for the test and validation data
            test_timestamps = df.index[self.window_size + 24:][val_split:]
            val_timestamps = df.index[self.window_size + 24:][train_split:val_split]
            
            np.save(self.test_data_dir / "test_timestamps.npy", test_timestamps)
            np.save(self.test_data_dir / "val_timestamps.npy", val_timestamps)
            
            logging.info(f"Saved test data: {len(X_test)} samples")
            logging.info(f"Saved validation data: {len(X_val)} samples")
            
            # Build and train model
            model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            history = self.train_model(model, X_train, y_train, X_val, y_val)
            
            # Save the model and scalers
            self.save_model_and_scalers(model)
            
            return model, history, df.index[self.window_size + 24:]

    def train_model(self, model, X_train, y_train, X_val, y_val):
        """Train the model with the provided data"""
        # Get training parameters from config
        training_params = self.feature_config.get_training_params()
        batch_size = training_params.get("batch_size", 32)
        max_epochs = training_params.get("max_epochs", 100)
        
        # Get callback parameters from config
        callback_params = self.feature_config.get_callback_params()
        
        # Early stopping callback
        early_stopping_params = callback_params.get("early_stopping", {})
        early_stopping = EarlyStopping(
            monitor=early_stopping_params.get("monitor", "val_loss"),
            patience=early_stopping_params.get("patience", 10),
            min_delta=early_stopping_params.get("min_delta", 0.001),
            restore_best_weights=True,
            verbose=1
        )
        
        # Reduce learning rate callback
        reduce_lr_params = callback_params.get("reduce_lr", {})
        reduce_lr = ReduceLROnPlateau(
            monitor=reduce_lr_params.get("monitor", "val_loss"),
            factor=reduce_lr_params.get("factor", 0.5),
            patience=reduce_lr_params.get("patience", 5),
            min_delta=reduce_lr_params.get("min_delta", 0.001),
            min_lr=reduce_lr_params.get("min_lr", 0.00001),
            verbose=1
        )
        
        # Train the model
        logging.info(f"Training model with batch_size={batch_size}, max_epochs={max_epochs}")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=max_epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history

    def save_model_and_scalers(self, model):
        """Save model and scalers to disk"""
        # Create timestamp for model version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine save directory based on training mode
        save_dir = self.saved_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = save_dir / self.model_name
        model.save(model_path)
        logging.info(f"Model saved to {model_path}")
        
        # Verify scalers before saving
        # Test the price scaler with a sample value to ensure it works correctly
        try:
            target_name = self.feature_config.get_target_name()
            target_index = 0  # Default to first column
            price_cols = self.feature_config.get_price_cols()
            if target_name in price_cols:
                target_index = price_cols.index(target_name)
            
            # Test the scaler with sample values
            test_val = np.array([100.0]).reshape(-1, 1)  # A typical price value
            dummy = np.zeros((1, len(self.price_scaler.scale_)))
            dummy[:, target_index] = test_val
            
            scaled = self.price_scaler.transform(dummy)
            unscaled = self.price_scaler.inverse_transform(scaled)
            
            # Check if the transformation is reversible
            original_val = unscaled[:, target_index]
            logging.info(f"Scaler test: original={test_val[0][0]}, scaled={scaled[0][target_index]}, unscaled={original_val[0]}")
            
            if not np.isclose(test_val[0][0], original_val[0], rtol=1e-5):
                logging.warning(f"Price scaler verification issue - output {original_val[0]} doesn't match input {test_val[0][0]}")
        except Exception as e:
            logging.error(f"Error verifying price scaler: {e}")
        
        # Save scalers
        joblib.dump(self.price_scaler, save_dir / "price_scaler.save")
        joblib.dump(self.grid_scaler, save_dir / "grid_scaler.save")
        logging.info(f"Scalers saved to {save_dir}")
        
        # Save target column information
        target_info = {
            "target_feature": self.feature_config.get_target_name(),
            "target_index": target_index,
            "price_cols": self.feature_config.get_price_cols(),
            "price_scaler_type": type(self.price_scaler).__name__,
            "price_scaler_params": self.price_scaler.get_params() if hasattr(self.price_scaler, 'get_params') else {},
            "timestamp": timestamp
        }
        
        with open(save_dir / "target_info.json", 'w') as f:
            json.dump(target_info, f, indent=4)
        logging.info(f"Target information saved to {save_dir / 'target_info.json'}")
        
        # Save feature configuration
        config_path = save_dir / "feature_config.json"
        try:
            with open(config_path, 'w') as f:
                json.dump({
                    "feature_groups": self.feature_config.feature_groups,
                    "feature_metadata": self.feature_config.metadata,
                    "model_config": self.feature_config.model_config
                }, f, indent=4)
            logging.info(f"Feature configuration saved to {config_path}")
        except Exception as e:
            logging.error(f"Error saving feature configuration: {e}")
            
        logging.info("Model, scalers, and configuration saved successfully")

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the model on test data and generate performance metrics and plots
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
        """
        logging.info("\nEvaluating model on test data...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = np.mean(np.abs(y_pred - y_test), axis=0)  # MAE per hour
        mape = np.mean(np.abs((y_pred - y_test) / (y_test + 1e-5)) * 100, axis=0)  # MAPE per hour
        rmse = np.sqrt(np.mean(np.square(y_pred - y_test), axis=0))  # RMSE per hour
        
        # Calculate overall metrics
        overall_mae = np.mean(mae)
        overall_mape = np.mean(mape)
        overall_rmse = np.mean(rmse)
        
        # Log metrics
        logging.info(f"Test MAE: {overall_mae:.2f} öre/kWh")
        logging.info(f"Test MAPE: {overall_mape:.2f}%")
        logging.info(f"Test RMSE: {overall_rmse:.2f} öre/kWh")
        
        # Log hourly metrics
        logging.info("\nHourly metrics:")
        for h in range(24):
            logging.info(f"Hour {h}: MAE={mae[h]:.2f}, MAPE={mape[h]:.2f}%, RMSE={rmse[h]:.2f}")
        
        # Create evaluation directory if it doesn't exist
        eval_dir = self.model_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Create plots
        self._create_evaluation_plots(y_test, y_pred, eval_dir)
        
        # Save numerical results
        self._save_evaluation_results(y_test, y_pred, mae, mape, rmse, eval_dir)
        
        return {
            "mae": overall_mae,
            "mape": overall_mape,
            "rmse": overall_rmse,
            "hourly_mae": mae.tolist(),
            "hourly_mape": mape.tolist(),
            "hourly_rmse": rmse.tolist()
        }
    
    def _create_evaluation_plots(self, y_true, y_pred, save_dir):
        """
        Create evaluation plots
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_dir: Directory to save plots
        """
        # Plot 1: Scatter plot of predicted vs actual
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true.flatten(), y_pred.flatten(), alpha=0.3)
        plt.plot([0, np.max(y_true)], [0, np.max(y_true)], 'r--')
        plt.xlabel('Actual Price (öre/kWh)')
        plt.ylabel('Predicted Price (öre/kWh)')
        plt.title('Predicted vs Actual Electricity Prices')
        plt.grid(True, alpha=0.3)
        scatter_path = save_dir / "pred_vs_actual_scatter.png"
        plt.savefig(scatter_path)
        plt.close()
        logging.info(f"Scatter plot saved to {scatter_path}")
        
        # Plot 2: Distribution of errors
        errors = y_pred.flatten() - y_true.flatten()
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, alpha=0.75)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Prediction Error (öre/kWh)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors')
        plt.grid(True, alpha=0.3)
        hist_path = save_dir / "error_distribution.png"
        plt.savefig(hist_path)
        plt.close()
        logging.info(f"Error distribution plot saved to {hist_path}")
        
        # Plot 3: Hourly MAE
        mae = np.mean(np.abs(y_pred - y_true), axis=0)
        plt.figure(figsize=(12, 6))
        plt.bar(range(24), mae)
        plt.xlabel('Hour of Day')
        plt.ylabel('Mean Absolute Error (öre/kWh)')
        plt.title('Prediction Error by Hour of Day')
        plt.xticks(range(24))
        plt.grid(True, alpha=0.3)
        hourly_path = save_dir / "hourly_mae.png"
        plt.savefig(hourly_path)
        plt.close()
        logging.info(f"Hourly MAE plot saved to {hourly_path}")
        
        # Plot 4: Sample of predictions vs actual
        sample_idx = np.random.choice(len(y_true), min(5, len(y_true)), replace=False)
        plt.figure(figsize=(15, 10))
        
        for i, idx in enumerate(sample_idx):
            plt.subplot(len(sample_idx), 1, i+1)
            plt.plot(range(24), y_true[idx], 'b-', label='Actual')
            plt.plot(range(24), y_pred[idx], 'r-', label='Predicted')
            plt.xlabel('Hour of Day')
            plt.ylabel('Price (öre/kWh)')
            plt.title(f'Sample {i+1}: Price Prediction vs Actual')
            plt.xticks(range(0, 24, 2))
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        sample_path = save_dir / "sample_predictions.png"
        plt.savefig(sample_path)
        plt.close()
        logging.info(f"Sample predictions plot saved to {sample_path}")
    
    def _save_evaluation_results(self, y_true, y_pred, mae, mape, rmse, save_dir):
        """
        Save numerical evaluation results
        
        Args:
            y_true: True values
            y_pred: Predicted values
            mae: Mean absolute error
            mape: Mean absolute percentage error
            rmse: Root mean squared error
            save_dir: Directory to save results
        """
        # Create results dictionary
        results = {
            "overall_metrics": {
                "mae": float(np.mean(mae)),
                "mape": float(np.mean(mape)),
                "rmse": float(np.mean(rmse))
            },
            "hourly_metrics": {
                "mae": mae.tolist(),
                "mape": mape.tolist(),
                "rmse": rmse.tolist()
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Save as JSON
        results_path = save_dir / "evaluation_metrics.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f"Evaluation metrics saved to {results_path}")
        
        # Save sample predictions as CSV
        sample_idx = np.random.choice(len(y_true), min(100, len(y_true)), replace=False)
        samples = {
            "hour": np.tile(np.arange(24), len(sample_idx)),
            "sample_id": np.repeat(np.arange(len(sample_idx)), 24),
            "actual": y_true[sample_idx].flatten(),
            "predicted": y_pred[sample_idx].flatten(),
            "error": y_pred[sample_idx].flatten() - y_true[sample_idx].flatten()
        }
        
        samples_df = pd.DataFrame(samples)
        samples_path = save_dir / "sample_predictions.csv"
        samples_df.to_csv(samples_path, index=False)
        logging.info(f"Sample predictions saved to {samples_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train electricity price prediction model using LSTM architecture')
    parser.add_argument('mode', choices=['production', 'evaluation'], 
                        help='Training mode: "production" uses all data for deployment, "evaluation" creates train/validation/test split for model assessment without running evaluation')
    parser.add_argument('--window', type=int, default=None,
                        help='Override window size (past hours to use for prediction). Default is from config.')
    args = parser.parse_args()

    # Set training mode based on argument
    train_from_all_data = args.mode == 'production'
    
    # Log the training mode
    logging.info(f"\n=== Training {args.mode.capitalize()} Model ===")
    logging.info(f"Training with {'all available' if train_from_all_data else 'train/val/test split'} data")
    
    # Initialize and train the model
    trainer = PriceModelTrainer(window_size=args.window, train_from_all_data=train_from_all_data)
    
    # Train the model
    model, history, dates = trainer.run_training()
    
    logging.info(f"{args.mode.capitalize()} model training complete!")
    logging.info("Model and scalers have been saved successfully")
    
    if args.mode == 'production':
        logging.info(f"Model saved to: {trainer.saved_dir / trainer.model_name}")
    else:
        logging.info(f"Model saved to: {trainer.saved_dir / trainer.model_name}")
        logging.info(f"Test data saved to: {trainer.test_data_dir}")
        logging.info("\nTo evaluate the model, run: python evaluate.py")

if __name__ == "__main__":
    main()


