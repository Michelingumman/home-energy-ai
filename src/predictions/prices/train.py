#!/usr/bin/env python
"""
Training script for electricity price forecasting with 3 models.
Trains a trend model using XGBOOST (with eXogenous variables).
Trains a peak model using TCN.
Trains a valley model using TCN.

Usage:
    TRAINING MODE: (trains on data split into train, val, test sets)
    python train.py --model trend
    python train.py --model peak
    python train.py --model valley
    
    PRODUCTION MODE: (trains on all data)
    python train.py --model trend --production
    python train.py --model peak --production
    python train.py --model valley --production
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Concatenate, Activation, BatchNormalization, Flatten, Reshape, Multiply, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.regularizers import l1_l2
from tcn import TCN
import statsmodels.api as sm
import matplotlib.pyplot as plt
import logging
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import joblib
from sklearn.preprocessing import StandardScaler
import sys
import traceback
from scipy.signal import medfilt
from scipy.signal import savgol_filter
import time  # Added for progress tracking
from tqdm import tqdm  # Added for progress bars
import psutil  # For memory usage tracking
import xgboost as xgb  # For Gradient Boosting model
from xgboost.callback import TrainingCallback  # Add TrainingCallback import
import pickle
from tensorflow.keras import backend as K
from tensorflow.keras.losses import BinaryCrossentropy

# Define custom layers that will be used in model architecture
class GlobalSumPooling1D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GlobalSumPooling1D, self).__init__(**kwargs)
        
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
        
    def get_config(self):
        config = super(GlobalSumPooling1D, self).get_config()
        return config

# Add the directory of this file to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import from utility file
from utils import (
    add_time_features, add_lag_features, add_rolling_features,
    add_price_spike_indicators, get_scaler, create_sequences,
    get_loss_function, plot_training_history, evaluate_model,
    plot_test_prediction, plot_feature_importance,
    add_spike_labels, spike_weighted_loss,
    detect_price_peaks, detect_price_valleys_derivative,
    detect_valleys_robust,
    xgb_smooth_trend_objective, xgb_trend_stability_objective
)

# Import config
from config import (
    MODELS_DIR, DATA_DIR, PLOTS_DIR, LOGS_DIR,
    TARGET_VARIABLE, LOOKBACK_WINDOW, PREDICTION_HORIZON,
    VALIDATION_SPLIT, TEST_SPLIT,
    DROPOUT_RATE, L1_REG, L2_REG,
    CORE_FEATURES, EXTENDED_FEATURES,
    SE3_PRICES_FILE, SWEDEN_GRID_FILE, TIME_FEATURES_FILE, 
    HOLIDAYS_FILE, WEATHER_DATA_FILE,
    PRICE_FEATURES, GRID_FEATURES, TIME_FEATURES,
    HOLIDAY_FEATURES, WEATHER_FEATURES, MARKET_FEATURES,
    LOSS_FUNCTION, WEIGHTED_LOSS_PARAMS,
    
    TREND_MODEL_DIR, PEAK_MODEL_DIR, VALLEY_MODEL_DIR, EVAL_DIR, TREND_EVAL_DIR,
    SPIKE_THRESHOLD_PERCENTILE, VALLEY_THRESHOLD_PERCENTILE,
    PEAK_CORE_FEATURES, VALLEY_CORE_FEATURES, SPIKE_CORE_FEATURES, SCALING_METHOD,
    
    
    PEAK_TCN_FILTERS, PEAK_TCN_KERNEL_SIZE, PEAK_TCN_DILATIONS,
    PEAK_TCN_NB_STACKS, PEAK_LEARNING_RATE, PEAK_EARLY_STOPPING_PATIENCE,
    PEAK_EPOCHS, PEAK_BATCH_SIZE,
    
    
    VALLEY_TCN_FILTERS, VALLEY_TCN_KERNEL_SIZE, VALLEY_TCN_DILATIONS,
    VALLEY_TCN_NB_STACKS, VALLEY_LEARNING_RATE, VALLEY_EARLY_STOPPING_PATIENCE,
    VALLEY_EPOCHS, VALLEY_BATCH_SIZE,
    
    
    TREND_EXOG_FEATURES,
    VALLEY_DETECTION_PARAMS,
    ROBUST_VALLEY_DETECTION_PARAMS,
    VALLEY_CLASS_WEIGHT_MULTIPLIER,
    FALSE_NEG_WEIGHT, FALSE_POS_WEIGHT,
    PRICE_LAG_HOURS
)

def configure_logging():
    """Configure logging for the model training."""
    # Create all necessary directories
    TREND_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PEAK_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    VALLEY_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Get command line arguments to determine which model we're training
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model', type=str, default='trend')
    known_args, _ = parser.parse_known_args()
    model_type = known_args.model

    # Choose the appropriate log file based on model type
    if model_type == 'trend':
        log_file = TREND_MODEL_DIR / "trend_model_training.log"
    elif model_type == 'peak':
        log_file = PEAK_MODEL_DIR / "peak_model_training.log"
    elif model_type == 'valley':
        log_file = VALLEY_MODEL_DIR / "valley_model_training.log"
    else:
        # Default to a general log file
        log_file = LOGS_DIR / "model_training.log"
        
    # Make sure the file exists
    if not log_file.exists():
        log_file.touch()
        
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )
    
    logging.info(f"Logging configured for {model_type} model. Log file: {log_file}")

def prepare_trend_data(use_extended_features=True):
    """Load and prepare data specifically for the SARIMAX trend model."""
    logging.info("Preparing data for SARIMAX trend model training...")
    
    # Load and merge the data
    df = load_and_merge_data()
    
    # Add lag features
    df = add_lag_features(df, TARGET_VARIABLE)
    
    # Add rolling window features
    df = add_rolling_features(df, TARGET_VARIABLE)
    
    # Add price spike indicators
    df = add_price_spike_indicators(df, TARGET_VARIABLE)
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to find timestamp column
        timestamp_col = None
        for col in ['timestamp', 'date', 'datetime']:
            if col in df.columns:
                timestamp_col = col
                break
        
        if timestamp_col:
            df = df.set_index(timestamp_col)
        else:
            # If no timestamp column found, use the current index but convert to datetime
            df.index = pd.to_datetime(df.index)
    
    # Make sure the data is sorted by index
    df = df.sort_index()
    
    # Create a copy for statsmodels
    model_df = df.copy()
    
    # Split the data into train, validation, and test sets
    train_size = int(len(model_df) * (1 - VALIDATION_SPLIT - TEST_SPLIT))
    val_size = int(len(model_df) * VALIDATION_SPLIT)
    
    train_df = model_df.iloc[:train_size].copy()
    val_df = model_df.iloc[train_size:train_size+val_size].copy()
    test_df = model_df.iloc[train_size+val_size:].copy()
    
    logging.info(f"Data split: Train {train_df.shape}, Validation {val_df.shape}, Test {test_df.shape}")
    
    # Extract target variable for training
    train_target = train_df[TARGET_VARIABLE]
    
    # Handle missing values in target
    if train_target.isnull().any():
        train_target = train_target.fillna(method='ffill')
    
    # Create data dictionary
    data = {
        'train_target': train_target,
        'val_df': val_df,
        'test_df': test_df,
        'full_df': model_df
    }
    
    return data

def add_peak_valley_labels(df, target_col=TARGET_VARIABLE):
    """Add binary labels for price spikes/peaks and valleys."""
    # Use the improved relative peak detection method by default
    logging.info("Using improved relative peak detection for peak/valley identification")
    
    # Apply relative peak detection
    df = detect_price_peaks(
        df, 
        target_col=target_col,
        window=24,            # 24 hour window for local context
        relative_height=0.2,  # Price must be 20% above local median
        distance=6,           # At least 6 hours between peaks
        prominence=0.15       # Peak must stand out by 15% of price range
    )
    
    # Use the new robust valley detection algorithm instead of derivative-based
    logging.info("Using new robust valley detection algorithm")
    df = detect_valleys_robust(
        df,
        target_col=target_col,
        **ROBUST_VALLEY_DETECTION_PARAMS  # Use parameters from config
    )
    
    # Use the robust valleys and relative peaks
    df['is_price_peak'] = df['is_price_peak_relative']
    df['is_price_valley'] = df['is_price_valley_robust']  # Use new robust valley detection
    
    # Log statistics
    total = len(df)
    num_peaks = df['is_price_peak'].sum()
    num_valleys = df['is_price_valley'].sum()
    
    logging.info(f"Detected {num_peaks} peaks ({num_peaks/total:.1%} of data) and {num_valleys} valleys ({num_valleys/total:.1%} of data)")
    
    # Traditional method just for reference in logs
    spike_threshold = df[target_col].quantile(SPIKE_THRESHOLD_PERCENTILE / 100)
    valley_threshold = df[target_col].quantile(VALLEY_THRESHOLD_PERCENTILE / 100)
    std_peaks = (df[target_col] >= spike_threshold).sum()
    
    logging.info(f"For reference - traditional method would detect {std_peaks} peaks ({std_peaks/total:.1%} of data)")
    logging.info(f"Traditional thresholds - spike: {spike_threshold:.2f}, valley: {valley_threshold:.2f}")
    
    # Save thresholds for later use in prediction
    thresholds = {
        "spike_threshold": float(spike_threshold),
        "valley_threshold": float(valley_threshold),
        "method": "robust",  # Now using robust method for valleys
        **ROBUST_VALLEY_DETECTION_PARAMS  # Include all valley detection parameters
    }
    
    return df, thresholds

def plot_valley_labels(df, output_dir=None, num_samples=3, days_per_sample=14):
    """
    Plot the price series with detected valleys highlighted to validate the labeling.
    
    Args:
        df: DataFrame with price data and valley labels
        output_dir: Directory to save plots (default: PLOTS_DIR / "valley_validation")
        num_samples: Number of time periods to sample for visualization
        days_per_sample: Number of days per sample
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from datetime import timedelta
    
    if output_dir is None:
        output_dir = PLOTS_DIR / "valley_validation"
        
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure we have the required columns
    if 'is_price_valley' not in df.columns or TARGET_VARIABLE not in df.columns:
        raise ValueError(f"DataFrame must include '{TARGET_VARIABLE}' and 'is_price_valley' columns")
    
    # Get time ranges for visualization (evenly distributed through the dataset)
    total_hours = len(df)
    hours_per_sample = 24 * days_per_sample
    
    if total_hours <= hours_per_sample:
        # If dataset is smaller than sample size, use the entire dataset
        sample_starts = [0]
    else:
        # Create evenly distributed sample start indices
        step = (total_hours - hours_per_sample) // (num_samples - 1) if num_samples > 1 else 0
        sample_starts = [i * step for i in range(num_samples)]
    
    # Create plots for each sample
    for i, start_idx in enumerate(sample_starts):
        end_idx = min(start_idx + hours_per_sample, total_hours)
        
        # Extract sample data
        sample = df.iloc[start_idx:end_idx].copy()
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Main price plot with valleys highlighted
        ax1.plot(sample.index, sample[TARGET_VARIABLE], 'b-', linewidth=2)
        
        # Highlight detected valleys
        valley_indices = sample[sample['is_price_valley'] == 1].index
        
        # Mark valleys with vertical spans
        for valley_idx in valley_indices:
            ax1.axvspan(valley_idx - timedelta(hours=1), 
                      valley_idx + timedelta(hours=1),
                      color='lightgreen', alpha=0.3)
        
        # Add points at valley locations
        valley_prices = sample.loc[valley_indices, TARGET_VARIABLE]
        ax1.scatter(valley_indices, valley_prices, color='red', s=100, marker='v')
        
        # Find local minima for comparison (simple algorithm)
        prices = sample[TARGET_VARIABLE].values
        window = 12  # 12-hour window for local minimum detection
        is_local_min = np.zeros(len(prices), dtype=bool)
        
        for j in range(window, len(prices) - window):
            if prices[j] == min(prices[j-window:j+window+1]):
                is_local_min[j] = True
        
        # Get local minima indices that aren't marked as valleys
        local_min_indices = np.where(is_local_min)[0]
        local_min_times = sample.index[local_min_indices]
        local_min_prices = sample.iloc[local_min_indices][TARGET_VARIABLE]
        
        # Mark local minima that aren't detected as valleys
        non_detected_mins = []
        for lm_time, lm_price in zip(local_min_times, local_min_prices):
            if lm_time not in valley_indices:
                non_detected_mins.append((lm_time, lm_price))
        
        if non_detected_mins:
            nd_times, nd_prices = zip(*non_detected_mins)
            ax1.scatter(nd_times, nd_prices, color='blue', s=80, marker='o', alpha=0.6)
        
        # Add title and labels
        start_date = sample.index[0].strftime('%Y-%m-%d')
        end_date = sample.index[-1].strftime('%Y-%m-%d')
        ax1.set_title(f'Valley Detection Validation ({start_date} to {end_date})')
        ax1.set_ylabel('Price (öre/kWh)')
        ax1.grid(True, alpha=0.3)
        
        # Second plot: Binary valley indicators
        ax2.step(sample.index, sample['is_price_valley'], 'g-', where='mid', linewidth=2)
        ax2.set_ylabel('Valley Label (0/1)')
        ax2.set_ylim(-0.1, 1.1)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gcf().autofmt_xdate()
        
        # Highlight weekends for context
        for j in range(len(sample) // 24):
            day_start = sample.index[0] + timedelta(days=j)
            if day_start.weekday() >= 5:  # Saturday or Sunday
                ax1.axvspan(day_start, day_start + timedelta(days=1), 
                           color='gray', alpha=0.1)
                ax2.axvspan(day_start, day_start + timedelta(days=1), 
                           color='gray', alpha=0.1)
        
        # Add descriptive statistics
        num_valleys = sample['is_price_valley'].sum()
        valleys_percent = (num_valleys / len(sample)) * 100
        ax1.text(0.02, 0.95, f'Detected valleys: {num_valleys} ({valleys_percent:.1f}% of data)', 
                transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / f"valley_validation_sample_{i+1}.png", dpi=300)
        plt.close()
    
    print(f"Generated {len(sample_starts)} valley validation plots in {output_dir}")
    return output_dir

def prepare_peak_valley_data(target_col='is_price_peak', production_mode=False):
    """Load and prepare data specifically for peak/valley detection model."""
    logging.info(f"Preparing data for {target_col} model training...")
    
    # Load and merge the data
    df = load_and_merge_data()
    
    # Add lag features
    df = add_lag_features(df, TARGET_VARIABLE)
    
    # Add rolling window features
    df = add_rolling_features(df, TARGET_VARIABLE)
    
    # Add binary labels for price spikes and valleys
    df, thresholds = add_peak_valley_labels(df, TARGET_VARIABLE)
    
    # Select features based on whether we're detecting peaks or valleys
    if target_col == 'is_price_peak':
        logging.info("Using peak-specific features for model training")
        features = PEAK_CORE_FEATURES.copy()
        # Peak models don't use SMOTE
        apply_smote = False
    else:  # valley detection
        logging.info("Using valley-specific features for model training")
        features = VALLEY_CORE_FEATURES.copy()
        # Valley models ALWAYS use SMOTE
        apply_smote = True
        logging.info("Valley model: Using SMOTE for class balancing")
    
    # Filter the DataFrame to include only selected features and the target
    selected_columns = features.copy()
    if target_col not in selected_columns:
        selected_columns.append(target_col)
    
    # Check if all selected features are available in the dataframe
    missing_features = [col for col in selected_columns if col not in df.columns]
    if missing_features:
        logging.warning(f"Missing features: {missing_features}")
        # Remove missing features from selection
        selected_columns = [col for col in selected_columns if col in df.columns]
    
    logging.info(f"Using {len(selected_columns)} features for {target_col} detection")
    
    df = df[selected_columns].copy()
    
    # Handle missing values
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype.kind in 'iuf':  # integer, unsigned int, or float
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    
    # Creating result dictionary to store data
    result = {}
    
    # In production mode, use all data for training
    if production_mode:
        logging.info("PRODUCTION MODE: Using ALL available data for training")
        combined_df = df.copy()  # Use all data
        logging.info(f"Combined data shape: {combined_df.shape}")
        logging.info(f"Target distribution in combined data: {combined_df[target_col].mean()*100:.2f}% positive samples")
        
        # Save full dataframe
        result['df'] = df
        result['train_df'] = combined_df
        
        # Scale the features (no scaling for target as it's binary)
        feature_scaler = get_scaler('standard')
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col != target_col]
        
        # Fit scaler on all data
        train_features = combined_df[feature_cols].values
        feature_scaler.fit(train_features)
        train_features_scaled = feature_scaler.transform(train_features)
        train_target = combined_df[target_col].values
        
        # Create sequences
        X_train, _ = create_sequences(
            train_features_scaled,
            LOOKBACK_WINDOW, 
            PREDICTION_HORIZON, 
            list(range(train_features_scaled.shape[1])),
            None
        )
        
        y_train = train_target[LOOKBACK_WINDOW:LOOKBACK_WINDOW+len(X_train)].reshape(-1, 1)
        
        # Apply SMOTE to balance the classes for valley models
        if apply_smote and target_col == 'is_price_valley':
            try:
                from imblearn.over_sampling import SMOTE
                # Reshape sequences for SMOTE (flatten the time dimension)
                n_samples, n_timesteps, n_features = X_train.shape
                X_reshaped = X_train.reshape(n_samples, n_timesteps * n_features)
                
                logging.info(f"Applying SMOTE for class balancing. Original class distribution: {np.mean(y_train)*100:.2f}% positive")
                
                # Apply SMOTE - aiming for approximately 1:3 ratio (25% positive samples)
                sampling_strategy = min(0.25, 3 * np.mean(y_train))
                smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X_reshaped, y_train.ravel())
                
                # Reshape back to sequences
                X_train = X_resampled.reshape(X_resampled.shape[0], n_timesteps, n_features)
                y_train = y_resampled.reshape(-1, 1)
                
                logging.info(f"After SMOTE: {X_train.shape[0]} samples with {np.mean(y_train)*100:.2f}% positive")
            except ImportError:
                logging.warning("imblearn not installed. SMOTE oversampling skipped.")
                logging.warning("Install with: pip install imbalanced-learn")
            except Exception as e:
                logging.error(f"Error applying SMOTE: {e}")
                logging.warning("Proceeding with original imbalanced data")
        
        # Store in result
        result['X_train'] = X_train
        result['y_train'] = y_train
        result['feature_scaler'] = feature_scaler
        result['feature_names'] = feature_cols
        result['thresholds'] = thresholds
        
        # Don't add dummy validation/test sets in production mode
        # The training function will handle this appropriately
        
        logging.info(f"Prepared training data: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
        return result
    
    # If not in production mode, proceed with regular train/val/test split
    # Split the data into train, validation, and test sets
    train_size = int(len(df) * (1 - VALIDATION_SPLIT - TEST_SPLIT))
    val_size = int(len(df) * VALIDATION_SPLIT)
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]
    
    logging.info(f"Data split: Train {train_df.shape}, Validation {val_df.shape}, Test {test_df.shape}")
    logging.info(f"Target distribution in train: {train_df[target_col].mean()*100:.2f}% positive samples")
    logging.info(f"Target distribution in val: {val_df[target_col].mean()*100:.2f}% positive samples")
    logging.info(f"Target distribution in test: {test_df[target_col].mean()*100:.2f}% positive samples")
    
    # Scale the features (no scaling for target as it's binary)
    feature_scaler = get_scaler('standard')
    
    # Fit and transform the feature data
    feature_cols = [col for col in df.columns if col != target_col]
    train_features = train_df[feature_cols].values
    
    feature_scaler.fit(train_features)
    
    train_features_scaled = feature_scaler.transform(train_features)
    val_features_scaled = feature_scaler.transform(val_df[feature_cols].values)
    test_features_scaled = feature_scaler.transform(test_df[feature_cols].values)
    
    # Get target values
    train_target = train_df[target_col].values
    val_target = val_df[target_col].values
    test_target = test_df[target_col].values
    
    # Create sequences for TCN model
    X_train, y_train = create_sequences(
        train_features_scaled,
        LOOKBACK_WINDOW, 
        PREDICTION_HORIZON, 
        list(range(train_features_scaled.shape[1])),
        None
    )
    
    y_train = train_target[LOOKBACK_WINDOW:LOOKBACK_WINDOW+len(X_train)].reshape(-1, 1)
    
    # Apply SMOTE to balance the classes for valley models
    if apply_smote and target_col == 'is_price_valley':
        try:
            from imblearn.over_sampling import SMOTE
            # Reshape sequences for SMOTE (flatten the time dimension)
            n_samples, n_timesteps, n_features = X_train.shape
            X_reshaped = X_train.reshape(n_samples, n_timesteps * n_features)
            
            logging.info(f"Applying SMOTE for class balancing. Original class distribution: {np.mean(y_train)*100:.2f}% positive")
            
            # Apply SMOTE - aiming for approximately 1:3 ratio (25% positive samples)
            sampling_strategy = min(0.25, 3 * np.mean(y_train))
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_reshaped, y_train.ravel())
            
            # Reshape back to sequences
            X_train = X_resampled.reshape(X_resampled.shape[0], n_timesteps, n_features)
            y_train = y_resampled.reshape(-1, 1)
            
            logging.info(f"After SMOTE: {X_train.shape[0]} samples with {np.mean(y_train)*100:.2f}% positive")
        except ImportError:
            logging.warning("imblearn not installed. SMOTE oversampling skipped.")
            logging.warning("Install with: pip install imbalanced-learn")
        except Exception as e:
            logging.error(f"Error applying SMOTE: {e}")
            logging.warning("Proceeding with original imbalanced data")
    
    X_val, y_val = create_sequences(
        val_features_scaled,
        LOOKBACK_WINDOW, 
        PREDICTION_HORIZON, 
        list(range(val_features_scaled.shape[1])),
        None
    )
    
    y_val = val_target[LOOKBACK_WINDOW:LOOKBACK_WINDOW+len(X_val)].reshape(-1, 1)
    
    X_test, y_test = create_sequences(
        test_features_scaled,
        LOOKBACK_WINDOW, 
        PREDICTION_HORIZON, 
        list(range(test_features_scaled.shape[1])),
        None
    )
    
    y_test = test_target[LOOKBACK_WINDOW:LOOKBACK_WINDOW+len(X_test)].reshape(-1, 1)
    
    logging.info(f"Sequence shapes: X_train {X_train.shape}, y_train {y_train.shape}")
    
    # Store all data in result dictionary
    result = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'feature_scaler': feature_scaler,
        'feature_names': feature_cols,
        'df': df,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'thresholds': thresholds
    }
    
    return result

def moving_average_filter(data, window=12):
    """
    Apply a simple moving average filter to smooth out high-frequency variations.
    """
    return pd.Series(data).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values

def exponential_smooth(predictions, alpha=0.3):
    """
    Apply exponential smoothing to predictions to reduce volatility.
    
    Args:
        predictions: Array of predictions to smooth
        alpha: Smoothing factor (0-1), lower values mean more smoothing
        
    Returns:
        Smoothed predictions array
    """
    smoothed = [predictions[0]]
    for i in range(1, len(predictions)):
        smoothed.append(alpha * predictions[i] + (1-alpha) * smoothed[i-1])
    return np.array(smoothed)

def median_filter(data, window=5):
    """
    Apply a median filter to remove outliers.
    """
    return medfilt(data, kernel_size=window)

def savitzky_golay_filter(data, window=11, polyorder=2):
    """
    Apply Savitzky-Golay filter for smoothing while preserving trends.
    """
    # Ensure window is odd
    if window % 2 == 0:
        window += 1
        
    # Ensure window is less than data length
    if window >= len(data):
        window = min(11, len(data) - 1)
        if window % 2 == 0:
            window -= 1
            
    return savgol_filter(data, window, polyorder)

def daily_averaging(data, timestamps):
    """
    Perform daily averaging on the forecast data for aggressive smoothing.
    Each hour of the day gets the average value for that day.
    
    Args:
        data: Array of predictions to smooth
        timestamps: DatetimeIndex corresponding to the data points
        
    Returns:
        Daily averaged predictions array
    """
    # Convert to pandas Series with timestamps
    temp_df = pd.DataFrame({'value': data}, index=timestamps)
    
    # Group by date and calculate daily average
    daily_means = temp_df.groupby(temp_df.index.date)['value'].mean()
    
    # Map each timestamp to its corresponding daily average
    smoothed = np.array([daily_means[ts.date()] for ts in timestamps])
    
    return smoothed

def weekly_averaging(data, timestamps):
    """
    Perform weekly averaging of the forecast data for very aggressive smoothing.
    Each hour gets the average value for its day of week and hour of day across
    all weeks in the dataset.
    
    Args:
        data: Array of predictions to smooth
        timestamps: DatetimeIndex corresponding to the data points
        
    Returns:
        Weekly pattern averaged predictions array
    """
    # Convert to pandas Series with timestamps
    temp_df = pd.DataFrame({'value': data}, index=timestamps)
    
    # Group by day of week and hour of day
    temp_df['dayofweek'] = temp_df.index.dayofweek
    temp_df['hour'] = temp_df.index.hour
    
    # Calculate average for each hour-day combination
    pattern_means = temp_df.groupby(['dayofweek', 'hour'])['value'].mean()
    
    # Map each timestamp to its corresponding pattern average
    smoothed = np.array([pattern_means[(ts.dayofweek, ts.hour)] for ts in timestamps])
    
    return smoothed

def adaptive_trend_smoothing(data, timestamps, smoothing_level='daily'):
    """
    Apply adaptive smoothing based on the desired smoothing level.
    
    Args:
        data: Array of predictions to smooth
        timestamps: DatetimeIndex corresponding to the data points
        smoothing_level: Level of smoothing ('light', 'medium', 'heavy', 'daily', 'weekly')
        
    Returns:
        Smoothed predictions array
    """
    if smoothing_level == 'light':
        # Light smoothing with exponential filter
        return exponential_smooth(data, alpha=0.3)
    
    elif smoothing_level == 'medium':
        # Medium smoothing with combined filters
        smoothed = exponential_smooth(data, alpha=0.1)
        smoothed = median_filter(smoothed, window=5)
        return savitzky_golay_filter(smoothed, window=11, polyorder=2)
    
    elif smoothing_level == 'heavy':
        # Heavy smoothing with larger windows
        smoothed = exponential_smooth(data, alpha=0.05)
        smoothed = median_filter(smoothed, window=7)
        return savitzky_golay_filter(smoothed, window=23, polyorder=2)
    
    elif smoothing_level == 'daily':
        # Daily average - very aggressive
        return daily_averaging(data, timestamps)
    
    elif smoothing_level == 'weekly':
        # Weekly pattern average - extremely aggressive
        return weekly_averaging(data, timestamps)
    
    else:
        # Default to medium smoothing
        smoothed = exponential_smooth(data, alpha=0.1)
        smoothed = median_filter(smoothed, window=5)
        return savitzky_golay_filter(smoothed, window=11, polyorder=2)

def train_trend_model(data, production_mode=False):
    """
    Train a time series model for price trend prediction using Gradient Boosting.
    
    Args:
        data: Dictionary with training data
        production_mode: If True, use all available data for training
        
    Returns:
        Trained model and evaluation metrics
    """
    logging.info("Starting Gradient Boosting trend model training...")
    
    # Get the data
    train_target = data['train_target']
    val_df = data['val_df']
    test_df = data['test_df'] 
    full_df = data['full_df']
    
    # In production mode, use combined dataset including train+val+test
    if production_mode:
        combined_target = data['combined_target']
        logging.info(f"Production mode: Using ALL available data: {len(combined_target)} samples from {combined_target.index[0]} to {combined_target.index[-1]}")
        # Calculate actual years of data
        years_of_data = (combined_target.index[-1] - combined_target.index[0]).days / 365.25
        logging.info(f"Total data span: {years_of_data:.1f} years")
    

    
    # Setup specific logging for debugging
    log_file = TREND_MODEL_DIR / "trend_model_training_debug.log"
    # make file if it doesn't exist
    if not log_file.exists():
        log_file.touch()
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler)
    
    # Step 1: Data preparation and inspection
    logging.info("=== STEP 1: DATA PREPARATION ===")
    logging.info(f"Target variable: {TARGET_VARIABLE}")
    
    if production_mode:
        logging.info(f"Training data shape: {combined_target.shape}")
        # Use all data in production mode
        train_target_trimmed = combined_target
    else:
        logging.info(f"Training data shape: {train_target.shape}")
        # Take a reasonable training window - gradient boosting can handle more data efficiently
        max_train_size = 3*365*24  # 3 years of hourly data by default
        if len(train_target) > max_train_size:
            logging.info(f"Limiting training data to last {max_train_size/365/24:.1f} years for balanced training")
            train_target_trimmed = train_target.iloc[-max_train_size:]
        else:
            train_target_trimmed = train_target
        
    logging.info(f"Training data length: {len(train_target_trimmed)}")

    # Step 2: Check for exogenous features
    logging.info("=== STEP 2: FEATURE INSPECTION ===")
    exog_features = TREND_EXOG_FEATURES
    logging.info(f"Potential features: {exog_features}")
    
    # Check which features are available
    available_features = [col for col in exog_features if col in full_df.columns]
    missing_features = [col for col in exog_features if col not in full_df.columns]
    
    if missing_features:
        logging.warning(f"Missing {len(missing_features)} features: {missing_features}")
    logging.info(f"Available features: {available_features}")
    
    # Step 3: Prepare aligned training data
    logging.info("=== STEP 3: ALIGNED DATA PREPARATION ===")
    
    # Get data for the training timestamps
    train_indices = train_target_trimmed.index
    
    if production_mode:
        # In production mode, use the entire dataset
        train_df = full_df.copy()
    else:
        # Otherwise, use only the training portion
        train_df = full_df.loc[full_df.index.isin(train_indices)].copy()
        
    logging.info(f"Extracted training data shape: {train_df.shape}")
    
    # Prepare feature matrix
    X_train = train_df[available_features].copy()
    y_train = train_df[TARGET_VARIABLE].copy()
    
    # Fill missing values
    for col in X_train.columns:
        if X_train[col].isnull().any():
            if X_train[col].dtype.kind in 'iuf':  # integer, unsigned int, or float
                X_train[col] = X_train[col].fillna(X_train[col].median())
            else:
                X_train[col] = X_train[col].fillna(X_train[col].mode()[0])
    
    logging.info(f"Prepared training data: X_train {X_train.shape}, y_train {y_train.shape}")
    
    # Step 4: Add time-based features optimal for Gradient Boosting
    logging.info("=== STEP 4: TIME FEATURE ENGINEERING ===")
    
    # Function to add additional time features to the feature matrix
    def add_gb_time_features(df):
        """Add time features specifically useful for Gradient Boosting models."""
        # Copy the dataframe to avoid modifying the original
        result = df.copy()
        
        # Extract datetime components
        if hasattr(result, 'index') and isinstance(result.index, pd.DatetimeIndex):
            idx = result.index
        else:
            # If not a DatetimeIndex, try to create one
            if 'datetime' in result.columns:
                idx = pd.DatetimeIndex(result['datetime'])
            else:
                raise ValueError("DataFrame must have DatetimeIndex or datetime column")
        
        # Basic time components
        result['hour'] = idx.hour
        result['day'] = idx.day
        result['month'] = idx.month
        result['dayofweek'] = idx.dayofweek
        result['quarter'] = idx.quarter
        
        # Cyclical encoding of time features (helps GB models detect periodic patterns)
        result['hour_sin'] = np.sin(2 * np.pi * result['hour']/24)
        result['hour_cos'] = np.cos(2 * np.pi * result['hour']/24)
        result['day_sin'] = np.sin(2 * np.pi * result['day']/31)
        result['day_cos'] = np.cos(2 * np.pi * result['day']/31)
        result['month_sin'] = np.sin(2 * np.pi * result['month']/12)
        result['month_cos'] = np.cos(2 * np.pi * result['month']/12)
        result['dayofweek_sin'] = np.sin(2 * np.pi * result['dayofweek']/7)
        result['dayofweek_cos'] = np.cos(2 * np.pi * result['dayofweek']/7)
        
        # Special time flags
        result['is_weekend'] = (result['dayofweek'] >= 5).astype(int)
        result['is_business_hour'] = ((result['hour'] >= 8) & (result['hour'] <= 18) & 
                                    (result['dayofweek'] < 5)).astype(int)
        result['is_morning_peak'] = ((result['hour'] >= 7) & (result['hour'] <= 9)).astype(int)
        result['is_evening_peak'] = ((result['hour'] >= 17) & (result['hour'] <= 20)).astype(int)
        
        # Drop original components that aren't needed anymore
        result.drop(['hour', 'day', 'month', 'dayofweek', 'quarter'], axis=1, inplace=True, errors='ignore')
        
        return result
    
    # Add time features to training data
    X_train = add_gb_time_features(X_train)
    logging.info(f"Added time features, new X_train shape: {X_train.shape}")
    
    # Step 5: Feature weighting/importance
    logging.info("=== STEP 5: FEATURE IMPORTANCE WEIGHTING ===")
    
    # Calculate correlation with target for feature importance logging
    feature_correlations = {}
    for col in X_train.columns:
        corr = np.corrcoef(X_train[col], y_train)[0, 1]
        feature_correlations[col] = corr
    
    # Group features by type for better analysis
    feature_groups = {
        "core_price": ["price_168h_avg", "hour_avg_price", "price_24h_avg"],
        "consumption": ["powerConsumptionTotal", "temperature_2m"],
        "production": ["powerProductionTotal", "hydro", "nuclear", "wind"],
        "trade": ["powerImportTotal", "powerExportTotal"],
        "market": ["Gas_Price"],
        "weather": ["wind_speed_100m", "cloud_cover"],
        "time": [col for col in X_train.columns if any(x in col for x in ['hour', 'day', 'month', 'weekend', 'peak'])]
    }
    
    # Log correlations by group for better analysis
    for group, features in feature_groups.items():
        available_group_features = [f for f in features if f in X_train.columns]
        if available_group_features:
            logging.info(f"Feature group '{group}' correlations:")
            for feature in available_group_features:
                if feature in feature_correlations:
                    corr = feature_correlations.get(feature, 0)
                    logging.info(f"  {feature}: {corr:.3f}")
    
    # Sort features by absolute correlation for logging
    sorted_corrs = sorted(feature_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    logging.info("Features sorted by correlation with target price:")
    for feature, corr in sorted_corrs:
        logging.info(f"  {feature}: {corr:.3f}")
    
    # Step 6: Train Gradient Boosting model
    logging.info("=== STEP 6: GRADIENT BOOSTING MODEL TRAINING ===")
    
    # Start timing
    start_time = time.time()
    
    # Prepare validation data
    X_val = val_df[available_features].copy()
    y_val = val_df[TARGET_VARIABLE].copy()
    
    # Fill missing values in validation data
    for col in X_val.columns:
        if X_val[col].isnull().any():
            if X_val[col].dtype.kind in 'iuf':
                X_val[col] = X_val[col].fillna(X_val[col].median())
            else:
                X_val[col] = X_val[col].fillna(X_val[col].mode()[0])
    
    # Add time features to validation data
    X_val = add_gb_time_features(X_val)
    
    # Ensure X_train and X_val have the same columns
    common_columns = list(set(X_train.columns) & set(X_val.columns))
    X_train = X_train[common_columns]
    X_val = X_val[common_columns]
    
    logging.info(f"Training with {X_train.shape[1]} features and {len(y_train)} samples")
    logging.info(f"Validation data: {X_val.shape}")
    
    # Define model hyperparameters - tuned for smooth trend prediction
    """
    params = {
        'objective': 'reg:pseudohubererror',  # Huber loss is less sensitive to outliers
        'learning_rate': 0.05,                # Lower learning rate for smoother convergence
        'max_depth': 8,                       # Reduce max depth to prevent overfitting to extreme values
        'min_child_weight': 5,                # Higher values prevent specializing on outliers
        'gamma': 0.2,                         # Increase pruning to reduce complexity
        'subsample': 0.7,                     # Reduce overfitting on specific price patterns
        'colsample_bytree': 0.7,              # Sample fewer features for more stable results
        'reg_alpha': 2.0,                     # Stronger L1 regularization for feature selection
        'reg_lambda': 10.0,                   # Much stronger L2 regularization for smoother predictions
        'huber_slope': 0.9,                   # For pseudohuber: control outlier sensitivity (high = less sensitive)
        'random_state': 42,
        'n_jobs': -1                          # Use all available cores
    }
    """

    # Uncomment to use our custom objective function that penalizes extreme values
    # Note: When using custom objective, we set objective to None and pass the function directly
    # Custom objective for smoother predictions with penalty for extremes
    params = {
        'objective': None,  # Will use custom objective
        'learning_rate': 0.05,
        'max_depth': 20,
        'min_child_weight': 5,
        'gamma': 0.2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 3.0,
        'reg_lambda': 10.0,
        'random_state': 42,
        'n_jobs': -1
    }
    custom_obj = xgb_smooth_trend_objective  # Reference our custom objective
    
    logging.info(f"XGBoost parameters: {params}")
    print(f"\nTraining XGBoost model with {len(y_train)} samples and {X_train.shape[1]} features")
    
    # Create DMatrix for XGBoost (faster training)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Define evaluation list
    eval_list = [(dtrain, 'train'), (dval, 'validation')]
    
    # Train model with early stopping
    num_boost_round = 1000
    early_stopping_rounds = 50
    
    # Initialize progress bar
    print(f"Training for up to {num_boost_round} boosting rounds with early stopping...")
    pbar = tqdm(total=num_boost_round, desc="XGBoost Training", unit="rounds")
    
    # Create a proper callback class for progress tracking
    class TQDMProgressCallback(TrainingCallback):
        """Custom callback for updating tqdm progress bar during XGBoost training."""
        def __init__(self, progress_bar):
            self.progress_bar = progress_bar
            
        def after_iteration(self, model, epoch, evals_log):
            """Update progress bar after each iteration."""
            self.progress_bar.update(1)
            return False
    
    # Check if we're using a custom objective
    if 'objective' in params and params['objective'] is None and 'custom_obj' in locals():
        # Train with custom objective
        model = xgb.train(
            params, 
            dtrain, 
            num_boost_round=num_boost_round,
            evals=eval_list,
            early_stopping_rounds=early_stopping_rounds,
            obj=custom_obj,  # Pass the custom objective function
            callbacks=[TQDMProgressCallback(pbar)],
            verbose_eval=False
        )
        logging.info("Trained model with custom objective function")
    else:
        # Train with standard objective
        model = xgb.train(
            params, 
            dtrain, 
            num_boost_round=num_boost_round,
            evals=eval_list,
            early_stopping_rounds=early_stopping_rounds,
            callbacks=[TQDMProgressCallback(pbar)],
            verbose_eval=False
        )
    
    # Close progress bar
    pbar.close()
    
    # Calculate training time
    elapsed_time = time.time() - start_time
    logging.info(f"Training completed in {timedelta(seconds=int(elapsed_time))}")
    logging.info(f"Best iteration: {model.best_iteration}")
    logging.info(f"Best score: {model.best_score}")
    
    print(f"\nXGBoost training completed in {timedelta(seconds=int(elapsed_time))}")
    print(f"Best iteration: {model.best_iteration}, Best score: {model.best_score:.4f}")
    
    # Step 7: Feature importance analysis
    logging.info("=== STEP 7: FEATURE IMPORTANCE ANALYSIS ===")
    
    # Get feature importance
    importance_scores = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'Feature': list(importance_scores.keys()),
        'Importance': list(importance_scores.values())
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Log top 20 features
    logging.info("Top 20 important features:")
    for i, (feature, importance) in enumerate(zip(importance_df['Feature'].head(20), 
                                                importance_df['Importance'].head(20))):
        logging.info(f"{i+1}. {feature}: {importance:.2f}")
    
    # Step 8: Validation predictions with smoothing
    logging.info("=== STEP 8: VALIDATION PREDICTIONS WITH SMOOTHING ===")
    
    # Make predictions on validation set
    val_dmatrix = xgb.DMatrix(X_val)
    val_pred_raw = model.predict(val_dmatrix)
    
    # Apply smoothing pipeline
    logging.info("Applying smoothing pipeline to predictions")
    
    # Use the heavy smoothing level rather than daily averaging
    # Choose from: 'light', 'medium', 'heavy', 'daily', 'weekly'
    smoothing_level = 'daily'  # Changed from 'daily' to 'heavy' since the model itself should be smoother
    logging.info(f"Using {smoothing_level} smoothing level")
    
    val_pred_values = adaptive_trend_smoothing(val_pred_raw, X_val.index, smoothing_level=smoothing_level)
    
    # Store the smoothing parameters for later use in evaluation
    smoothing_params = {
        'smoothing_level': smoothing_level,
        'exponential_alpha': 0.05,
        'median_window': 11,
        'savgol_window': 31,  # Increased from 11 to 23 for smoother trend
        'savgol_polyorder': 2
    }
    
    # Calculate validation metrics
    val_actual = y_val.values
    val_mae = np.mean(np.abs(val_actual - val_pred_values))
    val_rmse = np.sqrt(np.mean((val_actual - val_pred_values) ** 2))
    
    # Calculate direction accuracy
    val_actual_diff = np.diff(val_actual)
    val_pred_diff = np.diff(val_pred_values)
    val_direction_accuracy = np.mean((val_actual_diff > 0) == (val_pred_diff > 0))
    
    val_metrics = {
        'val_mae': float(val_mae),
        'val_rmse': float(val_rmse),
        'val_direction_accuracy': float(val_direction_accuracy)
    }
    
    logging.info(f"Validation metrics: MAE={val_mae:.2f}, RMSE={val_rmse:.2f}, Direction Accuracy={val_direction_accuracy:.2f}")
    
    # Step 9: Save model and metadata
    logging.info("=== STEP 9: MODEL SAVING ===")
    
    # Create directory if it doesn't exist
    os.makedirs(TREND_MODEL_DIR, exist_ok=True)
    
    # Save model
    import pickle
    model_path = TREND_MODEL_DIR / "best_trend_model.pkl"
    try:
        # First ensure parent directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model using pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Verify the file was actually created
        if model_path.exists():
            logging.info(f"Successfully saved model to {model_path} ({os.path.getsize(str(model_path))} bytes)")
        else:
            logging.error(f"Failed to save model: File {model_path} was not created")
    except Exception as e:
        logging.error(f"Error saving model to {model_path}: {e}")
        raise
    
    # Save feature names and their exact order
    feature_names_path = TREND_MODEL_DIR / "feature_names.json"
    try:
        with open(feature_names_path, 'w') as f:
            json.dump(list(X_train.columns), f)
        logging.info(f"Saved feature names to {feature_names_path}")
    except Exception as e:
        logging.error(f"Error saving feature names to {feature_names_path}: {e}")
    
    # Save model parameters and metadata
    model_params = {
        'model_type': 'XGBoost',
        'xgb_params': params,
        'best_iteration': model.best_iteration,
        'best_score': float(model.best_score),
        'feature_count': X_train.shape[1],
        'training_samples': len(y_train),
        'features': list(X_train.columns),
        'important_features': importance_df['Feature'].head(20).tolist(),
        'feature_importance': {f: float(i) for f, i in zip(importance_df['Feature'].head(20), 
                                                         importance_df['Importance'].head(20))},
        'smoothing': {
            'smoothing_level': smoothing_level,
            'exponential_alpha': 0.05,
            'median_window': 11,
            'savgol_window': 23,
            'savgol_polyorder': 2
        },
        'validation_metrics': val_metrics,
        'training_time_seconds': int(elapsed_time),
        'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'fitted_on': str(X_train.index[0]) + " to " + str(X_train.index[-1])
    }
    
    params_path = TREND_MODEL_DIR / "trend_model_params.json"
    try:
        with open(params_path, 'w') as f:
            json.dump(model_params, f, indent=4)
        logging.info(f"Saved model parameters to {params_path}")
    except Exception as e:
        logging.error(f"Error saving model parameters to {params_path}: {e}")
    
    # Save feature importance plot
    try:
        # Ensure plots directory exists
        plots_dir = PLOTS_DIR / "evaluation" / "trend"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        importance_plot = importance_df.head(20).plot(kind='barh', x='Feature', y='Importance', legend=False)
        plt.title('Feature Importance (Top 20)')
        plt.tight_layout()
        plt.savefig(plots_dir / "feature_importance.png")
        plt.close()
        logging.info(f"Saved feature importance plot to {plots_dir / 'feature_importance.png'}")
    except Exception as e:
        logging.error(f"Error saving feature importance plot: {e}")
    
    # Plot validation predictions
    try:
        plt.figure(figsize=(15, 8))
        plt.plot(X_val.index, val_actual, label='Actual')
        plt.plot(X_val.index, val_pred_values, label='XGBoost Forecast')
        plt.title('Validation Set Prediction vs Actual')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / "validation_prediction.png")
        plt.close()
        logging.info(f"Saved validation prediction plot to {plots_dir / 'validation_prediction.png'}")
    except Exception as e:
        logging.error(f"Error saving validation prediction plot: {e}")
    
    # Plot a sample week for better visibility
    if len(val_actual) >= 24*7:
        try:
            plt.figure(figsize=(15, 8))
            plt.plot(X_val.index[:24*7], val_actual[:24*7], label='Actual')
            plt.plot(X_val.index[:24*7], val_pred_values[:24*7], label='XGBoost Forecast')
            plt.title('Validation Set Prediction vs Actual (First Week)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(plots_dir / "validation_prediction_week.png")
            plt.close()
            logging.info(f"Saved weekly validation plot to {plots_dir / 'validation_prediction_week.png'}")
        except Exception as e:
            logging.error(f"Error saving weekly validation plot: {e}")
    
    logging.info("XGBoost trend model training completed")
    
    # Return the trained model and metrics
    return model, val_metrics

def build_tcn_model(input_shape, output_dim=PREDICTION_HORIZON, is_binary=False, model_type='trend'):
    """
    Build a TCN model for time series prediction.
    
    Args:
        input_shape: Shape of input sequences (lookback, num_features)
        output_dim: Number of output dimensions (prediction horizon)
        is_binary: Whether this is a binary classification model
        model_type: Type of model ('peak' or 'valley')
        
    Returns:
        Compiled model
    """
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # Get model-specific parameters
    if model_type == 'peak':
        tcn_filters = PEAK_TCN_FILTERS
        tcn_kernel_size = PEAK_TCN_KERNEL_SIZE
        tcn_dilations = PEAK_TCN_DILATIONS
        tcn_nb_stacks = PEAK_TCN_NB_STACKS
        learning_rate = PEAK_LEARNING_RATE
    elif model_type == 'valley':
        tcn_filters = VALLEY_TCN_FILTERS
        tcn_kernel_size = VALLEY_TCN_KERNEL_SIZE
        tcn_dilations = VALLEY_TCN_DILATIONS
        tcn_nb_stacks = VALLEY_TCN_NB_STACKS
        learning_rate = VALLEY_LEARNING_RATE
    else:
        # Default to peak model parameters if not specified
        tcn_filters = PEAK_TCN_FILTERS
        tcn_kernel_size = PEAK_TCN_KERNEL_SIZE
        tcn_dilations = PEAK_TCN_DILATIONS
        tcn_nb_stacks = PEAK_TCN_NB_STACKS
        learning_rate = PEAK_LEARNING_RATE
    
    # TCN layer
    tcn_layer = TCN(
        nb_filters=tcn_filters,
        kernel_size=tcn_kernel_size,
        nb_stacks=tcn_nb_stacks,
        dilations=tcn_dilations,
        padding='causal',
        use_skip_connections=True,
        dropout_rate=DROPOUT_RATE,
        return_sequences=False if model_type != 'valley' else True,  # For valley model, return sequences for attention
        activation='relu',
        kernel_initializer='he_normal',
        use_batch_norm=True,
        use_layer_norm=False
    )(input_layer)
    
    # For valley model, add attention mechanism to focus on relevant parts of the sequence
    if model_type == 'valley':
        # Apply attention mechanism
        # Simple self-attention implementation
        attention = Dense(1, activation='tanh')(tcn_layer)  # (batch, seq_len, 1)
        attention = Flatten()(attention)  # (batch, seq_len)
        attention_weights = Activation('softmax')(attention)  # (batch, seq_len)
        
        # Apply attention weights to TCN output
        expanded_weights = Reshape((input_shape[0], 1))(attention_weights)  # (batch, seq_len, 1)
        weighted_output = Multiply()([tcn_layer, expanded_weights])  # (batch, seq_len, nb_filters)
        
        # Replace Lambda layer with a more serialization-friendly approach using TensorFlow ops directly
        # Use the custom GlobalSumPooling1D layer defined at module level
        x = GlobalSumPooling1D()(weighted_output)
        
        # Add extra dense layer with strong regularization
        x = Dense(tcn_filters, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(DROPOUT_RATE * 1.5)(x)  # Increase dropout for valley model
    else:
        # For other models, standard architecture
        x = Dropout(DROPOUT_RATE)(tcn_layer)
    
    # Output layer
    if is_binary:
        # For binary classification (peak/valley detection)
        output_activation = 'sigmoid'
        output_dim = 1  # Always 1 for binary classification
    else:
        # For regression (trend prediction)
        output_activation = 'linear'
    
    # Add final dense layers
    if model_type == 'valley':
        # More specialized for valley detection with extra layer
        x = Dense(max(32, tcn_filters // 2), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(DROPOUT_RATE)(x)
        outputs = Dense(output_dim, activation=output_activation)(x)
    else:
        # Simpler architecture for other models with regularization
        outputs = Dense(output_dim, activation=output_activation, 
                        kernel_regularizer=l1_l2(l1=L1_REG, l2=L2_REG))(x)
    
    # Create and compile model
    model = Model(inputs=input_layer, outputs=outputs)
    
    # Logging model architecture
    logging.info(f"Built {model_type} model with input shape {input_shape}")
    
    return model

def get_focal_loss(alpha=0.75, gamma=3.0):
    """
    Create a focal loss function for imbalanced classification.
    
    Args:
        alpha: Weighting factor for the rare class (higher value gives more weight to valley class)
        gamma: Focusing parameter to down-weight easy examples (higher value focuses more on hard examples)
        
    Returns:
        Focal loss function
    """
    def focal_loss(y_true, y_pred):
        # Clip the prediction value to prevent extreme cases
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        
        # Apply class weighting
        class_weights = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        # Calculate focal loss with modulating factor based on how correct the prediction is
        # This down-weights easy examples and focuses on hard ones
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        modulating_factor = K.pow(1. - p_t, gamma)
        
        # Combine all factors
        loss = class_weights * modulating_factor * cross_entropy
        
        # Return mean loss
        return K.mean(loss)
    
    # Include parameters in the function name for better tracking
    focal_loss.__name__ = f'focal_loss_a{alpha}_g{gamma}'
    return focal_loss

def get_recall_oriented_loss(false_neg_weight=8.0, false_pos_weight=1.5):
    """
    Create a custom loss function that prioritizes recall over precision.
    This is an enhanced version with higher false negative penalty.
    
    Args:
        false_neg_weight: Weight for false negatives (missed valleys) - higher means better recall
        false_pos_weight: Weight for false positives (false alarms) - lower means more permissive predictions
        
    Returns:
        Recall-oriented loss function
    """
    def recall_oriented_loss(y_true, y_pred):
        # Clip predictions for numerical stability
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1 - epsilon)
        
        # Calculate binary cross entropy
        bce = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        
        # Apply weights based on error type:
        # - False Negatives (y_true=1, y_pred=0): Heavily penalized for better recall
        # - False Positives (y_true=0, y_pred=1): Less penalized to allow more predictions
        weights = y_true * false_neg_weight * (1 - y_pred) + (1 - y_true) * false_pos_weight * y_pred
        
        # Combine weights with base loss
        weighted_loss = weights * bce
        
        # Return mean loss
        return K.mean(weighted_loss)
    
    # Include parameters in function name for better tracking
    recall_oriented_loss.__name__ = f'recall_loss_fn{false_neg_weight}_fp{false_pos_weight}'
    return recall_oriented_loss

def train_peak_valley_model(data, model_type, production_mode=False):
    """
    Train the peak or valley detection model.
    
    Args:
        data: Dictionary with training data
        model_type: 'peak' or 'valley'
        production_mode: If True, use all available data for training
        
    Returns:
        Trained model and training history
    """
    logging.info(f"Training {model_type} detection model...")
    
    if production_mode:
        logging.info("PRODUCTION MODE: Using all available data for training")
    
    # Get the data
    X_train = data['X_train']
    y_train = data['y_train']
    
    # Only access validation data when not in production mode
    if not production_mode:
        X_val = data['X_val']
        y_val = data['y_val']
    
    # Determine the model directory and parameters
    if model_type == 'peak':
        model_dir = PEAK_MODEL_DIR
        model_name = 'peak_model'
        epochs = PEAK_EPOCHS
        batch_size = PEAK_BATCH_SIZE
        early_stopping_patience = PEAK_EARLY_STOPPING_PATIENCE
    else:  # valley
        model_dir = VALLEY_MODEL_DIR
        model_name = 'valley_model'
        epochs = VALLEY_EPOCHS
        batch_size = VALLEY_BATCH_SIZE
        early_stopping_patience = VALLEY_EARLY_STOPPING_PATIENCE
        # Valley models ALWAYS use recall-oriented loss
        logging.info("Valley model: Using recall-oriented loss for better valley detection")
        
        # Generate valley label validation plots when training valley model
        logging.info("Generating valley label validation plots...")
        validation_plots_dir = model_dir / "valley_validation"
        validation_plots_dir.mkdir(parents=True, exist_ok=True)
        plot_valley_labels(data['df'], output_dir=validation_plots_dir, num_samples=5, days_per_sample=14)
    
    # Save model artifacts early - before training begins
    feature_names = data['feature_names']
    feature_scaler = data['feature_scaler']
    thresholds = data['thresholds']
    
    # Save feature list
    with open(model_dir / f"feature_list_{model_name}.json", 'w') as f:
        json.dump(feature_names, f)
    logging.info(f"Saved feature list with {len(feature_names)} features")
    
    # Save feature scaler
    if feature_scaler:
        joblib.dump(feature_scaler, model_dir / f"feature_scaler_{model_name}.save")
        logging.info("Saved feature scaler")
    
    # Save price thresholds
    if thresholds:
        with open(model_dir / "thresholds.json", 'w') as f:
            json.dump(thresholds, f)
        logging.info(f"Saved thresholds: {thresholds}")
    
    # Input shape calculation
    _, lookback, n_features = X_train.shape
    input_shape = (lookback, n_features)
    
    # Build model
    model = build_tcn_model(input_shape, output_dim=1, is_binary=True, model_type=model_type)
    
    # Class weights to handle imbalance
    class_ratio = np.mean(y_train)
    if class_ratio < 0.5:
        # More 0s than 1s - class 1 is minority
        weight_for_0 = 1
        # Make weight for valley class (1) much higher - increase from the original calculation
        if model_type == 'valley':
            # Apply the multiplier from config to make the valley class weight even higher
            weight_for_1 = (1 - class_ratio) / class_ratio * VALLEY_CLASS_WEIGHT_MULTIPLIER
            logging.info(f"Using aggressive class weighting for valley model: {weight_for_1:.2f}x weight for valleys")
        else:
            weight_for_1 = (1 - class_ratio) / class_ratio
    else:
        # More 1s than 0s - class 0 is minority
        weight_for_1 = 1
        weight_for_0 = class_ratio / (1 - class_ratio)
    
    class_weight = {0: weight_for_0, 1: weight_for_1}
    logging.info(f"Class weights: {class_weight}")
    
    # Set up callbacks
    callbacks = []
    
    # Only use validation-dependent callbacks when not in production mode
    if not production_mode:
        callbacks.extend([
            EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                verbose=1,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=model_dir / f"best_{model_name}.keras",
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ])
    else:
        # In production mode, use training metrics instead of validation
        callbacks.extend([
            ModelCheckpoint(
                filepath=model_dir / f"best_{model_name}.keras",
                monitor='loss',  # Monitor training loss
                save_best_only=True,  # Still save only the best model here
                save_weights_only=False,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='loss',  # Monitor training loss
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ])
        logging.info("Added production-mode callbacks")
    
    # Always include TensorBoard callback
    callbacks.append(
        TensorBoard(
            log_dir=model_dir / 'logs',
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    )
    
    # For valley model, add a callback to save the model with best recall
    if model_type == 'valley':
        # Add a callback to save the model with the best recall
        callbacks.append(
            ModelCheckpoint(
                filepath=model_dir / f"best_recall_{model_name}.keras",
                monitor='recall',
                mode='max',  # Higher recall is better
                save_best_only=True,
                verbose=1
            )
        )
        logging.info("Added callback to save model with best recall")
        
        # Also add a callback to save the model with best F1 score
        # This optimizes for the balance between precision and recall
        callbacks.append(
            ModelCheckpoint(
                filepath=model_dir / f"best_f1_{model_name}.keras",
                monitor='val_precision',  # We'll use this as a proxy for F1
                mode='max',  # Higher F1 is better
                save_best_only=True,
                verbose=1
            )
        )
        logging.info("Added callback to save model with best precision (proxy for F1)")
    
    # Compile model based on model type
    if model_type == 'valley':
        # Valley models ALWAYS use recall-oriented loss
        logging.info("Using recall-oriented loss function to prioritize valley detection")
        # Use a high false negative penalty to improve recall, with values from config
        recall_loss_fn = get_recall_oriented_loss(FALSE_NEG_WEIGHT, FALSE_POS_WEIGHT)
        model.compile(
            optimizer=Adam(learning_rate=VALLEY_LEARNING_RATE),
            loss=recall_loss_fn,
            metrics=['accuracy', Precision(), Recall(), AUC()]
        )
    else:
        # Peak models use standard binary crossentropy
        model.compile(
            optimizer=Adam(learning_rate=PEAK_LEARNING_RATE),
            loss=BinaryCrossentropy(),
            metrics=['accuracy', Precision(), Recall(), AUC()]
        )
    
    # Train the model with or without validation data
    fit_args = {
        'x': X_train,
        'y': y_train,
        'epochs': epochs,
        'batch_size': batch_size,
        'callbacks': callbacks,
        'class_weight': class_weight,
        'verbose': 1
    }
    
    # Only add validation data when not in production mode
    if not production_mode:
        fit_args['validation_data'] = (X_val, y_val)
        logging.info(f"Training with validation data (shape: {X_val.shape})")
    else:
        logging.info("Training without validation in production mode")
    
    # Train the model
    history = model.fit(**fit_args)
    
    # Log training results
    if not production_mode and 'val_loss' in history.history:
        val_loss = history.history['val_loss']
        best_epoch = np.argmin(val_loss) + 1
        best_val_loss = val_loss[best_epoch - 1]
        logging.info(f"Training completed. Best validation loss {best_val_loss:.4f} at epoch {best_epoch}")
        
        # Find optimal probability threshold using Precision-Recall curve (if validation data available)
        if model_type == 'valley':
            try:
                # Get validation predictions
                val_preds = model.predict(X_val, verbose=0)
                
                # Calculate precision and recall at various thresholds
                from sklearn.metrics import precision_recall_curve, f1_score
                precision, recall, thresholds = precision_recall_curve(y_val, val_preds)
                
                # Calculate F1 score for each threshold
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
                
                # Find threshold with best F1 score
                best_f1_idx = np.argmax(f1_scores)
                best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
                
                # Save the threshold
                with open(model_dir / f"optimal_threshold.json", 'w') as f:
                    json.dump({"threshold": float(best_threshold), "f1_score": float(f1_scores[best_f1_idx])}, f)
                
                logging.info(f"Optimal probability threshold: {best_threshold:.4f} with F1 score: {f1_scores[best_f1_idx]:.4f}")
                
                # Plot Precision-Recall curve
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 8))
                plt.plot(recall, precision, 'b-', linewidth=2)
                plt.scatter(recall[best_f1_idx], precision[best_f1_idx], marker='o', color='red', s=100, 
                           label=f'Best threshold: {best_threshold:.4f}, F1: {f1_scores[best_f1_idx]:.4f}')
                
                # Plot random baseline
                plt.plot([0, 1], [np.mean(y_val), np.mean(y_val)], 'r--', linewidth=1, label=f'Random ({np.mean(y_val):.4f})')
                
                plt.title('Precision-Recall Curve for Valley Detection')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.savefig(model_dir / "precision_recall_curve.png")
                plt.close()
                
                logging.info("Generated Precision-Recall curve for threshold selection")
            except Exception as e:
                logging.error(f"Error generating PR curve: {e}")
    else:
        # For production mode, track best epoch based on training loss
        train_loss = history.history['loss']
        best_epoch = np.argmin(train_loss) + 1
        best_train_loss = train_loss[best_epoch - 1]
        logging.info(f"Training completed in production mode. Best training loss {best_train_loss:.4f} at epoch {best_epoch}")
        
        # Create a record of training progress
        training_summary = {
            "best_epoch": int(best_epoch),
            "best_training_loss": float(best_train_loss),
            "epochs_trained": len(train_loss),
            "training_losses": [float(loss) for loss in train_loss],
            "final_loss": float(train_loss[-1]),
            "training_accuracy": [float(acc) for acc in history.history.get('accuracy', [])],
            "date_trained": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save training summary
        with open(model_dir / f"{model_name}_training_summary.json", 'w') as f:
            json.dump(training_summary, f, indent=4)
        logging.info(f"Saved training summary to {model_dir / f'{model_name}_training_summary.json'}")
    
    # Save training history
    with open(model_dir / f"{model_name}_history.json", 'w') as f:
        # Convert numpy values to float for JSON serialization
        history_dict = {}
        for k, v in history.history.items():
            history_dict[k] = [float(val) for val in v]
        json.dump(history_dict, f)
    
    # Save the final model
    final_model_path = model_dir / f"final_{model_name}.keras"
    model.save(final_model_path)
    logging.info(f"Saved model to {final_model_path}")
    
    # In production mode, also save as "best" model since we don't have model checkpoints
    if production_mode:
        best_model_path = model_dir / f"best_{model_name}.keras"
        model.save(best_model_path)
        logging.info(f"In production mode, also saved model as {best_model_path}")
    
    # Evaluate model on validation data
    if not production_mode:
        val_loss, val_acc, val_precision, val_recall, val_auc = model.evaluate(X_val, y_val, verbose=0)
        logging.info(f"Validation metrics - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, "
                    f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, AUC: {val_auc:.4f}")
    
    # Return model and training history
    return model, history

def create_directories():
    """Create all necessary directories for model training and evaluation."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TREND_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create evaluation directories
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    TREND_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for detailed visualizations
    (TREND_EVAL_DIR / "weekly").mkdir(parents=True, exist_ok=True)
    (TREND_EVAL_DIR / "daily").mkdir(parents=True, exist_ok=True)
    (TREND_EVAL_DIR / "monthly").mkdir(parents=True, exist_ok=True)
    
    logging.info("Created all necessary directories")

def load_and_merge_data():
    """Load and merge data from various sources with proper date alignment."""
    logging.info("Loading and merging data...")
    
    # Load price data (our target variable source)
    try:
        price_df = pd.read_csv(SE3_PRICES_FILE)
        if 'HourSE' in price_df.columns:
            price_df['datetime'] = pd.to_datetime(price_df['HourSE'], utc=True)
            price_df.set_index('datetime', inplace=True)
        else:
            price_df.index = pd.to_datetime(price_df.index, utc=True)
        
        logging.info(f"Loaded price data: {price_df.shape}")
        logging.info(f"Price data date range: {price_df.index.min()} to {price_df.index.max()}")
        
        # Get the valid date range from price data - this will be our reference
        min_date = price_df.index.min()
        max_date = price_df.index.max()
        
        # Only keep rows with valid target values
        price_df = price_df.dropna(subset=[TARGET_VARIABLE])
        
        if len(price_df) < len(price_df.index):
            logging.info(f"Removed {len(price_df.index) - len(price_df)} rows with missing target values")
            # Update date range after dropping NAs
            min_date = price_df.index.min()
            max_date = price_df.index.max()
            
        # Create merged dataframe starting with price data
        merged_df = price_df.copy()
        
    except Exception as e:
        logging.error(f"Error loading price data: {e}")
        raise ValueError(f"Could not load price data: {e}")

    # Load and merge grid data
    try:
        grid_df = pd.read_csv(SWEDEN_GRID_FILE)
        if 'datetime' in grid_df.columns:
            grid_df['datetime'] = pd.to_datetime(grid_df['datetime'], utc=True)
            grid_df.set_index('datetime', inplace=True)
        else:
            grid_df.index = pd.to_datetime(grid_df.index, utc=True)
        
        # Filter grid data to match price data date range
        grid_df = grid_df.loc[(grid_df.index >= min_date) & (grid_df.index <= max_date)]
        logging.info(f"Grid data range after trimming: {grid_df.index.min()} to {grid_df.index.max()}")
        
        # Merge grid features
        merged_df = merged_df.join(grid_df[GRID_FEATURES], how='left')
        logging.info(f"Merged grid data: {merged_df.shape}")
        
    except Exception as e:
        logging.warning(f"Error loading grid data: {e}")
        logging.warning("Continuing without grid data")

    # Load and merge time features
    try:
        time_df = pd.read_csv(TIME_FEATURES_FILE)
        if 'Unnamed: 0' in time_df.columns:
            time_df['datetime'] = pd.to_datetime(time_df['Unnamed: 0'], utc=True)
            time_df.drop('Unnamed: 0', axis=1, inplace=True)
            time_df.set_index('datetime', inplace=True)
        else:
            time_df.index = pd.to_datetime(time_df.index, utc=True)
        
        # Filter time data to match price data date range
        time_df = time_df.loc[(time_df.index >= min_date) & (time_df.index <= max_date)]
        
        # Merge time features
        merged_df = merged_df.join(time_df[TIME_FEATURES], how='left')
        logging.info(f"Merged time features: {merged_df.shape}")
        
    except Exception as e:
        logging.warning(f"Error loading time features: {e}")
        logging.warning("Continuing without time features")

    # Load and merge holidays data
    try:
        holidays_df = pd.read_csv(HOLIDAYS_FILE)
        if 'Unnamed: 0' in holidays_df.columns:
            holidays_df['datetime'] = pd.to_datetime(holidays_df['Unnamed: 0'], utc=True)
            holidays_df.drop('Unnamed: 0', axis=1, inplace=True)
            holidays_df.set_index('datetime', inplace=True)
        else:
            holidays_df.index = pd.to_datetime(holidays_df.index, utc=True)
        
        # Filter holidays data to match price data date range
        holidays_df = holidays_df.loc[(holidays_df.index >= min_date) & (holidays_df.index <= max_date)]
        
        # Merge holiday features
        merged_df = merged_df.join(holidays_df[HOLIDAY_FEATURES], how='left')
        logging.info(f"Merged holiday data: {merged_df.shape}")
        
    except Exception as e:
        logging.warning(f"Error loading holidays data: {e}")
        logging.warning("Continuing without holidays data")

    # Load and merge weather data
    try:
        weather_df = pd.read_csv(WEATHER_DATA_FILE)
        if 'date' in weather_df.columns:
            weather_df['datetime'] = pd.to_datetime(weather_df['date'], utc=True)
            weather_df.drop('date', axis=1, inplace=True)
            weather_df.set_index('datetime', inplace=True)
        else:
            weather_df.index = pd.to_datetime(weather_df.index, utc=True)
        
        # Filter weather data to match price data date range
        weather_df = weather_df.loc[(weather_df.index >= min_date) & (weather_df.index <= max_date)]
        logging.info(f"Weather data range after trimming: {weather_df.index.min()} to {weather_df.index.max()}")
        
        # Merge weather features
        merged_df = merged_df.join(weather_df[WEATHER_FEATURES], how='left')
        logging.info(f"Merged weather data: {merged_df.shape}")
        
    except Exception as e:
        logging.warning(f"Error loading weather data: {e}")
        logging.warning("Continuing without weather data")
    
    # Sort by datetime
    merged_df.sort_index(inplace=True)
    
    # Check for remaining missing values
    missing_values = merged_df.isna().sum()
    if missing_values.any():
        logging.warning(f"Found missing values after merging: {missing_values[missing_values > 0]}")
    else:
        logging.info("No missing values in merged dataframe - all data ranges properly aligned")
    
    # Handle any remaining missing values
    for col in merged_df.columns:
        if merged_df[col].isnull().any():
            # Count missing values
            missing_count = merged_df[col].isnull().sum()
            missing_percent = (missing_count / len(merged_df)) * 100
            
            # Log detailed information about the gaps
            logging.warning(f"Missing values in '{col}': {missing_count} rows ({missing_percent:.2f}%)")
            
            # Find contiguous gaps (runs of NaN values)
            is_null = merged_df[col].isnull()
            
            # Find the indices where is_null changes (boundaries of runs)
            null_run_starts = is_null[~is_null.shift(1, fill_value=False)].index
            null_run_ends = is_null[~is_null.shift(-1, fill_value=False)].index
            
            # Pair start and end indices
            null_runs = list(zip(null_run_starts, null_run_ends))
            
            # Log the largest gaps
            if len(null_runs) > 0:
                # Calculate run lengths
                run_lengths = [(end - start).total_seconds() / 3600 + 1 for start, end in null_runs]
                
                # Sort runs by length (descending)
                sorted_runs = sorted(zip(null_runs, run_lengths), key=lambda x: x[1], reverse=True)
                
                # Log the 5 largest gaps
                logging.warning(f"Top gaps in '{col}':")
                for i, ((start, end), length) in enumerate(sorted_runs[:5]):
                    logging.warning(f"  Gap #{i+1}: {start} to {end} ({int(length)} hours)")
                
                # Log summary stats about all gaps
                if len(null_runs) > 5:
                    logging.warning(f"  ... and {len(null_runs)-5} more gaps")
                
                logging.warning(f"  Median gap size: {np.median(run_lengths):.1f} hours")
                logging.warning(f"  Average gap size: {np.mean(run_lengths):.1f} hours")
            
            # Choose appropriate fill method based on column type
            if merged_df[col].dtype.kind in 'iufc':  # Numeric
                merged_df[col] = merged_df[col].ffill().bfill()
                logging.info(f"Filled missing values in {col} with forward/backward fill")
            else:
                merged_df[col] = merged_df[col].ffill().fillna('')
                logging.info(f"Filled missing values in {col} with forward fill")
    
    logging.info(f"Final merged dataframe shape: {merged_df.shape}")
    logging.info(f"Date range: {merged_df.index.min()} to {merged_df.index.max()}")
    
    return merged_df

def load_data(test_mode=False):
    """
    Load and prepare data for model training.
    
    Args:
        test_mode: If True, use a smaller dataset for testing
        
    Returns:
        Dictionary with prepared data for model training
    """
    logging.info("Loading data...")
    
    # Load and merge all data sources with alignment
    df = load_and_merge_data()
    
    # In test mode, use a smaller dataset
    if test_mode:
        # Use only the most recent 3 months for quick testing
        test_period = timedelta(days=90)
        df = df.iloc[-int(24*test_period.days):]
        logging.warning(f"Test mode: Using reduced dataset with {len(df)} rows")
    
    # Train/validation/test split
    train_size = int(len(df) * (1 - VALIDATION_SPLIT - TEST_SPLIT))
    val_size = int(len(df) * VALIDATION_SPLIT)
    
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size+val_size].copy()
    test_df = df.iloc[train_size+val_size:].copy()
    
    # Log split sizes
    logging.info(f"Train size: {len(train_df)}, from {train_df.index[0]} to {train_df.index[-1]}")
    logging.info(f"Validation size: {len(val_df)}, from {val_df.index[0]} to {val_df.index[-1]}")
    logging.info(f"Test size: {len(test_df)}, from {test_df.index[0]} to {test_df.index[-1]}")
    
    # Create target variables
    train_target = train_df[TARGET_VARIABLE]
    val_target = val_df[TARGET_VARIABLE]
    test_target = test_df[TARGET_VARIABLE]
    
    # Create combined target for production mode (all data)
    combined_target = df[TARGET_VARIABLE]
    
    # Return all prepared data
    return {
        'full_df': df,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'train_target': train_target,
        'val_target': val_target,
        'test_target': test_target,
        'combined_target': combined_target  # Add combined target for production mode
    }

def add_lag_features(df, target_col):
    """Add lagged features of the target variable."""
    logging.info("Adding lag features for trend detection...")
    result_df = df.copy()
    
    # Create lag features for different time horizons
    for lag in PRICE_LAG_HOURS:
        col_name = f"{target_col}_lag_{lag}h"
        result_df[col_name] = result_df[target_col].shift(lag)
    
    # Add momentum features (price differences) for valley detection
    result_df['price_diff_1h'] = result_df[target_col].diff(1)
    result_df['price_diff_3h'] = result_df[target_col].diff(3)
    result_df['price_diff_6h'] = result_df[target_col].diff(6)
    
    # Calculate a simple price momentum (acceleration) feature
    result_df['price_momentum'] = result_df['price_diff_1h'] - result_df['price_diff_1h'].shift(1)
    
    # Create a detrended price series based on 24h moving average
    result_df['price_detrended'] = result_df[target_col] - result_df[target_col].rolling(window=24, center=True).mean()
    
    # Fill NA values that result from lag creation
    numeric_cols = result_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if result_df[col].isnull().any():
            result_df[col] = result_df[col].fillna(method='bfill').fillna(method='ffill')
    
    logging.info(f"Added {len(PRICE_LAG_HOURS)} lag features and 4 momentum/detrended features")
    return result_df

def main():
    """Main function to train electricity price forecasting models."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train electricity price forecasting models')
    parser.add_argument('--model', type=str, choices=['trend', 'peak', 'valley'], default='trend',
                      help='Model type: trend (XGBoost), peak (TCN), or valley (TCN)')
    parser.add_argument('--test-mode', action='store_true',
                      help='Run in test mode with reduced dataset')
    parser.add_argument('--production', action='store_true',
                      help='Train with all available data for production use')
    args = parser.parse_args()
    
    # Configure logging
    configure_logging()
    
    # Create necessary directories
    create_directories()
    
    # Check if we're in test mode
    test_mode = args.test_mode
    if test_mode:
        logging.warning("Running in TEST MODE with reduced dataset!")
    
    # Check if we're in production mode
    production_mode = args.production
    if production_mode:
        logging.info("Running in PRODUCTION MODE with all available data!")
    
    try:
        # Load the data
        data = load_data(test_mode=test_mode)
        
        # Train the selected model
        if args.model == 'trend':
            logging.info("=== Starting trend model training (XGBoost) ===")
            print("\n==================================")
            print("=== TRAINING TREND MODEL (XGBoost) ===")
            print("==================================\n")
            model, metrics = train_trend_model(data, production_mode=production_mode)
            logging.info(f"Trend model training complete with metrics: {metrics}")
            print(f"\nTrend model training complete with validation metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
        
        elif args.model == 'peak':
            logging.info("=== Starting peak model training (TCN) ===")
            print("\n==================================")
            print("=== TRAINING PEAK MODEL (TCN) ===")
            print("==================================\n")
            peak_data = prepare_peak_valley_data(target_col='is_price_peak', production_mode=production_mode)
            model, history = train_peak_valley_model(peak_data, 'peak', production_mode=production_mode)
            logging.info(f"Peak model training complete")
            
        elif args.model == 'valley':
            logging.info("=== Starting valley model training (TCN) ===")
            print("\n==================================")
            print("=== TRAINING VALLEY MODEL (TCN) ===")
            print("==================================\n")
            
            # Valley model always uses SMOTE and recall-oriented loss
            logging.info("Valley model always uses SMOTE for class balancing")
            print("SMOTE oversampling enabled for balanced class distribution")
            logging.info("Valley model always uses recall-oriented loss for better handling of class imbalance")
            print("Using recall-oriented loss for better valley detection")
            
            # Get valley data with SMOTE always enabled
            valley_data = prepare_peak_valley_data(
                target_col='is_price_valley', 
                production_mode=production_mode
            )
            
            # Create validation plots for valley labels before training
            validation_plots_dir = VALLEY_MODEL_DIR / "valley_validation"
            validation_plots_dir.mkdir(parents=True, exist_ok=True)
            plot_valley_labels(valley_data['df'], output_dir=validation_plots_dir, num_samples=5, days_per_sample=14)
            logging.info(f"Generated valley label validation plots in {validation_plots_dir}")
            
            model, history = train_peak_valley_model(
                valley_data, 
                'valley', 
                production_mode=production_mode
            )
            logging.info(f"Valley model training complete")
        
        logging.info(f"{args.model} model training completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error in training process: {e}")
        logging.error(traceback.format_exc())
        print(f"\nError during training: {e}")
        return False

if __name__ == "__main__":
    main() 