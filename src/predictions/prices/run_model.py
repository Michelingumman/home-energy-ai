#!/usr/bin/env python
"""
Simplified evaluation script for electricity price prediction models.
Currently focuses on valley detection with proper validation split.

Usage:
    python evaluate.py --model valley  # Only valley implemented for now
    python evaluate.py --model peak    
    python evaluate.py --model trend   
    python evaluate.py --model merged  # merged model implements prediction of trend, peaks and valleys
"""

import os
# (optional) still suppress TensorFlow's C++ banners and oneDNN info:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
# silence TensorFlow-emitted WARNING/INFO logs
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import warnings
# suppress all Python warnings (FutureWarning, DeprecationWarning, etc.)
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K  # Add this line to fix the Lambda layer issue
import matplotlib.pyplot as plt
import logging
import json
import joblib
import pickle  # Add pickle import
import traceback  # Add traceback import
from pathlib import Path
import argparse
from datetime import datetime, timedelta
from tcn import TCN
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import xgboost as xgb  # For loading and using XGBoost models
from scipy.signal import medfilt, savgol_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d


USE_ADVANCED_TREND_SMOOTHING = False

# This is not used if above is True
SIMPLE_TREND_MODEL_SMOOTHING_LEVEL = 'light' # 'light', 'medium', 'heavy', 'daily', 'weekly'


# Define custom layers needed for model loading
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

# Import utility functions
from utils import (
    add_lag_features, add_rolling_features, create_sequences,
)

# Import config parameters
from config import (
    MODELS_DIR, PLOTS_DIR, TARGET_VARIABLE,
    LOOKBACK_WINDOW, PREDICTION_HORIZON,
    SE3_PRICES_FILE, SWEDEN_GRID_FILE, TIME_FEATURES_FILE,
    HOLIDAYS_FILE, WEATHER_DATA_FILE,
    PRICE_FEATURES, GRID_FEATURES, TIME_FEATURES,
    HOLIDAY_FEATURES, WEATHER_FEATURES, MARKET_FEATURES,
    CORE_FEATURES, EXTENDED_FEATURES,
    VALIDATION_SPLIT, TEST_SPLIT,
    TREND_MODEL_DIR, VALLEY_DETECTION_PARAMS,
    ROBUST_VALLEY_DETECTION_PARAMS,
    PEAK_CORE_FEATURES, VALLEY_CORE_FEATURES
)


# Create necessary directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)

EVALUATION_DIR = PLOTS_DIR / "evaluation"
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

VALLEY_MODEL_DIR = MODELS_DIR / "valley_model"
VALLEY_MODEL_DIR.mkdir(parents=True, exist_ok=True)

PEAK_MODEL_DIR = MODELS_DIR / "peak_model"
PEAK_MODEL_DIR.mkdir(parents=True, exist_ok=True)

VALLEY_EVAL_DIR = EVALUATION_DIR / "valley"
VALLEY_EVAL_DIR.mkdir(parents=True, exist_ok=True)

PEAK_EVAL_DIR = EVALUATION_DIR / "peak"
PEAK_EVAL_DIR.mkdir(parents=True, exist_ok=True)

TREND_EVAL_DIR = EVALUATION_DIR / "trend"
TREND_EVAL_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = EVALUATION_DIR / "logs"
LOG_merged_DIR = EVALUATION_DIR / "merged" / "logs"
LOG_peak_DIR = EVALUATION_DIR / "peak" / "logs"
LOG_trend_DIR = EVALUATION_DIR / "trend" / "logs"
LOG_valley_DIR = EVALUATION_DIR / "valley" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_merged_DIR.mkdir(parents=True, exist_ok=True)
LOG_peak_DIR.mkdir(parents=True, exist_ok=True)
LOG_trend_DIR.mkdir(parents=True, exist_ok=True)
LOG_valley_DIR.mkdir(parents=True, exist_ok=True)

# Define smoothing functions at global scope so they're accessible everywhere
def exponential_smooth(predictions, alpha=0.01):
    """Apply exponential smoothing to predictions to reduce volatility."""
    smoothed = [predictions[0]]
    for i in range(1, len(predictions)):
        smoothed.append(alpha * predictions[i] + (1-alpha) * smoothed[i-1])
    return np.array(smoothed)

def median_filter(data, window=5):
    """Apply a median filter to remove outliers."""
    return medfilt(data, kernel_size=window)

def savitzky_golay_filter(data, window=11, polyorder=2):
    """Apply Savitzky-Golay filter for smoothing while preserving trends."""
    # Ensure window is odd
    if window % 2 == 0:
        window += 1
        
    # Ensure window is less than data length
    if window >= len(data):
        window = min(11, len(data) - 1)
        if window % 2 == 0:
            window -= 1
            
    return savgol_filter(data, window, polyorder)

def adaptive_trend_smoothing(data, timestamps, smoothing_level='light'):
    """
    Apply adaptive smoothing based on the level specified.
    
    Args:
        data: Array of price values to smooth
        timestamps: Datetime index for the data
        smoothing_level: One of 'light', 'medium', 'heavy', 'daily', 'weekly'
        
    Returns:
        Smoothed data array
    """
    result = np.array(data).copy()
    
    if smoothing_level == 'light':
        # Light smoothing
        result = exponential_smooth(result, alpha=0.6)
        result = median_filter(result, window=3)
        
    elif smoothing_level == 'medium':
        # Medium smoothing (default)
        result = exponential_smooth(result, alpha=0.30)
        result = median_filter(result, window=5)
        result = savitzky_golay_filter(result, window=11, polyorder=2)
        
    elif smoothing_level == 'heavy':
        # Heavy smoothing
        result = exponential_smooth(result, alpha=0.1) # alpha here is the smoothing factor where lower is more smoothing
        result = median_filter(result, window=7)
        result = savitzky_golay_filter(result, window=13, polyorder=2)
        
    elif smoothing_level == 'daily':
        # Daily average smoothing
        # Group by day and smooth with daily averages
        if timestamps is not None:
            daily_df = pd.DataFrame({'price': result, 'date': timestamps})
            daily_df['day'] = daily_df['date'].dt.date
            daily_means = daily_df.groupby('day')['price'].mean()
            
            for day in daily_df['day'].unique():
                # Replace with daily average
                daily_df.loc[daily_df['day'] == day, 'price'] = daily_means[day]
                
            result = daily_df['price'].values
            
    elif smoothing_level == 'weekly':
        # Weekly pattern smoothing
        # Use day of week and hour patterns
        if timestamps is not None:
            pattern_df = pd.DataFrame({'price': result, 'date': timestamps})
            pattern_df['hour'] = pattern_df['date'].dt.hour
            pattern_df['dayofweek'] = pattern_df['date'].dt.dayofweek
            
            # Get average price by day of week and hour
            pattern_means = pattern_df.groupby(['dayofweek', 'hour'])['price'].mean()
            
            # Apply the pattern
            for dow in pattern_df['dayofweek'].unique():
                for hour in pattern_df['hour'].unique():
                    if (dow, hour) in pattern_means:
                        pattern_df.loc[(pattern_df['dayofweek'] == dow) & 
                                       (pattern_df['hour'] == hour), 'price'] = pattern_means[dow, hour]
            
            result = pattern_df['price'].values
            
    return result

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def configure_logging(model_type):
    """Configure logging based on model type."""
    # Get the root logger
    logger = logging.getLogger('')
    
    # Clear any existing handlers to avoid duplication
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    # Set basic level
    logger.setLevel(logging.INFO)
    
    # Choose the appropriate log file based on model type
    if model_type == 'trend':
        log_file = TREND_EVAL_DIR / "logs" / "trend_model_evaluation.log"
    elif model_type == 'peak':
        log_file = PEAK_EVAL_DIR / "logs" / "peak_model_evaluation.log"
    elif model_type == 'valley':
        log_file = VALLEY_EVAL_DIR / "logs" / "valley_model_evaluation.log"
    elif model_type == 'merged':
        log_file = EVALUATION_DIR / "merged" / "logs" / "merged_model_evaluation.log"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Make sure the log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Add file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Add console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # Show what we're using
    logging.info(f"Logging to {log_file}")
    
    # Suppress deprecation warnings from TensorFlow/Keras
    tf.get_logger().setLevel('ERROR')

def get_test_model_dir(model_type):
    """
    Get the appropriate test model directory path based on model type.
    
    Args:
        model_type: Type of model ('trend', 'peak', or 'valley')
        
    Returns:
        Path object pointing to the test model directory
    """
    # Get base model directory
    if model_type == 'trend':
        base_dir = TREND_MODEL_DIR
    elif model_type == 'peak':
        base_dir = PEAK_MODEL_DIR
    elif model_type == 'valley':
        base_dir = VALLEY_MODEL_DIR
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Use test_model subdirectory
    model_dir = base_dir / "test_model"
    
    # Check if directory exists
    if not model_dir.exists():
        logging.warning(f"Test model directory {model_dir} does not exist. Falling back to base directory.")
        return base_dir
    
    logging.info(f"Using test model directory: {model_dir}")
    return model_dir

def get_production_model_dir(model_type):
    """
    Get the appropriate production model directory path based on model type.
    
    Args:
        model_type: Type of model ('trend', 'peak', or 'valley')
        
    Returns:
        Path object pointing to the production model directory
    """
    # Get base model directory
    if model_type == 'trend':
        base_dir = TREND_MODEL_DIR
    elif model_type == 'peak':
        base_dir = PEAK_MODEL_DIR
    elif model_type == 'valley':
        base_dir = VALLEY_MODEL_DIR
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Use production_model subdirectory
    model_dir = base_dir / "production_model"
    
    # Check if directory exists
    if not model_dir.exists():
        logging.warning(f"Production model directory {model_dir} does not exist. Falling back to base directory.")
        return base_dir
    
    logging.info(f"Using production model directory: {model_dir}")
    return model_dir

def load_data():
    """Load and prepare data for model evaluation."""
    try:
        # Load price data
        price_df = pd.read_csv(SE3_PRICES_FILE)
        if 'HourSE' in price_df.columns:
            price_df['datetime'] = pd.to_datetime(price_df['HourSE'], utc=True)
            price_df.set_index('datetime', inplace=True)
        else:
            price_df.index = pd.to_datetime(price_df.index, utc=True)
        logging.info(f"Loaded price data with shape {price_df.shape}")
        
        # Load grid data
        try:
            grid_df = pd.read_csv(SWEDEN_GRID_FILE)
            if 'datetime' in grid_df.columns:
                grid_df['datetime'] = pd.to_datetime(grid_df['datetime'], utc=True)
                grid_df.set_index('datetime', inplace=True)
            else:
                grid_df.index = pd.to_datetime(grid_df.index, utc=True)
            logging.info(f"Loaded grid data with shape {grid_df.shape}")
        except Exception as e:
            logging.warning(f"Error loading grid data: {e}")
            grid_df = None
        
        # Load time features
        try:
            time_df = pd.read_csv(TIME_FEATURES_FILE)
            if 'Unnamed: 0' in time_df.columns:
                time_df['datetime'] = pd.to_datetime(time_df['Unnamed: 0'], utc=True)
                time_df.drop('Unnamed: 0', axis=1, inplace=True)
                time_df.set_index('datetime', inplace=True)
            else:
                time_df.index = pd.to_datetime(time_df.index, utc=True)
            logging.info(f"Loaded time features with shape {time_df.shape}")
        except Exception as e:
            logging.warning(f"Error loading time features: {e}")
            time_df = None
        
        # Load holidays data
        try:
            holidays_df = pd.read_csv(HOLIDAYS_FILE)
            if 'Unnamed: 0' in holidays_df.columns:
                holidays_df['datetime'] = pd.to_datetime(holidays_df['Unnamed: 0'], utc=True)
                holidays_df.drop('Unnamed: 0', axis=1, inplace=True)
                holidays_df.set_index('datetime', inplace=True)
            else:
                holidays_df.index = pd.to_datetime(holidays_df.index, utc=True)
            logging.info(f"Loaded holidays data with shape {holidays_df.shape}")
        except Exception as e:
            logging.warning(f"Error loading holidays data: {e}")
            holidays_df = None
        
        # Load weather data
        try:
            weather_df = pd.read_csv(WEATHER_DATA_FILE)
            if 'date' in weather_df.columns:
                weather_df['datetime'] = pd.to_datetime(weather_df['date'], utc=True)
                weather_df.drop('date', axis=1, inplace=True)
                weather_df.set_index('datetime', inplace=True)
            else:
                weather_df.index = pd.to_datetime(weather_df.index, utc=True)
            logging.info(f"Loaded weather data with shape {weather_df.shape}")
        except Exception as e:
            logging.warning(f"Error loading weather data: {e}")
            weather_df = None
        
        # Merge all dataframes
        # Start with price data as the base
        merged_df = price_df.copy()
        
        # Merge with grid data
        if grid_df is not None:
            merged_df = merged_df.join(grid_df[GRID_FEATURES], how='left')
        
        # Merge with time features
        if time_df is not None:
            merged_df = merged_df.join(time_df[TIME_FEATURES], how='left')
        
        # Merge with holidays data
        if holidays_df is not None:
            merged_df = merged_df.join(holidays_df[HOLIDAY_FEATURES], how='left')
        
        # Merge with weather data
        if weather_df is not None:
            merged_df = merged_df.join(weather_df[WEATHER_FEATURES], how='left')
        
        # Sort by datetime
        merged_df.sort_index(inplace=True)
        
        # Fill missing values
        for col in merged_df.columns:
            if merged_df[col].isnull().any():
                # Choose appropriate fill method based on column type
                if merged_df[col].dtype.kind in 'iufc':  # Numeric
                    merged_df[col] = merged_df[col].ffill().bfill()
                else:
                    merged_df[col] = merged_df[col].ffill().fillna('')
        
        # Add lag features for valley detection
        merged_df = add_lag_features(merged_df, TARGET_VARIABLE)
        merged_df = add_rolling_features(merged_df, TARGET_VARIABLE)
        
        # Import the detect_valleys_robust function from train.py to ensure consistency
        sys.path.append(os.path.dirname(__file__))
        from train import detect_valleys_robust
        
        # Apply the robust valley detection (same as in training) instead of derivative-based
        merged_df = detect_valleys_robust(
            merged_df,
            target_col=TARGET_VARIABLE,
            **ROBUST_VALLEY_DETECTION_PARAMS  # Use parameters from config
        )
        
        # Use the robust valleys (same as training)
        merged_df['is_price_valley'] = merged_df['is_price_valley_robust']
        
        # Log statistics
        total = len(merged_df)
        num_valleys = merged_df['is_price_valley'].sum()
        
        logging.info(f"Identified {num_valleys} valleys ({num_valleys/total:.1%} of data) using robust valley detection")
        logging.info(f"Final merged data shape: {merged_df.shape}")
        
        return merged_df
    
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def load_valley_model(production_mode=False):
    """Load the valley detection model and associated artifacts.
    
    Args:
        production_mode: If True, load from production model directory instead of test model directory
    """
    logging.info("Loading valley detection model...")
    
    try:
        # Get the model directory based on mode
        if production_mode:
            model_dir = get_production_model_dir('valley')
            logging.info("Loading PRODUCTION valley model for future prediction")
        else:
            model_dir = get_test_model_dir('valley')
            logging.info("Loading TEST valley model for evaluation")
        
        # Determine the model path - check if model exists
        model_paths = [
            model_dir / "best_valley_model.keras",        # Best model by validation loss
            model_dir / "best_f1_valley_model.keras",     # Model optimized for F1 score
            model_dir / "final_valley_model.keras",       # Final model after training
            model_dir / "best_recall_valley_model.keras"  # Model optimized for recall
        ]
        
        # Find the first valid model path
        model_path = None
        for path in model_paths:
            if path.exists():
                model_path = path
                logging.info(f"Using model from {path.name}")
                break
        
        if model_path is None:
            # Try the parent directory paths if needed
            parent_paths = [
                VALLEY_MODEL_DIR / "best_valley_model.keras",
                VALLEY_MODEL_DIR / "best_f1_valley_model.keras",
                VALLEY_MODEL_DIR / "final_valley_model.keras",
                VALLEY_MODEL_DIR / "best_recall_valley_model.keras"
            ]
            
            for path in parent_paths:
                if path.exists():
                    model_path = path
                    logging.info(f"Falling back to model from parent directory: {path.name}")
                    break
            
            if model_path is None:
                raise ValueError("No valley model found in test_model or base directory")
                
        # Load optimal threshold from PR curve analysis if available
        threshold_path = model_dir / "optimal_threshold.json"
        if not threshold_path.exists():
            threshold_path = VALLEY_MODEL_DIR / "optimal_threshold.json"  # Fallback to parent directory
            
        prob_threshold = 0.5  # Default
        if threshold_path.exists():
            try:
                with open(threshold_path, 'r') as f:
                    threshold_data = json.load(f)
                    # Try different possible keys for the threshold
                    if 'threshold' in threshold_data:
                        prob_threshold = threshold_data['threshold']
                    elif 'optimal_threshold' in threshold_data:
                        prob_threshold = threshold_data['optimal_threshold']
                    else:
                        # Use the first value in the JSON if it's a simple structure
                        if isinstance(threshold_data, dict) and len(threshold_data) > 0:
                            prob_threshold = list(threshold_data.values())[0]
                    
                logging.info(f"Using optimal probability threshold: {prob_threshold}")
            except Exception as e:
                logging.warning(f"Could not load optimal threshold, using default: {e}")

        # Define the recall-oriented loss function to match what we used in training
        def get_recall_oriented_loss(false_neg_weight=8.0, false_pos_weight=1.0):
            """
            Create a custom loss function that prioritizes recall over precision.
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
        
        # Get the specific loss function with parameters matching those used in training
        recall_loss_fn = get_recall_oriented_loss(false_neg_weight=6.0, false_pos_weight=1.5)
        
        # Load custom objects if needed
        custom_objects = {
            'TCN': TCN,
            recall_loss_fn.__name__: recall_loss_fn,
            'sum_output_shape': lambda input_shape: (input_shape[0], input_shape[2]),
            'GlobalSumPooling1D': GlobalSumPooling1D  # Add the custom layer class
        }
        
        # Define the K.sum function for the Lambda layer
        def lambda_sum(x):
            return K.sum(x, axis=1)
            
        # Add the lambda function to custom objects
        custom_objects['lambda_sum'] = lambda_sum
        
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        logging.info(f"Loaded valley model from {model_path}")
    except Exception as e:
        logging.error(f"Error loading valley model: {e}")
        raise
    
    # Load feature list from test_model dir or fall back to parent dir
    feature_list_path = model_dir / "feature_list_valley_model.json"
    if not feature_list_path.exists():
        feature_list_path = VALLEY_MODEL_DIR / "feature_list_valley_model.json"  # Fallback
        
    try:
        with open(feature_list_path, 'r') as f:
            feature_names = json.load(f)
        logging.info(f"Loaded feature list with {len(feature_names)} features from {feature_list_path}")
    except Exception as e:
        logging.error(f"Error loading feature list: {e}")
        raise
    
    # Load feature scaler if it exists from test_model dir or fall back to parent dir
    feature_scaler = None
    feature_scaler_path = model_dir / "feature_scaler_valley_model.save"
    if not feature_scaler_path.exists():
        feature_scaler_path = VALLEY_MODEL_DIR / "feature_scaler_valley_model.save"  # Fallback
        
    if feature_scaler_path.exists():
        try:
            feature_scaler = joblib.load(feature_scaler_path)
            logging.info(f"Loaded feature scaler from {feature_scaler_path}")
        except Exception as e:
            logging.warning(f"Error loading feature scaler: {e}")
    
    return {
        'model': model,
        'feature_names': feature_names,
        'feature_scaler': feature_scaler,
        'prob_threshold': prob_threshold
    }

def evaluate_valley_model(df, model_artifacts, num_weeks=3, use_test_data=False, prob_threshold=None, find_optimal_threshold=False):
    """
    Evaluate the valley detection model on the validation or test set.
    
    Args:
        df: DataFrame with full data
        model_artifacts: Dictionary with model and preprocessing tools
        num_weeks: Number of weeks to plot
        use_test_data: If True, evaluate on test data instead of validation data
        prob_threshold: Probability threshold for valley detection (if None, use default 0.5)
        find_optimal_threshold: If True, find optimal threshold using F1 score
        
    Returns:
        Dictionary with evaluation results
    """
    logging.info(f"Evaluating valley detection model on {'test' if use_test_data else 'validation'} data...")
    
    # Get model artifacts
    model = model_artifacts['model']
    feature_scaler = model_artifacts['feature_scaler']
    feature_list = model_artifacts['feature_names']  # Changed from 'feature_list' to 'feature_names'
    
    # Use the artifacts' optimal threshold if none is specified
    if prob_threshold is None and 'prob_threshold' in model_artifacts:
        prob_threshold = model_artifacts['prob_threshold']
    elif prob_threshold is None:
        prob_threshold = 0.50  # Default
    
    logging.info(f"Using probability threshold: {prob_threshold:.4f}")
    
    # Add or ensure required features like detrended price and momentum features
    df = add_valley_features(df)
    
    # Split data
    train_size = int(len(df) * (1 - VALIDATION_SPLIT - TEST_SPLIT))
    val_size = int(len(df) * VALIDATION_SPLIT)
    
    if use_test_data:
        eval_df = df.iloc[train_size+val_size:].copy()
        eval_name = "test"
    else:
        eval_df = df.iloc[train_size:train_size+val_size].copy()
        eval_name = "validation"
    
    # Limit to most recent num_weeks if specified
    if num_weeks > 0:
        hours_to_use = 24 * 7 * num_weeks
        if len(eval_df) > hours_to_use:
            eval_df = eval_df.iloc[-hours_to_use:].copy()
    
    logging.info(f"Using {len(eval_df)} {eval_name} samples for evaluation")
    
    # Extract target and features
    if 'is_price_valley' in eval_df.columns:
        y_true = eval_df['is_price_valley'].values
    else:
        y_true = np.zeros(len(eval_df))
        logging.warning("No 'is_price_valley' column found. Using zeros as target.")
    
    # Check if all features are available
    missing_features = [f for f in feature_list if f not in eval_df.columns]
    if missing_features:
        logging.warning(f"Missing features: {missing_features}")
        # Drop missing features from list
        feature_list = [f for f in feature_list if f in eval_df.columns]
    
    # Extract features and scale
    X = eval_df[feature_list].values
    if feature_scaler:
        X = feature_scaler.transform(X)
    
    # Create sequences
    X_sequences, _ = create_sequences(
        X, LOOKBACK_WINDOW, PREDICTION_HORIZON, 
        list(range(X.shape[1])), None
    )
    
    # Get actual target values aligned with sequences
    y_actual = y_true[LOOKBACK_WINDOW:LOOKBACK_WINDOW+len(X_sequences)]
    
    # Get timestamps for plotting
    timestamps = eval_df.index[LOOKBACK_WINDOW:LOOKBACK_WINDOW+len(X_sequences)]
    
    # Get prices for visualization
    prices = eval_df[TARGET_VARIABLE].values[LOOKBACK_WINDOW:LOOKBACK_WINDOW+len(X_sequences)]
    
    # Make predictions
    y_pred_proba = model.predict(X_sequences, verbose=0)
    
    # Convert to binary predictions based on threshold
    y_pred = (y_pred_proba >= prob_threshold).astype(int)
    
    # If requested, find optimal threshold using PR curve
    if find_optimal_threshold:
        from sklearn.metrics import precision_recall_curve, f1_score
        precision, recall, thresholds = precision_recall_curve(y_actual, y_pred_proba)
        
        # Calculate F1 score at each threshold
        f1_scores = []
        for i in range(len(precision)):
            if i < len(thresholds):
                thresh = thresholds[i]
                y_pred_at_thresh = (y_pred_proba >= thresh).astype(int)
                f1 = f1_score(y_actual, y_pred_at_thresh)
                f1_scores.append((thresh, f1))
        
        # Find threshold with best F1 score
        best_threshold, best_f1 = max(f1_scores, key=lambda x: x[1])
        
        # Update predictions using best threshold
        logging.info(f"Found optimal threshold: {best_threshold:.4f} with F1 score: {best_f1:.4f}")
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        prob_threshold = best_threshold
    
    # Calculate evaluation metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(y_actual, y_pred)
    precision = precision_score(y_actual, y_pred, zero_division=0)
    recall = recall_score(y_actual, y_pred, zero_division=0)
    f1 = f1_score(y_actual, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_actual, y_pred_proba)
    except:
        roc_auc = 0
        logging.warning("Could not calculate ROC AUC - possibly only one class present")
    
    # Log metrics
    logging.info(f"Valley detection {eval_name} metrics (threshold={prob_threshold:.4f}):")
    logging.info(f"  Accuracy: {accuracy:.4f}")
    logging.info(f"  Precision: {precision:.4f}")
    logging.info(f"  Recall: {recall:.4f}")
    logging.info(f"  F1 Score: {f1:.4f}")
    logging.info(f"  ROC AUC: {roc_auc:.4f}")
    
    # Class distribution
    positive_rate = np.mean(y_actual)
    predicted_rate = np.mean(y_pred)
    logging.info(f"  Actual valley rate: {positive_rate:.4f} ({np.sum(y_actual)} valleys)")
    logging.info(f"  Predicted valley rate: {predicted_rate:.4f} ({np.sum(y_pred)} valleys)")
    
    # Generate visual evaluation
    output_dir = generate_weekly_valley_plots(
        timestamps, y_pred, y_pred_proba, y_actual, prices, eval_name=eval_name
    )
    
    # Create evaluation metrics dictionary
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'actual_valley_rate': float(positive_rate),
        'predicted_valley_rate': float(predicted_rate),
        'threshold': float(prob_threshold),
        'plots_dir': str(output_dir)
    }
    
    return metrics

def add_valley_features(df, enable_logging=True):
    """Add features specifically designed for valley detection."""
    if enable_logging:
        logging.info("Adding specialized features for valley detection...")
    result_df = df.copy()
    
    # Add momentum features (price differences) if they don't exist
    if 'price_diff_1h' not in result_df.columns:
        result_df['price_diff_1h'] = result_df[TARGET_VARIABLE].diff(1)
    
    if 'price_diff_3h' not in result_df.columns:
        result_df['price_diff_3h'] = result_df[TARGET_VARIABLE].diff(3)
    
    if 'price_diff_6h' not in result_df.columns:
        result_df['price_diff_6h'] = result_df[TARGET_VARIABLE].diff(6)
    
    # Calculate a simple price momentum (acceleration) feature
    if 'price_momentum' not in result_df.columns:
        result_df['price_momentum'] = result_df['price_diff_1h'] - result_df['price_diff_1h'].shift(1)
    
    # Create a detrended price series based on 24h moving average
    if 'price_detrended' not in result_df.columns:
        result_df['price_detrended'] = result_df[TARGET_VARIABLE] - result_df[TARGET_VARIABLE].rolling(window=24, center=True).mean()
    
    # Fill NA values that result from feature creation
    numeric_cols = result_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if result_df[col].isnull().any():
            result_df[col] = result_df[col].fillna(method='bfill').fillna(method='ffill')
    
    return result_df

def generate_weekly_valley_plots(timestamps, predictions, probabilities, actuals, prices, eval_name="validation"):
    """
    Generate weekly visualizations of valley predictions.
    
    Args:
        timestamps: Array of timestamps
        predictions: Array of binary predictions (0/1)
        probabilities: Array of prediction probabilities (0-1)
        actuals: Array of actual labels (0/1)
        prices: Array of price values
        eval_name: Name of the evaluation data ("validation" or "test")
    """
    # Create output directory
    output_dir = VALLEY_EVAL_DIR / eval_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure all arrays are 1-dimensional
    predictions = np.squeeze(predictions).flatten()
    probabilities = np.squeeze(probabilities).flatten()
    actuals = np.squeeze(actuals).flatten()
    prices = np.squeeze(prices).flatten()
    
    # Check dimensions for debugging
    logging.info(f"Data dimensions: timestamps {len(timestamps)}, predictions {predictions.shape}, " 
                 f"probabilities {probabilities.shape}, actuals {actuals.shape}, prices {prices.shape}")
    
    # Group by week
    df = pd.DataFrame({
        'timestamp': timestamps,
        'prediction': predictions,
        'probability': probabilities,
        'actual': actuals,
        'price': prices
    })
    
    # Add week start column
    df['week_start'] = df['timestamp'].apply(lambda x: (x.floor('D') - timedelta(days=x.weekday())))
    df['week_key'] = df['week_start'].dt.strftime('%Y-%m-%d')
    
    # Get list of weeks
    weeks = df['week_key'].unique()
    
    # Process each week
    for week_key in weeks:
        # Get data for this week
        week_data = df[df['week_key'] == week_key].copy()
        
        # Skip if we have less than 24 hours of data
        if len(week_data) < 24:
            logging.warning(f"Skipping week {week_key} - insufficient data (only {len(week_data)} hours)")
            continue
        
        # Sort by timestamp
        week_data = week_data.sort_values('timestamp')
        
        # Create the weekly plot
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot price data
        ax.step(week_data['timestamp'], week_data['price'], 'b-', linewidth=2, label='Price')
        
        # Highlight actual valleys
        for idx, row in week_data[week_data['actual'] == 1].iterrows():
            ax.axvspan(row['timestamp'] - timedelta(minutes=30), 
                       row['timestamp'] + timedelta(minutes=30),
                       color='green', alpha=0.3, label='_' if idx > week_data.index[0] else 'Actual Valley')
        
        # Mark predicted valleys with color and size based on probability
        predicted_valleys = week_data[week_data['prediction'] == 1]
        if len(predicted_valleys) > 0:
            # Create color map based on probability
            cmap = plt.cm.viridis  # Use a perceptually uniform colormap
            norm = plt.Normalize(vmin=0.5, vmax=1.0)  # Normalize probability from 0.5-1.0
            
            # Plot predicted valleys with varying marker sizes and colors based on probability
            for idx, row in predicted_valleys.iterrows():
                marker_size = 40 + int(row['probability'] * 160)  # Scale: 40-200 based on probability
                color = cmap(norm(row['probability']))
                ax.scatter(row['timestamp'], row['price'], marker='v', color=color, s=marker_size, 
                          edgecolors='black', linewidth=1,
                          label='_' if idx > predicted_valleys.index[0] else 'Predicted Valley')
                
                # Add annotation with probability value
                ax.annotate(f'{row["probability"]:.2f}', 
                           (row['timestamp'], row['price']),
                           xytext=(0, -20),
                           textcoords='offset points',
                           ha='center',
                           fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='black', alpha=0.7))
                
            # Add a colorbar to explain probability->color mapping
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label('Probability (Depth Score)', fontsize=10)
            
        # Calculate metrics for this week
        week_actuals = week_data['actual'].values
        week_predictions = week_data['prediction'].values
        
        true_positives = np.sum((week_predictions == 1) & (week_actuals == 1))
        false_positives = np.sum((week_predictions == 1) & (week_actuals == 0))
        false_negatives = np.sum((week_predictions == 0) & (week_actuals == 1))
        
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        
        # Add title with metrics
        ax.set_title(f'Week starting {week_key} ({eval_name} Data)\n' +
                    f'Valleys: {np.sum(week_actuals)} actual, {np.sum(week_predictions)} predicted | ' +
                    f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}',
                    fontsize=14)
        
        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (öre/kWh)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%a %d %b'))
        ax.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator())
        
        # Add day separators
        week_start = week_data['week_start'].iloc[0]
        for day in pd.date_range(start=week_start, periods=8, freq='D'):
            ax.axvline(day, color='gray', linestyle='--', alpha=0.5)
        
        # Annotate all actual valleys with price
        for idx, row in week_data[week_data['actual'] == 1].iterrows():
            ax.annotate(f'{row["price"]:.1f}', 
                       (row['timestamp'], row['price']),
                       xytext=(0, -15),
                       textcoords='offset points',
                       ha='center',
                       fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.7))
        
        # Mark true positives, false positives, and false negatives
        for idx, row in week_data.iterrows():
            if row['prediction'] == 1 and row['actual'] == 1:  # True positive
                ax.scatter(row['timestamp'], row['price'], marker='o', color='green', s=120, 
                          label='_' if idx > week_data.index[0] else 'True Positive')
            elif row['prediction'] == 1 and row['actual'] == 0:  # False positive
                ax.scatter(row['timestamp'], row['price'], marker='x', color='red', s=120, 
                          label='_' if idx > week_data.index[0] else 'False Positive')
            elif row['prediction'] == 0 and row['actual'] == 1:  # False negative
                ax.scatter(row['timestamp'], row['price'], marker='s', color='orange', s=120, 
                          label='_' if idx > week_data.index[0] else 'False Negative')
        
        # Add legend with better organization and more details
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / f"valley_week_{week_key}.png", dpi=300)
        plt.close()
        
        logging.info(f"Generated visualization for week starting {week_key}")
    
    return output_dir

def load_trend_model(production_mode=False):
    """Load the trained trend model (XGBoost).
    
    Args:
        production_mode: If True, load from production model directory instead of test model directory
    """
    logging.info("Searching for trend models...")
    
    # Get the appropriate model directory based on mode
    if production_mode:
        model_dir = get_production_model_dir('trend')
        logging.info("Loading PRODUCTION trend model for future prediction")
    else:
        model_dir = get_test_model_dir('trend')
        logging.info("Loading TEST trend model for evaluation")
    
    # Define file paths in model directory
    best_model_path = model_dir / "best_trend_model.pkl"
    fallback_model_path = model_dir / "trend_model.pkl"
    params_path = model_dir / "trend_model_params.json"
    feature_names_path = model_dir / "feature_names.json"
    
    # If files don't exist in model directory, fall back to parent directory
    if not best_model_path.exists() and not fallback_model_path.exists():
        logging.warning(f"No trend model found in {model_dir}, falling back to parent directory")
        best_model_path = TREND_MODEL_DIR / "best_trend_model.pkl"
        fallback_model_path = TREND_MODEL_DIR / "trend_model.pkl"
        params_path = TREND_MODEL_DIR / "trend_model_params.json"
        feature_names_path = TREND_MODEL_DIR / "feature_names.json"
    
    # Determine which model file to use - prioritize best_trend_model.pkl
    if best_model_path.exists():
        model_path = best_model_path
        logging.info(f"Using best trend model: {best_model_path}")
    elif fallback_model_path.exists():
        model_path = fallback_model_path
        logging.info(f"Using fallback trend model: {fallback_model_path}")
    else:
        logging.error(f"No valid trend model found in {model_dir} or parent directory")
        raise FileNotFoundError(f"No trend model files found in {model_dir} or parent directory")
    
    # Load model parameters if available, with fallback to parent directory
    if not params_path.exists():
        params_path = TREND_MODEL_DIR / "trend_model_params.json"
        
    model_params = {}
    if params_path.exists():
        try:
            with open(params_path, 'r') as f:
                model_params = json.load(f)
            model_type = model_params.get('model_type', 'XGBoost')
            logging.info(f"Loaded parameters for {model_type} model from {params_path}")
        except Exception as e:
            logging.warning(f"Error loading model parameters, using defaults: {e}")
            model_type = 'XGBoost'
    else:
        logging.warning("No parameter file found, using default parameter values")
        model_type = 'XGBoost'
    
    # Load the model
    try:
        with open(str(model_path), 'rb') as f:
            model = pickle.load(f)
        logging.info(f"Successfully loaded model from {model_path}")
        
        # Extract additional parameters
        features = model_params.get('features', [])
        important_features = model_params.get('important_features', [])
        smoothing_params = model_params.get('smoothing', {})
        
        # Load feature names/order if available, with fallback to parent directory
        if not feature_names_path.exists():
            feature_names_path = TREND_MODEL_DIR / "feature_names.json"
            
        if feature_names_path.exists():
            try:
                with open(feature_names_path, 'r') as f:
                    feature_order = json.load(f)
                logging.info(f"Loaded feature order list with {len(feature_order)} features from {feature_names_path}")
            except Exception as e:
                logging.warning(f"Error loading feature order: {e}")
                feature_order = features
        else:
            feature_order = features
            logging.warning("No feature_names.json found, using features from parameters")
        
        if features:
            logging.info(f"Model uses {len(features)} features")
            if important_features:
                logging.info(f"Top features: {important_features[:5] if len(important_features) >= 5 else important_features}")
        
        return {
            'model': model,
            'model_type': model_type,
            'features': features,
            'feature_order': feature_order,
            'important_features': important_features,
            'smoothing_params': smoothing_params
        }
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def evaluate_trend_model(df, model_artifacts, num_weeks=4):
    """
    Evaluate the trend model on test data with detailed visualizations at multiple time scales.
    
    Args:
        df: DataFrame with all data
        model_artifacts: Dictionary with model and associated artifacts
        num_weeks: Number of weeks to visualize
        
    Returns:
        Dictionary with evaluation results
    """
    model_type = model_artifacts.get('model_type', 'XGBoost') 
    
    # Get model information for logging
    model = model_artifacts['model']
    features = model_artifacts.get('features', [])
    feature_order = model_artifacts.get('feature_order', [])
    important_features = model_artifacts.get('important_features', [])
    smoothing_params = model_artifacts.get('smoothing_params', {})
    logging.info(f"Evaluating XGBoost trend model")
    logging.info(f"Using {len(features)} features with top important: {important_features[:3] if len(important_features) >= 3 else important_features}")
    
    # Get test data
    train_size = int(len(df) * (1 - VALIDATION_SPLIT - TEST_SPLIT))
    val_size = int(len(df) * VALIDATION_SPLIT)
    test_df = df.iloc[train_size+val_size:].copy()
    
    logging.info(f"Test data shape: {test_df.shape}, starting from {test_df.index[0]}")
    
    # Generate forecasts for test period
    test_steps = len(test_df)
    logging.info(f"Forecasting test period ({test_steps} steps)...")
    
    # Make forecast using the model
    try:
        # Function to add time features to match training
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
            
            # Cyclical encoding of time features
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
        
        # Prepare features for prediction using all available data
        exog_features = [
            "powerConsumptionTotal", "powerProductionTotal", "hydro", "nuclear",
            "wind", "powerImportTotal", "powerExportTotal", "Gas_Price",
            "Coal_Price", "CO2_Price", "temperature_2m", "wind_speed_100m",
            "cloud_cover", "price_168h_avg", "hour_avg_price", "price_24h_avg"
        ]
        
        # Create a DataFrame with all necessary features for prediction
        avail_exog = [f for f in exog_features if f in test_df.columns]
        X_test_all = test_df[avail_exog].copy()
        
        # Fill missing values
        for col in X_test_all.columns:
            if X_test_all[col].isnull().any():
                if X_test_all[col].dtype.kind in 'iuf':  # integer, unsigned int, or float
                    X_test_all[col] = X_test_all[col].fillna(X_test_all[col].median())
                else:
                    X_test_all[col] = X_test_all[col].fillna(X_test_all[col].mode()[0])
        
        # Add time features
        X_test_all = add_gb_time_features(X_test_all)
        
        # Check if we have the right feature order information
        if feature_order and len(feature_order) > 0:
            # Ensure we have all necessary features
            for feat in feature_order:
                if feat not in X_test_all.columns:
                    # Add missing features with zeros
                    X_test_all[feat] = 0.0
                    logging.warning(f"Added missing feature {feat} with zeros")
            
            # Reorder columns to match the model's training data
            X_test = X_test_all[feature_order].copy()
            logging.info(f"Using saved feature order with {len(feature_order)} features")
        else:
            # Just use what we have
            X_test = X_test_all
            logging.warning("No feature order information available, using all features")
        
        logging.info(f"Final test feature matrix: {X_test.shape}")
        
        # Skip the normal DMatrix creation which does name checking
        # Use a safer approach that avoids feature mismatch
        logging.info("Creating DMatrix with only feature values (no column names)")
        model.set_param({'predictor': 'cpu_predictor'})
        
        # Create DMatrix without feature names to avoid mismatch
        dtest = xgb.DMatrix(X_test.values)
        
        # Disable feature name validation - a workaround 
        # to avoid the feature mismatch error
        try:
            # First try setting the feature names to match exactly if needed
            if hasattr(model, 'feature_names') and feature_order:
                # Clear existing feature names to avoid mismatch
                orig_feature_names = getattr(model, 'feature_names', None)
                
                if orig_feature_names is not None:
                    logging.info(f"Clearing existing feature names in model: {orig_feature_names}")
                    model.feature_names = None
                
                # Create a new DMatrix with feature names that match the model's expected names
                logging.info("Creating DMatrix with matching feature order")
                dtest = xgb.DMatrix(X_test.values, feature_names=feature_order)
            
            # Make predictions
            test_pred_raw = model.predict(dtest)
            logging.info("Successfully made predictions")
            
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise e
        
        # Get the test actuals
        test_actual = test_df[TARGET_VARIABLE].values
        
        # Apply smoothing using same parameters from training
        alpha = smoothing_params.get('exponential_alpha', 0.35)
        median_window = smoothing_params.get('median_window', 5)
        savgol_window = smoothing_params.get('savgol_window', 11)
        savgol_polyorder = smoothing_params.get('savgol_polyorder', 2)
        smoothing_level = SIMPLE_TREND_MODEL_SMOOTHING_LEVEL
        
        logging.info(f"Applying {smoothing_level} smoothing level from trained model")
        
        if USE_ADVANCED_TREND_SMOOTHING:
            logging.info("Applying Advanced Trend Smoothing pipeline to predictions (exponential smoothing, median filter, savitzky-golay filter)")
            test_pred = exponential_smooth(test_pred_raw, alpha=alpha)
            test_pred = median_filter(test_pred, window=median_window)
            test_pred = savitzky_golay_filter(test_pred, window=savgol_window, polyorder=savgol_polyorder)
        # Use adaptive trend smoothing if smoothing level is specified
        elif smoothing_level in ['light', 'medium', 'heavy', 'daily', 'weekly']:
            test_pred = adaptive_trend_smoothing(test_pred_raw, X_test.index, smoothing_level)
        else:
            raise ValueError(f"Unknown smoothing level: {smoothing_level}")
        
        # Create index for forecasts
        forecast_index = test_df.index
    except Exception as e:
        logging.error(f"Error during XGBoost prediction: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise
        
    # Handle potential length mismatch
    min_length = min(len(test_actual), len(test_pred))
    if min_length < len(test_actual):
        logging.warning(f"Length mismatch between actual ({len(test_actual)}) and predicted ({len(test_pred)})")
    
    test_actual = test_actual[:min_length]
    test_pred = test_pred[:min_length]
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'actual': test_actual,
        'predicted': test_pred,
        'error': test_actual - test_pred,
        'abs_error': np.abs(test_actual - test_pred),
        'squared_error': (test_actual - test_pred)**2
    }, index=forecast_index[:min_length])
    
    # Save forecast data as CSV
    forecast_csv_path = TREND_EVAL_DIR / "trend_forecast_results.csv"
    forecast_df.to_csv(forecast_csv_path)
    logging.info(f"Saved forecast results to {forecast_csv_path}")
    
    # Compute metrics
    test_mae = np.mean(np.abs(test_actual - test_pred))
    test_rmse = np.sqrt(np.mean((test_actual - test_pred) ** 2))
    
    # For MAPE, handle potential zeros or very small values by adding small epsilon
    epsilon = 1.0  # 1 öre/kWh as a minimum to avoid division by zero
    test_mape = np.mean(np.abs((test_actual - test_pred) / (np.abs(test_actual) + epsilon))) * 100
    
    # Direction accuracy (if price goes up or down correctly)
    test_actual_diff = np.diff(test_actual)
    test_pred_diff = np.diff(test_pred)
    direction_accuracy = np.mean((test_actual_diff > 0) == (test_pred_diff > 0))
    
    # Calculate SMAPE (Symmetric Mean Absolute Percentage Error)
    smape = 100 * np.mean(2.0 * np.abs(test_actual - test_pred) / (np.abs(test_actual) + np.abs(test_pred) + epsilon))
    
    # Calculate median absolute error
    median_ae = np.median(np.abs(test_actual - test_pred))
    
    # Peak accuracy metrics
    std_dev = np.std(test_actual)
    mean_actual = np.mean(test_actual)
    peak_threshold = mean_actual + std_dev
    
    # Identify actual peaks
    actual_peaks = test_actual > peak_threshold
    
    # Check if predicted values correctly identify the peaks
    if np.sum(actual_peaks) > 0:
        peak_accuracy = np.mean((test_pred > peak_threshold) == actual_peaks)
        peak_mae = np.mean(np.abs(test_actual[actual_peaks] - test_pred[actual_peaks]))
    else:
        peak_accuracy = np.nan
        peak_mae = np.nan
    
    metrics = {
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'test_mape': float(test_mape),
        'smape': float(smape),
        'median_ae': float(median_ae),
        'direction_accuracy': float(direction_accuracy),
        'peak_accuracy': float(peak_accuracy),
        'peak_mae': float(peak_mae),
        'model_type': model_type,
        'num_observations': int(min_length),
        'test_period': f"{test_df.index[0]} to {test_df.index[min_length-1]}" if min_length > 0 else "Unknown"
    }
    
    logging.info(f"Test MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}, MAPE: {test_mape:.2f}%")
    logging.info(f"Direction Accuracy: {direction_accuracy:.2f}, SMAPE: {smape:.2f}%")
    
    # Save evaluation metrics
    metrics_path = TREND_EVAL_DIR / "trend_model_evaluation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Saved evaluation metrics to {metrics_path}")
    
    # Generate visualizations
    logging.info("Generating visualizations...")
    
    # Create subdirectories for visualizations if they don't exist
    weekly_dir = TREND_EVAL_DIR / "weekly"
    monthly_dir = TREND_EVAL_DIR / "monthly"
    weekly_dir.mkdir(parents=True, exist_ok=True)
    monthly_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot overall test period
    plt.figure(figsize=(16, 8))
    plt.step(forecast_index, test_actual, 'b-', label='Actual', alpha=0.7)
    plt.step(forecast_index, test_pred, 'r-', label='Predicted', alpha=0.7)
    plt.title(f'Price Forecast vs Actual (XGBoost) - Overall Test Period\nMAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}, Direction Accuracy: {direction_accuracy:.2f}')
    plt.xlabel('Date')
    plt.ylabel('Price (öre/kWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(TREND_EVAL_DIR / "overall_forecast.png")
    plt.close()
    
    # Generate weekly plots
    # Select weeks from the test period for detailed analysis
    logging.info(f"Generating weekly visualizations for {num_weeks} weeks...")
    
    # Find week boundaries in the test period
    week_starts = []
    for i in range(0, len(forecast_index), 24*7):
        if i + 24*7 <= len(forecast_index):
            week_starts.append(i)
    
    # Select weeks with reasonable spacing
    if len(week_starts) > num_weeks:
        # Space them out evenly across the test period
        step = len(week_starts) // num_weeks
        selected_weeks = week_starts[::step][:num_weeks]
    else:
        selected_weeks = week_starts
    
    # Generate plots for each selected week
    for week_idx in selected_weeks:
        week_end_idx = min(week_idx + 24*7, len(forecast_index))
        week_slice = slice(week_idx, week_end_idx)
        
        week_start_date = forecast_index[week_idx].strftime('%Y-%m-%d')
        week_end_date = forecast_index[week_end_idx-1].strftime('%Y-%m-%d')
        
        # Create weekly plot
        plt.figure(figsize=(16, 8))
        
        # Plot actual and predicted prices
        plt.step(forecast_index[week_slice], test_actual[week_slice], 'b-', label='Actual', linewidth=2)
        plt.step(forecast_index[week_slice], test_pred[week_slice], 'r-', label='Predicted', linewidth=2)
        
        # Highlight errors
        error = test_actual[week_slice] - test_pred[week_slice]
        abs_error = np.abs(error)
        
        # Calculate weekly metrics
        week_mae = np.mean(abs_error)
        week_rmse = np.sqrt(np.mean(error**2))
        
        # Calculate direction accuracy for the week
        week_actual_diff = np.diff(test_actual[week_slice])
        week_pred_diff = np.diff(test_pred[week_slice])
        week_dir_acc = np.mean((week_actual_diff > 0) == (week_pred_diff > 0))
        
        plt.title(f'Week: {week_start_date} to {week_end_date}\nMAE: {week_mae:.2f}, RMSE: {week_rmse:.2f}, Direction Accuracy: {week_dir_acc:.2f}')
        plt.xlabel('Date')
        plt.ylabel('Price (öre/kWh)')
        
        # Format x-axis for better readability
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%a %d %b'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        
        # Add day separators
        for day_offset in range(7):
            day_date = forecast_index[week_idx] + timedelta(days=day_offset)
            plt.axvline(day_date, color='gray', linestyle='--', alpha=0.5)
        
        # Add hour indicators for key hours (e.g., noon and midnight)
        for day_offset in range(7):
            for hour in [0, 12]:
                hour_date = forecast_index[week_idx] + timedelta(days=day_offset, hours=hour)
                if hour == 0:
                    plt.axvline(hour_date, color='black', linestyle='-', alpha=0.2)
                else:
                    plt.axvline(hour_date, color='gray', linestyle=':', alpha=0.3)
        
        # Add error visualization
        plt.fill_between(forecast_index[week_slice], test_actual[week_slice], test_pred[week_slice], 
                         color='gray', alpha=0.2, label='Error')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(weekly_dir / f"week_{week_start_date}.png")
        plt.close()
        
        logging.info(f"Generated visualization for week {week_start_date}")
    
    # Generate monthly plots
    logging.info("Generating monthly visualizations...")
    
    # Group by month
    forecast_df = pd.DataFrame({
        'date': forecast_index,
        'actual': test_actual,
        'predicted': test_pred,
        'error': test_actual - test_pred,
        'abs_error': np.abs(test_actual - test_pred)
    })
    
    forecast_df['year'] = forecast_df['date'].dt.year
    forecast_df['month'] = forecast_df['date'].dt.month
    forecast_df['day'] = forecast_df['date'].dt.day
    
    # Get unique months in the test period
    unique_months = forecast_df[['year', 'month']].drop_duplicates().values
    
    # Generate monthly plots
    for year, month in unique_months:
        # Filter data for this month
        month_data = forecast_df[(forecast_df['year'] == year) & (forecast_df['month'] == month)]
        
        # Skip if less than 3 days of data
        if len(month_data) < 3*24:
            continue
        
        month_name = pd.Timestamp(year=year, month=month, day=1).strftime('%B %Y')
        
        # Create monthly plot
        plt.figure(figsize=(16, 10))
        
        # Create a subplot grid
        gs = plt.GridSpec(3, 1, height_ratios=[3, 1, 1])
        
        # Price plot
        ax1 = plt.subplot(gs[0])
        ax1.step(month_data['date'], month_data['actual'], 'b-', label='Actual', linewidth=2)
        ax1.step(month_data['date'], month_data['predicted'], 'r-', label='Predicted', linewidth=2)
        
        # Calculate monthly metrics
        month_mae = month_data['abs_error'].mean()
        month_rmse = np.sqrt((month_data['error']**2).mean())
        
        # Calculate direction accuracy for the month
        month_actual_diff = np.diff(month_data['actual'])
        month_pred_diff = np.diff(month_data['predicted'])
        if len(month_actual_diff) > 0:
            month_dir_acc = np.mean((month_actual_diff > 0) == (month_pred_diff > 0))
        else:
            month_dir_acc = float('nan')
        
        ax1.set_title(f'Month: {month_name}\nMAE: {month_mae:.2f}, RMSE: {month_rmse:.2f}, Direction Accuracy: {month_dir_acc:.2f}')
        ax1.set_ylabel('Price (öre/kWh)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis for better readability
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        
        # Add day separators
        month_days = month_data['day'].unique()
        for day in month_days:
            day_date = pd.Timestamp(year=year, month=month, day=day)
            ax1.axvline(day_date, color='gray', linestyle='--', alpha=0.3)
        
        # Error plot
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.step(month_data['date'], month_data['error'], 'g-', label='Error', linewidth=1.5)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.fill_between(month_data['date'], month_data['error'], 0, 
                         where=(month_data['error'] > 0), color='green', alpha=0.3, interpolate=True)
        ax2.fill_between(month_data['date'], month_data['error'], 0, 
                         where=(month_data['error'] <= 0), color='red', alpha=0.3, interpolate=True)
        ax2.set_ylabel('Error')
        ax2.grid(True, alpha=0.3)
        
        # Daily aggregated metrics
        ax3 = plt.subplot(gs[2], sharex=ax1)
        
        # Group by day for daily metrics
        daily_metrics = month_data.groupby('day').agg({
            'abs_error': 'mean',
            'actual': ['mean', 'std']
        })
        
        daily_metrics.columns = ['_'.join(col).strip() for col in daily_metrics.columns.values]
        
        # Create daily metrics plot
        daily_dates = [pd.Timestamp(year=year, month=month, day=day) for day in daily_metrics.index]
        ax3.bar(daily_dates, daily_metrics['abs_error_mean'], width=0.8, alpha=0.6, color='orange', label='Daily MAE')
        
        # Add line showing daily price average
        ax4 = ax3.twinx()
        ax4.step(daily_dates, daily_metrics['actual_mean'], 'b--', label='Avg Price', linewidth=1.5)
        ax4.set_ylabel('Avg Daily Price', color='blue')
        
        # Set legends and labels
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax4.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax3.set_ylabel('Daily MAE')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(monthly_dir / f"month_{year}_{month:02d}.png")
        plt.close()
        
        logging.info(f"Generated visualization for month {month_name}")
    
    # Generate error distribution plot
    plt.figure(figsize=(12, 8))
    
    # Create histogram of errors
    plt.hist(test_actual - test_pred, bins=50, alpha=0.7, color='steelblue')
    plt.axvline(x=0, color='red', linestyle='--')
    
    # Add normal distribution fit
    from scipy import stats
    error_mean = np.mean(test_actual - test_pred)
    error_std = np.std(test_actual - test_pred)
    x = np.linspace(error_mean - 3*error_std, error_mean + 3*error_std, 100)
    plt.step(x, stats.norm.pdf(x, error_mean, error_std) * len(test_actual) * (6*error_std/50), 
             'r-', linewidth=2, label=f'Normal: μ={error_mean:.2f}, σ={error_std:.2f}')
    
    plt.title('Error Distribution')
    plt.xlabel('Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(TREND_EVAL_DIR / "error_distribution.png")
    plt.close()
    
    # Generate error vs actual plot to check for bias
    plt.figure(figsize=(12, 8))
    plt.scatter(test_actual, test_actual - test_pred, alpha=0.5, color='steelblue')
    plt.axhline(y=0, color='red', linestyle='--')
    
    # Add trend line
    z = np.polyfit(test_actual, test_actual - test_pred, 1)
    p = np.poly1d(z)
    plt.step(np.sort(test_actual), p(np.sort(test_actual)), "r--", 
             label=f'Trend: y = {z[0]:.4f}x + {z[1]:.2f}')
    
    plt.title('Error vs Actual Price (Checking for Bias)')
    plt.xlabel('Actual Price')
    plt.ylabel('Error (Actual - Predicted)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(TREND_EVAL_DIR / "error_vs_actual.png")
    plt.close()
    
    # Return the evaluation results
    return {
        'metrics': metrics,
        'actual': test_actual,
        'predicted': test_pred,
        'dates': forecast_index[:min_length],
        'model_type': model_type
    }

def generate_summary_report(results, model_type):
    """
    Generate a summary report of the evaluation results.
    
    Args:
        results: Dictionary with evaluation results
        model_type: Type of model ('trend', 'peak', or 'valley')
    """
    report_path = None
    
    if model_type == 'trend':
        report_path = TREND_EVAL_DIR / "trend_evaluation_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"TREND MODEL EVALUATION SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model type: {results['model_type']}\n")
            f.write(f"Test period: {results['metrics']['test_period']}\n")
            f.write(f"Number of observations: {results['metrics']['num_observations']}\n\n")
            
            f.write("METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mean Absolute Error (MAE): {results['metrics']['test_mae']:.4f}\n")
            f.write(f"Root Mean Squared Error (RMSE): {results['metrics']['test_rmse']:.4f}\n")
            f.write(f"Mean Absolute Percentage Error (MAPE): {results['metrics']['test_mape']:.2f}%\n")
            f.write(f"Symmetric Mean Absolute Percentage Error (SMAPE): {results['metrics']['smape']:.2f}%\n")
            f.write(f"Median Absolute Error: {results['metrics']['median_ae']:.4f}\n")
            f.write(f"Direction Accuracy: {results['metrics']['direction_accuracy']:.4f}\n")
            f.write(f"Peak Accuracy: {results['metrics']['peak_accuracy']:.4f}\n")
            f.write(f"Peak Mean Absolute Error: {results['metrics']['peak_mae']:.4f}\n\n")
            
            f.write("VISUALIZATION FILES\n")
            f.write("-" * 40 + "\n")
            for file in sorted(TREND_EVAL_DIR.glob("*.png")):
                f.write(f"- {file.name}\n")
            
            f.write("\n")
            f.write("DATA FILES\n")
            f.write("-" * 40 + "\n")
            for file in sorted(TREND_EVAL_DIR.glob("*.csv")):
                f.write(f"- {file.name}\n")
            for file in sorted(TREND_EVAL_DIR.glob("*.json")):
                f.write(f"- {file.name}\n")
            
            f.write("\n")
            f.write("NOTES\n")
            f.write("-" * 40 + "\n")
            f.write("- MAE and RMSE are in öre/kWh units\n")
            f.write("- Direction Accuracy measures how often the model correctly predicts price movements (up/down)\n")
            f.write("- Peak Accuracy measures how well the model identifies price peaks (>1 std deviation above mean)\n")
            
    elif model_type == 'valley':
        report_path = VALLEY_EVAL_DIR / "valley_evaluation_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"VALLEY DETECTION MODEL EVALUATION SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Use the metrics that are guaranteed to be there
            f.write(f"Evaluation metrics:\n")
            f.write(f"  Probability threshold: {results.get('threshold', 'N/A')}\n")
            
            # Write the metrics
            f.write("\nMETRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy: {results.get('accuracy', 'N/A')}\n")
            f.write(f"Precision: {results.get('precision', 'N/A')}\n")
            f.write(f"Recall: {results.get('recall', 'N/A')}\n")
            f.write(f"F1 Score: {results.get('f1_score', 'N/A')}\n")
            f.write(f"ROC AUC: {results.get('roc_auc', 'N/A')}\n")
            f.write(f"Actual valley rate: {results.get('actual_valley_rate', 'N/A')}\n")
            f.write(f"Predicted valley rate: {results.get('predicted_valley_rate', 'N/A')}\n\n")
            
            # Write info about the plot directory
            plots_dir = results.get('plots_dir', None)
            if plots_dir:
                f.write("VISUALIZATION FILES\n")
                f.write("-" * 40 + "\n")
                f.write(f"Plots directory: {plots_dir}\n\n")
                
                # List plot files if the directory exists
                plots_path = Path(plots_dir)
                if plots_path.exists():
                    for file in sorted(plots_path.glob("*.png")):
                        f.write(f"- {file.name}\n")
                else:
                    f.write("No visualization files found.\n")
            
            f.write("\n")
            f.write("NOTES\n")
            f.write("-" * 40 + "\n")
            f.write("- Valley detection is a binary classification task\n")
            f.write("- Precision is the ratio of correctly identified valleys to all predicted valleys\n")
            f.write("VISUALIZATION FILES\n")
            f.write("-" * 40 + "\n")
            for file in sorted(VALLEY_EVAL_DIR.glob("*.png")):
                f.write(f"- {file.name}\n")
    
    return report_path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate electricity price prediction models')
    parser.add_argument('--model', type=str, choices=['trend', 'peak', 'valley', 'merged'], default='merged',
                      help='Model type to evaluate (trend, peak, valley, merged)')
    parser.add_argument('--weeks', type=int, default=4,
                      help='Number of weeks to visualize in evaluation (or number of samples for peak/valley detection)')
    parser.add_argument('--test-data', action='store_true', dest='test_data',
                      help='Use test data instead of validation data for evaluation')
    parser.add_argument('--valley-threshold', type=float, default=0.5,
                      help='Probability threshold for valley detection (default: 0.5). Used when --model is \'valley\'. For \'merged\' model, also acts as peak threshold if --peak-threshold is not specified.')
    parser.add_argument('--peak-threshold', type=float, default=None,
                      help='Probability threshold for peak detection. Used when --model is \'peak\'. Also used for the peak component in \'merged\' model if explicitly set.')
    parser.add_argument('--optimize-threshold', action='store_true', dest='optimize_threshold',
                      help='Find the optimal threshold for peak/valley classification models based on F1 score on the current evaluation data.')
    
    # New arguments for production prediction mode
    parser.add_argument('--production-mode', action='store_true', dest='production_mode',
                      help='Enable production prediction mode using production models to predict future prices')
    parser.add_argument('--start', type=str, default=None,
                      help='Start date for predictions in YYYY-MM-DD format (default: current date)')
    parser.add_argument('--horizon', type=float, default=1.0,
                      help='Prediction horizon in days (default: 1 day)')
    
    return parser.parse_args()

# Add a new function to find optimal threshold
def find_optimal_threshold(probabilities, actuals):
    """Find the optimal threshold for binary classification.
    
    Args:
        probabilities: Numpy array of prediction probabilities
        actuals: Numpy array of actual binary values
        
    Returns:
        Dictionary with threshold analysis results
    """
    logging.info("Finding optimal threshold for classification...")
    results = []
    
    # Try different thresholds
    thresholds = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        
        # Skip if we have no positive predictions
        if predictions.sum() == 0:
            results.append({
                'threshold': threshold,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'predicted_positives': 0,
                'actual_positives': actuals.sum()
            })
            continue
        
        # Calculate metrics
        precision = precision_score(actuals, predictions, zero_division=0)
        recall = recall_score(actuals, predictions, zero_division=0)
        f1 = f1_score(actuals, predictions, zero_division=0)
        
        # Save results
        result = {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predicted_positives': predictions.sum(),
            'actual_positives': actuals.sum()
        }
        results.append(result)
        
        # Update best threshold
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # Log results
    logging.info(f"Threshold analysis results:")
    for r in results:
        logging.info(f"  Threshold {r['threshold']:.2f}: Precision={r['precision']:.4f}, "
                    f"Recall={r['recall']:.4f}, F1={r['f1']:.4f}, "
                    f"Predicted={r['predicted_positives']}, Actual={r['actual_positives']}")
    
    logging.info(f"Best threshold: {best_threshold:.4f} with F1 score: {best_f1:.4f}")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.plot([r['threshold'] for r in results], [r['precision'] for r in results], 'b-', marker='o', label='Precision')
    plt.plot([r['threshold'] for r in results], [r['recall'] for r in results], 'g-', marker='s', label='Recall')
    plt.plot([r['threshold'] for r in results], [r['f1'] for r in results], 'r-', marker='^', label='F1 Score')
    plt.axvline(x=best_threshold, color='k', linestyle='--', label=f'Best Threshold: {best_threshold:.2f}')
    
    plt.title('Classification Performance vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    return {
        'results': results,
        'best_threshold': best_threshold,
        'best_f1': best_f1
    }

def add_cyclical_time_features(df):
    """
    Add cyclical time features like sine and cosine transforms of day and day of week.
    These are important for capturing time-based patterns for the models.
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with additional cyclical time features
    """
    import numpy as np
    
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Make sure we have a datetime index
    if not isinstance(result.index, pd.DatetimeIndex):
        logging.warning("DataFrame doesn't have a DatetimeIndex. Cannot add time features.")
        return result
    
    # Extract date components
    result['day'] = result.index.day
    result['dayofweek'] = result.index.dayofweek
    
    # Create cyclical features
    result['day_sin'] = np.sin(2 * np.pi * result['day'] / 31)
    result['day_cos'] = np.cos(2 * np.pi * result['day'] / 31)
    result['dayofweek_sin'] = np.sin(2 * np.pi * result['dayofweek'] / 7)
    result['dayofweek_cos'] = np.cos(2 * np.pi * result['dayofweek'] / 7)
    
    # Remove original components that aren't needed (keeps them if the models need them)
    # result.drop(['day', 'dayofweek'], axis=1, inplace=True, errors='ignore')
    
    return result


def load_peak_model(production_mode=False):
    """Load the trained peak detection model.
    
    Args:
        production_mode: If True, load from production model directory instead of test model directory
    """
    logging.info("Loading peak detection model...")
    
    try:
        # Get the appropriate model directory based on mode
        if production_mode:
            model_dir = get_production_model_dir('peak')
            logging.info("Loading PRODUCTION peak model for future prediction")
        else:
            model_dir = get_test_model_dir('peak')
            logging.info("Loading TEST peak model for evaluation")
            
        # Look for model files - try different names
        model_paths = [
            model_dir / "best_peak_model.keras",        # Best model by validation loss
            model_dir / "best_f1_peak_model.keras",     # Model optimized for F1 score
            model_dir / "final_peak_model.keras",       # Final model after training
            model_dir / "best_recall_peak_model.keras"  # Model optimized for recall
        ]
        
        # Find the first valid model path
        model_path = None
        for path in model_paths:
            if path.exists():
                model_path = path
                logging.info(f"Using model from {path.name}")
                break
                
        # If no model found in test_model directory, check parent directory
        if model_path is None:
            parent_paths = [
                PEAK_MODEL_DIR / "best_peak_model.keras",
                PEAK_MODEL_DIR / "best_f1_peak_model.keras",
                PEAK_MODEL_DIR / "final_peak_model.keras",
                PEAK_MODEL_DIR / "best_recall_peak_model.keras"
            ]
            
            for path in parent_paths:
                if path.exists():
                    model_path = path
                    logging.info(f"Falling back to model from parent directory: {path.name}")
                    break
            
            if model_path is None:
                raise ValueError("No peak model found in test_model or base directory")
        
        # Load optimal threshold from PR curve analysis if available
        threshold_path = model_dir / "optimal_threshold.json"
        if not threshold_path.exists():
            threshold_path = PEAK_MODEL_DIR / "optimal_threshold.json"  # Fallback to parent directory
            
        prob_threshold = 0.5  # Default
        if threshold_path.exists():
            try:
                with open(threshold_path, 'r') as f:
                    threshold_data = json.load(f)
                    # Try different possible keys for the threshold
                    if 'threshold' in threshold_data:
                        prob_threshold = threshold_data['threshold']
                    elif 'optimal_threshold' in threshold_data:
                        prob_threshold = threshold_data['optimal_threshold']
                    else:
                        # Use the first value in the JSON if it's a simple structure
                        if isinstance(threshold_data, dict) and len(threshold_data) > 0:
                            prob_threshold = list(threshold_data.values())[0]
                    
                logging.info(f"Using optimal probability threshold: {prob_threshold}")
            except Exception as e:
                logging.warning(f"Could not load optimal threshold, using default: {e}")
                
        # Load custom objects if needed for categorical focal loss
        custom_objects = {'TCN': TCN}
        
        # Add focal loss function
        # Instead of using get_focal_loss_custom_objects(), define it directly here
        def binary_focal_loss(gamma=2.0, alpha=0.25):
            """
            Binary form of focal loss.
            """
            def binary_focal_loss_fixed(y_true, y_pred):
                # Clip the prediction value to prevent extreme cases
                epsilon = K.epsilon()
                y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
                
                # Calculate cross entropy
                cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
                
                # Calculate focal loss
                loss = alpha * K.pow(1 - y_pred, gamma) * y_true * K.log(y_pred) - \
                       (1 - alpha) * K.pow(y_pred, gamma) * (1 - y_true) * K.log(1 - y_pred)
                
                return -K.mean(loss)
            return binary_focal_loss_fixed
        
        focal_loss_fn = binary_focal_loss()
        focal_loss_fn.__name__ = 'binary_focal_loss_fixed'
        custom_objects['binary_focal_loss_fixed'] = focal_loss_fn
        
        # Add GlobalSumPooling1D custom layer
        custom_objects['GlobalSumPooling1D'] = GlobalSumPooling1D
        
        # Define the lambda_sum function
        def lambda_sum(x):
            return K.sum(x, axis=1)
            
        # Add the lambda function to custom objects
        custom_objects['lambda_sum'] = lambda_sum
        custom_objects['sum_output_shape'] = lambda input_shape: (input_shape[0], input_shape[2])
        
        # Add focal loss function
        # Instead of using get_focal_loss_custom_objects(), define it directly here
        def binary_focal_loss(gamma=2.0, alpha=0.25):
            """
            Binary form of focal loss.
            """
            def binary_focal_loss_fixed(y_true, y_pred):
                # Clip the prediction value to prevent extreme cases
                epsilon = K.epsilon()
                y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
                
                # Calculate cross entropy
                cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
                
                # Calculate focal loss
                loss = alpha * K.pow(1 - y_pred, gamma) * y_true * K.log(y_pred) - \
                       (1 - alpha) * K.pow(y_pred, gamma) * (1 - y_true) * K.log(1 - y_pred)
                
                return -K.mean(loss)
            return binary_focal_loss_fixed
        
        focal_loss_fn = binary_focal_loss()
        focal_loss_fn.__name__ = 'binary_focal_loss_fixed'
        custom_objects['binary_focal_loss_fixed'] = focal_loss_fn
        
        # Define recall-oriented loss function
        def get_recall_oriented_loss(false_neg_weight=5.0, false_pos_weight=1.0):
            """
            Create a custom loss function that prioritizes recall over precision.
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
                weights = y_true * false_neg_weight + (1 - y_true) * false_pos_weight
                
                # Combine weights with base loss
                weighted_loss = weights * bce
                
                # Return mean loss
                return K.mean(weighted_loss)
            
            # Include parameters in function name for better tracking
            recall_oriented_loss.__name__ = f'recall_loss_fn{false_neg_weight}_fp{false_pos_weight}'
            return recall_oriented_loss
            
        # Add the specific recall-oriented loss function
        recall_loss_fn = get_recall_oriented_loss(5.0, 1.0)
        custom_objects[recall_loss_fn.__name__] = recall_loss_fn
        
        # Load the model with custom objects
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        logging.info(f"Loaded peak model: {model_path}")
        
        # Try to load feature list from test_model directory first, then fall back to parent
        feature_paths = [
            model_dir / "feature_list_peak_model.json",  # Test model directory
            PEAK_MODEL_DIR / "feature_list_peak_model.json",  # Try global constant path
            VALLEY_MODEL_DIR / "feature_list_valley_model.json"  # Fallback to valley model's list as last resort
        ]
        
        feature_names = None
        for path in feature_paths:
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        feature_names = json.load(f)
                    logging.info(f"Loaded feature list with {len(feature_names)} features from {path}")
                    break
                except Exception as e:
                    logging.warning(f"Error loading feature list from {path}: {e}")
        
        if feature_names is None:
            logging.warning("Could not load feature list from any path. Will try to infer from model input shape.")
            # Try to infer feature count from model
            if hasattr(model, 'input_shape'):
                feature_count = model.input_shape[-1]
                logging.info(f"Inferred {feature_count} features from model input shape")
                # Create generic feature names
                feature_names = [f"feature_{i}" for i in range(feature_count)]
            else:
                raise ValueError("Could not determine feature list and model doesn't have input_shape")
        
        # Try to load feature scaler from test_model directory first, then fall back to parent
        scaler_paths = [
            model_dir / "feature_scaler_peak_model.save",  # Test model directory
            PEAK_MODEL_DIR / "feature_scaler_peak_model.save",  # Try global constant path
            VALLEY_MODEL_DIR / "feature_scaler_valley_model.save"  # Fallback to valley model's scaler as last resort
        ]
        
        feature_scaler = None
        for path in scaler_paths:
            if path.exists():
                try:
                    feature_scaler = joblib.load(path)
                    logging.info(f"Loaded feature scaler from {path}")
                    break
                except Exception as e:
                    logging.warning(f"Error loading feature scaler from {path}: {e}")
        
        if feature_scaler is None:
            logging.warning("Could not load any feature scaler. Will standardize data manually if needed.")
        
        return {
            'model': model,
            'feature_names': feature_names,
            'feature_scaler': feature_scaler,
            'prob_threshold': prob_threshold
        }
        
    except Exception as e:
        logging.error(f"Error loading peak model: {e}")
        traceback.print_exc()
        raise

def evaluate_peak_model(df, model_artifacts, num_weeks=3, use_test_data=False, prob_threshold=None, find_optimal_threshold=False):
    """
    Evaluate the peak detection model on validation or test data.
    
    Args:
        df: DataFrame with price data
        model_artifacts: Dictionary containing model artifacts
        num_weeks: Number of weeks to generate plots for
        use_test_data: Whether to use test data instead of validation data
        prob_threshold: Probability threshold for peak detection
        find_optimal_threshold: Whether to find optimal threshold using F1 score
        
    Returns:
        Dictionary containing evaluation metrics
    """
    logging.info("Evaluating peak detection model...")
    
    # Unpack model artifacts
    model = model_artifacts['model']
    feature_scaler = model_artifacts['feature_scaler']
    feature_names = model_artifacts['feature_names']
    
    # Use validation or test data
    if use_test_data:
        logging.info("Evaluating on test data...")
        # Use last 20% of data for testing
        test_size = int(len(df) * 0.2)
        eval_df = df.iloc[-test_size:].copy()
        eval_name = "test"
    else:
        logging.info("Evaluating on validation data...")
        # Use data from 60% to 80% mark for validation
        val_start = int(len(df) * 0.6)
        val_end = int(len(df) * 0.8)
        eval_df = df.iloc[val_start:val_end].copy()
        eval_name = "validation"
    
    logging.info(f"Evaluation data: {len(eval_df)} samples from {eval_df.index[0]} to {eval_df.index[-1]}")
    
    # Add peak features if needed
    eval_df = add_peak_features(eval_df)
    
    # Check for missing features in the feature list
    missing_features = [f for f in feature_names if f not in eval_df.columns]
    if missing_features:
        logging.warning(f"Missing {len(missing_features)} features: {missing_features}")
        # Add missing features with default values (zeros)
        for feature in missing_features:
            eval_df[feature] = 0
            logging.info(f"Added missing feature '{feature}' with default values")
    
    # Extract inputs and targets
    X = eval_df[feature_names].values
    y_true = eval_df['is_price_peak'].values
    
    # Scale features
    X_scaled = feature_scaler.transform(X)
    
    # Create sequences for TCN model - CHANGED to use 168 for peak model (7 days)
    from utils import create_sequences
    PEAK_LOOKBACK_WINDOW = 168  # 7 days of hourly data for peak model
    
    # Make sure we have enough data to create sequences
    if len(X_scaled) <= PEAK_LOOKBACK_WINDOW:
        logging.error(f"Not enough data for peak model evaluation. Need at least {PEAK_LOOKBACK_WINDOW+1} samples.")
        return None
    
    logging.info(f"Creating sequences with lookback window of {PEAK_LOOKBACK_WINDOW}")
    X_seq, _ = create_sequences(
        X_scaled, 
        PEAK_LOOKBACK_WINDOW,  # Use 168 (7 days) for peak model
        1,   # prediction horizon
        list(range(X_scaled.shape[1])), 
        None
    )
    
    # Get actual labels for the sequence data - adjust for the longer window
    y_true_seq = y_true[PEAK_LOOKBACK_WINDOW:PEAK_LOOKBACK_WINDOW+len(X_seq)]
    
    # Get timestamps for the sequence data - adjust for the longer window
    timestamps = eval_df.index[PEAK_LOOKBACK_WINDOW:PEAK_LOOKBACK_WINDOW+len(X_seq)]
    
    # Get price data for plotting - adjust for the longer window
    prices = eval_df[TARGET_VARIABLE].values[PEAK_LOOKBACK_WINDOW:PEAK_LOOKBACK_WINDOW+len(X_seq)]
    
    # Make predictions
    logging.info(f"Making predictions with model on {len(X_seq)} samples...")
    logging.info(f"Input shape: {X_seq.shape}")
    
    try:
        y_prob = model.predict(X_seq, verbose=0) # <-- Added verbose=0
    except Exception as e:
        logging.error(f"Error making predictions: {e}")
        # Try to get model's expected input shape
        if hasattr(model, 'input_shape'):
            expected_shape = model.input_shape
            actual_shape = X_seq.shape
            logging.error(f"Model expects input shape {expected_shape}, but got {actual_shape}")
        # Handle the error gracefully
        logging.error("Model prediction failed. This could be due to incompatible features or model issues.")
        return None
    
    logging.info(f"Successfully made predictions with shape {y_prob.shape}")
    
    # Determine the effective probability threshold
    effective_threshold = None
    probabilities_1d = y_prob.ravel() if y_prob.ndim > 1 else y_prob # Ensure 1D for sklearn metrics

    if find_optimal_threshold:
        from sklearn.metrics import precision_recall_curve, f1_score as f1_sklearn_score
        precision_vals, recall_vals, pr_thresholds_curve = precision_recall_curve(y_true_seq, probabilities_1d)
        
        f1_scores_list = []
        if len(pr_thresholds_curve) > 0:
            for thresh in pr_thresholds_curve:
                y_pred_at_thresh = (probabilities_1d >= thresh).astype(int)
                f1 = f1_sklearn_score(y_true_seq, y_pred_at_thresh, zero_division=0)
                f1_scores_list.append({'threshold': thresh, 'f1': f1})
        
        if f1_scores_list: # Check if any F1 scores were calculated
            best_result = max(f1_scores_list, key=lambda x: x['f1'])
            effective_threshold = best_result['threshold']
            logging.info(f"Dynamically found optimal peak threshold on evaluation data: {effective_threshold:.4f} (F1: {best_result['f1']:.4f})")
        else:
            logging.warning("Could not determine optimal peak threshold dynamically (no valid PR thresholds or F1 scores). Falling back.")
            # Fallback if dynamic optimization fails or PR curve is degenerate
            if prob_threshold is not None: # prob_threshold is CLI arg (args.peak_threshold)
                effective_threshold = prob_threshold
                logging.info(f"Using command-line specified peak threshold after dynamic optimization failed: {effective_threshold:.4f}")
            else:
                effective_threshold = model_artifacts.get('prob_threshold', 0.5) # From loaded model or default 0.5
                logging.info(f"Using peak threshold from model artifacts (or default 0.5) after dynamic optimization failed: {effective_threshold:.4f}")
    
    # This 'else' block is executed if find_optimal_threshold is False.
    # It contains the logic for when CLI threshold is given, or when to use artifacts.
    else: 
        if prob_threshold is not None: # CLI threshold provided
            effective_threshold = prob_threshold
            logging.info(f"Using command-line specified probability threshold for peaks: {effective_threshold:.4f}")
        else:
            # No CLI threshold, not optimizing dynamically. Use from artifacts.
            effective_threshold = model_artifacts.get('prob_threshold') 
            if effective_threshold is not None:
                 logging.info(f"Using probability threshold from loaded peak model artifacts: {effective_threshold:.4f}")
            else: # Should not be strictly necessary if load_peak_model ensures a value
                effective_threshold = 0.5 # Ultimate fallback
                logging.warning(f"Emergency fallback: Peak threshold not specified via CLI, not in artifacts. Defaulting to {effective_threshold:.4f}")
        
    y_pred = (probabilities_1d >= effective_threshold).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(y_true_seq, y_pred)
    precision = precision_score(y_true_seq, y_pred, zero_division=0)
    recall = recall_score(y_true_seq, y_pred, zero_division=0)
    f1 = f1_score(y_true_seq, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true_seq, probabilities_1d) 
    
    positive_rate = np.mean(y_true_seq)
    predicted_rate = np.mean(y_pred)
    
    logging.info(f"Peak detection {eval_name} metrics (threshold={effective_threshold:.4f}):")
    logging.info(f"  Accuracy: {accuracy:.4f}")
    logging.info(f"  Precision: {precision:.4f}")
    logging.info(f"  Recall: {recall:.4f}")
    logging.info(f"  F1 Score: {f1:.4f}")
    logging.info(f"  ROC AUC: {roc_auc:.4f}")
    logging.info(f"  Actual peak rate: {positive_rate:.4f} ({np.sum(y_true_seq)} peaks)")
    logging.info(f"  Predicted peak rate: {predicted_rate:.4f} ({np.sum(y_pred)} peaks)")
    
    output_dir = generate_weekly_peak_plots(
        timestamps, y_pred, probabilities_1d, y_true_seq, prices,
        eval_name=eval_name,
        num_weeks=num_weeks
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'actual_peak_rate': float(positive_rate),
        'predicted_peak_rate': float(predicted_rate),
        'threshold': float(effective_threshold),
        'plots_dir': str(output_dir)
    }
    
    return metrics

def add_peak_features(df, enable_logging=True):
    """
    Add necessary features for peak detection model.
    
    Args:
        df: DataFrame with base data
        enable_logging: Whether to log feature addition messages
        
    Returns:
        DataFrame with added peak detection features
    """
    if enable_logging:
        logging.info("Adding peak detection features...") # Standardized message

    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Ensure required columns exist
    if 'is_price_peak' not in result.columns:
        # Use the Scipy method for peak detection as established
        prices = result[TARGET_VARIABLE].values
        price_range = np.max(prices) - np.min(prices)
        
        from scipy.signal import find_peaks
        simple_peaks, _ = find_peaks(prices, 
                                   distance=4,                  # At least 4 hours between peaks
                                   prominence=0.03*price_range, # Minimum prominence relative to price range
                                   width=2)                     # Minimum width of 2 hours
        
        result['is_price_peak'] = 0
        result.iloc[simple_peaks, result.columns.get_loc('is_price_peak')] = 1
    
    # Add time-based features that help with peak detection
    if 'hour' not in result.columns and isinstance(result.index, pd.DatetimeIndex):
        result['hour'] = result.index.hour
        result['dayofweek'] = result.index.dayofweek
        result['month'] = result.index.month
        result['is_business_hour'] = ((result['hour'] >= 8) & (result['hour'] <= 18) & 
                                    (result['dayofweek'] < 5)).astype(int)
        result['is_morning_peak'] = ((result['hour'] >= 7) & (result['hour'] <= 9)).astype(int)
        result['is_evening_peak'] = ((result['hour'] >= 17) & (result['hour'] <= 20)).astype(int)
    
    # Add momentum/difference features (these were missing)
    result['price_diff_1h'] = result[TARGET_VARIABLE].diff(1)
    result['price_diff_3h'] = result[TARGET_VARIABLE].diff(3)
    result['price_diff_6h'] = result[TARGET_VARIABLE].diff(6)
    
    # Calculate price momentum (acceleration of price changes)
    result['price_momentum'] = result['price_diff_1h'] - result['price_diff_1h'].shift(1)
    
    # Create a detrended price series based on 24h moving average
    result['price_detrended'] = result[TARGET_VARIABLE] - result[TARGET_VARIABLE].rolling(window=24, center=True).mean()
    
    # Add relative price features
    result['price_vs_24h_avg'] = result[TARGET_VARIABLE] / result[TARGET_VARIABLE].rolling(24, center=True).mean()
    result['price_vs_7d_avg'] = result[TARGET_VARIABLE] / result[TARGET_VARIABLE].rolling(168, center=True).mean()
    
    # Add rate of change features
    result['price_roc_1h'] = result[TARGET_VARIABLE].pct_change(1)
    result['price_roc_3h'] = result[TARGET_VARIABLE].pct_change(3)
    
    # Fill missing values
    for col in result.columns:
        if result[col].isna().any():
            if np.issubdtype(result[col].dtype, np.number):
                # Use fill methods that don't generate deprecation warnings
                result[col] = result[col].ffill().bfill().fillna(0)
    
    return result

def generate_weekly_peak_plots(timestamps, predictions, probabilities, actuals, prices, eval_name="validation", num_weeks=3):
    """
    Generate weekly visualizations of peak predictions.
    
    Args:
        timestamps: Array of timestamps
        predictions: Array of binary predictions (0/1)
        probabilities: Array of prediction probabilities (0-1)
        actuals: Array of actual labels (0/1)
        prices: Array of price values
        eval_name: Name of the evaluation data ("validation" or "test")
        num_weeks: Number of weeks to generate plots for
    """
    # Create output directory
    output_dir = PEAK_EVAL_DIR / eval_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure all arrays are 1-dimensional
    predictions = np.squeeze(predictions).flatten()
    probabilities = np.squeeze(probabilities).flatten()
    actuals = np.squeeze(actuals).flatten()
    prices = np.squeeze(prices).flatten()
    
    # Check dimensions for debugging
    logging.info(f"Data dimensions: timestamps {len(timestamps)}, predictions {predictions.shape}, " 
                 f"probabilities {probabilities.shape}, actuals {actuals.shape}, prices {prices.shape}")
    
    # Group by week
    df = pd.DataFrame({
        'timestamp': timestamps,
        'prediction': predictions,
        'probability': probabilities,
        'actual': actuals,
        'price': prices
    })
    
    # Add week start column
    df['week_start'] = df['timestamp'].apply(lambda x: (x.floor('D') - timedelta(days=x.weekday())))
    df['week_key'] = df['week_start'].dt.strftime('%Y-%m-%d')
    
    # Get list of weeks and sort them
    weeks = sorted(df['week_key'].unique())
    
    # Calculate spacing between weeks
    total_weeks = len(weeks)
    if total_weeks <= num_weeks:
        # If we have fewer weeks than requested, use all weeks
        selected_weeks = weeks
    else:
        # Calculate spacing to evenly distribute weeks
        spacing = total_weeks // num_weeks
        selected_weeks = weeks[::spacing][:num_weeks]
    
    logging.info(f"Generating plots for {len(selected_weeks)} weeks out of {total_weeks} total weeks")
    
    # Process each selected week
    for week_key in selected_weeks:
        # Get data for this week
        week_data = df[df['week_key'] == week_key].copy()
        
        # Skip if we have less than 24 hours of data
        if len(week_data) < 24:
            logging.warning(f"Skipping week {week_key} - insufficient data (only {len(week_data)} hours)")
            continue
        
        # Sort by timestamp
        week_data = week_data.sort_values('timestamp')
        
        # Create the weekly plot
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot price data
        ax.step(week_data['timestamp'], week_data['price'], 'b-', linewidth=2, label='Price')
        
        # Highlight actual peaks
        for idx, row in week_data[week_data['actual'] == 1].iterrows():
            ax.axvspan(row['timestamp'] - timedelta(minutes=30), 
                       row['timestamp'] + timedelta(minutes=30),
                       color='red', alpha=0.3, label='_' if idx > week_data.index[0] else 'Actual Peak')
        
        # Mark predicted peaks with color and size based on probability
        predicted_peaks = week_data[week_data['prediction'] == 1]
        if len(predicted_peaks) > 0:
            # Create color map based on probability
            cmap = plt.cm.viridis  # Use a perceptually uniform colormap
            norm = plt.Normalize(vmin=0.5, vmax=1.0)  # Normalize probability from 0.5-1.0
            
            # Plot predicted peaks with varying marker sizes and colors based on probability
            for idx, row in predicted_peaks.iterrows():
                marker_size = 40 + int(row['probability'] * 160)  # Scale: 40-200 based on probability
                color = cmap(norm(row['probability']))
                ax.scatter(row['timestamp'], row['price'], marker='^', color=color, s=marker_size, 
                          edgecolors='black', linewidth=1,
                          label='_' if idx > predicted_peaks.index[0] else 'Predicted Peak')
                
                # Add annotation with probability value
                ax.annotate(f'{row["probability"]:.2f}', 
                           (row['timestamp'], row['price']),
                           xytext=(0, 20),
                           textcoords='offset points',
                           ha='center',
                           fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='black', alpha=0.7))
                
            # Add a colorbar to explain probability->color mapping
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.05)
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label('Probability (Height Score)', fontsize=10)
            
        # Calculate metrics for this week
        week_actuals = week_data['actual'].values
        week_predictions = week_data['prediction'].values
        
        true_positives = np.sum((week_predictions == 1) & (week_actuals == 1))
        false_positives = np.sum((week_predictions == 1) & (week_actuals == 0))
        false_negatives = np.sum((week_predictions == 0) & (week_actuals == 1))
        
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        
        # Add title with metrics
        ax.set_title(f'Week starting {week_key} ({eval_name} Data)\n' +
                    f'Peaks: {np.sum(week_actuals)} actual, {np.sum(week_predictions)} predicted | ' +
                    f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}',
                    fontsize=14)
        
        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (öre/kWh)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%a %d %b'))
        ax.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator())
        
        # Add day separators
        week_start = week_data['week_start'].iloc[0]
        for day in pd.date_range(start=week_start, periods=8, freq='D'):
            ax.axvline(day, color='gray', linestyle='--', alpha=0.5)
        
        # Annotate all actual peaks with price
        for idx, row in week_data[week_data['actual'] == 1].iterrows():
            ax.annotate(f'{row["price"]:.1f}', 
                       (row['timestamp'], row['price']),
                       xytext=(0, 15),
                       textcoords='offset points',
                       ha='center',
                       fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7))
        
        # Mark true positives, false positives, and false negatives
        for idx, row in week_data.iterrows():
            if row['prediction'] == 1 and row['actual'] == 1:  # True positive
                ax.scatter(row['timestamp'], row['price'], marker='o', color='green', s=120, 
                          label='_' if idx > week_data.index[0] else 'True Positive')
            elif row['prediction'] == 1 and row['actual'] == 0:  # False positive
                ax.scatter(row['timestamp'], row['price'], marker='x', color='red', s=120, 
                          label='_' if idx > week_data.index[0] else 'False Positive')
            elif row['prediction'] == 0 and row['actual'] == 1:  # False negative
                ax.scatter(row['timestamp'], row['price'], marker='s', color='orange', s=120, 
                          label='_' if idx > week_data.index[0] else 'False Negative')
        
        # Add legend with better organization and more details
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / f"peak_week_{week_key}.png", dpi=300)
        plt.close()
        
        logging.info(f"Generated visualization for week starting {week_key}")
    
    return output_dir

def evaluate_merged_models(data, trend_artifacts, peak_artifacts=None, valley_artifacts=None, 
                          use_test_data=False, peak_threshold=0.5, valley_threshold=0.4):
    """
    Simplified version of the merged model evaluation that makes it clear how peak and valley 
    predictions are combined with the trend model.
    
    Args:
        data: DataFrame with electricity price data
        trend_artifacts: Dictionary with trend model artifacts
        peak_artifacts: Dictionary with peak model artifacts (optional)
        valley_artifacts: Dictionary with valley model artifacts (optional)
        use_test_data: Whether to use test data instead of validation data
        peak_threshold: Probability threshold for peak detection
        valley_threshold: Probability threshold for valley detection
        
    Returns:
        DataFrame with the predictions and probabilities
    """
    logging.info("Starting simplified merged model evaluation...")
    
    # Copy the data
    df = data.copy()
    
    # Determine which subset to use
    subset_name = "test" if use_test_data else "validation"
    
    # Use a specific date range for more focused analysis
    if use_test_data:
        # Use the last 20% of data for test
        total_rows = len(df)
        test_size = int(total_rows * 0.2)
        df = df.iloc[-test_size:].copy()
    else:
        # Use the next-to-last 20% for validation
        total_rows = len(df)
        test_size = int(total_rows * 0.2)
        start_idx = total_rows - 2 * test_size
        end_idx = total_rows - test_size
        df = df.iloc[start_idx:end_idx].copy()
    
    logging.info(f"Using {subset_name} data with {len(df)} rows from {df.index[0]} to {df.index[-1]}")
    
    # ==================================================================================
    # 1. GET TREND PREDICTIONS
    # ==================================================================================
    print("\n" + "=" * 30)
    print("1. Getting trend predictions")
    print("=" * 30)
    # Add time features needed for trend prediction
    df = add_cyclical_time_features(df)
    
    # Add additional time features needed for the model
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_business_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 18) & 
                             (df['dayofweek'] < 5)).astype(int)
    df['is_morning_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_evening_peak'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    logging.info("Added cyclical time features")
    
    # Get trend model components
    trend_model = trend_artifacts['model']
    # In evaluate.py, feature names are stored under 'feature_order' key
    trend_feature_names = trend_artifacts.get('feature_order', [])
    if not trend_feature_names and 'features' in trend_artifacts:
        trend_feature_names = trend_artifacts['features']
    
    logging.info(f"Using {len(trend_feature_names)} features for trend prediction")
    
    # Prepare data for trend prediction
    df_trend = df[trend_feature_names].copy()
    
    # Make trend predictions - use set_param directly
    try:
        trend_model.set_param({'predictor': 'cpu_predictor'})
    except:
        # If setting the parameter fails, just continue
        pass
    
    # Use a DMatrix with matching feature order
    trend_model.feature_names = None
    dtest = xgb.DMatrix(df_trend.values, feature_names=trend_feature_names)
    trend_predictions = trend_model.predict(dtest)
    logging.info("Successfully made trend predictions")
    
    # Apply smoothing to trend predictions for a more stable base
    if USE_ADVANCED_TREND_SMOOTHING:
        logging.info("Applying advanced trend smoothing pipeline to trend predictions")
        trend_smooth = exponential_smooth(trend_predictions, alpha=0.5)
        trend_smooth = median_filter(trend_smooth, window=5)
        trend_smooth = savitzky_golay_filter(trend_smooth, window=11, polyorder=2)
    elif SIMPLE_TREND_MODEL_SMOOTHING_LEVEL in ['light', 'medium', 'heavy', 'daily', 'weekly']:
        logging.info(f"Applying {SIMPLE_TREND_MODEL_SMOOTHING_LEVEL} smoothing level to trend predictions")
        trend_smooth = adaptive_trend_smoothing(trend_predictions, df.index, SIMPLE_TREND_MODEL_SMOOTHING_LEVEL)
    else:
        raise ValueError(f"Unknown smoothing level: {SIMPLE_TREND_MODEL_SMOOTHING_LEVEL}")

    # Store the predictions in the dataframe
    df['trend_prediction'] = trend_predictions
    df['trend_prediction_smooth'] = trend_smooth
    
    # ==================================================================================
    # 2. GET PEAK PREDICTIONS
    # ==================================================================================
    print("\n" + "=" * 30)
    print("2. Getting peak predictions")
    print("=" * 30)
    peak_probabilities = None
    peak_predictions = None
    
    if peak_artifacts:
        try:
            # Add peak features
            df = add_peak_features(df)
            logging.info("Added peak detection features")
            
            # Get peak model components
            peak_model = peak_artifacts['model']
            
            # Compile the model if it's not already compiled
            # This is crucial for TensorFlow models loaded without compilation
            try:
                if hasattr(peak_model, 'compile') and not peak_model._is_compiled:
                    logging.info("Compiling peak model")
                    peak_model.compile(optimizer='adam', loss='binary_crossentropy')
            except Exception as e:
                logging.warning(f"Failed to compile peak model: {e}")
            
            peak_feature_names = peak_artifacts['feature_names']
            
            # Use the correct key for the feature scaler
            peak_scaler = peak_artifacts.get('scaler', peak_artifacts.get('feature_scaler', None))
            
            # Extract peak features directly from dataframe
            feature_data = df[peak_feature_names].values
            
            # Scale features if scaler is available
            if peak_scaler:
                feature_data = peak_scaler.transform(feature_data)
                logging.info(f"Scaled peak features with shape {feature_data.shape}")
            else:
                logging.warning("Peak model scaler not found, using unscaled features")
            
            # Create sequences with lookback window of 168 hours
            window_size = 168
            num_samples = len(feature_data) - window_size + 1
            
            logging.info(f"Creating {num_samples} sequences with window size {window_size}")
            
            # Initialize array for sequences
            X_peak = np.zeros((num_samples, window_size, len(peak_feature_names)))
            
            # Fill sequences
            for i in range(num_samples):
                X_peak[i] = feature_data[i:i+window_size]
            
            # Make predictions
            logging.info(f"Making peak predictions with {X_peak.shape}")
            peak_probabilities = peak_model.predict(X_peak, verbose=0) # <-- Added verbose=0
            
            # Ensure we get a 1D array
            peak_probabilities = peak_probabilities.flatten()
            
            # Print probability distribution
            logging.info(f"Peak probability range: {peak_probabilities.min():.4f} to {peak_probabilities.max():.4f}")
            
            # Apply threshold to get binary predictions
            peak_threshold_debug = peak_threshold  # Use the threshold from argument
            peak_predictions = (peak_probabilities >= peak_threshold_debug).astype(int)
            
            # Create empty columns for peak predictions
            df['peak_probability'] = np.nan
            df['peak_prediction'] = np.nan
            
            # Get the valid range for adding the predictions back to dataframe
            start_idx = window_size - 1
            end_idx = start_idx + len(peak_probabilities)
            
            # Ensure we don't go beyond dataframe bounds
            if end_idx > len(df):
                logging.warning(f"Truncating predictions to fit dataframe (from {len(peak_probabilities)} to {len(df) - start_idx})")
                peak_probabilities = peak_probabilities[:len(df) - start_idx]
                peak_predictions = peak_predictions[:len(df) - start_idx]
                end_idx = len(df)
            
            # Assign to dataframe
            probability_idx = df.columns.get_indexer(['peak_probability'])[0]
            prediction_idx = df.columns.get_indexer(['peak_prediction'])[0]
            
            for i in range(len(peak_probabilities)):
                df.iloc[start_idx + i, probability_idx] = peak_probabilities[i]
                df.iloc[start_idx + i, prediction_idx] = peak_predictions[i]
            
            # If no peaks were detected, use a simple rule-based approach as fallback
            if np.sum(peak_predictions) == 0:
                logging.warning("No peaks detected by model, using rule-based fallback detection")
                # Get the trend predictions from the range where we have predictions
                trend_preds = trend_predictions[start_idx:end_idx]
                
                # Simple rule: Mark the top 10% of prices as peaks
                threshold = np.percentile(trend_preds, 90)
                fallback_peaks = (trend_preds >= threshold).astype(int)
                
                # Update the peak predictions
                for i in range(len(fallback_peaks)):
                    if fallback_peaks[i] == 1:
                        df.iloc[start_idx + i, prediction_idx] = 1
                        df.iloc[start_idx + i, probability_idx] = 0.9  # Artificial high probability
                
                logging.info(f"Added {np.sum(fallback_peaks)} fallback peaks using price threshold {threshold:.2f}")
            else:
                logging.info(f"Added peak predictions with {np.sum(peak_predictions)} peaks detected using threshold {peak_threshold_debug}")
            
        except Exception as e:
            logging.error(f"Error in peak prediction: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
    
    # ==================================================================================
    # 3. GET VALLEY PREDICTIONS
    # ==================================================================================
    print("\n" + "=" * 30)
    print("3. Getting valley predictions")
    print("=" * 30)
    valley_probabilities = None
    valley_predictions = None
    
    if valley_artifacts:
        try:
            # Add valley features
            df = add_valley_features(df)
            logging.info("Added valley detection features")
            
            # Get valley model components
            valley_model = valley_artifacts['model']
            
            # Compile the model if it's not already compiled
            try:
                if hasattr(valley_model, 'compile') and not valley_model._is_compiled:
                    logging.info("Compiling valley model")
                    valley_model.compile(optimizer='adam', loss='binary_crossentropy')
            except Exception as e:
                logging.warning(f"Failed to compile valley model: {e}")
                
            valley_feature_names = valley_artifacts['feature_names']
            
            # Use the correct key for the feature scaler
            valley_scaler = valley_artifacts.get('scaler', valley_artifacts.get('feature_scaler', None))
            
            # Extract valley features directly from dataframe
            feature_data = df[valley_feature_names].values
            
            # Scale features if scaler is available
            if valley_scaler:
                feature_data = valley_scaler.transform(feature_data)
                logging.info(f"Scaled valley features with shape {feature_data.shape}")
            else:
                logging.warning("Valley model scaler not found, using unscaled features")
            
            # Create sequences with lookback window of 168 hours
            window_size = 168
            num_samples = len(feature_data) - window_size + 1
            
            logging.info(f"Creating {num_samples} sequences with window size {window_size}")
            
            # Initialize array for sequences
            X_valley = np.zeros((num_samples, window_size, len(valley_feature_names)))
            
            # Fill sequences
            for i in range(num_samples):
                X_valley[i] = feature_data[i:i+window_size]
            
            # Make predictions
            logging.info(f"Making valley predictions with {X_valley.shape}")
            valley_probabilities = valley_model.predict(X_valley, verbose=0) # <-- Added verbose=0
            
            # Ensure we get a 1D array
            valley_probabilities = valley_probabilities.flatten()
            
            # Print probability distribution
            logging.info(f"Valley probability range: {valley_probabilities.min():.4f} to {valley_probabilities.max():.4f}")
            
            # Apply threshold to get binary predictions
            valley_threshold_debug = valley_threshold  # Use the threshold from argument
            valley_predictions = (valley_probabilities >= valley_threshold_debug).astype(int)
            
            # Create empty columns for valley predictions
            df['valley_probability'] = np.nan
            df['valley_prediction'] = np.nan
            
            # Get the valid range for adding the predictions back to dataframe
            start_idx = window_size - 1
            end_idx = start_idx + len(valley_probabilities)
            
            # Ensure we don't go beyond dataframe bounds
            if end_idx > len(df):
                logging.warning(f"Truncating predictions to fit dataframe (from {len(valley_probabilities)} to {len(df) - start_idx})")
                valley_probabilities = valley_probabilities[:len(df) - start_idx]
                valley_predictions = valley_predictions[:len(df) - start_idx]
                end_idx = len(df)
            
            # Assign to dataframe
            probability_idx = df.columns.get_indexer(['valley_probability'])[0]
            prediction_idx = df.columns.get_indexer(['valley_prediction'])[0]
            
            for i in range(len(valley_probabilities)):
                df.iloc[start_idx + i, probability_idx] = valley_probabilities[i]
                df.iloc[start_idx + i, prediction_idx] = valley_predictions[i]
            
            # If no valleys were detected, use a simple rule-based approach as fallback
            if np.sum(valley_predictions) == 0:
                logging.warning("No valleys detected by model, using rule-based fallback detection")
                # Get the trend predictions from the range where we have predictions
                trend_preds = trend_predictions[start_idx:end_idx]
                
                # Simple rule: Mark the bottom 10% of prices as valleys
                threshold = np.percentile(trend_preds, 10)
                fallback_valleys = (trend_preds <= threshold).astype(int)
                
                # Update the valley predictions
                for i in range(len(fallback_valleys)):
                    if fallback_valleys[i] == 1:
                        df.iloc[start_idx + i, prediction_idx] = 1
                        df.iloc[start_idx + i, probability_idx] = 0.9  # Artificial high probability
                
                logging.info(f"Added {np.sum(fallback_valleys)} fallback valleys using price threshold {threshold:.2f}")
            else:
                logging.info(f"Added valley predictions with {np.sum(valley_predictions)} valleys detected using threshold {valley_threshold_debug}")
            
        except Exception as e:
            logging.error(f"Error in valley prediction: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
    
    # ==================================================================================
    # 4. MERGE PREDICTIONS
    # ==================================================================================
    print("\n" + "=" * 30)
    print("4. Merging predictions")
    print("=" * 30)
    logging.info(f"Simple merging complete - created predictions for {len(df)} samples")
    
    # Copy trend predictions to result column
    df['predicted_price'] = df['trend_prediction_smooth']
    
    # Count conflicts (both peak and valley)
    conflicts = 0
    peak_count = 0
    valley_count = 0
    
    # Calculate average peak and valley amplitudes for scaling the effects
    peak_amplitude = 0.8 * df[TARGET_VARIABLE].mean()  # 20% of average price
    valley_amplitude = 0.8 * df[TARGET_VARIABLE].mean()  # 20% of average price
    
    # Process each row where we have predictions
    for i in range(len(df)):
        # Skip rows without predictions
        if pd.isna(df.iloc[i]['peak_prediction']) or pd.isna(df.iloc[i]['valley_prediction']):
            continue
            
        is_peak = df.iloc[i]['peak_prediction'] == 1
        is_valley = df.iloc[i]['valley_prediction'] == 1
        
        # Check for conflicts
        if is_peak and is_valley:
            conflicts += 1
            # Resolve based on probability
            peak_prob = df.iloc[i]['peak_probability']
            valley_prob = df.iloc[i]['valley_probability']
            
            if peak_prob > valley_prob:
                is_valley = False
            else:
                is_peak = False
                
        # Apply peak effect
        if is_peak:
            peak_count += 1
            peak_prob = float(df.iloc[i]['peak_probability'])
            # Scale effect by probability (minimum 75% effect)
            effect = peak_amplitude * max(0.75, peak_prob)
            # Use iloc instead of loc to avoid indexing issues
            price_idx = df.columns.get_indexer(['predicted_price'])[0]
            df.iloc[i, price_idx] += effect
            
        # Apply valley effect
        if is_valley:
            valley_count += 1
            valley_prob = float(df.iloc[i]['valley_probability'])
            # Scale effect by probability (minimum 75% effect)
            effect = valley_amplitude * max(0.75, valley_prob)
            # Use iloc instead of loc to avoid indexing issues
            price_idx = df.columns.get_indexer(['predicted_price'])[0]  
            df.iloc[i, price_idx] -= effect
    
    if conflicts > 0:
        logging.info(f"Resolved {conflicts} conflicts (both peak and valley predicted)")
        
    logging.info(f"Applied {peak_count} peak effects and {valley_count} valley effects to trend predictions")
    
    # ==================================================================================
    # 5. GENERATE PLOTS
    # ==================================================================================
    print("\n" + "=" * 30)
    print("5. Generating plots")
    print("=" * 30)
    # Define output directory
    output_dir = EVALUATION_DIR / "merged" / subset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate weekly plots instead of one large summary plot
    logging.info(f"Generating weekly plots for merged model...")
    
    # Determine the number of weeks to plot (limit to 10 for clarity)
    total_hours = len(df)
    hours_per_week = 24 * 7
    num_weeks = min(10, total_hours // hours_per_week)
    
    for week in range(num_weeks):
        # Calculate start and end indices for this week
        start_idx = week * hours_per_week
        end_idx = min(start_idx + hours_per_week, len(df))
        
        # Get the data for this week
        week_timestamps = df.index[start_idx:end_idx]
        week_actual_prices = df[TARGET_VARIABLE].values[start_idx:end_idx]
        # Use smoothed trend predictions instead of raw predictions for consistency
        week_trend_predictions = df['trend_prediction_smooth'].values[start_idx:end_idx]
        week_merged_predictions = df['predicted_price'].values[start_idx:end_idx]
        
        # Get peak and valley data if available
        peaks = None
        valleys = None
        peak_probabilities = None
        valley_probabilities = None
        
        if 'peak_prediction' in df.columns:
            peaks = df['peak_prediction'].values[start_idx:end_idx]
            peak_probabilities = df['peak_probability'].values[start_idx:end_idx]
        
        if 'valley_prediction' in df.columns:
            valleys = df['valley_prediction'].values[start_idx:end_idx]
            valley_probabilities = df['valley_probability'].values[start_idx:end_idx]
        
        # Create the figure with 3 subplots (as in the example image)
        fig, axs = plt.subplots(3, 1, figsize=(15, 15), sharex=True, 
                                 gridspec_kw={'height_ratios': [1, 1, 0.5]})
        
        # Extract date range for the title
        start_date = pd.to_datetime(week_timestamps[0]).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(week_timestamps[-1]).strftime('%Y-%m-%d')
        
        # -------------------------------------------------------------------------
        # Plot 1: Actual Price and Base Trend Model
        # -------------------------------------------------------------------------
        axs[0].step(week_timestamps, week_actual_prices, 'b-', label='Actual Price', linewidth=2)
        axs[0].step(week_timestamps, week_trend_predictions, 'g-', label='Trend Prediction (Smoothed)', linewidth=2)
        
        axs[0].set_ylabel('Price (öre/kWh)')
        axs[0].set_title(f'Actual Price and Base Trend Model (Smoothed): {start_date} to {end_date}')
        axs[0].legend(loc='upper right')
        axs[0].grid(True, alpha=0.3)
        
        # -------------------------------------------------------------------------
        # Plot 2: Combined Model: Trend with Peak & Valley Volatility
        # -------------------------------------------------------------------------
        axs[1].step(week_timestamps, week_actual_prices, 'k-', label='Actual Price', linewidth=1, alpha=0.5)
        axs[1].step(week_timestamps, week_trend_predictions, 'g-', label='Trend Prediction (Smoothed)', linewidth=1.5)
        axs[1].step(week_timestamps, week_merged_predictions, 'm-', label='Merged Prediction', linewidth=2)
        
        # Add peak markers
        if peaks is not None and np.any(~np.isnan(peaks)):
            peak_mask = peaks == 1
            if np.any(peak_mask):
                axs[1].scatter(
                    week_timestamps[peak_mask], 
                    week_merged_predictions[peak_mask], 
                    color='red', marker='^', s=80, label='Predicted Peaks'
                )
        
        # Add valley markers
        if valleys is not None and np.any(~np.isnan(valleys)):
            valley_mask = valleys == 1
            if np.any(valley_mask):
                axs[1].scatter(
                    week_timestamps[valley_mask], 
                    week_merged_predictions[valley_mask], 
                    color='blue', marker='v', s=80, label='Predicted Valleys'
                )
        
        axs[1].set_ylabel('Price (öre/kWh)')
        axs[1].set_title('Combined Model: Trend with Peak & Valley Volatility')
        axs[1].legend(loc='upper right')
        axs[1].grid(True, alpha=0.3)
        
        # -------------------------------------------------------------------------
        # Plot 3: Peak and Valley Probabilities
        # -------------------------------------------------------------------------
        # Plot peak probabilities
        if peak_probabilities is not None and np.any(~np.isnan(peak_probabilities)):
            axs[2].step(week_timestamps, peak_probabilities, 'r-', label='Peak Probability')
            axs[2].axhline(y=peak_threshold, color='r', linestyle='--', alpha=0.6, 
                          label=f'Peak Threshold ({peak_threshold:.1f})')
            
            # Mark actual peaks
            if peaks is not None and np.any(~np.isnan(peaks)):
                peak_mask = peaks == 1
                if np.any(peak_mask):
                    axs[2].scatter(
                        week_timestamps[peak_mask], 
                        peak_probabilities[peak_mask], 
                        color='r', marker='o', s=40, label='Actual Peaks'
                    )
        
        # Plot valley probabilities
        if valley_probabilities is not None and np.any(~np.isnan(valley_probabilities)):
            axs[2].step(week_timestamps, valley_probabilities, 'b-', label='Valley Probability')
            axs[2].axhline(y=valley_threshold, color='b', linestyle='--', alpha=0.6, 
                          label=f'Valley Threshold ({valley_threshold:.1f})')
            
            # Mark actual valleys
            if valleys is not None and np.any(~np.isnan(valleys)):
                valley_mask = valleys == 1
                if np.any(valley_mask):
                    axs[2].scatter(
                        week_timestamps[valley_mask], 
                        valley_probabilities[valley_mask], 
                        color='b', marker='o', s=40, label='Actual Valleys'
                    )
        
        axs[2].set_ylim(0, 1)
        axs[2].set_ylabel('Probability')
        axs[2].set_title('Peak and Valley Probabilities')
        axs[2].legend(loc='upper right')
        axs[2].grid(True, alpha=0.3)
        
        # Format x-axis with dates
        axs[2].set_xlabel('Date')
        date_format = mdates.DateFormatter('%a %d %b')
        axs[2].xaxis.set_major_formatter(date_format)
        axs[2].xaxis.set_major_locator(mdates.DayLocator())
        plt.xticks(rotation=45)
        
        # Layout and save
        plt.tight_layout()
        plt.savefig(output_dir / f"week_{week+1}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.png", 
                  dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Generated plot for week {week+1}: {start_date} to {end_date}")
    
    logging.info(f"Simple merging complete - created predictions for {len(df)} samples")
    
    return df

def prepare_sequence_data(df, feature_names, window_size=LOOKBACK_WINDOW, stride=1, prediction_horizon=PREDICTION_HORIZON):
    """Prepare sequence data for prediction.
    
    Args:
        df: DataFrame with features
        feature_names: List of feature names to use
        window_size: Size of the lookback window
        stride: Stride between sequences
        prediction_horizon: Number of steps to predict ahead
        
    Returns:
        Numpy array with sequences
    """
    # Ensure all features are in the dataframe
    missing_features = list(set(feature_names) - set(df.columns))
    if missing_features:
        logging.warning(f"Missing features: {missing_features}")
        # This warning should not happen now since we add placeholder features earlier
    
    # Create sequences with the EXACT feature order specified by feature_names
    df_features = pd.DataFrame(index=df.index)
    for feature in feature_names:
        if feature in df.columns:
            df_features[feature] = df[feature]
        else:
            df_features[feature] = 0.0  # Use 0 as default for missing features
    
    # Number of sequences we can create
    n_sequences = (len(df_features) - window_size) // stride + 1
    
    # Create sequences
    sequences = np.zeros((n_sequences, window_size, len(feature_names)))
    
    for i in range(n_sequences):
        start_idx = i * stride
        end_idx = start_idx + window_size
        sequences[i] = df_features.iloc[start_idx:end_idx].values
    
    return sequences, feature_names

def predict_future_prices(model_type, start_date=None, horizon_days=1, peak_threshold=0.5, valley_threshold=0.5):
    """
    Predict future electricity prices using production models.
    Now supports true production prediction for trend, peak, valley, and merged models.
    """
    import matplotlib.dates as mdates
    PREDICTIONS_DIR = PLOTS_DIR / "predictions" / model_type
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Parse start date or use current date
    if start_date is None:
        start_datetime = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_datetime = pd.Timestamp(start_datetime, tz='UTC')
    else:
        try:
            start_datetime = pd.to_datetime(start_date).replace(hour=0, minute=0, second=0, microsecond=0)
            start_datetime = pd.Timestamp(start_datetime, tz='UTC')
        except:
            logging.error(f"Invalid start date format: {start_date}. Using current date.")
            start_datetime = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            start_datetime = pd.Timestamp(start_datetime, tz='UTC')

    logging.info(f"Predicting prices from {start_datetime} for {horizon_days} days")
    horizon_hours = int(horizon_days * 24)
    data = load_data()
    if data is None:
        logging.error("Failed to load data for prediction")
        return None
    logging.info(f"Loaded historical data with shape {data.shape}")

    # Filter data to include only up to start_date
    if start_datetime > data.index.min():
        historical_data = data[data.index < start_datetime].copy()
        if len(historical_data) == 0:
            logging.warning("No historical data before start date. Using all available data.")
            historical_data = data.copy()
    else:
        historical_data = data.copy()
    logging.info(f"Using {len(historical_data)} historical data points")
    latest_timestamp = historical_data.index.max()

    # --- Verification Logging ---
    logging.info(f"Production mode: Filtering historical data up to (exclusive) start_datetime: {start_datetime}")
    if not historical_data.empty:
        max_historical_ts = historical_data.index.max()
        logging.info(f"Production mode: Latest timestamp in filtered historical_data: {max_historical_ts}")
        if max_historical_ts < start_datetime:
            logging.info("Production mode: VERIFIED - historical_data does not contain dates on or after start_datetime.")
        else:
            logging.warning("Production mode: VERIFICATION FAILED - historical_data contains dates on or after start_datetime. This should not happen.")
    else:
        logging.warning("Production mode: historical_data is empty after filtering.")
    # --- End Verification Logging ---

    # Add cyclical and basic time features to historical data
    historical_data = add_cyclical_time_features(historical_data)
    historical_data['hour'] = historical_data.index.hour
    historical_data['dayofweek'] = historical_data.index.dayofweek
    historical_data['month'] = historical_data.index.month
    historical_data['is_business_hour'] = ((historical_data['hour'] >= 8) & (historical_data['hour'] <= 18) & (historical_data['dayofweek'] < 5)).astype(int)
    historical_data['is_morning_peak'] = ((historical_data['hour'] >= 7) & (historical_data['hour'] <= 9)).astype(int)
    historical_data['is_evening_peak'] = ((historical_data['hour'] >= 17) & (historical_data['hour'] <= 20)).astype(int)
    historical_data['is_weekend'] = (historical_data['dayofweek'] >= 5).astype(int)

    # Create prediction timestamps
    prediction_start = max(start_datetime, latest_timestamp + timedelta(hours=1))
    prediction_timestamps = pd.date_range(
        start=prediction_start,
        periods=horizon_hours,
        freq='h',
        tz='UTC'
    )
    logging.info(f"Prediction period: {prediction_timestamps[0]} to {prediction_timestamps[-1]}")

    # Prepare future_df for time features
    future_df = pd.DataFrame(index=prediction_timestamps)
    future_df = add_cyclical_time_features(future_df)
    future_df['hour'] = future_df.index.hour
    future_df['dayofweek'] = future_df.index.dayofweek
    future_df['month'] = future_df.index.month
    future_df['is_business_hour'] = ((future_df['hour'] >= 8) & (future_df['hour'] <= 18) & (future_df['dayofweek'] < 5)).astype(int)
    future_df['is_morning_peak'] = ((future_df['hour'] >= 7) & (future_df['hour'] <= 9)).astype(int)
    future_df['is_evening_peak'] = ((future_df['hour'] >= 17) & (future_df['hour'] <= 20)).astype(int)
    future_df['is_weekend'] = (future_df['dayofweek'] >= 5).astype(int)

    # --- Load models and artifacts ---
    trend_results = None
    peak_results = None
    valley_results = None
    trend_artifacts = None
    peak_artifacts = None
    valley_artifacts = None
    if model_type in ['trend', 'merged']:
        trend_artifacts = load_trend_model(production_mode=True)
    if model_type in ['peak', 'merged']:
        peak_artifacts = load_peak_model(production_mode=True)
    if model_type in ['valley', 'merged']:
        valley_artifacts = load_valley_model(production_mode=True)

    # --- Trend prediction (already true prediction) ---
    if trend_artifacts:
        feature_order = trend_artifacts.get('feature_order', [])
        for feature in feature_order:
            if feature not in future_df.columns:
                if feature in historical_data.columns:
                    future_df[feature] = historical_data[feature].iloc[-1]
                else:
                    future_df[feature] = 0
        X_test = future_df[feature_order].copy()
        model = trend_artifacts['model']
        if trend_artifacts.get('model_type', '').lower() == 'xgboost':
            model.feature_names = None
            dtest = xgb.DMatrix(X_test.values)
            raw_predictions = model.predict(dtest)
            smoothing_level = SIMPLE_TREND_MODEL_SMOOTHING_LEVEL
            smoothed_predictions = adaptive_trend_smoothing(raw_predictions, future_df.index, smoothing_level)
            trend_results = pd.DataFrame({
                'timestamp': prediction_timestamps,
                'predicted_price': raw_predictions,
                'trend_prediction_smooth': smoothed_predictions
            }).set_index('timestamp')

    # --- Peak/Valley prediction (true production, rolling window) ---
    def rolling_sequence_predict(historical_data, future_df, model_artifacts, add_features_fn, target_col, threshold, lookback_window=168):
        scaler = model_artifacts.get('feature_scaler')
        feature_names = model_artifacts['feature_names']
        model = model_artifacts['model']
        window_df = historical_data.copy()
        # Precompute features for the initial window, with logging enabled
        if add_features_fn.__name__ == 'add_peak_features' or add_features_fn.__name__ == 'add_valley_features':
            window_df = add_features_fn(window_df, enable_logging=True)
        else:
            window_df = add_features_fn(window_df) # For other feature functions that might not have the param

        preds = []
        probs = []
        for i, ts in enumerate(future_df.index):
            # Log progress every 24 steps (daily)
            if (i + 1) % 24 == 0 or (i + 1) == len(future_df.index):
                logging.info(f"Predicting for {target_col.capitalize()} - Step {i+1}/{len(future_df.index)} (Timestamp: {ts})")

            # Prepare new row with time features
            new_row = pd.DataFrame(index=[ts])
            for col in window_df.columns:
                if col in future_df.columns:
                    new_row[col] = future_df.loc[ts, col]
                elif col in window_df.columns:
                    new_row[col] = window_df.iloc[-1][col]
                else:
                    new_row[col] = 0
            if TARGET_VARIABLE not in new_row.columns:
                new_row[TARGET_VARIABLE] = window_df[TARGET_VARIABLE].iloc[-1]
            # Append to window
            # Use pd.concat instead of deprecated append
            window_df = pd.concat([window_df, new_row], axis=0, ignore_index=False) 

            # Only keep last lookback_window rows
            window_df = window_df.iloc[-lookback_window:]
            
            # Recompute features for the current window_df, with logging disabled for peak/valley inside the loop
            if add_features_fn.__name__ == 'add_peak_features' or add_features_fn.__name__ == 'add_valley_features':
                processed_window_df = add_features_fn(window_df, enable_logging=False)
            else:
                processed_window_df = add_features_fn(window_df)
            
            # Ensure all features required by the model are present in the processed window
            for f_name in feature_names:
                if f_name not in processed_window_df.columns:
                    # If a feature is missing after processing, add it with a default (e.g., 0 or ffill from historical)
                    # This case should be rare if feature engineering is comprehensive
                    processed_window_df[f_name] = 0 
                    logging.warning(f"Feature '{f_name}' was missing after add_features_fn in rolling_sequence_predict. Added with 0.")

            # Extract features for the last time step from the processed window
            # Ensure we are using the correct feature_names order for the model
            X_features_for_model = processed_window_df[feature_names].values

            if scaler:
                # Scale all features in the window then take the last sequence
                X_scaled_features = scaler.transform(X_features_for_model)
                X_seq = X_scaled_features[-lookback_window:].reshape(1, lookback_window, len(feature_names))
            else:
                # If no scaler, just take the last sequence of raw features
                X_seq = X_features_for_model[-lookback_window:].reshape(1, lookback_window, len(feature_names))
            
            # Make prediction with verbose=0
            prediction_output = model.predict(X_seq, verbose=0) 
            prob = prediction_output[0][0] if hasattr(prediction_output[0], '__len__') and len(prediction_output[0]) == 1 else prediction_output[0]
            pred = int(prob >= threshold)
            preds.append(pred)
            probs.append(prob)
        return pd.DataFrame({
            'timestamp': future_df.index,
            f'{target_col}_probability': probs,
            f'is_predicted_{target_col}': preds
        }).set_index('timestamp')

    # --- Run peak/valley prediction if needed ---
    if peak_artifacts and model_type in ['peak', 'merged']:
        peak_results = rolling_sequence_predict(
            historical_data,
            future_df,
            peak_artifacts,
            add_peak_features,
            'peak',
            peak_threshold,
            lookback_window=168
        )
    if valley_artifacts and model_type in ['valley', 'merged']:
        valley_results = rolling_sequence_predict(
            historical_data,
            future_df,
            valley_artifacts,
            add_valley_features,
            'valley',
            valley_threshold,
            lookback_window=168
        )

    # --- Merged model: combine predictions ---
    merged_results = None
    if model_type == 'merged' and trend_results is not None:
        merged_results = trend_results.copy()
        if peak_results is not None:
            merged_results = merged_results.join(peak_results, how='left')
        if valley_results is not None:
            merged_results = merged_results.join(valley_results, how='left')
        # Adjust price prediction based on peaks/valleys
        merged_results['price_prediction'] = merged_results['trend_prediction_smooth']
        if 'is_predicted_peak' in merged_results.columns and 'peak_probability' in merged_results.columns:
            merged_results.loc[merged_results['is_predicted_peak'] == 1, 'price_prediction'] = \
                merged_results['trend_prediction_smooth'] * (1 + 0.5 * merged_results['peak_probability'])
        if 'is_predicted_valley' in merged_results.columns and 'valley_probability' in merged_results.columns:
            merged_results.loc[merged_results['is_predicted_valley'] == 1, 'price_prediction'] = \
                merged_results['trend_prediction_smooth'] * (1 - 0.3 * merged_results['valley_probability'])

    # --- Plotting ---
    plt.figure(figsize=(15, 10))
    # Plot historical prices (last 48h)
    historical_end = latest_timestamp
    historical_start = historical_end - timedelta(hours=48)
    historical_subset = historical_data[(historical_data.index >= historical_start) & (historical_data.index <= historical_end)]
    if len(historical_subset) > 0:
        plt.step(historical_subset.index, historical_subset[TARGET_VARIABLE], 'b-', linewidth=2, label='Historical Prices')
    # Plot predictions
    if model_type == 'trend' and trend_results is not None:
        results_df = trend_results
        prediction_col = 'trend_prediction_smooth'
    elif model_type == 'peak' and peak_results is not None:
        results_df = peak_results
        prediction_col = None
    elif model_type == 'valley' and valley_results is not None:
        results_df = valley_results
        prediction_col = None
    elif model_type == 'merged' and merged_results is not None:
        results_df = merged_results
        prediction_col = 'price_prediction'
    else:
        results_df = None
        prediction_col = None
    # Plot predicted prices
    if results_df is not None and prediction_col is not None and prediction_col in results_df.columns:
        plt.step(results_df.index, results_df[prediction_col], 'r-', linewidth=2, marker='o', markersize=3, label='Price Prediction')

    # --- PLOT ACTUALS IN PREDICTION PERIOD IF AVAILABLE ---
    # Only for visualization, not for prediction
    try:
        # 'data' is the full loaded DataFrame
        actuals_in_pred_window = data.loc[data.index.intersection(results_df.index)]
        if len(actuals_in_pred_window) > 0:
            plt.step(actuals_in_pred_window.index, actuals_in_pred_window[TARGET_VARIABLE],
                     color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Actual Price (Future, for reference)')
    except Exception as e:
        pass

    # Mark predicted peaks/valleys
    if results_df is not None:
        if 'is_predicted_peak' in results_df.columns:
            peak_idx = results_df.index[results_df['is_predicted_peak'] == 1]
            if len(peak_idx) > 0:
                plt.scatter(peak_idx, results_df.loc[peak_idx, prediction_col if prediction_col else 'peak_probability'],
                            color='red', marker='^', s=120, label='Predicted Peaks', zorder=10)
        if 'is_predicted_valley' in results_df.columns:
            valley_idx = results_df.index[results_df['is_predicted_valley'] == 1]
            if len(valley_idx) > 0:
                plt.scatter(valley_idx, results_df.loc[valley_idx, prediction_col if prediction_col else 'valley_probability'],
                            color='blue', marker='v', s=120, label='Predicted Valleys', zorder=10)
    # Add vertical line at prediction start
    plt.axvline(x=prediction_timestamps[0], color='k', linestyle='--', alpha=0.7)
    plt.text(prediction_timestamps[0], plt.ylim()[0], 'Prediction Start', rotation=90, verticalalignment='bottom')

    # --- IMPROVED X-AXIS AND DAILY SEPARATORS ---
    import matplotlib.dates as mdates
    ax = plt.gca()
    # Set major ticks to daily, minor to hourly
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d %b'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    # Add vertical lines for each day in the prediction period
    all_dates = pd.date_range(start=historical_start.floor('D'), end=results_df.index[-1].ceil('D'), freq='D')
    for day in all_dates:
        plt.axvline(day, color='gray', linestyle=':', alpha=0.4, zorder=0)
    # Show every 3rd hour label for clarity
    hour_labels = []
    hour_ticks = []
    for i, ts in enumerate(results_df.index):
        if ts.hour % 3 == 0 and (i == 0 or ts.date() != results_df.index[i-1].date() or ts.hour == 0):
            hour_labels.append(ts.strftime('%H:%M'))
            hour_ticks.append(ts)
        else:
            hour_labels.append('')
    ax.set_xticks(hour_ticks, minor=True)
    ax.set_xticklabels([ts.strftime('%H:%M') for ts in hour_ticks], minor=True, rotation=90, fontsize=8)
    plt.gcf().autofmt_xdate(rotation=30, ha='right')

    # Format plot
    plt.title(f'Electricity Price Prediction - {horizon_days} Day Horizon ({model_type.capitalize()} Model)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (öre/kWh)', fontsize=12)
    plt.grid(True, alpha=0.3, which='both', axis='both')
    plt.legend(loc='best')
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = PREDICTIONS_DIR / f"prediction_{start_datetime.strftime('%Y%m%d')}_{horizon_days}days_{timestamp_str}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved prediction plot to {output_path}")
    # Save prediction data as CSV
    if results_df is not None:
        csv_path = PREDICTIONS_DIR / f"prediction_{start_datetime.strftime('%Y%m%d')}_{horizon_days}days_{timestamp_str}.csv"
        results_df.to_csv(csv_path)
        logging.info(f"Saved prediction data to {csv_path}")
    # Return the appropriate results
    if model_type == 'trend' and trend_results is not None:
        return trend_results
    elif model_type == 'peak' and peak_results is not None:
        return peak_results
    elif model_type == 'valley' and valley_results is not None:
        return valley_results
    elif model_type == 'merged' and merged_results is not None:
        return merged_results
    else:
        return None

def main():
    args = parse_arguments()
    
    # Configure logging based on model type
    configure_logging(args.model)
    
    # Check if we're in production prediction mode
    if args.production_mode:
        logging.info("RUNNING IN PRODUCTION PREDICTION MODE")
        
        # Parse start date
        start_date = args.start
        if start_date is None:
            start_date = datetime.now().strftime('%Y-%m-%d')
            logging.info(f"No start date provided, using current date: {start_date}")
        
        # Make predictions
        results = predict_future_prices(
            model_type=args.model,
            start_date=start_date,
            horizon_days=args.horizon,
            peak_threshold=args.peak_threshold if args.peak_threshold is not None else 0.5,
            valley_threshold=args.valley_threshold
        )
        
        if results is not None:
            logging.info(f"Successfully generated predictions for {args.horizon} days starting from {start_date}")
            print("\n" + "=" * 80)
            print(f"PRICE PREDICTION COMPLETE ({args.model.upper()} MODEL)")
            print("=" * 80)
            print(f"Start date: {start_date}")
            print(f"Horizon: {args.horizon} day(s)")
            print(f"Results saved to: {PLOTS_DIR / 'predictions' / args.model}")
        else:
            logging.error("Failed to generate predictions")
        
        return
    
    # Regular evaluation mode (existing code)
    # Load data
    data = load_data()
    
    if data is None:
        logging.error("Failed to load data. Exiting.")
        return
    
    if args.model == 'trend':
        # Load trend model
        trend_artifacts = load_trend_model(production_mode=args.production_mode)
        
        # Evaluate trend model
        results = evaluate_trend_model(data, trend_artifacts, num_weeks=args.weeks)
        
        if results:
            logging.info("Trend model evaluation complete")
            
            # Generate summary report
            generate_summary_report(results, 'trend')
            
            print("\n" + "=" * 80)
            print("TREND MODEL EVALUATION COMPLETE")
            print("=" * 80)
            print(f"Results saved to: {TREND_EVAL_DIR}")
            
            # Print key metrics
            print("\nKey performance metrics:")
            for metric, value in results.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    print(f"  {metric}: {value:.4f}")
    
    elif args.model == 'peak':
        # Load peak model
        peak_artifacts = load_peak_model(production_mode=args.production_mode)
        
        # Evaluate peak model
        results = evaluate_peak_model(
            data, 
            peak_artifacts, 
            num_weeks=args.weeks,
            use_test_data=args.test_data,
            prob_threshold=args.peak_threshold,  # Pass args.peak_threshold (can be None)
            find_optimal_threshold=args.optimize_threshold
        )
        
        if results:
            logging.info("Peak model evaluation complete")
            
            # Generate summary report
            generate_summary_report(results, 'peak')
            
            print("\n" + "=" * 80)
            print("PEAK MODEL EVALUATION COMPLETE")
            print("=" * 80)
            print(f"Results saved to: {PEAK_EVAL_DIR}")
            
            # Print key metrics
            print("\nKey performance metrics:")
            for metric, value in results.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    print(f"  {metric}: {value:.4f}")
    
    elif args.model == 'valley':
        # Load valley model
        valley_artifacts = load_valley_model(production_mode=args.production_mode)
        
        # Evaluate valley model
        results = evaluate_valley_model(
            data, 
            valley_artifacts, 
            num_weeks=args.weeks,
            use_test_data=args.test_data,
            prob_threshold=args.valley_threshold,
            find_optimal_threshold=args.optimize_threshold
        )
        
        if results:
            logging.info("Valley model evaluation complete")
            
            # Generate summary report
            generate_summary_report(results, 'valley')
            
            print("\n" + "=" * 80)
            print("VALLEY MODEL EVALUATION COMPLETE")
            print("=" * 80)
            print(f"Results saved to: {VALLEY_EVAL_DIR}")
            
            # Print key metrics
            print("\nKey performance metrics:")
            for metric, value in results.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    print(f"  {metric}: {value:.4f}")
    
    elif args.model == 'merged':
        # Load all models
        trend_artifacts = load_trend_model(production_mode=args.production_mode)
        peak_artifacts = load_peak_model(production_mode=args.production_mode)
        valley_artifacts = load_valley_model(production_mode=args.production_mode)
        
        # Evaluate with simplified merging
        result_df = evaluate_merged_models(
            data,
            trend_artifacts,
            peak_artifacts=peak_artifacts,
            valley_artifacts=valley_artifacts,
            use_test_data=args.test_data,
            peak_threshold=args.peak_threshold if args.peak_threshold is not None else 0.8,
            valley_threshold=args.valley_threshold if args.valley_threshold is not None else 0.5
        )
        
        if result_df is not None:
            logging.info("Simple merged model evaluation complete")
            
            print("\n" + "=" * 80)
            print("SIMPLE MERGED MODEL EVALUATION COMPLETE")
            print("=" * 80)
            print(f"Results saved to: {EVALUATION_DIR / 'merged'}")
            
            # Calculate and print metrics
            if 'predicted_price' in result_df.columns and TARGET_VARIABLE in result_df.columns:
                from sklearn.metrics import mean_absolute_error, mean_squared_error
                
                actual = result_df[TARGET_VARIABLE].values
                predicted = result_df['predicted_price'].values
                
                mae = mean_absolute_error(actual, predicted)
                rmse = np.sqrt(mean_squared_error(actual, predicted))
                
                print(f"\nMetrics:")
                print(f"  MAE: {mae:.4f}")
                print(f"  RMSE: {rmse:.4f}")
    
    else:
        logging.error(f"Unknown model type: {args.model}")
        return

if __name__ == "__main__":
    main() 