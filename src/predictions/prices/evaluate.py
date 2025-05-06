#!/usr/bin/env python
"""
Simplified evaluation script for electricity price prediction models.
Currently focuses on valley detection with proper validation split.

Usage:
    python evaluate.py --model valley  # Only valley implemented for now
    python evaluate.py --model peak    # Will prompt to implement
    python evaluate.py --model trend   # Will prompt to implement
"""

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
from pathlib import Path
import argparse
from datetime import datetime, timedelta
from tcn import TCN
import os
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
    add_lag_features, add_rolling_features, detect_price_valleys_derivative,
    create_sequences, detect_peaks_robust  # Removed plot_peak_labels from utils import
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

# Import plot_peak_labels from train
from train import plot_peak_labels

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
        result = exponential_smooth(result, alpha=0.5)
        result = median_filter(result, window=3)
        
    elif smoothing_level == 'medium':
        # Medium smoothing (default)
        result = exponential_smooth(result, alpha=0.35)
        result = median_filter(result, window=5)
        result = savitzky_golay_filter(result, window=11, polyorder=2)
        
    elif smoothing_level == 'heavy':
        # Heavy smoothing
        result = exponential_smooth(result, alpha=0.2)
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
    """Configure logging for the specific model type."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if model_type == "valley":
        log_file = VALLEY_EVAL_DIR / f"evaluation_{timestamp}.log"
    elif model_type == "trend":
        log_file = TREND_EVAL_DIR / f"evaluation_{timestamp}.log"
    else:
        log_file = EVALUATION_DIR / f"evaluation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )

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

def load_valley_model():
    """Load the valley detection model and associated artifacts."""
    logging.info("Loading valley detection model...")
    
    try:
        # Determine the model path - first check if model exists
        model_dir = Path(__file__).resolve().parent / "models" / "valley_model"
        model_paths = [
            model_dir / "best_valley_model.keras",        # Best model by validation loss
            model_dir / "best_f1_valley_model.keras",     # Model optimized for F1 score
            model_dir / "final_valley_model.keras",        # Final model after training
            model_dir / "best_recall_valley_model.keras" # Model optimized for recall
        ]
        
        # Find the first valid model path
        model_path = None
        for path in model_paths:
            if path.exists():
                model_path = path
                logging.info(f"Using model from {path.name}")
                break
        
        if model_path is None:
            raise ValueError("No valley model found in models directory")
        
        # Load optimal threshold from PR curve analysis if available
        threshold_path = model_dir / "optimal_threshold.json"
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
    
    # Load feature list
    feature_list_path = VALLEY_MODEL_DIR / "feature_list_valley_model.json"
    try:
        with open(feature_list_path, 'r') as f:
            feature_names = json.load(f)
        logging.info(f"Loaded feature list with {len(feature_names)} features")
    except Exception as e:
        logging.error(f"Error loading feature list: {e}")
        raise
    
    # Load feature scaler if it exists
    feature_scaler = None
    feature_scaler_path = VALLEY_MODEL_DIR / "feature_scaler_valley_model.save"
    if feature_scaler_path.exists():
        try:
            feature_scaler = joblib.load(feature_scaler_path)
            logging.info(f"Loaded feature scaler from {feature_scaler_path}")
        except Exception as e:
            logging.warning(f"Error loading feature scaler: {e}")
    
    return {
        'model': model,
        'feature_names': feature_names,
        'feature_scaler': feature_scaler
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

def add_valley_features(df):
    """Add features specifically designed for valley detection."""
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
        ax.plot(week_data['timestamp'], week_data['price'], 'b-', linewidth=2, label='Price')
        
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

def load_trend_model():
    """Load the trained trend model (XGBoost)."""
    logging.info("Searching for trend models...")
    
    # Define file paths
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
        logging.error(f"No valid trend model found in {TREND_MODEL_DIR}")
        raise FileNotFoundError(f"No trend model files found in {TREND_MODEL_DIR}")
    
    # Load model parameters if available
    model_params = {}
    if params_path.exists():
        try:
            with open(params_path, 'r') as f:
                model_params = json.load(f)
            model_type = model_params.get('model_type', 'XGBoost')
            logging.info(f"Loaded parameters for {model_type} model")
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
        
        # Load feature names/order if available
        if feature_names_path.exists():
            try:
                with open(feature_names_path, 'r') as f:
                    feature_order = json.load(f)
                logging.info(f"Loaded feature order list with {len(feature_order)} features")
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
            
            # Try alternative prediction approach as a fallback
            try:
                # Last resort solution: Hack around XGBoost validation check
                logging.warning("Trying fallback prediction approach...")
                # Save the original model features temporarily
                original_features = None
                if hasattr(model, 'feature_names'):
                    original_features = model.feature_names
                    model.feature_names = None  # Temporarily remove feature names
                
                # Make predictions without feature validation
                test_pred_raw = model.predict(dtest)
                logging.info("Successfully made predictions with fallback method")
                
                # Restore original feature names
                if original_features is not None:
                    model.feature_names = original_features
                    
            except Exception as fallback_error:
                logging.error(f"Fallback method also failed: {fallback_error}")
                raise
        
        # Get the test actuals
        test_actual = test_df[TARGET_VARIABLE].values
        
        # Apply smoothing using same parameters from training
        alpha = smoothing_params.get('exponential_alpha', 0.35)
        median_window = smoothing_params.get('median_window', 5)
        savgol_window = smoothing_params.get('savgol_window', 11)
        savgol_polyorder = smoothing_params.get('savgol_polyorder', 2)
        smoothing_level = smoothing_params.get('smoothing_level', 'light')
        
        logging.info(f"Applying {smoothing_level} smoothing level from trained model")
        
        # Use adaptive trend smoothing if smoothing level is specified
        if smoothing_level in ['light', 'medium', 'heavy', 'daily', 'weekly']:
            test_pred = adaptive_trend_smoothing(test_pred_raw, X_test.index, smoothing_level=smoothing_level)
        else:
            # Fall back to previous smoothing pipeline
            logging.info("Applying smoothing pipeline to predictions")
            test_pred = exponential_smooth(test_pred_raw, alpha=alpha)
            test_pred = median_filter(test_pred, window=median_window)
            test_pred = savitzky_golay_filter(test_pred, window=savgol_window, polyorder=savgol_polyorder)
        
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
    plt.plot(forecast_index, test_actual, 'b-', label='Actual', alpha=0.7)
    plt.plot(forecast_index, test_pred, 'r-', label='Predicted', alpha=0.7)
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
        plt.plot(forecast_index[week_slice], test_actual[week_slice], 'b-', label='Actual', linewidth=2)
        plt.plot(forecast_index[week_slice], test_pred[week_slice], 'r-', label='Predicted', linewidth=2)
        
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
        ax1.plot(month_data['date'], month_data['actual'], 'b-', label='Actual', linewidth=2)
        ax1.plot(month_data['date'], month_data['predicted'], 'r-', label='Predicted', linewidth=2)
        
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
        ax2.plot(month_data['date'], month_data['error'], 'g-', label='Error', linewidth=1.5)
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
        ax4.plot(daily_dates, daily_metrics['actual_mean'], 'b--', label='Avg Price', linewidth=1.5)
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
    plt.plot(x, stats.norm.pdf(x, error_mean, error_std) * len(test_actual) * (6*error_std/50), 
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
    plt.plot(np.sort(test_actual), p(np.sort(test_actual)), "r--", 
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
    parser.add_argument('--model', type=str, choices=['trend', 'peak', 'valley', 'merged'], default='trend',
                      help='Model type to evaluate (trend, peak, valley, or merged)')
    parser.add_argument('--weeks', type=int, default=4,
                      help='Number of weeks to visualize in evaluation (or number of samples for peak/valley detection)')
    parser.add_argument('--test-data', action='store_true', dest='test_data',
                      help='Use test data instead of validation data for evaluation')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Probability threshold for peak/valley detection (default: 0.5)')
    parser.add_argument('--peak-threshold', type=float, default=None,
                      help='Separate probability threshold for peak detection in merged model (default: None, uses --threshold)')
    parser.add_argument('--optimize-threshold', action='store_true', dest='optimize_threshold',
                      help='Find the optimal threshold for peak/valley detection')
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

# Add a new function to evaluate combined trend and valley models
def evaluate_merged_models(data, trend_artifacts, peak_artifacts=None, valley_artifacts=None, num_weeks=4, use_test_data=False, peak_threshold=0.5, valley_threshold=0.4):
    """
    Evaluate combined trend, peak, and valley models.
    
    Args:
        data: DataFrame with electricity price data
        trend_artifacts: Dictionary with trend model artifacts
        peak_artifacts: Dictionary with peak model artifacts (optional)
        valley_artifacts: Dictionary with valley model artifacts (optional)
        num_weeks: Number of weeks to visualize
        use_test_data: Whether to use test data instead of validation data
        peak_threshold: Probability threshold for peak detection
        valley_threshold: Probability threshold for valley detection
        
    Returns:
        Dictionary with evaluation results
    """
    logging.info("Starting combined model evaluation...")
    
    # The data is a single DataFrame, not a dictionary
    df = data.copy()
    
    # Determine which subset to use based on date range
    subset_name = "test" if use_test_data else "validation"
    
    # If using test data, take the last 20% of data
    # Otherwise use the next-to-last 20% (validation data)
    total_rows = len(df)
    test_size = int(total_rows * 0.2)
    
    if use_test_data:
        # Use the last 20% of data for test
        df = df.iloc[-test_size:].copy()
    else:
        # Use the next-to-last 20% for validation
        start_idx = total_rows - 2 * test_size
        end_idx = total_rows - test_size
        df = df.iloc[start_idx:end_idx].copy()
    
    logging.info(f"Using {subset_name} data with {len(df)} rows from {df.index[0]} to {df.index[-1]}")
    
    # Add features for peak and valley detection
    if peak_artifacts:
        df = add_peak_features(df)
        logging.info("Added peak detection features")
    
    if valley_artifacts:
        df = add_valley_features(df)
        logging.info("Added valley detection features")
    
    # Make trend predictions
    trend_model = trend_artifacts.get('model')
    trend_feature_names = trend_artifacts.get('feature_order', [])  # Try to use feature_order first
    if not trend_feature_names:
        trend_feature_names = trend_artifacts.get('feature_names', [])  # Fall back to feature_names
    
    # Check if we have the necessary features for trend prediction
    missing_trend_features = [f for f in trend_feature_names if f not in df.columns]
    if missing_trend_features:
        logging.warning(f"Missing {len(missing_trend_features)} trend features: {missing_trend_features[:5]}...")
        # Add dummy values for missing features
        for feature in missing_trend_features:
            df[feature] = 0
    
    # For XGBoost we need to ensure a specific ordering of features
    df_trend = df[trend_feature_names].copy()
    
    # Use XGBoost model for trend predictions
    trend_predictions = trend_model.predict(df_trend)
    
    # Create result dataframe with trend predictions
    result_df = df.copy()
    result_df['trend_prediction'] = trend_predictions
    
    # Apply smoothing to trend predictions if requested
    smoothing_params = trend_artifacts.get('smoothing_params', {})
    if smoothing_params:
        logging.info(f"Applying {smoothing_params.get('smoothing_level', 'medium')} smoothing to trend predictions")
        from utils import adaptive_trend_smoothing
        smooth_trend = adaptive_trend_smoothing(
            trend_predictions, 
            result_df.index,
            smoothing_level=smoothing_params.get('smoothing_level', 'medium')
        )
        result_df['trend_prediction_smooth'] = smooth_trend
    else:
        # Apply a default light smoothing
        logging.info("Applying default light smoothing to trend predictions")
        from utils import adaptive_trend_smoothing
        smooth_trend = adaptive_trend_smoothing(
            trend_predictions, 
            result_df.index,
            smoothing_level='light'
        )
        result_df['trend_prediction_smooth'] = smooth_trend
    
    # Make peak predictions if peak model was provided
    peak_probabilities = None
    peak_predictions = None
    if peak_artifacts:
        peak_model = peak_artifacts.get('model')
        peak_feature_names = peak_artifacts.get('feature_names', [])
        peak_feature_scaler = peak_artifacts.get('feature_scaler')
        
        # Check if we have all necessary features
        missing_peak_features = [f for f in peak_feature_names if f not in df.columns]
        if missing_peak_features:
            logging.warning(f"Missing {len(missing_peak_features)} peak model features. Adding placeholders.")
            # Add dummy features
            for feature in missing_peak_features:
                df[feature] = 0
        
        # Extract and scale features
        X_peak = df[peak_feature_names].values
        X_peak_scaled = peak_feature_scaler.transform(X_peak)
        
        # Create sequences for peak model - UPDATED to use 168 window
        from utils import create_sequences
        
        # Define the window size for peak model (1 week = 168 hours)
        PEAK_WINDOW_SIZE = 168
        
        # Check if we have enough data
        if len(X_peak_scaled) <= PEAK_WINDOW_SIZE:
            logging.warning(f"Not enough data for peak sequences. Need > {PEAK_WINDOW_SIZE} samples.")
            # Skip peak prediction
        else:
            logging.info(f"Creating peak sequences with window size {PEAK_WINDOW_SIZE}")
            X_peak_seq, _ = create_sequences(
                X_peak_scaled,
                PEAK_WINDOW_SIZE,  # Updated from 24 to 168
                1,
                list(range(X_peak_scaled.shape[1])),
                None
            )
            
            # Make predictions
            try:
                logging.info(f"Making peak predictions with input shape {X_peak_seq.shape}")
                peak_probabilities = peak_model.predict(X_peak_seq)
                
                # Flatten if needed
                if len(peak_probabilities.shape) > 1:
                    peak_probabilities = peak_probabilities.flatten()
                
                # Apply threshold
                peak_predictions = (peak_probabilities >= peak_threshold).astype(int)
                
                # Align with timestamps (accounting for sequence window)
                result_timestamps = result_df.index[PEAK_WINDOW_SIZE:PEAK_WINDOW_SIZE+len(peak_probabilities)]
                
                # Add to result dataframe - create a temporary df with the right index
                peak_df = pd.DataFrame(index=result_timestamps)
                peak_df['peak_probability'] = peak_probabilities
                peak_df['is_predicted_peak'] = peak_predictions
                
                # Join with result_df (some rows will remain NaN)
                result_df = result_df.join(peak_df, how='left')
                
                logging.info(f"Added peak predictions for {len(peak_probabilities)} samples")
            except Exception as e:
                logging.error(f"Error making peak predictions: {e}")
                logging.error("Peak model predictions failed, continuing without peak detection")
    
    # Make valley predictions if valley model was provided
    valley_probabilities = None
    valley_predictions = None
    if valley_artifacts:
        valley_model = valley_artifacts.get('model')
        valley_feature_names = valley_artifacts.get('feature_names', [])
        valley_feature_scaler = valley_artifacts.get('feature_scaler')
        
        # Check if we have all necessary features
        missing_valley_features = [f for f in valley_feature_names if f not in df.columns]
        if missing_valley_features:
            logging.warning(f"Missing {len(missing_valley_features)} valley model features. Adding placeholders.")
            # Add dummy features
            for feature in missing_valley_features:
                df[feature] = 0
        
        # Extract and scale features
        X_valley = df[valley_feature_names].values
        X_valley_scaled = valley_feature_scaler.transform(X_valley)
        
        # Create sequences for valley model
        from utils import create_sequences
        VALLEY_WINDOW_SIZE = 24  # Valley model uses 24-hour window
        
        # Check if we have enough data
        if len(X_valley_scaled) <= VALLEY_WINDOW_SIZE:
            logging.warning(f"Not enough data for valley sequences. Need > {VALLEY_WINDOW_SIZE} samples.")
            # Skip valley prediction
        else:
            X_valley_seq, _ = create_sequences(
                X_valley_scaled,
                VALLEY_WINDOW_SIZE,
                1,
                list(range(X_valley_scaled.shape[1])),
                None
            )
            
            # Make predictions
            try:
                valley_probabilities = valley_model.predict(X_valley_seq)
                
                # Flatten if needed
                if len(valley_probabilities.shape) > 1:
                    valley_probabilities = valley_probabilities.flatten()
                
                # Apply threshold
                valley_predictions = (valley_probabilities >= valley_threshold).astype(int)
                
                # Align with timestamps (accounting for sequence window)
                result_timestamps = result_df.index[VALLEY_WINDOW_SIZE:VALLEY_WINDOW_SIZE+len(valley_probabilities)]
                
                # Add to result dataframe - create a temporary df with the right index
                valley_df = pd.DataFrame(index=result_timestamps)
                valley_df['valley_probability'] = valley_probabilities
                valley_df['is_predicted_valley'] = valley_predictions
                
                # Join with result_df (some rows will remain NaN)
                result_df = result_df.join(valley_df, how='left')
                
                logging.info(f"Added valley predictions for {len(valley_probabilities)} samples")
            except Exception as e:
                logging.error(f"Error making valley predictions: {e}")
                logging.error("Valley model predictions failed, continuing without valley detection")
    
    # Extract actual peaks and valleys for evaluation
    actual_peaks = None
    actual_valleys = None
    
    if 'is_price_peak' in result_df.columns:
        actual_peaks = result_df['is_price_peak'].values
    
    if 'is_price_valley' in result_df.columns:
        actual_valleys = result_df['is_price_valley'].values
    
    # Generate plots
    output_dir = EVALUATION_DIR / "merged"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate weekly plots
    plot_dir = generate_merged_plots(
        result_df.index, 
        result_df[TARGET_VARIABLE].values, 
        result_df['trend_prediction_smooth'].values,
        peak_predictions, 
        result_df.get('peak_probability', None) if 'peak_probability' in result_df.columns else None,
        actual_peaks,
        valley_predictions, 
        result_df.get('valley_probability', None) if 'valley_probability' in result_df.columns else None,
        actual_valleys,
        output_dir=output_dir,
        eval_name=subset_name
    )
    
    # Calculate evaluation metrics
    results = {
        'model_type': 'merged',
        'eval_set': subset_name,
        'plot_dir': str(plot_dir)
    }
    
    # Calculate trend metrics
    if trend_predictions is not None and TARGET_VARIABLE in result_df.columns:
        actual_prices = result_df[TARGET_VARIABLE].values
        smooth_predictions = result_df['trend_prediction_smooth'].values
        
        # Calculate error metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(actual_prices, smooth_predictions)
        rmse = np.sqrt(mean_squared_error(actual_prices, smooth_predictions))
        
        # Calculate direction accuracy (up/down)
        actual_direction = np.diff(actual_prices) > 0
        pred_direction = np.diff(smooth_predictions) > 0
        direction_accuracy = np.mean(actual_direction == pred_direction)
        
        # Add metrics to results
        results.update({
            'trend_mae': float(mae),
            'trend_rmse': float(rmse),
            'trend_direction_accuracy': float(direction_accuracy)
        })
    
    # Calculate peak metrics
    if peak_predictions is not None and 'is_price_peak' in result_df.columns:
        # Align actual peaks with predictions (same length)
        peak_window = PEAK_WINDOW_SIZE  # Using the corrected peak window size (168)
        aligned_actuals = result_df['is_price_peak'].values[peak_window:peak_window+len(peak_predictions)]
        
        # Calculate classification metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        p_accuracy = accuracy_score(aligned_actuals, peak_predictions)
        p_precision = precision_score(aligned_actuals, peak_predictions, zero_division=0)
        p_recall = recall_score(aligned_actuals, peak_predictions, zero_division=0)
        p_f1 = f1_score(aligned_actuals, peak_predictions, zero_division=0)
        
        # Add metrics to results
        results.update({
            'peak_accuracy': float(p_accuracy),
            'peak_precision': float(p_precision),
            'peak_recall': float(p_recall),
            'peak_f1': float(p_f1),
            'peak_threshold': float(peak_threshold)
        })
    
    # Calculate valley metrics
    if valley_predictions is not None and 'is_price_valley' in result_df.columns:
        # Align actual valleys with predictions (same length)
        valley_window = VALLEY_WINDOW_SIZE
        aligned_actuals = result_df['is_price_valley'].values[valley_window:valley_window+len(valley_predictions)]
        
        # Calculate classification metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        v_accuracy = accuracy_score(aligned_actuals, valley_predictions)
        v_precision = precision_score(aligned_actuals, valley_predictions, zero_division=0)
        v_recall = recall_score(aligned_actuals, valley_predictions, zero_division=0)
        v_f1 = f1_score(aligned_actuals, valley_predictions, zero_division=0)
        
        # Add metrics to results
        results.update({
            'valley_accuracy': float(v_accuracy),
            'valley_precision': float(v_precision),
            'valley_recall': float(v_recall),
            'valley_f1': float(v_f1),
            'valley_threshold': float(valley_threshold)
        })
    
    # Generate summary report
    
    return results

def generate_merged_plots(timestamps, actual_prices, trend_predictions, 
                         peak_predictions=None, peak_probabilities=None, actual_peaks=None,
                         valley_predictions=None, valley_probabilities=None, actual_valleys=None,
                         output_dir=None, eval_name="validation"):
    """
    Generate plots with trend, peak, and valley predictions together.
    
    Args:
        timestamps: DatetimeIndex of prediction times
        actual_prices: Actual price values
        trend_predictions: Trend model predictions
        peak_predictions: Binary peak predictions (optional)
        peak_probabilities: Peak prediction probabilities (optional)
        actual_peaks: Actual peak labels (optional)
        valley_predictions: Binary valley predictions (optional)
        valley_probabilities: Valley prediction probabilities (optional)
        actual_valleys: Actual valley labels (optional)
        output_dir: Directory to save plots
        eval_name: Name of the evaluation set (e.g., 'validation' or 'test')
        
    Returns:
        Path to the output directory with generated plots
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    from datetime import timedelta
    import os
    
    # Create output directory
    if output_dir is None:
        output_dir = EVALUATION_DIR / "merged" / eval_name
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine the number of weeks in the data
    total_hours = len(timestamps)
    hours_per_week = 24 * 7
    num_weeks = max(1, total_hours // hours_per_week)
    
    logging.info(f"Generating {num_weeks} weekly plots for merged model visualization")
    
    # Create weekly plots
    for week in range(num_weeks):
        start_idx = week * hours_per_week
        end_idx = min(start_idx + hours_per_week, total_hours)
        
        if end_idx - start_idx < 24:  # Skip if less than a day of data
            continue
        
        # Extract data for this week
        week_timestamps = timestamps[start_idx:end_idx]
        week_actual_prices = actual_prices[start_idx:end_idx]
        week_trend_predictions = trend_predictions[start_idx:end_idx]
        
        # Extract peak data if available
        week_peak_predictions = None
        week_peak_probabilities = None
        week_actual_peaks = None
        
        if peak_predictions is not None:
            week_peak_predictions = peak_predictions[start_idx:end_idx]
            week_peak_probabilities = peak_probabilities[start_idx:end_idx]
            week_actual_peaks = actual_peaks[start_idx:end_idx] if actual_peaks is not None else None
        
        # Extract valley data if available
        week_valley_predictions = None
        week_valley_probabilities = None
        week_actual_valleys = None
        
        if valley_predictions is not None:
            week_valley_predictions = valley_predictions[start_idx:end_idx]
            week_valley_probabilities = valley_probabilities[start_idx:end_idx]
            week_actual_valleys = actual_valleys[start_idx:end_idx] if actual_valleys is not None else None
        
        # Start and end dates for this week
        start_date = week_timestamps[0]
        end_date = week_timestamps[-1]
        
        # Determine subplot configuration based on which models are included
        if peak_predictions is not None and valley_predictions is not None:
            # All three models: trend, peak, valley
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), 
                                              gridspec_kw={'height_ratios': [3, 1, 1]})
        elif peak_predictions is not None or valley_predictions is not None:
            # Trend + either peak or valley
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), 
                                         gridspec_kw={'height_ratios': [3, 1]})
        else:
            # Just trend model
            fig, ax1 = plt.subplots(1, 1, figsize=(15, 8))
        
        # Plot 1: Price and trend predictions
        ax1.plot(week_timestamps, week_actual_prices, 'b-', linewidth=2, label='Actual Price')
        ax1.plot(week_timestamps, week_trend_predictions, 'g-', linewidth=2, label='Trend Prediction')
        
        # Add peak markers if available
        if week_peak_predictions is not None:
            peak_indices = np.where(week_peak_predictions == 1)[0]
            if len(peak_indices) > 0:
                peak_times = [week_timestamps[i] for i in peak_indices]
                peak_prices = [week_actual_prices[i] for i in peak_indices]
                ax1.scatter(peak_times, peak_prices, color='red', s=80, marker='^', 
                          label='Detected Peaks')
            
            # Add actual peak markers if available
            if week_actual_peaks is not None and np.sum(week_actual_peaks) > 0:
                true_peak_indices = np.where(week_actual_peaks == 1)[0]
                true_peak_times = [week_timestamps[i] for i in true_peak_indices]
                true_peak_prices = [week_actual_prices[i] for i in true_peak_indices]
                ax1.scatter(true_peak_times, true_peak_prices, color='orange', s=60, marker='o', 
                          facecolors='none', label='Actual Peaks')
        
        # Add valley markers if available
        if week_valley_predictions is not None:
            valley_indices = np.where(week_valley_predictions == 1)[0]
            if len(valley_indices) > 0:
                valley_times = [week_timestamps[i] for i in valley_indices]
                valley_prices = [week_actual_prices[i] for i in valley_indices]
                ax1.scatter(valley_times, valley_prices, color='blue', s=80, marker='v', 
                          label='Detected Valleys')
            
            # Add actual valley markers if available
            if week_actual_valleys is not None and np.sum(week_actual_valleys) > 0:
                true_valley_indices = np.where(week_actual_valleys == 1)[0]
                true_valley_times = [week_timestamps[i] for i in true_valley_indices]
                true_valley_prices = [week_actual_prices[i] for i in true_valley_indices]
                ax1.scatter(true_valley_times, true_valley_prices, color='cyan', s=60, marker='o', 
                          facecolors='none', label='Actual Valleys')
        
        # Shade weekends for context
        for i in range(7):  # 7 days in a week
            day = start_date + timedelta(days=i)
            if day.weekday() >= 5:  # Saturday or Sunday
                ax1.axvspan(day, day + timedelta(days=1), color='lightgray', alpha=0.3)
        
        # Add title and labels for the price plot
        ax1.set_ylabel('Price (öre/kWh)', fontsize=12)
        ax1.set_title(f'Combined Model Evaluation: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}',
                     fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=10)
        
        # Plot 2: Peak probabilities (if available)
        if peak_predictions is not None:
            if valley_predictions is not None:  # Three models total
                ax_peak = ax2
            else:  # Only trend + peak
                ax_peak = ax2
                
            ax_peak.plot(week_timestamps, week_peak_probabilities, 'r-', linewidth=2, label='Peak Probability')
            ax_peak.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Threshold (0.5)')
            
            if week_actual_peaks is not None:
                ax_peak.step(week_timestamps, week_actual_peaks * 0.2, 'k-', where='mid', linewidth=1, label='Actual Peaks')
            
            # Shade weekends
            for i in range(7):
                day = start_date + timedelta(days=i)
                if day.weekday() >= 5:  # Saturday or Sunday
                    ax_peak.axvspan(day, day + timedelta(days=1), color='lightgray', alpha=0.3)
            
            ax_peak.set_ylabel('Peak Probability', fontsize=12)
            ax_peak.set_ylim(-0.05, 1.05)
            ax_peak.grid(True, alpha=0.3)
            ax_peak.legend(loc='upper right', fontsize=10)
        
        # Plot 3: Valley probabilities (if available)
        if valley_predictions is not None:
            if peak_predictions is not None:  # Three models total
                ax_valley = ax3
            else:  # Only trend + valley
                ax_valley = ax2
                
            ax_valley.plot(week_timestamps, week_valley_probabilities, 'b-', linewidth=2, label='Valley Probability')
            ax_valley.axhline(y=0.5, color='b', linestyle='--', alpha=0.7, label='Threshold (0.5)')
            
            if week_actual_valleys is not None:
                ax_valley.step(week_timestamps, week_actual_valleys * 0.2, 'k-', where='mid', linewidth=1, label='Actual Valleys')
            
            # Shade weekends
            for i in range(7):
                day = start_date + timedelta(days=i)
                if day.weekday() >= 5:  # Saturday or Sunday
                    ax_valley.axvspan(day, day + timedelta(days=1), color='lightgray', alpha=0.3)
            
            ax_valley.set_ylabel('Valley Probability', fontsize=12)
            ax_valley.set_ylim(-0.05, 1.05)
            ax_valley.grid(True, alpha=0.3)
            ax_valley.legend(loc='upper right', fontsize=10)
        
        # Format x-axis dates on all subplots
        for ax in fig.get_axes():
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d %b'))
            ax.xaxis.set_major_locator(mdates.DayLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save the plot with a descriptive filename
        filename = f"merged_models_{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}.png"
        file_path = os.path.join(str(output_dir), filename)
        plt.savefig(file_path, dpi=300)
        plt.close()
    
    # Also create a summary plot for the entire period
    try:
        create_merged_summary_plot(
            timestamps, actual_prices, trend_predictions,
            peak_predictions, peak_probabilities, actual_peaks,
            valley_predictions, valley_probabilities, actual_valleys,
            output_dir, eval_name
        )
    except Exception as e:
        logging.error(f"Error creating merged summary plot: {e}")
    
    logging.info(f"Generated merged model plots in {output_dir}")
    return output_dir

def load_peak_model():
    """Load the peak detection model and associated artifacts."""
    logging.info("Loading peak detection model...")
    
    try:
        # Determine the model path - first check if model exists
        model_dir = Path(__file__).resolve().parent / "models" / "peak_model"
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
        
        if model_path is None:
            raise ValueError("No peak model found in models directory")
        
        # Load optimal threshold from PR curve analysis if available
        threshold_path = model_dir / "optimal_threshold.json"
        prob_threshold = 0.5  # Default threshold
        if threshold_path.exists():
            try:
                with open(threshold_path, 'r') as f:
                    threshold_data = json.load(f)
                    prob_threshold = threshold_data.get('threshold', 0.5)
                logging.info(f"Loaded optimal probability threshold: {prob_threshold:.4f}")
            except Exception as e:
                logging.warning(f"Error loading optimal threshold: {e}")
        
        # Define custom objects for model loading
        from tcn import TCN
        
        # Define additional custom loss functions that might have been used during training
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
        
        def get_recall_oriented_loss(false_neg_weight=5.0, false_pos_weight=1.0):
            """Create a custom loss function that prioritizes recall over precision."""
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
        
        # Create a dictionary of custom objects
        custom_objects = {
            'TCN': TCN,
            'GlobalSumPooling1D': GlobalSumPooling1D,
            'binary_focal_loss_fixed': binary_focal_loss(),
            'recall_loss_fn5.0_fp1.0': get_recall_oriented_loss(5.0, 1.0),
            'recall_loss_fn8.0_fp1.0': get_recall_oriented_loss(8.0, 1.0)
        }
        
        # Try different compilation options if loading fails
        try:
            # First try loading without compiling
            model = load_model(model_path, custom_objects=custom_objects, compile=False)
            logging.info(f"Loaded peak model from {model_path} without compilation")
        except Exception as e:
            logging.warning(f"Error loading model without compilation: {e}")
            # Try with compilation
            try:
                model = load_model(model_path, custom_objects=custom_objects, compile=True)
                logging.info(f"Loaded peak model from {model_path} with compilation")
            except Exception as e:
                logging.error(f"Failed to load model with compilation: {e}")
                raise
                
    except Exception as e:
        logging.error(f"Error loading peak model: {e}")
        raise
    
    # Load feature list - FIXED to use PEAK model directory
    feature_list_path = model_dir / "feature_list_peak_model.json"
    
    # If feature list not found in peak directory, look for alternative files
    if not feature_list_path.exists():
        alternative_paths = [
            model_dir / "feature_names.json",
            model_dir / "features.json",
            PEAK_MODEL_DIR / "feature_list_peak_model.json",  # Try global constant path
            VALLEY_MODEL_DIR / "feature_list_valley_model.json"  # Fallback to valley model's list as last resort
        ]
        
        for alt_path in alternative_paths:
            if alt_path.exists():
                feature_list_path = alt_path
                logging.warning(f"Peak feature list not found at expected location. Using alternative: {alt_path}")
                break
    
    try:
        with open(feature_list_path, 'r') as f:
            feature_names = json.load(f)
        logging.info(f"Loaded feature list with {len(feature_names)} features")
    except Exception as e:
        logging.error(f"Error loading feature list: {e}")
        # Provide a fallback list of common features rather than failing
        logging.warning("Using fallback feature list")
        feature_names = [
            'price_lag_24h', 'price_diff_1h', 'price_diff_3h', 'price_diff_6h', 
            'price_momentum', 'price_detrended', 'price_vs_24h_avg', 'price_vs_7d_avg',
            'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 
            'is_business_hour', 'is_morning_peak', 'is_evening_peak',
            'price_roc_1h', 'price_roc_3h'
        ]
    
    # Load feature scaler if it exists - FIXED to use PEAK model directory first
    feature_scaler = None
    feature_scaler_path = model_dir / "feature_scaler_peak_model.save"
    
    # If scaler not found in peak directory, look for alternatives
    if not feature_scaler_path.exists():
        alternative_paths = [
            model_dir / "scaler.save",
            model_dir / "feature_scaler.save",
            PEAK_MODEL_DIR / "feature_scaler_peak_model.save",  # Try global constant path
            VALLEY_MODEL_DIR / "feature_scaler_valley_model.save"  # Fallback to valley model's scaler as last resort
        ]
        
        for alt_path in alternative_paths:
            if alt_path.exists():
                feature_scaler_path = alt_path
                logging.warning(f"Peak feature scaler not found at expected location. Using alternative: {alt_path}")
                break
    
    if feature_scaler_path.exists():
        try:
            feature_scaler = joblib.load(feature_scaler_path)
            logging.info(f"Loaded feature scaler from {feature_scaler_path}")
        except Exception as e:
            logging.warning(f"Error loading feature scaler: {e}")
            feature_scaler = StandardScaler()  # Create a new scaler as fallback
    else:
        logging.warning("No feature scaler found, creating a new StandardScaler")
        feature_scaler = StandardScaler()
    
    return {
        'model': model,
        'feature_names': feature_names,
        'feature_scaler': feature_scaler
    }

def evaluate_peak_model(df, model_artifacts, num_weeks=3, use_test_data=False, prob_threshold=None, find_optimal_threshold=False):
    """
    Evaluate the peak detection model on the validation or test set.
    
    Args:
        df: DataFrame with full data
        model_artifacts: Dictionary with model and preprocessing tools
        num_weeks: Number of weeks to plot
        use_test_data: If True, evaluate on test data instead of validation data
        prob_threshold: Probability threshold for peak detection (if None, use default 0.5)
        find_optimal_threshold: If True, find optimal threshold using F1 score
        
    Returns:
        Dictionary with evaluation results
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
        y_prob = model.predict(X_seq)
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
    
    # Apply threshold
    if prob_threshold is None:
        prob_threshold = 0.5
        logging.info(f"Using default probability threshold: {prob_threshold}")
    else:
        logging.info(f"Using specified probability threshold: {prob_threshold}")
        
    # Find optimal threshold if requested
    if find_optimal_threshold:
        optimal_threshold, f1_score = find_optimal_threshold(y_prob, y_true_seq)
        logging.info(f"Optimal threshold: {optimal_threshold:.4f} with F1 score: {f1_score:.4f}")
        prob_threshold = optimal_threshold
    
    y_pred = (y_prob >= prob_threshold).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    accuracy = accuracy_score(y_true_seq, y_pred)
    precision = precision_score(y_true_seq, y_pred, zero_division=0)
    recall = recall_score(y_true_seq, y_pred, zero_division=0)
    f1 = f1_score(y_true_seq, y_pred, zero_division=0)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_seq, y_pred, labels=[0, 1]).ravel()
    
    # Calculate precision at different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    precision_at_thresholds = {}
    recall_at_thresholds = {}
    f1_at_thresholds = {}
    
    for threshold in thresholds:
        y_pred_at_threshold = (y_prob >= threshold).astype(int)
        precision_at_thresholds[threshold] = precision_score(y_true_seq, y_pred_at_threshold, zero_division=0)
        recall_at_thresholds[threshold] = recall_score(y_true_seq, y_pred_at_threshold, zero_division=0)
        f1_at_thresholds[threshold] = f1_score(y_true_seq, y_pred_at_threshold, zero_division=0)
    
    # Log metrics
    logging.info(f"Peak model metrics:")
    logging.info(f"  Accuracy: {accuracy:.4f}")
    logging.info(f"  Precision: {precision:.4f}")
    logging.info(f"  Recall: {recall:.4f}")
    logging.info(f"  F1 Score: {f1:.4f}")
    logging.info(f"  True Positives: {tp}, False Positives: {fp}")
    logging.info(f"  False Negatives: {fn}, True Negatives: {tn}")
    
    # Generate evaluation plots
    plots_dir = PEAK_EVAL_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate weekly plots
    weekly_plots = generate_weekly_peak_plots(
        timestamps, 
        y_pred.flatten(), 
        y_prob.flatten(), 
        y_true_seq, 
        prices,
        eval_name=eval_name
    )
    
    # Create a summary plot with the whole evaluation period
    plt.figure(figsize=(16, 10))
    
    # Plot price data
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, prices, 'b-', label='Price')
    
    # Highlight detected peaks with vertical spans
    peak_indices = np.where(y_pred.flatten() == 1)[0]
    for idx in peak_indices:
        plt.axvspan(timestamps[idx] - pd.Timedelta(hours=1), 
                  timestamps[idx] + pd.Timedelta(hours=1), 
                  color='lightsalmon', alpha=0.3)
    
    # Mark actual peaks
    actual_peak_indices = np.where(y_true_seq == 1)[0]
    actual_peak_times = [timestamps[idx] for idx in actual_peak_indices]
    actual_peak_prices = [prices[idx] for idx in actual_peak_indices]
    plt.scatter(actual_peak_times, actual_peak_prices, color='red', marker='^', s=100, label='Actual Peaks')
    
    # Mark predicted peaks
    pred_peak_indices = np.where(y_pred.flatten() == 1)[0]
    pred_peak_times = [timestamps[idx] for idx in pred_peak_indices]
    pred_peak_prices = [prices[idx] for idx in pred_peak_indices]
    plt.scatter(pred_peak_times, pred_peak_prices, color='blue', marker='o', s=80, alpha=0.7, label='Predicted Peaks')
    
    plt.title(f'Peak Detection on {eval_name.capitalize()} Set')
    plt.ylabel('Price (öre/kWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot detection probabilities
    plt.subplot(2, 1, 2)
    plt.plot(timestamps, y_prob.flatten(), 'g-', label='Peak Probability')
    plt.axhline(y=prob_threshold, color='r', linestyle='--', label=f'Threshold ({prob_threshold:.2f})')
    
    # Mark actual peaks
    for peak_time in actual_peak_times:
        plt.axvline(x=peak_time, color='red', alpha=0.2)
    
    plt.title('Peak Detection Probability')
    plt.ylabel('Probability')
    plt.xlabel('Time')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / f"peak_detection_{eval_name}_summary.png")
    plt.close()
    
    # Create precision-recall curve
    plt.figure(figsize=(10, 6))
    
    precision_values = [precision_at_thresholds[t] for t in thresholds]
    recall_values = [recall_at_thresholds[t] for t in thresholds]
    
    plt.plot(thresholds, precision_values, 'b-', label='Precision')
    plt.plot(thresholds, recall_values, 'g-', label='Recall')
    plt.plot(thresholds, [f1_at_thresholds[t] for t in thresholds], 'r-', label='F1 Score')
    plt.axvline(x=prob_threshold, color='k', linestyle='--', label=f'Selected Threshold ({prob_threshold:.2f})')
    
    plt.title('Precision, Recall and F1 Score at Different Thresholds')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plots_dir / f"peak_detection_{eval_name}_precision_recall.png")
    plt.close()
    
    # Return evaluation results
    results = {
        'model_type': 'peak',
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn), 
            'true_negatives': int(tn)
        },
        'threshold': float(prob_threshold),
        'timestamps': timestamps,
        'predictions': y_pred.flatten().tolist(),
        'probabilities': y_prob.flatten().tolist(),
        'actual_values': y_true_seq.tolist(),
        'prices': prices.tolist(),
        'metrics_by_threshold': {
            'thresholds': thresholds,
            'precision': {str(t): float(p) for t, p in precision_at_thresholds.items()},
            'recall': {str(t): float(r) for t, r in recall_at_thresholds.items()},
            'f1_score': {str(t): float(f) for t, f in f1_at_thresholds.items()}
        },
        'plots': weekly_plots,
        'summary_plot': str(plots_dir / f"peak_detection_{eval_name}_summary.png"),
        'precision_recall_plot': str(plots_dir / f"peak_detection_{eval_name}_precision_recall.png")
    }
    
    return results

def add_peak_features(df):
    """
    Add necessary features for peak detection model.
    
    Args:
        df: DataFrame with base data
        
    Returns:
        DataFrame with added peak detection features
    """
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

def generate_weekly_peak_plots(timestamps, predictions, probabilities, actuals, prices, eval_name="validation"):
    """
    Generate weekly plots for peak detection evaluation.
    
    Args:
        timestamps: DatetimeIndex of evaluation timestamps
        predictions: Binary peak predictions (0/1)
        probabilities: Peak detection probabilities
        actuals: Actual peak labels (0/1)
        prices: Price values for plotting
        eval_name: Name of evaluation set ('validation' or 'test')
        
    Returns:
        List of generated plot paths
    """
    # Create output directory
    plots_dir = PEAK_MODEL_DIR / "weekly"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to more efficient data structures if needed
    if not isinstance(timestamps, pd.DatetimeIndex):
        timestamps = pd.DatetimeIndex(timestamps)
    
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    actuals = np.array(actuals)
    prices = np.array(prices)
    
    # Calculate start dates for weekly plots
    start_date = timestamps[0]
    end_date = timestamps[-1]
    
    # Generate dates at weekly intervals
    plot_dates = []
    current_date = start_date
    while current_date <= end_date:
        plot_dates.append(current_date)
        current_date += pd.Timedelta(days=7)
    
    # If we have less than 3 weeks, use different intervals
    if len(plot_dates) < 3:
        plot_dates = []
        days_interval = max(1, (end_date - start_date).days // 3)
        current_date = start_date
        while current_date <= end_date:
            plot_dates.append(current_date)
            current_date += pd.Timedelta(days=days_interval)
    
    # Generate plots for each week
    plot_paths = []
    
    for i, week_start in enumerate(plot_dates):
        week_end = week_start + pd.Timedelta(days=7)
        
        # Get indices for this week
        week_mask = (timestamps >= week_start) & (timestamps < week_end)
        if not np.any(week_mask):
            continue
            
        week_indices = np.where(week_mask)[0]
        
        # Get data for this week
        week_timestamps = timestamps[week_indices]
        week_predictions = predictions[week_indices]
        week_probabilities = probabilities[week_indices]
        week_actuals = actuals[week_indices]
        week_prices = prices[week_indices]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Price subplot with peaks
        ax1.plot(week_timestamps, week_prices, 'b-', linewidth=2)
        
        # Highlight predicted peaks
        predicted_peak_indices = np.where(week_predictions == 1)[0]
        for idx in predicted_peak_indices:
            ax1.axvspan(week_timestamps[idx] - pd.Timedelta(hours=1),
                      week_timestamps[idx] + pd.Timedelta(hours=1),
                      color='lightsalmon', alpha=0.3)
        
        # Plot actual peaks
        actual_peak_indices = np.where(week_actuals == 1)[0]
        actual_peak_times = [week_timestamps[idx] for idx in actual_peak_indices]
        actual_peak_prices = [week_prices[idx] for idx in actual_peak_indices]
        
        if actual_peak_times:
            ax1.scatter(actual_peak_times, actual_peak_prices, color='red', marker='^', s=100, 
                       label='Actual Peaks')
        
        # Plot predicted peaks
        pred_peak_indices = np.where(week_predictions == 1)[0]
        pred_peak_times = [week_timestamps[idx] for idx in pred_peak_indices]
        pred_peak_prices = [week_prices[idx] for idx in pred_peak_indices]
        
        if pred_peak_times:
            ax1.scatter(pred_peak_times, pred_peak_prices, color='blue', marker='o', s=80, 
                       label='Predicted Peaks', alpha=0.7)
        
        # Format the plot
        week_num = i + 1
        ax1.set_title(f'Peak Detection {eval_name.capitalize()} - Week {week_num}')
        ax1.set_ylabel('Price (öre/kWh)')
        ax1.grid(True, alpha=0.3)
        
        # Add legend only if we have predictions/actuals
        if len(actual_peak_indices) > 0 or len(pred_peak_indices) > 0:
            ax1.legend()
            
        # Highlight weekends
        for j in range(7):
            day_start = week_start + pd.Timedelta(days=j)
            if day_start.weekday() >= 5:  # Saturday or Sunday
                ax1.axvspan(day_start, day_start + pd.Timedelta(days=1), 
                           color='gray', alpha=0.1)
        
        # Probability subplot
        ax2.plot(week_timestamps, week_probabilities, 'g-', linewidth=2)
        ax2.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
        
        # Highlight actual peaks in probability plot
        for peak_time in actual_peak_times:
            ax2.axvline(x=peak_time, color='red', alpha=0.2)
        
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Peak Probability')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis to show dates nicely
        fig.autofmt_xdate()
        
        # Add stats annotation
        week_stats = f"Week {week_num} Statistics:\n"
        
        # Add metrics if we have actual values
        if len(week_actuals) > 0:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # Only calculate metrics if we have both classes present
            if len(set(week_actuals)) > 1:
                week_accuracy = accuracy_score(week_actuals, week_predictions)
                week_precision = precision_score(week_actuals, week_predictions, zero_division=0)
                week_recall = recall_score(week_actuals, week_predictions, zero_division=0)
                week_f1 = f1_score(week_actuals, week_predictions, zero_division=0)
                
                week_stats += f"Accuracy: {week_accuracy:.2f}\n"
                week_stats += f"Precision: {week_precision:.2f}\n"
                week_stats += f"Recall: {week_recall:.2f}\n"
                week_stats += f"F1 Score: {week_f1:.2f}\n"
            else:
                week_stats += "Only one class present - metrics not calculated\n"
        
        # Add counts
        week_actual_count = week_actuals.sum()
        week_predicted_count = week_predictions.sum()
        
        week_stats += f"Actual Peaks: {week_actual_count}\n"
        week_stats += f"Predicted Peaks: {week_predicted_count}\n"
        
        ax1.text(0.02, 0.02, week_stats, transform=ax1.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8), fontsize=9)
        
        # Save plot
        plot_path = plots_dir / f"peak_week_{week_num}_{eval_name}.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        plot_paths.append(str(plot_path))
    
    return plot_paths

def main():
    args = parse_arguments()
    
    # Configure logging for the selected model
    configure_logging(args.model)
    
    # Load data before using it
    data = load_data()
    logging.info(f"Loaded data with {len(data)} samples")
    
    if args.model == 'trend':
        # Load trend model
        trend_artifacts = load_trend_model()
        
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
            
            # Print trend metrics
            print("\nTrend Model:")
            for metric, value in results.items():
                if metric.startswith('trend_') and isinstance(value, (int, float)) and not isinstance(value, bool):
                    print(f"  {metric}: {value:.4f}")
    elif args.model == 'peak':
        # Load peak model
        peak_artifacts = load_peak_model()
        
        # Evaluate peak model
        results = evaluate_peak_model(data, peak_artifacts, num_weeks=args.weeks)
        
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
            
            # Print peak metrics
            print("\nPeak Model:")
            for metric, value in results.items():
                if metric.startswith('peak_') and isinstance(value, (int, float)) and not isinstance(value, bool):
                    print(f"  {metric}: {value:.4f}")
    elif args.model == 'valley':
        # Load valley model
        valley_artifacts = load_valley_model()
        
        # Evaluate valley model
        results = evaluate_valley_model(data, valley_artifacts, num_weeks=args.weeks)
        
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
            
            # Print valley metrics
            print("\nValley Model:")
            for metric, value in results.items():
                if metric.startswith('valley_') and isinstance(value, (int, float)) and not isinstance(value, bool):
                    print(f"  {metric}: {value:.4f}")
    elif args.model == 'merged':
        # Create required directories for merged model
        MERGED_EVAL_DIR = EVALUATION_DIR / "merged"
        MERGED_EVAL_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load trend, peak and valley models
        trend_artifacts = load_trend_model()
        peak_artifacts = load_peak_model()
        valley_artifacts = load_valley_model()
        
        # Evaluate merged models with available models
        peak_threshold = args.threshold
        valley_threshold = args.threshold
        
        # Evaluate with the models that were successfully loaded
        if trend_artifacts:
            logging.info(f"Starting merged model evaluation with threshold={args.threshold}")
            
            results = evaluate_merged_models(
                data,
                trend_artifacts,
                peak_artifacts if peak_artifacts else None,
                valley_artifacts if valley_artifacts else None,
                num_weeks=args.weeks,
                use_test_data=args.test_data,
                peak_threshold=peak_threshold,
                valley_threshold=valley_threshold
            )
            
            if results:
                logging.info("Merged model evaluation complete")
                
                # Generate summary report
                generate_summary_report(results, 'merged')
                
                print("\n" + "=" * 80)
                print("MERGED MODEL EVALUATION COMPLETE")
                print("=" * 80)
                print(f"Results saved to: {MERGED_EVAL_DIR}")
                
                # Print key metrics
                print("\nKey performance metrics:")
                for metric, value in results.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and metric != 'threshold':
                        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main() 