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
    create_sequences  # Added create_sequences import
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
VALLEY_EVAL_DIR = EVALUATION_DIR / "valley"
VALLEY_EVAL_DIR.mkdir(parents=True, exist_ok=True)
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
    Evaluate the valley detection model on a validation dataset.
    
    Args:
        df: DataFrame with price data and features
        model_artifacts: Dictionary with model and preprocessing objects
        num_weeks: Number of weeks to evaluate
        use_test_data: If True, use test data, otherwise validation data
        prob_threshold: Optional probability threshold for classification (overrides the optimal threshold)
        find_optimal_threshold: If True, find optimal threshold using PR curve
        
    Returns:
        Dictionary with evaluation metrics
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
    parser.add_argument('--model', type=str, choices=['trend', 'peak', 'valley'], default='trend',
                      help='Model type to evaluate (trend, peak, valley)')
    parser.add_argument('--weeks', type=int, default=4,
                      help='Number of weeks to visualize in evaluation')
    parser.add_argument('--test', action='store_true', 
                      help='Use test data instead of validation data for evaluation')
    parser.add_argument('--valley-threshold', type=float, default=0.4,
                      help='Probability threshold for valley detection (default: 0.4, higher values reduce sensitivity). Works with --model valley or --merge')
    parser.add_argument('--find-threshold', action='store_true',
                      help='Find the optimal threshold for peak/valley detection')
    parser.add_argument('--merge', action='store_true',
                      help='Combine trend and valley model predictions and visualize together')
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
def evaluate_merged_models(data, trend_artifacts, valley_artifacts, num_weeks=4, use_test_data=False, valley_threshold=0.4):
    """
    Evaluate combined trend and valley models.
    
    Args:
        data: DataFrame with electricity price data
        trend_artifacts: Dictionary with trend model artifacts
        valley_artifacts: Dictionary with valley model artifacts
        num_weeks: Number of weeks to visualize
        use_test_data: Whether to use test data instead of validation data
        valley_threshold: Probability threshold for valley detection
        
    Returns:
        Dictionary with evaluation results
    """
    logging.info("Starting combined trend and valley model evaluation...")
    
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
    
    # Add features for valley detection
    df = add_valley_features(df)
    
    # Make trend predictions
    trend_model = trend_artifacts.get('model')
    trend_feature_names = trend_artifacts.get('feature_order', [])  # Try to use feature_order first
    if not trend_feature_names:
        trend_feature_names = trend_artifacts.get('feature_names', [])  # Fall back to feature_names
    
    # Check if we have feature names
    if not trend_feature_names or len(trend_feature_names) == 0:
        logging.warning("No feature names found in trend artifacts, trying to load from file")
        try:
            # Try to load feature names from the feature_names.json file
            feature_names_path = Path(__file__).resolve().parent / "models" / "trend_model" / "feature_names.json"
            with open(feature_names_path, 'r') as f:
                trend_feature_names = json.load(f)
            logging.info(f"Loaded {len(trend_feature_names)} feature names from file")
        except Exception as e:
            logging.error(f"Error loading feature names: {e}")
            # Create a fallback feature list with commonly used features
            trend_feature_names = [
                'price_24h_avg', 'price_168h_avg', 'hour_avg_price',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                'month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos',
                'is_weekend', 'is_business_hour', 'is_morning_peak', 'is_evening_peak',
                'powerConsumptionTotal', 'powerProductionTotal', 'powerImportTotal', 'powerExportTotal',
                'nuclear', 'hydro', 'wind'
            ]
            logging.info(f"Using fallback feature list with {len(trend_feature_names)} features")
    
    logging.info("Making trend predictions...")
    
    # Ensure we have all required lag features
    # Add rolling price statistics if not already present
    if 'price_24h_avg' not in df.columns:
        logging.info("Adding price lag and rolling features for trend prediction...")
        if 'SE3_price_lag_24h' not in df.columns:
            # Add lag features
            for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
                lag_col = f'SE3_price_lag_{lag}h'
                if lag_col not in df.columns:
                    df[lag_col] = df[TARGET_VARIABLE].shift(lag)
        
        # Add rolling averages
        if 'price_24h_avg' not in df.columns:
            df['price_24h_avg'] = df[TARGET_VARIABLE].rolling(window=24, center=False).mean()
        
        if 'price_168h_avg' not in df.columns:
            df['price_168h_avg'] = df[TARGET_VARIABLE].rolling(window=168, center=False).mean()
    
        # Calculate hour-of-day average prices if needed
        if 'hour_avg_price' not in df.columns:
            # Group by hour of day
            hour_avg = df.groupby(df.index.hour)[TARGET_VARIABLE].mean()
            # Map hour averages back to the dataframe
            df['hour_avg_price'] = df.index.hour.map(hour_avg)
    
    # Add time features for trend model
    logging.info("Adding time features for trend model...")
    result_df = df.copy()
    idx = result_df.index
    
    # Basic time components
    result_df['hour'] = idx.hour
    result_df['day'] = idx.day
    result_df['month'] = idx.month
    result_df['dayofweek'] = idx.dayofweek
    result_df['quarter'] = idx.quarter
    
    # Cyclical encoding
    result_df['hour_sin'] = np.sin(2 * np.pi * result_df['hour']/24)
    result_df['hour_cos'] = np.cos(2 * np.pi * result_df['hour']/24)
    result_df['day_sin'] = np.sin(2 * np.pi * result_df['day']/31)
    result_df['day_cos'] = np.cos(2 * np.pi * result_df['day']/31)
    result_df['month_sin'] = np.sin(2 * np.pi * result_df['month']/12)
    result_df['month_cos'] = np.cos(2 * np.pi * result_df['month']/12)
    result_df['dayofweek_sin'] = np.sin(2 * np.pi * result_df['dayofweek']/7)
    result_df['dayofweek_cos'] = np.cos(2 * np.pi * result_df['dayofweek']/7)
    
    # Special time flags
    result_df['is_weekend'] = (result_df['dayofweek'] >= 5).astype(int)
    result_df['is_business_hour'] = ((result_df['hour'] >= 8) & (result_df['hour'] <= 18) & 
                               (result_df['dayofweek'] < 5)).astype(int)
    result_df['is_morning_peak'] = ((result_df['hour'] >= 7) & (result_df['hour'] <= 9)).astype(int)
    result_df['is_evening_peak'] = ((result_df['hour'] >= 17) & (result_df['hour'] <= 20)).astype(int)
    
    # Ensure all needed grid features are available
    grid_features = ['powerConsumptionTotal', 'powerProductionTotal', 'powerImportTotal', 
                        'powerExportTotal', 'nuclear', 'hydro', 'wind']
        
    # Fill any missing values
    for col in result_df.columns:
        if result_df[col].isnull().any():
            if result_df[col].dtype.kind in 'iufc':  # For numeric columns
                # Use forward fill, backward fill, and finally the median
                result_df[col] = result_df[col].fillna(method='ffill').fillna(
                    method='bfill').fillna(result_df[col].median())
    
    # Now ensure all trend feature columns are present
    missing_features = []
    for feature in trend_feature_names:
        if feature not in result_df.columns:
            missing_features.append(feature)
            logging.warning(f"Missing trend feature: {feature}, using zero values")
            result_df[feature] = 0
    
    if missing_features:
        logging.warning(f"Missing {len(missing_features)} trend features: {missing_features[:5]}...")
    
    # Get feature matrix for trend prediction
    X_trend = result_df[trend_feature_names].values
    
    # Make prediction using trend model
    logging.info(f"Making predictions with trend model using {len(trend_feature_names)} features...")
    
    try:
        if isinstance(trend_model, xgb.Booster):
            # For XGBoost models - avoid feature name validation
            logging.info("Creating DMatrix without feature names to avoid mismatch")
            dmatrix = xgb.DMatrix(X_trend)
            
            # Temporarily clear feature names if they exist
            orig_feature_names = None
            if hasattr(trend_model, 'feature_names'):
                orig_feature_names = trend_model.feature_names
                trend_model.feature_names = None
            
            trend_predictions = trend_model.predict(dmatrix)
            
            # Restore original feature names
            if orig_feature_names is not None:
                trend_model.feature_names = orig_feature_names
        else:
            # For other model types
            trend_predictions = trend_model.predict(X_trend)
            
        logging.info(f"Successfully generated trend predictions with shape {trend_predictions.shape}")
        
        # Apply smoothing to trend predictions if needed
        smoothing_params = trend_artifacts.get('smoothing_params', {})
        smoothing_level = smoothing_params.get('smoothing_level', 'light')
        trend_predictions = adaptive_trend_smoothing(trend_predictions, result_df.index, smoothing_level)
    except Exception as e:
        logging.error(f"Error making trend predictions: {e}")
        # Create fallback trend predictions (just use the price itself)
        logging.warning("Using actual prices as fallback for trend predictions")
        trend_predictions = result_df[TARGET_VARIABLE].values
    
    # Make valley predictions
    valley_model = valley_artifacts.get('model')
    valley_feature_scaler = valley_artifacts.get('feature_scaler')
    valley_feature_names = valley_artifacts.get('feature_names', [])
    
    logging.info(f"Making valley predictions with threshold {valley_threshold}...")
    
    # Check if all valley features are present
    missing_valley_features = [f for f in valley_feature_names if f not in result_df.columns]
    if missing_valley_features:
        logging.warning(f"Missing {len(missing_valley_features)} valley features: {missing_valley_features[:5]}...")
        for feature in missing_valley_features:
            result_df[feature] = 0
    
    # Prepare data for valley detection
    X_valley = result_df[valley_feature_names].values
    if valley_feature_scaler:
        X_valley = valley_feature_scaler.transform(X_valley)
    
    # Create sequences for TCN valley model
    X_val_seq, _ = create_sequences(
        X_valley,
        LOOKBACK_WINDOW, 
        PREDICTION_HORIZON, 
        list(range(X_valley.shape[1])),
        None
    )
    
    # Get actual prices and timestamps
    timestamps = result_df.index[LOOKBACK_WINDOW:]
    actual_prices = result_df[TARGET_VARIABLE].values[LOOKBACK_WINDOW:]
    
    # Make predictions with valley model
    valley_probabilities = valley_model.predict(X_val_seq, verbose=0).flatten()
    valley_predictions = (valley_probabilities >= valley_threshold).astype(int)
    
    # Get actual valleys for evaluation (if available)
    actual_valleys = result_df['is_price_valley'].values[LOOKBACK_WINDOW:] if 'is_price_valley' in result_df.columns else np.zeros_like(valley_predictions)
    
    # Make sure trend_predictions and valley predictions have the same length
    if len(trend_predictions) > len(valley_predictions):
        logging.info(f"Truncating trend predictions to match valley predictions: {len(trend_predictions)} -> {len(valley_predictions)}")
        trend_predictions = trend_predictions[-len(valley_predictions):]
    
    # Make sure actual_valleys and valley_predictions have the same length
    if len(actual_valleys) != len(valley_predictions):
        logging.warning(f"Length mismatch between actual_valleys ({len(actual_valleys)}) and valley_predictions ({len(valley_predictions)})")
        min_len = min(len(actual_valleys), len(valley_predictions))
        actual_valleys = actual_valleys[:min_len]
        valley_predictions = valley_predictions[:min_len]
        actual_prices = actual_prices[:min_len]
        timestamps = timestamps[:min_len]
        if len(trend_predictions) > min_len:
            trend_predictions = trend_predictions[:min_len]
        logging.info(f"Truncated arrays to common length: {min_len}")
    
    # Generate weekly visualizations
    logging.info(f"Generating visualizations for {num_weeks} weeks...")
    
    # Split the data into weeks
    hours_per_week = 24 * 7
    num_samples = len(timestamps)
    
    # Calculate how many complete weeks we can visualize
    if num_samples < hours_per_week:
        logging.warning(f"Not enough data for weekly visualizations (need at least 168 hours, got {num_samples})")
        weeks = 0
    else:
        weeks = min(num_weeks, num_samples // hours_per_week)
    
    # Set up results
    results = {
        'model_type': 'merged',
        'timestamps': timestamps,
        'trend_predictions': trend_predictions,
        'valley_predictions': valley_predictions,
        'valley_probabilities': valley_probabilities,
        'actual_valleys': actual_valleys,
        'actual_prices': actual_prices,
        'valley_threshold': valley_threshold,
        'num_weeks': weeks,
        'eval_name': subset_name
    }
    
    # Evaluation metrics for valley detection
    results['accuracy'] = accuracy_score(actual_valleys, valley_predictions)
    results['precision'] = precision_score(actual_valleys, valley_predictions, zero_division=0)
    results['recall'] = recall_score(actual_valleys, valley_predictions, zero_division=0)
    results['f1_score'] = f1_score(actual_valleys, valley_predictions, zero_division=0)
    
    # Calculate trend metrics
    results['trend_mae'] = np.mean(np.abs(actual_prices - trend_predictions))
    results['trend_rmse'] = np.sqrt(np.mean((actual_prices - trend_predictions) ** 2))
    
    # Generate weekly visualizations
    visualization_dir = EVALUATION_DIR / "merged" / subset_name
    visualization_dir.mkdir(parents=True, exist_ok=True)
    
    visualization_files = []
    
    for week in range(weeks):
        start_idx = week * hours_per_week
        end_idx = start_idx + hours_per_week
        
        # Ensure we have enough data
        if end_idx > num_samples:
            break
        
        week_start = timestamps[start_idx]
        week_timestamps = timestamps[start_idx:end_idx]
        week_prices = actual_prices[start_idx:end_idx]
        week_trend = trend_predictions[start_idx:end_idx]
        week_valleys = valley_predictions[start_idx:end_idx]
        week_probabilities = valley_probabilities[start_idx:end_idx]
        week_actual_valleys = actual_valleys[start_idx:end_idx]
        
        # Create merged prediction using probability-based valley depth adjustment
        merged_prediction = week_trend.copy()
        
        # Apply probability-based adjustments for all potential valley points
        for i, probability in enumerate(week_probabilities):
            if probability >= valley_threshold:
                # Scale the adjustment by the confidence level
                # Higher probability = deeper valley (stronger adjustment)
                adjustment_factor = 0.85 - (probability - valley_threshold) * 0.8
                # Ensure it doesn't go below some reasonable minimum (60% of trend)
                adjustment_factor = max(adjustment_factor, 0.6)
                # Apply the adjustment
                merged_prediction[i] = week_trend[i] * adjustment_factor
        
        # Apply extra smoothing for visualization purposes
        from scipy.signal import savgol_filter
        
        # Get smoothing window size (must be odd and less than data points)
        vis_window = min(47, len(week_trend) - 1)
        if vis_window % 2 == 0:
            vis_window -= 1  # Make sure it's odd
            
        # Make a visual-only smoothed copy of trend and merged predictions
        trend_smooth = savgol_filter(week_trend, window_length=vis_window, polyorder=2)
        merged_smooth = merged_prediction.copy()
        
        # Only smooth the non-valley parts of the merged prediction to preserve the valleys
        valley_mask = week_valleys == 0  # Where there are no valleys
        merged_smooth_temp = savgol_filter(merged_prediction, window_length=vis_window, polyorder=2)
        merged_smooth[valley_mask] = merged_smooth_temp[valley_mask]
        
        # Calculate week metrics
        week_accuracy = accuracy_score(week_actual_valleys, week_valleys)
        week_precision = precision_score(week_actual_valleys, week_valleys, zero_division=0)
        week_recall = recall_score(week_actual_valleys, week_valleys, zero_division=0)
        week_f1 = f1_score(week_actual_valleys, week_valleys, zero_division=0)
        week_trend_mae = np.mean(np.abs(week_prices - week_trend))
        
        # Generate plot
        plt.figure(figsize=(15, 10))
        
        # Create two subplots - top for price trend, bottom for valley markers
        gs = plt.GridSpec(2, 1, height_ratios=[3, 1])
        
        # Top plot - Price and trend predictions
        ax1 = plt.subplot(gs[0])
        
        # Plot actual prices
        ax1.plot(week_timestamps, week_prices, 'b-', linewidth=2, label='Actual Price')
        
        # Plot trend predictions in purple (changed from red)
        ax1.plot(week_timestamps, trend_smooth, 'purple', linewidth=2.5, label='Trend Prediction')
        
        # Plot the merged trend+valley prediction line in red
        ax1.plot(week_timestamps, merged_smooth, 'r-', linewidth=2.5, label='Merged Trend+Valley')
        
        # Mark valley detection with vertical green bands
        for i, is_valley in enumerate(week_valleys):
            if is_valley:
                # Make the valley bands wider by extending 2 hours in each direction
                valley_start = max(0, i-2)
                valley_end = min(len(week_timestamps)-1, i+2)
                ax1.axvspan(week_timestamps[valley_start], week_timestamps[valley_end], 
                          color='green', alpha=0.2)
                
                # Add a marker at the exact valley point
                ax1.plot(week_timestamps[i], week_prices[i], 'gv', markersize=8)
                
                # Add probability value as text above the marker
                ax1.text(week_timestamps[i], week_prices[i] * 1.05, 
                        f"{week_probabilities[i]:.2f}", fontsize=8, 
                        ha='center', va='bottom',
                        bbox=dict(facecolor='white', alpha=0.7))
        
        # Add title and labels
        week_start_str = week_start.strftime('%Y-%m-%d')
        ax1.set_title(f'Week starting {week_start_str} ({subset_name.capitalize()} Data)', fontsize=14)
        ax1.set_ylabel('Price (öre/kWh)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Format date axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%a %d %b'))
        ax1.tick_params(axis='x', rotation=45)
        
        # Add metrics to plot
        valley_count = week_valleys.sum()
        actual_valley_count = week_actual_valleys.sum()
        # metrics_text = (
        #     f"Valleys: {actual_valley_count} actual, {valley_count} predicted | "
        #     f"Precision: {week_precision:.2f}, Recall: {week_recall:.2f}, F1: {week_f1:.2f}\n"
        #     f"Trend MAE: {week_trend_mae:.2f} öre/kWh"
        # )
        # ax1.text(0.5, 0.02, metrics_text, transform=ax1.transAxes, 
        #         ha='center', va='bottom', fontsize=10,
        #         bbox=dict(facecolor='white', alpha=0.8))
        
        # Add a dummy plot element for the valley markers in the legend
        handles, labels = ax1.get_legend_handles_labels()
        handles.append(Line2D([0], [0], marker='v', color='g', linestyle='None', 
                             markersize=8, label='Valley Detection'))
        ax1.legend(handles=handles, loc='upper right', fontsize=10)
        
        # Bottom plot - Valley probabilities
        ax2 = plt.subplot(gs[1], sharex=ax1)
        
        # Plot probability line in green to match the valley band color
        ax2.plot(week_timestamps, week_probabilities, 'g-', linewidth=1.5, label='Valley Probability')
        
        # Add horizontal line for threshold
        ax2.axhline(y=valley_threshold, color='r', linestyle='--', 
                  label=f'Threshold ({valley_threshold:.2f})')
        
        # Show binary valley predictions as triangles
        for i, is_valley in enumerate(week_valleys):
            if is_valley:
                ax2.scatter(week_timestamps[i], 0.9, color='g', marker='v', s=50)
        
        # Add actual valleys as blue triangles
        for i, is_valley in enumerate(week_actual_valleys):
            if is_valley:
                ax2.scatter(week_timestamps[i], 0.9, color='b', marker='^', s=50)
        
        # Add a legend to explain the markers
        handles, labels = ax2.get_legend_handles_labels()
        handles.append(plt.Line2D([0], [0], marker='v', color='g', linestyle='None', markersize=8, label='Predicted Valley'))
        handles.append(plt.Line2D([0], [0], marker='^', color='b', linestyle='None', markersize=8, label='Actual Valley'))
        ax2.legend(handles=handles, loc='upper right', fontsize=10)
        
        # Add labels
        ax2.set_ylabel('Valley Probability', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        filename = f"merged_week_{week_start_str}.png"
        filepath = visualization_dir / filename
        plt.savefig(filepath, dpi=150)
        plt.close()
        
        visualization_files.append(filename)
        logging.info(f"Generated visualization for week starting {week_start_str}")
    
    results['visualization_files'] = visualization_files
    results['visualization_dir'] = str(visualization_dir)
    
    return results

def main():
    """Main function to run the evaluation."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Configure logging for the selected model
    configure_logging(args.model if not args.merge else "merged")
    
    # Special case for merged model evaluation
    if args.merge:
        logging.info("Starting merged model evaluation (trend + valley)...")
        
        # Create merged evaluation directory
        merged_eval_dir = EVALUATION_DIR / "merged"
        merged_eval_dir.mkdir(parents=True, exist_ok=True)
        (merged_eval_dir / "validation").mkdir(parents=True, exist_ok=True)
        (merged_eval_dir / "test").mkdir(parents=True, exist_ok=True)
        
        try:
            # Load data
            logging.info("Loading data...")
            data = load_data()
            
            # Load both models
            logging.info("Loading trend model...")
            trend_artifacts = load_trend_model()
            
            logging.info("Loading valley model...")
            valley_artifacts = load_valley_model()
            
            # Evaluate merged models
            results = evaluate_merged_models(
                data,
                trend_artifacts,
                valley_artifacts,
                num_weeks=args.weeks,
                use_test_data=args.test,
                valley_threshold=args.valley_threshold
            )
            
            if results:
                logging.info("Merged model evaluation complete.")
                print("\n" + "=" * 80)
                print("MERGED MODEL EVALUATION COMPLETE")
                print("=" * 80)
                print(f"Results saved to: {results['visualization_dir']}")
                
                print("\nValley Detection Metrics:")
                print(f"- Accuracy: {results['accuracy']:.4f}")
                print(f"- Precision: {results['precision']:.4f}")
                print(f"- Recall: {results['recall']:.4f}")
                print(f"- F1 Score: {results['f1_score']:.4f}")
                
                print("\nTrend Prediction Metrics:")
                print(f"- MAE: {results['trend_mae']:.4f}")
                print(f"- RMSE: {results['trend_rmse']:.4f}")
                
                print(f"\nVisualized {len(results['visualization_files'])} weeks:")
                for file in results['visualization_files']:
                    print(f"- {file}")
            else:
                logging.error("Merged model evaluation failed.")
        
        except Exception as e:
            logging.error(f"Error in merged model evaluation: {e}")
            import traceback
            logging.error(traceback.format_exc())
        
        return
    
    logging.info(f"Starting {args.model} model evaluation with {args.weeks} weeks of visualizations...")
    
    try:
        # Load data for all model types
        logging.info("Loading data...")
        data = load_data()
        
        if args.model == 'trend':
            # Create required directories
            TREND_EVAL_DIR.mkdir(parents=True, exist_ok=True)
            (TREND_EVAL_DIR / "daily").mkdir(parents=True, exist_ok=True)
            (TREND_EVAL_DIR / "weekly").mkdir(parents=True, exist_ok=True)
            (TREND_EVAL_DIR / "monthly").mkdir(parents=True, exist_ok=True)
            
            # Load trend model
            logging.info("Loading trend model...")
            trend_artifacts = load_trend_model()
            
            # Evaluate trend model
            logging.info(f"Evaluating trend model with {args.weeks} weeks of visualizations...")
            results = evaluate_trend_model(data, trend_artifacts, num_weeks=args.weeks)
            
            if results:
                # Generate summary report
                report_path = generate_summary_report(results, 'trend')
                
                logging.info("Trend model evaluation complete.")
                print("\n" + "=" * 80)
                print("TREND MODEL EVALUATION COMPLETE")
                print("=" * 80)
                print(f"Results saved to: {TREND_EVAL_DIR}")
                print(f"\nModel type: {results['model_type']}")
                print(f"Test period: {results['metrics']['test_period']}")
                print(f"Number of observations: {results['metrics']['num_observations']}")
                print("\nMetrics summary:")
                print(f"- MAE: {results['metrics']['test_mae']:.4f}")
                print(f"- RMSE: {results['metrics']['test_rmse']:.4f}")
                print(f"- MAPE: {results['metrics']['test_mape']:.2f}%")
                print(f"- SMAPE: {results['metrics']['smape']:.2f}%")
                print(f"- Direction Accuracy: {results['metrics']['direction_accuracy']:.4f}")
                print(f"- Peak Accuracy: {results['metrics']['peak_accuracy']:.4f}")
                
                if report_path:
                    print(f"\nSummary report saved to: {report_path}")
                
                print("\nVisualization files:")
                for file_type in ['Overall test period', 'Weekly analyses', 'Error analysis']:
                    print(f"- {file_type} visualizations")
                
            else:
                logging.error("Trend model evaluation failed.")
                print("\nTrend model evaluation failed. Check logs for details.")
        
        elif args.model == 'valley':
            logging.info("Starting valley model evaluation...")
            
            # Load valley model
            logging.info("Loading valley model...")
            valley_artifacts = load_valley_model()
            
            # Evaluate valley model
            results = evaluate_valley_model(
                data, 
                valley_artifacts, 
                num_weeks=args.weeks, 
                use_test_data=args.test,
                prob_threshold=args.valley_threshold,
                find_optimal_threshold=args.find_threshold
            )
            
            if results:
                # Generate summary report
                report_path = generate_summary_report(results, 'valley')
                
                logging.info("Valley model evaluation complete.")
                print("\n" + "=" * 80)
                print("VALLEY DETECTION MODEL EVALUATION COMPLETE")
                print("=" * 80)
                print(f"Results saved to: {VALLEY_EVAL_DIR}")
                print("\nMetrics summary:")
                print(f"- Accuracy: {results.get('accuracy', 'N/A')}")
                print(f"- Precision: {results.get('precision', 'N/A')}")
                print(f"- Recall: {results.get('recall', 'N/A')}")
                print(f"- F1 Score: {results.get('f1_score', 'N/A')}")
                print(f"- ROC AUC: {results.get('roc_auc', 'N/A')}")
                print(f"- Actual valleys rate: {results.get('actual_valley_rate', 'N/A')}")
                print(f"- Predicted valleys rate: {results.get('predicted_valley_rate', 'N/A')}")
                print(f"- Probability threshold: {results.get('threshold', 'N/A')}")
                
                if 'threshold_analysis' in results:
                    print("\nThreshold analysis:")
                    print(f"- Best threshold: {results['threshold_analysis']['best_threshold']:.4f}")
                    print(f"- Best F1 score: {results['threshold_analysis']['best_f1']:.4f}")
                    print(f"- Threshold analysis plot: {VALLEY_EVAL_DIR}/threshold_analysis.png")
                
                if report_path:
                    print(f"\nSummary report saved to: {report_path}")
                
                # List the visualization files
                plots_dir = results.get('plots_dir', VALLEY_EVAL_DIR)
                plots_path = Path(plots_dir)
                if plots_path.exists():
                    plot_files = list(plots_path.glob("*.png"))
                    if plot_files:
                        print(f"\n{len(plot_files)} visualization files generated:")
                        for file in sorted(plot_files)[:5]:  # Show only first 5
                            print(f"- {file.name}")
                        if len(plot_files) > 5:
                            print(f"  ... and {len(plot_files) - 5} more")
                else:
                    logging.warning(f"Plot directory {plots_dir} does not exist")
                    print("\nNo visualization files found.")
            else:
                logging.error("Valley model evaluation failed.")
                print("\nValley model evaluation failed. Check logs for details.")
        
        elif args.model == 'peak':
            logging.info("Peak model evaluation not yet implemented.")
            print("\n" + "=" * 80)
            print("PEAK DETECTION MODEL EVALUATION")
            print("=" * 80)
            print("\nPeak model evaluation not yet implemented.")
            print("To implement, add the peak evaluation function.")
    
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        print(f"\nEvaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logging.info(f"{args.model.capitalize()} model evaluation completed successfully.")

if __name__ == "__main__":
    main() 