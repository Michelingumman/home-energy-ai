#!/usr/bin/env python
"""
Unified prediction script for electricity price forecasting.
Supports trend model, peak detection model, and valley detection model.
Can combine all models for optimal predictions.

Usage:
    python predict.py                   (use all models with default 1-day horizon)
    python predict.py --model trend     (use trend model with default 1-day horizon)
    python predict.py --model trend --horizon 7  (predict for 7 days)
    python predict.py --model peak 
    python predict.py --model valley
    python predict.py --model all --show_plot
    python predict.py --production_mode
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pickle
import logging
import json
import joblib
from pathlib import Path
import argparse
from datetime import datetime, timedelta
from tcn import TCN
import os
import sys
import xgboost as xgb  # Added XGBoost import

# Add the directory of this file to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import utility functions
from utils import (
    add_time_features, add_lag_features, add_rolling_features,
    add_price_spike_indicators, LogTransformScaler, CustomBoundedScaler,
    load_and_merge_data, spike_weighted_loss, detect_price_valleys_derivative
)

# Import scipy for filters
from scipy.signal import medfilt, savgol_filter

# Import config
from config import (
    MODELS_DIR, PLOTS_DIR, TARGET_VARIABLE,
    LOOKBACK_WINDOW, PREDICTION_HORIZON,
    SE3_PRICES_FILE, SWEDEN_GRID_FILE, TIME_FEATURES_FILE, 
    HOLIDAYS_FILE, WEATHER_DATA_FILE,
    PRICE_FEATURES, GRID_FEATURES, TIME_FEATURES,
    HOLIDAY_FEATURES, WEATHER_FEATURES, MARKET_FEATURES,
    CORE_FEATURES, EXTENDED_FEATURES,
    TREND_MODEL_DIR, PEAK_MODEL_DIR, VALLEY_MODEL_DIR,
    DEFAULT_PREDICTION_HORIZON, DEFAULT_CONFIDENCE_LEVEL, ROLLING_PREDICTION_WINDOW
)

# Define smoothing functions locally since they're not in utils.py
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

def adaptive_trend_smoothing(data, timestamps, smoothing_level='medium'):
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

# Create necessary directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR = PLOTS_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
TREND_MODEL_DIR.mkdir(parents=True, exist_ok=True)
PEAK_MODEL_DIR.mkdir(parents=True, exist_ok=True)
VALLEY_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(MODELS_DIR / "price_prediction.log"),
        logging.StreamHandler()
    ]
)

def load_trend_model():
    """Load the trained trend model and associated artifacts."""
    # First try to load the XGBoost model
    xgb_model_path = TREND_MODEL_DIR / "best_trend_model.pkl"
    
    model = None
    model_type = None
    
    # Try loading XGBoost model first
    if xgb_model_path.exists():
        try:
            model = xgb.Booster()
            model.load_model(str(xgb_model_path))
            model_type = "xgboost"
            logging.info(f"Loaded XGBoost trend model from {xgb_model_path}")
        except Exception as e:
            logging.error(f"Error loading XGBoost trend model: {e}")

    # Load feature list - for XGBoost we use a different file
    if model_type == "xgboost":
        feature_list_path = TREND_MODEL_DIR / "feature_names.json"
    else:
        feature_list_path = TREND_MODEL_DIR / "feature_list_trend_model.json"
        
    try:
        with open(feature_list_path, 'r') as f:
            feature_names = json.load(f)
        logging.info(f"Loaded feature list with {len(feature_names)} features")
    except Exception as e:
        logging.error(f"Error loading feature list: {e}")
        raise
    
    # Load target scaler
    target_scaler_json_path = TREND_MODEL_DIR / "target_scaler_trend_model.json"
    target_scaler_joblib_path = TREND_MODEL_DIR / "target_scaler_trend_model.save"
    
    target_scaler = None
    
    if target_scaler_json_path.exists():
        # Load custom scaler
        try:
            with open(target_scaler_json_path, 'r') as f:
                scaler_params = json.load(f)
            
            if scaler_params.get('type') == 'LogTransformScaler':
                target_scaler = LogTransformScaler.load(target_scaler_json_path)
            elif scaler_params.get('type') == 'CustomBoundedScaler':
                target_scaler = CustomBoundedScaler.load(target_scaler_json_path)
            else:
                raise ValueError(f"Unknown scaler type: {scaler_params.get('type')}")
                
            logging.info(f"Loaded custom target scaler from {target_scaler_json_path}")
        except Exception as e:
            logging.error(f"Error loading custom target scaler: {e}")
    elif target_scaler_joblib_path.exists():
        # Load standard sklearn scaler
        try:
            target_scaler = joblib.load(target_scaler_joblib_path)
            logging.info(f"Loaded target scaler from {target_scaler_joblib_path}")
        except Exception as e:
            logging.error(f"Error loading target scaler: {e}")
    
    # For XGBoost, no target scaler is typically necessary
    if target_scaler is None and model_type == "xgboost":
        logging.info("No target scaler found, but using XGBoost model which doesn't require one")
    elif target_scaler is None:
        logging.error(f"No target scaler found for Keras model")
        raise FileNotFoundError("No target scaler found")
    
    # Load feature scaler if it exists
    feature_scaler = None
    feature_scaler_path = TREND_MODEL_DIR / "feature_scaler_trend_model.save"
    if feature_scaler_path.exists():
        try:
            feature_scaler = joblib.load(feature_scaler_path)
            logging.info(f"Loaded feature scaler from {feature_scaler_path}")
        except Exception as e:
            logging.warning(f"Error loading feature scaler: {e}")
    
    # Load smoothing parameters for XGBoost
    smoothing_params = None
    if model_type == "xgboost":
        params_path = TREND_MODEL_DIR / "trend_model_params.json" 
        try:
            with open(params_path, 'r') as f:
                model_params = json.load(f)
                smoothing_params = model_params.get('smoothing', {
                    'smoothing_level': 'medium',
                    'exponential_alpha': 0.05,
                    'median_window': 11,
                    'savgol_window': 23,
                    'savgol_polyorder': 2
                })
            logging.info(f"Loaded smoothing parameters: {smoothing_params}")
        except Exception as e:
            logging.warning(f"Error loading model parameters: {e}")
    
    return {
        'model': model,
        'model_type': model_type,
        'feature_names': feature_names,
        'target_scaler': target_scaler,
        'feature_scaler': feature_scaler,
        'smoothing_params': smoothing_params
    }

def load_peak_valley_model(model_type):
    """Load the trained peak or valley model and associated artifacts."""
    if model_type == 'peak':
        model_dir = PEAK_MODEL_DIR
        model_name = 'peak_model'
    elif model_type == 'valley':
        model_dir = VALLEY_MODEL_DIR
        model_name = 'valley_model'
    else:
        raise ValueError(f"Invalid model_type: {model_type}")
    
    model_path = model_dir / f"best_{model_name}.keras"
    
    try:
        # Load custom objects if needed
        custom_objects = {'TCN': TCN}
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        logging.info(f"Loaded {model_type} model from {model_path}")
    except Exception as e:
        logging.error(f"Error loading {model_type} model: {e}")
        raise
    
    # Load feature list
    feature_list_path = model_dir / f"feature_list_{model_name}.json"
    try:
        with open(feature_list_path, 'r') as f:
            feature_names = json.load(f)
        logging.info(f"Loaded feature list with {len(feature_names)} features")
    except Exception as e:
        logging.error(f"Error loading feature list: {e}")
        raise
    
    # Load feature scaler if it exists
    feature_scaler = None
    feature_scaler_path = model_dir / f"feature_scaler_{model_name}.save"
    if feature_scaler_path.exists():
        try:
            feature_scaler = joblib.load(feature_scaler_path)
            logging.info(f"Loaded feature scaler from {feature_scaler_path}")
        except Exception as e:
            logging.warning(f"Error loading feature scaler: {e}")
    
    # Load thresholds if they exist
    thresholds = None
    thresholds_path = model_dir / "thresholds.json"
    if thresholds_path.exists():
        try:
            with open(thresholds_path, 'r') as f:
                thresholds = json.load(f)
            logging.info(f"Loaded thresholds: {thresholds}")
        except Exception as e:
            logging.warning(f"Error loading thresholds: {e}")
    
    return {
        'model': model,
        'feature_names': feature_names,
        'feature_scaler': feature_scaler,
        'thresholds': thresholds
    } 

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
    # Filter to include only the features we need
    available_features = list(set(feature_names).intersection(set(df.columns)))
    missing_features = list(set(feature_names) - set(available_features))
    
    if missing_features:
        logging.warning(f"Missing features: {missing_features}")
    
    df_filtered = df[available_features].copy()
    
    # Number of sequences we can create
    n_sequences = (len(df_filtered) - window_size) // stride + 1
    
    # Create sequences
    sequences = np.zeros((n_sequences, window_size, len(available_features)))
    
    for i in range(n_sequences):
        start_idx = i * stride
        end_idx = start_idx + window_size
        sequences[i] = df_filtered.iloc[start_idx:end_idx].values
    
    return sequences, available_features

def predict_with_trend_model(trend_artifacts, df, horizon=PREDICTION_HORIZON, show_plot=False):
    """Make predictions using the trend model.
    
    Args:
        trend_artifacts: Dict with model and artifacts
        df: DataFrame with features
        horizon: Number of hours to predict ahead (can be hours or day-converted to hours)
        show_plot: Whether to show a plot of predictions
        
    Returns:
        DataFrame with predictions
    """
    model = trend_artifacts['model']
    model_type = trend_artifacts['model_type']
    feature_names = trend_artifacts['feature_names']
    target_scaler = trend_artifacts['target_scaler']
    feature_scaler = trend_artifacts['feature_scaler']
    smoothing_params = trend_artifacts.get('smoothing_params', None)
    
    # We can assume horizon is already in hours from the predict_prices function
    horizon_hours = horizon
    
    # Get the latest timestamp in the data
    latest_timestamp = df.index.max()
    
    # Create prediction timestamps
    prediction_timestamps = pd.date_range(
        start=latest_timestamp + timedelta(hours=1),
        periods=horizon_hours,
        freq='h'
    )
    
    # Common data preparation for both model types
    # Prepare the features differently based on model type
    if model_type == "xgboost":
        # For XGBoost, we need to prepare the features directly
        # Add time features
        input_df = df.copy()
        
        # Add time features for future timestamps
        future_df = pd.DataFrame(index=prediction_timestamps)
        future_df = add_time_features(future_df)
        
        # Other features to include from the most recent data
        for feature in feature_names:
            if feature not in future_df.columns and feature in input_df.columns:
                # For lag features, rolling stats, etc., use the most recent values
                future_df[feature] = input_df[feature].iloc[-1]
        
        # Remove any features not in the feature list
        for feat in list(future_df.columns):
            if feat not in feature_names:
                future_df.drop(feat, axis=1, inplace=True)
        
        # Make sure all features from the feature list are included
        missing_features = set(feature_names) - set(future_df.columns)
        if missing_features:
            logging.warning(f"Missing features for prediction: {missing_features}")
            for feat in missing_features:
                # Add zeros for missing features
                future_df[feat] = 0
        
        # Make sure columns are in the same order as feature_names
        future_df = future_df[feature_names]
        
        # Prepare for XGBoost prediction
        dtest = xgb.DMatrix(future_df)
        
        # Make predictions
        predictions = model.predict(dtest)
        
        # Apply smoothing if parameters are available
        if smoothing_params:
            predictions = adaptive_trend_smoothing(
                predictions, 
                future_df.index, 
                smoothing_level=smoothing_params.get('smoothing_level', 'medium')
            )
    else:
        # For the Keras model, we need to create sequences
        # Prepare sequence data
        X_pred, used_features = prepare_sequence_data(
            df, 
            feature_names,
            window_size=LOOKBACK_WINDOW,
            stride=1,
            prediction_horizon=horizon_hours
        )
        
        # Apply feature scaling if needed
        if feature_scaler is not None:
            # Reshape for scaling
            original_shape = X_pred.shape
            X_pred_2d = X_pred.reshape(-1, X_pred.shape[-1])
            
            # Scale features
            X_pred_2d = feature_scaler.transform(X_pred_2d)
            
            # Reshape back
            X_pred = X_pred_2d.reshape(original_shape)
        
        # Make predictions
        predictions = model.predict(X_pred)
        
        # Apply inverse transform if target scaler is available
        if target_scaler is not None:
            # Reshape for inverse transform
            original_shape = predictions.shape
            predictions_2d = predictions.reshape(-1, 1)
            
            # Inverse transform
            predictions_2d = target_scaler.inverse_transform(predictions_2d)
            
            # Reshape back
            predictions = predictions_2d.reshape(original_shape)
        
        # Get only the first sequence predictions as we only need one forecast
        predictions = predictions[0]
    
    # Create a DataFrame with the predictions
    predictions_df = pd.DataFrame({
        'timestamp': prediction_timestamps,
        'predicted_price': predictions
    })
    predictions_df.set_index('timestamp', inplace=True)
    
    # Plot if requested
    if show_plot:
        plt.figure(figsize=(15, 8))
        
        # Historical prices - last 7 days if available
        hist_days = 7
        hist_hours = hist_days * 24
        if len(df) >= hist_hours:
            plt.plot(df.index[-hist_hours:], df[TARGET_VARIABLE][-hist_hours:], 'b-', label='Historical Prices')
        else:
            plt.plot(df.index, df[TARGET_VARIABLE], 'b-', label='Historical Prices')
        
        # Predicted prices
        plt.plot(predictions_df.index, predictions_df['predicted_price'], 'r--', marker='x', label='Predicted Prices')
        
        # Add vertical line at current time
        plt.axvline(x=latest_timestamp, color='k', linestyle='--', alpha=0.5)
        
        # Calculate the number of days in the forecast for the title
        forecast_days = horizon_hours / 24
        
        # Customize the plot
        plt.title(f'{forecast_days:.1f} Day Price Forecast')
        plt.xlabel('Date')
        plt.ylabel(f'{TARGET_VARIABLE}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gcf().autofmt_xdate()
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(PREDICTIONS_DIR / f"trend_prediction_{horizon_hours}h.png")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    return predictions_df

def predict_with_peak_valley_model(model_artifacts, df, model_type, horizon=PREDICTION_HORIZON, show_plot=False):
    """Make predictions using the peak or valley model.
    
    Args:
        model_artifacts: Dict with model and artifacts
        df: DataFrame with features
        model_type: 'peak' or 'valley'
        horizon: Number of steps to predict ahead
        show_plot: Whether to show a plot of predictions
        
    Returns:
        DataFrame with predictions and probabilities
    """
    model = model_artifacts['model']
    feature_names = model_artifacts['feature_names']
    feature_scaler = model_artifacts['feature_scaler']
    thresholds = model_artifacts.get('thresholds')
    
    # Prepare sequence data - we only need one prediction per sequence
    X_sequences, available_features = prepare_sequence_data(
        df, feature_names, prediction_horizon=1
    )
    
    if feature_scaler:
        # Reshape for scaling
        original_shape = X_sequences.shape
        X_sequences_reshaped = X_sequences.reshape(-1, len(available_features))
        
        # Scale features
        X_sequences_scaled = feature_scaler.transform(X_sequences_reshaped)
        
        # Reshape back
        X_sequences = X_sequences_scaled.reshape(original_shape)
    
    # Make predictions
    y_prob = model.predict(X_sequences)
    
    # For valley model, adjust threshold based on the derivative approach
    if model_type == 'valley' and thresholds and thresholds.get('method') == 'derivative':
        # Use a higher threshold (0.7) for derivative-based valleys to improve precision
        threshold = 0.7
        logging.info(f"Using higher threshold ({threshold}) for derivative-based valley detection")
        y_pred = (y_prob >= threshold).astype(int)
    else:
        # Default threshold of 0.5 for other models
        y_pred = (y_prob >= 0.5).astype(int)
    
    # Create DataFrame with predictions
    prediction_times = []
    for i in range(len(y_pred)):
        # Start from the end of the sequence
        # Fix the indexing to avoid out of bounds error
        idx = min(LOOKBACK_WINDOW + i, len(df) - 1)  # Ensure we don't go beyond the last index
        prediction_times.append(df.index[idx])  # Get the current timestamp (not +1, since we prepare data correctly)
    
    # Create DataFrame
    if model_type == 'peak':
        pred_df = pd.DataFrame({
            'timestamp': prediction_times,
            'peak_probability': y_prob.flatten(),
            'is_predicted_peak': y_pred.flatten()
        }).set_index('timestamp')
    else:
        pred_df = pd.DataFrame({
            'timestamp': prediction_times,
            'valley_probability': y_prob.flatten(),
            'is_predicted_valley': y_pred.flatten()
        }).set_index('timestamp')
    
    # If we have duplicates (from overlapping sequences), take the latest prediction
    pred_df = pred_df.groupby(pred_df.index).last()
    
    # Plot predictions if requested
    if show_plot:
        plt.figure(figsize=(15, 6))
        
        # Plot historical prices
        ax1 = plt.gca()
        ax1.plot(df.index[-48:], df[TARGET_VARIABLE].values[-48:], label='Price', color='blue')
        
        # Add second y-axis for probabilities
        ax2 = ax1.twinx()
        
        # Get the appropriate threshold based on model type
        if model_type == 'peak' and thresholds:
            threshold = thresholds.get('spike_threshold', None)
            if threshold:
                ax1.axhline(y=threshold, color='green', linestyle='--', label=f'Peak Threshold ({threshold:.1f})')
            
            # Plot peak probabilities
            pred_indices = pred_df.index[:horizon]
            ax2.plot(pred_indices, pred_df['peak_probability'].values[:horizon], label='Peak Probability', color='red', marker='o')
            
            # Mark predicted peaks
            peak_indices = pred_df[pred_df['is_predicted_peak'] == 1].index[:horizon]
            if len(peak_indices) > 0:
                ax1.scatter(peak_indices, df.loc[peak_indices, TARGET_VARIABLE], color='red', s=100, marker='*', label='Predicted Peaks')
            
            title = 'Price Peak Detection'
            
        elif model_type == 'valley' and thresholds:
            # For derivative method, show derivative threshold differently
            if thresholds.get('method') == 'derivative':
                method_name = 'Derivative-based Valley Detection'
                # If we have derivative-specific thresholds
                slope_threshold = thresholds.get('slope_threshold', 0.15)
                curvature_threshold = thresholds.get('curvature_threshold', 0.1)
                # Draw a horizontal line at the bottom of the chart for reference
                price_min = df[TARGET_VARIABLE][-48:].min()
                ax1.text(df.index[-35], price_min, 
                         f'Using derivative method (slope={slope_threshold}, curvature={curvature_threshold})',
                         color='green', fontsize=8)
            else:
                method_name = 'Threshold-based Valley Detection'
                threshold = thresholds.get('valley_threshold', None)
                if threshold:
                    ax1.axhline(y=threshold, color='green', linestyle='--', label=f'Valley Threshold ({threshold:.1f})')
            
            # Plot valley probabilities
            pred_indices = pred_df.index[:horizon]
            ax2.plot(pred_indices, pred_df['valley_probability'].values[:horizon], label='Valley Probability', color='purple', marker='o')
            
            # Draw threshold line for probability
            if model_type == 'valley' and thresholds and thresholds.get('method') == 'derivative':
                ax2.axhline(y=0.7, color='purple', linestyle=':', label='Probability Threshold (0.7)')
            
            # Mark predicted valleys
            valley_indices = pred_df[pred_df['is_predicted_valley'] == 1].index[:horizon]
            if len(valley_indices) > 0:
                ax1.scatter(valley_indices, df.loc[valley_indices, TARGET_VARIABLE], color='purple', s=100, marker='*', label='Predicted Valleys')
            
            title = method_name
        
        ax1.set_xlabel('Datetime')
        ax1.set_ylabel('Price (öre/kWh)')
        ax2.set_ylabel('Probability')
        ax2.set_ylim(0, 1)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title(title)
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(PREDICTIONS_DIR / f"{model_type}_prediction.png")
        
        if show_plot == True:
            plt.show()
    
    return pred_df

def combine_predictions(trend_results, peak_results, valley_results):
    """Combine predictions from multiple models.
    
    Args:
        trend_results: DataFrame with trend predictions
        peak_results: DataFrame with peak predictions
        valley_results: DataFrame with valley predictions
        
    Returns:
        DataFrame with combined predictions
    """
    # Start with trend predictions
    combined_df = trend_results.copy()
    
    # Add peak predictions if available
    if peak_results is not None:
        # Join on index (timestamp)
        combined_df = combined_df.join(peak_results, how='left')
        
        # If a peak is predicted, we might want to adjust the trend prediction
        # Here we could add some custom logic or apply a different model
        if 'is_predicted_peak' in combined_df.columns:
            peak_indices = combined_df[combined_df['is_predicted_peak'] == 1].index
            logging.info(f"Detected {len(peak_indices)} potential price peaks")
    
    # Add valley predictions if available
    if valley_results is not None:
        # Join on index (timestamp)
        combined_df = combined_df.join(valley_results, how='left')
        
        # If a valley is predicted, we might want to adjust the trend prediction
        # Here we could add some custom logic or apply a different model
        if 'is_predicted_valley' in combined_df.columns:
            valley_indices = combined_df[combined_df['is_predicted_valley'] == 1].index
            logging.info(f"Detected {len(valley_indices)} potential price valleys")
    
    # Create a combined prediction column
    combined_df['price_prediction'] = combined_df['predicted_price'].copy()
    
    # Apply rules to adjust predictions based on peak/valley detection
    if 'is_predicted_peak' in combined_df.columns and 'peak_probability' in combined_df.columns:
        # For peaks, we might increase the prediction by some factor based on probability
        peak_adjustment = lambda row: row['predicted_price'] * (1 + 0.5 * row['peak_probability']) if row['is_predicted_peak'] == 1 else row['predicted_price']
        combined_df['price_prediction'] = combined_df.apply(peak_adjustment, axis=1)
    
    if 'is_predicted_valley' in combined_df.columns and 'valley_probability' in combined_df.columns:
        # For valleys, we might decrease the prediction by some factor based on probability
        valley_adjustment = lambda row: row['price_prediction'] * (1 - 0.3 * row['valley_probability']) if row['is_predicted_valley'] == 1 else row['price_prediction']
        combined_df['price_prediction'] = combined_df.apply(valley_adjustment, axis=1)
    
    # Add a confidence score column (simplified version)
    # Higher confidence for "normal" prices, lower for peaks/valleys
    combined_df['confidence_score'] = 0.85  # Default confidence
    
    if 'peak_probability' in combined_df.columns:
        combined_df.loc[combined_df['peak_probability'] > 0.5, 'confidence_score'] = 0.7
    
    if 'valley_probability' in combined_df.columns:
        combined_df.loc[combined_df['valley_probability'] > 0.5, 'confidence_score'] = 0.75
    
    # Calculate prediction intervals (simple approach)
    confidence_width = 1.96  # ~95% confidence interval
    combined_df['prediction_lower'] = combined_df['price_prediction'] - confidence_width * (1 - combined_df['confidence_score']) * combined_df['price_prediction'].abs()
    combined_df['prediction_upper'] = combined_df['price_prediction'] + confidence_width * (1 - combined_df['confidence_score']) * combined_df['price_prediction'].abs()
    
    # Ensure no negative prices if that's not expected
    combined_df['prediction_lower'] = combined_df['prediction_lower'].clip(lower=0)
    
    return combined_df

def predict_prices(model_type='all', horizon=1, show_plot=False, production_mode=False):
    """Make predictions using the specified model(s).
    
    Args:
        model_type: Type of model to use ('trend', 'peak', 'valley', 'all')
        horizon: Prediction horizon in days (default: 1 day)
        show_plot: Whether to show a plot of predictions
        production_mode: Whether to run in production mode
        
    Returns:
        DataFrame with predictions
    """
    # Convert horizon from days to hours
    try:
        horizon_days = int(horizon)
        horizon_hours = horizon_days * 24
        logging.info(f"Using {horizon_days} day horizon ({horizon_hours} hours)")
    except (ValueError, TypeError):
        logging.warning(f"Invalid horizon value: {horizon}, using default 1 day (24 hours)")
        horizon_days = 1
        horizon_hours = 24
    
    # Load data
    logging.info("Loading data for prediction...")
    try:
        df = load_and_merge_data()
        logging.info(f"Loaded data with shape {df.shape}")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise
    
    # Make predictions based on model type
    if model_type == 'trend' or model_type == 'all':
        logging.info("Making trend model predictions...")
        try:
            # Load trend model
            trend_artifacts = load_trend_model()
            
            # Make predictions
            trend_results = predict_with_trend_model(
                trend_artifacts,
                df,
                horizon=horizon_hours,
                show_plot=show_plot
            )
            
            logging.info(f"Trend predictions shape: {trend_results.shape}")
            
            if model_type == 'trend':
                predictions = trend_results.copy()
                predictions.columns = ['predicted_price']
        except Exception as e:
            logging.error(f"Error in trend prediction: {e}")
            if model_type == 'trend':
                raise
            else:
                logging.warning("Continuing with other models...")
                trend_results = None
    else:
        trend_results = None
    
    if model_type == 'peak' or model_type == 'all':
        logging.info("Making peak model predictions...")
        try:
            # Load peak model
            peak_artifacts = load_peak_valley_model('peak')
            
            # Make predictions
            peak_results = predict_with_peak_valley_model(
                peak_artifacts,
                df,
                'peak',
                horizon=horizon_hours,
                show_plot=show_plot
            )
            
            logging.info(f"Peak predictions shape: {peak_results.shape}")
            
            if model_type == 'peak':
                predictions = peak_results.copy()
        except Exception as e:
            logging.error(f"Error in peak prediction: {e}")
            if model_type == 'peak':
                raise
            else:
                logging.warning("Continuing with other models...")
                peak_results = None
    else:
        peak_results = None
    
    if model_type == 'valley' or model_type == 'all':
        logging.info("Making valley model predictions...")
        try:
            # Load valley model
            valley_artifacts = load_peak_valley_model('valley')
            
            # Make predictions
            valley_results = predict_with_peak_valley_model(
                valley_artifacts,
                df,
                'valley',
                horizon=horizon_hours,
                show_plot=show_plot
            )
            
            logging.info(f"Valley predictions shape: {valley_results.shape}")
            
            if model_type == 'valley':
                predictions = valley_results.copy()
        except Exception as e:
            logging.error(f"Error in valley prediction: {e}")
            if model_type == 'valley':
                raise
            else:
                logging.warning("Continuing with other models...")
                valley_results = None
    else:
        valley_results = None
    
    # Combine all predictions if requested
    if model_type == 'all':
        logging.info("Combining all predictions...")
        predictions = combine_predictions(trend_results, peak_results, valley_results)
    
    return predictions

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict electricity prices using trained models')
    
    parser.add_argument('--model', type=str, choices=['trend', 'peak', 'valley', 'all'],
                      default='all', help='Model to use for prediction (default: all)')
    
    parser.add_argument('--horizon', type=int, default=1,
                      help='Prediction horizon in days (default: 1 day)')
    
    parser.add_argument('--show_plot', action='store_true',
                      help='Show plot of predictions')
    
    parser.add_argument('--production_mode', action='store_true',
                      help='Run in production mode')
    
    return parser.parse_args()

def main():
    """Main function to run predictions."""
    args = parse_arguments()
    
    try:
        # Try to run predictions
        predictions = predict_prices(
            model_type=args.model,
            horizon=args.horizon,
            show_plot=args.show_plot,
            production_mode=args.production_mode
        )
        
        # Print prediction summary
        print("\nPrediction Summary:")
        print(f"Model: {args.model}")
        print(f"Horizon: {args.horizon} days ({args.horizon * 24} hours)")
        print(f"Period: {predictions.index.min()} to {predictions.index.max()}")
        
        # Print statistics
        if 'predicted_price' in predictions.columns:
            print(f"Predicted price stats - Min: {predictions['predicted_price'].min():.2f}, Max: {predictions['predicted_price'].max():.2f}, Mean: {predictions['predicted_price'].mean():.2f}")
        
        print(f"Production mode: {args.production_mode}")
        
        return True
    
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 