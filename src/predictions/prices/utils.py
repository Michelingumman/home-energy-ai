"""
Utility functions for the improved TCN model for electricity price forecasting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import logging
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import os
from datetime import datetime, timedelta
import sys
from scipy.signal import find_peaks

# Import config
from config import (
    TARGET_VARIABLE, PRICE_LAG_HOURS, ROLLING_WINDOWS,
    SCALING_METHOD, LOG_TRANSFORM_PARAMS, CUSTOM_SCALING_BOUNDS,
    LOOKBACK_WINDOW, PREDICTION_HORIZON, LOSS_FUNCTION, WEIGHTED_LOSS_PARAMS
)

# ----- DATA PREPARATION FUNCTIONS -----

def add_time_features(df):
    """Add time-based features to the dataframe."""
    # Create basic time features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['week_of_year'] = df.index.isocalendar().week
    
    # Create cyclical encoding of time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    # Indicator variables for special times
    df['is_business_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 18) & (~df['is_weekend'])).astype(int)
    df['is_peak_hour'] = ((df['hour'] >= 17) & (df['hour'] <= 20) & (df['day_of_week'] < 5)).astype(int)
    df['is_morning_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 9) & (df['day_of_week'] < 5)).astype(int)
    df['is_evening_peak'] = ((df['hour'] >= 17) & (df['hour'] <= 20) & (df['day_of_week'] < 5)).astype(int)
    
    # Add season information
    month = df.index.month
    df['season'] = np.where(month.isin([12, 1, 2]), 0,  # Winter
                  np.where(month.isin([3, 4, 5]), 1,    # Spring
                  np.where(month.isin([6, 7, 8]), 2,    # Summer
                           3)))                         # Fall
    
    return df

def add_lag_features(df, target_col, lag_hours=None):
    """Add lagged features for the target variable."""
    if lag_hours is None:
        lag_hours = PRICE_LAG_HOURS
        
    for lag in lag_hours:
        df[f'{target_col}_lag_{lag}h'] = df[target_col].shift(lag)
    
    return df

def add_rolling_features(df, target_col, windows=None):
    """Add rolling window features for the target variable."""
    if windows is None:
        windows = ROLLING_WINDOWS
        
    for window_config in windows:
        window = window_config['window']
        features = window_config['features']
        
        rolling_window = df[target_col].rolling(window=window, min_periods=1)
        
        if 'mean' in features:
            df[f'{target_col}_rolling_{window}h_mean'] = rolling_window.mean()
        
        if 'std' in features:
            df[f'{target_col}_rolling_{window}h_std'] = rolling_window.std().fillna(0)
        
        if 'min' in features:
            df[f'{target_col}_rolling_{window}h_min'] = rolling_window.min()
        
        if 'max' in features:
            df[f'{target_col}_rolling_{window}h_max'] = rolling_window.max()
    
    return df

def add_price_spike_indicators(df, target_col, percentile_threshold=95):
    """Add indicators for price spikes based on historical patterns."""
    # Get the threshold value based on the specified percentile
    spike_threshold = df[target_col].quantile(percentile_threshold / 100)
    
    # Create the spike indicator
    df[f'{target_col}_spike'] = (df[target_col] > spike_threshold).astype(int)
    
    # Add a column for negative prices
    df[f'{target_col}_negative'] = (df[target_col] < 0).astype(int)
    
    return df

def detect_price_peaks(df, target_col=TARGET_VARIABLE, window=24, relative_height=0.2, distance=6, prominence=0.15):
    """
    A more sophisticated approach to detect price peaks based on relative movements.
    
    Args:
        df: DataFrame with price data
        target_col: Target column name containing price data
        window: Window size for local maxima detection (hours)
        relative_height: Minimum relative height compared to median price in window
        distance: Minimum distance between detected peaks (hours)
        prominence: Minimum prominence required for a peak (0-1 range, relative to price range)
        
    Returns:
        DataFrame with added 'is_price_peak_relative' and 'is_price_valley_relative' columns
    """
    # Make a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Add rolling statistics with centered window for better detection (using proper methods)
    result_df['price_rolling_median'] = result_df[target_col].rolling(window=window, center=True).median().ffill().bfill()
    result_df['price_rolling_std'] = result_df[target_col].rolling(window=window, center=True).std().ffill().bfill()
    result_df['price_rolling_max'] = result_df[target_col].rolling(window=window, center=True).max().ffill().bfill()
    result_df['price_rolling_min'] = result_df[target_col].rolling(window=window, center=True).min().ffill().bfill()
    
    # Calculate daily pattern
    result_df['hour'] = result_df.index.hour
    hourly_avg = result_df.groupby('hour')[target_col].mean()
    result_df['hourly_pattern'] = result_df['hour'].map(hourly_avg)
    
    # Detrend using hourly pattern for better peak detection
    result_df['detrended'] = result_df[target_col] - result_df['hourly_pattern'] + result_df[target_col].mean()
    
    # Initialize peak indicators
    result_df['is_price_peak_relative'] = 0
    result_df['is_price_valley_relative'] = 0
    
    # Get price data as numpy array for scipy peak detection
    prices = result_df[target_col].values
    detrended_prices = result_df['detrended'].values
    
    # Calculate price range for prominence scaling
    price_range = np.max(prices) - np.min(prices)
    min_prominence = prominence * price_range
    
    # Find peaks using signal processing on both raw and detrended data
    peak_indices, _ = find_peaks(
        prices, 
        distance=distance,
        prominence=min_prominence
    )
    
    detrended_peak_indices, _ = find_peaks(
        detrended_prices, 
        distance=distance,
        prominence=min_prominence
    )
    
    # Combine both peak detection methods
    all_peak_indices = np.unique(np.concatenate([peak_indices, detrended_peak_indices]))
    
    # Find valleys (invert the signal to find valleys as peaks)
    valley_indices, _ = find_peaks(
        -prices,
        distance=distance,
        prominence=min_prominence
    )
    
    detrended_valley_indices, _ = find_peaks(
        -detrended_prices,
        distance=distance,
        prominence=min_prominence
    )
    
    # Combine both valley detection methods
    all_valley_indices = np.unique(np.concatenate([valley_indices, detrended_valley_indices]))
    
    # Set peak indicators
    result_df.iloc[all_peak_indices, result_df.columns.get_loc('is_price_peak_relative')] = 1
    result_df.iloc[all_valley_indices, result_df.columns.get_loc('is_price_valley_relative')] = 1
    
    # Apply relative threshold filters (peaks must be X% above local median)
    result_df['is_price_peak_relative'] = (
        (result_df['is_price_peak_relative'] == 1) & 
        (result_df[target_col] > result_df['price_rolling_median'] * (1 + relative_height))
    ).astype(int)
    
    # For valleys, they must be X% below local median
    result_df['is_price_valley_relative'] = (
        (result_df['is_price_valley_relative'] == 1) & 
        (result_df[target_col] < result_df['price_rolling_median'] * (1 - relative_height))
    ).astype(int)
    
    # Apply time-based refinements - limit to 2 peaks per day maximum
    days = result_df.index.date
    for day in np.unique(days):
        day_mask = (result_df.index.date == day) & (result_df['is_price_peak_relative'] == 1)
        day_peaks = result_df[day_mask]
        
        if len(day_peaks) > 2:
            # Keep only the top 2 peaks for this day
            keep_idx = day_peaks[target_col].nlargest(2).index
            drop_idx = [idx for idx in day_peaks.index if idx not in keep_idx]
            result_df.loc[drop_idx, 'is_price_peak_relative'] = 0
    
    # Record statistics
    num_peaks = result_df['is_price_peak_relative'].sum()
    num_valleys = result_df['is_price_valley_relative'].sum()
    total_points = len(result_df)
    
    logging.info(f"Detected {num_peaks} peaks ({num_peaks/total_points:.1%} of data) and {num_valleys} valleys ({num_valleys/total_points:.1%} of data)")
    logging.info(f"Using window={window}h, relative_height={relative_height:.2f}, distance={distance}h, prominence={prominence:.2f}")
    
    # Cleanup temporary columns
    result_df = result_df.drop(['price_rolling_median', 'price_rolling_std', 'price_rolling_max', 
                               'price_rolling_min', 'hour', 'hourly_pattern', 'detrended'], axis=1)
    
    return result_df

def detect_price_valleys_derivative(df, target_col=TARGET_VARIABLE, window=24, slope_threshold=0.15, 
                                   curvature_threshold=0.1, distance=8, smoothing_window=3,
                                   detect_daily_valleys=True, daily_lookback=6, daily_lookahead=6,
                                   detect_relative_valleys=True, relative_depth_threshold=0.15):
    """
    A derivative-based approach to detect price valleys based on price movement patterns.
    This method focuses on identifying significant price drops followed by increases.
    
    Args:
        df: DataFrame with price data
        target_col: Target column name containing price data
        window: Window size for context (in hours)
        slope_threshold: Minimum relative slope needed to consider a potential valley
        curvature_threshold: Minimum curvature (second derivative) to detect direction changes
        distance: Minimum distance between detected valleys (hours)
        smoothing_window: Window size for smoothing derivatives
        detect_daily_valleys: Whether to also detect daily valleys (lowest point in each day)
        daily_lookback: Hours to lookback when finding daily valleys
        daily_lookahead: Hours to look ahead when finding daily valleys
        detect_relative_valleys: Whether to detect relative valleys (local minima between peaks)
        relative_depth_threshold: Minimum relative depth required to identify a relative valley
        
    Returns:
        DataFrame with added 'is_price_valley_derivative' column
    """
    # Make a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Calculate price statistics for context
    result_df['price_rolling_median'] = result_df[target_col].rolling(window=window, center=True).median().fillna(method='bfill').fillna(method='ffill')
    result_df['price_rolling_max'] = result_df[target_col].rolling(window=window, center=True).max().fillna(method='bfill').fillna(method='ffill')
    result_df['price_rolling_min'] = result_df[target_col].rolling(window=window, center=True).min().fillna(method='bfill').fillna(method='ffill')
    
    # Calculate the price range for scaling
    result_df['price_range'] = result_df['price_rolling_max'] - result_df['price_rolling_min']
    
    # Calculate first derivative (rate of change)
    result_df['price_diff'] = result_df[target_col].diff()
    
    # Smooth the first derivative to reduce noise
    result_df['smooth_diff'] = result_df['price_diff'].rolling(window=smoothing_window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    
    # Calculate the second derivative (acceleration/curvature)
    result_df['price_diff2'] = result_df['smooth_diff'].diff()
    result_df['smooth_diff2'] = result_df['price_diff2'].rolling(window=smoothing_window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    
    # Calculate relative derivatives (scaled by price range)
    result_df['rel_diff'] = result_df['smooth_diff'] / result_df['price_range']
    result_df['rel_diff2'] = result_df['smooth_diff2'] / result_df['price_range']
    
    # Identify potential valleys based on second derivative (inflection points)
    # A valley occurs when:
    # 1. Price was decreasing (negative first derivative)
    # 2. Then starts increasing (positive first derivative)
    # 3. Which means second derivative goes from negative to positive
    result_df['inflection_up'] = (result_df['smooth_diff2'] > 0) & (result_df['smooth_diff2'].shift(1) <= 0)
    
    # Filter to significant valleys - only consider points where:
    # 1. The price is in the bottom half of its local range
    # 2. The slope change is significant (valley is steep enough)
    result_df['rel_price'] = (result_df[target_col] - result_df['price_rolling_min']) / result_df['price_range']
    
    # Mark potential valleys where:
    # - It's an inflection point from down to up
    # - The price is in the lower part of the range
    # - The slope changes are significant enough
    
    # Relax slope and curvature thresholds even more
    result_df['is_valley_candidate'] = (
        result_df['inflection_up'] & 
        (result_df['rel_price'] < 0.7) &  # Relaxed from 0.5 to catch higher valleys
        (result_df['rel_diff'].abs() > slope_threshold * 0.7) &  # Further reduced threshold by 30%
        (result_df['rel_diff2'].abs() > curvature_threshold * 0.7)  # Further reduced threshold by 30%
    )
    
    # Find the actual valley points as local minima in a small window around candidates
    result_df['is_price_valley_derivative'] = 0
    window_size = 3  # Look 3 hours before and after
    
    # For each candidate, check if it's a local minimum
    for idx in result_df[result_df['is_valley_candidate']].index:
        try:
            # Get the window around this point
            start_idx = max(0, result_df.index.get_loc(idx) - window_size)
            end_idx = min(len(result_df), result_df.index.get_loc(idx) + window_size + 1)
            window_indices = result_df.index[start_idx:end_idx]
            
            # Check if this is the minimum in the window
            if result_df.loc[idx, target_col] == result_df.loc[window_indices, target_col].min():
                result_df.loc[idx, 'is_price_valley_derivative'] = 1
        except Exception as e:
            logging.warning(f"Error processing valley candidate at {idx}: {e}")
    
    # ADDITIONAL FEATURE: Detect daily valleys (lowest price in each day)
    if detect_daily_valleys:
        # Add date column for grouping by day
        result_df['date'] = result_df.index.date
        
        # For each day, find the lowest price point
        daily_valleys = []
        
        for day, group in result_df.groupby('date'):
            if len(group) < 12:  # Skip partial days
                continue
            
            # Find the hour with minimum price in this day
            min_price_idx = group[target_col].idxmin()
            
            # Only add if not already detected by derivative method
            if result_df.loc[min_price_idx, 'is_price_valley_derivative'] == 0:
                # Make sure it's a true local minimum by checking surrounding hours
                try:
                    # Get the integer position in the original dataframe
                    idx_pos = result_df.index.get_loc(min_price_idx)
                    if isinstance(idx_pos, slice):  # Handle slice return from get_loc
                        # Convert slice to a single integer position (midpoint)
                        idx_pos = idx_pos.start
                        
                    # Now use regular integer positions
                    start_pos = max(0, idx_pos - daily_lookback)
                    end_pos = min(len(result_df), idx_pos + daily_lookahead + 1)
                    
                    # Get surrounding window prices (using iloc for integer-based positions)
                    surrounding_prices = result_df.iloc[start_pos:end_pos][target_col].values
                    local_min_pos = np.argmin(surrounding_prices)
                    
                    # If this point is the minimum in its surrounding window
                    if start_pos + local_min_pos == idx_pos:
                        daily_valleys.append(min_price_idx)
                except Exception as e:
                    logging.warning(f"Error processing daily valley at {min_price_idx}: {e}")
        
        # Mark daily valleys
        for idx in daily_valleys:
            result_df.loc[idx, 'is_price_valley_derivative'] = 1
            
        logging.info(f"Added {len(daily_valleys)} additional daily valleys")
    
    # NEW FEATURE: Detect relative valleys (local drops between peaks)
    if detect_relative_valleys:
        # Find all local minima that aren't already detected
        relative_valleys = []
        
        # Create a small sliding window to find local minima
        small_window = 3  # Hours to look before/after
        
        for i in range(small_window, len(result_df) - small_window):
            idx = result_df.index[i]
            
            # Skip if already marked as a valley
            valley_col_idx = result_df.columns.get_loc('is_price_valley_derivative')
            if result_df.iloc[i, valley_col_idx] == 1:
                continue
                
            # Get price at this point
            current_price = result_df.iloc[i][target_col]
            
            # Get local window
            start_idx = max(0, i - small_window)
            end_idx = min(len(result_df), i + small_window + 1)
            
            # Get prices in window before and after
            before_prices = [result_df.iloc[j][target_col] for j in range(start_idx, i)]
            after_prices = [result_df.iloc[j][target_col] for j in range(i+1, end_idx)]
            
            # Check if it's a local minimum relative to nearby points
            if len(before_prices) > 0 and len(after_prices) > 0:
                min_before = min(before_prices)
                min_after = min(after_prices)
                max_before = max(before_prices)
                max_after = max(after_prices)
                
                # Calculate drop characteristics
                drop_before = max_before - current_price
                rise_after = max_after - current_price
                
                # Various conditions to detect different types of valleys:
                is_valley = False
                
                # 1. Deep valley between peaks
                if (drop_before > current_price * relative_depth_threshold and 
                    rise_after > current_price * relative_depth_threshold):
                    is_valley = True
                # 2. Local minimum (price lower than all surrounding points)
                elif current_price < min_before and current_price < min_after:
                    # Additional check: must be at least slightly lower to be meaningful
                    min_surrounding = min(min_before, min_after)
                    if (min_surrounding - current_price) / current_price > 0.02:  # At least 2% relative difference
                        is_valley = True
                # 3. One-sided valley (sharp drop followed by plateau or gradual rise)
                elif (drop_before > current_price * relative_depth_threshold and 
                      current_price <= min_after and
                      max_after > current_price * 1.05):  # At least 5% higher after
                    is_valley = True
                # 4. Plateau bottom (flat bottom with higher prices on both sides)
                elif (current_price <= min_before * 1.01 and  # Within 1% of min before
                      current_price <= min_after * 1.01 and   # Within 1% of min after
                      max_before > current_price * 1.1 and    # At least 10% higher before
                      max_after > current_price * 1.1):       # At least 10% higher after
                    is_valley = True
                
                if is_valley:
                    relative_valleys.append(idx)
        
        # Mark relative valleys in the dataframe
        if relative_valleys:
            result_df.loc[relative_valleys, 'is_price_valley_derivative'] = 1
            logging.info(f"Added {len(relative_valleys)} relative valleys between peaks")
            
    # Add a second pass for detecting minor valleys that were missed
    # This uses a very simple algorithm - just find local minima in small windows
    minor_valley_window = 2  # Smaller window for finer detection
    minor_valleys = []
    
    # Only run this if we haven't detected too many valleys already (to avoid over-detection)
    valley_pct = result_df['is_price_valley_derivative'].mean() * 100
    if valley_pct < 10.0:  # Increased from 8% to 10% to allow more valleys
        for i in range(minor_valley_window, len(result_df) - minor_valley_window):
            # Skip if already marked as a valley
            if result_df.iloc[i]['is_price_valley_derivative'] == 1:
                continue
                
            # Simple local minimum check
            window_prices = [result_df.iloc[i-minor_valley_window:i][target_col].min(),
                           result_df.iloc[i][target_col],
                           result_df.iloc[i+1:i+minor_valley_window+1][target_col].min()]
            
            # If this point is the minimum in its immediate vicinity
            if window_prices[1] <= window_prices[0] and window_prices[1] <= window_prices[2]:
                # Calculate price range in this local area
                local_min = min(window_prices)
                local_max = max(
                    result_df.iloc[max(0, i-6):min(i+7, len(result_df))][target_col].max(),
                    result_df.iloc[max(0, i-6):i][target_col].max(), 
                    result_df.iloc[i+1:min(i+7, len(result_df))][target_col].max()
                )
                price_range = local_max - local_min
                
                # Check for significant drop and rise, but with more relaxed criteria
                left_max = result_df.iloc[max(0, i-8):i][target_col].max()
                right_max = result_df.iloc[i+1:min(i+9, len(result_df))][target_col].max()
                current_price = result_df.iloc[i][target_col]
                
                # More relaxed conditions for minor valleys
                is_minor_valley = False
                
                # Condition 1: Any price dip between higher values
                if (left_max > current_price * 1.03 or right_max > current_price * 1.03):  # Only 3% difference required
                    is_minor_valley = True
                    
                # Condition 2: Local minimum within larger context
                elif (current_price <= window_prices[0] * 0.99 and current_price <= window_prices[2] * 0.99):  # Just 1% lower
                    is_minor_valley = True
                    
                # Condition 3: Relative valley in a flat area (prices don't change much)
                elif price_range > 0 and (local_max - current_price) / price_range > 0.2:  # Only 20% of range
                    is_minor_valley = True
                    
                if is_minor_valley:
                    minor_valleys.append(result_df.index[i])
                    
        # Add minor valleys to the detection
        if minor_valleys:
            result_df.loc[minor_valleys, 'is_price_valley_derivative'] = 1
            logging.info(f"Added {len(minor_valleys)} minor local valley points")

    # Apply minimum distance between valleys
    # Get indices of detected valleys
    valley_mask = result_df['is_price_valley_derivative'] == 1
    valley_indices = result_df.index[valley_mask]
    
    # ADDITIONAL APPROACH: Use the simple local minima detection that produces the blue dots
    # This is the same algorithm used in the plot_valley_labels function
    simple_window = 6  # Use 6-hour window for simple local minimum detection
    simple_valleys = []
    
    # Get current valley percentage
    valley_pct = result_df['is_price_valley_derivative'].mean() * 100
    if valley_pct < 15.0:  # Cap at 15% to avoid excessive valleys
        prices = result_df[target_col].values
        # Find points that are the minimum in their local window
        for j in range(simple_window, len(prices) - simple_window):
            # Skip if already marked as a valley
            # Fix: Use iloc instead of loc to avoid ambiguity with Series
            if result_df.iloc[j]['is_price_valley_derivative'] == 1:
                continue
                
            # Check if this point is the minimum in its window
            if prices[j] == min(prices[j-simple_window:j+simple_window+1]):
                # Additional check to avoid flat areas - must be at least slightly lower than neighbors
                if (prices[j] < prices[j-1] * 0.996) and (prices[j] < prices[j+1] * 0.996):
                    simple_valleys.append(result_df.index[j])
        
        # Add these simple valleys to our detection
        if simple_valleys:
            result_df.loc[simple_valleys, 'is_price_valley_derivative'] = 1
            logging.info(f"Added {len(simple_valleys)} simple local minima as valleys")
        
    # Now reapply distance filtering with all valleys
    valley_mask = result_df['is_price_valley_derivative'] == 1
    valley_indices = result_df.index[valley_mask]
    
    if len(valley_indices) > 1:
        # Convert to positions in the DataFrame
        # Use enumerate to get integer positions directly
        valley_positions = []
        for i, (idx, row) in enumerate(result_df.iterrows()):
            if row['is_price_valley_derivative'] == 1:
                valley_positions.append(i)  # i is the integer position
        
        # Sort positions (should already be sorted, but just to be safe)
        valley_positions.sort()
        
        # Keep the first valley, then filter others based on distance
        filtered_positions = [valley_positions[0]]
        
        for pos in valley_positions[1:]:
            if pos - filtered_positions[-1] >= distance:
                filtered_positions.append(pos)
        
        # Reset all valleys and only set the filtered ones
        result_df['is_price_valley_derivative'] = 0
        for pos in filtered_positions:
            result_df.iloc[pos, result_df.columns.get_loc('is_price_valley_derivative')] = 1
    
    # Record statistics
    num_valleys = result_df['is_price_valley_derivative'].sum()
    total_points = len(result_df)
    
    logging.info(f"Detected {num_valleys} derivative-based valleys ({num_valleys/total_points:.1%} of data)")
    logging.info(f"Using derivative method with slope_threshold={slope_threshold:.2f}, curvature_threshold={curvature_threshold:.2f}, distance={distance}")
    
    # Cleanup temporary columns
    columns_to_drop = [
        'price_rolling_median', 'price_rolling_max', 'price_rolling_min', 
        'price_range', 'price_diff', 'smooth_diff', 'price_diff2', 
        'smooth_diff2', 'rel_diff', 'rel_diff2', 'inflection_up', 
        'rel_price', 'is_valley_candidate', 'date'
    ]
    result_df = result_df.drop(columns_to_drop, axis=1, errors='ignore')
    
    return result_df

def detect_valleys_robust(df, target_col=TARGET_VARIABLE, min_prominence=0.05, min_width=2, distance=4, 
                         depth_percentile=10, smoothing_window=3):
    """
    A more robust valley detection system that captures all prominent valleys.
    
    Args:
        df: DataFrame with price data
        target_col: Column name for the target price variable
        min_prominence: Minimum relative prominence (as fraction of price range)
        min_width: Minimum width of valley in hours
        distance: Minimum distance between valleys
        depth_percentile: Percentile threshold for valley depth comparison
        smoothing_window: Window size for optional smoothing
        
    Returns:
        DataFrame with valley detection columns added
    """
    import numpy as np
    import pandas as pd
    from scipy.signal import find_peaks, peak_prominences
    
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    prices = result_df[target_col].values
    
    # Optional: Apply light smoothing to reduce noise
    if smoothing_window > 1:
        smoothed_prices = pd.Series(prices).rolling(window=smoothing_window, center=True).mean()
        # Fix deprecation warning: Use bfill/ffill directly instead of fillna(method=...)
        smoothed_prices = smoothed_prices.bfill().ffill().values
    else:
        smoothed_prices = prices
        
    # Invert prices to find valleys instead of peaks
    inverted_prices = -smoothed_prices
    
    # Calculate price range for relative thresholds
    price_range = np.max(prices) - np.min(prices)
    prominence_threshold = min_prominence * price_range
    
    # Method 1: Find valleys using scipy's find_peaks with prominence
    valley_indices, properties = find_peaks(
        inverted_prices,
        prominence=prominence_threshold,
        width=min_width,
        distance=distance
    )
    
    # Get prominences for ranking
    prominences = properties['prominences']
    
    # Method 2: Find local minima using a rolling window approach
    local_min_indices = []
    local_min_values = []
    
    # Window size for local minimum detection (adaptive based on data frequency)
    local_window = max(6, min(12, len(prices) // 100))
    
    for i in range(local_window, len(prices) - local_window):
        if prices[i] == min(prices[i-local_window:i+local_window+1]):
            local_min_indices.append(i)
            local_min_values.append(prices[i])
    
    # Method 3: Derivative-based detection (looking for slope changes)
    # First derivative (price change)
    price_diff = np.diff(smoothed_prices)
    # Add a zero at the beginning to maintain length
    price_diff = np.insert(price_diff, 0, 0)
    
    # Second derivative (change in price change)
    price_diff2 = np.diff(price_diff)
    price_diff2 = np.insert(price_diff2, 0, 0)
    
    # Valley if: prior slope negative, subsequent slope positive, and second derivative positive
    derivative_valleys = []
    for i in range(local_window, len(prices) - local_window):
        # Look back and forward a few steps to be more robust
        back_slope = np.mean(price_diff[i-3:i])
        forward_slope = np.mean(price_diff[i:i+3])
        
        if (back_slope < 0 and forward_slope > 0 and price_diff2[i] > 0):
            derivative_valleys.append(i)
    
    # Combine all detected valleys (from all methods)
    all_valley_indices = set(valley_indices) | set(local_min_indices) | set(derivative_valleys)
    all_valley_indices = sorted(list(all_valley_indices))
    
    # Calculate depth scores for ranking
    depth_scores = []
    for idx in all_valley_indices:
        # Calculate how deep this valley is relative to surrounding prices
        # Look 12 hours before and after
        look_window = 12
        
        # Handle edge cases
        start_idx = max(0, idx - look_window)
        end_idx = min(len(prices) - 1, idx + look_window)
        
        # Get surrounding prices excluding the valley itself
        surrounding = list(prices[start_idx:idx]) + list(prices[idx+1:end_idx+1])
        
        # Calculate mean of surrounding prices
        if surrounding:
            mean_surrounding = np.mean(surrounding)
            # Depth is difference between valley and mean of surroundings
            depth = mean_surrounding - prices[idx]
            
            # Normalize by price range
            relative_depth = depth / price_range
            depth_scores.append(relative_depth)
        else:
            depth_scores.append(0)
    
    # Filter valleys by depth score
    depth_threshold = np.percentile(depth_scores, depth_percentile) if depth_scores else 0
    
    # Initialize valley flags
    result_df['is_price_valley_robust'] = 0
    
    # Mark valleys that meet all criteria
    for idx, score in zip(all_valley_indices, depth_scores):
        if score >= depth_threshold:
            if idx < len(result_df):
                result_df.iloc[idx, result_df.columns.get_loc('is_price_valley_robust')] = 1
    
    # Add depth scores for analysis and tuning
    result_df['valley_depth_score'] = 0.0  # Initialize as float
    for idx, score in zip(all_valley_indices, depth_scores):
        if idx < len(result_df):
            result_df.iloc[idx, result_df.columns.get_loc('valley_depth_score')] = float(score)
    
    # Log statistics
    num_valleys = result_df['is_price_valley_robust'].sum()
    logging.info(f"Robust valley detection found {num_valleys} valleys ({num_valleys/len(result_df):.1%} of data)")
    
    return result_df

def detect_peaks_robust(df, target_col=TARGET_VARIABLE, min_prominence=0.05, min_width=2, distance=4, 
                        height_percentile=90, smoothing_window=3):
    """
    A robust peak detection system that captures all prominent price peaks.
    
    Args:
        df: DataFrame with price data
        target_col: Column name for the target price variable
        min_prominence: Minimum relative prominence (as fraction of price range)
        min_width: Minimum width of peak in hours
        distance: Minimum distance between peaks
        height_percentile: Percentile threshold for peak height comparison
        smoothing_window: Window size for optional smoothing
        
    Returns:
        DataFrame with peak detection columns added
    """
    import numpy as np
    import pandas as pd
    from scipy.signal import find_peaks, peak_prominences
    
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    prices = result_df[target_col].values
    
    # Optional: Apply light smoothing to reduce noise
    if smoothing_window > 1:
        smoothed_prices = pd.Series(prices).rolling(window=smoothing_window, center=True).mean()
        smoothed_prices = smoothed_prices.bfill().ffill().values
    else:
        smoothed_prices = prices
    
    # Calculate price range for relative thresholds
    price_range = np.max(prices) - np.min(prices)
    prominence_threshold = min_prominence * price_range
    
    # Method 1: Find peaks using scipy's find_peaks with prominence
    peak_indices, properties = find_peaks(
        smoothed_prices,  # No inversion needed for peaks
        prominence=prominence_threshold,
        width=min_width,
        distance=distance
    )
    
    # Get prominences for ranking
    prominences = properties['prominences']
    
    # Method 2: Find local maxima using a rolling window approach
    local_max_indices = []
    local_max_values = []
    
    # Window size for local maximum detection (adaptive based on data frequency)
    local_window = max(6, min(12, len(prices) // 100))
    
    for i in range(local_window, len(prices) - local_window):
        if prices[i] == max(prices[i-local_window:i+local_window+1]):
            local_max_indices.append(i)
            local_max_values.append(prices[i])
    
    # Method 3: Derivative-based detection (looking for slope changes)
    # First derivative (price change)
    price_diff = np.diff(smoothed_prices)
    # Add a zero at the beginning to maintain length
    price_diff = np.insert(price_diff, 0, 0)
    
    # Second derivative (change in price change)
    price_diff2 = np.diff(price_diff)
    price_diff2 = np.insert(price_diff2, 0, 0)
    
    # Peak if: prior slope positive, subsequent slope negative, and second derivative negative
    derivative_peaks = []
    for i in range(local_window, len(prices) - local_window):
        # Look back and forward a few steps to be more robust
        back_slope = np.mean(price_diff[i-3:i])
        forward_slope = np.mean(price_diff[i:i+3])
        
        if (back_slope > 0 and forward_slope < 0 and price_diff2[i] < 0):
            derivative_peaks.append(i)
    
    # Combine all detected peaks (from all methods)
    all_peak_indices = set(peak_indices) | set(local_max_indices) | set(derivative_peaks)
    all_peak_indices = sorted(list(all_peak_indices))
    
    # Calculate height scores for ranking
    height_scores = []
    for idx in all_peak_indices:
        # Calculate how high this peak is relative to surrounding prices
        # Look 12 hours before and after
        look_window = 12
        
        # Handle edge cases
        start_idx = max(0, idx - look_window)
        end_idx = min(len(prices) - 1, idx + look_window)
        
        # Get surrounding prices excluding the peak itself
        surrounding = list(prices[start_idx:idx]) + list(prices[idx+1:end_idx+1])
        
        # Calculate mean of surrounding prices
        if surrounding:
            mean_surrounding = np.mean(surrounding)
            # Height is difference between peak and mean of surroundings
            height = prices[idx] - mean_surrounding
            
            # Normalize by price range
            relative_height = height / price_range
            height_scores.append(relative_height)
        else:
            height_scores.append(0)
    
    # Filter peaks by height score
    height_threshold = np.percentile(height_scores, 100 - height_percentile) if height_scores else 0
    
    # Initialize peak flags
    result_df['is_price_peak_robust'] = 0
    
    # Mark peaks that meet all criteria
    for idx, score in zip(all_peak_indices, height_scores):
        if score >= height_threshold:
            if idx < len(result_df):
                result_df.iloc[idx, result_df.columns.get_loc('is_price_peak_robust')] = 1
    
    # Add height scores for analysis and tuning
    result_df['peak_height_score'] = 0.0  # Initialize as float
    for idx, score in zip(all_peak_indices, height_scores):
        if idx < len(result_df):
            result_df.iloc[idx, result_df.columns.get_loc('peak_height_score')] = float(score)
    
    # Log statistics
    num_peaks = result_df['is_price_peak_robust'].sum()
    logging.info(f"Robust peak detection found {num_peaks} peaks ({num_peaks/len(result_df):.1%} of data)")
    
    return result_df

def load_and_merge_data():
    """
    Load and merge all the data sources for price prediction.
    Data from other sources will be limited to the date range of the price data.
    
    Returns:
        pd.DataFrame: DataFrame with merged price, grid, time features, holiday, and weather data.
    """
    from config import (
        DATA_DIR, SE3_PRICES_FILE, SWEDEN_GRID_FILE, TIME_FEATURES_FILE,
        HOLIDAYS_FILE, WEATHER_DATA_FILE, TARGET_VARIABLE
    )
    
    logging.info("Loading and merging data...")
    
    # Load price data first - this determines our date range
    price_file = DATA_DIR / SE3_PRICES_FILE
    if not price_file.exists():
        raise FileNotFoundError(f"Price data file not found: {price_file}")
    
    # Determine if we need to parse the HourSE column
    try:
        # First try reading with HourSE as date
        price_data = pd.read_csv(price_file)
        if 'HourSE' in price_data.columns:
            price_data['datetime'] = pd.to_datetime(price_data['HourSE'])
            price_data.set_index('datetime', inplace=True)
        logging.info(f"Loaded price data: {price_data.shape}")
    except Exception as e:
        logging.error(f"Error loading price data: {e}")
        raise
    
    # Ensure the index is timezone naive
    if price_data.index.tz is not None:
        price_data.index = price_data.index.tz_localize(None)
    
    # Extract the date range from price data - this is our target date range
    price_start_date = price_data.index.min()
    price_end_date = price_data.index.max()
    logging.info(f"Price data date range: {price_start_date} to {price_end_date}")
    
    # Initialize the merged dataframe with price data
    merged_df = price_data.copy()
    
    # Load and merge grid data if available
    grid_file = DATA_DIR / SWEDEN_GRID_FILE
    if grid_file.exists():
        try:
            grid_data = pd.read_csv(grid_file)
            if 'datetime' in grid_data.columns:
                grid_data['datetime'] = pd.to_datetime(grid_data['datetime'])
                grid_data.set_index('datetime', inplace=True)
                
                # Ensure the index is timezone naive
                if grid_data.index.tz is not None:
                    grid_data.index = grid_data.index.tz_localize(None)
                
                # Trim grid data to price data date range
                grid_data = grid_data[(grid_data.index >= price_start_date) & 
                                       (grid_data.index <= price_end_date)]
                
                logging.info(f"Grid data range after trimming: {grid_data.index.min()} to {grid_data.index.max()}")
                
                # Align grid data with price data's exact timestamps
                grid_data_aligned = grid_data.reindex(merged_df.index, method='ffill')
                
                # Join with aligned grid data
                for col in grid_data.columns:
                    if col in merged_df.columns:
                        logging.warning(f"Column '{col}' exists in both price and grid data. Using grid data version.")
                    merged_df[col] = grid_data_aligned[col]
            else:
                # If no datetime column, use direct merge but still trim to date range
                logging.warning("No datetime column in grid data, attempting direct merge")
                merged_df = pd.merge(merged_df, grid_data, left_index=True, right_index=True, how='left')
                
            logging.info(f"Merged grid data: {grid_data.shape}")
        except Exception as e:
            logging.warning(f"Error merging grid data: {e}")
    else:
        logging.warning(f"Grid data file not found: {grid_file}")
    
    # Load and merge time features if available
    time_file = DATA_DIR / TIME_FEATURES_FILE
    if time_file.exists():
        try:
            # For time_features.csv, the first column might be index
            time_data = pd.read_csv(time_file, index_col=0)
            # Convert index to datetime if it's not already
            if not isinstance(time_data.index, pd.DatetimeIndex):
                time_data.index = pd.to_datetime(time_data.index)
                
                # Ensure the index is timezone naive
                if time_data.index.tz is not None:
                    time_data.index = time_data.index.tz_localize(None)
                
            # Trim time data to price data date range
            time_data = time_data[(time_data.index >= price_start_date) & 
                                  (time_data.index <= price_end_date)]
                
            # Align time data with price data index
            time_data_aligned = time_data.reindex(merged_df.index, method='ffill')
            
            # Add columns directly
            for col in time_data.columns:
                if col in merged_df.columns:
                    logging.warning(f"Column '{col}' exists in both merged and time data. Using time data version.")
                merged_df[col] = time_data_aligned[col]
                
            logging.info(f"Merged time features: {time_data.shape}")
        except Exception as e:
            logging.warning(f"Error merging time features: {e}")
            # Generate time features if file doesn't exist or has errors
            logging.warning("Generating time features instead...")
            merged_df = add_time_features(merged_df)
    else:
        # Generate time features if file doesn't exist
        logging.warning(f"Time features file not found: {time_file}. Generating time features...")
        merged_df = add_time_features(merged_df)
    
    # Load and merge holiday data if available
    holiday_file = DATA_DIR / HOLIDAYS_FILE
    if holiday_file.exists():
        try:
            # For holidays.csv, the first column might be index
            holiday_data = pd.read_csv(holiday_file, index_col=0)
            # Convert index to datetime if it's not already
            if not isinstance(holiday_data.index, pd.DatetimeIndex):
                holiday_data.index = pd.to_datetime(holiday_data.index)
                
                # Ensure the index is timezone naive
                if holiday_data.index.tz is not None:
                    holiday_data.index = holiday_data.index.tz_localize(None)
                
            # Trim holiday data to price data date range
            holiday_data = holiday_data[(holiday_data.index >= price_start_date) & 
                                       (holiday_data.index <= price_end_date)]
                
            # Align holiday data with price data index  
            holiday_data_aligned = holiday_data.reindex(merged_df.index, method='ffill')
            
            # Add columns directly
            for col in holiday_data.columns:
                if col in merged_df.columns:
                    logging.warning(f"Column '{col}' exists in both merged and holiday data. Using holiday data version.")
                merged_df[col] = holiday_data_aligned[col]
                
            logging.info(f"Merged holiday data: {holiday_data.shape}")
        except Exception as e:
            logging.warning(f"Error merging holiday data: {e}")
    else:
        logging.warning(f"Holiday data file not found: {holiday_file}")
    
    # Load and merge weather data if available
    weather_file = DATA_DIR / WEATHER_DATA_FILE
    if weather_file.exists():
        try:
            weather_data = pd.read_csv(weather_file)
            if 'date' in weather_data.columns:
                weather_data['datetime'] = pd.to_datetime(weather_data['date'])
                weather_data.set_index('datetime', inplace=True)
                
                # Ensure the index is timezone naive
                if weather_data.index.tz is not None:
                    weather_data.index = weather_data.index.tz_localize(None)
                
            # Trim weather data to price data date range
            weather_data = weather_data[(weather_data.index >= price_start_date) & 
                                        (weather_data.index <= price_end_date)]
                
            logging.info(f"Weather data range after trimming: {weather_data.index.min()} to {weather_data.index.max()}")
            
            # Align weather data with price data index
            weather_data_aligned = weather_data.reindex(merged_df.index, method='ffill')
            
            # Add columns directly
            for col in weather_data.columns:
                if col not in ['date']: # Skip the date column if it exists
                    if col in merged_df.columns:
                        logging.warning(f"Column '{col}' exists in both merged and weather data. Using weather data version.")
                    merged_df[col] = weather_data_aligned[col]
                
            logging.info(f"Merged weather data: {weather_data.shape}")
        except Exception as e:
            logging.warning(f"Error merging weather data: {e}")
    else:
        logging.warning(f"Weather data file not found: {weather_file}")
    
    # Sort the data by datetime
    merged_df = merged_df.sort_index()
    
    # Check for missing values after the merge - there shouldn't be any for the price data range
    missing_values = merged_df.isnull().sum()
    missing_columns = missing_values[missing_values > 0]
    
    if len(missing_columns) > 0:
        logging.warning(f"Found missing values in data after merging: {missing_columns}")
        
        # Print information about missing data ranges
        for col in missing_columns.index:
            missing_indices = merged_df[merged_df[col].isnull()].index
            if len(missing_indices) > 0:
                logging.warning(f"Missing '{col}' data between {missing_indices.min()} and {missing_indices.max()}")
                
                # For missing data within price range, use forward/backward fill (preserves patterns)
                merged_df[col] = merged_df[col].fillna(method='ffill').fillna(method='bfill')
                
                # Check if we still have missing values
                if merged_df[col].isnull().any():
                    logging.warning(f"Still missing values in {col} after forward/backward fill, using fallback method")
                    
                    # Fallback filling method (only if absolutely necessary)
                    if merged_df[col].dtype.kind in 'iuf':  # integer, unsigned int, or float
                        median_val = merged_df[col].median()
                        merged_df[col] = merged_df[col].fillna(median_val)
                        logging.info(f"Filled remaining missing values in {col} with median")
                    else:
                        # For categorical values, use mode
                        mode_val = merged_df[col].mode()[0] if not merged_df[col].mode().empty else ""
                        merged_df[col] = merged_df[col].fillna(mode_val)
                        logging.info(f"Filled remaining missing values in {col} with mode")
    else:
        logging.info("No missing values in merged dataframe - all data ranges properly aligned")
    
    # Check for required columns and add them if they're missing
    required_features = [
        "temperature_2m", "cloud_cover", "relative_humidity_2m", 
        "wind_speed_100m", "wind_direction_100m", "shortwave_radiation_sum"
    ]
    
    for feature in required_features:
        if feature not in merged_df.columns:
            logging.warning(f"Required feature '{feature}' not found in data. Adding dummy values.")
            # Add a dummy column with median values
            merged_df[feature] = 0
    
    logging.info(f"Final merged dataframe shape: {merged_df.shape}")
    logging.info(f"Date range: {merged_df.index.min()} to {merged_df.index.max()}")
    
    return merged_df

def add_spike_labels(df, target_col=TARGET_VARIABLE, spike_percentile=95, valley_percentile=5):
    """Add binary labels for price spikes and valleys."""
    # Calculate thresholds
    spike_threshold = df[target_col].quantile(spike_percentile / 100)
    valley_threshold = df[target_col].quantile(valley_percentile / 100)
    
    # Add spike and valley indicators
    df['is_price_peak'] = (df[target_col] >= spike_threshold).astype(int)
    df['is_price_valley'] = (df[target_col] <= valley_threshold).astype(int)
    
    # Log the thresholds
    logging.info(f"Price spike threshold (P{spike_percentile}): {spike_threshold:.2f}")
    logging.info(f"Price valley threshold (P{valley_percentile}): {valley_threshold:.2f}")
    
    return df

# ----- CUSTOM SCALERS -----

class LogTransformScaler:
    """Custom scaler that applies a logarithmic transformation to handle wide value ranges."""
    def __init__(self, offset=100, base=10):
        self.offset = offset
        self.base = base
        self.min_value = None
        self.fitted = False
        
    def fit(self, X):
        X_array = np.array(X)
        self.min_value = np.min(X_array)
        if self.min_value <= -self.offset:
            self.offset = abs(self.min_value) + 1
        self.fitted = True
        return self
    
    def transform(self, X):
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet.")
        X_array = np.array(X)
        # Add offset to make all values positive
        X_positive = X_array + self.offset
        # Apply log transform
        if self.base == np.e:
            return np.log(X_positive)
        else:
            return np.log10(X_positive) / np.log10(self.base)
    
    def inverse_transform(self, X):
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet.")
        X_array = np.array(X)
        # Apply inverse log transform
        if self.base == np.e:
            return np.exp(X_array) - self.offset
        else:
            return self.base ** X_array - self.offset
    
    def save(self, path):
        params = {
            'offset': self.offset,
            'base': self.base,
            'min_value': self.min_value,
            'fitted': self.fitted,
            'type': 'LogTransformScaler'
        }
        with open(path, 'w') as f:
            json.dump(params, f)
    
    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            params = json.load(f)
        scaler = cls(offset=params['offset'], base=params['base'])
        scaler.min_value = params['min_value']
        scaler.fitted = params['fitted']
        return scaler

class CustomBoundedScaler:
    """Custom scaler that scales data to a predefined range."""
    def __init__(self, feature_min=-100, feature_max=1000, range_min=-1, range_max=1):
        self.feature_min = feature_min
        self.feature_max = feature_max
        self.range_min = range_min
        self.range_max = range_max
        self.fitted = False
    
    def fit(self, X):
        self.fitted = True
        return self
    
    def transform(self, X):
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet.")
        X_array = np.array(X)
        # Clip values to the specified bounds
        X_clipped = np.clip(X_array, self.feature_min, self.feature_max)
        # Scale to the specified range
        return self.range_min + (X_clipped - self.feature_min) * (self.range_max - self.range_min) / (self.feature_max - self.feature_min)
    
    def inverse_transform(self, X):
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet.")
        X_array = np.array(X)
        # Clip values to the scaling range
        X_clipped = np.clip(X_array, self.range_min, self.range_max)
        # Scale back to the original range
        return self.feature_min + (X_clipped - self.range_min) * (self.feature_max - self.feature_min) / (self.range_max - self.range_min)
    
    def save(self, path):
        params = {
            'feature_min': self.feature_min,
            'feature_max': self.feature_max,
            'range_min': self.range_min,
            'range_max': self.range_max,
            'fitted': self.fitted,
            'type': 'CustomBoundedScaler'
        }
        with open(path, 'w') as f:
            json.dump(params, f)
    
    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            params = json.load(f)
        scaler = cls(
            feature_min=params['feature_min'],
            feature_max=params['feature_max'],
            range_min=params['range_min'],
            range_max=params['range_max']
        )
        scaler.fitted = params['fitted']
        return scaler

def get_scaler(method=SCALING_METHOD):
    """Get the appropriate scaler based on the configuration."""
    if method == 'standard':
        return StandardScaler()
    elif method == 'robust':
        return RobustScaler()
    elif method == 'minmax':
        return MinMaxScaler()
    elif method == 'log_transform':
        return LogTransformScaler(
            offset=LOG_TRANSFORM_PARAMS['offset'],
            base=LOG_TRANSFORM_PARAMS['base']
        )
    elif method == 'custom':
        return CustomBoundedScaler(
            feature_min=CUSTOM_SCALING_BOUNDS['price_min'],
            feature_max=CUSTOM_SCALING_BOUNDS['price_max']
        )
    else:
        raise ValueError(f"Unknown scaling method: {method}")

# ----- SEQUENCE CREATION -----

def create_sequences(data, lookback, horizon, feature_columns=None, target_column=None):
    """Create sequences for time series forecasting."""
    if isinstance(data, pd.DataFrame):
        if feature_columns is None:
            feature_columns = data.columns
        if target_column is None:
            target_column = TARGET_VARIABLE
            
        data_array = data[feature_columns].values
        target_array = data[target_column].values
    else:
        data_array = data
        target_array = data[:, 0]  # Assume first column is the target
    
    X, y = [], []
    
    for i in range(len(data_array) - lookback - horizon + 1):
        # Get lookback sequence
        X.append(data_array[i:i+lookback])
        
        # Get target sequence
        y.append(target_array[i+lookback:i+lookback+horizon])
    
    return np.array(X), np.array(y)

# ----- CUSTOM LOSS FUNCTIONS -----

def spike_weighted_loss(y_true, y_pred):
    """
    Custom loss function that heavily penalizes missed spikes
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Weighted mean squared error
    """
    # Base MSE loss
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    
    # Create weight matrix based on true values
    baseline_weight = WEIGHTED_LOSS_PARAMS["baseline_weight"]
    spike_threshold = WEIGHTED_LOSS_PARAMS["spike_threshold"]
    spike_weight = WEIGHTED_LOSS_PARAMS["spike_weight"] * 2  # Double the spike weight for spike model
    negative_weight = WEIGHTED_LOSS_PARAMS["negative_weight"] * 2  # Double the negative weight
    
    # Weight matrix - start with baseline
    weights = tf.ones_like(y_true) * baseline_weight
    
    # Apply spike weights (extra high for this model)
    weights = tf.where(y_true > spike_threshold, tf.ones_like(y_true) * spike_weight, weights)
    
    # Apply negative price weights
    weights = tf.where(y_true < 0, tf.ones_like(y_true) * negative_weight, weights)
    
    # Apply weights to MSE
    weighted_mse = mse * weights
    
    return tf.reduce_mean(weighted_mse)

def smape_loss(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error loss function.
    More robust to outliers than standard MAE or MSE.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        SMAPE loss value
    """
    epsilon = tf.keras.backend.epsilon()
    
    # Add a small constant to avoid division by zero
    denominator = (tf.abs(y_true) + tf.abs(y_pred) + epsilon) / 2.0
    
    # Calculate SMAPE
    smape = tf.abs(y_pred - y_true) / denominator
    
    return tf.reduce_mean(smape) * 100  # Multiply by 100 to get percentage

def weighted_mae_loss(y_true, y_pred):
    """
    Custom weighted Mean Absolute Error loss function.
    Applies higher weights to price spikes and negative prices.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Weighted MAE loss value
    """
    # Base MAE loss
    mae = tf.abs(y_true - y_pred)
    
    # Get parameters from config
    baseline_weight = WEIGHTED_LOSS_PARAMS["baseline_weight"]
    spike_threshold = WEIGHTED_LOSS_PARAMS["spike_threshold"]
    spike_weight = WEIGHTED_LOSS_PARAMS["spike_weight"]
    negative_weight = WEIGHTED_LOSS_PARAMS["negative_weight"]
    
    # Weight matrix - start with baseline
    weights = tf.ones_like(y_true) * baseline_weight
    
    # Apply spike weights
    weights = tf.where(y_true > spike_threshold, tf.ones_like(y_true) * spike_weight, weights)
    
    # Apply negative price weights
    weights = tf.where(y_true < 0, tf.ones_like(y_true) * negative_weight, weights)
    
    # Apply weights to MAE
    weighted_mae = mae * weights
    
    return tf.reduce_mean(weighted_mae)

def get_loss_function(loss_name):
    """
    Get the appropriate loss function for the given name.
    
    Args:
        loss_name: Name of the loss function
        
    Returns:
        Loss function
    """
    if loss_name == 'mae':
        return 'mae'
    elif loss_name == 'mse':
        return 'mse'
    elif loss_name == 'huber':
        return tf.keras.losses.Huber(delta=1.0)
    elif loss_name == 'log_cosh':
        return 'log_cosh'
    elif loss_name == 'custom_weighted':
        return weighted_mae_loss
    elif loss_name == 'smape_loss':
        return smape_loss
    elif loss_name == 'spike_weighted':
        return spike_weighted_loss
    else:
        return 'mse'

# Add a new custom objective function for XGBoost that penalizes extreme values
def xgb_smooth_trend_objective(y_pred, dtrain):
    """
    Custom XGBoost objective function that increasingly penalizes extreme values.
    This creates a strong incentive for the model to produce smooth, moderate predictions.
    
    Args:
        y_pred: Predicted values
        dtrain: Training data (including labels)
        
    Returns:
        gradient and hessian values for XGBoost
    """
    y_true = dtrain.get_label()
    error = y_pred - y_true
    
    # Get statistics from the true values to determine what's "extreme"
    all_labels = dtrain.get_label()
    mean_price = np.mean(all_labels)
    std_price = np.std(all_labels)
    
    # Calculate how far each prediction is from the mean
    pred_deviation = np.abs(y_pred - mean_price) / std_price
    
    # Create multipliers that increase as predictions get more extreme
    # Square the deviation to penalize extreme values more heavily
    multipliers = 1.0 + np.square(pred_deviation) * 0.5
    
    # Apply the multipliers to the gradients and hessians
    # For extreme predictions, the gradient will be stronger
    grad = multipliers * error
    hess = multipliers * np.ones_like(error)
    
    return grad, hess

# Add another objective function option for XGBoost that focuses on trend stability
def xgb_trend_stability_objective(y_pred, dtrain):
    """
    Custom XGBoost objective function that penalizes volatile predictions.
    This encourages the model to produce stable trends by penalizing
    rapid changes between consecutive predictions.
    
    Note: This requires sequential data in dtrain. Not supported out of the box,
    but included as a reference implementation.
    
    Args:
        y_pred: Predicted values
        dtrain: Training data
        
    Returns:
        gradient and hessian values for XGBoost
    """
    y_true = dtrain.get_label()
    
    # Basic squared error components
    error = y_pred - y_true
    grad = error
    hess = np.ones_like(error)
    
    # Add penalty for prediction volatility (changes between consecutive points)
    # This is a simplification - in practice we would need to track the actual sequence
    if len(y_pred) > 1:
        # Calculate volatility penalty - difference between consecutive predictions
        volatility = np.abs(np.diff(np.append(y_pred, y_pred[-1])))
        volatility_penalty = volatility * 0.5  # Scale factor for volatility penalty
        
        # Add volatility penalty to gradient
        # Positive gradient will decrease the value, negative will increase it
        # We want to move predictions closer to their neighbors
        grad_adjustment = np.zeros_like(grad)
        
        # For points with higher value than neighbors, increase gradient (push down)
        # For points with lower value than neighbors, decrease gradient (pull up)
        for i in range(1, len(y_pred)-1):
            if y_pred[i] > y_pred[i-1] and y_pred[i] > y_pred[i+1]:
                grad_adjustment[i] = volatility_penalty[i]
            elif y_pred[i] < y_pred[i-1] and y_pred[i] < y_pred[i+1]:
                grad_adjustment[i] = -volatility_penalty[i]
        
        grad += grad_adjustment
    
    return grad, hess

# ----- EVALUATION METRICS -----

def calculate_smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error."""
    # Add small constant to avoid division by zero
    epsilon = 1e-8
    
    # Calculate absolute difference
    abs_diff = np.abs(y_true - y_pred)
    
    # Calculate sum of absolute values
    abs_sum = np.abs(y_true) + np.abs(y_pred)
    
    # Calculate SMAPE
    smape = 200.0 * np.mean(abs_diff / (abs_sum + epsilon))
    
    return smape

def calculate_direction_accuracy(y_true, y_pred):
    """Calculate direction accuracy (percentage of correct up/down movement predictions)."""
    # Calculate direction of changes in true values
    true_diff = np.diff(y_true, axis=0)
    true_direction = np.sign(true_diff)
    
    # Calculate direction of changes in predicted values
    pred_diff = np.diff(y_pred, axis=0)
    pred_direction = np.sign(pred_diff)
    
    # Calculate direction accuracy
    correct_directions = (true_direction == pred_direction).sum()
    total_directions = true_direction.size
    
    return 100.0 * correct_directions / total_directions

def evaluate_model(model, X_test, y_test, target_scaler=None):
    """Evaluate the model on test data and calculate various metrics."""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Reshape data if needed
    if len(y_pred.shape) == 3 and y_pred.shape[2] == 1:
        y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
    
    # Apply inverse transform if scaler is provided
    if target_scaler is not None:
        # Reshape for inverse transform
        y_true_2d = y_test.reshape(-1, 1)
        y_pred_2d = y_pred.reshape(-1, 1)
        
        # Inverse transform
        y_true_original = target_scaler.inverse_transform(y_true_2d).reshape(y_test.shape)
        y_pred_original = target_scaler.inverse_transform(y_pred_2d).reshape(y_pred.shape)
    else:
        y_true_original = y_test
        y_pred_original = y_pred
    
    # Calculate metrics
    mae = np.mean(np.abs(y_true_original - y_pred_original))
    mse = np.mean((y_true_original - y_pred_original) ** 2)
    rmse = np.sqrt(mse)
    median_ae = np.median(np.abs(y_true_original - y_pred_original))
    
    # Calculate more advanced metrics
    smape = calculate_smape(y_true_original, y_pred_original)
    direction_accuracy = calculate_direction_accuracy(y_true_original, y_pred_original)
    
    # For MAPE, handle zero/near-zero values
    epsilon = 1.0
    adjusted_y_true = np.maximum(np.abs(y_true_original), epsilon)
    mape = 100.0 * np.mean(np.abs(y_true_original - y_pred_original) / adjusted_y_true)
    
    # Calculate horizon-wise metrics (average by prediction step)
    horizon_mae = np.mean(np.abs(y_true_original - y_pred_original), axis=0)
    horizon_rmse = np.sqrt(np.mean((y_true_original - y_pred_original) ** 2, axis=0))
    
    metrics = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'mape': float(mape),
        'smape': float(smape),
        'median_ae': float(median_ae),
        'direction_accuracy': float(direction_accuracy),
        'horizon_mae': horizon_mae.tolist() if isinstance(horizon_mae, np.ndarray) else float(horizon_mae),
        'horizon_rmse': horizon_rmse.tolist() if isinstance(horizon_rmse, np.ndarray) else float(horizon_rmse)
    }
    
    return metrics, y_pred_original

# ----- VISUALIZATION FUNCTIONS -----

def plot_training_history(history, save_path=None):
    """Plot training history (loss and metrics)."""
    # Create figure with subplots
    has_metrics = 'mae' in history.history
    num_plots = 2 if has_metrics else 1
    fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 6))
    
    # Convert to list if only one plot
    if num_plots == 1:
        axes = [axes]
    
    # Plot training and validation loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot training and validation MAE if available
    if has_metrics:
        axes[1].plot(history.history['mae'], label='Training MAE')
        if 'val_mae' in history.history:
            axes[1].plot(history.history['val_mae'], label='Validation MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Training and Validation MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_test_prediction(y_true, y_pred, save_path=None, start_idx=0, num_samples=5):
    """Plot test predictions alongside actual values."""
    # Create figure with subplots
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    # Plot each sample
    for i in range(num_samples):
        idx = start_idx + i
        if idx >= len(y_true):
            break
            
        true_values = y_true[idx]
        pred_values = y_pred[idx]
        
        # Create x-axis (hours ahead)
        hours_ahead = range(1, len(true_values) + 1)
        
        # Plot actual vs. predicted
        axes[i].plot(hours_ahead, true_values, 'b-', label='Actual', marker='o')
        axes[i].plot(hours_ahead, pred_values, 'r--', label='Predicted', marker='x')
        axes[i].set_xlabel('Hours Ahead')
        axes[i].set_ylabel(f'{TARGET_VARIABLE}')
        axes[i].set_title(f'Sample {idx} - {len(true_values)}h Forecast')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Calculate metrics for this sample
        sample_mae = np.mean(np.abs(true_values - pred_values))
        sample_rmse = np.sqrt(np.mean((true_values - pred_values) ** 2))
        sample_smape = calculate_smape(true_values, pred_values)
        
        # Add metrics to the plot
        axes[i].text(0.02, 0.95, f'MAE: {sample_mae:.2f}, RMSE: {sample_rmse:.2f}, SMAPE: {sample_smape:.2f}%',
                   transform=axes[i].transAxes, fontsize=10, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_feature_importance(model, feature_names, save_path=None):
    """Plot feature importance by analyzing gradients."""
    try:
        # Create a sample input
        sample_input = np.random.normal(size=(1, model.input_shape[1], model.input_shape[2]))
        
        # Create a gradient tape to track operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Convert the input to a tensor
            input_tensor = tf.convert_to_tensor(sample_input, dtype=tf.float32)
            tape.watch(input_tensor)
            
            # Get the prediction
            prediction = model(input_tensor)
            
            # Calculate the mean prediction
            mean_prediction = tf.reduce_mean(prediction)
        
        # Calculate the gradients with respect to the input
        gradients = tape.gradient(mean_prediction, input_tensor)
        
        # Calculate the importance of each feature (mean absolute gradient)
        importance = np.mean(np.abs(gradients.numpy()), axis=(0, 1))
        
        # Create a DataFrame for the feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        bars = plt.barh(importance_df['Feature'], importance_df['Importance'])
        
        # Add labels and title
        plt.xlabel('Importance (Mean Absolute Gradient)')
        plt.ylabel('Feature')
        plt.title('Feature Importance Analysis')
        plt.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    except Exception as e:
        logging.warning(f"Could not calculate feature importance: {e}") 