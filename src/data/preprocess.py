# src/data/preprocess.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_raw_data(csv_path):
    """
    Load raw data from a CSV file.
    
    Expected columns: 'entity_id', 'state', 'last_changed'.
    The 'last_changed' column is parsed as datetime and set as the DataFrame index.
    
    If your CSV file includes a header row (with column names), no additional arguments are required.
    If not, you can specify header=None and provide the column names via the 'names' parameter.
    """
    df = pd.read_csv(csv_path)  # Adjust if your file does not contain a header row.
    df['last_changed'] = pd.to_datetime(df['last_changed'])
    df.set_index('last_changed', inplace=True)
    return df

def clean_data(df, missing_method='ffill'):
    """
    Clean the data by converting the 'state' column to numeric and handling non-numeric values.
    
    Args:
        df (pd.DataFrame): The raw DataFrame.
        missing_method (str): How to handle missing values. Options:
            - 'ffill': Forward-fill then backward-fill (to cover initial NaNs)
            - 'interpolate': Linear interpolation (with additional fill)
            - 'drop': Drop rows with missing 'state'
    
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Convert 'state' to numeric, coercing non-numeric values to NaN.
    df['state'] = pd.to_numeric(df['state'], errors='coerce')
    
    if missing_method == 'ffill':
        # Fill missing values using forward fill then backward fill to catch initial NaNs.
        df['state'] = df['state'].ffill().bfill()
    elif missing_method == 'interpolate':
        # Interpolate, then forward and backward fill as a safeguard.
        df['state'] = df['state'].interpolate(method='linear').ffill().bfill()
    else:  # 'drop'
        df = df.dropna(subset=['state'])
        
    return df

def resample_data(df, rule='1h', agg='mean'):
    """
    Resample the DataFrame to a common time frequency.
    
    Args:
        df (pd.DataFrame): The cleaned DataFrame.
        rule (str): The resampling frequency (e.g., '1h' for hourly).
        agg (str): Aggregation method to use ('mean' or 'sum').
    
    Returns:
        pd.DataFrame: The resampled DataFrame (only numeric columns are aggregated).
    """
    if agg == 'mean':
        return df.resample(rule).mean(numeric_only=True)
    elif agg == 'sum':
        return df.resample(rule).sum(numeric_only=True)
    else:
        return df.resample(rule).mean(numeric_only=True)

def normalize_series(series):
    """
    Normalize a pandas Series to the range [0, 1] using MinMaxScaler.
    
    This function first ensures that the series has no missing values
    (using forward and backward filling), then applies scaling.
    
    Args:
        series (pd.Series): The input time series.
    
    Returns:
        tuple: A tuple (normalized_series, scaler) where normalized_series is a pandas Series
            with values scaled to [0, 1], and scaler is the fitted MinMaxScaler.
    """
    # Ensure the series has no missing values.
    series = series.ffill().bfill()
    
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    # Replace any residual non-finite values, if any.
    scaled_values = np.nan_to_num(scaled_values, nan=0.0, posinf=1.0, neginf=0.0)
    
    normalized_series = pd.Series(scaled_values, index=series.index)
    return normalized_series, scaler
