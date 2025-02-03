# data/preprocess.py

import pandas as pd
import numpy as np

def load_raw_data(csv_path):
    """
    Load raw data from a CSV file.
    
    Expected columns: 'entity_id', 'state', 'last_changed'.
    The 'last_changed' column is parsed as datetime and set as the index.
    """
    df = pd.read_csv(csv_path)
    df['last_changed'] = pd.to_datetime(df['last_changed'])
    df.set_index('last_changed', inplace=True)
    return df

def clean_data(df, missing_method='ffill'):
    """
    Clean the data by converting the 'state' column to numeric and handling non-numeric values.
    
    Args:
        df (DataFrame): The raw data.
        missing_method (str): How to handle missing values. Options are:
            - 'ffill': Forward fill
            - 'interpolate': Linear interpolation
            - 'drop': Drop missing values
    
    Returns:
        DataFrame: The cleaned data.
    """
    df['state'] = pd.to_numeric(df['state'], errors='coerce')
    if missing_method == 'ffill':
        df['state'] = df['state'].fillna(method='ffill')
    elif missing_method == 'interpolate':
        df['state'] = df['state'].interpolate(method='linear')
    else:
        df = df.dropna(subset=['state'])
    return df

def resample_data(df, rule='1H', agg='mean'):
    """
    Resample the DataFrame to a common time frequency.
    
    Args:
        df (DataFrame): The cleaned data.
        rule (str): The resampling frequency (e.g., '1H' for hourly).
        agg (str): Aggregation method to use. Options include 'mean' or 'sum'.
    
    Returns:
        DataFrame: The resampled data.
    """
    if agg == 'mean':
        return df.resample(rule).mean()
    elif agg == 'sum':
        return df.resample(rule).sum()
    else:
        return df.resample(rule).mean()
