import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_power_consumption_data(csv_path):
    """
    Load and preprocess a power consumption dataset from a CSV file.
    The CSV is expected to have columns: 'entity_id', 'state', 'last_changed'.
    The 'last_changed' column is parsed as datetime and set as the DataFrame index.
    """
    df = pd.read_csv(csv_path)
    df['last_changed'] = pd.to_datetime(df['last_changed'])
    df.set_index('last_changed', inplace=True)
    return df

def load_multiple_power_consumption_data(csv_paths):
    """
    Load and combine multiple power consumption datasets.
    
    Args:
        csv_paths (list of str): List of CSV file paths.
        
    Returns:
        pd.DataFrame: Combined and sorted DataFrame from all CSV files.
    """
    dfs = []
    for path in csv_paths:
        df = load_power_consumption_data(path)
        dfs.append(df)
    combined_df = pd.concat(dfs).sort_index()
    return combined_df

def get_power_consumption_series(df):
    """
    Extract the power consumption series from the DataFrame.
    The 'state' column is assumed to contain numeric values.
    """
    return pd.to_numeric(df['state'])

def df_to_X_y(series, input_window=168, forecast_horizon=168):
    """
    Convert a time series into samples and multi-step targets.
    
    For each sample, the model will receive `input_window` time steps as input
    and must predict the following `forecast_horizon` time steps.
    
    Args:
        series (pd.Series): The time series data.
        input_window (int): Number of historical time steps for input.
        forecast_horizon (int): Number of future time steps to forecast.
    
    Returns:
        tuple: (X, y) as numpy arrays.
            - X shape: (n_samples, input_window, 1)
            - y shape: (n_samples, forecast_horizon, 1)
    """
    data = series.to_numpy()
    X, y = [], []
    total_length = len(data)
    # We need enough points for both input and forecast horizon.
    for i in range(total_length - input_window - forecast_horizon + 1):
        X.append(data[i:i+input_window].reshape(input_window, 1))
        y.append(data[i+input_window:i+input_window+forecast_horizon].reshape(forecast_horizon, 1))
    return np.array(X), np.array(y)

def plot_series(series, title='Power Consumption Over Time'):
    """Plot the time series data."""
    series.plot(title=title)
    plt.xlabel('Time')
    plt.ylabel('Power Consumption')
    plt.show()
