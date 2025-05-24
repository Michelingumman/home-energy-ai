# Train file for demand predictions
import pandas as pd
import numpy as np
import logging
import pickle
import os
from holidays import country_holidays
from hmmlearn import hmm
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import optuna
from xgboost.callback import EarlyStopping # Import EarlyStopping callback
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Print XGBoost version for debugging
print(f"XGBoost Version: {xgb.__version__}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add progress bar support
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Create a simple fallback
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Constants
CONSUMPTION_DATA_PATH = 'data/processed/villamichelin/VillamichelinEnergyData.csv'
HEAT_PUMP_DATA_PATH = 'data/processed/villamichelin/Thermia/HeatPumpPower.csv'
WEATHER_DATA_PATH = 'data/processed/weather_data.csv'
MODEL_SAVE_PATH = 'src/predictions/demand/models/villamichelin_demand_model.pkl'
HMM_MODEL_SAVE_PATH = 'src/predictions/demand/models/villamichelin_hmm_model.pkl'
FEATURE_IMPORTANCE_SAVE_PATH = 'src/predictions/demand/plots/feature_importance.png'
N_HMM_STATES = 3
TARGET_COL = 'consumption'
FORECAST_HORIZON = 1 # Predicting t+1

# Ensure model directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(FEATURE_IMPORTANCE_SAVE_PATH), exist_ok=True)

# --- Data Pipeline Functions ---
def load_consumption_data(file_path: str) -> pd.DataFrame:
    logging.info(f"Loading consumption data from {file_path}")
    try:
        df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
        df.index = df.index.tz_localize('Europe/Stockholm', ambiguous=True, nonexistent='shift_forward')
        df = df[['consumption']]
        logging.info(f"Consumption data loaded successfully. Shape: {df.shape}")
        print("\nConsumption data head: \n", df.head())
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading consumption data: {e}")
        raise

def load_heat_pump_data(file_path: str) -> pd.DataFrame:
    logging.info(f"Loading heat pump data from {file_path}")
    try:
        df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
        df.index = df.index.tz_localize('Europe/Stockholm', ambiguous=True, nonexistent='shift_forward')
        df = df[['power_input_kw']]
        logging.info(f"Heat pump data loaded successfully. Shape: {df.shape}")
        print("\nHeat pump data head: \n", df.head())
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading heat pump data: {e}")
        raise

def load_weather_data(file_path: str) -> pd.DataFrame:
    logging.info(f"Loading weather data from {file_path}")
    try:
        df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
        df.index = df.index.tz_convert('Europe/Stockholm')
        df = df[['temperature_2m', 'cloud_cover', 'relative_humidity_2m', 'wind_speed_100m', 'shortwave_radiation_sum']]
        logging.info(f"Weather data loaded successfully. Shape: {df.shape}")
        print("\nWeather data head: \n", df.head())
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading weather data: {e}")
        raise

def resample_heat_pump_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Resampling heat pump data to hourly")
    try:
        df['power_input_kwh'] = df['power_input_kw'] * 0.25 # Convert to kWh
        df = df.drop(columns=['power_input_kw'])
        df_resampled = df['power_input_kwh'].resample('h').sum().to_frame()
        logging.info(f"Heat pump data resampled successfully. Shape: {df_resampled.shape}")
        return df_resampled
    except Exception as e:
        logging.error(f"Error resampling heat pump data: {e}")
        raise

def merge_data(consumption_df: pd.DataFrame, heat_pump_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Merging dataframes")
    try:
        merged_df = consumption_df.join(heat_pump_df, how='inner')
        merged_df = merged_df.join(weather_df, how='inner')
        logging.info(f"Dataframes merged successfully. Shape: {merged_df.shape}")
        print("\nMerged data head: \n", merged_df.head(), "\n", merged_df.tail())
        return merged_df
    except Exception as e:
        logging.error(f"Error merging dataframes: {e}")
        raise

def clean_and_impute_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Cleaning and imputing missing values")
    initial_missing = df.isnull().sum().sum()
    if initial_missing > 0:
        logging.info(f"Found {initial_missing} missing values before imputation.")
    df_cleaned = df.ffill().bfill()
    final_missing = df_cleaned.isnull().sum().sum()
    if final_missing > 0:
        logging.warning(f"Could not impute all missing values. {final_missing} missing values remain.")
    else:
        logging.info("Missing values imputed successfully.")
    return df_cleaned

def run_data_pipeline(consumption_path: str, heat_pump_path: str, weather_path: str) -> pd.DataFrame:
    logging.info("Starting data pipeline")
    consumption_data = load_consumption_data(consumption_path)
    heat_pump_data = load_heat_pump_data(heat_pump_path)
    weather_data = load_weather_data(weather_path)
    heat_pump_hourly = resample_heat_pump_data(heat_pump_data)
    merged_df = merge_data(consumption_data, heat_pump_hourly, weather_data)
    final_df = clean_and_impute_data(merged_df)
    logging.info(f"Data pipeline finished. Final dataset shape: {final_df.shape}")
    return final_df

# --- HMM Occupancy Functions ---
def fit_hmm(series: pd.Series, n_states: int, n_iterations: int = 100, random_state: int = 42) -> hmm.GaussianHMM:
    logging.info(f"Fitting HMM with {n_states} states.")
    data = series.values.reshape(-1, 1)
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=n_iterations, random_state=random_state)
    try:
        model.fit(data)
        logging.info("HMM fitting complete.")
        return model
    except Exception as e:
        logging.error(f"Error fitting HMM: {e}")
        if np.all(data == 0):
            logging.warning("Input data for HMM is all zeros. This can cause fitting issues.")
        raise

def decode_states(model: hmm.GaussianHMM, series: pd.Series) -> np.ndarray:
    logging.info("Decoding HMM states.")
    data = series.values.reshape(-1, 1)
    try:
        states = model.predict(data)
        logging.info("HMM states decoded.")
        return states
    except Exception as e:
        logging.error(f"Error decoding HMM states: {e}")
        raise

def get_state_posteriors(model: hmm.GaussianHMM, series: pd.Series) -> np.ndarray:
    logging.info("Calculating HMM state posterior probabilities.")
    data = series.values.reshape(-1, 1)
    try:
        posteriors = model.predict_proba(data)
        logging.info("HMM state posterior probabilities calculated.")
        return posteriors
    except Exception as e:
        logging.error(f"Error calculating HMM state posterior probabilities: {e}")
        raise

def add_hmm_features(df: pd.DataFrame, consumption_col: str, n_states: int) -> pd.DataFrame:
    if consumption_col not in df.columns:
        logging.error(f"Consumption column '{consumption_col}' not found in DataFrame.")
        raise ValueError(f"Consumption column '{consumption_col}' not found.")
    if df[consumption_col].isnull().any():
        logging.warning(f"Missing values found in '{consumption_col}'. HMM might behave unexpectedly.")
    
    series_for_hmm = df[consumption_col]
    if series_for_hmm.empty or len(series_for_hmm) < n_states:
        logging.error("Not enough data points to fit HMM.")
        # Add placeholder columns if HMM cannot be fit, to ensure consistency
        df['hmm_state'] = 0 
        for i in range(n_states):
            df[f'hmm_state_posterior_{i}'] = 1.0 / n_states
        logging.warning("Added placeholder HMM features as HMM could not be fit.")
        return df

    hmm_model = fit_hmm(series_for_hmm, n_states=n_states)
    
    # Save the HMM model
    try:
        with open(HMM_MODEL_SAVE_PATH, 'wb') as f:
            pickle.dump(hmm_model, f)
        logging.info(f"HMM model saved to {HMM_MODEL_SAVE_PATH}")
    except Exception as e:
        logging.error(f"Error saving HMM model: {e}")
        # Continue without HMM model if saving fails, but log error

    mean_consumptions = hmm_model.means_.flatten()
    state_order = np.argsort(mean_consumptions)
    
    decoded_states_raw = decode_states(hmm_model, series_for_hmm)
    posterior_probs_raw = get_state_posteriors(hmm_model, series_for_hmm)

    state_map = {old_label: new_label for new_label, old_label in enumerate(state_order)}
    df['hmm_state'] = np.array([state_map[s] for s in decoded_states_raw])
    
    posterior_probs_sorted = posterior_probs_raw[:, state_order]
    for i in range(n_states):
        df[f'hmm_state_posterior_{i}'] = posterior_probs_sorted[:, i]
        
    logging.info("HMM features added to DataFrame.")
    print("\nHMM features added to DataFrame: \n", df.head(), "\n", df.tail())
    return df

# --- Feature Engineering Functions ---
def add_lagged_features(df: pd.DataFrame, target_col: str, lags: list = [1, 2, 3, 24, 48, 72, 168]) -> pd.DataFrame:
    """
    Add advanced lagged and time series features to capture temporal patterns in energy consumption.
    
    Args:
        df: DataFrame with target column
        target_col: Name of target column (consumption)
        lags: List of lag periods to include (default includes hours, days, and week)
        
    Returns:
        DataFrame with added lag features
    """
    logging.info(f"Adding advanced lagged features for '{target_col}'")
    df_copy = df.copy()
    
    # Basic lag features
    for lag in lags:
        df_copy[f'{target_col}_lag_{lag}h'] = df_copy[target_col].shift(lag)
    
    # Moving averages at different time scales
    windows = [6, 12, 24, 48, 72, 168]  # 6h, 12h, 24h, 2d, 3d, 7d
    for window in windows:
        # Moving average
        df_copy[f'{target_col}_ma_{window}h'] = df_copy[target_col].rolling(
            window=window, min_periods=1).mean().shift(1)
        
        # Moving standard deviation (volatility)
        df_copy[f'{target_col}_std_{window}h'] = df_copy[target_col].rolling(
            window=window, min_periods=1).std().shift(1)
        
        # Moving min/max (range)
        df_copy[f'{target_col}_min_{window}h'] = df_copy[target_col].rolling(
            window=window, min_periods=1).min().shift(1)
        df_copy[f'{target_col}_max_{window}h'] = df_copy[target_col].rolling(
            window=window, min_periods=1).max().shift(1)
    
    # Differencing features
    df_copy[f'{target_col}_diff_1h'] = df_copy[target_col].diff(1)
    df_copy[f'{target_col}_diff_24h'] = df_copy[target_col].diff(24)
    df_copy[f'{target_col}_diff_168h'] = df_copy[target_col].diff(168)
    
    # Percentage change features
    df_copy[f'{target_col}_pct_1h'] = df_copy[target_col].pct_change(1)
    df_copy[f'{target_col}_pct_24h'] = df_copy[target_col].pct_change(24)
    
    # Rate of change features (momentum)
    df_copy[f'{target_col}_roc_24h'] = (df_copy[target_col] / df_copy[target_col].shift(24) - 1) * 100
    
    # Time of day patterns
    # Same hour in previous days
    for days_ago in [1, 2, 3, 7, 14]:
        df_copy[f'{target_col}_same_hour_{days_ago}d_ago'] = df_copy[target_col].shift(24 * days_ago)
    
    # Weekly patterns - same hour and day in previous weeks
    for weeks_ago in [1, 2, 4]:
        df_copy[f'{target_col}_same_hour_dow_{weeks_ago}w_ago'] = df_copy[target_col].shift(168 * weeks_ago)
    
    # Hour of day average consumption (captures daily patterns)
    if isinstance(df_copy.index, pd.DatetimeIndex):
        df_copy['hour'] = df_copy.index.hour
        hour_avg = df_copy.groupby('hour')[target_col].transform('mean')
        df_copy[f'{target_col}_hour_avg_ratio'] = df_copy[target_col] / hour_avg
        df_copy.drop('hour', axis=1, inplace=True)
    
    # Day of week average consumption (captures weekly patterns)
    if isinstance(df_copy.index, pd.DatetimeIndex):
        df_copy['dayofweek'] = df_copy.index.dayofweek
        dow_avg = df_copy.groupby('dayofweek')[target_col].transform('mean')
        df_copy[f'{target_col}_dow_avg_ratio'] = df_copy[target_col] / dow_avg
        df_copy.drop('dayofweek', axis=1, inplace=True)
    
    # Lag features for power input if available
    power_col = 'power_input_kwh'
    if power_col in df_copy.columns:
        for lag in [1, 24, 168]:
            df_copy[f'{power_col}_lag_{lag}h'] = df_copy[power_col].shift(lag)
        
        # Moving averages for power
        for window in [24, 168]:
            df_copy[f'{power_col}_ma_{window}h'] = df_copy[power_col].rolling(
                window=window, min_periods=1).mean().shift(1)
        
        # Interaction between power and consumption
        df_copy[f'{target_col}_power_ratio'] = df_copy[target_col] / (df_copy[power_col] + 0.1)
    
    # Fill NaN values in lag features with appropriate statistical measures
    # Create a list of columns that might have NaN values
    cols_to_fill = []
    for col in df_copy.columns:
        if (col.startswith(f'{target_col}_lag_') or 
            col.startswith(f'{target_col}_ma_') or 
            col.startswith(f'{target_col}_std_') or 
            col.startswith(f'{target_col}_diff_') or
            col.startswith(f'{target_col}_pct_') or 
            col.startswith(f'{target_col}_roc_') or
            col.startswith(f'{target_col}_same_hour_')):
            if df_copy[col].isnull().any():
                cols_to_fill.append(col)
        
        if power_col in df_copy.columns:
            if (col.startswith(f'{power_col}_lag_') or 
                col.startswith(f'{power_col}_ma_')):
                if df_copy[col].isnull().any():
                    cols_to_fill.append(col)
    
    # Calculate means for columns with NaNs
    fill_values = {col: df_copy[col].mean() for col in cols_to_fill}
    
    # Fill NaNs all at once
    df_copy = df_copy.fillna(fill_values)
    
    logging.info(f"Added {len(df_copy.columns) - len(df.columns)} new lagged and time series features.")
    return df_copy

def add_calendar_features(df: pd.DataFrame, country_code: str = 'SE', years_range: tuple = (2020, 2025)) -> pd.DataFrame:
    """
    Add advanced calendar features to capture time-based patterns in energy consumption.
    
    Args:
        df: DataFrame with datetime index
        country_code: Country code for holidays
        years_range: Range of years to consider for holidays
        
    Returns:
        DataFrame with added calendar features
    """
    logging.info("Adding advanced calendar features.")
    df_copy = df.copy()
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        logging.error("DataFrame index is not DatetimeIndex.")
        raise TypeError("DataFrame index must be DatetimeIndex for calendar features.")

    # Store timezone info for consistent handling
    tz_info = df_copy.index.tz

    # Basic time components
    df_copy['hour_of_day'] = df_copy.index.hour
    df_copy['day_of_week_num'] = df_copy.index.dayofweek  # 0=Monday, 6=Sunday
    df_copy['month_num'] = df_copy.index.month
    df_copy['year'] = df_copy.index.year
    df_copy['day_of_month'] = df_copy.index.day
    df_copy['week_of_year'] = df_copy.index.isocalendar().week
    
    # Circular encoding for cyclical features
    df_copy['hour_sin'] = np.sin(2 * np.pi * df_copy['hour_of_day'] / 24.0)
    df_copy['hour_cos'] = np.cos(2 * np.pi * df_copy['hour_of_day'] / 24.0)
    df_copy['day_of_week_sin'] = np.sin(2 * np.pi * df_copy['day_of_week_num'] / 7.0)
    df_copy['day_of_week_cos'] = np.cos(2 * np.pi * df_copy['day_of_week_num'] / 7.0)
    df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month_num'] / 12.0)
    df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month_num'] / 12.0)
    df_copy['day_of_month_sin'] = np.sin(2 * np.pi * df_copy['day_of_month'] / 31.0)
    df_copy['day_of_month_cos'] = np.cos(2 * np.pi * df_copy['day_of_month'] / 31.0)
    
    # Time period indicators
    df_copy['is_weekend'] = (df_copy.index.dayofweek >= 5).astype(int)
    df_copy['is_night'] = ((df_copy['hour_of_day'] >= 22) | (df_copy['hour_of_day'] <= 6)).astype(int)
    
    # Morning/Evening peak indicators
    df_copy['is_morning_peak'] = ((df_copy['hour_of_day'] >= 6) & (df_copy['hour_of_day'] <= 9)).astype(int)
    df_copy['is_evening_peak'] = ((df_copy['hour_of_day'] >= 17) & (df_copy['hour_of_day'] <= 21)).astype(int)
    
    # Day segments - more granular time periods
    conditions = [
        (df_copy['hour_of_day'] >= 0) & (df_copy['hour_of_day'] < 6),
        (df_copy['hour_of_day'] >= 6) & (df_copy['hour_of_day'] < 12),
        (df_copy['hour_of_day'] >= 12) & (df_copy['hour_of_day'] < 18),
        (df_copy['hour_of_day'] >= 18) & (df_copy['hour_of_day'] <= 23)
    ]
    values = ['night', 'morning', 'afternoon', 'evening']
    df_copy['day_segment'] = np.select(conditions, values, default='unknown')
    
    # One-hot encode day segment
    df_copy = pd.get_dummies(df_copy, columns=['day_segment'], prefix='segment', dtype=int)
    
    # Seasons in Northern Hemisphere
    # Winter: Dec-Feb, Spring: Mar-May, Summer: Jun-Aug, Fall: Sep-Nov
    conditions = [
        (df_copy['month_num'] >= 3) & (df_copy['month_num'] <= 5),
        (df_copy['month_num'] >= 6) & (df_copy['month_num'] <= 8),
        (df_copy['month_num'] >= 9) & (df_copy['month_num'] <= 11),
        (df_copy['month_num'] == 12) | (df_copy['month_num'] <= 2)
    ]
    values = ['spring', 'summer', 'fall', 'winter']
    df_copy['season'] = np.select(conditions, values, default='unknown')
    
    # One-hot encode season
    df_copy = pd.get_dummies(df_copy, columns=['season'], prefix='season', dtype=int)
    
    # Holiday features
    min_year = df_copy.index.year.min()
    max_year = df_copy.index.year.max()
    data_years = df_copy.index.year.unique().tolist()
    all_years_for_holidays = sorted(list(set(data_years + list(range(min_year, max_year + 1)))))
    
    try:
        # Create a normalized index (date only, no time) to compare with holiday dates
        df_copy['date_normalized'] = df_copy.index.normalize()
        
        # Get holidays using country_holidays
        country_holidays_list = country_holidays(country_code, years=all_years_for_holidays)
        
        # Create a localized DatetimeIndex from holiday dates matching the DataFrame's timezone
        holiday_dates = pd.DatetimeIndex([pd.Timestamp(date).tz_localize('UTC').tz_convert(tz_info) 
                                         for date in country_holidays_list.keys()])
        
        # Normalize the holiday dates to remove time component
        holiday_dates_normalized = holiday_dates.normalize()
        
        # Check if each date is a holiday
        df_copy['is_holiday'] = df_copy['date_normalized'].isin(holiday_dates_normalized).astype(int)
        
        # Holiday proximity features
        days_to_next_holiday = []
        days_since_prev_holiday = []
        
        # Iterate through each row to calculate holiday proximity
        for idx, row in df_copy.iterrows():
            day_idx = row['date_normalized']
            
            # Find future holidays
            future_holidays = holiday_dates_normalized[holiday_dates_normalized > day_idx]
            if len(future_holidays) > 0:
                next_holiday = future_holidays[0]
                # Calculate days difference - both timestamps are timezone-aware
                days_to = (next_holiday - day_idx).days
            else:
                days_to = 365  # Default if no future holidays in data
            days_to_next_holiday.append(days_to)
            
            # Find previous holidays
            past_holidays = holiday_dates_normalized[holiday_dates_normalized < day_idx]
            if len(past_holidays) > 0:
                prev_holiday = past_holidays[-1]
                days_since = (day_idx - prev_holiday).days
            else:
                days_since = 365  # Default if no past holidays in data
            days_since_prev_holiday.append(days_since)
        
        df_copy['days_to_next_holiday'] = days_to_next_holiday
        df_copy['days_since_holiday'] = days_since_prev_holiday
        
        # Holiday eve indicator (day before holiday)
        df_copy['is_holiday_eve'] = (df_copy['days_to_next_holiday'] == 1).astype(int)
        
    except Exception as e:
        logging.warning(f"Error processing holidays for country code '{country_code}': {e}")
        logging.warning("Defaulting to no holiday features.")
        df_copy['is_holiday'] = 0
        df_copy['days_to_next_holiday'] = 365
        df_copy['days_since_holiday'] = 365
        df_copy['is_holiday_eve'] = 0
    
    # Special time periods (work hours, quiet hours)
    df_copy['is_work_hour'] = ((df_copy['hour_of_day'] >= 8) & 
                              (df_copy['hour_of_day'] <= 17) & 
                              (~df_copy['is_weekend'].astype(bool)) &
                              (~df_copy['is_holiday'].astype(bool))).astype(int)
    
    df_copy['is_quiet_hour'] = ((df_copy['hour_of_day'] >= 22) | 
                               (df_copy['hour_of_day'] <= 6)).astype(int)
    
    # Drop temporary columns
    if 'date_normalized' in df_copy.columns:
        df_copy.drop('date_normalized', axis=1, inplace=True)
    
    # Drop original time columns used for derivatives
    cols_to_drop = ['hour_of_day', 'day_of_week_num', 'month_num', 'year', 'day_of_month', 'week_of_year']
    df_copy.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    logging.info(f"Added {len(df_copy.columns) - len(df.columns)} new calendar features.")
    return df_copy

def add_weather_transforms(df: pd.DataFrame, temp_col: str = 'temperature_2m', t_base: float = 18.0) -> pd.DataFrame:
    """
    Add advanced weather transformation features to improve model performance.
    
    Args:
        df: DataFrame with weather data
        temp_col: Name of temperature column
        t_base: Base temperature for heating/cooling degree calculations (Celsius)
        
    Returns:
        DataFrame with added weather features
    """
    logging.info("Adding advanced weather transformation features.")
    df_copy = df.copy()
    
    if temp_col not in df_copy.columns:
        logging.error(f"Temperature column '{temp_col}' not found.")
        raise ValueError(f"Temperature column '{temp_col}' not found.")
    
    # Basic heating/cooling degree hours
    df_copy['heating_degree_hours'] = np.maximum(0, t_base - df_copy[temp_col])
    df_copy['cooling_degree_hours'] = np.maximum(0, df_copy[temp_col] - t_base)
    
    # Exponential temperature impact
    # These capture the non-linear relationship between temperature and energy consumption
    df_copy['temp_exp_heating'] = np.exp(-np.maximum(0, t_base - df_copy[temp_col]) / 10)
    df_copy['temp_exp_cooling'] = np.exp(-np.maximum(0, df_copy[temp_col] - t_base) / 10)
    
    # Temperature rate of change (hourly delta)
    df_copy['temp_change_1h'] = df_copy[temp_col].diff()
    
    # Temperature moving average features
    df_copy['temp_ma_6h'] = df_copy[temp_col].rolling(window=6, min_periods=1).mean()
    df_copy['temp_ma_24h'] = df_copy[temp_col].rolling(window=24, min_periods=1).mean()
    
    # Temperature variability
    df_copy['temp_std_24h'] = df_copy[temp_col].rolling(window=24, min_periods=1).std()
    
    # Seasonal temperature normalization
    # Calculate monthly averages for temperature (climate normal)
    df_copy['month'] = df_copy.index.month
    monthly_avg_temp = df_copy.groupby('month')[temp_col].transform('mean')
    df_copy['temp_vs_monthly_avg'] = df_copy[temp_col] - monthly_avg_temp
    df_copy.drop('month', axis=1, inplace=True)
    
    # Weather interaction features
    if 'cloud_cover' in df_copy.columns:
        # Cloud cover affects solar radiation and heating/cooling needs
        df_copy['clear_sky_heating'] = df_copy['heating_degree_hours'] * (100 - df_copy['cloud_cover']) / 100
        df_copy['clear_sky_cooling'] = df_copy['cooling_degree_hours'] * (100 - df_copy['cloud_cover']) / 100
    
    if 'relative_humidity_2m' in df_copy.columns:
        # Humidity affects perceived temperature
        df_copy['humid_heat_index'] = df_copy[temp_col] * (1 + 0.005 * df_copy['relative_humidity_2m'])
        
    if 'wind_speed_100m' in df_copy.columns:
        # Wind chill effect for heating
        df_copy['wind_chill_factor'] = np.where(
            df_copy[temp_col] < 10,  # Only apply wind chill when cold
            df_copy[temp_col] - (0.1 * df_copy['wind_speed_100m']),
            df_copy[temp_col]
        )
        df_copy['wind_enhanced_hdh'] = np.maximum(0, t_base - df_copy['wind_chill_factor'])
    
    if 'shortwave_radiation_sum' in df_copy.columns:
        # Solar radiation impact on heating/cooling
        df_copy['solar_heating_offset'] = df_copy['heating_degree_hours'] * np.exp(-df_copy['shortwave_radiation_sum'] / 100)
        
        # Day/night indicator based on solar radiation
        df_copy['is_daylight'] = (df_copy['shortwave_radiation_sum'] > 10).astype(int)
        
        # Interaction between solar radiation and temperature
        df_copy['solar_temp_interaction'] = df_copy[temp_col] * df_copy['shortwave_radiation_sum'] / 100
    
    # Weather change indicators
    for col in ['cloud_cover', 'relative_humidity_2m', 'wind_speed_100m', 'shortwave_radiation_sum']:
        if col in df_copy.columns:
            # Rate of change in weather parameters
            df_copy[f'{col}_change'] = df_copy[col].diff()
            
            # 24h difference to capture daily patterns
            df_copy[f'{col}_24h_diff'] = df_copy[col].diff(24)
    
    logging.info(f"Added {len(df_copy.columns) - len(df.columns)} new weather transformation features.")
    return df_copy

def add_interaction_terms(df: pd.DataFrame, hmm_state_col: str = 'hmm_state', temp_col: str = 'temperature_2m') -> pd.DataFrame:
    """
    Add advanced interaction features to capture complex relationships between variables.
    
    Args:
        df: DataFrame with features
        hmm_state_col: Name of HMM state column
        temp_col: Name of temperature column
        
    Returns:
        DataFrame with added interaction features
    """
    logging.info("Adding advanced interaction features.")
    df_copy = df.copy()
    
    feature_count_before = len(df_copy.columns)
    
    # Base features to interact with
    base_features = []
    
    # HMM state interactions
    if hmm_state_col in df_copy.columns:
        base_features.append(hmm_state_col)
    else:
        logging.warning(f"HMM state column '{hmm_state_col}' not found. Skipping HMM interactions.")
    
    # Temperature interactions
    if temp_col in df_copy.columns:
        base_features.append(temp_col)
        
        # Special temperature interaction with HMM
        if hmm_state_col in df_copy.columns:
            df_copy[f'{hmm_state_col}_x_{temp_col}'] = df_copy[hmm_state_col] * df_copy[temp_col]
            
            # More complex temperature-state relationship (quadratic)
            df_copy[f'{hmm_state_col}_x_{temp_col}_squared'] = df_copy[hmm_state_col] * (df_copy[temp_col] ** 2)
    
    # Weather derivatives interactions
    weather_features = [
        'heating_degree_hours', 'cooling_degree_hours',
        'temp_exp_heating', 'temp_exp_cooling', 
        'humid_heat_index', 'wind_chill_factor'
    ]
    
    weather_cols = [col for col in weather_features if col in df_copy.columns]
    
    # Time features
    time_features = [col for col in df_copy.columns if 
                     col.startswith('hour_') or 
                     col.startswith('day_of_week_') or
                     col.startswith('is_weekend') or
                     col.startswith('is_holiday') or
                     col.startswith('is_morning_peak') or 
                     col.startswith('is_evening_peak')]
    
    # For each feature in base_features, create interactions with time features
    for base_feature in base_features:
        for time_feature in time_features:
            if time_feature in df_copy.columns:
                df_copy[f'{time_feature}_x_{base_feature}'] = df_copy[time_feature] * df_copy[base_feature]
    
    # Weather feature interactions with time features
    for weather_col in weather_cols:
        # Interact with weekend/holiday/peak indicators
        for indicator in ['is_weekend', 'is_holiday', 'is_morning_peak', 'is_evening_peak']:
            if indicator in df_copy.columns:
                df_copy[f'{indicator}_x_{weather_col}'] = df_copy[indicator] * df_copy[weather_col]
    
    # Heating/cooling degree hours with seasonal indicators
    for season in ['season_winter', 'season_summer']:
        if season in df_copy.columns:
            if 'heating_degree_hours' in df_copy.columns:
                df_copy[f'{season}_x_heating_degree_hours'] = df_copy[season] * df_copy['heating_degree_hours']
            if 'cooling_degree_hours' in df_copy.columns:
                df_copy[f'{season}_x_cooling_degree_hours'] = df_copy[season] * df_copy['cooling_degree_hours']
    
    # Solar radiation interactions 
    if 'shortwave_radiation_sum' in df_copy.columns:
        # Solar x Temperature
        if temp_col in df_copy.columns:
            df_copy[f'solar_x_{temp_col}'] = df_copy['shortwave_radiation_sum'] * df_copy[temp_col]
        
        # Solar x Time of day
        if 'hour_sin' in df_copy.columns and 'hour_cos' in df_copy.columns:
            df_copy['solar_x_hour_sin'] = df_copy['shortwave_radiation_sum'] * df_copy['hour_sin']
            df_copy['solar_x_hour_cos'] = df_copy['shortwave_radiation_sum'] * df_copy['hour_cos']
    
    # Power consumption with weather
    if 'power_input_kwh' in df_copy.columns:
        power_col = 'power_input_kwh'
        
        # Power x Temperature
        if temp_col in df_copy.columns:
            df_copy[f'{power_col}_x_{temp_col}'] = df_copy[power_col] * df_copy[temp_col]
        
        # Power x Weather features
        for weather_col in weather_cols:
            df_copy[f'{power_col}_x_{weather_col}'] = df_copy[power_col] * df_copy[weather_col]
    
    # Create special consumption ratio features if lag features exist
    consumption_col = 'consumption'
    lag_cols = [col for col in df_copy.columns if col.startswith(f'{consumption_col}_lag_')]
    
    # If we have enough lag columns, create ratio features
    if len(lag_cols) >= 3:
        # Ratio between different timeframes
        df_copy['consumption_ratio_day'] = df_copy[lag_cols[0]] / (df_copy[lag_cols[1]] + 1e-6)  # day/day ratio
        df_copy['consumption_ratio_week'] = df_copy[lag_cols[1]] / (df_copy[lag_cols[2]] + 1e-6)  # day/week ratio
    
    # Limit the number of interaction features to avoid dimensionality explosion
    # Keep track of how many features we've added and limit if necessary
    feature_count_after = len(df_copy.columns)
    num_features_added = feature_count_after - feature_count_before
    
    logging.info(f"Added {num_features_added} new interaction features.")
    return df_copy

def add_heat_pump_baseload_features(df: pd.DataFrame, consumption_col: str = 'consumption', 
                                    power_col: str = 'power_input_kwh', temp_col: str = 'temperature_2m') -> pd.DataFrame:
    """
    Add specialized features for heat pump baseload demand prediction.
    
    Heat pump baseload is primarily driven by:
    1. Heating/cooling needs based on outdoor temperature
    2. Building thermal dynamics (thermal mass effects)
    3. Heat pump efficiency curves
    4. Occupancy patterns
    
    Args:
        df: DataFrame with consumption, power, and temperature data
        consumption_col: Name of consumption column
        power_col: Name of heat pump power consumption column  
        temp_col: Name of temperature column
        
    Returns:
        DataFrame with heat pump baseload features
    """
    logging.info("Adding heat pump baseload specific features")
    df_copy = df.copy()
    
    if temp_col not in df_copy.columns:
        logging.warning(f"Temperature column '{temp_col}' not found. Skipping heat pump specific features.")
        return df_copy
    
    # Heat pump efficiency features based on temperature
    # Heat pump efficiency drops significantly at extreme temperatures
    df_copy['temp_squared'] = df_copy[temp_col] ** 2
    df_copy['temp_cubed'] = df_copy[temp_col] ** 3
    
    # COP (Coefficient of Performance) estimation based on outdoor temperature
    # Simplified COP model: COP decreases as temperature difference increases
    t_indoor = 21.0  # Assumed indoor temperature
    df_copy['temp_diff_indoor'] = np.abs(df_copy[temp_col] - t_indoor)
    df_copy['estimated_cop'] = np.maximum(2.0, 5.0 - 0.1 * df_copy['temp_diff_indoor'])
    
    # Heat pump operating regime indicators
    df_copy['extreme_cold'] = (df_copy[temp_col] < -5).astype(int)  # Below -5°C
    df_copy['cold'] = ((df_copy[temp_col] >= -5) & (df_copy[temp_col] < 5)).astype(int)  # -5 to 5°C
    df_copy['mild'] = ((df_copy[temp_col] >= 5) & (df_copy[temp_col] < 15)).astype(int)  # 5 to 15°C
    df_copy['warm'] = ((df_copy[temp_col] >= 15) & (df_copy[temp_col] < 25)).astype(int)  # 15 to 25°C
    df_copy['hot'] = (df_copy[temp_col] >= 25).astype(int)  # Above 25°C
    
    # Thermal mass effects - building responds slowly to temperature changes
    # Rolling temperature features capture thermal inertia
    df_copy['temp_ma_3h'] = df_copy[temp_col].rolling(window=3, min_periods=1).mean()
    df_copy['temp_ma_6h'] = df_copy[temp_col].rolling(window=6, min_periods=1).mean()
    df_copy['temp_ma_12h'] = df_copy[temp_col].rolling(window=12, min_periods=1).mean()
    
    # Temperature trend (building heating/cooling inertia)
    df_copy['temp_trend_3h'] = df_copy[temp_col] - df_copy['temp_ma_3h']
    df_copy['temp_trend_6h'] = df_copy[temp_col] - df_copy['temp_ma_6h']
    
    # Degree hours with more precise base temperatures for heat pumps
    # Heat pumps typically start significant heating around 15-18°C
    for base_temp in [15.0, 16.0, 17.0, 18.0, 19.0, 20.0]:
        df_copy[f'hdh_{base_temp}c'] = np.maximum(0, base_temp - df_copy[temp_col])
        df_copy[f'cdh_{base_temp}c'] = np.maximum(0, df_copy[temp_col] - base_temp)
    
    # Heat pump power consumption features if available
    if power_col in df_copy.columns:
        # Heat pump efficiency (consumption per unit of heat delivered)
        # Higher efficiency when power consumption is lower for same heating need
        df_copy['hp_efficiency_proxy'] = df_copy[f'hdh_18.0c'] / (df_copy[power_col] + 0.1)
        
        # Heat pump cycling behavior (frequent on/off indicates inefficiency)
        df_copy['hp_power_change'] = df_copy[power_col].diff().abs()
        df_copy['hp_cycling_indicator'] = df_copy['hp_power_change'].rolling(window=6).sum()
        
        # Heat pump utilization rate
        max_power = df_copy[power_col].quantile(0.95)  # Assume 95th percentile is near max capacity
        if max_power > 0:
            df_copy['hp_utilization'] = df_copy[power_col] / max_power
        else:
            # If max power is 0 or very small, set utilization to 0
            df_copy['hp_utilization'] = 0.0
        
        # Power consumption in different temperature regimes
        df_copy['hp_power_extreme_cold'] = df_copy[power_col] * df_copy['extreme_cold']
        df_copy['hp_power_cold'] = df_copy[power_col] * df_copy['cold']
        df_copy['hp_power_mild'] = df_copy[power_col] * df_copy['mild']
        
        # Non-heat pump baseload estimation
        # Estimate minimum consumption when heat pump is not running much
        min_consumption_percentile = df_copy[consumption_col].quantile(0.05)
        df_copy['estimated_baseload'] = np.minimum(df_copy[consumption_col], min_consumption_percentile * 1.2)
        df_copy['hp_contribution'] = df_copy[consumption_col] - df_copy['estimated_baseload']
    
    # Weather-related heat loss/gain features
    if 'wind_speed_100m' in df_copy.columns:
        # Wind increases heat loss from building
        df_copy['wind_chill_heat_loss'] = df_copy['temp_diff_indoor'] * (1 + 0.1 * df_copy['wind_speed_100m'])
    
    if 'shortwave_radiation_sum' in df_copy.columns:
        # Solar gain reduces heating need during day
        df_copy['solar_heat_gain'] = df_copy['shortwave_radiation_sum'] / (df_copy['temp_diff_indoor'] + 1)
        
        # Solar availability affects heat pump supplement strategies
        df_copy['solar_available'] = (df_copy['shortwave_radiation_sum'] > 50).astype(int)
        df_copy['solar_low'] = ((df_copy['shortwave_radiation_sum'] > 10) & 
                               (df_copy['shortwave_radiation_sum'] <= 50)).astype(int)
        df_copy['no_solar'] = (df_copy['shortwave_radiation_sum'] <= 10).astype(int)
    
    # Seasonal adjustment for heat pump baseload
    if isinstance(df_copy.index, pd.DatetimeIndex):
        df_copy['month'] = df_copy.index.month
        
        # Different base temperatures for different seasons
        winter_months = [12, 1, 2]
        summer_months = [6, 7, 8]
        
        df_copy['is_winter'] = df_copy['month'].isin(winter_months).astype(int)
        df_copy['is_summer'] = df_copy['month'].isin(summer_months).astype(int)
        df_copy['is_transition'] = (~df_copy['month'].isin(winter_months + summer_months)).astype(int)
        
        # Seasonal degree hours
        df_copy['winter_hdh'] = df_copy['is_winter'] * df_copy['hdh_18.0c']
        df_copy['summer_cdh'] = df_copy['is_summer'] * df_copy['cdh_18.0c']
        df_copy['transition_hdh'] = df_copy['is_transition'] * df_copy['hdh_18.0c']
        
        df_copy.drop('month', axis=1, inplace=True)
    
    # Heat pump defrost cycle indicators (typically needed in cold, humid conditions)
    if 'relative_humidity_2m' in df_copy.columns and temp_col in df_copy.columns:
        # Defrost cycles typically needed when temp is below 5°C and humidity is high
        df_copy['defrost_conditions'] = ((df_copy[temp_col] < 5) & 
                                        (df_copy['relative_humidity_2m'] > 80)).astype(int)
        df_copy['frost_risk'] = ((df_copy[temp_col] < 2) & 
                                (df_copy['relative_humidity_2m'] > 70)).astype(int)
    
    # Thermal comfort zone indicators
    df_copy['in_comfort_zone'] = ((df_copy[temp_col] >= 18) & (df_copy[temp_col] <= 22)).astype(int)
    df_copy['below_comfort'] = (df_copy[temp_col] < 18).astype(int)
    df_copy['above_comfort'] = (df_copy[temp_col] > 22).astype(int)
    
    logging.info(f"Added {len(df_copy.columns) - len(df.columns)} heat pump baseload features")
    return df_copy

def engineer_features(df: pd.DataFrame, target_col: str, country_code_for_holidays: str = 'SE', create_target: bool = True) -> pd.DataFrame:
    logging.info("Starting feature engineering pipeline.")
    df_featured = df.copy()
    df_featured = add_lagged_features(df_featured, target_col=target_col)
    df_featured = add_calendar_features(df_featured, country_code=country_code_for_holidays)
    df_featured = add_weather_transforms(df_featured)
    df_featured = add_interaction_terms(df_featured)
    df_featured = add_heat_pump_baseload_features(df_featured)
    
    if create_target:
        # Define target variable y (demand at t+1)
        df_featured['y'] = df_featured[target_col].shift(-FORECAST_HORIZON)
        
        initial_rows = len(df_featured)
        df_featured.dropna(inplace=True) # Drops rows with NaNs from lags and from shifting for y
        rows_dropped = initial_rows - len(df_featured)
        if rows_dropped > 0:
            logging.info(f"Dropped {rows_dropped} rows due to NaN values (lags/target shift).")
    else:
        # When not creating target (e.g., for prediction), we still need to handle NaNs from feature creation
        # Typically, for prediction, the last row(s) will have NaNs in lagged features if not enough history is provided.
        # The model expects a full feature set. The prediction function should ensure enough data is passed to engineer_features
        # so that the row(s) for which predictions are needed have complete features.
        # Or, if NaNs are unavoidable for the very first few predictions with limited history, they need a strategy (e.g. imputation or error).
        # For now, assume the calling function (make_predictions) handles providing sufficient historical data.
        # We will drop rows where *all* features might be NaN if that somehow occurs, but typically lags create NaNs at the start.
        # df_featured.dropna(how='all', inplace=True) # Example: if a row became all NaNs

        # More importantly, ensure columns match training set, excluding 'y'
        pass

    logging.info(f"Feature engineering pipeline finished. Final dataset shape: {df_featured.shape}")
    return df_featured

# --- New Feature Importance Visualization Function ---
def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """
    Visualize feature importance from the trained model
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        top_n: Number of top features to display
        save_path: Path to save the plot
    """
    logging.info(f"Generating feature importance visualization for top {top_n} features")
    
    # Get feature importance from the model
    importance_dict = {}
    
    # Try different methods to get feature importance
    try:
        # Method 1: Using feature_importances_ attribute (sklearn-style)
        if hasattr(model, 'feature_importances_'):
            importance_dict['importance'] = dict(zip(feature_names, model.feature_importances_))
            logging.info("Successfully extracted feature importance using feature_importances_")
    except Exception as e:
        logging.warning(f"Could not get feature_importances_: {e}")
    
    try:
        # Method 2: Using get_score method (if model is trained)
        if hasattr(model, 'get_score'):
            for importance_type in ['weight', 'gain', 'cover', 'total_gain', 'total_cover']:
                try:
                    importance = model.get_score(importance_type=importance_type)
                    importance_dict[importance_type] = importance
                    logging.info(f"Successfully extracted {importance_type} importance")
                except Exception as e:
                    logging.warning(f"Could not get {importance_type} importance: {e}")
    except Exception as e:
        logging.warning(f"Could not use get_score method: {e}")
    
    try:
        # Method 3: Using booster feature importance (if booster exists)
        if hasattr(model, 'get_booster'):
            booster = model.get_booster()
            if hasattr(booster, 'get_score'):
                for importance_type in ['weight', 'gain', 'cover']:
                    try:
                        importance = booster.get_score(importance_type=importance_type)
                        importance_dict[f'booster_{importance_type}'] = importance
                        logging.info(f"Successfully extracted booster {importance_type} importance")
                    except Exception as e:
                        logging.warning(f"Could not get booster {importance_type} importance: {e}")
    except Exception as e:
        logging.warning(f"Could not access booster: {e}")
    
    if not importance_dict:
        logging.error("Could not retrieve feature importance from model using any method")
        return None
    
    # Create a DataFrame for visualization
    importance_df = pd.DataFrame({k: pd.Series(v) for k, v in importance_dict.items()})
    importance_df = importance_df.fillna(0)
    
    # Normalize importance scores for better comparison
    for col in importance_df.columns:
        if importance_df[col].sum() > 0:
            importance_df[col] = importance_df[col] / importance_df[col].sum()
    
    # Select best available importance type for sorting
    sort_by = None
    for preferred_type in ['gain', 'booster_gain', 'importance', 'weight', 'booster_weight']:
        if preferred_type in importance_df.columns:
            sort_by = preferred_type
            break
    
    if sort_by is None:
        sort_by = importance_df.columns[0]
    
    importance_df = importance_df.sort_values(by=sort_by, ascending=False).head(top_n)
    
    # Prepare a more readable index
    importance_df.index = [str(idx).replace('_', ' ').title() for idx in importance_df.index]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    sns.set_style('whitegrid')
    
    # Plot bars
    ax = sns.barplot(x=importance_df[sort_by], y=importance_df.index, palette='viridis')
    plt.title(f'Top {top_n} Features (by {sort_by.replace("_", " ").title()})', fontsize=16)
    plt.xlabel(f'Normalized {sort_by.replace("_", " ").title()}', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    
    # Add values as text
    for i, v in enumerate(importance_df[sort_by]):
        ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=10)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Feature importance plot saved to {save_path}")
    
    return importance_df

# --- Model Training Functions ---
def preprocess_data_for_xgboost(X, y):
    """
    Preprocess data to ensure it's compatible with XGBoost
    
    Args:
        X: Features DataFrame
        y: Target Series
        
    Returns:
        Cleaned X, y
    """
    # Make copies to avoid modifying originals
    X_clean = X.copy()
    y_clean = y.copy()
    
    # Handle infinity values
    X_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Check for and log any remaining infinity values
    inf_counts = np.isinf(X_clean.values).sum()
    if inf_counts > 0:
        logging.warning(f"Found {inf_counts} infinity values even after replacement.")
    
    # Get numeric columns
    numeric_cols = X_clean.select_dtypes(include=np.number).columns
    
    # Count total extreme values and NaNs for reporting
    total_nans = 0
    total_extremes = 0
    cols_with_significant_nans = []
    cols_with_significant_extremes = []
    
    # For each numeric column, calculate stats and cap extreme values
    for col in numeric_cols:
        # Calculate column statistics
        col_values = X_clean[col].dropna()
        if len(col_values) == 0:
            X_clean[col] = 0
            continue
            
        col_mean = col_values.mean()
        col_std = col_values.std()
        col_median = col_values.median()
        
        # Check for NaN values
        nan_count = X_clean[col].isna().sum()
        if nan_count > 0:
            total_nans += nan_count
            # Use median for imputation instead of mean for robustness
            X_clean[col] = X_clean[col].fillna(col_median)
            if nan_count > 50:  # Only log significant NaN counts
                cols_with_significant_nans.append((col, nan_count, col_median))
        
        # Use more conservative outlier detection (6 standard deviations instead of 5)
        # and use IQR method for highly skewed data
        if col_std > 0:
            # Method 1: Standard deviation (for normal-ish distributions)
            lower_bound_std = col_mean - 6 * col_std
            upper_bound_std = col_mean + 6 * col_std
            
            # Method 2: IQR method (for skewed distributions)
            q1 = col_values.quantile(0.25)
            q3 = col_values.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower_bound_iqr = q1 - 3 * iqr  # 3 * IQR instead of 1.5 for more conservative approach
                upper_bound_iqr = q3 + 3 * iqr
            else:
                lower_bound_iqr = lower_bound_std
                upper_bound_iqr = upper_bound_std
            
            # Use the more conservative (wider) bounds
            lower_bound = min(lower_bound_std, lower_bound_iqr)
            upper_bound = max(upper_bound_std, upper_bound_iqr)
            
            # Check for extreme values
            extreme_mask = (X_clean[col] < lower_bound) | (X_clean[col] > upper_bound)
            extreme_count = extreme_mask.sum()
            
            if extreme_count > 0:
                total_extremes += extreme_count
                # Only cap if there are significant outliers (>1% of data or >5 values)
                if extreme_count > max(len(X_clean) * 0.01, 5):
                    X_clean[col] = np.clip(X_clean[col], lower_bound, upper_bound)
                    if extreme_count > 50:  # Only log significant extreme counts
                        cols_with_significant_extremes.append((col, extreme_count, lower_bound, upper_bound))
    
    # Handle NaN values in target if any
    if y_clean.isna().sum() > 0:
        target_nans = y_clean.isna().sum()
        logging.warning(f"Found {target_nans} NaN values in target. Filling with median.")
        y_clean.fillna(y_clean.median(), inplace=True)
    
    # Check for final NaN values
    final_X_nans = X_clean.isna().sum().sum()
    final_y_nans = y_clean.isna().sum()
    
    if final_X_nans > 0:
        logging.warning(f"Still found {final_X_nans} NaN values in features after cleaning.")
        # Last resort: fill remaining NaNs with 0 instead of dropping columns to maintain feature consistency
        nan_cols = X_clean.columns[X_clean.isna().any()].tolist()
        if len(nan_cols) > 10:
            logging.warning(f"Filling remaining NaNs in {len(nan_cols)} columns (showing first 10): {nan_cols[:10]}")
        else:
            logging.warning(f"Filling remaining NaNs in columns: {nan_cols}")
        X_clean.fillna(0, inplace=True)
        
        final_X_nans_after_fill = X_clean.isna().sum().sum()
        if final_X_nans_after_fill > 0:
            logging.error(f"Still have {final_X_nans_after_fill} NaN values after filling with 0")
    
    if final_y_nans > 0:
        logging.error(f"Target still has {final_y_nans} NaN values after cleaning.")
    
    # Check for infinity values again
    if np.any(np.isinf(X_clean.values)) or np.any(np.isinf(y_clean.values)):
        logging.error("Still found infinity values after cleaning.")
        # Replace any remaining infinities
        X_clean.replace([np.inf, -np.inf], 0, inplace=True)
        y_clean.replace([np.inf, -np.inf], y_clean.median(), inplace=True)
    
    # Summary log - only show significant events
    if total_nans > 0 or total_extremes > 0 or cols_with_significant_nans or cols_with_significant_extremes:
        logging.info(f"Data preprocessing: X shape {X_clean.shape}")
        if total_nans > 0:
            logging.info(f"  • Total NaN values imputed: {total_nans}")
        if total_extremes > 0:
            logging.info(f"  • Total extreme values processed: {total_extremes}")
        
        # Log significant NaN imputations
        for col, count, median_val in cols_with_significant_nans:
            logging.info(f"  • Column '{col}': {count} NaNs filled with median ({median_val:.3f})")
        
        # Log significant extreme value capping
        for col, count, lower, upper in cols_with_significant_extremes:
            logging.info(f"  • Column '{col}': {count} extreme values capped to [{lower:.3f}, {upper:.3f}]")
    
    return X_clean, y_clean

def objective_heat_pump_baseload(trial, X, y, cv):
    """
    Enhanced objective function specifically optimized for heat pump baseload demand prediction.
    
    This objective function focuses on:
    1. Better performance during heating season (higher weight)
    2. Accuracy in different temperature regimes
    3. Balanced performance across different times of day
    4. Lower penalty for small errors in low consumption periods
    
    Args:
        trial: Optuna trial object
        X: Features DataFrame
        y: Target Series
        cv: Cross-validation strategy
        
    Returns:
        Weighted RMSE optimized for heat pump applications
    """
    
    # Enhanced hyperparameter space for heat pump applications
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': 42,
        'tree_method': 'hist',
        'early_stopping_rounds': 20,
    }
    
    # Cross-validation with custom scoring
    cv_scores = []
    
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X)):
        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]
        
        # Preprocess data for XGBoost
        X_train_fold, y_train_fold = preprocess_data_for_xgboost(X_train_fold, y_train_fold)
        X_valid_fold, y_valid_fold = preprocess_data_for_xgboost(X_valid_fold, y_valid_fold)
        
        model = xgb.XGBRegressor(**params)
        
        # Fit model with early stopping
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_valid_fold, y_valid_fold)],
            verbose=False
        )
        
        # Make predictions on validation set
        y_pred = model.predict(X_valid_fold)
        
        # Custom weighted scoring for heat pump baseload
        score = calculate_heat_pump_score(y_valid_fold, y_pred, X_valid_fold)
        cv_scores.append(score)
        
        # Log progress
        trial.set_user_attr(f"fold_{fold+1}_heat_pump_score", score)
        if hasattr(model, 'best_iteration'):
            trial.set_user_attr(f"best_iteration_fold_{fold+1}", model.best_iteration)
    
    # Calculate mean score across all folds
    mean_score = np.mean(cv_scores)
    
    # Log additional information about the trial
    trial.set_user_attr("mean_heat_pump_score", mean_score)
    trial.set_user_attr("std_heat_pump_score", np.std(cv_scores))
    
    return mean_score

def calculate_heat_pump_score(y_true, y_pred, X_valid=None):
    """
    Calculate a custom scoring metric optimized for heat pump baseload demand prediction.
    
    Args:
        y_true: True consumption values
        y_pred: Predicted consumption values  
        X_valid: Validation features (optional, for context-aware scoring)
        
    Returns:
        Custom weighted score (lower is better)
    """
    # Base RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    if X_valid is None:
        return rmse
    
    # Try to access temperature and time features for weighted scoring
    weights = np.ones(len(y_true))
    
    try:
        # Higher weight during heating season (when heat pump is working harder)
        if 'temperature_2m' in X_valid.columns:
            temp = X_valid['temperature_2m'].values
            # More weight when temperature is low (heating needed)
            heating_weight = np.where(temp < 10, 2.0, 1.0)  # 2x weight below 10°C
            weights *= heating_weight
            
            # Additional weight for extreme conditions where heat pump efficiency matters most
            extreme_weight = np.where(temp < -5, 1.5, 1.0)  # 1.5x more weight below -5°C
            weights *= extreme_weight
        
        # Higher weight during peak hours when demand prediction is most critical
        if hasattr(X_valid.index, 'hour'):
            hour = X_valid.index.hour
            # Morning and evening peaks
            peak_weight = np.where((hour >= 6) & (hour <= 9) | (hour >= 17) & (hour <= 21), 1.3, 1.0)
            weights *= peak_weight
        elif 'hour_of_day' in X_valid.columns:
            hour = X_valid['hour_of_day'].values
            peak_weight = np.where((hour >= 6) & (hour <= 9) | (hour >= 17) & (hour <= 21), 1.3, 1.0)
            weights *= peak_weight
        
        # Lower penalty for small errors in very low consumption periods (baseload only)
        consumption_percentile_20 = np.percentile(y_true, 20)
        low_consumption_weight = np.where(y_true < consumption_percentile_20, 0.7, 1.0)
        weights *= low_consumption_weight
        
        # Normalize weights to prevent score inflation
        weights = weights / np.mean(weights)
        
    except Exception as e:
        logging.warning(f"Could not apply custom weighting in heat pump score: {e}")
        # Fall back to unweighted RMSE
        return rmse
    
    # Calculate weighted RMSE
    squared_errors = (y_true - y_pred) ** 2
    weighted_mse = np.average(squared_errors, weights=weights)
    weighted_rmse = np.sqrt(weighted_mse)
    
    return weighted_rmse

def train_demand_model(df: pd.DataFrame, target_col_name: str = 'y', n_splits: int = 5, n_trials_optuna: int = 50, 
                       use_proper_splits: bool = True, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15):
    """
    Train an XGBoost model for energy demand prediction with proper temporal splits
    
    Args:
        df: Featured dataframe with target column
        target_col_name: Name of target column
        n_splits: Number of CV splits for hyperparameter optimization
        n_trials_optuna: Number of Optuna trials
        use_proper_splits: Whether to use proper temporal splits (True) or legacy approach (False)
        train_ratio: Proportion for training (default 0.70)
        val_ratio: Proportion for validation (default 0.15) 
        test_ratio: Proportion for test (default 0.15)
        
    Returns:
        model, splits_info
    """
    logging.info("Starting model training with improved temporal splitting")
    
    # Prepare data
    X = df.drop(columns=[target_col_name])
    y = df[target_col_name]
    
    if use_proper_splits:
        # IMPROVED APPROACH: Proper temporal splits with dedicated test set
        logging.info(f"Using proper temporal splits: {train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%}")
        
        # Sort by time
        df_sorted = df.sort_index()
        total_samples = len(df_sorted)
        
        # Account for lag buffer at the beginning (168 hours for weekly patterns)
        lag_buffer = 168
        effective_start_idx = lag_buffer
        effective_samples = total_samples - lag_buffer
        
        # Calculate split indices
        train_size = int(effective_samples * train_ratio)
        val_size = int(effective_samples * val_ratio)
        
        # Split indices (all relative to effective start)
        train_start_idx = effective_start_idx
        train_end_idx = train_start_idx + train_size
        val_start_idx = train_end_idx
        val_end_idx = val_start_idx + val_size
        test_start_idx = val_end_idx
        test_end_idx = total_samples
        
        # Create splits - NO OVERLAP
        train_df = df_sorted.iloc[train_start_idx:train_end_idx].copy()
        val_df = df_sorted.iloc[val_start_idx:val_end_idx].copy()
        test_df = df_sorted.iloc[test_start_idx:test_end_idx].copy()
        
        # Verify no overlap
        assert train_df.index[-1] < val_df.index[0], "Train and validation overlap!"
        assert val_df.index[-1] < test_df.index[0], "Validation and test overlap!"
        
        # Prepare training data
        X_train = train_df.drop(columns=[target_col_name])
        y_train = train_df[target_col_name]
        X_val = val_df.drop(columns=[target_col_name])
        y_val = val_df[target_col_name]
        X_test = test_df.drop(columns=[target_col_name])
        y_test = test_df[target_col_name]
        
        logging.info("✅ Proper temporal splits created:")
        logging.info(f"  Train: {train_df.index[0]} to {train_df.index[-1]} ({len(train_df)} samples, {(train_df.index[-1] - train_df.index[0]).days} days)")
        logging.info(f"  Val:   {val_df.index[0]} to {val_df.index[-1]} ({len(val_df)} samples, {(val_df.index[-1] - val_df.index[0]).days} days)")
        logging.info(f"  Test:  {test_df.index[0]} to {test_df.index[-1]} ({len(test_df)} samples, {(test_df.index[-1] - test_df.index[0]).days} days)")
        
        # Create proper splits info structure
        splits_info = {
            'approach': 'proper_temporal_splits',
            'lag_buffer_hours': lag_buffer,
            'ratios': {'train': train_ratio, 'val': val_ratio, 'test': test_ratio},
            'train': {
                'start_date': train_df.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': train_df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                'n_samples': len(train_df),
                'days': (train_df.index[-1] - train_df.index[0]).days
            },
            'validation': {
                'start_date': val_df.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': val_df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                'n_samples': len(val_df),
                'days': (val_df.index[-1] - val_df.index[0]).days
            },
            'test': {
                'start_date': test_df.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': test_df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                'n_samples': len(test_df),
                'days': (test_df.index[-1] - test_df.index[0]).days
            },
            'full_dataset': {
                'start_date': df.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                'n_samples': len(df)
            }
        }
        
    else:
        # LEGACY APPROACH: For backward compatibility
        logging.warning("Using legacy split approach - only for backward compatibility!")
        X_train, X_val = X.iloc[:-180], X.iloc[-180:]
        y_train, y_val = y.iloc[:-180], y.iloc[-180:]
        X_test, y_test = None, None  # No test set in legacy approach
        
        splits_info = {
            'approach': 'legacy_no_test_set',
            'final_train': {
                'start_date': df.iloc[:-180].index[0].strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': df.iloc[:-180].index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                'n_samples': len(df.iloc[:-180])
            },
            'final_val': {
                'start_date': df.iloc[-180:].index[0].strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': df.iloc[-180:].index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                'n_samples': len(df.iloc[-180:])
            }
        }
    
    # Create TimeSeriesSplit for hyperparameter optimization (only on training data)
    ts_cv = TimeSeriesSplit(n_splits=n_splits)
    cv_splits = list(ts_cv.split(X_train))
    
    logging.info(f"Using {len(cv_splits)} CV folds for hyperparameter optimization on training data")
    
    # Use Optuna for hyperparameter optimization
    logging.info("Starting hyperparameter optimization with Optuna")
    study = optuna.create_study(direction='minimize')
    
    # Improved objective function with robust MAPE
    def objective_improved(trial):
        # Enhanced hyperparameter space for time series prediction
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1500),  # Increased range for better fit
            'max_depth': trial.suggest_int('max_depth', 4, 12),  # Slightly deeper trees
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),  # Lower minimum for stability
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),  # More conservative range
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),  # More conservative range
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),  # L1 regularization
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),  # L2 regularization
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),  # Add min_child_weight
            'gamma': trial.suggest_float('gamma', 0, 0.5),  # Add gamma for complexity control
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'random_state': 42,
            'tree_method': 'hist',
            'grow_policy': 'depthwise'  # Better for time series patterns
        }
        
        cv_scores = []
        for train_idx, val_idx in cv_splits:
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Preprocess data
            X_train_clean, y_train_clean = preprocess_data_for_xgboost(X_train_fold, y_train_fold)
            X_val_clean, y_val_clean = preprocess_data_for_xgboost(X_val_fold, y_val_fold)
            
            # Train model
            model = xgb.XGBRegressor(**params)
            model.fit(X_train_clean, y_train_clean, verbose=False)
            
            # Predict and score
            y_pred = model.predict(X_val_clean)
            
            # Improved scoring with robust MAPE
            rmse = np.sqrt(np.mean((y_val_clean - y_pred) ** 2))
            mae = np.mean(np.abs(y_val_clean - y_pred))
            
            # Robust MAPE calculation (only for values above threshold)
            threshold = 0.1
            mask = np.abs(y_val_clean) > threshold
            if mask.sum() > 0:
                mape_values = np.abs((y_val_clean[mask] - y_pred[mask]) / y_val_clean[mask]) * 100
                robust_mape = np.mean(mape_values)
            else:
                robust_mape = 200  # High penalty if no valid MAPE values
            
            # Combined score (cap MAPE contribution)
            mape_capped = min(robust_mape, 200)
            combined_score = 0.4 * rmse + 0.4 * mae + 0.2 * (mape_capped / 100)
            cv_scores.append(combined_score)
        
        return np.mean(cv_scores)
    
    # Add progress callback if tqdm is available
    if TQDM_AVAILABLE:
        def progress_callback(study, trial):
            tqdm.write(f"Trial {trial.number + 1}/{n_trials_optuna}: Score = {trial.value:.4f}")
        
        with tqdm(total=n_trials_optuna, desc="Hyperparameter optimization", unit="trial") as pbar:
            def callback_with_progress(study, trial):
                progress_callback(study, trial)
                pbar.update(1)
            
            study.optimize(objective_improved, n_trials=n_trials_optuna, callbacks=[callback_with_progress])
    else:
        logging.info(f"Running {n_trials_optuna} optimization trials...")
        study.optimize(objective_improved, n_trials=n_trials_optuna)
    
    # Get best parameters
    best_params = study.best_params
    logging.info(f"Best hyperparameters: {best_params}")
    
    # Add fixed parameters
    best_params.update({
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': 42,
        'tree_method': 'hist',
        'early_stopping_rounds': 20
    })
    
    # Train final model with best parameters
    logging.info("Training final model with best parameters")
    final_model = xgb.XGBRegressor(**best_params)
    
    # Preprocess data for final training
    X_train_clean, y_train_clean = preprocess_data_for_xgboost(X_train, y_train)
    X_val_clean, y_val_clean = preprocess_data_for_xgboost(X_val, y_val)
    
    # Fit the model
    final_model.fit(
        X_train_clean, y_train_clean,
        eval_set=[(X_val_clean, y_val_clean)],
        verbose=False
    )
    
    # Calculate validation metrics
    y_val_pred = final_model.predict(X_val_clean)
    val_rmse = np.sqrt(np.mean((y_val_clean - y_val_pred) ** 2))
    val_mae = np.mean(np.abs(y_val_clean - y_val_pred))
    
    # Robust MAPE calculation
    threshold = 0.1
    mask = np.abs(y_val_clean) > threshold
    if mask.sum() > 0:
        mape_values = np.abs((y_val_clean[mask] - y_val_pred[mask]) / y_val_clean[mask]) * 100
        val_mape = np.mean(mape_values)
    else:
        val_mape = np.inf
    
    logging.info(f"Validation metrics:")
    logging.info(f"  RMSE: {val_rmse:.4f}")
    logging.info(f"  MAE:  {val_mae:.4f}")
    logging.info(f"  Robust MAPE: {val_mape:.2f}%")
    
    # Add validation metrics to splits_info
    splits_info['validation_metrics'] = {
        'rmse': val_rmse,
        'mae': val_mae,
        'mape': val_mape
    }
    
    # Add optimization info
    splits_info['optimization'] = {
        'n_trials': n_trials_optuna,
        'best_params': best_params,
        'best_score': study.best_value
    }
    
    if use_proper_splits:
        logging.info("⚠️  IMPORTANT: Test set is reserved for final evaluation only!")
        logging.info("   Do NOT use test set for model selection or hyperparameter tuning!")
        logging.info(f"   Test set: {len(test_df)} samples from {test_df.index[0]} to {test_df.index[-1]}")
    
    # Save the model and split information
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(final_model, f)
    logging.info(f"Final model saved to {MODEL_SAVE_PATH}")
    
    splits_info_path = MODEL_SAVE_PATH.replace('.pkl', '_splits_info.pkl')
    with open(splits_info_path, 'wb') as f:
        pickle.dump(splits_info, f)
    logging.info(f"Model splits info saved to {splits_info_path}")
    
    # Save feature columns for future use
    feature_columns_path = MODEL_SAVE_PATH.replace('.pkl', '_feature_columns.pkl')
    with open(feature_columns_path, 'wb') as f:
        pickle.dump(list(X_train_clean.columns), f)
    logging.info(f"Model feature columns saved to {feature_columns_path}")
    
    # Plot feature importance
    plot_feature_importance(final_model, X_train_clean.columns, save_path=FEATURE_IMPORTANCE_SAVE_PATH)
    
    return final_model, splits_info

# --- Main Function ---
def main():
    """
    Main function to train the demand prediction model
    """
    # Load and process data
    df = run_data_pipeline(CONSUMPTION_DATA_PATH, HEAT_PUMP_DATA_PATH, WEATHER_DATA_PATH)
    
    # Add HMM features
    df = add_hmm_features(df, TARGET_COL, N_HMM_STATES)
    
    # Feature engineering
    featured_df = engineer_features(df, TARGET_COL, create_target=True)
    
    # Use more trials for better model performance (increased from 5)
    n_trials = 30  # Increased default for better hyperparameter optimization
    
    # Train model
    model, splits_info = train_demand_model(featured_df, n_trials_optuna=n_trials)
    
    # Check if tqdm is available and suggest installation if not
    if not TQDM_AVAILABLE:
        logging.info("💡 Tip: Install tqdm for progress bars: pip install tqdm")
    
    print("Training complete.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train demand prediction model')
    parser.add_argument('--trials', type=int, default=30,
                        help='Number of trials for hyperparameter optimization (increased default)')
    parser.add_argument('--save-path', type=str, default=MODEL_SAVE_PATH,
                        help='Path to save the trained model')
    parser.add_argument('--legacy-splits', action='store_true',
                        help='Use legacy split approach (not recommended)')
    parser.add_argument('--train-ratio', type=float, default=0.70,
                        help='Proportion of data for training (default: 0.70)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Proportion of data for validation (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='Proportion of data for test (default: 0.15)')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.01:
        logging.error("Train, validation, and test ratios must sum to 1.0")
        exit(1)
    
    # Update global variables if needed
    if args.save_path != MODEL_SAVE_PATH:
        MODEL_SAVE_PATH = args.save_path
    
    # Determine split approach
    use_proper_splits = not args.legacy_splits
    
    if use_proper_splits:
        logging.info("🎯 Using PROPER temporal splits (recommended)")
        logging.info(f"   Ratios: Train={args.train_ratio:.0%}, Val={args.val_ratio:.0%}, Test={args.test_ratio:.0%}")
        logging.info("   This creates a dedicated test set for final evaluation")
    else:
        logging.warning("⚠️  Using LEGACY splits (not recommended - no test set!)")
        logging.warning("   This approach has data leakage issues")
    
    # Start the main process
    try:
        # Load and process data
        df = run_data_pipeline(CONSUMPTION_DATA_PATH, HEAT_PUMP_DATA_PATH, WEATHER_DATA_PATH)
        
        # Add HMM features
        df = add_hmm_features(df, TARGET_COL, N_HMM_STATES)
        
        # Feature engineering
        featured_df = engineer_features(df, TARGET_COL, create_target=True)
        
        # Train model with improved splits
        model, splits_info = train_demand_model(
            featured_df, 
            n_trials_optuna=args.trials,
            use_proper_splits=use_proper_splits,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
        
        # Print final summary
        logging.info("="*60)
        logging.info("🎉 TRAINING COMPLETED SUCCESSFULLY")
        logging.info("="*60)
        
        if use_proper_splits:
            logging.info("✅ Model trained with proper temporal splits")
            logging.info(f"✅ Validation RMSE: {splits_info['validation_metrics']['rmse']:.4f}")
            logging.info(f"✅ Validation MAE: {splits_info['validation_metrics']['mae']:.4f}")
            logging.info(f"✅ Validation Robust MAPE: {splits_info['validation_metrics']['mape']:.2f}%")
            logging.info("")
            logging.info("📋 Data splits:")
            logging.info(f"   Train: {splits_info['train']['n_samples']} samples ({splits_info['train']['days']} days)")
            logging.info(f"   Val:   {splits_info['validation']['n_samples']} samples ({splits_info['validation']['days']} days)")
            logging.info(f"   Test:  {splits_info['test']['n_samples']} samples ({splits_info['test']['days']} days)")
            logging.info("")
            logging.info("⚠️  IMPORTANT NOTES:")
            logging.info("   • Test set is RESERVED for final evaluation only")
            logging.info("   • Use evaluate.py with the test set for final performance")
            logging.info("   • Do NOT retrain using test set results")
            
            # Performance advice
            mape = splits_info['validation_metrics']['mape']
            if mape > 50:
                logging.warning("⚠️  High MAPE detected. Consider:")
                logging.warning("   • Increasing --trials for better hyperparameters")
                logging.warning("   • Checking data quality and feature engineering")
                logging.warning("   • Using domain-specific preprocessing")
            elif mape > 30:
                logging.info("💡 Moderate MAPE. Model could benefit from:")
                logging.info("   • More trials for hyperparameter optimization")
                logging.info("   • Additional feature engineering")
        else:
            logging.warning("⚠️  Model trained with legacy splits (no test set)")
            logging.warning("   Consider retraining with --legacy-splits flag removed")
        
        # Check if tqdm is available and suggest installation if not
        if not TQDM_AVAILABLE:
            logging.info("💡 Tip: Install tqdm for progress bars: pip install tqdm")
        
        logging.info(f"📁 Model saved to: {MODEL_SAVE_PATH}")
        logging.info("="*60)
        
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        raise