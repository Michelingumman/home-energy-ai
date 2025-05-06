import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime, timedelta
import os
import numpy as np
import logging
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Get the last date from the existing CSV file or use a default start date
csv_path = project_root / 'data/processed/weather_data.csv'
today = datetime.now().strftime('%Y-%m-%d')
yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

if os.path.exists(csv_path):
    try:
        existing_data = pd.read_csv(csv_path)
        existing_data['date'] = pd.to_datetime(existing_data['date'])
        last_date = existing_data['date'].max()
        start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        logging.info(f"Found existing data up to {last_date.strftime('%Y-%m-%d')}. Starting from {start_date}.")
    except Exception as e:
        logging.error(f"Error reading existing CSV: {str(e)}")
        start_date = "2017-01-01"
else:
    logging.info("No existing data found. Starting from 2017-01-01.")
    start_date = "2017-01-01"

# Define variable names
hourly_vars = ["temperature_2m", "cloud_cover", "relative_humidity_2m", "wind_speed_100m", "wind_direction_100m"]
daily_vars = ["shortwave_radiation_sum"]

def process_hourly_data(hourly):
    """Process hourly data from the API response and convert to dataframe"""
    if not hourly:
        return pd.DataFrame()
    
    # Generate the correct time array
    hourly_time = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )
    
    # Create dataframe with the time array
    df = pd.DataFrame({"date": hourly_time})
    
    # Add each hourly variable to the dataframe
    for i, name in enumerate(hourly_vars):
        if i < hourly.VariablesLength():
            values = hourly.Variables(i).ValuesAsNumpy()
            if len(values) == len(hourly_time):
                df[name] = values.round(3)
            else:
                logging.warning(f"Length mismatch for {name}: values={len(values)}, dates={len(hourly_time)}. Attempting to fix...")
                # Try to fix length mismatch by truncating or padding
                if len(values) > len(hourly_time):
                    df[name] = values[:len(hourly_time)].round(3)
                else:
                    # Pad with NaN and warn
                    padded_values = np.pad(values, (0, len(hourly_time) - len(values)), 
                                          constant_values=np.nan)
                    df[name] = padded_values.round(3)
    
    return df

def process_daily_data(daily):
    """Process daily data from the API response and convert to dataframe"""
    if not daily:
        return pd.DataFrame()
    
    # Generate the correct time array
    daily_time = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )
    
    # Create dataframe with the time array
    df = pd.DataFrame({"date": daily_time})
    
    # Add each daily variable to the dataframe
    for i, name in enumerate(daily_vars):
        if i < daily.VariablesLength():
            values = daily.Variables(i).ValuesAsNumpy()
            if len(values) == len(daily_time):
                df[name] = values.round(3)
            else:
                logging.warning(f"Length mismatch for {name}: values={len(values)}, dates={len(daily_time)}. Attempting to fix...")
                # Try to fix length mismatch by truncating or padding
                if len(values) > len(daily_time):
                    df[name] = values[:len(daily_time)].round(3)
                else:
                    # Pad with NaN and warn
                    padded_values = np.pad(values, (0, len(daily_time) - len(values)), 
                                          constant_values=np.nan)
                    df[name] = padded_values.round(3)
    
    return df

# Initialize dataframes for collection
all_hourly_data = []
all_daily_data = []

# Only fetch archive data if we need historical data (not just today)
if start_date < today:
    logging.info(f"Fetching historical data from Archive API ({start_date} to {yesterday})")
    archive_url = "https://archive-api.open-meteo.com/v1/archive"
    try:
        archive_params = {
            "latitude": 59.295923,
            "longitude": 18.010528,
            "start_date": start_date,
            "end_date": yesterday,  # Archive API only up to yesterday
            "hourly": hourly_vars,
            "daily": daily_vars,
            "timezone": "Europe/Berlin"
        }
        
        archive_resp = openmeteo.weather_api(archive_url, params=archive_params)[0]
        
        # Process hourly archive data
        hourly_df = process_hourly_data(archive_resp.Hourly())
        if not hourly_df.empty:
            all_hourly_data.append(hourly_df)
            logging.info(f"Retrieved {len(hourly_df)} hourly records from Archive API")
        
        # Process daily archive data
        daily_df = process_daily_data(archive_resp.Daily())
        if not daily_df.empty:
            all_daily_data.append(daily_df)
            logging.info(f"Retrieved {len(daily_df)} daily records from Archive API")
    except Exception as e:
        logging.error(f"Error fetching archive data: {str(e)}")

# Always fetch forecast data for the most current data
logging.info(f"Fetching forecast data from Forecast API (today and future)")
forecast_url = "https://api.open-meteo.com/v1/forecast"
try:
    forecast_params = {
        "latitude": 59.295923,
        "longitude": 18.010528,
        "hourly": hourly_vars,
        "daily": daily_vars,
        "timezone": "Europe/Berlin"
    }
    
    forecast_resp = openmeteo.weather_api(forecast_url, params=forecast_params)[0]
    
    # Process hourly forecast data
    hourly_df = process_hourly_data(forecast_resp.Hourly())
    if not hourly_df.empty:
        all_hourly_data.append(hourly_df)
        logging.info(f"Retrieved {len(hourly_df)} hourly records from Forecast API")
    
    # Process daily forecast data
    daily_df = process_daily_data(forecast_resp.Daily())
    if not daily_df.empty:
        all_daily_data.append(daily_df)
        logging.info(f"Retrieved {len(daily_df)} daily records from Forecast API")
except Exception as e:
    logging.error(f"Error fetching forecast data: {str(e)}")

# Combine all collected data
combined_hourly = pd.DataFrame()
if all_hourly_data:
    combined_hourly = pd.concat(all_hourly_data).drop_duplicates(subset=['date'], keep='first').sort_values('date')
    logging.info(f"Combined hourly data: {len(combined_hourly)} records")

combined_daily = pd.DataFrame()
if all_daily_data:
    combined_daily = pd.concat(all_daily_data).drop_duplicates(subset=['date'], keep='first').sort_values('date')
    logging.info(f"Combined daily data: {len(combined_daily)} records")

# Final merge of hourly and daily data
if not combined_hourly.empty and not combined_daily.empty:
    weather_dataframe = pd.merge(combined_hourly, combined_daily, on="date", how="outer")
    weather_dataframe.sort_values('date', inplace=True)
    
    # Fill forward daily values for hourly data points
    weather_dataframe.ffill(inplace=True)
    
    # Handle missing directory
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # If there's existing data, merge it with the new data
    if os.path.exists(csv_path):
        try:
            existing_data = pd.read_csv(csv_path)
            existing_data['date'] = pd.to_datetime(existing_data['date'])
            weather_dataframe['date'] = pd.to_datetime(weather_dataframe['date'])
            
            # Combine existing and new data, removing any duplicates
            # Use 'first' to keep the actual data (from archive) over forecast data
            full_data = pd.concat([existing_data, weather_dataframe])
            full_data = full_data.drop_duplicates(subset=['date'], keep='first')
            full_data = full_data.sort_values('date')
            
            # Round all numeric columns to 3 decimal places
            numeric_cols = full_data.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                if col != 'date':
                    full_data[col] = full_data[col].round(3)
            
            # Save the combined data
            full_data.to_csv(csv_path, index=False)
            logging.info(f"Updated weather data from {start_date} to {today}")
        except Exception as e:
            logging.error(f"Error merging with existing data: {str(e)}")
    else:
        # Round all numeric columns to 3 decimal places
        numeric_cols = weather_dataframe.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if col != 'date':
                weather_dataframe[col] = weather_dataframe[col].round(3)
        
        # If no existing data, just save the new data
        weather_dataframe.to_csv(csv_path, index=False)
        logging.info(f"Created new weather data file from {start_date} to {today}")
elif len(all_hourly_data) > 0:
    logging.warning("No daily data was collected, proceeding with hourly data only.")
    weather_dataframe = combined_hourly
    # Save logic as above...
    if os.path.exists(csv_path):
        # Same merging logic as above
        try:
            existing_data = pd.read_csv(csv_path)
            existing_data['date'] = pd.to_datetime(existing_data['date'])
            weather_dataframe['date'] = pd.to_datetime(weather_dataframe['date'])
            
            full_data = pd.concat([existing_data, weather_dataframe])
            full_data = full_data.drop_duplicates(subset=['date'], keep='first')
            full_data = full_data.sort_values('date')
            
            numeric_cols = full_data.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                if col != 'date':
                    full_data[col] = full_data[col].round(3)
            
            full_data.to_csv(csv_path, index=False)
            logging.info(f"Updated weather data with hourly data only from {start_date} to {today}")
        except Exception as e:
            logging.error(f"Error merging with existing data: {str(e)}")
    else:
        weather_dataframe.to_csv(csv_path, index=False)
        logging.info(f"Created new weather data file with hourly data only from {start_date} to {today}")
elif len(all_daily_data) > 0:
    logging.warning("No hourly data was collected, proceeding with daily data only.")
    weather_dataframe = combined_daily
    # Similar save logic
    if os.path.exists(csv_path):
        try:
            existing_data = pd.read_csv(csv_path)
            existing_data['date'] = pd.to_datetime(existing_data['date'])
            weather_dataframe['date'] = pd.to_datetime(weather_dataframe['date'])
            
            full_data = pd.concat([existing_data, weather_dataframe])
            full_data = full_data.drop_duplicates(subset=['date'], keep='first')
            full_data = full_data.sort_values('date')
            
            numeric_cols = full_data.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                if col != 'date':
                    full_data[col] = full_data[col].round(3)
            
            full_data.to_csv(csv_path, index=False)
            logging.info(f"Updated weather data with daily data only from {start_date} to {today}")
        except Exception as e:
            logging.error(f"Error merging with existing data: {str(e)}")
    else:
        weather_dataframe.to_csv(csv_path, index=False)
        logging.info(f"Created new weather data file with daily data only from {start_date} to {today}")
else:
    logging.warning("No weather data was collected. No updates made to CSV.")

print(f"Data update process completed. Range: {start_date} to {today}")
