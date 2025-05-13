import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime, timedelta
import os
import numpy as np
import logging
from pathlib import Path
import argparse
import sys

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

# Hardcoded coordinates from the original script
LATITUDE = 59.295923
LONGITUDE = 18.010528
TIMEZONE = "Europe/Berlin"

archive_url = "https://archive-api.open-meteo.com/v1/archive"
forecast_url = "https://api.open-meteo.com/v1/forecast"

def setup_logging():
    # Ensure logging is configured. This can be called at the start of main().
    # To avoid multiple handlers if the script is imported, check existing handlers.
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def _combine_and_prepare_dataframe(hourly_data_list, daily_data_list):
    """Helper to combine hourly and daily data lists into a single dataframe."""
    combined_hourly_df = pd.DataFrame()
    if hourly_data_list:
        combined_hourly_df = pd.concat(hourly_data_list).drop_duplicates(subset=['date'], keep='first').sort_values('date')

    combined_daily_df = pd.DataFrame()
    if daily_data_list:
        combined_daily_df = pd.concat(daily_data_list).drop_duplicates(subset=['date'], keep='first').sort_values('date')

    if not combined_hourly_df.empty and not combined_daily_df.empty:
        final_df = pd.merge(combined_hourly_df, combined_daily_df, on="date", how="outer")
    elif not combined_hourly_df.empty:
        final_df = combined_hourly_df
    elif not combined_daily_df.empty:
        final_df = combined_daily_df
    else:
        return pd.DataFrame()

    final_df.sort_values('date', inplace=True)
    final_df.ffill(inplace=True) # Forward fill daily values for hourly records

    # Round all numeric columns to 3 decimal places
    numeric_cols = final_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if 'date' not in col.lower(): # Avoid trying to round date/time columns if any
             final_df[col] = pd.to_numeric(final_df[col], errors='ignore').round(3)
    return final_df

def fetch_and_save_specific_forecast(num_days):
    logging.info(f"Fetching {num_days}-day forecast only, aligned to calendar days.")
    
    # Calculate start_date (today) and end_date (today + num_days - 1)
    # Use a consistent "now" for date calculations and filename
    current_script_run_time = datetime.now()
    # start_date will be the calendar day the script is run on
    calendar_start_date = current_script_run_time.replace(hour=0, minute=0, second=0, microsecond=0)
    start_date_str_param = calendar_start_date.strftime("%Y-%m-%d") # For API param
    
    # For API call: request an extra day to ensure we get all hours of the last day
    # For an N-day forecast, we request N+1 days to ensure we have all 24 hours for each day
    calendar_end_date = calendar_start_date + timedelta(days=num_days)  # Request an extra day
    end_date_str_param = calendar_end_date.strftime("%Y-%m-%d") # For API param

    # Date for the filename should reflect the start of the forecast period requested
    filename_date_str = calendar_start_date.strftime('%Y-%m-%d')

    # For filtering: the actual requested calendar day range (00:00 - 23:59 for each day)
    actual_start_datetime = calendar_start_date
    actual_end_datetime = calendar_start_date + timedelta(days=num_days) - timedelta(seconds=1)  # Last second of last day
    
    logging.info(f"Forecast period requested: {start_date_str_param} to {(calendar_start_date + timedelta(days=num_days-1)).strftime('%Y-%m-%d')}")
    logging.info(f"API request date range: {start_date_str_param} to {end_date_str_param} (including extra day to ensure complete data)")

    forecast_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": hourly_vars,
        "daily": daily_vars,
        "timezone": TIMEZONE, # Important for how API interprets start/end date
        "start_date": start_date_str_param,
        "end_date": end_date_str_param
    }
    
    forecast_hourly_data = []
    forecast_daily_data = []

    try:
        logging.info(f"Fetching forecast data from API")
        api_responses = openmeteo.weather_api(forecast_url, params=forecast_params)
        if not api_responses:
            logging.error(f"API call for forecast returned no response.")
            return
        forecast_resp = api_responses[0]
        
        hourly_df = process_hourly_data(forecast_resp.Hourly())
        if not hourly_df.empty:
            forecast_hourly_data.append(hourly_df)
            logging.info(f"Retrieved {len(hourly_df)} hourly records from Forecast API.")
        
        daily_df = process_daily_data(forecast_resp.Daily())
        if not daily_df.empty:
            forecast_daily_data.append(daily_df)
            logging.info(f"Retrieved {len(daily_df)} daily records from Forecast API.")
            
    except Exception as e:
        logging.error(f"Error fetching forecast data: {str(e)}")
        return

    if not forecast_hourly_data and not forecast_daily_data:
        logging.warning(f"No forecast data retrieved. No file will be saved.")
        return

    forecast_dataframe = _combine_and_prepare_dataframe(forecast_hourly_data, forecast_daily_data)

    if forecast_dataframe.empty:
        logging.warning(f"Processed forecast data is empty. No file will be saved.")
        return

    # Now filter the dataframe to keep only the specified calendar days (00:00 to 23:59 of each day)
    # First ensure the date column is datetime
    forecast_dataframe['date'] = pd.to_datetime(forecast_dataframe['date'])
    
    # Get the original row count for logging
    original_rows = len(forecast_dataframe)
    
    # Filter to keep only rows within the actual requested calendar days
    # Note: If forecast_dataframe['date'] is timezone-aware, make actual_start_datetime and actual_end_datetime
    # timezone-aware as well. The API returns UTC dates, so we'll handle that.
    if forecast_dataframe['date'].dt.tz is not None:
        import pytz
        actual_start_datetime = actual_start_datetime.replace(tzinfo=pytz.UTC)
        actual_end_datetime = actual_end_datetime.replace(tzinfo=pytz.UTC)
    
    # Filter: keep rows from actual_start_datetime to actual_end_datetime (inclusive)
    forecast_dataframe = forecast_dataframe[
        (forecast_dataframe['date'] >= actual_start_datetime) &
        (forecast_dataframe['date'] <= actual_end_datetime)
    ].copy()
    
    logging.info(f"Filtered forecast data from {original_rows} to {len(forecast_dataframe)} rows to match requested calendar days exactly.")
    
    # Check if we now have empty data after filtering
    if forecast_dataframe.empty:
        logging.warning(f"After filtering for calendar days, forecast data is empty. No file will be saved.")
        return

    output_dir = project_root / 'data' / 'processed' / 'forecasts' / "weather"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{filename_date_str}_{num_days}days.csv" # Use filename_date_str
    filepath = output_dir / filename
    
    forecast_dataframe.to_csv(filepath, index=False)
    logging.info(f"Successfully saved {num_days}-day forecast to {filepath}")

def fetch_and_update_main_weather_data():
    logging.info("Starting main weather data update process (historical and current forecast).")
    csv_path = project_root / 'data' / 'processed' / 'weather_data.csv'
    
    current_start_date = "2017-01-01" # Default start date
    if os.path.exists(csv_path):
        try:
            existing_data_df = pd.read_csv(csv_path)
            existing_data_df['date'] = pd.to_datetime(existing_data_df['date'])
            last_date = existing_data_df['date'].max()
            current_start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            logging.info(f"Found existing data up to {last_date.strftime('%Y-%m-%d')}. Fetching new data from {current_start_date}.")
        except Exception as e:
            logging.error(f"Error reading existing CSV to determine start date: {str(e)}. Defaulting to {current_start_date}.")
    else:
        logging.info(f"No existing data found at {csv_path}. Starting from {current_start_date}.")

    all_hourly_data = []
    all_daily_data = []

    # Fetch archive data if needed
    if current_start_date < today:
        logging.info(f"Fetching historical data from Archive API ({current_start_date} to {yesterday})")
        archive_params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": current_start_date,
            "end_date": yesterday,
            "hourly": hourly_vars,
            "daily": daily_vars,
            "timezone": TIMEZONE
        }
        try:
            archive_resp_list = openmeteo.weather_api(archive_url, params=archive_params)
            if archive_resp_list:
                archive_resp = archive_resp_list[0]
                hourly_df_archive = process_hourly_data(archive_resp.Hourly())
                if not hourly_df_archive.empty:
                    all_hourly_data.append(hourly_df_archive)
                daily_df_archive = process_daily_data(archive_resp.Daily())
                if not daily_df_archive.empty:
                    all_daily_data.append(daily_df_archive)
            logging.info(f"Archive data fetch complete. Hourly chunks: {len(all_hourly_data)}, Daily chunks: {len(all_daily_data)}")
        except Exception as e:
            logging.error(f"Error fetching archive data: {str(e)}")

    # Fetch current forecast data (standard, e.g., next 7-16 days)
    logging.info(f"Fetching standard forecast data from Forecast API (today and future for main update)")
    forecast_params_main = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": hourly_vars,
        "daily": daily_vars,
        "timezone": TIMEZONE
        # No "forecast_days" here to get default (e.g., 7 or 16 days)
    }
    try:
        forecast_resp_list = openmeteo.weather_api(forecast_url, params=forecast_params_main)
        if forecast_resp_list:
            forecast_resp_main = forecast_resp_list[0]
            hourly_df_forecast = process_hourly_data(forecast_resp_main.Hourly())
            if not hourly_df_forecast.empty:
                all_hourly_data.append(hourly_df_forecast)
            daily_df_forecast = process_daily_data(forecast_resp_main.Daily())
            if not daily_df_forecast.empty:
                all_daily_data.append(daily_df_forecast)
        logging.info(f"Standard forecast data fetch complete. Total hourly chunks: {len(all_hourly_data)}, Total daily chunks: {len(all_daily_data)}")
    except Exception as e:
        logging.error(f"Error fetching standard forecast data: {str(e)}")

    if not all_hourly_data and not all_daily_data:
        logging.warning("No new data (archive or forecast) was collected. No updates will be made to main CSV.")
        return

    newly_fetched_dataframe = _combine_and_prepare_dataframe(all_hourly_data, all_daily_data)

    if newly_fetched_dataframe.empty:
        logging.warning("Processed newly fetched data is empty. No updates will be made to main CSV.")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path):
        try:
            existing_data_df = pd.read_csv(csv_path)
            existing_data_df['date'] = pd.to_datetime(existing_data_df['date'])
            newly_fetched_dataframe['date'] = pd.to_datetime(newly_fetched_dataframe['date'])
            
            # Combine, drop duplicates (keeping existing data for overlaps), sort
            full_data = pd.concat([existing_data_df, newly_fetched_dataframe])
            full_data.drop_duplicates(subset=['date'], keep='first', inplace=True)
            full_data.sort_values('date', inplace=True)
            
            # Re-apply rounding after concat as types might change or NaNs introduced
            numeric_cols_full = full_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols_full:
                 if 'date' not in col.lower():
                    full_data[col] = pd.to_numeric(full_data[col], errors='coerce').round(3)

            full_data.to_csv(csv_path, index=False)
            logging.info(f"Successfully updated weather data at {csv_path}. New data from {current_start_date} was processed.")
        except Exception as e:
            logging.error(f"Error merging new data with existing data at {csv_path}: {str(e)}")
    else:
        newly_fetched_dataframe.to_csv(csv_path, index=False)
        logging.info(f"Successfully created new weather data file at {csv_path} with data from {current_start_date}.")
    
    logging.info(f"Main weather data update process completed. Range processed started from: {current_start_date} up to latest forecast.")

def main():
    setup_logging() # Call logging setup
    parser = argparse.ArgumentParser(description="Fetch weather data from Open-Meteo.")
    parser.add_argument(
        "--forecast",
        action="store_true",
        help="Fetch only forecast data for a specified number of days and save to a separate file."
    )
    parser.add_argument(
        "--days",
        type=int,
        metavar="N",
        help="Number of days for the forecast (1-16). Required and used only if --forecast is specified."
    )
    args = parser.parse_args()

    if args.forecast:
        if args.days is None:
            parser.error("--days is required when --forecast is specified.")
        if not (1 <= args.days <= 16): # Open-Meteo free tier typically allows up to 16 days
            parser.error("--days must be between 1 and 16.")
        
        fetch_and_save_specific_forecast(args.days)
    else:
        fetch_and_update_main_weather_data()

if __name__ == "__main__":
    main()
