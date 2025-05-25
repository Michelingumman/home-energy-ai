"""
This script fetches weather data from the Open-Meteo API and saves it to a CSV file.

The script is designed to be run daily to update the weather data.

Usage: 

- python FetchWeatherData.py --forecast --days N 
- python FetchWeatherData.py # This will update the weather data and store in data/processed/weather_data.csv

predictions is saved to `data/processed/forecasts/weather/`
"""

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

project_root = Path(__file__).resolve().parents[3]

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('src/predictions/demand/logs/weather_data_fetch.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

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
        logger.info(f"EXISTING DATA: Found existing data up to {last_date.strftime('%Y-%m-%d')}. Starting from {start_date}")
    except Exception as e:
        logger.error(f"FILE ERROR: Error reading existing CSV: {str(e)}")
        start_date = "2017-01-01"
else:
    logger.info("NO EXISTING DATA: Starting from 2017-01-01")
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

def validate_weather_data_quality(df, data_type="weather"):
    """
    Comprehensive data quality validation for weather data.
    
    Args:
        df: DataFrame to validate
        data_type: Type of data for logging context
    
    Returns:
        dict: Quality metrics and status
    """
    logger.info(f"Starting data quality validation for {data_type}")
    
    if df.empty:
        logger.error(f"CRITICAL: {data_type} DataFrame is empty!")
        return {"status": "FAILED", "reason": "Empty DataFrame"}
    
    quality_report = {
        "status": "PASSED",
        "total_records": len(df),
        "date_range": None,
        "duplicates": 0,
        "missing_values": {},
        "gaps": [],
        "outliers": {},
        "issues": [],
        "weather_columns": [],
        "problematic_records": {}
    }
    
    # Basic info
    logger.info(f"{data_type} - Total records: {len(df)}")
    
    # Check for weather columns
    expected_weather_cols = hourly_vars + daily_vars
    found_weather_cols = [col for col in expected_weather_cols if col in df.columns]
    quality_report["weather_columns"] = found_weather_cols
    
    if not found_weather_cols:
        logger.error(f"CRITICAL: {data_type} - No weather data columns found!")
        quality_report["status"] = "FAILED"
        quality_report["issues"].append("No weather data columns found")
        return quality_report
    else:
        logger.info(f"COLUMNS FOUND: {data_type} - Found weather columns: {found_weather_cols}")
    
    # Date range analysis
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'])
            start_date = df['date'].min()
            end_date = df['date'].max()
            quality_report["date_range"] = f"{start_date} to {end_date}"
            logger.info(f"{data_type} - Date range: {start_date} to {end_date}")
            
            # Gap detection for hourly data
            if len(df) > 1:
                expected_freq = pd.infer_freq(df['date'])
                if expected_freq:
                    logger.info(f"FREQUENCY DETECTED: {data_type} - Detected frequency: {expected_freq}")
                    # Create expected date range
                    if 'H' in expected_freq:  # Hourly data
                        expected_range = pd.date_range(start=start_date, end=end_date, freq='H')
                    elif 'D' in expected_freq:  # Daily data
                        expected_range = pd.date_range(start=start_date, end=end_date, freq='D')
                    else:
                        expected_range = None
                    
                    if expected_range is not None:
                        missing_dates = expected_range.difference(df['date'])
                        if len(missing_dates) > 0:
                            quality_report["gaps"] = [f"{len(missing_dates)} missing time periods"]
                            logger.warning(f"GAP WARNING: {data_type} - Found {len(missing_dates)} missing time periods")
                            quality_report["issues"].append(f"Missing {len(missing_dates)} time periods")
                            
                            # Show specific missing dates (sample)
                            sample_missing = missing_dates[:20] if len(missing_dates) > 20 else missing_dates
                            logger.warning(f"MISSING TIMESTAMPS: Sample missing dates: {sample_missing.tolist()}")
                            if len(missing_dates) > 20:
                                logger.warning(f"... and {len(missing_dates) - 20} more missing timestamps")
                            quality_report["problematic_records"]["missing_timestamps"] = missing_dates[:50].tolist()
                        else:
                            logger.info(f"NO GAPS: {data_type} - No gaps detected in time series")
                            
        except Exception as e:
            logger.error(f"DATE ERROR: {data_type} - Error processing dates: {e}")
            quality_report["issues"].append(f"Date processing error: {e}")
    
    # Duplicate detection
    if 'date' in df.columns:
        duplicates = df.duplicated(subset=['date']).sum()
        quality_report["duplicates"] = duplicates
        if duplicates > 0:
            duplicate_mask = df.duplicated(subset=['date'], keep=False)
            duplicate_records = df[duplicate_mask]
            logger.warning(f"DUPLICATE WARNING: {data_type} - Found {duplicates} duplicate timestamps")
            quality_report["issues"].append(f"{duplicates} duplicate timestamps")
            
            # Show specific duplicate timestamps
            duplicate_timestamps = duplicate_records['date'].tolist()
            sample_duplicates = duplicate_timestamps[:20] if len(duplicate_timestamps) > 20 else duplicate_timestamps
            logger.warning(f"DUPLICATE TIMESTAMPS: Sample duplicates: {sample_duplicates}")
            if len(duplicate_timestamps) > 20:
                logger.warning(f"... and {len(duplicate_timestamps) - 20} more duplicate timestamps")
            quality_report["problematic_records"]["duplicate_timestamps"] = duplicate_timestamps[:50]
        else:
            logger.info(f"NO DUPLICATES: {data_type} - No duplicate timestamps found")
    
    # Missing values analysis for weather columns
    missing_summary = df[found_weather_cols].isnull().sum()
    missing_dict = missing_summary[missing_summary > 0].to_dict()
    quality_report["missing_values"] = missing_dict
    
    if missing_dict:
        logger.warning(f"MISSING VALUES WARNING: {data_type} - Missing values detected:")
        for col, count in missing_dict.items():
            pct = (count / len(df)) * 100
            logger.warning(f"   {col}: {count} missing ({pct:.1f}%)")
            
            # Show specific timestamps with missing values
            missing_mask = df[col].isnull()
            missing_timestamps = df[missing_mask]['date'].tolist()
            sample_missing = missing_timestamps[:10] if len(missing_timestamps) > 10 else missing_timestamps
            logger.warning(f"   MISSING VALUE TIMESTAMPS for {col}: {sample_missing}")
            if len(missing_timestamps) > 10:
                logger.warning(f"   ... and {len(missing_timestamps) - 10} more missing value timestamps")
            quality_report["problematic_records"][f"missing_{col}"] = missing_timestamps[:30]
            
            if pct > 15:  # More than 15% missing is concerning for weather data
                quality_report["issues"].append(f"{col} has {pct:.1f}% missing values")
    else:
        logger.info(f"NO MISSING VALUES: {data_type} - No missing values found in weather columns")
    
    # Value validation for weather columns
    for col in found_weather_cols:
        if col in df.columns and not df[col].empty:
            # Check for extreme outliers based on weather variable type
            if 'temperature' in col:
                # Temperature should be reasonable (-50 to 50°C)
                extreme_mask = (df[col] < -50) | (df[col] > 50)
                extreme_count = extreme_mask.sum()
                if extreme_count > 0:
                    logger.warning(f"EXTREME VALUES WARNING: {data_type} - {col}: {extreme_count} extreme temperature values")
                    quality_report["issues"].append(f"{col} has {extreme_count} extreme values")
                    
                    # Show specific extreme temperature records
                    extreme_records = df[extreme_mask][['date', col]]
                    sample_extreme = extreme_records.head(10)
                    logger.warning(f"EXTREME TEMPERATURE RECORDS for {col}:")
                    for _, row in sample_extreme.iterrows():
                        logger.warning(f"   {row['date']}: {row[col]:.2f}°C")
                    if len(extreme_records) > 10:
                        logger.warning(f"   ... and {len(extreme_records) - 10} more extreme temperature records")
                    
                    quality_report["problematic_records"][f"extreme_{col}"] = [
                        {"timestamp": str(row['date']), "value": float(row[col])} 
                        for _, row in extreme_records.head(20).iterrows()
                    ]
                    
            elif 'humidity' in col:
                # Humidity should be 0-100%
                out_of_range_mask = (df[col] < 0) | (df[col] > 100)
                out_of_range = out_of_range_mask.sum()
                if out_of_range > 0:
                    logger.warning(f"OUT OF RANGE WARNING: {data_type} - {col}: {out_of_range} out of range values")
                    quality_report["issues"].append(f"{col} has {out_of_range} out of range values")
                    
                    # Show specific out of range humidity records
                    out_of_range_records = df[out_of_range_mask][['date', col]]
                    sample_out_of_range = out_of_range_records.head(10)
                    logger.warning(f"OUT OF RANGE HUMIDITY RECORDS for {col}:")
                    for _, row in sample_out_of_range.iterrows():
                        logger.warning(f"   {row['date']}: {row[col]:.2f}%")
                    if len(out_of_range_records) > 10:
                        logger.warning(f"   ... and {len(out_of_range_records) - 10} more out of range humidity records")
                    
                    quality_report["problematic_records"][f"out_of_range_{col}"] = [
                        {"timestamp": str(row['date']), "value": float(row[col])} 
                        for _, row in out_of_range_records.head(20).iterrows()
                    ]
                    
            elif 'wind_speed' in col:
                # Wind speed should be non-negative and reasonable
                negative_mask = df[col] < 0
                negative_count = negative_mask.sum()
                extreme_mask = df[col] > 200  # >200 m/s is extreme
                extreme_count = extreme_mask.sum()
                
                if negative_count > 0:
                    logger.warning(f"NEGATIVE VALUES WARNING: {data_type} - {col}: {negative_count} negative values")
                    quality_report["issues"].append(f"{col} has {negative_count} negative values")
                    
                    # Show specific negative wind speed records
                    negative_records = df[negative_mask][['date', col]]
                    sample_negative = negative_records.head(10)
                    logger.warning(f"NEGATIVE WIND SPEED RECORDS for {col}:")
                    for _, row in sample_negative.iterrows():
                        logger.warning(f"   {row['date']}: {row[col]:.2f} m/s")
                    if len(negative_records) > 10:
                        logger.warning(f"   ... and {len(negative_records) - 10} more negative wind speed records")
                    
                    quality_report["problematic_records"][f"negative_{col}"] = [
                        {"timestamp": str(row['date']), "value": float(row[col])} 
                        for _, row in negative_records.head(20).iterrows()
                    ]
                    
                if extreme_count > 0:
                    logger.warning(f"EXTREME VALUES WARNING: {data_type} - {col}: {extreme_count} extreme values")
                    quality_report["issues"].append(f"{col} has {extreme_count} extreme values")
                    
                    # Show specific extreme wind speed records
                    extreme_records = df[extreme_mask][['date', col]]
                    sample_extreme = extreme_records.head(10)
                    logger.warning(f"EXTREME WIND SPEED RECORDS for {col}:")
                    for _, row in sample_extreme.iterrows():
                        logger.warning(f"   {row['date']}: {row[col]:.2f} m/s")
                    if len(extreme_records) > 10:
                        logger.warning(f"   ... and {len(extreme_records) - 10} more extreme wind speed records")
                    
                    quality_report["problematic_records"][f"extreme_wind_{col}"] = [
                        {"timestamp": str(row['date']), "value": float(row[col])} 
                        for _, row in extreme_records.head(20).iterrows()
                    ]
            
            # General outlier detection using IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                outlier_pct = (outlier_count / len(df)) * 100
                quality_report["outliers"][col] = outlier_count
                
                # Show statistics
                logger.info(f"OUTLIER STATS for {col}: Q1={Q1:.3f}, Q3={Q3:.3f}, IQR={IQR:.3f}")
                logger.info(f"OUTLIER BOUNDS for {col}: Lower={lower_bound:.3f}, Upper={upper_bound:.3f}")
                
                if outlier_pct > 8:  # More than 8% outliers is concerning for weather data
                    logger.warning(f"OUTLIER WARNING: {data_type} - {col}: {outlier_count} outliers ({outlier_pct:.1f}%)")
                    quality_report["issues"].append(f"{col} has {outlier_pct:.1f}% outliers")
                    
                    # Show specific outlier records
                    outlier_records = df[outlier_mask][['date', col]].sort_values(col)
                    
                    # Show most extreme outliers
                    extreme_low = outlier_records.head(5)
                    extreme_high = outlier_records.tail(5)
                    
                    logger.warning(f"EXTREME LOW OUTLIERS for {col}:")
                    for _, row in extreme_low.iterrows():
                        logger.warning(f"   {row['date']}: {row[col]:.3f}")
                    
                    logger.warning(f"EXTREME HIGH OUTLIERS for {col}:")
                    for _, row in extreme_high.iterrows():
                        logger.warning(f"   {row['date']}: {row[col]:.3f}")
                    
                    if outlier_count > 10:
                        logger.warning(f"   ... and {outlier_count - 10} more outlier records")
                    
                    quality_report["problematic_records"][f"outliers_{col}"] = [
                        {"timestamp": str(row['date']), "value": float(row[col])} 
                        for _, row in outlier_records.head(10).iterrows()
                    ] + [
                        {"timestamp": str(row['date']), "value": float(row[col])} 
                        for _, row in outlier_records.tail(10).iterrows()
                    ]
                else:
                    logger.info(f"OUTLIER INFO: {data_type} - {col}: {outlier_count} outliers ({outlier_pct:.1f}%) - within acceptable range")
                    
                    # Still log a few examples for reference
                    outlier_records = df[outlier_mask][['date', col]].head(5)
                    sample_outliers = [(str(row['date']), float(row[col])) for _, row in outlier_records.iterrows()]
                    logger.info(f"SAMPLE OUTLIERS for {col}: {sample_outliers}")
    
    # Final status determination
    if quality_report["issues"]:
        if any("FAILED" in str(issue) or "No weather" in str(issue) for issue in quality_report["issues"]):
            quality_report["status"] = "FAILED"
        else:
            quality_report["status"] = "WARNING"
        
        logger.warning(f"QUALITY SUMMARY: {data_type} - Data quality issues found: {len(quality_report['issues'])}")
        for issue in quality_report["issues"]:
            logger.warning(f"   Issue: {issue}")
    else:
        logger.info(f"QUALITY PASSED: {data_type} - All data quality checks passed")
    
    return quality_report

def process_hourly_data(hourly):
    """Process hourly data from the API response and convert to dataframe"""
    if not hourly:
        logger.warning("API WARNING: No hourly data received from API")
        return pd.DataFrame()
    
    logger.info("PROCESSING: Processing hourly weather data from API")
    
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
    processing_errors = 0
    for i, name in enumerate(hourly_vars):
        if i < hourly.VariablesLength():
            values = hourly.Variables(i).ValuesAsNumpy()
            if len(values) == len(hourly_time):
                df[name] = values.round(3)
                logger.info(f"   Added {name}: {len(values)} values")
            else:
                logger.warning(f"LENGTH MISMATCH: {name}: values={len(values)}, dates={len(hourly_time)}. Attempting to fix...")
                processing_errors += 1
                # Try to fix length mismatch by truncating or padding
                if len(values) > len(hourly_time):
                    df[name] = values[:len(hourly_time)].round(3)
                else:
                    # Pad with NaN and warn
                    padded_values = np.pad(values, (0, len(hourly_time) - len(values)), 
                                          constant_values=np.nan)
                    df[name] = padded_values.round(3)
    
    if processing_errors > 0:
        logger.warning(f"PROCESSING WARNING: Encountered {processing_errors} length mismatches in hourly data")
    
    logger.info(f"HOURLY DATA: Successfully processed {len(df)} hourly records")
    return df

def process_daily_data(daily):
    """Process daily data from the API response and convert to dataframe"""
    if not daily:
        logger.warning("API WARNING: No daily data received from API")
        return pd.DataFrame()
    
    logger.info("PROCESSING: Processing daily weather data from API")
    
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
    processing_errors = 0
    for i, name in enumerate(daily_vars):
        if i < daily.VariablesLength():
            values = daily.Variables(i).ValuesAsNumpy()
            if len(values) == len(daily_time):
                df[name] = values.round(3)
                logger.info(f"   Added {name}: {len(values)} values")
            else:
                logger.warning(f"LENGTH MISMATCH: {name}: values={len(values)}, dates={len(daily_time)}. Attempting to fix...")
                processing_errors += 1
                # Try to fix length mismatch by truncating or padding
                if len(values) > len(daily_time):
                    df[name] = values[:len(daily_time)].round(3)
                else:
                    # Pad with NaN and warn
                    padded_values = np.pad(values, (0, len(daily_time) - len(values)), 
                                          constant_values=np.nan)
                    df[name] = padded_values.round(3)
    
    if processing_errors > 0:
        logger.warning(f"PROCESSING WARNING: Encountered {processing_errors} length mismatches in daily data")
    
    logger.info(f"DAILY DATA: Successfully processed {len(df)} daily records")
    return df

def setup_logging():
    # Ensure logging is configured. This can be called at the start of main().
    # To avoid multiple handlers if the script is imported, check existing handlers.
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _combine_and_prepare_dataframe(hourly_data_list, daily_data_list):
    """Helper to combine hourly and daily data lists into a single dataframe."""
    logger.info("COMBINING: Combining hourly and daily weather data")
    
    combined_hourly_df = pd.DataFrame()
    if hourly_data_list:
        combined_hourly_df = pd.concat(hourly_data_list).drop_duplicates(subset=['date'], keep='first').sort_values('date')
        logger.info(f"   Combined {len(hourly_data_list)} hourly chunks into {len(combined_hourly_df)} records")

    combined_daily_df = pd.DataFrame()
    if daily_data_list:
        combined_daily_df = pd.concat(daily_data_list).drop_duplicates(subset=['date'], keep='first').sort_values('date')
        logger.info(f"   Combined {len(daily_data_list)} daily chunks into {len(combined_daily_df)} records")

    if not combined_hourly_df.empty and not combined_daily_df.empty:
        final_df = pd.merge(combined_hourly_df, combined_daily_df, on="date", how="outer")
        logger.info("   Merged hourly and daily data using outer join")
    elif not combined_hourly_df.empty:
        final_df = combined_hourly_df
        logger.info("   Using hourly data only")
    elif not combined_daily_df.empty:
        final_df = combined_daily_df
        logger.info("   Using daily data only")
    else:
        logger.warning("COMBINE WARNING: No data available to combine")
        return pd.DataFrame()

    final_df.sort_values('date', inplace=True)
    
    # Forward fill daily values for hourly records
    missing_before = final_df.isnull().sum().sum()
    final_df.ffill(inplace=True)
    missing_after = final_df.isnull().sum().sum()
    filled_count = missing_before - missing_after
    
    if filled_count > 0:
        logger.info(f"   Forward filled {filled_count} missing values")

    # Round all numeric columns to 3 decimal places
    numeric_cols = final_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if 'date' not in col.lower(): # Avoid trying to round date/time columns if any
             final_df[col] = pd.to_numeric(final_df[col], errors='ignore').round(3)
    
    logger.info(f"COMBINE SUCCESS: Final combined dataset has {len(final_df)} records")
    return final_df

def fetch_and_save_specific_forecast(num_days):
    logger.info(f"FORECAST START: Fetching {num_days}-day forecast only, aligned to calendar days")
    
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
    
    logger.info(f"FORECAST PERIOD: Requested period: {start_date_str_param} to {(calendar_start_date + timedelta(days=num_days-1)).strftime('%Y-%m-%d')}")
    logger.info(f"API REQUEST: Date range: {start_date_str_param} to {end_date_str_param} (including extra day to ensure complete data)")

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
        logger.info("API CALL: Fetching forecast data from API")
        api_responses = openmeteo.weather_api(forecast_url, params=forecast_params)
        if not api_responses:
            logger.error("API ERROR: API call for forecast returned no response")
            return
        forecast_resp = api_responses[0]
        
        hourly_df = process_hourly_data(forecast_resp.Hourly())
        if not hourly_df.empty:
            forecast_hourly_data.append(hourly_df)
            logger.info(f"API SUCCESS: Retrieved {len(hourly_df)} hourly records from Forecast API")
        
        daily_df = process_daily_data(forecast_resp.Daily())
        if not daily_df.empty:
            forecast_daily_data.append(daily_df)
            logger.info(f"API SUCCESS: Retrieved {len(daily_df)} daily records from Forecast API")
            
    except Exception as e:
        logger.error(f"API ERROR: Error fetching forecast data: {str(e)}")
        return

    if not forecast_hourly_data and not forecast_daily_data:
        logger.warning("API WARNING: No forecast data retrieved. No file will be saved")
        return

    forecast_dataframe = _combine_and_prepare_dataframe(forecast_hourly_data, forecast_daily_data)

    if forecast_dataframe.empty:
        logger.warning("PROCESSING WARNING: Processed forecast data is empty. No file will be saved")
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
    
    logger.info(f"FILTERING: Filtered forecast data from {original_rows} to {len(forecast_dataframe)} rows to match requested calendar days exactly")
    
    # Check if we now have empty data after filtering
    if forecast_dataframe.empty:
        logger.warning("FILTERING WARNING: After filtering for calendar days, forecast data is empty. No file will be saved")
        return

    # Validate forecast data quality
    quality_report = validate_weather_data_quality(forecast_dataframe, "forecast weather data")

    output_dir = project_root / 'data' / 'processed' / 'forecasts' / "weather"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{filename_date_str}_{num_days}days.csv" # Use filename_date_str
    filepath = output_dir / filename
    
    try:
        forecast_dataframe.to_csv(filepath, index=False)
        logger.info(f"SAVE SUCCESS: Successfully saved {num_days}-day forecast to {filepath}")
        
        if quality_report["status"] == "PASSED":
            logger.info("VALIDATION PASSED: Forecast data quality validation successful")
        elif quality_report["status"] == "WARNING":
            logger.warning("VALIDATION WARNING: Forecast data quality validation completed with warnings")
        else:
            logger.error("VALIDATION FAILED: Forecast data quality validation failed")
            
    except Exception as e:
        logger.error(f"SAVE ERROR: Error saving forecast data: {str(e)}")

def fetch_and_update_main_weather_data():
    logger.info("STARTING: Main weather data update process (historical and current forecast)")
    csv_path = project_root / 'data' / 'processed' / 'weather_data.csv'
    
    current_start_date = "2017-01-01" # Default start date
    if os.path.exists(csv_path):
        try:
            existing_data_df = pd.read_csv(csv_path)
            existing_data_df['date'] = pd.to_datetime(existing_data_df['date'])
            last_date = existing_data_df['date'].max()
            current_start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            logger.info(f"EXISTING FILE: Found existing data up to {last_date.strftime('%Y-%m-%d')}. Fetching new data from {current_start_date}")
        except Exception as e:
            logger.error(f"FILE ERROR: Error reading existing CSV to determine start date: {str(e)}. Defaulting to {current_start_date}")
    else:
        logger.info(f"NEW FILE: No existing data found at {csv_path}. Starting from {current_start_date}")

    all_hourly_data = []
    all_daily_data = []

    # Fetch archive data if needed
    if current_start_date < today:
        logger.info(f"ARCHIVE FETCH: Fetching historical data from Archive API ({current_start_date} to {yesterday})")
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
            logger.info(f"ARCHIVE SUCCESS: Archive data fetch complete. Hourly chunks: {len(all_hourly_data)}, Daily chunks: {len(all_daily_data)}")
        except Exception as e:
            logger.error(f"ARCHIVE ERROR: Error fetching archive data: {str(e)}")

    # Fetch current forecast data (standard, e.g., next 7-16 days)
    logger.info("FORECAST FETCH: Fetching standard forecast data from Forecast API (today and future for main update)")
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
        logger.info(f"FORECAST SUCCESS: Standard forecast data fetch complete. Total hourly chunks: {len(all_hourly_data)}, Total daily chunks: {len(all_daily_data)}")
    except Exception as e:
        logger.error(f"FORECAST ERROR: Error fetching standard forecast data: {str(e)}")

    if not all_hourly_data and not all_daily_data:
        logger.warning("NO DATA: No new data (archive or forecast) was collected. No updates will be made to main CSV")
        return False

    newly_fetched_dataframe = _combine_and_prepare_dataframe(all_hourly_data, all_daily_data)

    if newly_fetched_dataframe.empty:
        logger.warning("NO DATA: Processed newly fetched data is empty. No updates will be made to main CSV")
        return False

    # Validate newly fetched data quality
    quality_report = validate_weather_data_quality(newly_fetched_dataframe, "newly fetched weather data")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path):
        try:
            existing_data_df = pd.read_csv(csv_path)
            existing_data_df['date'] = pd.to_datetime(existing_data_df['date'])
            newly_fetched_dataframe['date'] = pd.to_datetime(newly_fetched_dataframe['date'])
            
            logger.info(f"MERGING: Combining new data ({len(newly_fetched_dataframe)} records) with existing data ({len(existing_data_df)} records)")
            
            # Combine, drop duplicates (keeping existing data for overlaps), sort
            full_data = pd.concat([existing_data_df, newly_fetched_dataframe])
            original_count = len(full_data)
            full_data.drop_duplicates(subset=['date'], keep='first', inplace=True)
            duplicate_count = original_count - len(full_data)
            full_data.sort_values('date', inplace=True)
            
            if duplicate_count > 0:
                logger.info(f"DUPLICATES: Removed {duplicate_count} duplicate records during merge")
            
            # Re-apply rounding after concat as types might change or NaNs introduced
            numeric_cols_full = full_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols_full:
                 if 'date' not in col.lower():
                    full_data[col] = pd.to_numeric(full_data[col], errors='coerce').round(3)

            # Final quality validation
            final_quality = validate_weather_data_quality(full_data, "final merged weather data")

            full_data.to_csv(csv_path, index=False)
            logger.info(f"SAVE SUCCESS: Successfully updated weather data at {csv_path}. New data from {current_start_date} was processed")
            logger.info(f"FILE STATS: Final file contains {len(full_data)} total records")
            
            if final_quality["status"] == "PASSED":
                logger.info("VALIDATION PASSED: Final merged data quality validation successful")
                return True
            elif final_quality["status"] == "WARNING":
                logger.warning("VALIDATION WARNING: Final merged data quality validation completed with warnings")
                return True
            else:
                logger.error("VALIDATION FAILED: Final merged data quality validation failed")
                return False
                
        except Exception as e:
            logger.error(f"MERGE ERROR: Error merging new data with existing data at {csv_path}: {str(e)}")
            return False
    else:
        newly_fetched_dataframe.to_csv(csv_path, index=False)
        logger.info(f"FILE CREATED: Successfully created new weather data file at {csv_path} with data from {current_start_date}")
        logger.info(f"FILE STATS: New file contains {len(newly_fetched_dataframe)} records")
        
        if quality_report["status"] == "PASSED":
            logger.info("VALIDATION PASSED: New weather data quality validation successful")
            return True
        elif quality_report["status"] == "WARNING":
            logger.warning("VALIDATION WARNING: New weather data quality validation completed with warnings")
            return True
        else:
            logger.error("VALIDATION FAILED: New weather data quality validation failed")
            return False
    
    logger.info(f"PROCESS COMPLETE: Main weather data update process completed. Range processed started from: {current_start_date} up to latest forecast")

def main():
    logger.info("SYSTEM START: Starting weather data fetch system")
    
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

    try:
        if args.forecast:
            if args.days is None:
                logger.error("ARGUMENT ERROR: --days is required when --forecast is specified")
                parser.error("--days is required when --forecast is specified.")
            if not (1 <= args.days <= 16): # Open-Meteo free tier typically allows up to 16 days
                logger.error(f"ARGUMENT ERROR: --days must be between 1 and 16, got {args.days}")
                parser.error("--days must be between 1 and 16.")
            
            logger.info(f"MODE: Forecast mode - fetching {args.days} days")
            fetch_and_save_specific_forecast(args.days)
            logger.info("SYSTEM SUCCESS: Forecast fetch completed successfully")
        else:
            logger.info("MODE: Main mode - updating historical and forecast weather data")
            success = fetch_and_update_main_weather_data()
            if success:
                logger.info("SYSTEM SUCCESS: Main weather data update completed successfully")
            else:
                logger.error("SYSTEM FAILED: Main weather data update failed")
                exit(1)
                
    except Exception as e:
        logger.error(f"SYSTEM ERROR: Critical error in main function: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
