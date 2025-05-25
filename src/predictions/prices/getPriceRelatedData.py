import os
# (optional) still suppress TensorFlow's C++ banners and oneDNN info:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
# suppress all Python warnings (FutureWarning, DeprecationWarning, etc.)
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import requests
from datetime import datetime, timedelta
import json
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import logging
import sys

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('src/predictions/prices/logs/price_data_fetch.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Import from config.py
from config import (
    TARGET_VARIABLE, DATA_DIR, SE3_PRICES_FILE, SWEDEN_GRID_FILE,
    TIME_FEATURES_FILE, HOLIDAYS_FILE, WEATHER_DATA_FILE,
    GRID_FEATURES, MARKET_FEATURES, PRICE_FEATURES, TIME_FEATURES, HOLIDAY_FEATURES,
    FEATURE_GROUPS, WEATHER_FEATURES
)

# Get project root from DATA_DIR
project_root = DATA_DIR.resolve().parents[1]

def validate_data_quality(df, data_type="unknown", expected_columns=None):
    """
    Comprehensive data quality validation with detailed logging.
    
    Args:
        df: DataFrame to validate
        data_type: Type of data for logging context
        expected_columns: List of expected columns
    
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
        "problematic_records": {}
    }
    
    # Basic info
    logger.info(f"{data_type} - Total records: {len(df)}")
    
    # Date range analysis
    if 'HourSE' in df.columns:
        date_col = 'HourSE'
    elif 'datetime' in df.columns:
        date_col = 'datetime'
    else:
        date_col = df.columns[0] if len(df.columns) > 0 else None
    
    if date_col and not df[date_col].empty:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            start_date = df[date_col].min()
            end_date = df[date_col].max()
            quality_report["date_range"] = f"{start_date} to {end_date}"
            logger.info(f"{data_type} - Date range: {start_date} to {end_date}")
            
            # Gap detection for hourly data
            if data_type in ["price", "grid"]:
                expected_hours = (end_date - start_date).total_seconds() / 3600 + 1
                actual_hours = len(df)
                gap_count = int(expected_hours - actual_hours)
                
                if gap_count > 0:
                    quality_report["gaps"] = [f"{gap_count} missing hours"]
                    logger.warning(f"DATA QUALITY WARNING: {data_type} - Found {gap_count} missing hours (expected {int(expected_hours)}, got {actual_hours})")
                    quality_report["issues"].append(f"Missing {gap_count} hours of data")
                    
                    # Find specific missing hours
                    expected_range = pd.date_range(start=start_date, end=end_date, freq='H')
                    missing_hours = expected_range.difference(df[date_col])
                    sample_missing = missing_hours[:20] if len(missing_hours) > 20 else missing_hours
                    logger.warning(f"MISSING HOUR TIMESTAMPS: Sample missing hours: {sample_missing.tolist()}")
                    if len(missing_hours) > 20:
                        logger.warning(f"... and {len(missing_hours) - 20} more missing hour timestamps")
                    quality_report["problematic_records"]["missing_hours"] = missing_hours[:50].tolist()
                else:
                    logger.info(f"DATA QUALITY OK: {data_type} - No gaps detected in hourly data")
                    
        except Exception as e:
            logger.error(f"DATA QUALITY ERROR: {data_type} - Error processing dates: {e}")
            quality_report["issues"].append(f"Date processing error: {e}")
    
    # Duplicate detection
    if date_col:
        duplicates = df.duplicated(subset=[date_col]).sum()
        quality_report["duplicates"] = duplicates
        if duplicates > 0:
            duplicate_mask = df.duplicated(subset=[date_col], keep=False)
            duplicate_records = df[duplicate_mask]
            logger.warning(f"DATA QUALITY WARNING: {data_type} - Found {duplicates} duplicate records")
            quality_report["issues"].append(f"{duplicates} duplicate records")
            
            # Show specific duplicate timestamps
            duplicate_timestamps = duplicate_records[date_col].tolist()
            sample_duplicates = duplicate_timestamps[:20] if len(duplicate_timestamps) > 20 else duplicate_timestamps
            logger.warning(f"DUPLICATE TIMESTAMPS: Sample duplicates: {sample_duplicates}")
            if len(duplicate_timestamps) > 20:
                logger.warning(f"... and {len(duplicate_timestamps) - 20} more duplicate timestamps")
            quality_report["problematic_records"]["duplicate_timestamps"] = duplicate_timestamps[:50]
        else:
            logger.info(f"DATA QUALITY OK: {data_type} - No duplicates found")
    
    # Missing values analysis
    missing_summary = df.isnull().sum()
    missing_dict = missing_summary[missing_summary > 0].to_dict()
    quality_report["missing_values"] = missing_dict
    
    if missing_dict:
        logger.warning(f"DATA QUALITY WARNING: {data_type} - Missing values detected:")
        for col, count in missing_dict.items():
            pct = (count / len(df)) * 100
            logger.warning(f"   {col}: {count} missing ({pct:.1f}%)")
            
            # Show specific timestamps with missing values
            if date_col and col != date_col:
                missing_mask = df[col].isnull()
                missing_timestamps = df[missing_mask][date_col].tolist()
                sample_missing = missing_timestamps[:10] if len(missing_timestamps) > 10 else missing_timestamps
                logger.warning(f"   MISSING VALUE TIMESTAMPS for {col}: {sample_missing}")
                if len(missing_timestamps) > 10:
                    logger.warning(f"   ... and {len(missing_timestamps) - 10} more missing value timestamps")
                quality_report["problematic_records"][f"missing_{col}"] = missing_timestamps[:30]
            
            if pct > 10:  # More than 10% missing is concerning
                quality_report["issues"].append(f"{col} has {pct:.1f}% missing values")
    else:
        logger.info(f"DATA QUALITY OK: {data_type} - No missing values found")
    
    # Column validation
    if expected_columns:
        missing_cols = set(expected_columns) - set(df.columns)
        extra_cols = set(df.columns) - set(expected_columns)
        
        if missing_cols:
            logger.error(f"DATA QUALITY ERROR: {data_type} - Missing expected columns: {missing_cols}")
            quality_report["issues"].append(f"Missing columns: {missing_cols}")
            quality_report["status"] = "FAILED"
        
        if extra_cols:
            logger.info(f"DATA QUALITY INFO: {data_type} - Extra columns found: {extra_cols}")
    
    # Outlier detection for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df.columns and not df[col].empty:
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
                
                if outlier_pct > 5:  # More than 5% outliers is concerning
                    logger.warning(f"DATA QUALITY WARNING: {data_type} - {col}: {outlier_count} outliers ({outlier_pct:.1f}%)")
                    quality_report["issues"].append(f"{col} has {outlier_pct:.1f}% outliers")
                    
                    # Show specific outlier records
                    outlier_records = df[outlier_mask]
                    if date_col:
                        outlier_data = outlier_records[[date_col, col]].sort_values(col)
                        
                        # Show most extreme outliers
                        extreme_low = outlier_data.head(5)
                        extreme_high = outlier_data.tail(5)
                        
                        logger.warning(f"EXTREME LOW OUTLIERS for {col}:")
                        for _, row in extreme_low.iterrows():
                            logger.warning(f"   {row[date_col]}: {row[col]:.3f}")
                        
                        logger.warning(f"EXTREME HIGH OUTLIERS for {col}:")
                        for _, row in extreme_high.iterrows():
                            logger.warning(f"   {row[date_col]}: {row[col]:.3f}")
                        
                        if outlier_count > 10:
                            logger.warning(f"   ... and {outlier_count - 10} more outlier records")
                        
                        quality_report["problematic_records"][f"outliers_{col}"] = [
                            {"timestamp": str(row[date_col]), "value": float(row[col])} 
                            for _, row in outlier_data.head(10).iterrows()
                        ] + [
                            {"timestamp": str(row[date_col]), "value": float(row[col])} 
                            for _, row in outlier_data.tail(10).iterrows()
                        ]
                    else:
                        outlier_values = df[outlier_mask][col].sort_values()
                        sample_outliers = outlier_values.head(10).tolist() + outlier_values.tail(10).tolist()
                        logger.warning(f"OUTLIER VALUES for {col}: {sample_outliers}")
                        quality_report["problematic_records"][f"outliers_{col}"] = sample_outliers[:20]
                else:
                    logger.info(f"DATA QUALITY INFO: {data_type} - {col}: {outlier_count} outliers ({outlier_pct:.1f}%) - within acceptable range")
                    
                    # Still log a few examples for reference
                    if date_col:
                        outlier_records = df[outlier_mask][[date_col, col]].head(5)
                        sample_outliers = [(str(row[date_col]), float(row[col])) for _, row in outlier_records.iterrows()]
                        logger.info(f"SAMPLE OUTLIERS for {col}: {sample_outliers}")
    
    # Final status determination
    if quality_report["issues"]:
        if any("FAILED" in str(issue) or "Missing columns" in str(issue) for issue in quality_report["issues"]):
            quality_report["status"] = "FAILED"
        else:
            quality_report["status"] = "WARNING"
        
        logger.warning(f"DATA QUALITY SUMMARY: {data_type} - Data quality issues found: {len(quality_report['issues'])}")
        for issue in quality_report["issues"]:
            logger.warning(f"   Issue: {issue}")
    else:
        logger.info(f"DATA QUALITY PASSED: {data_type} - All data quality checks passed")
    
    return quality_report

def update_price_data():
    """Update the price data from the API and calculate features"""
    logger.info("STARTING: Price data update process")
    
    # Use paths from config
    csv_file_path = DATA_DIR / SE3_PRICES_FILE.name
    
    # Check if the file exists
    if not csv_file_path.exists():
        df = pd.DataFrame(columns=['HourSE', 'PriceArea', TARGET_VARIABLE])
        # Start from 30 days ago if creating a new file to avoid excessive API calls
        latest_timestamp = pd.Timestamp.now() - timedelta(days=30)
        logger.info(f"FILE CREATION: Creating new price data file. Starting from {latest_timestamp}")
    else:
        # Read the existing CSV file
        try:
            df = pd.read_csv(csv_file_path)
            df['HourSE'] = pd.to_datetime(df['HourSE'])
            logger.info(f"FILE LOADED: Existing price data loaded - {len(df)} records")
            
            # If the file exists but is empty or corrupted, start from 30 days ago
            if df.empty:
                latest_timestamp = pd.Timestamp.now() - timedelta(days=30)
                logger.warning(f"FILE WARNING: Existing file is empty. Starting from {latest_timestamp}")
            else:
                # Get the latest timestamp from the existing data
                latest_timestamp = df['HourSE'].max()
                logger.info(f"CONTINUING FROM: Latest timestamp in file: {latest_timestamp}")
        except Exception as e:
            logger.error(f"FILE ERROR: Error reading existing price data file: {e}")
            df = pd.DataFrame(columns=['HourSE', 'PriceArea', TARGET_VARIABLE])
            latest_timestamp = pd.Timestamp.now() - timedelta(days=30)

    # Get the current time
    current_time = datetime.now()
    new_data = []
    api_failures = 0
    max_failures = 5

    logger.info(f"API FETCH: Fetching price data from {latest_timestamp} to {current_time}")

    # Loop through missing hours until the present
    hours_to_fetch = 0
    temp_timestamp = latest_timestamp
    while temp_timestamp < current_time:
        temp_timestamp += timedelta(hours=1)
        hours_to_fetch += 1

    logger.info(f"API FETCH: Need to fetch {hours_to_fetch} hours of price data")

    # Reset for actual fetching
    while latest_timestamp < current_time:
        latest_timestamp += timedelta(hours=1)
        next_date_str = latest_timestamp.strftime('%Y-%m-%d')
        next_hour_int = latest_timestamp.hour
        next_hour_str = latest_timestamp.strftime('%H:00:00')

        # Fetch data from the API
        api_url = f'https://mgrey.se/espot?format=json&date={next_date_str}'
        try:
            response = requests.get(api_url, timeout=10)  # Add timeout for safety
            
            if response.status_code == 200:
                data = response.json()
                if 'SE3' in data:
                    se3_data = data['SE3']
                    hour_data = next((item for item in se3_data if item['hour'] == next_hour_int), None)
                    
                    if hour_data:
                        new_data.append({
                            'HourSE': f'{next_date_str} {next_hour_str}',
                            'PriceArea': 'SE3',
                            TARGET_VARIABLE: hour_data['price_sek'],
                        })
                    else:
                        logger.warning(f"API WARNING: No hour data available for {next_date_str} {next_hour_str}")
                        api_failures += 1
                else:
                    logger.warning(f"API WARNING: No SE3 data available for {next_date_str}")
                    api_failures += 1
            else:
                logger.error(f"API ERROR: Request failed for {next_date_str}. Status code: {response.status_code}")
                api_failures += 1
                if api_failures >= max_failures:
                    logger.error(f"API ERROR: Too many API failures ({api_failures}). Stopping data fetch.")
                    break
        except Exception as e:
            logger.error(f"API ERROR: Error fetching price data for {next_date_str}: {e}")
            api_failures += 1
            if api_failures >= max_failures:
                logger.error(f"API ERROR: Too many API failures ({api_failures}). Stopping data fetch.")
                break

    # Update DataFrame with new data
    if new_data:
        logger.info(f"PROCESSING: Processing {len(new_data)} new price records")
        new_df = pd.DataFrame(new_data)
        new_df['HourSE'] = pd.to_datetime(new_df['HourSE'])
        df = pd.concat([df, new_df], ignore_index=True)
        df = df.sort_values(by="HourSE")

        # Calculate features
        logger.info("FEATURE CALCULATION: Calculating price features...")
        df["price_24h_avg"] = df[TARGET_VARIABLE].rolling(window=24, min_periods=1).mean()
        df["price_168h_avg"] = df[TARGET_VARIABLE].rolling(window=168, min_periods=1).mean()
        df["price_24h_std"] = df[TARGET_VARIABLE].rolling(window=24, min_periods=1).std().fillna(0)
        df["hour_avg_price"] = df.groupby(df["HourSE"].dt.hour)[TARGET_VARIABLE].transform("mean")
        df["price_vs_hour_avg"] = df[TARGET_VARIABLE] / df["hour_avg_price"]

        # Validate data quality
        quality_report = validate_data_quality(df, "price", expected_columns=['HourSE', 'PriceArea', TARGET_VARIABLE])

        # Save to CSV
        try:
            df.to_csv(csv_file_path, index=False)
            logger.info(f'SAVE SUCCESS: Price data saved - {len(new_data)} new records added')
            logger.info(f'FILE STATS: Total records in file: {len(df)}')
            
            if quality_report["status"] == "PASSED":
                logger.info("VALIDATION PASSED: Price data quality validation successful")
            elif quality_report["status"] == "WARNING":
                logger.warning("VALIDATION WARNING: Price data quality validation completed with warnings")
            else:
                logger.error("VALIDATION FAILED: Price data quality validation failed")
                
        except Exception as e:
            logger.error(f"SAVE ERROR: Error saving price data to CSV: {e}")
            return False
    else:
        logger.info("NO UPDATE: Price data is already up-to-date - no new records to add")
        
        # Still validate existing data
        if not df.empty:
            quality_report = validate_data_quality(df, "price")
            if quality_report["status"] != "PASSED":
                logger.warning("VALIDATION WARNING: Existing price data has quality issues")

    if api_failures > 0:
        logger.warning(f"API SUMMARY: Encountered {api_failures} API failures during price data fetch")
        
    return True

def update_grid_data():
    """Update the grid data using Electricity Maps API with hourly resolution"""
    logger.info("STARTING: Grid data update process")
    
    # Use path from config
    grid_file_path = DATA_DIR / SWEDEN_GRID_FILE.name
    
    # Load API key from env file
    dotenv_path = project_root / 'api.env'
    load_dotenv(dotenv_path=dotenv_path)
    api_key = os.getenv('ELECTRICITYMAPS')
    
    if not api_key:
        logger.error("API ERROR: ELECTRICITYMAPS API key not found in api.env")
        return False
    else:
        logger.info("API SUCCESS: Electricity Maps API key loaded successfully")
    
    # Use grid columns from feature groups config
    grid_cols = FEATURE_GROUPS["grid_cols"]
    logger.info(f"CONFIGURATION: Expected grid columns: {len(grid_cols)} columns")
    
    # Define the essential zones for import/export
    zones = {
        'SE-SE2': 'Main connection from northern Sweden',
        'SE-SE4': 'Main connection to southern Sweden',
        'NO-NO1': 'Norway connection',
        'DK-DK1': 'Denmark connection',
        'FI': 'Finland connection',
    }
    
    # Define the power sources matching config
    power_sources = ['nuclear', 'wind', 'hydro', 'solar', 'unknown']
    
    # Make API request for history data - limit to the last 7 days
    try:
        # Calculate start date (7 days ago)
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.info(f"API FETCH: Fetching grid data from {start_date} (last 7 days)")
        
        response = requests.get(
            f"https://api.electricitymap.org/v3/power-breakdown/history?zone=SE-SE3&start={start_date}",
            headers={
                "auth-token": api_key
            },
            timeout=30  # Added timeout
        )
        
        if response.status_code != 200:
            logger.error(f"API ERROR: Grid API request failed. Status code: {response.status_code}")
            logger.error(f"Response content: {response.content.decode() if hasattr(response, 'content') else 'No response content'}")
            return False
        
        data = response.json()
        logger.info(f"API SUCCESS: Received grid data with {len(data.get('history', []))} records from API")
        
        # Process records at hourly resolution
        hourly_records = []
        processing_errors = 0
        
        for i, entry in enumerate(data['history']):
            try:
                # Convert datetime to proper format with explicit UTC handling
                try:
                    # Parse the timestamp without specifying format (pandas will auto-detect)
                    entry_time = pd.to_datetime(entry['datetime'])
                    
                    # If the timestamp has timezone info, convert to UTC then make naive
                    if entry_time.tzinfo is not None:
                        # Convert to UTC
                        entry_time = entry_time.tz_convert('UTC')
                        # Remove timezone info to make it naive
                        entry_time = entry_time.tz_localize(None)
                except Exception as tz_err:
                    logger.warning(f"TIMESTAMP WARNING: Error processing timestamp {entry['datetime']}: {tz_err}")
                    # Fallback: try string manipulation if parsing fails
                    try:
                        # If it's a string with timezone info like "2025-04-15 19:00:00+00:00"
                        ts_str = str(entry['datetime'])
                        if '+' in ts_str:
                            # Remove the timezone part
                            base_ts = ts_str.split('+')[0]
                            entry_time = pd.to_datetime(base_ts)
                        else:
                            entry_time = pd.to_datetime(ts_str)
                    except Exception as parse_err:
                        logger.error(f"TIMESTAMP ERROR: Severe error parsing timestamp: {parse_err}. Skipping record {i}")
                        processing_errors += 1
                        continue
                
                # Create record for this hour
                record = {col: 0 for col in grid_cols}  # Initialize all columns from config
                record['datetime'] = entry_time
                
                # Set core metrics and round to 2 decimals
                record['fossilFreePercentage'] = round(entry.get('fossilFreePercentage', 0) or 0, 2)
                record['renewablePercentage'] = round(entry.get('renewablePercentage', 0) or 0, 2)
                record['powerConsumptionTotal'] = round(entry.get('powerConsumptionTotal', 0) or 0, 2)
                record['powerProductionTotal'] = round(entry.get('powerProductionTotal', 0) or 0, 2)
                record['powerImportTotal'] = round(entry.get('powerImportTotal', 0) or 0, 2)
                record['powerExportTotal'] = round(entry.get('powerExportTotal', 0) or 0, 2)
                
                # Get production breakdown
                prod = entry.get('powerProductionBreakdown', {})
                
                # Set power sources and round
                for source in power_sources:
                    record[source] = round(prod.get(source, 0) or 0, 2)
                
                # Set import/export for each zone and round
                import_breakdown = entry.get('powerImportBreakdown', {})
                export_breakdown = entry.get('powerExportBreakdown', {})
                
                for zone in zones:
                    record[f'import_{zone}'] = round(import_breakdown.get(zone, 0) or 0, 2)
                    record[f'export_{zone}'] = round(export_breakdown.get(zone, 0) or 0, 2)
                
                hourly_records.append(record)
                
            except Exception as record_err:
                logger.error(f"PROCESSING ERROR: Error processing grid record {i}: {record_err}")
                processing_errors += 1
                continue
        
        if processing_errors > 0:
            logger.warning(f"PROCESSING WARNING: Encountered {processing_errors} errors while processing grid records")
        
        # Create DataFrame with specified columns from config
        df = pd.DataFrame(hourly_records)
        logger.info(f"DATAFRAME CREATED: Created DataFrame with {len(df)} grid records")
        
        # Set datetime as index and ensure it's properly formatted
        if not df.empty:
            df.set_index('datetime', inplace=True)
            
            # Ensure all required columns are present (initialize with zeros if missing)
            missing_columns = []
            for col in grid_cols:
                if col not in df.columns:
                    df[col] = 0
                    missing_columns.append(col)
            
            if missing_columns:
                logger.warning(f"COLUMN WARNING: Added missing columns with zero values: {missing_columns}")
            
            # Select only the columns defined in grid_cols and in the correct order
            df = df[grid_cols]
            df.sort_index(inplace=True)
            
            # Check for missing values
            missing_count = df.isna().sum().sum()
            if missing_count > 0:
                # Fill any missing values with 0
                df = df.fillna(0)
                logger.warning(f"MISSING VALUES: Found and filled {missing_count} missing values in grid data with zeros")
            
            # Validate data quality before merging
            quality_report = validate_data_quality(df.reset_index(), "grid", expected_columns=['datetime'] + grid_cols)
            
            # Merge with existing data if it exists and is not empty
            if grid_file_path.exists():
                try:
                    logger.info("FILE MERGE: Loading existing grid data for merging")
                    existing_df = pd.read_csv(grid_file_path)
                    if not existing_df.empty:
                        # Ensure datetime column exists and convert to proper datetime
                        if 'datetime' in existing_df.columns:
                            existing_df['datetime'] = pd.to_datetime(existing_df['datetime'])
                            existing_df.set_index('datetime', inplace=True)
                        else:
                            # If no datetime column, try using the first column as index
                            existing_df = pd.read_csv(grid_file_path, index_col=0, parse_dates=True)
                        
                        # Handle timezone issues
                        if existing_df.index.tzinfo is not None:
                            existing_df.index = existing_df.index.tz_convert('UTC').tz_localize(None)
                        
                        logger.info(f"EXISTING DATA: Existing grid data has {len(existing_df)} records")
                        
                        # Combine data, keeping the newer data for duplicated timestamps
                        combined_df = pd.concat([existing_df, df])
                        original_count = len(combined_df)
                        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                        duplicate_count = original_count - len(combined_df)
                        combined_df.sort_index(inplace=True)
                        
                        if duplicate_count > 0:
                            logger.info(f"DUPLICATES REMOVED: Removed {duplicate_count} duplicate timestamps (kept newest)")
                        
                        # Reset index to save datetime as column
                        combined_df.reset_index(inplace=True)
                        combined_df.rename(columns={'index': 'datetime'}, inplace=True)
                        
                        # Final quality validation
                        final_quality = validate_data_quality(combined_df, "grid (final)", expected_columns=['datetime'] + grid_cols)
                        
                        # Save the updated data
                        combined_df.to_csv(grid_file_path, index=False)
                        logger.info(f'SAVE SUCCESS: Grid data updated - {len(df)} new records merged')
                        logger.info(f'FILE STATS: Total records in file: {len(combined_df)}')
                        
                        if final_quality["status"] == "PASSED":
                            logger.info("VALIDATION PASSED: Grid data quality validation successful")
                        elif final_quality["status"] == "WARNING":
                            logger.warning("VALIDATION WARNING: Grid data quality validation completed with warnings")
                        else:
                            logger.error("VALIDATION FAILED: Grid data quality validation failed")
                        
                    else:
                        # If existing file is empty or corrupted, just save the new data
                        df.reset_index(inplace=True)
                        df.to_csv(grid_file_path, index=False)
                        logger.info(f'FILE CREATED: New grid data file created - {len(df)} records (existing file was empty)')
                        
                except Exception as e:
                    logger.error(f"MERGE ERROR: Error merging with existing grid data: {e}")
                    # If there's an error with the existing file, save the new data as backup
                    backup_path = grid_file_path.with_suffix('.backup.csv')
                    df.reset_index(inplace=True)
                    df.to_csv(backup_path, index=False)
                    logger.warning(f'BACKUP CREATED: Saved backup grid data to {backup_path} due to merge error')
                    return False
            else:
                # No existing file, just save the new data
                df.reset_index(inplace=True)
                df.to_csv(grid_file_path, index=False)
                logger.info(f'FILE CREATED: New grid data file created - {len(df)} records')
                
                if quality_report["status"] == "PASSED":
                    logger.info("VALIDATION PASSED: Grid data quality validation successful")
                elif quality_report["status"] == "WARNING":
                    logger.warning("VALIDATION WARNING: Grid data quality validation completed with warnings")
                else:
                    logger.error("VALIDATION FAILED: Grid data quality validation failed")
        else:
            logger.error("API ERROR: No grid data records received from API")
            return False
            
    except Exception as e:
        logger.error(f"CRITICAL ERROR: Error updating grid data: {e}")
        return False
    
    return True

def main():
    """Main function to update all data files with comprehensive reporting"""
    logger.info("SYSTEM START: Starting comprehensive price and grid data update")
    
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"DIRECTORY: Data directory confirmed: {DATA_DIR}")
    
    success_count = 0
    total_tasks = 2
    
    # Update price data
    logger.info("=" * 60)
    logger.info("PRICE DATA UPDATE")
    logger.info("=" * 60)
    try:
        if update_price_data():
            success_count += 1
            logger.info("TASK SUCCESS: Price data update completed successfully")
        else:
            logger.error("TASK FAILED: Price data update failed")
    except Exception as e:
        logger.error(f"TASK EXCEPTION: Price data update failed with exception: {e}")
    
    # Update grid data
    logger.info("=" * 60)
    logger.info("GRID DATA UPDATE")
    logger.info("=" * 60)
    try:
        if update_grid_data():
            success_count += 1
            logger.info("TASK SUCCESS: Grid data update completed successfully")
        else:
            logger.error("TASK FAILED: Grid data update failed")
    except Exception as e:
        logger.error(f"TASK EXCEPTION: Grid data update failed with exception: {e}")
    
    # Final summary
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Tasks completed successfully: {success_count}/{total_tasks}")
    
    if success_count == total_tasks:
        logger.info("SYSTEM SUCCESS: ALL DATA UPDATES COMPLETED SUCCESSFULLY!")
        return True
    elif success_count > 0:
        logger.warning(f"SYSTEM PARTIAL: PARTIAL SUCCESS - {success_count}/{total_tasks} tasks completed")
        return False
    else:
        logger.error("SYSTEM FAILED: ALL DATA UPDATES FAILED!")
        return False

if __name__ == "__main__":
    main()