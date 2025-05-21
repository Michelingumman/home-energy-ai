"""
this script will calculate the actual load not just he tibber power import.
Actula load is the following:
actual_consumption (load) = import - export + solar + battery_flow

where battery_flow is positive when discharging to the house and negative when charging


"""

import pandas as pd 
import requests
import json
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import logging
import sys
from dateutil.parser import parse
import os
import pathlib as Path
import subprocess
import math

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# Load environment variables
load_dotenv('api.env')
TIBBER_TOKEN = os.getenv('TIBBER_TOKEN')

if not TIBBER_TOKEN:
    logger.error("TIBBER_TOKEN not found in api.env file!")
    sys.exit(1)


def check_and_fill_gaps(df, source_name):
    """
    Check for gaps in hourly time series data and fill them using ffill and bfill
    
    Args:
        df (pandas.DataFrame): DataFrame with datetime index
        source_name (str): Name of the data source for logging purposes
        
    Returns:
        pandas.DataFrame: DataFrame with gaps filled
    """
    if df.empty:
        logger.warning(f"No data available for {source_name}")
        return df
    
    # Check original length
    original_length = len(df)
    
    # Check for duplicate timestamps
    duplicates = df.index.duplicated()
    if duplicates.any():
        dup_count = duplicates.sum()
        logger.warning(f"Found {dup_count} duplicate timestamps in {source_name} data")
        
        # Show example duplicates
        dup_idx = df.index[duplicates]
        sample_count = min(3, len(dup_idx))
        for i in range(sample_count):
            logger.warning(f"  • Duplicate example {i+1}: {dup_idx[i].isoformat()}")
        
        # Keep the last occurrence of each duplicate
        logger.warning(f"Removing duplicates from {source_name} data (keeping latest value)")
        df = df[~df.index.duplicated(keep='last')]
        logger.info(f"Removed {dup_count} duplicate rows from {source_name} data")
    
    if isinstance(df.index, pd.DatetimeIndex):
        tz = df.index.tz
    else:
        tz = None
    
    # Build complete hourly index
    full_idx = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq='h',
        tz=tz
    )
    
    # Check for missing timestamps
    missing = full_idx.difference(df.index)
    
    if not missing.empty:
        logger.warning(f"Found {len(missing)} gaps in {source_name} data")
        
        # Log some example gaps
        sample_count = min(5, len(missing))
        for i in range(sample_count):
            logger.warning(f"  • Gap example {i+1}: {missing[i].isoformat()}")
        
        # Reindex with the full index
        df = df.reindex(full_idx)
        
        # Apply forward fill then backward fill
        df = df.ffill().bfill()
        
        # Log statistics
        filled_count = len(df) - original_length
        logger.info(f"Filled {filled_count} missing values in {source_name} data using ffill and bfill")
    else:
        logger.info(f"No gaps found in {source_name} data")
    
    return df


def tibber_import_export_data():
    """
    Get consumption and production data from Tibber
    """
    print('='*40)
    print("Fetching Tibber data...")
    print('='*40)
    
    df = pd.read_csv('data/processed/villamichelin/VillamichelinEnergyData.csv', index_col='timestamp', parse_dates=['timestamp'])
    df.index = df.index.tz_localize('Europe/Stockholm', ambiguous=True, nonexistent='shift_forward')
    df = df[['consumption', 'production']]
    
    # Check for gaps and fill them
    df = check_and_fill_gaps(df, "Tibber import/export")
    
    print("\nTibber data Import and Export: ", df.head(3), df.tail(3), '\n\n\n')
    return df


def solar_data():
    """
    Get solar production data
    """
    print('='*40)
    print("Fetching Solar production data...")
    print('='*40)
    
    df = pd.read_csv('src/predictions/solar/actual_data/ActualSolarProductionData.csv', index_col='Timestamp', parse_dates=['Timestamp'])
    df.index = df.index.tz_convert('Europe/Stockholm')
    df = df[['solar_production_kwh']]
    
    # Check for gaps and fill them
    df = check_and_fill_gaps(df, "Solar production")
    
    print("\nSolar data: ", df.head(3), df.tail(3), '\n\n\n')
    return df


def sonnen_data():
    """
    Get battery flow data from Sonnen battery:
    1. Check existing data in sonnenbattery.csv
    2. If not up to date, download latest data using downloadEntityData.py
    3. Merge existing and new data
    4. Update the original sonnenbattery.csv file with combined data
    5. Return the combined dataframe
    """
    print('='*40)
    print("Fetching Sonnen battery data...")
    print('='*40)
    # File paths
    base_sonnen_file = 'data/HomeAssistant/sonnenbattery.csv'
    hourly_sonnen_file = 'data/HomeAssistant/sonnenbatterie_307421_state_battery_inout_hourly.csv'
    
    # Get current time (rounded to the nearest hour)
    current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    current_time = current_time.astimezone(timezone(timedelta(hours=2)))  # Europe/Stockholm
    
    # Check latest timestamp in existing data
    try:
        # Read the CSV with parse_dates to properly handle timezone info
        existing_df = pd.read_csv(base_sonnen_file, parse_dates=['timestamp'])
        
        # Set the index properly - the timestamps already have timezone info
        existing_df.set_index('timestamp', inplace=True)
        
        if not existing_df.empty:
            latest_timestamp = existing_df.index.max()
            
            # Make sure the timestamp has timezone info
            if latest_timestamp.tzinfo is None:
                latest_timestamp = latest_timestamp.tz_localize('Europe/Stockholm', ambiguous=True, nonexistent='shift_forward')
            
            # Calculate days difference
            time_diff = current_time - latest_timestamp
            days_missing = math.ceil(time_diff.total_seconds() / (24 * 3600))
            
            logger.info(f"Latest Sonnen data timestamp: {latest_timestamp}")
            logger.info(f"Current time: {current_time}")
            logger.info(f"Days missing: {days_missing}")
            
            # If data is not up to date, download latest data
            if days_missing > 0:
                logger.info(f"Downloading {days_missing} days of Sonnen battery data...")
                
                # Run the download script
                cmd = [
                    sys.executable,
                    "c:/_Projects/home-energy-ai/src/downloadEntityData.py",
                    "--entity", "sensor.sonnenbatterie_307421_state_battery_inout",
                    "--days", str(days_missing),
                    "--res", "hourly"
                ]
                
                try:
                    subprocess.run(cmd, check=True)
                    logger.info("Successfully downloaded new Sonnen battery data")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error downloading Sonnen data: {e}")
        else:
            logger.warning(f"Existing Sonnen data file {base_sonnen_file} is empty")
            # Run with a default of 7 days if file exists but is empty
            cmd = [
                sys.executable,
                "c:/_Projects/home-energy-ai/src/downloadEntityData.py",
                "--entity", "sensor.sonnenbatterie_307421_state_battery_inout",
                "--days", "7",
                "--res", "hourly"
            ]
            subprocess.run(cmd, check=True)
            
    except FileNotFoundError:
        logger.warning(f"Sonnen data file {base_sonnen_file} not found, downloading last 7 days")
        # Run with a default of 7 days if file doesn't exist
        cmd = [
            sys.executable,
            "c:/_Projects/home-energy-ai/src/downloadEntityData.py",
            "--entity", "sensor.sonnenbatterie_307421_state_battery_inout",
            "--days", "7",
            "--res", "hourly"
        ]
        subprocess.run(cmd, check=True)
    
    # Now read the hourly data file that was created/updated
    try:
        hourly_df = pd.read_csv(hourly_sonnen_file, parse_dates=['timestamp'])
        hourly_df.set_index('timestamp', inplace=True)
        
        # Ensure timezone info is present
        if hourly_df.index.tzinfo is None:
            hourly_df.index = hourly_df.index.tz_localize('Europe/Stockholm', ambiguous=True, nonexistent='shift_forward')
        
        # Convert to kWh
        hourly_df['battery_flow'] = hourly_df['state']/1000  # from Watt to kWh
        hourly_df = hourly_df.drop(columns=['state'])
        
        # If we also have existing data, merge them
        try:
            # Read the CSV with parse_dates to properly handle timezone info
            existing_df = pd.read_csv(base_sonnen_file, parse_dates=['timestamp'])
            
            # Set the index properly
            existing_df.set_index('timestamp', inplace=True)
            
            # Ensure timezone info is present
            if existing_df.index.tzinfo is None:
                existing_df.index = existing_df.index.tz_localize('Europe/Stockholm', ambiguous=True, nonexistent='shift_forward')
            
            # Convert columns if needed
            if 'state' in existing_df.columns:
                existing_df['battery_flow'] = existing_df['state']/1000
                existing_df = existing_df.drop(columns=['state'])
            
            # Combine the dataframes, keeping the more recent data if there are duplicates
            combined_df = pd.concat([existing_df, hourly_df])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df = combined_df.sort_index()
            
            logger.info(f"Combined Sonnen data: {len(existing_df)} existing + {len(hourly_df)} new = {len(combined_df)} total records")
            
            # Update the original sonnenbattery.csv file with the combined data
            if len(hourly_df) > 0:
                logger.info(f"Updating {base_sonnen_file} with newly fetched data")
                # Ensure the directory exists
                os.makedirs(os.path.dirname(base_sonnen_file), exist_ok=True)
                # Save the combined data back to the original file
                combined_df.to_csv(base_sonnen_file)
                logger.info(f"Successfully updated {base_sonnen_file} with {len(hourly_df)} new records")
            
            # Check for gaps and fill them
            combined_df = check_and_fill_gaps(combined_df, "Sonnen battery")
            
            print("\nSonnenbatterie data: ", combined_df.head(), combined_df.tail(), '\n\n\n')
            return combined_df
            
        except FileNotFoundError:
            # If base file doesn't exist, create it with hourly data
            logger.info(f"Creating new {base_sonnen_file} with hourly data ({len(hourly_df)} records)")
            # Ensure the directory exists
            os.makedirs(os.path.dirname(base_sonnen_file), exist_ok=True)
            # Save the hourly data to the base file
            hourly_df.to_csv(base_sonnen_file)
            logger.info(f"Successfully created {base_sonnen_file} with {len(hourly_df)} records")
            
            # Check for gaps and fill them
            hourly_df = check_and_fill_gaps(hourly_df, "Sonnen battery")
            
            print("\nSonnenbatterie data: ", hourly_df.head(), hourly_df.tail(), '\n\n\n')
            return hourly_df
            
    except FileNotFoundError:
        logger.error(f"Hourly Sonnen data file {hourly_sonnen_file} not found even after attempted download")
        # Return empty DataFrame with the expected columns
        empty_df = pd.DataFrame(columns=['battery_flow'])
        empty_df.index.name = 'timestamp'
        return empty_df


def merge_data():
    """
    Merge all data sources into a single DataFrame
    """
    print('='*40)
    print("Merging data...")
    print('='*40)
    df_import_export = tibber_import_export_data()
    df_solar = solar_data()
    df_sonnen = sonnen_data()
    
    # Perform outer joins to include all timestamps
    df = pd.merge(df_import_export, df_solar, left_index=True, right_index=True, how='outer')
    df = pd.merge(df, df_sonnen, left_index=True, right_index=True, how='outer')
    
    # Fill any NaN values created by the outer join
    df = df.ffill().bfill()
    
    print("\nMerged data: ", df.head(), df.tail(), '\n\n\n')
    return df


def calculate_actual_load():
    """
    Calculate actual load using the formula:
    actual_consumption = import - export + solar + battery_flow
    """
    print('='*40)
    print("Calculating actual load...")
    print('='*40)
    df = merge_data()

    # 1) Check for gaps in the hourly index
    # -------------------------------------
    # Ensure index is DatetimeIndex with proper timezone
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("Index is not a DatetimeIndex. Converting...")
        # Convert to datetime index with UTC timezone first
        df.index = pd.to_datetime(df.index, utc=True)
        # Then convert to Europe/Stockholm if needed
        if str(df.index.tz) != 'Europe/Stockholm':
            try:
                df.index = df.index.tz_convert('Europe/Stockholm')
            except Exception as e:
                logger.warning(f"Could not convert timezone to Europe/Stockholm: {e}. Using UTC timezone.")
    
    # Get timezone from index for date_range
    tz_info = df.index.tz if hasattr(df.index, 'tz') else 'Europe/Stockholm'
    
    # Build the complete hourly index over the observed timespan
    full_idx = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq='h',
        tz=tz_info
    )
    # Identify missing timestamps
    missing = full_idx.difference(df.index)
    if not missing.empty:
        logger.warning(f"Gaps detected in merged time series! Missing {len(missing)} hourly timestamps")
        # Fill gaps using forward and backward fill
        df = df.reindex(full_idx).ffill().bfill()
        logger.info(f"Filled {len(missing)} gaps in merged data using ffill and bfill")

    # 2) Calculate actual consumption
    # --------------------------------
    df['actual_consumption'] = (
        df['consumption']
        - df['production']
        + df['solar_production_kwh']
        + df['battery_flow']
    )
    df.index.name = 'Timestamp'
    df = df.round(3)

    # 3) Validate for null/NaN/space values as before
    # -----------------------------------------------
    if df.isnull().any().any():
        logger.error("Null values found in the DataFrame. Please check the data.")
        print(df[df.isnull().any(axis=1)])
    elif df.isna().any().any():
        logger.error("NaN values found in the DataFrame. Please check the data.")
        print(df[df.isna().any(axis=1)])
    else:
        logger.info("No null or NaN values found in the DataFrame.")

    # 4) Persist results
    # ------------------
    df.to_csv('data/processed/villamichelin/VillamichelinActualLoad.csv')
    
    logger.info("Actual load calculated and saved to data/processed/villamichelin/VillamichelinActualLoad.csv")
    logger.info("Script completed successfully.")
    return df

if __name__ == "__main__":
    try:
        calculate_actual_load()
    except Exception as e:
        logger.error(f"Error in script execution: {e}")
        import traceback
        logger.error(traceback.format_exc())