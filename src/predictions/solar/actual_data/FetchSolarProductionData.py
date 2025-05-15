#!/usr/bin/env python3
"""
Merge Actual Solar Power Data
============================

This script:
1. Checks for existing data in merged_cleaned_actual_data.csv
2. Determines how many days of data need to be fetched.
3. Tries to fetch data from SolarEdge API (outputting Timestamp, solar_production_kwh).
4. If SolarEdge fails, fetches from Home Assistant (transforming to Timestamp, solar_production_kwh).
5. Merges the new data with existing data, handling overlaps.
6. Saves the updated data to the CSV with Timestamp, solar_production_kwh columns.
7. Optionally, fetches all historical data from SolarEdge API using --fetch-all.

This script is designed to run daily to keep the solar generation data up to date.
"""

import pandas as pd
import os
import subprocess
import sys
from datetime import datetime, timedelta, date, timezone 
from pathlib import Path
from dotenv import load_dotenv # Changed from import dotenv
import requests 
import argparse 
# import time # For potential rate limiting, if needed
import pytz # For timezone handling

# Load environment variables from the correct path
project_root = Path(__file__).resolve().parents[4]
dotenv_path = project_root / 'api.env'
load_dotenv(dotenv_path=dotenv_path)


# Constants
ENTITY_ID = "sensor.solar_generated_power_2" # Used for Home Assistant fallback
RESOLUTION = "hourly" # Used for Home Assistant fallback filename
OUTPUT_FILE = "ActualSolarProductionData.csv" # CSV filename
CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(__file__).resolve().parents[4] / "data/HomeAssistant" # HA data download directory
DOWNLOAD_SCRIPT = Path(__file__).resolve().parents[3] / "downloadEntityData.py" # HA download script

# SolarEdge API Constants
SOLAREDGE_API_KEY = os.getenv("SOLAREDGE_API_KEY")
SOLAREDGE_SITE_ID = os.getenv("SOLAREDGE_SITE_ID")
SOLAREDGE_BASE_URL = f"https://monitoringapi.solaredge.com/site/{SOLAREDGE_SITE_ID}" if SOLAREDGE_SITE_ID else None
SOLAREDGE_SITE_TIMEZONE = os.getenv("SOLAREDGE_SITE_TIMEZONE", "Europe/Stockholm")


def get_solaredge_api_data_period() -> tuple[date | None, date | None]:
    """Fetches data period from SolarEdge and returns (api_start_date, api_end_date). Fallback to None if API call fails."""
    if not SOLAREDGE_API_KEY or not SOLAREDGE_SITE_ID or not SOLAREDGE_BASE_URL:
        print("SolarEdge API not configured, cannot fetch data period.")
        return None, None

    data_period_url = f"{SOLAREDGE_BASE_URL}/dataPeriod.json"
    params_period = {"api_key": SOLAREDGE_API_KEY}
    try:
        response = requests.get(data_period_url, params=params_period, timeout=30)
        response.raise_for_status()
        data_period_json = response.json()["dataPeriod"]
        api_start_date = datetime.strptime(data_period_json["startDate"], "%Y-%m-%d").date()
        api_end_date = datetime.strptime(data_period_json["endDate"], "%Y-%m-%d").date()
        print(f"SolarEdge API reports data available from: {api_start_date} to {api_end_date}")
        return api_start_date, api_end_date
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data period from SolarEdge: {e}.")
        return None, None
    except (KeyError, ValueError) as e:
        print(f"Error parsing data period response from SolarEdge: {e}.")
        return None, None


def fetch_solaredge_power_data(start_date_str: str, end_date_str: str) -> pd.DataFrame | None:
    """
    Fetches energy data from SolarEdge API for the given date range.
    Timestamps from API are assumed to be in SOLAREDGE_SITE_TIMEZONE, converted to UTC.
    Data is fetched in hourly resolution (default of power.json) and converted to kWh.
    Dates should be in YYYY-MM-DD format.
    Returns a DataFrame with 'Timestamp' (datetime, UTC) and 'solar_production_kwh' (float)
    or None if an error occurs or no data.
    """
    if not SOLAREDGE_API_KEY or not SOLAREDGE_SITE_ID or not SOLAREDGE_BASE_URL:
        print("SolarEdge API Key, Site ID, or Base URL not configured. Skipping SolarEdge fetch.")
        return None

    # The API expects startTime and endTime to include time, but dates work for full day fetches.
    # For hourly data, YYYY-MM-DD should cover the whole day from 00:00 to 23:59 site time.
    api_url = f"{SOLAREDGE_BASE_URL}/power.json" 
    params = {
        "api_key": SOLAREDGE_API_KEY,
        "startTime": f"{start_date_str} 00:00:00",
        "endTime": f"{end_date_str} 23:59:59",
        "timeUnit": "HOUR"  # Explicitly request HOURLY data
    }
    print(f"Attempting to fetch HOURLY data from SolarEdge API from {start_date_str} to {end_date_str}")
    print(f"Using site timezone for initial parsing: {SOLAREDGE_SITE_TIMEZONE}")

    try:
        response = requests.get(api_url, params=params, timeout=60) # Increased timeout
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from SolarEdge API: {e}")
        return None
    except ValueError as e: # Includes JSONDecodeError
        print(f"Error decoding JSON response from SolarEdge API: {e}")
        print(f"Response text: {response.text[:500]}") # Print part of response if not JSON
        return None

    if "power" not in data or not data["power"]["values"]:
        print(f"No power data found in API response for {start_date_str} to {end_date_str}.")
        return pd.DataFrame(columns=['Timestamp', 'solar_production_kwh']) # Return empty DF

    raw_values = data["power"]["values"]
    if raw_values:
        print(f"Raw timestamp from API for first entry: {raw_values[0]['date']}")

    df = pd.DataFrame(raw_values)
    if df.empty:
        print("DataFrame is empty after initial creation from API values.")
        return pd.DataFrame(columns=['Timestamp', 'solar_production_kwh']) 

    # API 'value' is power in Watts for the interval.
    # Rename 'value' to 'solar_power_watts' for clarity.
    df.rename(columns={'date': 'Timestamp', 'value': 'solar_power_watts'}, inplace=True)
    
    # Convert power from Watts to kiloWatts (kW).
    # pd.to_numeric handles potential nulls or non-numeric data from the API.
    df['solar_power_kw'] = pd.to_numeric(df['solar_power_watts'], errors='coerce') / 1000.0
    
    # Calculate energy in kWh for the 15-minute interval: Energy (kWh) = Power (kW) * Time (h)
    # Time interval is 15 minutes = 15.0/60.0 hours = 0.25 hours.
    df['solar_production_kwh'] = df['solar_power_kw'] * (15.0 / 60.0)
    
    # Drop intermediate columns and rows where conversion might have failed (e.g., if 'value' was null)
    df.drop(columns=['solar_power_watts', 'solar_power_kw'], inplace=True)
    df.dropna(subset=['solar_production_kwh'], inplace=True)

    if df.empty:
        print(f"No valid numeric production data after Watts to kWh conversion for {start_date_str} to {end_date_str}.")
        return pd.DataFrame(columns=['Timestamp', 'solar_production_kwh'])

    # Timestamp processing:
    # 1. Parse string to naive datetime objects
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df.dropna(subset=['Timestamp'], inplace=True) # Drop rows where date parsing failed

    if df.empty:
        print("DataFrame empty after parsing string timestamps.")
        return pd.DataFrame(columns=['Timestamp', 'solar_production_kwh'])

    # 2. Localize naive datetime to site's timezone
    try:
        df['Timestamp'] = df['Timestamp'].dt.tz_localize(SOLAREDGE_SITE_TIMEZONE, ambiguous='infer', nonexistent='shift_forward')
    except pytz.exceptions.UnknownTimeZoneError:
        print(f"ERROR: Unknown Timezone: {SOLAREDGE_SITE_TIMEZONE}. Please check your .env file. Defaulting to UTC localization.")
        df['Timestamp'] = df['Timestamp'].dt.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')
    except Exception as e:
        print(f"Error localizing timestamps to {SOLAREDGE_SITE_TIMEZONE}: {e}. Timestamps might be incorrect.")
        # Potentially return None or raise error, depending on desired strictness
        return None 

    # 3. Convert to UTC
    df['Timestamp'] = df['Timestamp'].dt.tz_convert('UTC')

    # --- BEGIN HOURLY RESAMPLING ---
    if not df.empty and 'Timestamp' in df.columns and 'solar_production_kwh' in df.columns:
        # Store original count for logging if verbose_logging is enabled or for debugging
        # original_point_count = len(df) 
        
        # Set Timestamp as index for resampling. It's already datetime64[ns, UTC].
        df.set_index('Timestamp', inplace=True)
        
        # Resample to hourly ('H'), summing the kWh values. 
        # label='left' ensures the timestamp for the hour is the start of the hour.
        df_resampled = df['solar_production_kwh'].resample('H', label='left').sum()
        
        # Convert the resampled Series back to a DataFrame
        df = df_resampled.reset_index() 
        
        # Ensure Timestamp column is UTC after resampling and reset_index.
        # Typically, reset_index on a TZ-aware DatetimeIndex preserves the TZ.
        # But it's good to be explicit.
        if not df.empty:
            if df['Timestamp'].dt.tz is None: # If it became naive (e.g., due to pandas version/specifics)
                df['Timestamp'] = df['Timestamp'].dt.tz_localize('UTC')
            elif str(df['Timestamp'].dt.tz) != 'UTC': # If it's some other timezone (should not happen here)
                df['Timestamp'] = df['Timestamp'].dt.tz_convert('UTC')
        
        # The existing print statements for data points count, head, and info will now reflect hourly data.
    # --- END HOURLY RESAMPLING ---

    df = df[['Timestamp', 'solar_production_kwh']].sort_values(by='Timestamp').reset_index(drop=True)

    print(f"Successfully fetched and processed {len(df)} data points from SolarEdge for {start_date_str} to {end_date_str}.")
    if not df.empty:
        print("DataFrame head after processing (should be UTC):")
        print(df.head())
        print("DataFrame info:")
        df.info()
    
    return df


def fetch_all_solaredge_data():
    """
    Fetches all available historical data from SolarEdge API, converts to UTC, and saves to OUTPUT_FILE.
    Includes data up to the latest available date from the API (which can be today).
    """
    print("--- Fetch-all mode activated --- Starting to fetch all historical data from SolarEdge ---")
    
    api_start_date, api_end_date = get_solaredge_api_data_period()

    if api_start_date is None or api_end_date is None:
        print("Could not determine API data period from SolarEdge. Aborting fetch-all.")
        return

    # For fetch-all, we want everything the API says it has.
    effective_fetch_start_date = api_start_date
    effective_fetch_end_date = api_end_date 

    print(f"Target fetch range for fetch-all: {effective_fetch_start_date} to {effective_fetch_end_date}.")

    all_data_frames = []
    current_period_start_date = effective_fetch_start_date
    CHUNK_DAYS = 28  # Fetch in approx 4-week chunks

    while current_period_start_date <= effective_fetch_end_date:
        current_period_end_date = current_period_start_date + timedelta(days=CHUNK_DAYS - 1)
        if current_period_end_date > effective_fetch_end_date:
            current_period_end_date = effective_fetch_end_date

        print(f"Fetching SolarEdge chunk: {current_period_start_date.strftime('%Y-%m-%d')} to {current_period_end_date.strftime('%Y-%m-%d')}")
        
        period_df = fetch_solaredge_power_data(
            current_period_start_date.strftime("%Y-%m-%d"),
            current_period_end_date.strftime("%Y-%m-%d")
        )

        if period_df is not None and not period_df.empty:
            all_data_frames.append(period_df)
            # Debug print for each chunk is already in fetch_solaredge_power_data
        elif period_df is None: 
            print(f"WARNING: API call failed for period {current_period_start_date.strftime('%Y-%m-%d')} to {current_period_end_date.strftime('%Y-%m-%d')}.")
        else: # API success, but no data returned
             print(f"No data returned from API for period {current_period_start_date.strftime('%Y-%m-%d')} to {current_period_end_date.strftime('%Y-%m-%d')}.")
        
        current_period_start_date = current_period_end_date + timedelta(days=1)
        # time.sleep(1) # Optional: throttle if hitting API limits

    if not all_data_frames:
        print("No data successfully fetched from SolarEdge after all attempts in fetch-all mode.")
        # Create an empty file or leave it as is? Let's create/overwrite with empty if no data.
        # To ensure the file is overwritten as per user request for fetch-all:
        pd.DataFrame(columns=['Timestamp', 'solar_production_kwh']).to_csv(CURRENT_DIR / OUTPUT_FILE, index=False)
        print(f"Output file {OUTPUT_FILE} has been overwritten (empty as no data fetched).")
        return

    print(f"Concatenating {len(all_data_frames)} fetched dataframes for fetch-all mode...")
    combined_df = pd.concat(all_data_frames, ignore_index=True)
    
    if combined_df.empty:
        print("Combined data from SolarEdge is empty. Nothing to save.")
        pd.DataFrame(columns=['Timestamp', 'solar_production_kwh']).to_csv(CURRENT_DIR / OUTPUT_FILE, index=False)
        print(f"Output file {OUTPUT_FILE} has been overwritten (empty as combined data was empty).")
        return
        
    # Ensure Timestamps are UTC and drop duplicates
    if not isinstance(combined_df['Timestamp'].dtype, pd.DatetimeTZDtype) or str(combined_df['Timestamp'].dt.tz) != 'UTC':
         print("Warning: Timestamps in combined_df are not consistently UTC-aware. Attempting conversion...")
         combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], errors='coerce').dt.tz_convert('UTC')
         combined_df.dropna(subset=['Timestamp'], inplace=True)

    combined_df.sort_values('Timestamp', inplace=True)
    initial_rows = len(combined_df)
    combined_df.drop_duplicates(subset=['Timestamp'], keep='first', inplace=True)
    deduplicated_count = initial_rows - len(combined_df)
    if deduplicated_count > 0:
        print(f"Removed {deduplicated_count} duplicate rows based on 'Timestamp' for fetch-all.")
    print(f"Number of rows after deduplication: {len(combined_df)}")

    if not combined_df.empty:
        print("Final combined DataFrame head before saving (fetch-all, should be UTC):")
        print(combined_df.head())
        print("Final combined DataFrame info (fetch-all, should be UTC):")
        combined_df.info()
        
        # Perform gap analysis (optional, can be kept if useful)
        # ... (existing gap analysis code can be re-inserted here if desired, ensuring it works with UTC timestamps) ...
        # For now, skipping re-insertion of gap analysis to simplify, can be added back.
        pass # Placeholder for gap analysis if re-added

    # Save to CSV, Timestamps as UTC strings
    output_path = CURRENT_DIR / OUTPUT_FILE
    combined_df['Timestamp'] = combined_df['Timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    
    try:
        combined_df.to_csv(output_path, index=False) # Overwrites existing file
        print(f"Successfully saved all fetched historical SolarEdge data ({len(combined_df)} rows) to {output_path}")
        if not combined_df.empty:
            print(f"Data range in CSV: {combined_df['Timestamp'].min()} to {combined_df['Timestamp'].max()}")
    except Exception as e:
        print(f"Error saving fetch-all data to {output_path}: {e}")
    print("--- Fetch-all mode completed ---")


def calculate_latest_timestamp_from_csv(file_path: Path) -> pd.Timestamp | None:
    """Reads the CSV, parses 'Timestamp' as UTC, and returns the max Timestamp or None."""
    if not file_path.exists():
        print(f"Data file not found: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
        if df.empty or 'Timestamp' not in df.columns:
            print(f"Data file is empty or missing 'Timestamp' column: {file_path}")
            return None
        
        # Ensure Timestamp is string type before attempting string operations
        df['Timestamp'] = df['Timestamp'].astype(str)
        
        # The format YYYY-MM-DDTHH:MM:SS.000Z is ISO 8601 and directly parseable by pd.to_datetime with UTC inference.
        # Explicitly handle cases where it might not be perfectly formatted or to ensure UTC.
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df.dropna(subset=['Timestamp'], inplace=True)

        if df.empty or df['Timestamp'].isnull().all():
             print(f"No valid timestamps found in {file_path} after parsing.")
             return None

        # Ensure all timestamps are UTC. If naive, assume UTC. If localized, convert to UTC.
        if df['Timestamp'].dt.tz is None:
            df['Timestamp'] = df['Timestamp'].dt.tz_localize('UTC')
        else:
            df['Timestamp'] = df['Timestamp'].dt.tz_convert('UTC')
            
        latest_timestamp = df['Timestamp'].max()
        if pd.isna(latest_timestamp):
            print(f"Could not determine a valid latest timestamp from {file_path} (all NaT after processing).")
            return None
        
        print(f"Latest timestamp from CSV ({file_path}): {latest_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        return latest_timestamp
    except Exception as e:
        print(f"Error reading or parsing timestamps from {file_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Fetch and merge solar production data.")
    parser.add_argument(
        "--fetch-all",
        action="store_true",
        help="Fetch all available historical data from SolarEdge API and overwrite existing file."
    )
    args = parser.parse_args()

    output_path = CURRENT_DIR / OUTPUT_FILE
    DATA_DIR.mkdir(exist_ok=True, parents=True) # Ensure HA data dir exists for fallback

    if args.fetch_all:
        fetch_all_solaredge_data() # This function now handles its own printing and saving
        return

    # --- Regular daily update mode --- 
    print(f"--- Starting daily update process for solar production data ({output_path}) ---")
    
    latest_timestamp_in_file_utc = calculate_latest_timestamp_from_csv(output_path)
    
    # Determine start date for fetching
    fetch_start_date_obj = None
    if latest_timestamp_in_file_utc is None:
        print("No valid latest timestamp in CSV or file doesn't exist. Attempting to fetch from API start date.")
        api_start_date, _ = get_solaredge_api_data_period()
        if api_start_date:
            fetch_start_date_obj = api_start_date
        else:
            # Fallback if API period also fails
            print("Could not get API start date, defaulting to 30 days ago.")
            fetch_start_date_obj = datetime.now(timezone.utc).date() - timedelta(days=30)
    else:
        # Fetch from the date of the last record to potentially update that day and get subsequent days.
        fetch_start_date_obj = latest_timestamp_in_file_utc.date() 

    # Determine end date for fetching (always today UTC)
    fetch_target_end_date_obj = datetime.now(timezone.utc).date()

    if fetch_start_date_obj > fetch_target_end_date_obj:
        print(f"Latest data in file ({fetch_start_date_obj}) is already up to or newer than target end date ({fetch_target_end_date_obj}). No new SolarEdge data needed.")
        new_solar_data_df = None
    else:
        fetch_start_date_str = fetch_start_date_obj.strftime("%Y-%m-%d")
        fetch_target_end_date_str = fetch_target_end_date_obj.strftime("%Y-%m-%d")
        print(f"Attempting to fetch new SolarEdge data from {fetch_start_date_str} to {fetch_target_end_date_str}.")
        new_solar_data_df = fetch_solaredge_power_data(fetch_start_date_str, fetch_target_end_date_str)

    final_df_to_process = None
    source_name = ""

    if new_solar_data_df is not None and not new_solar_data_df.empty:
        print(f"Successfully fetched {len(new_solar_data_df)} new/updated rows from SolarEdge.")
        final_df_to_process = new_solar_data_df
        source_name = "SolarEdge API"
    else:
        print("No new data fetched from SolarEdge API or API fetch failed.")
        # --- Attempt Home Assistant Fallback --- 
        # This part requires careful thought on date ranges if SolarEdge fails partially.
        # For simplicity now, if SolarEdge fails, HA tries to fetch data for a fixed recent period.
        # A more robust HA fallback would need to calculate its required range based on latest_timestamp_in_file_utc.
        print("\nAttempting to fetch new data from Home Assistant as a fallback...")
        
        # Determine days needed for HA based on the original latest_timestamp_in_file_utc
        days_for_ha_fallback = 30 # Default
        if latest_timestamp_in_file_utc:
            days_diff = (datetime.now(timezone.utc).date() - latest_timestamp_in_file_utc.date()).days
            days_for_ha_fallback = max(1, days_diff + 1) # Fetch at least 1 day, up to today
        else: # No existing CSV, HA fetches default period
            print("No existing CSV for HA fallback reference, fetching default 30 days for HA.")

        print(f"HA fallback will attempt to download data for the last {days_for_ha_fallback} days.")
        download_new_data(days_for_ha_fallback) # This function needs to exist and work
    
        entity_name_for_file = ENTITY_ID.split(".", 1)[1] if "." in ENTITY_ID else ENTITY_ID
        downloaded_ha_file = DATA_DIR / f"{entity_name_for_file}_{RESOLUTION}.csv"

        if downloaded_ha_file.exists():
            try:
                ha_data_raw = pd.read_csv(downloaded_ha_file)
                if not ha_data_raw.empty:
                    # HA data needs its own robust UTC conversion if not already UTC
                    # Assuming transform_data_format handles conversion to our standard UTC DataFrame
                    print("Processing HA data...")
                    ha_df_transformed = transform_data_format(ha_data_raw) # MUST return UTC-aware Timestamp
                    if ha_df_transformed is not None and not ha_df_transformed.empty:
                        if not isinstance(ha_df_transformed['Timestamp'].dtype, pd.DatetimeTZDtype) or str(ha_df_transformed['Timestamp'].dt.tz) != 'UTC':
                            print("WARNING: transform_data_format for HA did not return UTC-aware timestamps. Attempting conversion.")
                            ha_df_transformed['Timestamp'] = pd.to_datetime(ha_df_transformed['Timestamp']).dt.tz_convert('UTC')
                        final_df_to_process = ha_df_transformed
                        source_name = "Home Assistant"
                        print(f"Successfully loaded and transformed {len(final_df_to_process)} rows from Home Assistant.")
                    else:
                        print("HA data transformation resulted in empty data.")
                else:
                    print(f"Home Assistant data file {downloaded_ha_file} is empty.")
            except Exception as e:
                print(f"Error loading or transforming Home Assistant data from {downloaded_ha_file}: {e}")
        else:
            print(f"Home Assistant downloaded data file not found: {downloaded_ha_file}")
    # --- End of HA Fallback ---

    if final_df_to_process is not None and not final_df_to_process.empty:
        print(f"\nProceeding to merge data obtained from {source_name}...")
        merge_data(output_path, final_df_to_process)
    elif output_path.exists():
        print("\nNo new data successfully fetched or processed from any source. Existing data file remains unchanged.")
    else:
        print("\nNo new data successfully fetched and no existing data file. Nothing to save.")
    
    print(f"--- Daily update process finished. Check {output_path} for results. ---")


def download_new_data(days):
    """Run the HA download script to fetch new data for a number of past days"""
    try:
        cmd = [
            sys.executable,
            str(DOWNLOAD_SCRIPT),
            "--entity", ENTITY_ID,
            "--days", str(days),
            "--res", RESOLUTION
        ]
        
        print(f"Running Home Assistant download command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("HA Download script output:")
        print(result.stdout)
        
        if result.stderr:
            print("HA Download script errors:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"Error running Home Assistant download script: {e}")
        print(f"Script output: {e.output}")
        print(f"Script errors: {e.stderr}")
    except FileNotFoundError:
        print(f"Error: The download script {DOWNLOAD_SCRIPT} was not found.")


def merge_data(output_path: Path, new_data_df: pd.DataFrame):
    """
    Merges new UTC data with existing UTC data from CSV, saves as UTC.
    New data df must have 'Timestamp' (UTC datetime) and 'solar_production_kwh'.
    """
    print(f"Preparing to merge {len(new_data_df)} new rows.")
    if not isinstance(new_data_df['Timestamp'].dtype, pd.DatetimeTZDtype) or str(new_data_df['Timestamp'].dt.tz) != 'UTC':
        print("ERROR in merge_data: new_data_df['Timestamp'] is not UTC-aware. Aborting merge.")
        # Optionally, attempt to convert/localize, but safer to ensure it comes in correctly.
        # new_data_df['Timestamp'] = pd.to_datetime(new_data_df['Timestamp']).dt.tz_convert('UTC') # Risky if not aware of source tz
        return

    new_data_df['solar_production_kwh'] = pd.to_numeric(new_data_df['solar_production_kwh'], errors='coerce').fillna(0.0)
    new_data_df.dropna(subset=['Timestamp', 'solar_production_kwh'], inplace=True)

    existing_data_list = []
    if output_path.exists():
        try:
            # Use the robust CSV timestamp reader
            existing_df = pd.read_csv(output_path)
            if not existing_df.empty and 'Timestamp' in existing_df.columns:
                existing_df['Timestamp'] = existing_df['Timestamp'].astype(str)
                existing_df['Timestamp'] = pd.to_datetime(existing_df['Timestamp'], errors='coerce')
                existing_df.dropna(subset=['Timestamp'], inplace=True)
                if not existing_df.empty:
                    if existing_df['Timestamp'].dt.tz is None:
                        existing_df['Timestamp'] = existing_df['Timestamp'].dt.tz_localize('UTC')
                    else:
                        existing_df['Timestamp'] = existing_df['Timestamp'].dt.tz_convert('UTC')
                    existing_df['solar_production_kwh'] = pd.to_numeric(existing_df['solar_production_kwh'], errors='coerce')
                    existing_df.dropna(subset=['Timestamp','solar_production_kwh'], inplace=True)
                    existing_data_list.append(existing_df)
                    print(f"Loaded {len(existing_df)} valid rows from existing data file {output_path}.")
            else:
                print(f"Existing data file {output_path} is empty or has incorrect format.")
        except Exception as e:
            print(f"Error loading existing data from {output_path}: {e}. It will be ignored/overwritten if new data exists.")
    else:
        print(f"No existing data file at {output_path}. Will save new data directly.")

    all_dfs_to_combine = existing_data_list + [new_data_df]
    combined_data = pd.concat(all_dfs_to_combine, ignore_index=True)

    if combined_data.empty:
        print("No data to save after combining existing and new data.")
        return

    # Ensure all timestamps are proper datetimes and UTC before deduplication
    combined_data['Timestamp'] = pd.to_datetime(combined_data['Timestamp'])
    if combined_data['Timestamp'].dt.tz is None:
        combined_data['Timestamp'] = combined_data['Timestamp'].dt.tz_localize('UTC')
    else:
        combined_data['Timestamp'] = combined_data['Timestamp'].dt.tz_convert('UTC')

    combined_data.sort_values(by='Timestamp', inplace=True)
    # Keep the 'last' entry for a given timestamp, which should favor new_data_df if timestamps overlap
    initial_rows = len(combined_data)
    combined_data.drop_duplicates(subset=['Timestamp'], keep='last', inplace=True)
    deduplicated_count = initial_rows - len(combined_data)
    if deduplicated_count > 0:
        print(f"Removed {deduplicated_count} duplicate/overlapping rows based on Timestamp.")

    combined_data.sort_values('Timestamp', inplace=True) # Sort again after potential drops
    
    # Format Timestamp as ISO 8601 UTC string for CSV output
    combined_data['Timestamp'] = combined_data['Timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    
    try:
        combined_data.to_csv(output_path, index=False)
        print(f"Successfully merged and saved data. Total rows in {output_path}: {len(combined_data)}")
        if not combined_data.empty:
            print(f"Data range in CSV: {combined_data['Timestamp'].min()} to {combined_data['Timestamp'].max()}")
    except Exception as e:
        print(f"Error saving combined data to {output_path}: {e}")


def transform_data_format(data_ha_style: pd.DataFrame) -> pd.DataFrame | None:
    """
    Transform data from Home Assistant style (timestamp, state) to 
    the standard format (Timestamp, solar_production_kwh) with UTC timestamps.
    Assumes HA 'timestamp' is ISO8601 format or parsable by pd.to_datetime.
    HA timestamps are often UTC, but if they are naive or local, this function
    attempts to standardize them to UTC.
    """
    if data_ha_style.empty or 'timestamp' not in data_ha_style.columns or 'state' not in data_ha_style.columns:
        print("HA data is empty or missing required columns ('timestamp', 'state').")
        return None

    data_transformed = data_ha_style.rename(columns={'timestamp': 'Timestamp'})
    
    # Ensure Timestamp is parsed as datetime
    data_transformed['Timestamp'] = pd.to_datetime(data_transformed['Timestamp'], errors='coerce')
    data_transformed.dropna(subset=['Timestamp'], inplace=True)
    if data_transformed.empty:
        print("HA data empty after initial Timestamp parsing.")
        return None

    # Timezone handling for HA data:
    # 1. If timezone-aware, convert to UTC.
    # 2. If naive, assume it's local time as per HA config (often UTC by default, but could be system local).
    #    For robustness, we'll assume naive timestamps from HA are UTC. If they are local, this might need adjustment
    #    or a specific timezone configured for HA data source.
    if data_transformed['Timestamp'].dt.tz is not None:
        print("HA Timestamps are timezone-aware. Converting to UTC.")
        data_transformed['Timestamp'] = data_transformed['Timestamp'].dt.tz_convert('UTC')
    else:
        print("HA Timestamps are naive. Assuming UTC and localizing.")
        try:
            data_transformed['Timestamp'] = data_transformed['Timestamp'].dt.tz_localize('UTC')
        except Exception as e: # Catches errors like re-localizing already localized data if logic is complex
            print(f"Error localizing naive HA timestamps to UTC: {e}. Attempting UTC conversion if already localized somehow.")
            # As a fallback, try to force conversion if it was an unexpected type
            try:
                 data_transformed['Timestamp'] = pd.to_datetime(data_transformed['Timestamp']).dt.tz_convert('UTC')
            except Exception as e2:
                 print(f"Could not convert HA timestamps to UTC: {e2}. Returning None.")
                 return None

    data_transformed['solar_production_kwh'] = pd.to_numeric(data_transformed['state'], errors='coerce') / 1000.0
    data_transformed.dropna(subset=['solar_production_kwh', 'Timestamp'], inplace=True)
    
    data_transformed = data_transformed[['Timestamp', 'solar_production_kwh']]
    print(f"Transformed HA data. {len(data_transformed)} rows. Head (UTC):")
    if not data_transformed.empty:
        print(data_transformed.head())
    return data_transformed

if __name__ == "__main__":
    main()
