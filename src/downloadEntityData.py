#!/usr/bin/env python3
"""
Home Assistant Entity Data Downloader
====================================

This script downloads historical sensor data from a Home Assistant instance, processes it 
to calculate averages at different time resolutions, and saves the data to a CSV file.

Usage:
------
1. Basic usage (defaults to raw data with complete time coverage):
   python downloadEntityData.py
   
2. Specify a different entity to download:
     python downloadEntityData.py --entity entity_id
   
3. Specify number of days to retrieve (e.g., last 7 days):
   python downloadEntityData.py --days 7
   
4. Specify data resolution (raw, hourly, or daily):
   python downloadEntityData.py --res hourly
   python downloadEntityData.py --res daily
   
5. Combine options:
   python downloadEntityData.py --entity entity_id --days 10 --res hourly
   
All entity IDs should be provided with their full domain prefix (e.g., 'sensor.temperature', 'binary_sensor.motion', 'device_tracker.phone').

Output:
-------
Data is saved to a CSV file in the "Data/HomeAssistant" folder with naming format:
"[entity_id].csv" for raw data, "[entity_id]_hourly.csv" for hourly data, or "[entity_id]_daily.csv" for daily data.

Each file contains columns:
- timestamp: The timestamp in ISO format
- state: The numerical value

For all resolutions, the script ensures complete time coverage for the entire requested date range:
- Raw: Preserves all original data points
- Hourly: Creates one timestamp per hour with averaged values
- Daily: Creates one timestamp per day with averaged values

Example output file contents:
For raw data:
timestamp,state
2023-11-01T00:05:23,120.5
2023-11-01T00:10:45,118.7
2023-11-01T00:15:12,121.3
...

For hourly data (--res hourly):
timestamp,state
2023-11-01T00:00:00,119.8
2023-11-01T01:00:00,115.3
2023-11-01T02:00:00,0
...

For daily data (--res daily):
timestamp,state
2023-11-01T00:00:00,118.2
2023-11-02T00:00:00,120.6
2023-11-03T00:00:00,0
...

Notes:
------
- The script aggregates multiple readings by calculating mean values
- Non-numeric states are filtered out for hourly and daily resolutions
- The date range includes full days from midnight to midnight
- When using --latest_value, only the state is printed to stdout. 
  All other logs/errors go to stderr.
"""

import requests
import argparse
import json
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os
from pathlib import Path
import pandas as pd
import traceback
import sys
import time


# Home Assistant URL and token
project_root = Path(__file__).resolve().parents[1]
dotenv_path = project_root / 'api.env'
load_dotenv(dotenv_path=dotenv_path)
token = os.getenv('HOMEASSISTANT_TOKEN_DataDownloadToken')
ha_ip = os.getenv('HOMEASSISTANT_IP')

# Check if required environment variables are available
if not token or not ha_ip:
    print("Error: Missing required environment variables.", file=sys.stderr)
    print("Please create an api.env file containing:", file=sys.stderr)
    print("HOMEASSISTANT_TOKEN_DataDownloadToken=your_token_here", file=sys.stderr)
    print("HOMEASSISTANT_IP=your_home_assistant_ip_here", file=sys.stderr)
    print("\nOr set these environment variables directly.", file=sys.stderr)
    sys.exit(1)

ha_url = f"http://{ha_ip}:8123"


def get_latest_entity_state(entity_id: str) -> str | None:
    """Fetches the current state of a single entity."""
    url = f"{ha_url}/api/states/{entity_id}"
    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.ok:
            data = response.json()
            return data.get("state")
        else:
            print(f"Error fetching latest state for {entity_id}: {response.status_code} {response.text}", file=sys.stderr)
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed for latest state of {entity_id}: {e}", file=sys.stderr)
        return None

def fetch_entity_data(entity_id, days_back, resolution):
    try:
        # Calculate the start time based on days_back
        now = datetime.now()
        # print(f"Debug - Current time: {now}", file=sys.stderr) # Keep logs to stderr for this path
        
        # Calculate days back properly - subtract the full number of days requested
        start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_back)
        # Calculate end time to end of current day
        end = now.replace(hour=23, minute=59, second=59, microsecond=999999)

        # print(f"Debug - Start date: {start}", file=sys.stderr)
        # print(f"Debug - End date: {end}", file=sys.stderr)

        # Format timestamps in ISO8601 format that Home Assistant expects
        start_iso = start.strftime("%Y-%m-%dT00:00:00")  # Explicitly set to start of day
        end_iso = end.strftime("%Y-%m-%dT23:59:59")      # Explicitly set to end of day

        print(f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr)
        print(f"Requesting data for: {days_back} days (from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')})", file=sys.stderr)
        print(f"Entity: {entity_id}", file=sys.stderr)
        print(f"Resolution: {resolution}", file=sys.stderr)

        # Build the URL for the history API with explicit start and end times
        # We disable minimal_response and significant_changes_only to get ALL data points
        url = f"{ha_url}/api/history/period/{start_iso}?end_time={end_iso}&filter_entity_id={entity_id}&minimal_response=false&significant_changes_only=false"
        # print(f"Request URL: {url}", file=sys.stderr)

        headers = {
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        }

        print("Sending API request for historical data...", file=sys.stderr)
        response = requests.get(url, headers=headers)
        print(f"API response status code: {response.status_code}", file=sys.stderr)
        
        if response.ok:
            data = response.json()
            # print(f"API returned data of type: {type(data)}", file=sys.stderr)
            # print(f"API data length: {len(data) if isinstance(data, list) else 'not a list'}", file=sys.stderr)
            
            if not data or len(data) == 0:
                print(f"No data returned for entity: {entity_id}", file=sys.stderr)
                # For historical, we might still want to create an empty CSV or a CSV with headers
                # Depending on desired behavior. For now, exiting.
                # sys.exit(0) 
                # Create an empty df to signify no data, but allow file creation
                result_df = pd.DataFrame(columns=['timestamp', 'state'])

            else: # data is not empty
                try:
                    print("Processing historical data...", file=sys.stderr)
                    # Debug: Print first items to understand structure
                    # print(f"First data item type: {type(data[0])}", file=sys.stderr)
                    
                    # Extract entity data from API response
                    if isinstance(data, list) and len(data) > 0:
                        if isinstance(data[0], list):
                            # print(f"Found list-of-lists format with {len(data)} items", file=sys.stderr)
                            for i, item in enumerate(data):
                                if isinstance(item, list) and len(item) > 0:
                                    # print(f"Using item {i} with {len(item)} records", file=sys.stderr)
                                    raw_df = pd.DataFrame(item)
                                    break
                            else:
                                print("No non-empty lists found in data", file=sys.stderr)
                                result_df = pd.DataFrame(columns=['timestamp', 'state']) # empty df
                                # sys.exit(1) # Or handle gracefully
                        elif isinstance(data[0], dict):
                            # print("Found direct list of dictionaries", file=sys.stderr)
                            raw_df = pd.DataFrame(data)
                        else:
                            print(f"Unexpected data format: {type(data[0])}", file=sys.stderr)
                            result_df = pd.DataFrame(columns=['timestamp', 'state']) # empty df
                            # sys.exit(1) # Or handle gracefully
                    else:
                        print("Data is not a list or is empty", file=sys.stderr)
                        result_df = pd.DataFrame(columns=['timestamp', 'state']) # empty df
                        # sys.exit(1) # Or handle gracefully
                    
                    if 'raw_df' not in locals() or raw_df.empty: # Check if raw_df was successfully created
                         if result_df.empty: # if it's already set to empty, use it
                            pass
                         else: # otherwise, initialize to empty
                            result_df = pd.DataFrame(columns=['timestamp', 'state'])
                    else: # raw_df exists and is not empty
                        # print(f"Raw DataFrame created with {len(raw_df)} rows", file=sys.stderr)
                        # print(f"Columns: {raw_df.columns.tolist()}", file=sys.stderr)
                        
                        # Make a copy of the raw data to preserve it
                        # orig_df = raw_df.copy() # Not used currently
                        
                        # Check for required columns
                        if 'last_changed' not in raw_df.columns:
                            print(f"Warning: 'last_changed' column not found. Available columns: {raw_df.columns.tolist()}", file=sys.stderr)
                            time_cols = [col for col in raw_df.columns if 'time' in col.lower() or 'date' in col.lower()]
                            if time_cols:
                                print(f"Using '{time_cols[0]}' as timestamp column", file=sys.stderr)
                                raw_df = raw_df.rename(columns={time_cols[0]: 'last_changed'})
                            else:
                                print("No suitable timestamp column found", file=sys.stderr)
                                result_df = pd.DataFrame(columns=['timestamp', 'state']) # empty df
                                # sys.exit(1)
                        
                        if 'state' not in raw_df.columns and 'result_df' not in locals(): # only if result_df not already set
                            print(f"Warning: 'state' column not found. Available columns: {raw_df.columns.tolist()}", file=sys.stderr)
                            value_cols = [col for col in raw_df.columns if 'state' in col.lower() or 'value' in col.lower()]
                            if value_cols:
                                print(f"Using '{value_cols[0]}' as state column", file=sys.stderr)
                                raw_df = raw_df.rename(columns={value_cols[0]: 'state'})
                            else:
                                print("No suitable state column found", file=sys.stderr)
                                result_df = pd.DataFrame(columns=['timestamp', 'state']) # empty df
                                # sys.exit(1)
                        
                        # Now process based on resolution, only if result_df isn't already set (e.g. from error)
                        if 'result_df' not in locals() or result_df.empty and not raw_df.empty:
                            if resolution == 'raw':
                                print("Processing raw data...", file=sys.stderr)
                                result_df = raw_df[['last_changed', 'state']].copy()
                                result_df.rename(columns={'last_changed': 'timestamp'}, inplace=True)
                                result_df['timestamp'] = pd.to_datetime(result_df['timestamp'], format='ISO8601')
                                result_df = result_df.sort_values('timestamp')
                                # print(f"Raw data has {len(result_df)} data points", file=sys.stderr)
                                # if not result_df.empty:
                                #     print(f"Time range: {result_df['timestamp'].min()} to {result_df['timestamp'].max()}", file=sys.stderr)
                            
                            else:  # hourly or daily
                                print(f"Processing {resolution} data...", file=sys.stderr)
                                df_proc = raw_df.copy() # Use a different name to avoid confusion
                                df_proc['last_changed'] = pd.to_datetime(df_proc['last_changed'], format='ISO8601')
                                
                                # if not df_proc.empty:
                                    # print("\nDEBUG INFO - Original timestamps:", file=sys.stderr)
                                    # print(f"First timestamp: {df_proc['last_changed'].iloc[0]}", file=sys.stderr)
                                    # print(f"Timestamp timezone info: {str(df_proc['last_changed'].dt.tz) if hasattr(df_proc['last_changed'].dt, 'tz') and df_proc['last_changed'].dt.tz else 'None'}", file=sys.stderr)
                                    # current_time_debug = datetime.now() # local 'now'
                                    # print(f"Current system time: {current_time_debug}", file=sys.stderr)
                                    # print(f"Current system timezone: {datetime.now().astimezone().tzinfo}", file=sys.stderr)
                                    
                                    # first_timestamp = df_proc['last_changed'].iloc[0]
                                    # if hasattr(first_timestamp, 'year') and hasattr(current_time_debug, 'year') and first_timestamp.year > current_time_debug.year + 1:
                                    #     print(f"WARNING: Timestamps appear to be from the future year {first_timestamp.year}.", file=sys.stderr)
                                    #     print(f"System year: {current_time_debug.year}, Data year: {first_timestamp.year}", file=sys.stderr)
                                
                                df_proc['state'] = pd.to_numeric(df_proc['state'], errors='coerce')
                                original_len = len(df_proc)
                                df_proc = df_proc.dropna(subset=['state'])
                                # if len(df_proc) < original_len:
                                #     print(f"Dropped {original_len - len(df_proc)} rows with non-numeric values", file=sys.stderr)
                                
                                if resolution == 'hourly':
                                    df_proc['timestamp'] = df_proc['last_changed'].dt.floor('h')
                                    freq = 'h'
                                else:  # daily
                                    df_proc['timestamp'] = df_proc['last_changed'].dt.floor('D')
                                    freq = 'D'
                                
                                result_df_agg = df_proc.groupby('timestamp')['state'].mean().reset_index()
                                
                                start_naive = start # Use original 'start' from function args
                                end_naive = end     # Use original 'end' from function args
                                
                                if hasattr(result_df_agg['timestamp'].dt, 'tz') and result_df_agg['timestamp'].dt.tz is not None:
                                    # print("Converting timezone-aware timestamps to naive for range creation", file=sys.stderr)
                                    try:
                                        result_df_agg['timestamp'] = result_df_agg['timestamp'].dt.tz_convert('Europe/Stockholm').dt.tz_localize(None)
                                    except Exception as e_tz:
                                        print(f"Error converting timezone for aggregation: {e_tz}", file=sys.stderr)
                                        result_df_agg['timestamp'] = result_df_agg['timestamp'].dt.tz_localize(None) # Fallback: just strip
                                
                                all_periods = pd.date_range(start=start_naive.replace(tzinfo=None), end=end_naive.replace(tzinfo=None), freq=freq)
                                template_df = pd.DataFrame({'timestamp': all_periods})
                                
                                # Merge template with aggregated data
                                result_df = pd.merge(template_df, result_df_agg, on='timestamp', how='left')
                                result_df['state'] = result_df['state'].fillna(0)
                                # print(f"{resolution.capitalize()} data has {len(result_df)} data points after merge", file=sys.stderr)
                                # if not result_df.empty:
                                #     print(f"Time range after merge: {result_df['timestamp'].min()} to {result_df['timestamp'].max()}", file=sys.stderr)
                    
                    # Common saving logic for historical data
                    output_dir = Path("Data/HomeAssistant") # Relative to where script is run or CWD
                    output_dir.mkdir(parents=True, exist_ok=True) # Use parents=True
                    
                    entity_name_part = entity_id.split(".", 1)[1] if "." in entity_id else entity_id
                    res_suffix = f"_{resolution}" if resolution != "raw" else ""
                    output_filename = f'{entity_name_part}{res_suffix}.csv'
                    output_file = output_dir / output_filename
                    
                    if not result_df.empty and pd.api.types.is_datetime64_any_dtype(result_df['timestamp']):
                        if result_df['timestamp'].dt.tz is not None:
                            try:
                                result_df['timestamp'] = result_df['timestamp'].dt.tz_convert('Europe/Stockholm').dt.tz_localize(None)
                            except Exception as e_save_tz:
                                print(f"Error converting output timezone for saving: {e_save_tz}", file=sys.stderr)
                                result_df['timestamp'] = result_df['timestamp'].dt.tz_localize(None) # Fallback

                        # Filter out future dates robustly
                        # current_time_for_filter = datetime.now(timezone.utc) # Ensure timezone aware comparison
                        # future_cutoff_for_filter = current_time_for_filter + timedelta(days=1)
                        # result_df['timestamp'] = pd.to_datetime(result_df['timestamp']).dt.tz_localize('UTC', ambiguous='infer') # Assume UTC if naive
                        
                        # To avoid timezone issues here, compare with naive datetime if timestamps are naive
                        # current_time_naive = datetime.now().replace(tzinfo=None)
                        # future_cutoff_naive = current_time_naive + timedelta(days=1)
                        # future_dates_mask = result_df['timestamp'] > future_cutoff_naive
                        # future_dates_count_val = future_dates_mask.sum()

                        # if future_dates_count_val > 0:
                        #     print(f"WARNING: Found {future_dates_count_val} timestamps in the future. Removing them.", file=sys.stderr)
                        #     result_df = result_df[~future_dates_mask]
                    
                    result_df.to_csv(output_file, index=False)
                    print(f"Saved {len(result_df)} rows to {output_file}", file=sys.stderr)
                    
                except Exception as e_proc:
                    print(f"Error processing historical data: {str(e_proc)}", file=sys.stderr)
                    print("Exception details:", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    if 'data' in locals():
                         print("First few records of data received:", file=sys.stderr)
                         print(str(data)[:1000], file=sys.stderr)
                    # sys.exit(1) # Decide if failure here should stop script for historical
        else: # response not ok for historical
            print("Error fetching historical data:", response.status_code, response.text, file=sys.stderr)
            print(f"URL: {url}", file=sys.stderr)
            # sys.exit(1) # Decide if failure here should stop script

    except Exception as e_outer:
        print(f"Unexpected error in fetch_entity_data: {str(e_outer)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


def update_entity(entity_id: str) -> bool:
    """
    Call the homeassistant.update_entity service to force-refresh a single entity.
    Returns True on HTTP 200, False otherwise.
    """
    url = f"{ha_url}/api/services/homeassistant/update_entity"
    # print(f"Request URL: {url} for {entity_id}", file=sys.stderr)
    
    payload = {"entity_id": entity_id}
    
    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
    }

    # print("Sending API update entity request...", file=sys.stderr)
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        # print(f"API response status code for update: {resp.status_code}", file=sys.stderr)
        return resp.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Request failed during update_entity for {entity_id}: {e}", file=sys.stderr)
        return False

def main():
    # Check system time
    current_time_main = datetime.now()
    print(f"System time check: Current date and time is {current_time_main}", file=sys.stderr)
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Download entity data from Home Assistant.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
        python downloadEntityData.py --entity sensor.solar_power_hourly_average --days 1
        python downloadEntityData.py --entity sensor.electricity_consumption --days 7 --res hourly
        python downloadEntityData.py --entity binary_sensor.water_consumption --days 30 --res daily
        python downloadEntityData.py --entity sensor.battery_soc --latest_value --refresh
        """
    )
    parser.add_argument('--entity', type=str, default='sensor.solar_power_hourly_average',
                        help='Full entity ID (e.g., sensor.temperature) or alias ("battery_soc").')
    parser.add_argument('--refresh', action='store_true', default=False,
                        help='Refresh the entity data before fetching (applies to both latest_value and historical).')
    parser.add_argument('--days', type=int, default=1,
                        help='Number of days for historical data (default: 1). Ignored if --latest_value is used.')
    parser.add_argument('--res', type=str, choices=['raw', 'hourly', 'daily'], default='raw',
                        help='Resolution for historical data (default: raw). Ignored if --latest_value is used.')
    parser.add_argument('--latest_value', action='store_true', default=False,
                        help='Fetch only the latest value and print to stdout. Ignores --days and --res.')
    args = parser.parse_args()

    # Resolve entity_id, including alias
    entity_to_query = args.entity
    # Standardize known aliases
    if args.entity.lower() == "battery_soc" or args.entity.lower() == "sensor.battery_soc":
        entity_to_query = "sensor.sonnenbatterie_307421_state_battery_percentage_user"
    # Add other aliases here if needed:
    # elif args.entity.lower() == "another_alias":
    #     entity_to_query = "actual.sensor_id_for_alias"

    if args.latest_value:
        # Handle fetching and printing only the latest value
        if args.refresh:
            print(f"Refreshing entity: {entity_to_query}...", file=sys.stderr)
            if not update_entity(entity_to_query):
                print(f"Warning: Failed to trigger update for {entity_to_query}. Attempting to fetch current state anyway.", file=sys.stderr)
            else:
                print(f"Update triggered for {entity_to_query}. Waiting 5 seconds...", file=sys.stderr)
                time.sleep(5) # Wait for Home Assistant to poll/update

        # print(f"Fetching latest state for: {entity_to_query}...", file=sys.stderr) # Redundant if get_latest_entity_state logs
        current_state = get_latest_entity_state(entity_to_query)

        if current_state is not None:
            print(current_state) # Print only the state string to stdout
            sys.exit(0)
        else:
            # get_latest_entity_state already prints error to stderr
            sys.exit(1)
    else:
        # Existing logic for fetching historical data and saving to CSV
        print(f"Fetching historical data for entity: {entity_to_query}", file=sys.stderr)
        if args.refresh:
            print(f"Refreshing entity: {entity_to_query} before fetching history...", file=sys.stderr)
            if not update_entity(entity_to_query):
                print(f"Warning: Failed to trigger update for {entity_to_query} before fetching history.", file=sys.stderr)
            else:
                print(f"Update triggered for {entity_to_query}. Waiting 5 seconds...", file=sys.stderr)
                time.sleep(5)
        
        fetch_entity_data(entity_to_query, args.days, args.res)

if __name__ == "__main__":
    main()
