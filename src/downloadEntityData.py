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

# Check system time
current_time = datetime.now()
print(f"System time check: Current date and time is {current_time}")
# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Download entity data from Home Assistant and calculate hourly/daily averages',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
    python downloadEntityData.py --entity sensor.solar_power_hourly_average --days 1
    python downloadEntityData.py --entity sensor.electricity_consumption --days 7 --res hourly
    python downloadEntityData.py --entity binary_sensor.water_consumption --days 30 --res daily
    """
)
parser.add_argument('--entity', type=str, default='sensor.solar_power_hourly_average',
                    help='Full entity ID to fetch data for (including domain prefix like sensor., binary_sensor., etc.)')
parser.add_argument('--days', type=int, default=1,
                    help='Number of days to look back from today (default: 1 day)')
parser.add_argument('--res', type=str, choices=['raw', 'hourly', 'daily'], default='raw',
                    help='Data resolution: "raw" for original data points, "hourly" for hourly averages, "daily" for daily averages (default: raw)')
args = parser.parse_args()

entity_id = args.entity
days_back = args.days
resolution = args.res

try:
    # Home Assistant URL and token
    project_root = Path(__file__).resolve().parents[1]
    dotenv_path = project_root / 'api.env'
    load_dotenv(dotenv_path=dotenv_path)
    token = os.getenv('HOMEASSISTANT_TOKEN_DataDownloadToken')
    ha_ip = os.getenv('HOMEASSISTANT_IP')

    # Check if required environment variables are available
    if not token or not ha_ip:
        print("Error: Missing required environment variables.")
        print("Please create an api.env file containing:")
        print("HOMEASSISTANT_TOKEN_DataDownloadToken=your_token_here")
        print("HOMEASSISTANT_IP=your_home_assistant_ip_here")
        print("\nOr set these environment variables directly.")
        sys.exit(1)

    ha_url = f"http://{ha_ip}:8123"

    # Calculate the start time based on days_back
    now = datetime.now()
    print(f"Debug - Current time: {now}")
    
    # Calculate days back properly - subtract the full number of days requested
    start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_back)
    # Calculate end time to end of current day
    end = now.replace(hour=23, minute=59, second=59, microsecond=999999)

    print(f"Debug - Start date: {start}")
    print(f"Debug - End date: {end}")

    # Format timestamps in ISO8601 format that Home Assistant expects
    start_iso = start.strftime("%Y-%m-%dT00:00:00")  # Explicitly set to start of day
    end_iso = end.strftime("%Y-%m-%dT23:59:59")      # Explicitly set to end of day

    print(f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Requesting data for: {days_back} days (from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')})")
    print(f"Entity: {entity_id}")
    print(f"Resolution: {resolution}")

    # Build the URL for the history API with explicit start and end times
    # We disable minimal_response and significant_changes_only to get ALL data points
    url = f"{ha_url}/api/history/period/{start_iso}?end_time={end_iso}&filter_entity_id={entity_id}&minimal_response=false&significant_changes_only=false"
    print(f"Request URL: {url}")

    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
    }

    print("Sending API request...")
    response = requests.get(url, headers=headers)
    print(f"API response status code: {response.status_code}")
    
    if response.ok:
        data = response.json()
        print(f"API returned data of type: {type(data)}")
        print(f"API data length: {len(data) if isinstance(data, list) else 'not a list'}")
        
        if not data or len(data) == 0:
            print(f"No data returned for entity: {entity_id}")
            sys.exit(0)
            
        else:
            try:
                print("Processing data...")
                # Debug: Print first items to understand structure
                print(f"First data item type: {type(data[0])}")
                
                # Extract entity data from API response
                if isinstance(data, list) and len(data) > 0:
                    # Home Assistant's API format is typically:
                    # [ [entity1_data_points], [entity2_data_points], ... ]
                    if isinstance(data[0], list):
                        print(f"Found list-of-lists format with {len(data)} items")
                        # Use first non-empty list
                        for i, item in enumerate(data):
                            if isinstance(item, list) and len(item) > 0:
                                print(f"Using item {i} with {len(item)} records")
                                raw_df = pd.DataFrame(item)
                                break
                        else:
                            print("No non-empty lists found in data")
                            sys.exit(1)
                    elif isinstance(data[0], dict):
                        print("Found direct list of dictionaries")
                        raw_df = pd.DataFrame(data)
                    else:
                        print(f"Unexpected data format: {type(data[0])}")
                        sys.exit(1)
                else:
                    print("Data is not a list or is empty")
                    sys.exit(1)
                
                print(f"Raw DataFrame created with {len(raw_df)} rows")
                print(f"Columns: {raw_df.columns.tolist()}")
                
                # Make a copy of the raw data to preserve it
                orig_df = raw_df.copy()
                
                # Check for required columns
                if 'last_changed' not in raw_df.columns:
                    print(f"Warning: 'last_changed' column not found. Available columns: {raw_df.columns.tolist()}")
                    # Try to find a suitable timestamp column
                    time_cols = [col for col in raw_df.columns if 'time' in col.lower() or 'date' in col.lower()]
                    if time_cols:
                        print(f"Using '{time_cols[0]}' as timestamp column")
                        raw_df = raw_df.rename(columns={time_cols[0]: 'last_changed'})
                    else:
                        print("No suitable timestamp column found")
                        sys.exit(1)
                
                if 'state' not in raw_df.columns:
                    print(f"Warning: 'state' column not found. Available columns: {raw_df.columns.tolist()}")
                    # Try to find a suitable value column
                    value_cols = [col for col in raw_df.columns if 'state' in col.lower() or 'value' in col.lower()]
                    if value_cols:
                        print(f"Using '{value_cols[0]}' as state column")
                        raw_df = raw_df.rename(columns={value_cols[0]: 'state'})
                    else:
                        print("No suitable state column found")
                        sys.exit(1)
                
                # Now process based on resolution
                if resolution == 'raw':
                    print("Processing raw data...")
                    # For raw resolution, use the original data without conversion
                    result_df = raw_df[['last_changed', 'state']].copy()
                    result_df.rename(columns={'last_changed': 'timestamp'}, inplace=True)
                    
                    # Convert timestamps to datetime but keep original state values
                    result_df['timestamp'] = pd.to_datetime(result_df['timestamp'], format='ISO8601')
                    
                    # Sort chronologically
                    result_df = result_df.sort_values('timestamp')
                    
                    print(f"Raw data has {len(result_df)} data points")
                    if not result_df.empty:
                        print(f"Time range: {result_df['timestamp'].min()} to {result_df['timestamp'].max()}")
                    
                else:  # hourly or daily
                    print(f"Processing {resolution} data...")
                    # Need to convert state to numeric for aggregation
                    df = raw_df.copy()
                    df['last_changed'] = pd.to_datetime(df['last_changed'], format='ISO8601')
                    
                    # Print debug information about the timestamps
                    if not df.empty:
                        print("\nDEBUG INFO - Original timestamps:")
                        print(f"First timestamp: {df['last_changed'].iloc[0]}")
                        print(f"Timestamp timezone info: {str(df['last_changed'].dt.tz) if hasattr(df['last_changed'].dt, 'tz') and df['last_changed'].dt.tz else 'None'}")
                        current_time = datetime.now()
                        print(f"Current system time: {current_time}")
                        print(f"Current system timezone: {datetime.now().astimezone().tzinfo}")
                        
                        # Check if system time might be wrong (more than a year in the future)
                        first_timestamp = df['last_changed'].iloc[0]
                        if hasattr(first_timestamp, 'year') and first_timestamp.year > current_time.year + 1:
                            print(f"WARNING: Timestamps appear to be from the future year {first_timestamp.year}.")
                            print("This suggests your system clock might be incorrectly set.")
                            print(f"System year: {current_time.year}, Data year: {first_timestamp.year}")
                    
                    df['state'] = pd.to_numeric(df['state'], errors='coerce')
                    
                    # Drop rows with non-numeric states
                    original_len = len(df)
                    df = df.dropna(subset=['state'])
                    if len(df) < original_len:
                        print(f"Dropped {original_len - len(df)} rows with non-numeric values")
                    
                    if resolution == 'hourly':
                        # Floor to the hour
                        df['timestamp'] = df['last_changed'].dt.floor('h')
                        freq = 'h'  # Updated from 'H' to 'h' to avoid deprecation warning
                    else:  # daily
                        # Floor to the day
                        df['timestamp'] = df['last_changed'].dt.floor('D')
                        freq = 'D'
                    
                    # Calculate mean for each time period
                    result_df = df.groupby('timestamp')['state'].mean().reset_index()
                    
                    # Remove timezone info before creating date range
                    start_naive = start
                    end_naive = end
                    
                    if hasattr(result_df['timestamp'].dt, 'tz') and result_df['timestamp'].dt.tz is not None:
                        # If timestamps have timezone, convert to naive datetimes
                        print("Converting timezone-aware timestamps to naive datetimes")
                        try:
                            # First convert to Stockholm timezone, then remove timezone info
                            result_df['timestamp'] = result_df['timestamp'].dt.tz_convert('Europe/Stockholm').dt.tz_localize(None)
                            print("Timestamps converted to Europe/Stockholm timezone before stripping timezone info")
                        except Exception as e:
                            print(f"Error converting timezone: {e}")
                            print("Trying fallback method with simple timezone offset adjustment...")
                            # Fallback: Apply +2 hours offset (Stockholm timezone is usually UTC+2)
                            result_df['timestamp'] = result_df['timestamp'].dt.tz_localize(None) + timedelta(hours=2)
                            print("Applied +2 hours timezone adjustment as fallback")
                    
                    # Create a complete range for all periods with naive datetimes
                    all_periods = pd.date_range(start=start_naive, end=end_naive, freq=freq)
                    template_df = pd.DataFrame({'timestamp': all_periods})
                    
                    # Merge with template to ensure all periods are included
                    # Use concat and groupby instead of merge to avoid timezone issues
                    combined_df = pd.concat([template_df, result_df])
                    result_df = combined_df.groupby('timestamp', as_index=False)['state'].first()
                    
                    # Fill NaN values with 0
                    result_df['state'] = result_df['state'].fillna(0)
                    
                    print(f"{resolution.capitalize()} data has {len(result_df)} data points")
                    if not result_df.empty:
                        print(f"Time range: {result_df['timestamp'].min()} to {result_df['timestamp'].max()}")
                
                # Save the result
                output_dir = Path("Data/HomeAssistant")
                output_dir.mkdir(exist_ok=True)
                
                # Extract entity name without domain for file naming
                entity_name = entity_id.split(".", 1)[1] if "." in entity_id else entity_id
                output_filename = f'{entity_name}{"_hourly" if resolution == "hourly" else ""}{"_daily" if resolution == "daily" else ""}.csv'
                output_file = output_dir / output_filename
                
                # Ensure timezone consistency for saving
                if not result_df.empty and hasattr(result_df['timestamp'].dt, 'tz') and result_df['timestamp'].dt.tz is not None:
                    try:
                        # Convert to Stockholm timezone first, then remove timezone info
                        result_df['timestamp'] = result_df['timestamp'].dt.tz_convert('Europe/Stockholm').dt.tz_localize(None)
                        print("Output timestamps converted to Europe/Stockholm timezone")
                    except Exception as e:
                        print(f"Error converting output timezone: {e}")
                        print("Using fallback timezone adjustment (+2 hours)...")
                        # Fallback: Apply +2 hours offset (Stockholm timezone is usually UTC+2)
                        result_df['timestamp'] = result_df['timestamp'].dt.tz_localize(None) + timedelta(hours=2)
                        print("Applied +2 hours timezone adjustment as fallback")
                
                # Filter out future dates (more than 1 day ahead of current time)
                # This helps catch any date parsing issues
                current_time = datetime.now()
                future_cutoff = current_time + timedelta(days=1)
                future_dates_count = len(result_df[result_df['timestamp'] > future_cutoff])
                if future_dates_count > 0:
                    print(f"WARNING: Found {future_dates_count} timestamps in the future (after {future_cutoff})")
                    print(f"First few future timestamps: {result_df[result_df['timestamp'] > future_cutoff]['timestamp'].head(5).tolist()}")
                    print("Removing future timestamps as they likely indicate a date parsing issue")
                    result_df = result_df[result_df['timestamp'] <= future_cutoff]
                
                # Additional debug info
                if not result_df.empty:
                    print(f"Final timestamp range: {result_df['timestamp'].min()} to {result_df['timestamp'].max()}")
                    print(f"Sample timestamps: {result_df['timestamp'].head(3).tolist()}")
                
                # Save the file
                result_df.to_csv(output_file, index=False)
                print(f"Saved {len(result_df)} rows to {output_file}")
                
            except Exception as e:
                print(f"Error processing data: {str(e)}")
                print("Exception details:")
                traceback.print_exc()
                print("First few records of data received:")
                print(str(data)[:1000])
                sys.exit(1)
    else:
        print("Error:", response.status_code, response.text)
        print(f"URL: {url}")
        sys.exit(1)

except Exception as e:
    print(f"Unexpected error: {str(e)}")
    print("Exception details:")
    traceback.print_exc()
    sys.exit(1) 