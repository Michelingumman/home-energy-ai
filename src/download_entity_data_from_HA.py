#!/usr/bin/env python3
"""
Home Assistant Entity Data Downloader
====================================

This script downloads historical sensor data from a Home Assistant instance, processes it 
to calculate averages at different time resolutions, and saves the data to a CSV file.

Usage:
------
1. Basic usage (defaults to raw data with complete time coverage):
    python download_entity_data_from_HA.py
    
2. Specify a different entity to download:
    python download_entity_data_from_HA.py --entity entity_id
    
3. Specify number of days to retrieve (e.g., last 7 days):
    python download_entity_data_from_HA.py --days 7
    
4. Specify data resolution (raw, hourly, or daily):
    python download_entity_data_from_HA.py --res raw
    python download_entity_data_from_HA.py --res hourly
    python download_entity_data_from_HA.py --res daily
    
5. Combine options:
    python download_entity_data_from_HA.py --entity entity_id --days 10 --res hourly 
    
All entity IDs should be provided without the 'sensor.' prefix as the script will add it automatically.

Output:
-------
Data is saved to a CSV file in the "Data/HomeAssistant" folder with naming format:
"[entity_id].csv" for raw data, "[entity_id]_hourly.csv" for hourly data, or "[entity_id]_daily.csv" for daily data.

Each file contains columns:
- timestamp: The timestamp in ISO format
- state: The numerical value

For all resolutions, the script ensures complete time coverage for the entire requested date range:
- Raw: Creates timestamps at regular intervals (detected from the data)
- Hourly: Creates one timestamp per hour
- Daily: Creates one timestamp per day

Any missing data points will be filled with zeros (0).

Notes:
------
- The script aggregates multiple readings by calculating mean values
- Non-numeric states are filtered out
- The date range includes full days from midnight to midnight
- The script always creates a complete time series, filling in missing values with 0
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

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Download entity data from Home Assistant and calculate hourly/daily averages',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
        Examples:
        python download_entity_data_from_HA.py --entity solar_generated_power_2 --days 1
        python download_entity_data_from_HA.py --entity electricity_consumption --days 7 --res hourly
        python download_entity_data_from_HA.py --entity water_consumption --days 30 --res daily
        python download_entity_data_from_HA.py --entity temperature --days 14
    """
)
parser.add_argument('--entity', type=str, default='solar_generated_power_2',
                    help='Entity ID to fetch data for (without the "sensor." prefix)')
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
    print(f"Entity: sensor.{entity_id}")
    print(f"Resolution: {resolution}")

    # Build the URL for the history API with explicit start and end times
    # We disable minimal_response and significant_changes_only to get ALL data points
    url = f"{ha_url}/api/history/period/{start_iso}?end_time={end_iso}&filter_entity_id=sensor.{entity_id}&minimal_response=false&significant_changes_only=false"
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
            print(f"No data returned for entity: sensor.{entity_id}")
            # Create empty DataFrame with expected columns for consistent output
            empty_df = pd.DataFrame(columns=['timestamp', 'state'])
            
            # Create a range of timestamps based on resolution
            if resolution == 'hourly':
                all_times = pd.date_range(start=start, end=end, freq='h')
            elif resolution == 'daily':
                all_times = pd.date_range(start=start, end=end, freq='D')
            else:  # raw
                all_times = pd.date_range(start=start, end=end, freq='5min')  # Default to 5min intervals
                
            empty_df['timestamp'] = all_times
            
            # Save empty dataframe with NaN values
            output_dir = project_root / "Data/HomeAssistant"
            output_dir.mkdir(exist_ok=True)
            output_filename = f'{entity_id}{"_hourly" if resolution == "hourly" else ""}{"_daily" if resolution == "daily" else ""}.csv'
            output_file = output_dir / output_filename
            empty_df.to_csv(output_file, index=False)
            print(f"Saved empty dataset with timestamps to {output_file}")
            sys.exit(0)
            
        else:
            try:
                print("Processing data...")
                # Debug: Print first few items to see structure
                print(f"First data item: {str(data[0])[:500]}..." if data and len(data) > 0 else "No data items")
                
                # Check if data has the expected format
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], list) and len(data[0]) > 0:
                        # Sometimes Home Assistant returns data as a list within a list
                        df = pd.DataFrame(data[0])
                    elif isinstance(data[0], dict):
                        df = pd.DataFrame(data[0])
                    else:
                        print(f"Unexpected data format in first item. Type: {type(data[0])}")
                        # Try to extract any data we can
                        flat_data = []
                        for item in data:
                            if isinstance(item, list) and len(item) > 0:
                                flat_data.extend(item)
                            elif isinstance(item, dict):
                                flat_data.append(item)
                        
                        if flat_data:
                            print(f"Extracted {len(flat_data)} records from nested structure")
                            df = pd.DataFrame(flat_data)
                        else:
                            print("Could not extract valid data")
                            print(f"Data sample: {str(data)[:1000]}")
                            sys.exit(1)
                else:
                    print(f"Unexpected data format received. Raw data: {str(data)[:1000]}")
                    sys.exit(1)
                
                # Print columns to help debug
                print(f"DataFrame columns: {df.columns.tolist()}")
                
                # Ensure required columns exist
                if 'last_changed' not in df.columns or 'state' not in df.columns:
                    print(f"Missing required columns. Available columns: {df.columns.tolist()}")
                    # Try to identify alternative columns
                    time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower() or 'changed' in col.lower()]
                    value_cols = [col for col in df.columns if 'state' in col.lower() or 'value' in col.lower()]
                    
                    if time_cols and value_cols:
                        print(f"Using alternative columns: {time_cols[0]} for timestamp and {value_cols[0]} for state")
                        df = df.rename(columns={time_cols[0]: 'last_changed', value_cols[0]: 'state'})
                    else:
                        print("Cannot find suitable columns for timestamp and state")
                        sys.exit(1)
                
                # Print data shape and time range if data exists
                if not df.empty:
                    print(f"Retrieved data with {len(df)} rows")
                    
                    # Convert the 'last_changed' column to datetime
                    df['last_changed'] = pd.to_datetime(df['last_changed'], format='ISO8601')
                    print(f"Data time range: {df['last_changed'].min()} to {df['last_changed'].max()}")
                else:
                    print("Retrieved empty dataset")
                
                # Convert 'state' to numeric (non-numeric entries will become NaN)
                df['state'] = pd.to_numeric(df['state'], errors='coerce')
                
                # Drop rows with NaN values in state
                original_len = len(df)
                df = df.dropna(subset=['state'])
                if len(df) < original_len:
                    print(f"Dropped {original_len - len(df)} rows with non-numeric values")
                
                if resolution == 'hourly':
                    # Create a new 'timestamp' column by flooring 'last_changed' to the hour
                    df['timestamp'] = df['last_changed'].dt.floor('h')

                    # Check timezone
                    has_timezone = df['timestamp'].dt.tz is not None
                    if has_timezone:
                        print(f"Timestamps have timezone: {df['timestamp'].dt.tz}")
                        tz = df['timestamp'].dt.tz
                    else:
                        tz = None

                    # Group by the floored timestamp and calculate the mean of the 'state' values
                    result_df = df.groupby('timestamp')['state'].mean().reset_index()
                    
                    # Get actual start and end timestamps from data
                    if not result_df.empty:
                        actual_start = result_df['timestamp'].min()
                        actual_end = result_df['timestamp'].max()
                        print(f"Actual data time range: {actual_start} to {actual_end}")
                    else:
                        print("WARNING: No data found in the response!")
                    
                    # Create a complete hourly range from start to end date, with matching timezone
                    if has_timezone:
                        all_hours = pd.date_range(start=start, end=end, freq='h', tz=tz)
                        print(f"Creating timezone-aware hourly intervals with timezone: {tz}")
                    else:
                        all_hours = pd.date_range(start=start, end=end, freq='h')
                        # Make result timestamps timezone-naive if they have timezone
                        if not result_df.empty and result_df['timestamp'].dt.tz is not None:
                            result_df['timestamp'] = result_df['timestamp'].dt.tz_localize(None)
                            print("Converting timestamps to timezone-naive for hourly aggregation")
                            
                    print(f"Requested time range: {all_hours[0]} to {all_hours[-1]}")
                    print(f"Creating {len(all_hours)} hourly intervals")
                    
                    # Create a template dataframe with all hours
                    template_df = pd.DataFrame({'timestamp': all_hours})
                    
                    # Merge with proper timezone handling
                    try:
                        # Try the merge
                        result_df = pd.merge(template_df, result_df, on='timestamp', how='left')
                    except ValueError as e:
                        print(f"Timezone merge error: {str(e)}")
                        print("Making all timestamps timezone-naive and trying again")
                        
                        # Convert to naive datetimes
                        template_df['timestamp'] = template_df['timestamp'].dt.tz_localize(None)
                        result_df['timestamp'] = result_df['timestamp'].dt.tz_localize(None)
                        
                        # Try merge again
                        result_df = pd.merge(template_df, result_df, on='timestamp', how='left')
                    
                    # Fill NaN values with 0 instead of leaving them empty
                    result_df['state'] = result_df['state'].fillna(0)
                    
                    print(f"Aggregated {len(df)} data points into {len(result_df)} hourly intervals")
                    
                    # Check for missing hours
                    missing_hours = result_df['state'].isna().sum()
                    if missing_hours > 0:
                        print(f"WARNING: {missing_hours} hours ({(missing_hours/len(result_df))*100:.1f}%) have no data")
                        
                        # Find the largest gap
                        result_df['has_data'] = ~result_df['state'].isna()
                        result_df['gap_group'] = (result_df['has_data'].shift(1) != result_df['has_data']).cumsum()
                        gaps = result_df[~result_df['has_data']].groupby('gap_group').size()
                        if not gaps.empty:
                            max_gap = gaps.max()
                            max_gap_hours = max_gap
                            if max_gap_hours > 12:
                                print(f"WARNING: Largest gap is {max_gap_hours} consecutive hours with no data!")
                        
                        # Remove the temporary columns
                        result_df = result_df.drop(['has_data', 'gap_group'], axis=1)
                        
                elif resolution == 'daily':
                    # Create a new 'timestamp' column by flooring 'last_changed' to the day
                    df['timestamp'] = df['last_changed'].dt.floor('D')
                    
                    # Check timezone
                    has_timezone = df['timestamp'].dt.tz is not None
                    if has_timezone:
                        print(f"Timestamps have timezone: {df['timestamp'].dt.tz}")
                        tz = df['timestamp'].dt.tz
                    else:
                        tz = None

                    # Group by the floored timestamp and calculate the mean of the 'state' values
                    result_df = df.groupby('timestamp')['state'].mean().reset_index()
                    
                    # Create a complete daily range from start to end date with proper timezone
                    if has_timezone:
                        all_days = pd.date_range(start=start, end=end, freq='D', tz=tz)
                        print(f"Creating timezone-aware daily intervals with timezone: {tz}")
                    else:
                        all_days = pd.date_range(start=start, end=end, freq='D')
                        # Make result timestamps timezone-naive if they have timezone
                        if not result_df.empty and result_df['timestamp'].dt.tz is not None:
                            result_df['timestamp'] = result_df['timestamp'].dt.tz_localize(None)
                            print("Converting timestamps to timezone-naive for daily aggregation")
                    
                    # Create a template dataframe with all days
                    template_df = pd.DataFrame({'timestamp': all_days})
                    
                    # Merge with proper timezone handling
                    try:
                        # Try the merge
                        result_df = pd.merge(template_df, result_df, on='timestamp', how='left')
                    except ValueError as e:
                        print(f"Timezone merge error: {str(e)}")
                        print("Making all timestamps timezone-naive and trying again")
                        
                        # Convert to naive datetimes
                        template_df['timestamp'] = template_df['timestamp'].dt.tz_localize(None)
                        result_df['timestamp'] = result_df['timestamp'].dt.tz_localize(None)
                        
                        # Try merge again
                        result_df = pd.merge(template_df, result_df, on='timestamp', how='left')
                    
                    # Fill NaN values with 0 instead of leaving them empty
                    result_df['state'] = result_df['state'].fillna(0)
                    
                    print(f"Aggregated {len(df)} data points into {len(result_df)} daily averages")
                    
                    if not result_df.empty:
                        print(f"Date range in output: {result_df['timestamp'].min()} to {result_df['timestamp'].max()}")
                    
                    # Check for missing days
                    missing_days = result_df['state'].isna().sum()
                    if missing_days > 0:
                        print(f"WARNING: {missing_days} days ({(missing_days/len(result_df))*100:.1f}%) have no data")
                else:  # raw resolution
                    # Use the original timestamps
                    result_df = df[['last_changed', 'state']].copy()
                    result_df.rename(columns={'last_changed': 'timestamp'}, inplace=True)
                    
                    # Get time range info
                    if not result_df.empty:
                        actual_start = result_df['timestamp'].min()
                        actual_end = result_df['timestamp'].max()
                        print(f"Actual data range from API: {actual_start} to {actual_end}")
                        
                        # Handle timezone - note if timestamps are timezone-aware
                        has_timezone = result_df['timestamp'].dt.tz is not None
                        if has_timezone:
                            print(f"Timestamps have timezone: {result_df['timestamp'].dt.tz}")
                    else:
                        print("WARNING: No data found in the response!")
                        has_timezone = False
                        
                    # Create complete time series based on average interval
                    if len(result_df) > 1:
                        # Find the most common interval between timestamps
                        time_diffs = result_df['timestamp'].diff().dropna()
                        if not time_diffs.empty:
                            # Calculate median time difference for more stable results
                            median_diff_seconds = time_diffs.dt.total_seconds().median()
                            # Convert to minutes and round to nearest minute
                            minutes = round(median_diff_seconds / 60)
                            # Use 5 minute default if we can't determine
                            minutes = max(1, min(60, minutes))  # Constrain between 1-60 minutes
                        else:
                            minutes = 5  # Default to 5 minutes
                        
                        print(f"Detected average interval: {minutes} minutes")
                        
                        # Create a complete range of timestamps from start to end
                        freq = f"{minutes}min"
                        
                        # If original timestamps have timezone info, add timezone to our template
                        if has_timezone:
                            # Get the timezone from the data
                            tz = result_df['timestamp'].dt.tz
                            # Create timezone-aware timestamps
                            all_times = pd.date_range(start=start, end=end, freq=freq, tz=tz)
                            print(f"Creating timezone-aware timestamps with timezone: {tz}")
                        else:
                            # Create timezone-naive timestamps
                            all_times = pd.date_range(start=start, end=end, freq=freq)
                            # Make result timestamps timezone-naive if they have timezone
                            if not result_df.empty and result_df['timestamp'].dt.tz is not None:
                                result_df['timestamp'] = result_df['timestamp'].dt.tz_localize(None)
                                print("Converting timestamps to timezone-naive for compatibility")
                        
                        print(f"Creating complete time series with {len(all_times)} timestamps")
                        
                        # Create a template dataframe with all timestamps
                        template_df = pd.DataFrame({'timestamp': all_times})
                        
                        # Alternatively, make both timezone-naive
                        if has_timezone:
                            try:
                                # Try the merge with timezone-aware timestamps
                                complete_df = pd.merge(template_df, result_df, on='timestamp', how='left')
                            except ValueError as e:
                                print(f"Timezone merge error: {str(e)}")
                                print("Making all timestamps timezone-naive and trying again")
                                
                                # Convert to naive datetimes
                                template_df['timestamp'] = template_df['timestamp'].dt.tz_localize(None)
                                result_df['timestamp'] = result_df['timestamp'].dt.tz_localize(None)
                                
                                # Try merge again
                                complete_df = pd.merge(template_df, result_df, on='timestamp', how='left')
                        else:
                            # Simple merge for timezone-naive data
                            complete_df = pd.merge(template_df, result_df, on='timestamp', how='left')
                        
                        # Check for missing times
                        missing_points = complete_df['state'].isna().sum()
                        if missing_points > 0:
                            print(f"{missing_points} timestamps ({(missing_points/len(complete_df))*100:.1f}%) have no data")
                        
                        # Fill NaN values with 0 instead of leaving them empty
                        complete_df['state'] = complete_df['state'].fillna(0)
                        
                        # Use complete dataframe instead
                        result_df = complete_df
                    else:
                        print("Not enough data points to create a complete time series")
                    
                    print(f"Processed {len(result_df)} raw data points")

                # Make sure the output directory exists
                print("Preparing to save data...")
                output_dir = project_root / "Data/HomeAssistant"
                # Create the directory if it doesn't exist
                output_dir.mkdir(exist_ok=True)
                
                output_filename = f'{entity_id}{"_hourly" if resolution == "hourly" else ""}{"_daily" if resolution == "daily" else ""}.csv'
                output_file = output_dir / output_filename
                
                # Ensure the result_df is not empty
                if result_df.empty:
                    print("WARNING: Result dataframe is empty, creating a template with timestamps")
                    if resolution == 'hourly':
                        times = pd.date_range(start=start, end=end, freq='h')
                    elif resolution == 'daily':
                        times = pd.date_range(start=start, end=end, freq='D')
                    else:
                        times = pd.date_range(start=start, end=end, freq='5min')
                    
                    result_df = pd.DataFrame({'timestamp': times})
                    # Add state column with zeros
                    result_df['state'] = 0
                
                # For better compatibility, convert all timestamps to timezone-naive before saving
                if not result_df.empty and hasattr(result_df['timestamp'].dt, 'tz') and result_df['timestamp'].dt.tz is not None:
                    print("Converting timestamps to timezone-naive format for output file")
                    result_df['timestamp'] = result_df['timestamp'].dt.tz_localize(None)

                # Check if the directory exists before saving
                if not output_dir.exists():
                    print(f"Creating directory: {output_dir}")
                    output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save with error handling
                try:
                    result_df.to_csv(output_file, index=False)
                    print(f"Data saved to {output_file}")
                    print(f"CSV file contains {len(result_df)} rows")
                    # Check if file was actually created
                    if output_file.exists():
                        print(f"Verified: File exists with size {output_file.stat().st_size} bytes")
                    else:
                        print(f"ERROR: File was not created at {output_file}")
                except Exception as e:
                    print(f"Error saving CSV file: {str(e)}")
                    # Try alternative save approach
                    try:
                        alt_path = f"Data/HomeAssistant/{entity_id}_{resolution}.csv"
                        result_df.to_csv(alt_path, index=False)
                        print(f"Data saved to alternative path: {alt_path}")
                    except Exception as e2:
                        print(f"Alternative save also failed: {str(e2)}")
                
            except Exception as e:
                print(f"Error processing data: {str(e)}")
                print("Exception details:")
                traceback.print_exc()
                print("First few records of data received:")
                print(json.dumps(data[:2] if data and len(data) > 0 else data, indent=2)[:1000])
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

