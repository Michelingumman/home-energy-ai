import os
# (optional) still suppress TensorFlow’s C++ banners and oneDNN info:
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


# Import from config.py
from config import (
    TARGET_VARIABLE, DATA_DIR, SE3_PRICES_FILE, SWEDEN_GRID_FILE,
    TIME_FEATURES_FILE, HOLIDAYS_FILE, WEATHER_DATA_FILE,
    GRID_FEATURES, MARKET_FEATURES, PRICE_FEATURES, TIME_FEATURES, HOLIDAY_FEATURES,
    FEATURE_GROUPS, WEATHER_FEATURES
)

# Get project root from DATA_DIR
project_root = DATA_DIR.resolve().parents[1]

def update_price_data():
    """Update the price data from the API and calculate features"""
    # Use paths from config
    csv_file_path = DATA_DIR / SE3_PRICES_FILE.name

    # Check if the file exists
    if not csv_file_path.exists():
        df = pd.DataFrame(columns=['HourSE', 'PriceArea', TARGET_VARIABLE])
        # Start from 30 days ago if creating a new file to avoid excessive API calls
        latest_timestamp = pd.Timestamp.now() - timedelta(days=30)
        print(f"Creating new price data file. Starting from {latest_timestamp}")
    else:
        # Read the existing CSV file
        df = pd.read_csv(csv_file_path)
        df['HourSE'] = pd.to_datetime(df['HourSE'])
        
        # If the file exists but is empty or corrupted, start from 30 days ago
        if df.empty:
            latest_timestamp = pd.Timestamp.now() - timedelta(days=30)
            print(f"Existing file is empty. Starting from {latest_timestamp}")
        else:
            # Get the latest timestamp from the existing data
            latest_timestamp = df['HourSE'].max()
            print(f"Continuing from latest timestamp in file: {latest_timestamp}")

    # Get the current time
    current_time = datetime.now()
    new_data = []

    # Loop through missing hours until the present
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
                        print(f"No data available for {next_date_str} {next_hour_str}")
                else:
                    print(f"No SE3 data available for {next_date_str}")
            else:
                print(f"Failed to fetch price data for {next_date_str}. Status code: {response.status_code}")
                break
        except Exception as e:
            print(f"Error fetching price data: {e}")
            break

    # Update DataFrame with new data
    if new_data:
        new_df = pd.DataFrame(new_data)
        new_df['HourSE'] = pd.to_datetime(new_df['HourSE'])
        df = pd.concat([df, new_df], ignore_index=True)
        df = df.sort_values(by="HourSE")

        # Calculate features
        df["price_24h_avg"] = df[TARGET_VARIABLE].rolling(window=24, min_periods=1).mean()
        df["price_168h_avg"] = df[TARGET_VARIABLE].rolling(window=168, min_periods=1).mean()
        df["price_24h_std"] = df[TARGET_VARIABLE].rolling(window=24, min_periods=1).std().fillna(0)
        df["hour_avg_price"] = df.groupby(df["HourSE"].dt.hour)[TARGET_VARIABLE].transform("mean")
        df["price_vs_hour_avg"] = df[TARGET_VARIABLE] / df["hour_avg_price"]

        # Save to CSV
        try:
            df.to_csv(csv_file_path, index=False)
            print(f'Successfully added {len(new_data)} missing hourly price records and updated features.')
        except Exception as e:
            print(f"Error saving price data to CSV: {e}")
    else:
        print("Price data is already up-to-date.")


def update_grid_data():
    """Update the grid data using Electricity Maps API with hourly resolution"""
    # Use path from config
    grid_file_path = DATA_DIR / SWEDEN_GRID_FILE.name
    
    # Load API key from env file
    dotenv_path = project_root / 'api.env'
    load_dotenv(dotenv_path=dotenv_path)
    api_key = os.getenv('ELECTRICITYMAPS')
    
    if not api_key:
        print("Error: ELECTRICITYMAPS API key not found in api.env.")
        return
    
    # Use grid columns from feature groups config
    grid_cols = FEATURE_GROUPS["grid_cols"]
    
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
        
        response = requests.get(
            f"https://api.electricitymap.org/v3/power-breakdown/history?zone=SE-SE3&start={start_date}",
            headers={
                "auth-token": api_key
            }
        )
        
        if response.status_code != 200:
            print(f"Failed to fetch grid data. Status code: {response.status_code}")
            print("Response content:", response.content.decode() if hasattr(response, 'content') else "No response content")
            return
            
        data = response.json()
        
        # Process records at hourly resolution
        hourly_records = []
        
        for entry in data['history']:
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
                print(f"Error processing timestamp {entry['datetime']}: {tz_err}")
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
                    print(f"Severe error parsing timestamp: {parse_err}. Using current time as fallback.")
                    entry_time = pd.Timestamp.now()
            
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
        
        # Create DataFrame with specified columns from config
        df = pd.DataFrame(hourly_records)
        
        # Set datetime as index and ensure it's properly formatted
        if not df.empty:
            df.set_index('datetime', inplace=True)
            
            # Ensure all required columns are present (initialize with zeros if missing)
            for col in grid_cols:
                if col not in df.columns:
                    df[col] = 0
            
            # Select only the columns defined in grid_cols and in the correct order
            df = df[grid_cols]
            df.sort_index(inplace=True)
            
            if df.isna().sum().any():
                # Fill any missing values with 0
                df = df.fillna(0)
                print("Found missing values in grid data: ", df.isna().sum())
            # Merge with existing data if it exists and is not empty
            if grid_file_path.exists():
                try:
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
                        
                        # Combine data, keeping the newer data for duplicated timestamps
                        combined_df = pd.concat([existing_df, df])
                        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                        combined_df.sort_index(inplace=True)
                        
                        # Reset index to save datetime as column
                        combined_df.reset_index(inplace=True)
                        combined_df.rename(columns={'index': 'datetime'}, inplace=True)
                        
                        # Save the updated data
                        combined_df.to_csv(grid_file_path, index=False)
                        print(f'Updated grid data with {len(df)} new records.')
                    else:
                        # If existing file is empty or corrupted, just save the new data
                        df.reset_index(inplace=True)
                        df.to_csv(grid_file_path, index=False)
                        print(f'Created new grid data file with {len(df)} records.')
                except Exception as e:
                    print(f"Error merging with existing grid data: {e}")
                    # If there's an error with the existing file, save the new data as backup
                    backup_path = grid_file_path.with_suffix('.backup.csv')
                    df.reset_index(inplace=True)
                    df.to_csv(backup_path, index=False)
                    print(f'Saved backup grid data to {backup_path}')
            else:
                # No existing file, just save the new data
                df.reset_index(inplace=True)
                df.to_csv(grid_file_path, index=False)
                print(f'Created new grid data file with {len(df)} records.')
        else:
            print("No grid data records received from API.")
    except Exception as e:
        print(f"Error updating grid data: {e}")

def main():
    """Main function to update all data files"""
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Updating price data...")
    update_price_data()
    
    print("\nUpdating grid data...")
    update_grid_data()
    
    print("\nData update complete.")

if __name__ == "__main__":
    main()