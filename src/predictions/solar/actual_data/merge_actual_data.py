#!/usr/bin/env python3
"""
Merge Actual Solar Power Data
============================

This script:
1. Checks for existing data in merged_cleaned_actual_data.csv
2. Determines how many days of data need to be fetched from Home Assistant
3. Runs download_entity_data_from_HA.py to get the new data
4. Merges the new data with existing data, handling overlaps
5. Saves the updated data to merged_cleaned_actual_data.csv

This script is designed to run daily to keep the solar generation data up to date.
"""

import pandas as pd
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Constants
ENTITY_ID = "solar_generated_power_2"
RESOLUTION = "hourly"
OUTPUT_FILE = "merged_cleaned_actual_data.csv"
CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("Data/HomeAssistant")
DOWNLOAD_SCRIPT = Path(__file__).resolve().parents[3] / "download_entity_data_from_HA.py"

def main():
    print(f"Starting merge process for {ENTITY_ID} data")
    
    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    output_path = CURRENT_DIR / OUTPUT_FILE
    
    # Calculate days to fetch based on existing data
    days_to_fetch = calculate_days_to_fetch(output_path)
    
    if days_to_fetch <= 0:
        print("Data is already up to date, no new data needed")
        return
    
    print(f"Fetching {days_to_fetch} days of new data")
    
    # Run the download script to get new data
    download_new_data(days_to_fetch)
    
    # Merge the data
    merge_data(output_path)
    
    print(f"Data successfully merged and saved to {output_path}")

def calculate_days_to_fetch(output_path):
    """Calculate how many days of data need to be fetched"""
    if not output_path.exists():
        print(f"No existing data file found at {output_path}")
        return 30  # Default to 30 days if no existing data
    
    try:
        # Read existing data
        existing_data = pd.read_csv(output_path)
        
        # Check if data exists and has the expected timestamp column
        if existing_data.empty or 'last_changed' not in existing_data.columns:
            print("Existing data file is empty or missing last_changed column")
            return 30
        
        # Convert timestamps to datetime
        existing_data['last_changed'] = pd.to_datetime(existing_data['last_changed'], format='ISO8601')
        
        # Find the latest timestamp
        latest_timestamp = existing_data['last_changed'].max()
        
        # Calculate days since latest timestamp
        days_since = (datetime.now() - latest_timestamp.replace(tzinfo=None)).days + 1
        
        print(f"Latest data timestamp: {latest_timestamp}")
        print(f"Days since latest data: {days_since}")
        
        return max(1, days_since)  # Fetch at least 1 day
    
    except Exception as e:
        print(f"Error reading existing data: {e}")
        return 30  # Default to 30 days if error occurs

def download_new_data(days):
    """Run the download script to fetch new data"""
    try:
        cmd = [
            sys.executable,
            str(DOWNLOAD_SCRIPT),
            "--entity", ENTITY_ID,
            "--days", str(days),
            "--res", RESOLUTION
        ]
        
        print(f"Running download command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("Download script output:")
        print(result.stdout)
        
        if result.stderr:
            print("Download script errors:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"Error running download script: {e}")
        print(f"Script output: {e.output}")
        print(f"Script errors: {e.stderr}")
        sys.exit(1)

def merge_data(output_path):
    """Merge new data with existing data"""
    # Path to the downloaded data
    downloaded_file = DATA_DIR / f"{ENTITY_ID}_{RESOLUTION}.csv"
    
    if not downloaded_file.exists():
        print(f"Downloaded data file not found: {downloaded_file}")
        sys.exit(1)
    
    # Load the new data
    try:
        new_data = pd.read_csv(downloaded_file)
        new_data['timestamp'] = pd.to_datetime(new_data['timestamp'], format='ISO8601')
        
        # Ensure timestamps are timezone-naive
        if new_data['timestamp'].dt.tz is not None:
            new_data['timestamp'] = new_data['timestamp'].dt.tz_localize(None)
            print("Converted new data timestamps to timezone-naive")
            
        # Transform new data to match the expected format in merged_cleaned_actual_data.csv
        new_data = transform_data_format(new_data)
        
        print(f"Loaded and transformed {len(new_data)} rows of new data")
    except Exception as e:
        print(f"Error loading new data: {e}")
        sys.exit(1)
    
    # Check if the output file exists
    if not output_path.exists():
        # If no existing file, just save the new data
        print(f"No existing data file, saving new data to {output_path}")
        new_data.to_csv(output_path, index=False)
        return
    
    # Load existing data
    try:
        existing_data = pd.read_csv(output_path)
        existing_data['last_changed'] = pd.to_datetime(existing_data['last_changed'], format='ISO8601')
        
        # Ensure timestamps are timezone-naive
        if existing_data['last_changed'].dt.tz is not None:
            existing_data['last_changed'] = existing_data['last_changed'].dt.tz_localize(None)
            print("Converted existing data timestamps to timezone-naive")
            
        print(f"Loaded {len(existing_data)} rows of existing data")
    except Exception as e:
        print(f"Error loading existing data: {e}")
        sys.exit(1)
    
    # Combine the data, with new data taking precedence
    combined_data = pd.concat([existing_data, new_data])
    
    # Remove duplicates, keeping the last occurrence (which will be from the new data)
    combined_data = combined_data.drop_duplicates(subset=['last_changed'], keep='last')
    
    # Sort by timestamp
    combined_data = combined_data.sort_values('last_changed')
    
    # Convert datetime to string format for saving to CSV
    combined_data['last_changed'] = combined_data['last_changed'].dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    
    # Save the merged data
    combined_data.to_csv(output_path, index=False)
    print(f"Saved {len(combined_data)} rows of merged data")
    print(f"Data range: {combined_data['last_changed'].min()} to {combined_data['last_changed'].max()}")

def transform_data_format(data):
    """Transform data from timestamp,state to entity_id,state,last_changed format"""
    # Rename timestamp column to last_changed
    data = data.rename(columns={'timestamp': 'last_changed'})
    
    # Add entity_id column
    data['entity_id'] = f"sensor.{ENTITY_ID}"
    
    # Reorder columns to match the expected format
    data = data[['entity_id', 'state', 'last_changed']]
    
    return data

if __name__ == "__main__":
    main()
