import os
import pandas as pd
from datetime import datetime, timedelta
import subprocess
import sys

# Define the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

# Define the path to the main CSV file
MAIN_CSV_FILE = os.path.join(ROOT_DIR, 'data', 'processed', 'villamichelin', 'Thermia', 'HeatPumpPower.csv')

# Define the path to the script that fetches data
EXTRACT_SCRIPT_PATH = os.path.join(ROOT_DIR, 'src', 'predictions', 'demand', 'Thermia', 'ExtractHistoricData.py')

# Temporary file to store newly fetched data
TEMP_CSV_FILE = os.path.join(ROOT_DIR, 'data', 'processed', 'villamichelin', 'Thermia', 'temp_heat_pump_data.csv')

def get_latest_timestamp(csv_file_path):
    """Reads the CSV and returns the latest timestamp."""
    try:
        if not os.path.exists(csv_file_path) or os.path.getsize(csv_file_path) == 0:
            print(f"Main CSV file '{csv_file_path}' not found or empty. Will attempt to fetch initial data.")
            return None
        
        df = pd.read_csv(csv_file_path)
        if 'timestamp' not in df.columns:
            print(f"Error: 'timestamp' column not found in {csv_file_path}")
            return None
        if df.empty:
            print(f"CSV file {csv_file_path} is empty. No latest timestamp to read.")
            return None
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        latest_timestamp = df['timestamp'].max()
        print(f"Latest timestamp in {csv_file_path}: {latest_timestamp}")
        return latest_timestamp
    except Exception as e:
        print(f"Error reading latest timestamp from {csv_file_path}: {e}")
        return None

def main():
    latest_timestamp = get_latest_timestamp(MAIN_CSV_FILE)
    
    days_to_fetch = 7 # Default days to fetch if no existing data
    if latest_timestamp:
        # Calculate days from latest_timestamp to now
        # Add a small buffer (e.g., 1 day) to ensure overlap and catch any very recent data
        # The Thermia API seems to sometimes have a slight delay in data availability for the current day.
        # We also need to ensure we fetch at least 1 day if the data is very recent.
        time_since_latest = datetime.now() - latest_timestamp
        days_to_fetch = max(1, time_since_latest.days + 1) 
        print(f"Time since last data point: {time_since_latest}. Will fetch {days_to_fetch} day(s) of data.")
    else:
        print(f"No existing data or unable to read latest timestamp. Fetching default {days_to_fetch} days of data.")

    # Run the ExtractHistoricData.py script
    print(f"Running {EXTRACT_SCRIPT_PATH} to fetch {days_to_fetch} day(s) of data...")
    try:
        # Ensure the script is executed with the Python interpreter from the current environment
        python_executable = sys.executable
        process = subprocess.Popen(
            [python_executable, EXTRACT_SCRIPT_PATH, '--days', str(days_to_fetch), '--output-file', TEMP_CSV_FILE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, # Ensures stdout and stderr are strings
            encoding='utf-8' # Specify encoding
        )
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print("ExtractHistoricData.py executed successfully.")
            if stdout:
                print("Script output:\n", stdout)
        else:
            print(f"Error running ExtractHistoricData.py. Return code: {process.returncode}")
            if stdout:
                print("Script output:\n", stdout)
            if stderr:
                print("Script errors:\n", stderr)
            # Clean up temp file if script failed before creating it or created an empty/invalid one
            if os.path.exists(TEMP_CSV_FILE):
                 os.remove(TEMP_CSV_FILE)
            return # Exit if data fetching failed

    except FileNotFoundError:
        print(f"Error: The script {EXTRACT_SCRIPT_PATH} was not found.")
        return
    except Exception as e:
        print(f"An error occurred while running ExtractHistoricData.py: {e}")
        if os.path.exists(TEMP_CSV_FILE):
             os.remove(TEMP_CSV_FILE)
        return

    # Check if the temp file was created and has data
    if not os.path.exists(TEMP_CSV_FILE) or os.path.getsize(TEMP_CSV_FILE) == 0:
        print(f"Temporary data file {TEMP_CSV_FILE} was not created or is empty. No new data to append.")
        # No need to remove if it doesn't exist, but ensure it's gone if empty
        if os.path.exists(TEMP_CSV_FILE):
            os.remove(TEMP_CSV_FILE)
        return

    # Append new data to the main CSV
    try:
        new_data_df = pd.read_csv(TEMP_CSV_FILE)
        if new_data_df.empty:
            print(f"Fetched data in {TEMP_CSV_FILE} is empty. Nothing to append.")
            os.remove(TEMP_CSV_FILE)
            return
            
        new_data_df['timestamp'] = pd.to_datetime(new_data_df['timestamp'])

        if os.path.exists(MAIN_CSV_FILE) and os.path.getsize(MAIN_CSV_FILE) > 0:
            existing_data_df = pd.read_csv(MAIN_CSV_FILE)
            existing_data_df['timestamp'] = pd.to_datetime(existing_data_df['timestamp'])
            combined_df = pd.concat([existing_data_df, new_data_df], ignore_index=True)
            print(f"Appended {len(new_data_df)} new records to existing {len(existing_data_df)} records.")
        else:
            combined_df = new_data_df
            print(f"Main CSV file did not exist or was empty. Initializing with {len(new_data_df)} new records.")
            # Ensure the directory for the main CSV exists if it's the first time
            os.makedirs(os.path.dirname(MAIN_CSV_FILE), exist_ok=True)

        # Remove duplicates based on timestamp, keeping the last occurrence (new data if overlapping)
        # It is assumed that if there are overlaps, the newly fetched data is more up-to-date for those specific timestamps.
        # However, ExtractHistoricData resamples to 15min intervals, so identical timestamps should ideally have identical data.
        # If power_status or other values can change for the same 15min slot on a re-fetch, keep='last' is appropriate.
        # If not, keep='first' would also work and be slightly more robust against unexpected changes in historical re-fetches.
        # For simplicity and given the source, keep='last' is fine.
        combined_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
        
        # Sort by timestamp
        combined_df.sort_values(by='timestamp', inplace=True)
        
        # Save the updated DataFrame
        combined_df.to_csv(MAIN_CSV_FILE, index=False, date_format='%Y-%m-%d %H:%M:%S')
        print(f"Successfully updated {MAIN_CSV_FILE}. Total records: {len(combined_df)}")
        final_latest_timestamp = combined_df['timestamp'].max()
        print(f"New latest timestamp in {MAIN_CSV_FILE}: {final_latest_timestamp}")

    except Exception as e:
        print(f"Error processing or appending data: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(TEMP_CSV_FILE):
            os.remove(TEMP_CSV_FILE)
            print(f"Removed temporary file {TEMP_CSV_FILE}.")

if __name__ == "__main__":
    main() 