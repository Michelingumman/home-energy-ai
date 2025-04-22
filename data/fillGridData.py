from pathlib import Path
import pandas as pd
import requests
from datetime import datetime, timedelta
import json
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import os
import time
import argparse

project_root = Path(__file__).resolve().parents[1]
class FeatureConfig:
    def __init__(self):
        self.config_path = project_root / "src" / "predictions" / "prices" / "config.json"
        self.load_config()
    
    def load_config(self):
        """Load the feature configuration from JSON"""
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Set attributes for easy access
        self.feature_groups = self.config["feature_groups"]
        self.metadata = self.config["feature_metadata"]
        self.model_config = self.config["model_config"]
        
        # Set individual feature groups
        self.price_cols = self.feature_groups["price_cols"]
        self.grid_cols = self.feature_groups["grid_cols"]
        self.cyclical_cols = self.feature_groups["cyclical_cols"]
        self.binary_cols = self.feature_groups["binary_cols"]
        
        # Set important metadata
        self.target_feature = self.metadata["target_feature"]
        self.feature_order = self.metadata["feature_order"]
        
        # Set model configuration
        self.architecture = self.model_config["architecture"]
        self.training = self.model_config["training"]
        self.callbacks = self.model_config["callbacks"]
        self.data_split = self.model_config["data_split"]
        self.scaling = self.model_config["scaling"]
    
    @property
    def total_features(self):
        """Calculate total number of features dynamically"""
        return len(self.get_all_features())
    
    def get_all_features(self):
        """Get all features in the correct order"""
        all_features = []
        for group in self.feature_order:
            all_features.extend(self.feature_groups[group])
        return all_features
    
    def get_feature_group(self, group_name):
        """Get features for a specific group"""
        return self.feature_groups.get(group_name, [])
    
    def get_ordered_features(self):
        """Get all features in the training order with target first"""
        features = [self.target_feature]  # Target always first
        features.extend([f for f in self.price_cols if f != self.target_feature])
        
        # Add other features in order
        for group in self.feature_order[1:]:  # Skip price_cols as we handled it
            features.extend(self.feature_groups[group])
        
        return features
    
    def verify_features(self, available_features):
        """Verify that all required features are available"""
        required_features = set(self.get_all_features())
        missing_features = required_features - set(available_features)
        return list(missing_features)
    
    def get_model_architecture(self):
        """Get the model architecture configuration"""
        return self.architecture
    
    def get_training_params(self):
        """Get training parameters"""
        return self.training
    
    def get_callback_params(self):
        """Get callback configurations"""
        return self.callbacks
    
    def get_data_split_ratios(self):
        """Get data split ratios"""
        return self.data_split
    
    def get_scaling_params(self):
        """Get scaling configurations"""
        return self.scaling

def update_grid_data(project_root, fetch_total=False):
    """Update the grid data using Electricity Maps API with hourly resolution
    
    Args:
        project_root: Path to the project root
        fetch_total: If True, fetch all data from 2017-01-01 regardless of existing data
    """
    grid_file_path = project_root / 'data' / 'processed' / 'SwedenGrid.csv'
    
    # Load API key from env file
    dotenv_path = project_root / 'api.env'
    load_dotenv(dotenv_path=dotenv_path)
    api_key = os.getenv('ELECTRICITYMAPS')
    
    if not api_key:
        print("Error: ELECTRICITYMAPS API key not found in api.env.")
        return

    # Load config to ensure correct columns
    config = FeatureConfig()
    grid_cols = config.grid_cols
    
    # Define the essential zones for import/export
    zones = {
        'SE-SE2': 'Main connection from northern Sweden',
        'SE-SE4': 'Main connection to southern Sweden',
        'NO-NO1': 'Norway connection',
        'DK-DK1': 'Denmark connection',
        'FI': 'Finland connection',
        'AX': 'Åland connection'
    }
    
    # Define the power sources matching config
    power_sources = ['nuclear', 'wind', 'hydro', 'solar', 'unknown']
    
    # Set the start date based on the fetch_total flag
    if fetch_total:
        # If --total is specified, start from January 1, 2017
        start_date = pd.Timestamp('2017-01-01', tz='UTC')
        print(f"Total fetch requested. Will attempt to fetch all data from {start_date} to present.")
        
        # Create a new empty DataFrame if fetching total
        existing_df = pd.DataFrame(columns=grid_cols)
        existing_df.index.name = 'datetime'
        latest_timestamp = start_date
    else:
        # Load existing data or create new DataFrame
        if grid_file_path.exists():
            try:
                existing_df = pd.read_csv(grid_file_path, index_col=0)
                if not existing_df.empty:
                    existing_df.index = pd.to_datetime(existing_df.index)
                    
                    # Ensure existing index is timezone-aware UTC
                    if existing_df.index.tzinfo is None:
                        print("Existing index is timezone-naive. Assuming UTC and localizing.")
                        # *** Assumption: Naive timestamps in CSV represent UTC. ***
                        # If they represent local time (e.g., Europe/Stockholm), use:
                        # existing_df.index = existing_df.index.tz_localize('Europe/Stockholm', ambiguous='infer', nonexistent='shift_forward').tz_convert('UTC')
                        try:
                            existing_df.index = existing_df.index.tz_localize('UTC')
                        except Exception as tz_err:
                            print(f"Error localizing naive index to UTC: {tz_err}. Proceeding without timezone conversion for existing data.")
                            # Handle error - perhaps skip update or log? Here, we proceed but sorting might fail later.
                    elif str(existing_df.index.tzinfo) != 'UTC':
                        print(f"Existing index has timezone {existing_df.index.tzinfo}. Converting to UTC.")
                        try:
                            existing_df.index = existing_df.index.tz_convert('UTC')
                        except Exception as tz_err:
                            print(f"Error converting existing index to UTC: {tz_err}. Proceeding without timezone conversion.")

                    # Ensure existing data has all required columns from config
                    for col in grid_cols:
                        if col not in existing_df.columns:
                            existing_df[col] = 0.0 # Initialize with float
                    existing_df = existing_df[grid_cols] # Keep only config columns and ensure order
                    existing_df = existing_df.round(2) # Round existing data
                    existing_df = existing_df.fillna(0.0) # Fill NA
                    latest_timestamp = existing_df.index.max() # Get latest timestamp *after* potential UTC conversion
                else:
                    existing_df = pd.DataFrame(columns=grid_cols) # Empty file, create empty df
                    existing_df.index.name = 'datetime'
                    latest_timestamp = pd.Timestamp.utcnow() - timedelta(days=7) # Start from 7 days ago
            except pd.errors.EmptyDataError:
                print("Existing file was empty or corrupted, starting fresh.")
                existing_df = pd.DataFrame(columns=grid_cols)
                existing_df.index.name = 'datetime'
                latest_timestamp = pd.Timestamp.utcnow() - timedelta(days=7) # Start from 7 days ago
            except Exception as e:
                print(f"Error reading existing grid data: {e}. Starting fresh with last 7 days.")
                existing_df = pd.DataFrame(columns=grid_cols)
                existing_df.index.name = 'datetime'
                latest_timestamp = pd.Timestamp.utcnow() - timedelta(days=7) # Start from 7 days ago
        else:
            print("Grid data file not found, creating new file starting from 7 days ago.")
            existing_df = pd.DataFrame(columns=grid_cols)
            existing_df.index.name = 'datetime'
            latest_timestamp = pd.Timestamp.utcnow() - timedelta(days=7) # Start from 7 days ago

    # Ensure latest_timestamp is timezone-aware (UTC) for comparison
    if latest_timestamp.tzinfo is None:
        latest_timestamp = latest_timestamp.tz_localize('UTC')
    else:
        # Use str comparison for timezone instead of direct object comparison
        if str(latest_timestamp.tzinfo) != 'UTC':
            latest_timestamp = latest_timestamp.tz_convert('UTC')

    # Get current time in UTC, rounded down to the nearest hour
    current_time = pd.Timestamp.utcnow().floor('h')
    
    # If fetching total data, we start from Jan 1, 2017, otherwise from the latest timestamp + 1 hour
    start_time = latest_timestamp if fetch_total else latest_timestamp + timedelta(hours=1)
    
    # Create timestamps to fetch, limited by API restrictions
    timestamps_to_fetch = pd.date_range(start=start_time, end=current_time, freq='h')
    
    # Prepare for incremental saves for large dataset fetches
    save_interval = 500  # Save every 500 records to avoid losing data if script crashes
    last_save = 0
    new_hourly_records = []

    print(f"Found {len(existing_df)} existing records.")
    if fetch_total:
        print(f"Will fetch ALL data from {start_time} to {current_time}, total of {len(timestamps_to_fetch)} hours")
    else:
        print(f"Latest record: {latest_timestamp}. Current time: {current_time}")
        
    if not timestamps_to_fetch.empty:
        if not fetch_total:
            print(f"Attempting to fetch data for {len(timestamps_to_fetch)} missing hours from {timestamps_to_fetch[0]} to {timestamps_to_fetch[-1]}...")

        for i, timestamp in enumerate(timestamps_to_fetch):
            # Format timestamp for API call (ISO 8601 format with Z for UTC)
            datetime_str = timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')
            api_url = f"https://api.electricitymap.org/v3/power-breakdown/past?zone=SE-SE3&datetime={datetime_str}"
            
            # Progress update for long fetches
            if i % 50 == 0 or i == len(timestamps_to_fetch) - 1:
                progress_pct = (i+1) / len(timestamps_to_fetch) * 100
                print(f"Progress: {i+1}/{len(timestamps_to_fetch)} ({progress_pct:.1f}%) - Current timestamp: {timestamp}")
            
            try:
                response = requests.get(
                    api_url,
                    headers={"auth-token": api_key}
                )
                
                if response.status_code == 200:
                    entry = response.json()
                    
                    # Process the single entry for this hour
                    record = {col: 0.0 for col in grid_cols} # Initialize with float
                    
                    # Use the timestamp we requested as the index
                    record_time = timestamp 

                    # Check if data is valid (sometimes API returns success but empty/null data)
                    if not entry or entry.get('powerProductionBreakdown') is None:
                        print(f"Warning: No valid data returned for {datetime_str}. Skipping.")
                        continue

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
                    
                    # Use the timestamp as the key for the record
                    new_hourly_records.append(pd.Series(record, name=record_time))
                    
                    # Incremental save for large fetches
                    if fetch_total and len(new_hourly_records) >= last_save + save_interval:
                        print(f"Performing incremental save with {len(new_hourly_records)} records...")
                        save_progress(grid_file_path, existing_df, new_hourly_records, grid_cols)
                        last_save = len(new_hourly_records)
                        print(f"Saved progress up to {timestamp}")

                elif response.status_code == 429: # Rate limit hit
                    print("Rate limit hit. Waiting 60 seconds...")
                    time.sleep(60)
                    # Consider adding logic here to retry the same timestamp
                    print(f"Retrying fetch for {datetime_str}")
                    # Simple retry (could be more robust)
                    response = requests.get(api_url, headers={"auth-token": api_key})
                    if response.status_code == 200:
                        # Reprocess if successful after retry
                        print("Retry successful.")
                        entry = response.json()
                        
                        # Process the single entry for this hour (same code as above)
                        record = {col: 0.0 for col in grid_cols}
                        record_time = timestamp
                        
                        if not entry or entry.get('powerProductionBreakdown') is None:
                            print(f"Warning: No valid data returned for {datetime_str} after retry. Skipping.")
                            continue
                            
                        record['fossilFreePercentage'] = round(entry.get('fossilFreePercentage', 0) or 0, 2)
                        record['renewablePercentage'] = round(entry.get('renewablePercentage', 0) or 0, 2)
                        record['powerConsumptionTotal'] = round(entry.get('powerConsumptionTotal', 0) or 0, 2)
                        record['powerProductionTotal'] = round(entry.get('powerProductionTotal', 0) or 0, 2)
                        record['powerImportTotal'] = round(entry.get('powerImportTotal', 0) or 0, 2)
                        record['powerExportTotal'] = round(entry.get('powerExportTotal', 0) or 0, 2)
                        
                        prod = entry.get('powerProductionBreakdown', {})
                        for source in power_sources:
                            record[source] = round(prod.get(source, 0) or 0, 2)
                            
                        import_breakdown = entry.get('powerImportBreakdown', {})
                        export_breakdown = entry.get('powerExportBreakdown', {})
                        for zone in zones:
                            record[f'import_{zone}'] = round(import_breakdown.get(zone, 0) or 0, 2)
                            record[f'export_{zone}'] = round(export_breakdown.get(zone, 0) or 0, 2)
                            
                        new_hourly_records.append(pd.Series(record, name=record_time))
                    else:
                        print(f"Retry failed for {datetime_str}. Status: {response.status_code}. Skipping.")
                else:
                    print(f"Failed to fetch grid data for {datetime_str}. Status code: {response.status_code}")
                    if response.status_code == 402: # Payment required
                        print("API subscription may have expired. Stopping fetch.")
                        break
                    elif response.status_code == 403: # Forbidden
                        print("API key may be invalid. Stopping fetch.")
                        break
                    # For other errors, continue with next timestamp

            except requests.exceptions.RequestException as e:
                print(f"Network error fetching data for {datetime_str}: {e}")
                # Wait a bit before continuing to avoid hammering the API
                time.sleep(5)
            except Exception as e:
                print(f"Error processing grid data for {datetime_str}: {e}")
                # Continue with next timestamp
        
        # Final save
        if new_hourly_records:
            save_final_results(grid_file_path, existing_df, new_hourly_records, grid_cols)
        else:
            print("No new grid data fetched or processed.")
    else:
        print("Grid data appears to be up-to-date.")

def save_progress(grid_file_path, existing_df, new_records, grid_cols):
    """Save progress for incremental updates to avoid losing data"""
    try:
        new_df = pd.DataFrame(new_records)
        new_df.index.name = 'datetime'
        
        # Ensure new_df has correct columns in correct order
        for col in grid_cols:
            if col not in new_df.columns:
                new_df[col] = 0.0
        new_df = new_df[grid_cols]
        
        # Combine with existing data
        combined_df = pd.concat([existing_df, new_df])
        
        # Remove duplicates, keeping newest data
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        
        # Sort, round, and save
        combined_df.sort_index(inplace=True)
        combined_df = combined_df.round(2)
        combined_df = combined_df.fillna(0.0)
        
        combined_df.to_csv(grid_file_path)
        return True
    except Exception as e:
        print(f"Error during incremental save: {e}")
        return False

def save_final_results(grid_file_path, existing_df, new_records, grid_cols):
    """Save final results after all data fetching is complete"""
    try:
        new_df = pd.DataFrame(new_records)
        new_df.index.name = 'datetime'
        
        # Ensure new_df has the correct columns in the correct order, fill missing with 0.0
        for col in grid_cols:
            if col not in new_df.columns:
                new_df[col] = 0.0
        new_df = new_df[grid_cols] 

        # Combine with existing data
        combined_df = pd.concat([existing_df, new_df])
        
        # Remove duplicates, keeping the newly fetched data
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        
        # Sort by date, round all values, and save
        combined_df.sort_index(inplace=True)
        combined_df = combined_df.round(2) # Final rounding
        combined_df = combined_df.fillna(0.0) # Ensure no NaNs remain

        combined_df.to_csv(grid_file_path)
        print(f"Successfully added/updated {len(new_records)} hourly grid records. New latest record: {combined_df.index.max()}")
    except Exception as e:
        print(f"Error writing updated grid data to CSV: {e}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Update grid data from Electricity Maps API")
    parser.add_argument("--total", action="store_true", help="Fetch all data from 2017-01-01 regardless of existing data")
    args = parser.parse_args()
    
    feature_config = FeatureConfig() 
    print(feature_config.get_ordered_features())
    
    project_root = Path(__file__).resolve().parents[1]
    
    # Create processed data directory if it doesn't exist
    processed_dir = project_root / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Update grid data
    print("\nUpdating grid data...")
    update_grid_data(project_root, fetch_total=args.total)

if __name__ == "__main__":
    main()