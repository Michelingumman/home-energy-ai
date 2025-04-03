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
import logging
import sys
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parents[3]
class FeatureConfig:
    def __init__(self):
        self.config_path = Path(__file__).parent / "config.json"
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






def update_price_data(project_root):
    """Update the price data from the API and calculate features"""
csv_file_path = project_root / 'data' / 'processed' / 'SE3prices.csv'

# Check if the file exists
if not csv_file_path.exists():
        df = pd.DataFrame(columns=['HourSE', 'PriceArea', 'SE3_price_ore'])
        latest_timestamp = pd.Timestamp('2020-01-01')  # Start from 2020 if no file exists
else:
    # Read the existing CSV file
    df = pd.read_csv(csv_file_path)
    df['HourSE'] = pd.to_datetime(df['HourSE'])
    latest_timestamp = df['HourSE'].max()

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
    response = requests.get(api_url)
    
    if response.status_code == 200:
        data = response.json()
        if 'SE3' in data:
            se3_data = data['SE3']
            hour_data = next((item for item in se3_data if item['hour'] == next_hour_int), None)
            
            if hour_data:
                new_data.append({
                    'HourSE': f'{next_date_str} {next_hour_str}',
                    'PriceArea': 'SE3',
                        'SE3_price_ore': hour_data['price_sek'],
                })
            else:
                print(f"No data available for {next_date_str} {next_hour_str}")
        else:
            print(f"No SE3 data available for {next_date_str}")
    else:
            print(f"Failed to fetch price data for {next_date_str}. Status code: {response.status_code}")
            break

    # Update DataFrame with new data
if new_data:
    new_df = pd.DataFrame(new_data)
    new_df['HourSE'] = pd.to_datetime(new_df['HourSE'])
    df = pd.concat([df, new_df], ignore_index=True)
    df = df.sort_values(by="HourSE")

    # Calculate features
    df["price_24h_avg"] = df["SE3_price_ore"].rolling(window=24, min_periods=1).mean()
    df["price_168h_avg"] = df["SE3_price_ore"].rolling(window=168, min_periods=1).mean()
    df["price_24h_std"] = df["SE3_price_ore"].rolling(window=24, min_periods=1).std()
    df["hour_avg_price"] = df.groupby(df["HourSE"].dt.hour)["SE3_price_ore"].transform("mean")
    df["price_vs_hour_avg"] = df["SE3_price_ore"] / df["hour_avg_price"]

    df.to_csv(csv_file_path, index=False)
    print(f'Successfully added {len(new_data)} missing hourly price records and updated features.')
else:
    print("Price data is already up-to-date.")




def update_grid_data_hourly(project_root):
    """Update the grid data using Electricity Maps API with hourly resolution for the past 24 hours"""
    grid_file_path = project_root / 'data' / 'processed' / 'SwedenGrid.csv'
    
    # Create processed data directory if it doesn't exist
    processed_dir = project_root / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load API key from env file
    dotenv_path = project_root / 'api.env'
    load_dotenv(dotenv_path=dotenv_path)
    api_key = os.getenv('ELECTRICITYMAPS')
    
    if not api_key:
        print("Error: ELECTRICITYMAPS API key not found in api.env file")
        return
    
    print(f"API key loaded: {api_key[:4]}...{api_key[-4:] if len(api_key) > 8 else ''}")
    
    # Load config to ensure correct columns
    config = FeatureConfig()
    grid_cols = config.grid_cols if hasattr(config, 'grid_cols') else None
    
    # Define the essential zones for import/export
    zones = {
        'SE-SE2': 'Main connection from northern Sweden',
        'SE-SE4': 'Main connection to southern Sweden',
        'NO-NO1': 'Norway connection',
        'DK-DK1': 'Denmark connection',
        'FI': 'Finland connection',
        'AX': 'Åland connection'
    }
    
    # Define the power sources
    power_sources = ['nuclear', 'wind', 'hydro', 'solar', 'unknown']
    
    try:
        # Determine the latest timestamp in existing data to avoid gaps
        latest_timestamp = None
        if grid_file_path.exists():
            try:
                existing_df = pd.read_csv(grid_file_path, index_col=0)
                if not existing_df.empty:
                    existing_df.index = pd.to_datetime(existing_df.index)
                    latest_timestamp = pd.to_datetime(existing_df.index.max()).tz_localize('UTC')
                    print(f"Latest data timestamp: {latest_timestamp}")
            except pd.errors.EmptyDataError:
                print("Existing file was empty, will create new file with current data.")
            except Exception as e:
                print(f"Warning: Could not read existing file: {str(e)}")
                
        # Make API request for history data (past 24 hours)
        print("Fetching power breakdown history for Sweden...")
        api_url = "https://api.electricitymap.org/v3/power-breakdown/history?zone=SE-SE3"
        print(f"API URL: {api_url}")
        
        response = requests.get(
            api_url,
            headers={
                "auth-token": api_key
            }
        )
        
        if response.status_code != 200:
            print(f"Failed to fetch grid data. Status code: {response.status_code}")
            error_info = response.json() if response.content else "No error details"
            print(f"API Error: {error_info.get('message', str(error_info)[:100]) if isinstance(error_info, dict) else str(error_info)[:100]}")
            return
            
        data = response.json()
        
        if 'history' not in data or not data['history']:
            print("No history data found in API response")
            print(f"Response keys: {list(data.keys())}")
            print(f"Response sample: {str(data)[:200]}...")
            return
            
        # Show some debug info about the data
        history_length = len(data['history'])
        first_entry = data['history'][0] if history_length > 0 else {}
        last_entry = data['history'][-1] if history_length > 0 else {}
        
        print(f"Received {history_length} records from API")
        if history_length > 0:
            print(f"Time range: {first_entry.get('datetime', 'N/A')} to {last_entry.get('datetime', 'N/A')}")
        
        # Process each hourly record
        records = []
        skipped_records = 0
        
        for entry in data['history']:
            # Create a record for each hourly data point
            timestamp = pd.to_datetime(entry['datetime'])  # This will preserve the UTC timezone
            
            # Skip if we already have this timestamp in the existing data
            if latest_timestamp is not None and timestamp <= latest_timestamp:
                skipped_records += 1
                continue
                
            record = {}
            
            # Add core metrics
            record['fossilFreePercentage'] = entry.get('fossilFreePercentage', 0)
            record['renewablePercentage'] = entry.get('renewablePercentage', 0)
            record['powerConsumptionTotal'] = entry.get('powerConsumptionTotal', 0)
            record['powerProductionTotal'] = entry.get('powerProductionTotal', 0)
            record['powerImportTotal'] = entry.get('powerImportTotal', 0)
            record['powerExportTotal'] = entry.get('powerExportTotal', 0)
            
            # Get production breakdown
            prod = entry.get('powerProductionBreakdown', {})
            
            # Add power sources
            for source in power_sources:
                record[source] = prod.get(source, 0) or 0
            
            # Add import/export values
            import_breakdown = entry.get('powerImportBreakdown', {})
            export_breakdown = entry.get('powerExportBreakdown', {})
            
            for zone in zones:
                record[f'import_{zone}'] = import_breakdown.get(zone, 0) or 0
                record[f'export_{zone}'] = export_breakdown.get(zone, 0) or 0
            
            # Add to records with timestamp as key
            records.append((timestamp, record))
        
        print(f"Fetched {len(records)} new records (skipped {skipped_records} existing records)")
        
        if not records:
            print("No new data to add. Data is already up-to-date.")
            return
            
        # Create DataFrame with hourly data
        new_df = pd.DataFrame([record for _, record in records], 
                             index=[timestamp.tz_localize(None) for timestamp, _ in records])  # Remove timezone for consistency
        
        # Round values to 2 decimals
        new_df = new_df.round(2)
        
        # Fill any missing values with 0
        new_df = new_df.fillna(0)
        
        # Merge with existing data if file exists and has content
        final_df = new_df.copy()
        if grid_file_path.exists():
            try:
                existing_df = pd.read_csv(grid_file_path, index_col=0)
                if not existing_df.empty:
                    # Ensure index is datetime without timezone
                    existing_df.index = pd.to_datetime(existing_df.index)
                    
                    # Get all columns from both dataframes
                    all_columns = list(set(existing_df.columns).union(set(new_df.columns)))
                    
                    # Ensure both dataframes have all columns
                    for col in all_columns:
                        if col not in existing_df.columns:
                            existing_df[col] = 0
                        if col not in new_df.columns:
                            new_df[col] = 0
                    
                    # Combine dataframes
                    combined_df = pd.concat([existing_df, new_df])
                    
                    # Remove duplicates, keeping the newer data
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                    
                    # Sort by timestamp
                    combined_df = combined_df.sort_index()
                    
                    # Save the final combined dataframe
                    final_df = combined_df
                    print(f"Successfully merged new data with existing data")
                    
            except Exception as e:
                print(f"Error processing existing data: {str(e)}")
                print("Creating new file with current data.")
                
        # Sort by datetime, round values
        final_df.sort_index(inplace=True)
        final_df = final_df.round(2)
        
        # Apply column order if grid_cols is available
        if grid_cols:
            # Ensure all required columns exist
            missing_cols = []
            for col in grid_cols:
                if col not in final_df.columns:
                    final_df[col] = 0
                    missing_cols.append(col)
            
            if missing_cols:
                print(f"Added missing columns: {missing_cols}")
                
            # Order columns according to configuration
            final_df = final_df[grid_cols]
        
        # Save to file
        try:
            final_df.to_csv(grid_file_path)
            print(f"Successfully saved data to {grid_file_path}")
            print(f"Updated hourly grid data with {len(records)} new records")
            print(f"Total records in file: {len(final_df)}")
            print(f"Data range: {final_df.index.min()} to {final_df.index.max()}")
        except Exception as e:
            print(f"Error saving data to {grid_file_path}: {str(e)}")
            # Try saving to a backup file
            backup_path = str(grid_file_path) + ".backup"
            try:
                final_df.to_csv(backup_path)
                print(f"Saved data to backup file: {backup_path}")
            except Exception as e2:
                print(f"Error saving to backup file: {str(e2)}")
        
    except Exception as e:
        print(f"Error updating hourly grid data: {str(e)}")
        import traceback
        print(traceback.format_exc().splitlines()[-3:])  # Only show last 3 lines of traceback for brevity

def fetch_historical_grid_data(project_root):
    """
    Fetch historical power breakdown data for Sweden SE3 from ElectricityMaps API
    starting from 2017-01-01 until today, using the /past-range endpoint with hourly resolution.
    
    This function processes data in 9-day chunks, saving results after each chunk to avoid data loss.
    """
    grid_file_path = project_root / 'data' / 'processed' / 'SwedenGrid.csv'
    
    # Create processed data directory if it doesn't exist
    processed_dir = project_root / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load API key from env file
    dotenv_path = project_root / 'api.env'
    load_dotenv(dotenv_path=dotenv_path)
    api_key = os.getenv('ELECTRICITYMAPS')
    
    if not api_key:
        logger.error("ELECTRICITYMAPS API key not found in api.env file")
        return
    
    logger.info(f"API key loaded: {api_key[:4]}...{api_key[-4:] if len(api_key) > 8 else ''}")
    
    # Load config to ensure correct columns
    config = FeatureConfig()
    grid_cols = config.grid_cols if hasattr(config, 'grid_cols') else None
    
    # Define the essential zones for import/export
    zones = {
        'SE-SE2': 'Main connection from northern Sweden',
        'SE-SE4': 'Main connection to southern Sweden',
        'NO-NO1': 'Norway connection',
        'DK-DK1': 'Denmark connection',
        'FI': 'Finland connection',
        'AX': 'Åland connection'
    }
    
    # Define the power sources
    power_sources = ['nuclear', 'wind', 'hydro', 'solar', 'unknown']
    
    # Fixed start date for historical data
    historical_start_date = datetime(2017, 1, 1)
    end_date = datetime.now()
    
    # Get existing data to identify the dates we already have
    existing_timestamps = set()
    
    if grid_file_path.exists():
        try:
            existing_df = pd.read_csv(grid_file_path, index_col=0)
            if not existing_df.empty:
                # Ensure index is datetime
                existing_df.index = pd.to_datetime(existing_df.index)
                
                # Get all existing timestamps
                existing_timestamps = set(existing_df.index)
                
                logger.info(f"Found existing data with {len(existing_timestamps)} records")
                logger.info(f"Date range: {min(existing_timestamps)} to {max(existing_timestamps)}")
        except Exception as e:
            logger.warning(f"Could not read existing file: {e}")
    
    # Create date ranges to fetch (9 days per chunk)
    chunk_size = 9  # Days per API call
    date_ranges = []
    
    current_date = historical_start_date
    while current_date < end_date:
        # Calculate end date of chunk (9 days or until end_date)
        chunk_end = min(current_date + timedelta(days=chunk_size-1), end_date)
        
        # Check if we need to fetch this chunk (if any hour is missing)
        missing_hours = False
        check_date = current_date
        while check_date <= chunk_end:
            # Check each hour of the day
            for hour in range(24):
                timestamp = check_date.replace(hour=hour)
                if timestamp not in existing_timestamps:
                    missing_hours = True
                    break
            if missing_hours:
                break
            check_date += timedelta(days=1)
        
        # If any hour is missing, add this chunk to our list
        if missing_hours:
            date_ranges.append((current_date, chunk_end))
        
        # Move to next chunk
        current_date += timedelta(days=chunk_size)
    
    if not date_ranges:
        logger.info("All historical data is already present. Nothing to fetch.")
        return
    
    total_chunks = len(date_ranges)
    logger.info(f"Need to fetch {total_chunks} chunks of data (9 days per chunk)")
    
    # API rate limits: Using 1000 requests per hour (as per user's license)
    # We'll use a slightly conservative limit of 900 requests per hour to be safe
    api_calls_per_hour = 900
    api_delay = 4  # seconds between API calls (4 seconds = 900 calls per hour)
    
    # Create a progress bar for all chunks
    with tqdm(total=total_chunks, desc="Processing data chunks", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} chunks [{elapsed}<{remaining}]") as chunk_pbar:
        
        total_hours_saved = 0
        
        for chunk_index, (start_date, end_date) in enumerate(date_ranges):
            try:
                # Format dates for API request
                start_str = start_date.strftime('%Y-%m-%d')
                end_str = end_date.strftime('%Y-%m-%d')
                
                # Update progress bar description
                chunk_pbar.set_description(f"Fetching {start_str} to {end_str}")
                
                # Construct API URL for the past-range endpoint with 60-minute resolution
                api_url = f"https://api.electricitymap.org/v3/power-breakdown/past-range?start={start_str}&end={end_str}&zone=SE-SE3&resolution=60"
                
                # Make API request
                response = requests.get(
                    api_url,
                    headers={"auth-token": api_key}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check if we have history data
                    if 'data' not in data or not data['data']:
                        logger.warning(f"No data returned for range {start_str} to {end_str}")
                        chunk_pbar.set_postfix(status="No data")
                        chunk_pbar.update(1)
                        continue
                    
                    # Process all hourly records in the response
                    chunk_records = {}
                    hours_saved = 0
                    
                    for entry in data['data']:
                        # Get timestamp
                        record_timestamp = pd.to_datetime(entry['datetime'])
                        localized_timestamp = record_timestamp.tz_localize(None)
                        
                        # Skip if we already have this timestamp
                        if localized_timestamp in existing_timestamps:
                            continue
                        
                        # Create a record
                        record = {}
                        
                        # Add core metrics
                        record['fossilFreePercentage'] = entry.get('fossilFreePercentage', 0)
                        record['renewablePercentage'] = entry.get('renewablePercentage', 0)
                        record['powerConsumptionTotal'] = entry.get('powerConsumptionTotal', 0)
                        record['powerProductionTotal'] = entry.get('powerProductionTotal', 0)
                        record['powerImportTotal'] = entry.get('powerImportTotal', 0)
                        record['powerExportTotal'] = entry.get('powerExportTotal', 0)
                        
                        # Get production breakdown
                        prod = entry.get('powerProductionBreakdown', {})
                        
                        # Add power sources
                        for source in power_sources:
                            record[source] = prod.get(source, 0) or 0
                        
                        # Add import/export values
                        import_breakdown = entry.get('powerImportBreakdown', {})
                        export_breakdown = entry.get('powerExportBreakdown', {})
                        
                        for zone in zones:
                            record[f'import_{zone}'] = import_breakdown.get(zone, 0) or 0
                            record[f'export_{zone}'] = export_breakdown.get(zone, 0) or 0
                        
                        # Add to records
                        chunk_records[localized_timestamp] = record
                        hours_saved += 1
                    
                    # Save all records for this chunk
                    if chunk_records:
                        success = save_chunk_data(chunk_records, grid_file_path, grid_cols)
                        
                        if success:
                            # Update tracking
                            existing_timestamps.update(chunk_records.keys())
                            total_hours_saved += hours_saved
                            
                            # Update progress bar
                            chunk_pbar.set_postfix(status=f"Saved {hours_saved} hrs")
                        else:
                            chunk_pbar.set_postfix(status="Save Error")
                    else:
                        chunk_pbar.set_postfix(status="Already complete")
                
                elif response.status_code == 429:
                    # Rate limit exceeded - wait longer
                    logger.warning(f"Rate limit exceeded at chunk {chunk_index+1}. Waiting before retrying...")
                    
                    # Show countdown timer in the progress bar
                    for countdown in range(5, 0, -1):
                        chunk_pbar.set_postfix(rate_limit_wait=f"{countdown}min")
                        time.sleep(60)  # Sleep for 1 minute
                    
                    # Try this chunk again (don't increment chunk_index)
                    chunk_pbar.set_postfix(status="Retrying")
                    continue
                
                elif response.status_code in [401, 403]:
                    error_msg = f"Authentication failed: Status {response.status_code}"
                    try:
                        error_msg += f", {response.json()}"
                    except:
                        pass
                    logger.error(error_msg)
                    chunk_pbar.set_postfix(status="Auth Error")
                    # Exit the process
                    break
                
                else:
                    error_info = "Unknown error"
                    try:
                        error_info = response.json()
                    except:
                        error_info = response.text
                    
                    logger.error(f"API error for chunk {chunk_index+1}: Status {response.status_code}, {error_info}")
                    chunk_pbar.set_postfix(status=f"Error {response.status_code}")
                
                # Update progress bar
                chunk_pbar.update(1)
                
                # Wait between API calls to respect rate limits
                if chunk_index < len(date_ranges) - 1:
                    time.sleep(api_delay)
            
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_index+1}: {e}")
                chunk_pbar.set_postfix(status="Exception")
                import traceback
                logger.error('\n'.join(traceback.format_exc().splitlines()[-10:]))
                chunk_pbar.update(1)
                
                # Continue with next chunk
                continue
        
        # Log final status
        logger.info(f"Historical data backfill complete. Saved {total_hours_saved} hours of data.")

def save_chunk_data(chunk_records, grid_file_path, grid_cols):
    """Save a chunk of hourly records to the CSV file"""
    if not chunk_records:
        return False
    
    try:
        # Create DataFrame with the chunk's records
        timestamps = sorted(chunk_records.keys())
        records = [chunk_records[ts] for ts in timestamps]
        df = pd.DataFrame(records, index=timestamps)
        
        # Round values and fill NAs
        df = df.round(2).fillna(0)
        
        # First time case - no existing data
        if not grid_file_path.exists():
            # Apply column order if grid_cols is available
            if grid_cols:
                # Ensure all required columns exist
                for col in grid_cols:
                    if col not in df.columns:
                        df[col] = 0
                # Order columns according to configuration
                df = df[grid_cols]
            
            # Save to file
            df.to_csv(grid_file_path)
            return True
        
        # Read the existing file
        full_df = pd.read_csv(grid_file_path, index_col=0)
        full_df.index = pd.to_datetime(full_df.index)
        
        # Ensure all columns are in both dataframes
        all_columns = list(set(full_df.columns).union(set(df.columns)))
        for col in all_columns:
            if col not in full_df.columns:
                full_df[col] = 0
            if col not in df.columns:
                df[col] = 0
        
        # Combine dataframes
        combined_df = pd.concat([full_df, df])
        
        # Remove duplicates, keeping the newer data
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        
        # Sort, round, and apply column order
        combined_df = combined_df.sort_index().round(2)
        
        # Apply column order if grid_cols is available
        if grid_cols:
            # Ensure all required columns exist
            for col in grid_cols:
                if col not in combined_df.columns:
                    combined_df[col] = 0
            # Order columns according to configuration
            combined_df = combined_df[grid_cols]
        
        # Save to file
        combined_df.to_csv(grid_file_path)
        return True
        
    except Exception as e:
        logger.error(f"Error saving chunk data: {e}")
        return False

def main():
    feature_config = FeatureConfig() 
    print(feature_config.get_ordered_features())
    
    project_root = Path(__file__).resolve().parents[3]
    
    """Update both price and grid data"""
    # Create processed data directory if it doesn't exist
    processed_dir = project_root / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Update datasets
    print("\nUpdating price data...")
    update_price_data(project_root)
    
    print("\nUpdating grid data (latest 24 hours)...")
    update_grid_data_hourly(project_root)
    
    print("\nFetching historical grid data (from 2017-01-01)...")
    # Show a nice ASCII art progress header
    print("\n" + "="*80)
    print("  Historical Data Backfill Process")
    print("  - Using /past-range endpoint with 60-minute resolution")
    print("  - Processing data in 9-day chunks")
    print("  - Using licensed API rate limit (1000 calls/hour)")
    print("  - Progress is saved after each chunk")
    print("="*80 + "\n")
    
    fetch_historical_grid_data(project_root)
    
    print("\n" + "="*80)
    print("  Process Complete")
    print("  Check the logs for details on the data fetched")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
