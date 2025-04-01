from pathlib import Path
import pandas as pd
import requests
from datetime import datetime, timedelta
import json
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import os

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
    
    print("\nUpdating grid data (hourly resolution)...")
    update_grid_data_hourly(project_root)

if __name__ == "__main__":
    main()
