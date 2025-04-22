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
        # Start from 7 days ago instead of 1999 to avoid excessive API calls
        latest_timestamp = pd.Timestamp.now() - timedelta(days=7)
    else:
        # Read the existing CSV file
        df = pd.read_csv(csv_file_path)
        df['HourSE'] = pd.to_datetime(df['HourSE'])
        
        # If the file exists but is empty or corrupted, start from 7 days ago
        if df.empty:
            latest_timestamp = pd.Timestamp.now() - timedelta(days=7)
        else:
            # Get the latest timestamp from the existing data
            latest_timestamp = df['HourSE'].max()
            
            # Ensure we're not going back too far (limit to 7 days ago)
            seven_days_ago = pd.Timestamp.now() - timedelta(days=7)
            if latest_timestamp < seven_days_ago:
                print("Limiting price data update to the last 7 days")
                latest_timestamp = seven_days_ago

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
                            'SE3_price_ore': hour_data['price_sek'],
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
        df["price_24h_avg"] = df["SE3_price_ore"].rolling(window=24, min_periods=1).mean()
        df["price_168h_avg"] = df["SE3_price_ore"].rolling(window=168, min_periods=1).mean()
        df["price_24h_std"] = df["SE3_price_ore"].rolling(window=24, min_periods=1).std()
        df["hour_avg_price"] = df.groupby(df["HourSE"].dt.hour)["SE3_price_ore"].transform("mean")
        df["price_vs_hour_avg"] = df["SE3_price_ore"] / df["hour_avg_price"]

        # Save to CSV
        try:
            df.to_csv(csv_file_path, index=False)
            print(f'Successfully added {len(new_data)} missing hourly price records and updated features.')
        except Exception as e:
            print(f"Error saving price data to CSV: {e}")
    else:
        print("Price data is already up-to-date.")




def update_grid_data(project_root):
    """Update the grid data using Electricity Maps API with hourly resolution"""
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
            df = df[grid_cols]  # Ensure columns are in correct order
            df.sort_index(inplace=True)
            
            # Fill any missing values with 0
            df = df.fillna(0)
            
            # Merge with existing data if it exists and is not empty
            if grid_file_path.exists():
                try:
                    existing_df = pd.read_csv(grid_file_path, index_col=0, parse_dates=True)
                    if not existing_df.empty:
                        # Handle different timestamp formats in existing data
                        try:
                            # Convert string index to datetime objects with proper handling
                            # for timezone information like "2025-04-15 19:00:00+00:00"
                            existing_df.index = pd.to_datetime(existing_df.index)
                            
                            # If timestamps have timezone info, convert to UTC then make naive
                            if existing_df.index.tzinfo is not None:
                                existing_df.index = existing_df.index.tz_convert('UTC').tz_localize(None)
                        except AttributeError:
                            # Handle case where index elements have mixed timezone info
                            # Convert each timestamp individually
                            new_index = []
                            for ts in existing_df.index:
                                ts_dt = pd.to_datetime(ts)
                                if hasattr(ts_dt, 'tzinfo') and ts_dt.tzinfo is not None:
                                    ts_dt = ts_dt.tz_convert('UTC').tz_localize(None)
                                new_index.append(ts_dt)
                            existing_df.index = pd.DatetimeIndex(new_index)
                        
                        # Ensure existing data has all required columns
                        for col in grid_cols:
                            if col not in existing_df.columns:
                                existing_df[col] = 0
                        
                        # Keep only config columns and ensure order
                        existing_df = existing_df[grid_cols]
                        
                        # Round existing data to 2 decimals
                        existing_df = existing_df.round(2)
                        
                        # Fill any missing values with 0
                        existing_df = existing_df.fillna(0)
                        
                        # Combine old and new data
                        df = pd.concat([existing_df, df])
                        df = df[~df.index.duplicated(keep='last')]  # Remove duplicates
                except pd.errors.EmptyDataError:
                    print("Existing file was empty, creating new file with current data.")
                except Exception as e:
                    print(f"Error processing existing grid data: {str(e)}")
                    print("Creating new file with current data only.")
            
            # Sort by date, round all values, and save
            df.sort_index(inplace=True)
            df = df.round(2)  # Final rounding of all values
            df.to_csv(grid_file_path)
            print(f"Successfully updated grid data through {df.index.max().strftime('%Y-%m-%d %H:%M')}")
        else:
            print("No grid data records found in API response.")
        
    except Exception as e:
        print(f"Error updating grid data: {str(e)}")
        if 'response' in locals():
            print("Response content:", response.content.decode() if hasattr(response, 'content') else "No response")

def main():
    feature_config = FeatureConfig() 
    print(feature_config.get_ordered_features())
    
    # Use the same path as defined at the top of the file for consistency
    project_root = Path(__file__).resolve().parents[3]
    
    """Update both price and grid data"""
    # Create processed data directory if it doesn't exist
    processed_dir = project_root / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Update both datasets
    print("\nUpdating price data...")
    update_price_data(project_root)
    
    print("\nUpdating grid data...")
    update_grid_data(project_root)

if __name__ == "__main__":
    main()