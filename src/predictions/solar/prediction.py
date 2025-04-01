import requests
import pandas as pd
import json
import datetime
from typing import Dict, List
from pathlib import Path
import sys
import os
from dotenv import load_dotenv


# Define direction to azimuth mapping   
azimuth_map = {
    "north": 180,
    "northeast": -135,
    "east": -90,
    "southeast": -45,
    "south": 0,
    "southwest": 45,
    "west": 90,
    "northwest": 135
}

class SolarPrediction:
    def __init__(self, config_path="config.json"):
        """
        Initialize the solar prediction with configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration JSON file
        """
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Extract location information
        location = self.config["system_specs"]["location"]
        self.latitude = location["latitude"]
        self.longitude = location["longitude"]
        
        # Load API key from environment variables
        self.api_key = self.load_api_key()
        
        # Base URL for the forecast.solar API
        self.base_url = "https://api.forecast.solar"
        
    def load_config(self, config_path):
        """Load configuration from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            raise
            
    def load_api_key(self):
        """Load API key from env file."""
        # Try to load from environment variable first
        api_key = os.environ.get("FORECASTSOLAR")
        
        # If not found, try to load from api.env file
        if not api_key:
            try:
                # Look for api.env in project root or current directory
                env_paths = [
                    Path.cwd() / "api.env",
                    Path.cwd().parent.parent.parent / "api.env",  # Try project root
                    Path(__file__).parent.parent.parent.parent / "api.env"  # Another approach
                ]
                
                for env_path in env_paths:
                    if env_path.exists():
                        # Load the .env file
                        load_dotenv(env_path)
                        api_key = os.environ.get("FORECASTSOLAR")
                        if api_key:
                            print(f"Loaded API key from {env_path}")
                            break
                
                if not api_key:
                    print("Warning: FORECASTSOLAR API key not found. Using public API with limited features.")
            except Exception as e:
                print(f"Error loading API key: {e}")
                print("Continuing with public API (limited features)")
        
        return api_key
    
    def get_panel_groups(self):
        """
        Create panel groups based on the configuration.
        
        Returns:
            List of dictionaries containing panel group information
        """
        specs = self.config["system_specs"]
        tilt = specs["tilt_degrees"]
        panel_power = specs["panel_power_w"]

        # Create panel groups based on the configuration
        panel_groups = []
        
        for direction in ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]:
            panel_count_key = f"panel_count_{direction}"
            
            # Skip if panel count key is not in config
            if panel_count_key not in specs:
                continue
                
            panel_count = specs[panel_count_key]
            
            # Skip if no panels in this direction
            if panel_count <= 0:
                continue
                
            panel_groups.append({
                "tilt": tilt,
                "azimuth": azimuth_map[direction],
                "panel_count": panel_count,
                "panel_power_w": panel_power
            })
        
        return panel_groups
    
    def get_prediction_for_panel_group(
        self, 
        tilt: float, 
        azimuth: float, 
        panel_count: int, 
        panel_power_w: int,
        start: str = "now"
    ) -> Dict:
        """
        Get prediction for a group of panels with the same orientation.
        
        Args:
            tilt: Panel tilt in degrees (0 = horizontal, 90 = vertical)
            azimuth: Panel azimuth in degrees (0 = north, 90 = east, 180 = south, 270 = west)
            panel_count: Number of panels in this group
            panel_power_w: Power rating of each panel in watts
            start: When to start the forecast (API only supports "now")
            
        Returns:
            Dictionary containing hourly energy predictions for the next 3 days
        """
        total_power_kw = (panel_count * panel_power_w) / 1000
        
        # Parameters for the API request
        params = {
            'full': 1,     # Get full 24-hour data with 0 values outside daylight
            'limit': 4,    # Get forecast for 4 days (today + 3 days ahead)
            'damping': 1,  # Use default damping factor for realistic forecasts
            'resolution': 60  # Request hourly data (60 minutes)
        }
        
        # Construct the URL differently based on whether API key is available
        if self.api_key:
            # Authenticated API call using personal API key
            base_endpoint = f"{self.base_url}/{self.api_key}/estimate/watthours/{self.latitude}/{self.longitude}/{tilt}/{azimuth}/{total_power_kw}"
        else:
            # Public API call (limited features)
            base_endpoint = f"{self.base_url}/estimate/watthours/{self.latitude}/{self.longitude}/{tilt}/{azimuth}/{total_power_kw}"
        
        # Add query parameters
        url = f"{base_endpoint}?" + "&".join(f"{k}={v}" for k, v in params.items())
        print("URL", url, '\n')
        
        try:
            direction = list(azimuth_map.keys())[list(azimuth_map.values()).index(azimuth)]
            if self.api_key:
                print(f"Making authenticated API request for {panel_count} panels facing {direction} (azimuth {azimuth}°)...")
            else:
                print(f"Making public API request for {panel_count} panels facing {direction} (azimuth {azimuth}°)...")
            print(f"Parameters: Start=now, Days=4, Resolution=hourly")
            
            # Request JSON format
            headers = {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
            
            # Make the request
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            # Check for rate limiting info in response
            if 'message' in data and 'ratelimit' in data['message']:
                rate_info = data['message']['ratelimit']
                print(f"API Rate Limit: {rate_info.get('remaining', 'N/A')}/{rate_info.get('limit', 'N/A')} requests remaining in this period")
            
            # Check if we have a result section
            if 'result' in data:
                # Extract the raw result (could be directly in 'result' or in 'result.watt_hours_period')
                predictions = data['result']
                
                # Check different possible structures of the response
                if isinstance(predictions, dict):
                    # Direct timestamp-value pairs in the result
                    if predictions and any(isinstance(v, (int, float)) for v in predictions.values()):
                        return {"watt_hours_period": predictions}
                    # Check if watt_hours_period key exists
                    elif 'watt_hours_period' in predictions and predictions['watt_hours_period']:
                        return {"watt_hours_period": predictions['watt_hours_period']}
            
            print("No valid predictions found in response")
            print("API Response:", data)
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"Error making API request: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print("Raw response:", response.text)
            return None
    
    def combine_predictions(self, predictions: List[Dict]) -> pd.DataFrame:
        """
        Combine predictions from multiple panel groups.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            DataFrame with hourly energy predictions in kWh
        """
        combined_df = None
        
        print("\nProcessing predictions from each panel group:")
        for i, prediction in enumerate(predictions, 1):
            if prediction and 'watt_hours_period' in prediction:
                print(f"\nPanel group {i}:")
                # Create a DataFrame from this prediction
                wh = prediction['watt_hours_period']
                df = pd.DataFrame(list(wh.items()), columns=['timestamp', 'watt_hours'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df = df.sort_index()  # Ensure timestamps are sorted
                
                # Group by date and calculate differences within each day
                df['date'] = df.index.date
                df['watt_hours_diff'] = df.groupby('date')['watt_hours'].diff().fillna(df['watt_hours'])
                
                # Handle edge cases: if diff is negative or unreasonably large, use the original value
                mask = (df['watt_hours_diff'] < 0) | (df['watt_hours_diff'] > 10000)  # 10kWh threshold
                df.loc[mask, 'watt_hours_diff'] = df.loc[mask, 'watt_hours']
                
                df['watt_hours'] = df['watt_hours_diff']
                df = df.drop(['date', 'watt_hours_diff'], axis=1)
                
                print(f"Time range: {df.index.min()} to {df.index.max()}")
                print(f"Number of entries: {len(df)}")
                
                # Ensure hourly resampling
                df = df.resample('h').sum()
                print(f"After resampling to hourly: {len(df)} entries")
                
                if combined_df is None:
                    combined_df = df
                else:
                    # Add energy values
                    combined_df = combined_df.join(df, how='outer', rsuffix='_new')
                    combined_df['watt_hours'] = combined_df['watt_hours'].fillna(0) + combined_df['watt_hours_new'].fillna(0)
                    combined_df.drop('watt_hours_new', axis=1, inplace=True)
                    print("Combined with previous predictions")
        
        if combined_df is not None:
            # Convert watt-hours to kilowatt-hours
            combined_df['kilowatt_hours'] = combined_df['watt_hours'] / 1000
            print(f"\nFinal combined predictions:")
            print(f"Time range: {combined_df.index.min()} to {combined_df.index.max()}")
            print(f"Total entries: {len(combined_df)}")
            return combined_df
        
        return None

    def append_to_merged_file(self, df: pd.DataFrame, forecasted_dir: Path) -> None:
        """
        Append new predictions to a merged CSV file, handling potential duplicates.
        
        Args:
            df: DataFrame with new predictions
            forecasted_dir: Directory to save the merged file
        """
        # Ensure forecasted_dir exists
        forecasted_dir.mkdir(parents=True, exist_ok=True)
        merged_file_path = forecasted_dir / "merged_predictions.csv"
        
        # Reset index to make timestamp a column for easier CSV handling
        df_to_save = df.reset_index()
        
        # If merged file already exists, read it and append new data
        if merged_file_path.exists():
            try:
                print(f"Reading existing merged file: {merged_file_path}")
                existing_df = pd.read_csv(merged_file_path)
                
                # Convert timestamp column to datetime
                existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                
                # Remove any existing entries with the same timestamps as new data
                existing_timestamps = set(df_to_save['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S'))
                existing_df = existing_df[~existing_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').isin(existing_timestamps)]
                
                # Append new data
                merged_df = pd.concat([existing_df, df_to_save], ignore_index=True)
                
                # Sort by timestamp
                merged_df = merged_df.sort_values('timestamp')
                
                # Save back to file
                merged_df.to_csv(merged_file_path, index=False)
                print(f"Added {len(df_to_save)} new entries to merged file. Total entries: {len(merged_df)}")
                
            except Exception as e:
                print(f"Error appending to merged file: {e}")
                # If error occurs, just save the new data (overwriting existing file)
                df_to_save.to_csv(merged_file_path, index=False)
                print(f"Saved new predictions to {merged_file_path} (overwrote due to error)")
        else:
            # If file doesn't exist, create it with the new data
            df_to_save.to_csv(merged_file_path, index=False)
            print(f"Created new merged file with {len(df_to_save)} entries: {merged_file_path}")
    
    def run_prediction(self, output_dir=None, forecasted_dir=None, start="now"):
        """
        Run the prediction process and save results to CSV.
        
        Args:
            output_dir: Directory to save individual date CSV files
            forecasted_dir: Directory to save the merged CSV file 
            start: When to start the forecast (API only supports "now")
        
        Returns:
            Dictionary of DataFrames with hourly energy predictions for the current and next 3 days
        """
        # Display warning if start is not "now"
        if start != "now":
            print("Warning: The forecast.solar API only supports forecasts starting from the current day.")
            print("The provided start date will be ignored and today's date will be used instead.")
            start = "now"
            
        # Get panel groups
        panel_groups = self.get_panel_groups()
        
        if not panel_groups:
            print("No valid panel groups found in configuration.")
            return None
        
        # Get predictions for each panel group
        predictions = []
        for group in panel_groups:
            prediction = self.get_prediction_for_panel_group(
                group["tilt"], 
                group["azimuth"], 
                group["panel_count"], 
                group["panel_power_w"],
                start=start
            )
            predictions.append(prediction)
        
        # Combine predictions
        combined_df = self.combine_predictions(predictions)
        
        if combined_df is not None:
            # Parse the start date to determine the requested date
            if start == "now":
                start_date = datetime.datetime.now().date()
            else:
                # Handle both YYYY-MM-DD and datetime objects
                if isinstance(start, str):
                    start_date = datetime.datetime.strptime(start, "%Y-%m-%d").date()
                else:
                    start_date = start.date()
            
            # Create a dictionary to store DataFrames for each date
            date_dfs = {}
            
            # Get all unique dates in the prediction
            unique_dates = pd.Series(combined_df.index.date).unique()
            
            print(f"\nProcessing predictions for {len(unique_dates)} days (4-day forecast):")
            
            # Process each date
            for date in unique_dates:
                date_mask = combined_df.index.date == date
                date_df = combined_df[date_mask]
                
                if not date_df.empty:
                    date_dfs[date] = date_df
                    print(f"\nHourly Energy Production (kWh) for {date}:")
                    print(date_df[['kilowatt_hours']])
                    
                    # Save to CSV with date-based filename
                    if output_dir:
                        # Create data directory if it doesn't exist
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Format the date for the filename
                        filename = f"{date.strftime('%Y%m%d')}.csv"
                        csv_path = output_dir / filename
                        
                        date_df.to_csv(csv_path)
                        print(f"Predictions for {date} saved to {csv_path}")
                else:
                    print(f"No prediction data available for {date}.")
            
            # Append to merged file if forecasted_dir is provided
            if forecasted_dir and combined_df is not None:
                self.append_to_merged_file(combined_df, forecasted_dir)
            
            return date_dfs
        else:
            print("Failed to get predictions.")
            return None


def main():
    script_dir = Path(__file__).parent
    config_path = script_dir / "config.json"
    
    # New directory structure for individual day files
    forecasted_dir = script_dir / "forecasted_data"
    per_day_dir = forecasted_dir / "per_day"
    
    try:
        # Check for command line arguments but only for backward compatibility
        if len(sys.argv) > 1:
            print("WARNING: The forecast.solar API only supports forecasts starting from the current day.")
            print("The provided date argument will be ignored.")
            
        # Use today's date
        start_date = "now"
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        print(f"Starting predictions from today: {today}")
        
        # Show forecast period
        print(f"Fetching 4-day forecast (today + 3 days ahead)")
        
        # Initialize solar prediction with config file
        solar = SolarPrediction(config_path=config_path)
        
        # Run prediction with specific parameters and also save to merged file
        date_dfs = solar.run_prediction(
            output_dir=per_day_dir,
            forecasted_dir=forecasted_dir,
            start=start_date
        )
        
        if date_dfs:
            print(f"\nSummary: Generated predictions for {len(date_dfs)} days:")
            for date, df in date_dfs.items():
                print(f"  - {date}: {len(df)} hourly values, saved to {per_day_dir / date.strftime('%Y%m%d')}.csv")
            print(f"  - All predictions also added to {forecasted_dir / 'merged_predictions.csv'}")
            
            # Show forecast range
            dates = list(date_dfs.keys())
            if dates:
                min_date = min(dates)
                max_date = max(dates)
                if min_date != max_date:
                    print(f"\nForecast period: {min_date} to {max_date} (4-day forecast)")
                    
    except Exception as e:
        print(f"Error running solar prediction: {e}")


if __name__ == "__main__":
    main()