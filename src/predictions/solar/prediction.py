import requests
import pandas as pd
import json
import datetime
from typing import Dict, List
from pathlib import Path
import sys


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
        
        # Base URL for the forecast.solar API
        self.base_url = "https://api.forecast.solar/estimate"
        
    def load_config(self, config_path):
        """Load configuration from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            raise
    
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
        start: str = "now",
        days_horizon: int = 7,
        resolution_minutes: int = 60
    ) -> Dict:
        """
        Get prediction for a group of panels with the same orientation.
        
        Args:
            tilt: Panel tilt in degrees (0 = horizontal, 90 = vertical)
            azimuth: Panel azimuth in degrees (0 = north, 90 = east, 180 = south, 270 = west)
            panel_count: Number of panels in this group
            panel_power_w: Power rating of each panel in watts
            start: When to start the forecast ("now" or time like "12:00")
            days_horizon: Number of days to forecast (1-8)
            resolution_minutes: Time resolution in minutes (15, 30, or 60)
            
        Returns:
            Dictionary containing hourly energy predictions
        """
        total_power_kw = (panel_count * panel_power_w) / 1000
        
        # Validate parameters
        if days_horizon < 1 or days_horizon > 8:
            raise ValueError("days_horizon must be between 1 and 8")
        if resolution_minutes not in [15, 30, 60]:
            raise ValueError("resolution_minutes must be 15, 30, or 60")
        
        # Construct the API URL for watthours with parameters
        params = {
            'full': 1,  # Get full 24-hour data with 0 values outside daylight
            'limit': days_horizon,  # Number of days to forecast
            'start': start,  # Start time
            'resolution': resolution_minutes  # Time resolution
        }
        
        base_url = f"{self.base_url}/watthours/{self.latitude}/{self.longitude}/{tilt}/{azimuth}/{total_power_kw}"
        url = f"{base_url}?" + "&".join(f"{k}={v}" for k, v in params.items())
        print("URL", url, '\n')
        
        try:
            direction = list(azimuth_map.keys())[list(azimuth_map.values()).index(azimuth)]
            print(f"Making API request for {panel_count} panels facing {direction} (azimuth {azimuth}°)...")
            print(f"Parameters: Start={start}, Horizon={days_horizon} days, Resolution={resolution_minutes} minutes")
            
            # Request JSON format
            headers = {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            # Check if we have a result section
            if 'result' in data:
                # The result section directly contains timestamp: value pairs
                predictions = data['result']
                if predictions:  # Check if we have any predictions
                    return {"watt_hours_period": predictions}
            
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
    
    def run_prediction(self, output_dir=None, start="now", days_horizon=7, resolution_minutes=60):
        """
        Run the prediction process and save results to CSV.
        
        Args:
            output_dir: Directory to save the CSV file
            start: When to start the forecast ("now" or time like "12:00")
            days_horizon: Number of days to forecast (1-8)
            resolution_minutes: Time resolution in minutes (15, 30, or 60)
        
        Returns:
            DataFrame with hourly energy predictions
        """
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
                start=start,
                days_horizon=days_horizon,
                resolution_minutes=resolution_minutes
            )
            predictions.append(prediction)
        
        # Combine predictions
        combined_df = self.combine_predictions(predictions)
        
        if combined_df is not None:
            # Print the predictions
            print("\nHourly Energy Production (kWh):")
            print(combined_df[['kilowatt_hours']])
            
            # Save to CSV with date-based filename
            if output_dir:
                # Create data directory if it doesn't exist
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Get the start date from the DataFrame for the filename
                start_date = combined_df.index[0].strftime("%Y%m%d")
                
                # Create filename with just the start date
                filename = f"{start_date}.csv"
                csv_path = output_dir / filename
                
                combined_df.to_csv(csv_path)
                print(f"Predictions saved to {csv_path}")
            
            return combined_df
        else:
            print("Failed to get predictions.")
            return None


def main():
    config_path = Path(__file__).parent / "config.json"
    data_dir = Path(__file__).parent / "data"
    
    try:
        # Get the date from command line argument or use today
        if len(sys.argv) > 1:
            # Accept either YYYYMMDD or YYYY-MM-DD format
            date_str = sys.argv[1].replace('-', '')
            # Convert to YYYY-MM-DD format for the API
            start_date = datetime.datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
        else:
            # Use today's date
            start_date = datetime.datetime.now().strftime("%Y-%m-%d")
            
        # Get the horizon window from command line argument or use default
        days_horizon = 7  # Default to 7 days
        if len(sys.argv) > 2:
            try:
                days_horizon = int(sys.argv[2])
                if days_horizon < 1 or days_horizon > 8:
                    raise ValueError("Horizon must be between 1 and 8 days")
            except ValueError as e:
                print(f"Error: Invalid horizon value. {str(e)}")
                print("Please specify a number between 1 and 8 days.")
                return
        
        # Initialize solar prediction with config file
        solar = SolarPrediction(config_path=config_path)
        
        # Run prediction with specific parameters
        df = solar.run_prediction(
            output_dir=data_dir,
            start=start_date,  # Use the formatted date
            days_horizon=days_horizon,  # Use specified or default horizon
            resolution_minutes=60
        )
        
    except ValueError as e:
        print(f"Error: Invalid date format. Please use YYYYMMDD or YYYY-MM-DD format.")
        print(f"Example: python prediction.py 20250227 [days] or python prediction.py 2025-02-27 [days]")
        print(f"Where [days] is optional and must be between 1 and 8 (default is 7)")
    except Exception as e:
        print(f"Error running solar prediction: {e}")


if __name__ == "__main__":
    main()