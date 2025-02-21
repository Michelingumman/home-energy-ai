from pathlib import Path
import pandas as pd
import requests
from datetime import datetime, timedelta

# Determine the project root directory dynamically
project_root = Path(__file__).resolve().parent.parent.parent.parent  # Moves up two levels to the root

# Define the path to the SE3prices.csv file dynamically
csv_file_path = project_root / 'data' / 'processed' / 'SE3prices.csv'

# Check if the file exists
if not csv_file_path.exists():
    raise FileNotFoundError(f"CSV file not found at {csv_file_path}")

# Read the existing CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Convert the 'HourSE' column to datetime objects
df['HourSE'] = pd.to_datetime(df['HourSE'])

# Find the latest timestamp in the CSV
latest_timestamp = df['HourSE'].max()

# Get the current time
current_time = datetime.now()

# List to store new data
new_data = []

# Loop through missing hours until the present
while latest_timestamp < current_time:
    # Move to the next hour
    latest_timestamp += timedelta(hours=1)
    
    # Extract date and hour in required formats
    next_date_str = latest_timestamp.strftime('%Y-%m-%d')
    next_hour_int = latest_timestamp.hour
    next_hour_str = latest_timestamp.strftime('%H:00:00')

    # Fetch data from the API for the respective date
    api_url = f'https://mgrey.se/espot?format=json&date={next_date_str}'
    response = requests.get(api_url)
    
    if response.status_code == 200:
        data = response.json()
        
        # Check if 'SE3' data is available for the fetched date
        if 'SE3' in data:
            se3_data = data['SE3']
            
            # Find the data for the specific hour
            hour_data = next((item for item in se3_data if item['hour'] == next_hour_int), None)
            
            if hour_data:
                # Store new row in list
                new_data.append({
                    'HourSE': f'{next_date_str} {next_hour_str}',
                    'PriceArea': 'SE3',
                    'SE3_price_ore': hour_data['price_sek'], # price_sek is in öre/kwh according to API --> https://mgrey.se/espot/api
                })
            else:
                print(f"No data available for {next_date_str} {next_hour_str}")
        else:
            print(f"No SE3 data available for {next_date_str}")
    else:
        print(f"Failed to fetch data for {next_date_str}. Status code: {response.status_code}")
        break  # Stop fetching if API fails

# If new data was added, update the DataFrame
if new_data:
    new_df = pd.DataFrame(new_data)
    
    # Convert HourSE to datetime for merging
    new_df['HourSE'] = pd.to_datetime(new_df['HourSE'])
    
    # Concatenate new data to existing DataFrame
    df = pd.concat([df, new_df], ignore_index=True)

    # ** FEATURE CALCULATION **
    df = df.sort_values(by="HourSE")  # Ensure data is sorted before calculating rolling values

    # 24-hour rolling mean of price
    df["price_24h_avg"] = df["SE3_price_ore"].rolling(window=24, min_periods=1).mean()

    # 168-hour (7-day) rolling mean
    df["price_168h_avg"] = df["SE3_price_ore"].rolling(window=168, min_periods=1).mean()

    # 24-hour rolling standard deviation
    df["price_24h_std"] = df["SE3_price_ore"].rolling(window=24, min_periods=1).std()

    # Calculate hourly average price over multiple days
    df["hour_avg_price"] = df.groupby(df["HourSE"].dt.hour)["SE3_price_ore"].transform("mean")

    # Price vs hourly average (how high/low price is compared to avg for that hour)
    df["price_vs_hour_avg"] = df["SE3_price_ore"] / df["hour_avg_price"]

    # Save the updated DataFrame back to the CSV
    df.to_csv(csv_file_path, index=False)
    print(f'Successfully added {len(new_data)} missing hourly records and updated features.')
else:
    print("CSV file is already up-to-date.")
