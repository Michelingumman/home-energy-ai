import pandas as pd
import numpy as np

def parse_timme(timme_str):
    """
    Converts a string like "2025-01-01 00-01" to a datetime.
    We assume the hour range "HH-HH" indicates the start of the hour.
    """
    parts = timme_str.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Unexpected time format: {timme_str}")
    date_part = parts[0]
    hour_range = parts[1]
    start_hour = hour_range.split('-')[0]
    new_str = f"{date_part} {start_hour}:00:00"
    return pd.to_datetime(new_str)

def calculate_features(df):
    """
    Calculate rolling statistics and hour-based features.
    All prices should already be in öre/kWh.
    """
    # Sort by time to ensure correct rolling calculations
    df = df.sort_values('HourSE')
    
    # Calculate rolling 24h and 168h statistics
    df['price_24h_avg'] = df['SE3_price_ore'].rolling(window=24, min_periods=1).mean()
    df['price_168h_avg'] = df['SE3_price_ore'].rolling(window=168, min_periods=1).mean()
    df['price_24h_std'] = df['SE3_price_ore'].rolling(window=24, min_periods=1).std()
    
    # Calculate hour averages
    df['hour'] = df['HourSE'].dt.hour
    hour_avgs = df.groupby('hour')['SE3_price_ore'].transform('mean')
    df['hour_avg_price'] = hour_avgs
    
    # Calculate price vs hour average
    df['price_vs_hour_avg'] = df['SE3_price_ore'] - df['hour_avg_price']
    
    # Drop temporary columns
    df = df.drop('hour', axis=1)
    
    return df

def process_vsc_file(filename):
    """
    Reads a VSC file with structure:
      Timme ,SEK/MWh 
      2025-01-01 00-01,36.19
      ...
    Converts the time and price (from SEK/MWh to öre/kWh) 
    and returns a DataFrame matching the SE3Prices structure.
    """
    # Read CSV with comma delimiter and remove extra spaces from headers
    df = pd.read_csv(filename, skipinitialspace=True, delimiter=',')
    df.columns = df.columns.str.strip()  # Remove whitespace around column names
    
    # Clean up the price column - remove any thousand separators and convert to float
    df['SEK/MWh'] = df['SEK/MWh'].str.replace(',', '').astype(float)
    
    # Parse the time information and convert price to öre/kWh (multiply by 0.1)
    df['HourSE'] = df['Timme'].apply(parse_timme)
    df['SE3_price_ore'] = df['SEK/MWh'] * 0.1
    
    # Add PriceArea column
    df['PriceArea'] = 'SE3'
    
    # Select and order columns
    df = df[['HourSE', 'PriceArea', 'SE3_price_ore']]
    
    return df

def main():
    # Process the 2024 and 2025 files
    df_2024 = process_vsc_file("data/raw/Elspotprices/nordpools priser 2020-2025/2024.csv")
    df_2025 = process_vsc_file("data/raw/Elspotprices/nordpools priser 2020-2025/2025.csv")
    
    # Merge the data
    merged_df = pd.concat([df_2024, df_2025], ignore_index=True)
    merged_df = merged_df.sort_values('HourSE')
    
    # Calculate all required features
    merged_df = calculate_features(merged_df)
    
    # Ensure all required columns are present and in correct order
    required_columns = [
        'HourSE', 'PriceArea', 'SE3_price_ore', 'price_24h_avg',
        'price_168h_avg', 'price_24h_std', 'hour_avg_price',
        'price_vs_hour_avg'
    ]
    merged_df = merged_df[required_columns]
    
    # Save the merged data
    output_path = "data/raw/Elspotprices/nordpools priser 2020-2025/SE3prices.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"Merged CSV saved as {output_path}")

if __name__ == "__main__":
    main()
