import os
import pandas as pd
import yfinance as yf
import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]

# Path to the SE3Prices file
se3_prices_path = project_root / 'data/processed/SE3prices.csv'

# Load the existing SE3Prices.csv file
se3_prices = pd.read_csv(se3_prices_path, parse_dates=True, index_col=0)
print(f"SE3Prices index name: {se3_prices.index.name}")

# Determine the latest date with commodity data
latest_date = None
if all(col in se3_prices.columns for col in ["CO2_Price", "Gas_Price", "Coal_Price"]):
    # First, make sure the index is actually datetime
    se3_prices.index = pd.to_datetime(se3_prices.index)
    
    # Check if we have hourly data
    is_hourly = 'hour' in str(se3_prices.index.dtype).lower() or 'time' in str(se3_prices.index.dtype).lower()
    
    # Find the last timestamp that has non-NaN values for all three commodities
    mask = se3_prices[["CO2_Price", "Gas_Price", "Coal_Price"]].notna().all(axis=1)
    if mask.any():
        # Get the maximum date with valid data
        valid_timestamps = se3_prices.index[mask]
        latest_date = max(valid_timestamps)
        print(f"Latest timestamp with complete commodity data: {latest_date}")
        
        # For display purposes, get the date part
        latest_date_only = latest_date.strftime('%Y-%m-%d')
        latest_time_only = latest_date.strftime('%H:%M')
        print(f"Latest date: {latest_date_only}, time: {latest_time_only}")
    else:
        print("No existing commodity data found")
else:
    print("Commodity price columns not found in existing data")

# If no valid date found, start from a default date
if latest_date is None:
    start_date = "2017-01-01"
    print(f"Using default start date: {start_date}")
else:
    # For daily data sources like Yahoo Finance, we need to get the last calendar date 
    # that needs updating (earliest date missing data)
    
    # Get the last calendar date in the dataset
    last_calendar_date = latest_date.date()
    
    # Get today's date object for comparison
    today_date = datetime.datetime.now().date()
    
    # If the latest data is from today and it's the last hour (or close to it),
    # then we might be fully up to date
    if last_calendar_date == today_date and latest_date.hour >= 22:
        print("Data appears to be up to date for today")
        is_up_to_date = True
        start_date = today_date.strftime('%Y-%m-%d')
    else:
        # Otherwise, we need to update from the last calendar date
        is_up_to_date = False
        start_date = last_calendar_date.strftime('%Y-%m-%d')
        print(f"Need to update commodity prices from {start_date} onwards")

# Get today's date
today = datetime.datetime.now().strftime('%Y-%m-%d')

print(f"Fetching new data from {start_date} to {today}")

# Only fetch new data if needed
if start_date <= today and not (is_up_to_date if 'is_up_to_date' in locals() else False):
    try:
        # Use Yahoo Finance for similar commodities
        # Natural Gas futures
        gas = yf.download("NG=F", start=start_date, end=today)
        print(f"Gas data points: {len(gas)}")
        
        # Coal - use BTU (Peabody Energy) instead of KOL (ETF which may be delisted)
        coal = yf.download("BTU", start=start_date, end=today)
        print(f"Coal data points: {len(coal)}")
        
        # Carbon allowances ETF (KRBN)
        co2 = yf.download("KRBN", start=start_date, end=today)
        print(f"CO2 data points: {len(co2)}")
        
        # Check if we have at least some data
        has_data = not (co2.empty and gas.empty and coal.empty)
        
        if has_data:
            print(f"Downloaded new price data:")
            
            # Important: Create a DataFrame first to avoid keeping the original ticker in the MultiIndex
            co2_prices = pd.DataFrame()
            if not co2.empty:
                co2_prices["CO2_Price"] = co2["Close"]

            gas_prices = pd.DataFrame()
            if not gas.empty:
                gas_prices["Gas_Price"] = gas["Close"]

            coal_prices = pd.DataFrame()
            if not coal.empty:
                coal_prices["Coal_Price"] = coal["Close"]

            # Combine the data
            new_energy_prices = pd.concat([co2_prices, gas_prices, coal_prices], axis=1)
            
            if not new_energy_prices.empty:
                # Make sure indices are datetime
                new_energy_prices.index = pd.to_datetime(new_energy_prices.index)
                
                print(f"New energy prices data dates: {min(new_energy_prices.index)} to {max(new_energy_prices.index)}")
                print(f"Columns available in new data: {', '.join(new_energy_prices.columns)}")

                # Check if SE3Prices is hourly data
                if 'hour' in str(se3_prices.index.dtype).lower() or 'time' in str(se3_prices.index.dtype).lower():
                    print("SE3Prices is hourly data, forward-filling daily commodity prices")
                    
                    # For each date in energy_prices, assign the same price to all hours in that day
                    updated_hours = 0
                    for date, row in new_energy_prices.iterrows():
                        date_str = date.strftime('%Y-%m-%d')
                        mask = se3_prices.index.strftime('%Y-%m-%d') == date_str
                        
                        if mask.any():
                            for col in new_energy_prices.columns:
                                if pd.notna(row[col]):  # Only update if we have a value
                                    se3_prices.loc[mask, col] = row[col]
                                    print(f"Updated {sum(mask)} hours for {date_str} with {col}: {row[col]}")
                                    updated_hours += sum(mask)
                    
                    print(f"Total updated hours: {updated_hours}")
                else:
                    print("SE3Prices is not hourly, using direct merge")
                    # Convert indices to proper datetime if needed
                    
                    # Create new columns if they don't exist
                    for col in new_energy_prices.columns:
                        if col not in se3_prices.columns:
                            se3_prices[col] = None
                    
                    # Update the new data
                    updated_rows = 0
                    for date, row in new_energy_prices.iterrows():
                        if date in se3_prices.index:
                            for col in new_energy_prices.columns:
                                if pd.notna(row[col]):  # Only update if we have a value
                                    se3_prices.loc[date, col] = row[col]
                                    updated_rows += 1
                        else:
                            # This handles new dates that aren't in the original dataset
                            new_row = pd.Series(index=se3_prices.columns)
                            for col in new_energy_prices.columns:
                                if pd.notna(row[col]):
                                    new_row[col] = row[col]
                            se3_prices.loc[date] = new_row
                            updated_rows += 1
                    
                    print(f"Total updated rows: {updated_rows}")

                # Fill any missing values with forward fill then backward fill
                # First check if we had data before
                had_data_before = se3_prices[["CO2_Price", "Gas_Price", "Coal_Price"]].notna().any().any()
                
                # Only apply fill methods if we already had some data
                if had_data_before:
                    se3_prices = se3_prices.ffill().bfill()

                # Round all numeric columns to 3 decimal places
                numeric_columns = se3_prices.select_dtypes(include=['number']).columns
                se3_prices[numeric_columns] = se3_prices[numeric_columns].round(3)

                # Clean up old ticker columns if they exist
                for col in ['KRBN', 'NG=F', 'KOL', 'BTU']:
                    if col in se3_prices.columns:
                        se3_prices = se3_prices.drop(columns=[col])

                # Save the updated data
                se3_prices.to_csv(se3_prices_path)
                print(f"Updated {se3_prices_path} with new energy commodity prices")
                
                # Verify the file was saved
                try:
                    updated_file = pd.read_csv(se3_prices_path, nrows=5)
                    print(f"Verified file was saved with columns: {', '.join(updated_file.columns)}")
                except Exception as e:
                    print(f"WARNING: Could not verify file was saved: {str(e)}")
            else:
                print("No new price data was combined. Check for API issues.")
        else:
            print("No data available for the specified date range from any source")
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
else:
    print("Data is already up to date, no new data to fetch")
