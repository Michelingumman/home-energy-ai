#!/usr/bin/env python
import sys
import os
import csv
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from ThermiaOnlineAPI import Thermia
from dotenv import load_dotenv
import argparse

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'api.env'))


CHANGE_HEAT_PUMP_DATA_DURING_TEST = (
    False  # Set to True if you want to change heat pump data during test
)

USERNAME = os.getenv("THERMIA_USERNAME")
PASSWORD = os.getenv("THERMIA_PASSWORD")

if not USERNAME or not PASSWORD:
    print("No username or password found in api.env")
    exit()

thermia = Thermia(USERNAME, PASSWORD)

if not thermia.connected:
    print("Failed to connect to Thermia API.")
    exit()

print("Connected to Thermia API successfully.")

heat_pump = thermia.heat_pumps[0]
print(f"Operating on heat pump: {str(heat_pump.model)}")


def calculate_power_status(data):
    """
    Calculate when the compressor is drawing power based on changes in operation time.
    Returns a dataframe with timestamps and power status (1=on, 0=off).
    Assumes 'data' is a list of dicts [{'time': ..., 'value': ...}] from the API.
    """
    if not data or len(data) < 2:
        print("Not enough data to calculate power status.")
        return pd.DataFrame(columns=["timestamp", "power_status"])
    
    df = pd.DataFrame(data)
    # API returns 'time' and 'value' keys
    df = df.rename(columns={'time': 'timestamp', 'value': 'operation_time'})
    
    # Ensure timestamp is datetime and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values("timestamp")
    
    # Calculate the difference in operation time between consecutive readings
    df["operation_time_diff"] = df["operation_time"].diff()
    
    # If operation time increased, the compressor was running (drawing power)
    df["power_status"] = (df["operation_time_diff"] > 0).astype(int)
    
    # First row will have NaN diff, assume off initially or based on first valid reading
    if not df.empty:
        df.loc[df.index[0], "power_status"] = 0 # Default first status to off
        # A more robust way might be to see if the first operation_time is > 0 and previous was 0,
        # but diff() handles changes well. For the very first point, 0 is a safe assumption.

    result = df[["timestamp", "power_status"]]
    return result

def plot_power_input(file_to_plot):
    """Loads data from the CSV and plots power_input_kw over time."""
    if file_to_plot is None or not os.path.exists(file_to_plot):
        print(f"Error: CSV file not found at {file_to_plot} for plotting.")
        return

    try:
        df = pd.read_csv(file_to_plot)
    except Exception as e:
        print(f"Error loading CSV file for plotting: {e}")
        return

    if 'timestamp' not in df.columns or 'power_input_kw' not in df.columns:
        print("Error: CSV file must contain 'timestamp' and 'power_input_kw' columns for plotting.")
        return

    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        print(f"Error converting 'timestamp' column to datetime for plotting: {e}")
        return

    df = df.sort_values('timestamp')

    plt.figure(figsize=(15, 7))
    plt.step(df['timestamp'], df['power_input_kw'], label='Estimated Power Input (kW)', color='blue', linewidth=1)
    
    plt.xlabel('Time')
    plt.ylabel('Estimated Power Input (kW)')
    plt.title('Heat Pump Estimated Power Input Over Time (15-min intervals)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    print("\nDisplaying plot...")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Extract and process historical data from Thermia heat pump')
    parser.add_argument('--days', type=int, default=7, help='Number of days of historical data to fetch')
    parser.add_argument('--plot', action='store_true', help='Display a plot of the power_input_kw over time after processing')
    args = parser.parse_args()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    print(f"Fetching data from {start_date} to {end_date}")

    processed_data_frames = [] # List to hold individually resampled DataFrames

    # 1. Process Compressor Operation Time (Power Status)
    print("Fetching and processing REG_OPER_TIME_COMPRESSOR for power_status...")
    compressor_raw_data = heat_pump.get_historical_data_for_register("REG_OPER_TIME_COMPRESSOR", start_date, end_date)
    power_status_df = calculate_power_status(compressor_raw_data)
    
    if power_status_df is not None and not power_status_df.empty:
        power_status_df = power_status_df.set_index('timestamp').resample('15T')['power_status'].max().to_frame()
        if not power_status_df.empty:
            processed_data_frames.append(power_status_df)
            print(f"Processed power_status: {len(power_status_df)} 15-min records")
        else:
            print("Power status data became empty after resampling.")
    else:
        print("No power status data to process or it was empty initially.")

    # 2. Define other registers to fetch, clean, and resample
    registers_to_process = {
        "REG_SUPPLY_LINE": "supply_temp",
        "REG_OPER_DATA_RETURN": "return_temp",
        "REG_BRINE_IN": "brine_in_temp",
        "REG_BRINE_OUT": "brine_out_temp",
        "REG_OUTDOOR_TEMPERATURE": "outdoor_temp",
        "REG_DESIRED_SYS_SUPPLY_LINE_TEMP": "desired_supply_temp"
    }

    for reg_api_name, col_name in registers_to_process.items():
        print(f"Fetching and processing {reg_api_name} as {col_name}...")
        raw_data = heat_pump.get_historical_data_for_register(reg_api_name, start_date, end_date)
        
        if raw_data:
            temp_df = pd.DataFrame(raw_data)
            if temp_df.empty:
                print(f"No raw data for {reg_api_name}.")
                continue

            temp_df = temp_df.rename(columns={'time': 'timestamp', 'value': col_name})
            temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
            temp_df[col_name] = pd.to_numeric(temp_df[col_name], errors='coerce')
            
            min_val, max_val = None, None
            if col_name in ['supply_temp', 'return_temp']:
                min_val, max_val = -5, 90
            elif col_name == 'desired_supply_temp':
                min_val, max_val = 0, 70
            elif col_name in ['brine_in_temp', 'brine_out_temp']:
                min_val, max_val = -20, 40
            elif col_name == 'outdoor_temp':
                min_val, max_val = -40, 55
            
            if min_val is not None and max_val is not None:
                original_count = len(temp_df)
                temp_df.loc[~temp_df[col_name].between(min_val, max_val, inclusive='both'), col_name] = pd.NA
                if original_count - temp_df[col_name].count() > 0:
                    print(f"Cleaned {original_count - temp_df[col_name].count()} outliers for {col_name} (range {min_val} to {max_val}).")
            
            temp_df.dropna(subset=[col_name], inplace=True)
            if temp_df.empty:
                print(f"No valid data for {reg_api_name} after cleaning.")
                continue

            resampled_series = temp_df.set_index('timestamp').resample('15T')[col_name].mean().to_frame()
            if not resampled_series.empty:
                processed_data_frames.append(resampled_series)
                print(f"Processed {col_name}: {len(resampled_series)} 15-min records")
            else:
                print(f"{col_name} data became empty after resampling.")
        else:
            print(f"No data returned for {reg_api_name}.")

    # --- Combine all resampled data --- 
    if not processed_data_frames:
        print("\nNo data available from any register after individual processing. Exiting.")
        return
    
    print("\n--- Combining all individually resampled dataframes ---")
    resampled_df = pd.concat(processed_data_frames, axis=1) # Joins on index (15T timestamps)

    if resampled_df.empty:
        print("Combined dataframe is empty. Exiting.")
        return

    # Ensure power_status column exists, vital for subsequent calculations
    if 'power_status' not in resampled_df.columns:
        print("\nCritical 'power_status' column is missing after concatenation. Cannot proceed.")
        # Optionally save the partial df for debugging
        # resampled_df.reset_index().to_csv(os.path.join(output_dir, f"debug_missing_power_status_data.csv")) 
        return
    
    # Drop rows where resampled 'power_status' is NaN (intervals with no compressor data at all)
    resampled_df.dropna(subset=['power_status'], inplace=True)
    if resampled_df.empty:
        print("\nDataFrame empty after dropping rows with NaN power_status. Exiting.")
        return
        
    print(f"Combined and primary filtered dataframe: {len(resampled_df)} records")

    # --- CALCULATIONS on the combined 15-minute data ---
    print("\n--- Performing calculations on 15-minute data ---")

    # 1. Delta T (Raw)
    resampled_df['delta_temp_raw'] = resampled_df.get('supply_temp', pd.Series(dtype=float)) - resampled_df.get('return_temp', pd.Series(dtype=float))
    
    # 2. Clamp Delta T
    # Clamped to 2-10 K. If original delta_temp_raw is NaN, delta_temp_clamped will be NaN.
    resampled_df['delta_temp_clamped'] = resampled_df['delta_temp_raw'].clip(lower=2, upper=10)
    print("Calculated and clamped delta_T.")

    # Constants for heat output calculation
    MASS_FLOW_RATE_KGS = 0.55  # kg/s (estimated)
    SPECIFIC_HEAT_WATER_KJ_KGK = 4.18  # kJ/kgK

    # 3. Heat Output (kW) using CLAMPED Delta T
    resampled_df['heat_output_kw'] = 0.0
    # Mask for valid calculation: power_status is 1, and clamped delta_T is a valid number.
    heat_calc_mask = (resampled_df['power_status'] == 1) & resampled_df['delta_temp_clamped'].notna()
    resampled_df.loc[heat_calc_mask, 'heat_output_kw'] = MASS_FLOW_RATE_KGS * SPECIFIC_HEAT_WATER_KJ_KGK * resampled_df.loc[heat_calc_mask, 'delta_temp_clamped']
    print("Calculated heat_output_kw.")

    # 4. Estimate COP (Coefficient of Performance)
    FIXED_COP = 4.75  # Updated COP for Thermia Atlas 12 at B0/W35
    resampled_df['cop'] = FIXED_COP 
    # COP is only relevant if heat output is positive, otherwise set to NaN.
    resampled_df.loc[resampled_df['heat_output_kw'] <= 0, 'cop'] = pd.NA
    print(f"Applied fixed COP: {FIXED_COP}")

    # 5. Calculate Power Input (kW)
    resampled_df['power_input_kw'] = 0.0
    # Mask for valid calculation: heat_output > 0 and COP is valid and positive.
    power_calc_mask = (resampled_df['heat_output_kw'] > 0) & resampled_df['cop'].notna() & (resampled_df['cop'] > 0)
    resampled_df.loc[power_calc_mask, 'power_input_kw'] = resampled_df.loc[power_calc_mask, 'heat_output_kw'] / resampled_df.loc[power_calc_mask, 'cop']
    print("Calculated power_input_kw.")

    # 6. Clamp Power Input to physical limits (max 4.7 kW)
    MAX_POWER_INPUT_KW = 4.7
    resampled_df['power_input_kw'] = resampled_df['power_input_kw'].clip(upper=MAX_POWER_INPUT_KW)
    print(f"Clamped power_input_kw to max {MAX_POWER_INPUT_KW} kW.")
    
    resampled_df = resampled_df.reset_index() # Move timestamp back to column

    # Reorder columns for better readability (optional, but good practice)
    desired_column_order = [
        'timestamp', 'power_status', 
        'supply_temp', 'return_temp', 'delta_temp_raw', 'delta_temp_clamped',
        'brine_in_temp', 'brine_out_temp', 'outdoor_temp', 'desired_supply_temp', 
        'heat_output_kw', 'cop', 'power_input_kw'
    ]
    # Include only columns that actually exist in resampled_df to avoid errors
    final_columns = [col for col in desired_column_order if col in resampled_df.columns]
    # Add any other columns that might have been created but not in desired_order (e.g. if a register was added temporarily)
    for col in resampled_df.columns:
        if col not in final_columns:
            final_columns.append(col)
    resampled_df = resampled_df[final_columns]

    # --- Save to CSV ---
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'processed', 'villamichelin', 'Thermia')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename based on actual data range if possible
    if not resampled_df.empty and 'timestamp' in resampled_df.columns:
        min_date_str = resampled_df['timestamp'].min().strftime('%Y%m%d')
        max_date_str = resampled_df['timestamp'].max().strftime('%Y%m%d')
    else:
        min_date_str = (datetime.now() - timedelta(days=args.days)).strftime('%Y%m%d')
        max_date_str = datetime.now().strftime('%Y%m%d')
    
    output_file = os.path.join(output_dir, f"heat_pump_power_15min_{min_date_str}_to_{max_date_str}.csv")
    
    resampled_df.to_csv(output_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
    
    print(f"\nSaved 15-min interval data with power calculations to {output_file}")
    if not resampled_df.empty:
        print(f"Total records in final CSV: {len(resampled_df)}")
        print(f"Time period: {resampled_df['timestamp'].min()} to {resampled_df['timestamp'].max()}")
        print("\nFinal DataFrame columns:", resampled_df.columns.tolist())
        print("\nSample data (calculated):")
        print(resampled_df.head(10))
        if len(resampled_df) > 10:
            print("\nSample data (calculated, tail):")
            print(resampled_df.tail(10))
        
        if args.plot:
            print("\n--plot flag specified, generating plot...")
            plot_power_input(output_file)
    else:
        print("\nFinal dataset is empty.")
    
if __name__ == "__main__":
    main() 