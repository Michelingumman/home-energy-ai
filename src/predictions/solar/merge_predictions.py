import pandas as pd
from pathlib import Path
import glob
import os

def merge_predictions():
    # Get the data directory path
    data_dir = Path(__file__).parent / "actual_data/per_day" 
    
    # Get all CSV files in the directory
    csv_files = glob.glob(str(data_dir / "*.csv"))
    
    if not csv_files:
        print("No CSV files found in the data directory.")
        return
    
    # Create an empty list to store all dataframes
    all_dfs = []
    
    print(f"Found {len(csv_files)} CSV files to merge.")
    
    # Read and combine all CSV files
    for file in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(file)
            
            # Convert timestamp column to datetime if it exists
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            # Add the dataframe to our list
            all_dfs.append(df)
            print(f"Successfully read: {os.path.basename(file)}")
            
        except Exception as e:
            print(f"Error reading {os.path.basename(file)}: {str(e)}")
    
    if not all_dfs:
        print("No valid data to merge.")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs)
    
    # Sort by timestamp
    combined_df.sort_index(inplace=True)
    
    # Remove any duplicate timestamps
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    
    # Save the merged data
    output_file = data_dir / "merged_predictions.csv"
    combined_df.to_csv(output_file)
    
    print(f"\nMerged {len(all_dfs)} files into: {output_file}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")

if __name__ == "__main__":
    merge_predictions() 