import pandas as pd

def merge_price_files():
    """
    Merges SE3prices_new and SE3prices files, ensuring no duplicate entries
    and maintaining chronological order.
    """
    # Read both files
    df_new = pd.read_csv('data/raw/Elspotprices/nordpools priser 2020-2025/SE3prices_new.csv')
    df_existing = pd.read_csv('data/raw/Elspotprices/nordpools priser 2020-2025/SE3prices.csv')
    
    # Convert HourSE to datetime for both dataframes
    df_new['HourSE'] = pd.to_datetime(df_new['HourSE'])
    df_existing['HourSE'] = pd.to_datetime(df_existing['HourSE'])
    
    # Combine the dataframes and remove duplicates based on HourSE
    # Keep the entry from df_new if there's a duplicate
    combined_df = pd.concat([df_existing, df_new])
    combined_df = combined_df.drop_duplicates(subset='HourSE', keep='last')
    
    # Sort by datetime
    combined_df = combined_df.sort_values('HourSE')
    
    # Ensure all required columns are present and in correct order
    required_columns = [
        'HourSE', 'PriceArea', 'SE3_price_ore', 'price_24h_avg',
        'price_168h_avg', 'price_24h_std', 'hour_avg_price',
        'price_vs_hour_avg'
    ]
    combined_df = combined_df[required_columns]
    
    # Save the merged result
    output_path = 'data/raw/Elspotprices/nordpools priser 2020-2025/SE3prices_merged.csv'
    combined_df.to_csv(output_path, index=False)
    
    # Print some statistics about the merge
    print(f"Original files: {len(df_existing)} rows in SE3prices, {len(df_new)} rows in SE3prices_new")
    print(f"Merged file: {len(combined_df)} total rows")
    print(f"Merged file saved as {output_path}")

if __name__ == "__main__":
    merge_price_files()
