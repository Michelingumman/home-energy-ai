import pandas as pd
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import sys


def load_actual_data(date: datetime.date, actual_dir: Path) -> pd.DataFrame:
    """
    Load actual solar production data from Home Assistant export and convert to hourly means.
    
    Args:
        date: The date to load data for
        actual_dir: Directory containing actual data files
        
    Returns:
        DataFrame with actual data or None if not found
    """
    try:
        # Construct the path for actual data file
        actual_file = actual_dir / f"{date.strftime('%Y%m%d')}.csv"
        
        if not actual_file.exists():
            return None
            
        # Read Home Assistant data
        df = pd.read_csv(actual_file)
        
        # Convert last_changed to datetime and state to float
        df['last_changed'] = pd.to_datetime(df['last_changed'])
        # Convert to timezone naive by removing timezone info
        df['last_changed'] = df['last_changed'].dt.tz_localize(None)
        df['state'] = pd.to_numeric(df['state'], errors='coerce')
        
        # Drop any rows where state couldn't be converted to number
        df = df.dropna(subset=['state'])
        
        # Set the timestamp as index
        df.set_index('last_changed', inplace=True)
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Calculate hourly means of the state values (in watts)
        hourly_means = df['state'].resample('h').mean()
        
        # Convert watts to kilowatt-hours (mean power * 1 hour / 1000)
        hourly_energy = hourly_means / 1000
        
        # Create DataFrame with hourly energy
        result_df = pd.DataFrame({'kilowatt_hours': hourly_energy})
        
        # Only keep data for the requested date
        start_time = pd.Timestamp(date)
        end_time = start_time + pd.Timedelta(days=1)
        result_df = result_df[start_time:end_time]
        
        print(f"Processed actual data for {date}:")
        print(f"Number of readings: {len(df)}")
        print(f"Time range: {df.index.min()} to {df.index.max()}")
        print(f"Number of hourly means: {len(result_df)}")
        
        return result_df
        
    except Exception as e:
        print(f"Error loading actual data: {e}")
        print(f"File: {actual_file}")
        if 'df' in locals():
            print("Available columns:", df.columns.tolist())
            if 'last_changed' in df.columns:
                print("Sample last_changed values:", df['last_changed'].head())
        return None


def plot_solar_prediction(csv_file: Path):
    """
    Plot solar prediction data from a CSV file.
    
    Args:
        csv_file: Path to the CSV file containing solar predictions
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Create images directory if it doesn't exist
    images_dir = csv_file.parent.parent / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Try to load actual data for each day in the prediction
    actual_dir = csv_file.parent.parent / "actual"
    has_actual_data = False
    actual_data = []
    
    # Convert index.date to pandas Series before calling unique()
    unique_dates = pd.Series(df.index.date).unique()
    for date in unique_dates:
        actual_df = load_actual_data(date, actual_dir)
        if actual_df is not None:
            has_actual_data = True
            actual_data.append(actual_df)
    
    if has_actual_data:
        actual_combined = pd.concat(actual_data)
    
    # Set style for better visualization
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(15, 8))
    
    bar_width = 0.35  # Width of the bars
    bar_positions = range(len(df))
    
    # Create bar plot for predictions with gradient colors
    norm = plt.Normalize(0, df['kilowatt_hours'].max())
    colors = plt.cm.YlOrRd(norm(df['kilowatt_hours'].values))
    pred_bars = ax.bar(
        [x - bar_width/2 for x in bar_positions], 
        df['kilowatt_hours'],
        bar_width,
        color=colors,
        edgecolor='#FF8C00',
        linewidth=1,
        alpha=0.8,
        label='Predicted'
    )
    
    # Add actual data bars if available
    if has_actual_data:
        # Align actual data with prediction timestamps
        actual_aligned = actual_combined.reindex(df.index)
        actual_bars = ax.bar(
            [x + bar_width/2 for x in bar_positions],
            actual_aligned['kilowatt_hours'],
            bar_width,
            color='skyblue',
            edgecolor='blue',
            linewidth=1,
            alpha=0.8,
            label='Actual'
        )
    
    # Get date range for title
    start_date = df.index[0].strftime("%Y-%m-%d")
    end_date = df.index[-1].strftime("%Y-%m-%d")
    
    # Customize plot
    title = f'Predicted vs Actual Solar Energy Production by Hour\n{start_date} to {end_date}'
    ax.set_title(title, fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel('Date and Hour', fontsize=12, labelpad=10)
    ax.set_ylabel('Energy Production (kWh)', fontsize=12, labelpad=10)
    
    # Set x-axis ticks with dates
    x_labels = []
    prev_date = None
    for timestamp in df.index:
        current_date = timestamp.date()
        if current_date != prev_date:
            # If it's a new date, show date and time
            label = timestamp.strftime('%Y-%m-%d\n%H:00')
            prev_date = current_date
        else:
            # If it's the same date, show only time
            label = timestamp.strftime('%H:00')
        x_labels.append(label)
    
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    # Add value labels on bars for non-zero values
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0.1:  # Only show labels for values > 0.1 kWh
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )
    
    add_value_labels(pred_bars)
    if has_actual_data:
        add_value_labels(actual_bars)
    
    # Customize grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)  # Put grid behind bars
    
    # Add light background color
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Add total daily production
    df['date'] = df.index.date
    daily_totals = df.groupby('date')['kilowatt_hours'].sum()
    
    daily_totals_text = "Daily Totals (kWh):\n"
    for date, total in daily_totals.items():
        line = f"{date}: {total:.1f}"
        if has_actual_data and date in actual_combined.index.date:
            actual_total = actual_combined[actual_combined.index.date == date]['kilowatt_hours'].sum()
            line += f" (Actual: {actual_total:.1f})"
        daily_totals_text += line + "\n"
    
    plt.figtext(1.02, 0.5, daily_totals_text, fontsize=10, va='center')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the plot with the same date as the CSV
    filename = f"{csv_file.stem}.png"
    plt.savefig(images_dir / filename, dpi=300, bbox_inches='tight', bbox_extra_artists=[plt.figtext(0, 0, '')], pad_inches=0.5)
    print(f"\nPlot saved as: {filename}")
    
    plt.show()


def main():
    try:
        # Get the data directory path
        data_dir = Path(__file__).parent / "data"
        
        if not data_dir.exists():
            print(f"Error: Data directory not found at {data_dir}")
            return
            
        # Get the date from command line argument or use today
        if len(sys.argv) > 1:
            # Accept either YYYYMMDD or YYYY-MM-DD format
            date_str = sys.argv[1].replace('-', '')
            try:
                # Validate the date format
                datetime.datetime.strptime(date_str, "%Y%m%d")
            except ValueError:
                print(f"Error: Invalid date format. Please use YYYYMMDD or YYYY-MM-DD format.")
                print(f"Example: python plot_solar.py 20250224 or python plot_solar.py 2025-02-24")
                return
        else:
            # Use today's date
            date_str = datetime.datetime.now().strftime("%Y%m%d")
            
        # Construct the CSV file path
        csv_file = data_dir / f"{date_str}.csv"
        
        if not csv_file.exists():
            print(f"Error: No prediction file found for date {date_str}")
            print(f"Please run prediction.py first to generate predictions:")
            print(f"python prediction.py {date_str}")
            return
            
        print(f"Processing prediction file for date: {date_str}")
        
        # Plot the predictions
        plot_solar_prediction(csv_file)
        
    except Exception as e:
        print(f"Error plotting solar prediction: {e}")


if __name__ == "__main__":
    main() 