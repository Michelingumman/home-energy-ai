import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from pathlib import Path
import sys
import calendar
import numpy as np


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
        
        return result_df
        
    except Exception as e:
        print(f"Error loading actual data: {e}")
        return None


def load_multiple_days(start_date: datetime.date, end_date: datetime.date, data_dir: Path, actual_dir: Path):
    """
    Load multiple days of prediction and actual data.
    
    Args:
        start_date: Start date for the range
        end_date: End date for the range
        data_dir: Directory containing prediction data
        actual_dir: Directory containing actual data
        
    Returns:
        Dictionary with prediction and actual data for the date range
    """
    current_date = start_date
    prediction_dfs = []
    actual_dfs = []
    available_dates = []
    
    while current_date <= end_date:
        # Check for prediction file
        pred_file = data_dir / f"{current_date.strftime('%Y%m%d')}.csv"
        
        if pred_file.exists():
            # Load prediction data
            df_pred = pd.read_csv(pred_file)
            df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'])
            df_pred.set_index('timestamp', inplace=True)
            df_pred['date'] = df_pred.index.date
            prediction_dfs.append(df_pred)
            available_dates.append(current_date)
            
            # Try to load actual data for this date
            actual_df = load_actual_data(current_date, actual_dir)
            if actual_df is not None:
                actual_df['date'] = actual_df.index.date
                actual_dfs.append(actual_df)
        
        current_date += datetime.timedelta(days=1)
    
    # Combine all data frames
    if not prediction_dfs:
        return None
        
    all_predictions = pd.concat(prediction_dfs)
    
    # Check for and handle duplicate indices
    if all_predictions.index.duplicated().any():
        print("Found duplicate timestamps in prediction data, aggregating...")
        # Save the date column before aggregation 
        dates = all_predictions.index.date
        
        # Only select numeric columns for aggregation
        numeric_cols = all_predictions.select_dtypes(include=['number']).columns
        
        # Group by index and take the mean of only numeric columns
        all_predictions = all_predictions[numeric_cols].groupby(level=0).mean()
        
        # Re-add the date column
        all_predictions['date'] = pd.Series(dates, index=all_predictions.index).values
    
    all_actuals = pd.concat(actual_dfs) if actual_dfs else None
    
    # Also check for duplicates in actual data
    if all_actuals is not None and all_actuals.index.duplicated().any():
        print("Found duplicate timestamps in actual data, aggregating...")
        # Save the date column before aggregation
        dates = all_actuals.index.date
        
        # Only select numeric columns for aggregation
        numeric_cols = all_actuals.select_dtypes(include=['number']).columns
        
        # Group by index and take the mean of only numeric columns
        all_actuals = all_actuals[numeric_cols].groupby(level=0).mean()
        
        # Re-add the date column
        all_actuals['date'] = pd.Series(dates, index=all_actuals.index).values
    
    return {
        'predictions': all_predictions,
        'actuals': all_actuals,
        'dates': available_dates
    }


def plot_daily_summary(data, output_path, period_name="Week"):
    """
    Create a daily summary plot showing total production for each day.
    
    Args:
        data: Dictionary containing prediction and actual data
        output_path: Path to save the plot
        period_name: Name of the period (Week/Month)
    """
    predictions = data['predictions']
    actuals = data['actuals']
    
    # Group by date and sum to get daily totals
    daily_pred = predictions.groupby('date')['kilowatt_hours'].sum()
    daily_pred_index = pd.DatetimeIndex(daily_pred.index)
    
    # Set up the plot
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define color palette
    bar_colors = plt.cm.YlOrRd(np.linspace(0.3, 0.8, len(daily_pred)))
    
    # Plot prediction bars
    bars = ax.bar(
        range(len(daily_pred)), 
        daily_pred.values,
        width=0.7,
        color=bar_colors,
        edgecolor='#FF8C00',
        alpha=0.7 if actuals is not None else 0.9,
        label='Predicted'
    )
    
    # Add actual data if available
    if actuals is not None:
        daily_actual = actuals.groupby('date')['kilowatt_hours'].sum()
        daily_actual_aligned = daily_actual.reindex(daily_pred.index)
        
        # Add actual data as line with markers
        actual_line = ax.plot(
            range(len(daily_pred)),
            daily_actual_aligned.values,
            marker='o',
            color='blue',
            linewidth=2,
            markersize=8,
            label='Actual'
        )
        
        # Add value annotations for actuals
        for i, value in enumerate(daily_actual_aligned.values):
            if not pd.isna(value):
                ax.annotate(
                    f'{value:.1f}',
                    xy=(i, value),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    color='blue'
                )
    
    # Format date labels for x-axis
    date_labels = [d.strftime('%a\n%b %d') for d in daily_pred_index]
    ax.set_xticks(range(len(daily_pred)))
    ax.set_xticklabels(date_labels)
    
    # Set title and labels
    start_date = daily_pred_index.min().strftime('%b %d, %Y')
    end_date = daily_pred_index.max().strftime('%b %d, %Y')
    title = f'Solar Energy Production - {start_date} to {end_date}'
    subtitle = 'Daily Totals (kWh)'
    
    ax.set_title(f'{title}\n{subtitle}', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Energy Production (kWh)', fontsize=12, labelpad=10)
    
    # Add grid and set color
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    # Add weekly/monthly total in text box
    pred_total = daily_pred.sum()
    text = f'Total {period_name} Production:\n{pred_total:.1f} kWh (Predicted)'
    
    if actuals is not None:
        actual_total = daily_actual.sum()
        text += f'\n{actual_total:.1f} kWh (Actual)'
        
        # Add percentage difference
        if actual_total > 0:
            pct_diff = ((actual_total - pred_total) / pred_total) * 100
            text += f'\n{pct_diff:.1f}% Difference'
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(
        0.02, 0.97, text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=props
    )
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_path}")
    
    plt.show()


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
    
    bar_width = 0.8 if has_actual_data else 0.6
    bar_positions = range(len(df))
    
    # Create bar plot for predictions with gradient colors
    norm = plt.Normalize(0, df['kilowatt_hours'].max())
    colors = plt.cm.YlOrRd(norm(df['kilowatt_hours'].values))
    
    pred_bars = ax.bar(
        bar_positions, 
        df['kilowatt_hours'],
        bar_width,
        color=colors,
        edgecolor='#FF8C00',
        linewidth=1,
        alpha=0.7 if has_actual_data else 0.9,
        label='Predicted'
    )
    
    # Add actual data visualization if available
    if has_actual_data:
        # Align actual data with prediction timestamps
        actual_aligned = actual_combined.reindex(df.index)
        
        # Add hatching pattern to actual bars to create overlay effect
        actual_bars = ax.bar(
            bar_positions,
            actual_aligned['kilowatt_hours'],
            bar_width,
            fill=False,
            hatch='////',
            edgecolor='blue',
            linewidth=1.5,
            label='Actual'
        )
        
        # Add connecting line for the actual values to highlight the trend
        ax.plot(
            bar_positions,
            actual_aligned['kilowatt_hours'],
            color='blue',
            linewidth=1.5,
            alpha=0.7,
            marker='o',
            markersize=4
        )
    
    # Get date range for title
    start_date = df.index[0].strftime("%Y-%m-%d")
    end_date = df.index[-1].strftime("%Y-%m-%d")
    
    # Format dates for display in title
    start_display = df.index[0].strftime("%b %d, %Y")
    end_display = df.index[-1].strftime("%b %d, %Y")
    
    # Create title with formatted date range
    if start_date == end_date:
        # For single day display
        title = f'Solar Energy Production - {start_display}'
    else:
        # For multi-day display
        title = f'Solar Energy Production - {start_display} to {end_display}'
    
    # Add subtitle with data types
    if has_actual_data:
        subtitle = 'Predicted vs Actual (Hourly)'
    else:
        subtitle = 'Predicted (Hourly)'
        
    # Customize plot
    ax.set_title(title + '\n' + subtitle, fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel('Date and Hour', fontsize=12, labelpad=10)
    ax.set_ylabel('Energy Production (kWh)', fontsize=12, labelpad=10)
    
    # Set x-axis ticks with dates
    x_labels = []
    tick_positions = []
    prev_date = None
    date_tick_positions = []
    date_labels = []
    
    for i, timestamp in enumerate(df.index):
        current_date = timestamp.date()
        
        # Add to tick positions
        tick_positions.append(i)
        
        # Track date changes for major ticks
        if current_date != prev_date:
            date_tick_positions.append(i)
            date_labels.append(current_date.strftime('%b %d\n%Y'))
            # If it's a new date, show date and time
            label = timestamp.strftime('%H:00')
            prev_date = current_date
        else:
            # If it's the same date, show only time
            label = timestamp.strftime('%H:00')
        
        x_labels.append(label)
    
    # Set regular ticks for hours
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    
    # Add prominent date markers
    date_text_height = -0.12  # Position below the regular x-axis labels
    
    # Add date separators and labels
    for i, (pos, date_label) in enumerate(zip(date_tick_positions, date_labels)):
        # Add vertical line for date change (except for the first one)
        if pos > 0:
            ax.axvline(x=pos-0.5, color='#888888', linestyle='--', alpha=0.5)
        
        # Add date label centered on the date's hours
        if i < len(date_tick_positions) - 1:
            # For dates except the last one
            midpoint = (pos + date_tick_positions[i+1] - 1) / 2
            width = date_tick_positions[i+1] - pos
        else:
            # For the last date
            midpoint = (pos + len(df) - 1) / 2
            width = len(df) - pos
        
        # Add a background color block for each date
        if i % 2 == 0:  # Alternate background colors
            rect = plt.Rectangle((pos-0.5, 0), width, -5, 
                              transform=ax.get_xaxis_transform(),
                              color='#f0f0f0', alpha=0.3, zorder=-1)
            ax.add_patch(rect)
        
        # Add date label
        ax.text(midpoint, date_text_height, date_label,
               ha='center', va='top', fontsize=10, fontweight='bold',
               transform=ax.get_xaxis_transform())
    
    # Add value labels on bars for non-zero values
    def add_value_labels(bars, values, y_offset=0, color=None):
        for i, bar in enumerate(bars):
            height = values[i]
            if height > 0.1:  # Only show labels for values > 0.1 kWh
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.05 + y_offset,  # Small offset to avoid overlap
                    f'{height:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color=color or 'black'
                )
    
    # Add labels for prediction bars
    pred_values = df['kilowatt_hours'].values
    add_value_labels(pred_bars, pred_values, y_offset=0, color='#BB5500')
    
    # Add labels for actual bars if available, with a slight vertical offset
    if has_actual_data:
        actual_values = actual_aligned['kilowatt_hours'].values
        if has_actual_data:
            # Find where actual > predicted to offset in the right direction
            offsets = [-0.2 if actual_values[i] <= pred_values[i] else 0.2 for i in range(len(actual_values))]
            for i, bar in enumerate(actual_bars):
                height = actual_values[i]
                if height > 0.1:  # Only show labels for values > 0.1 kWh
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + offsets[i],
                        f'{height:.1f}',
                        ha='center',
                        va='bottom' if offsets[i] > 0 else 'top',
                        fontsize=8,
                        color='blue'
                    )
    
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


def plot_weekly_detailed(data, output_path):
    """
    Create a detailed hourly plot for a week of data, similar to daily plot but spanning multiple days.
    
    Args:
        data: Dictionary containing prediction and actual data
        output_path: Path to save the plot
    """
    predictions = data['predictions']
    actuals = data['actuals']
    has_actual_data = actuals is not None
    
    # Set style for better visualization
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(18, 9))
    
    # Create a new index from the earliest to latest timestamp with hourly frequency
    min_timestamp = predictions.index.min().floor('h')
    max_timestamp = predictions.index.max().ceil('h')
    hourly_index = pd.date_range(start=min_timestamp, end=max_timestamp, freq='h')
    
    # Reindex predictions to ensure complete hourly data
    # First ensure there are no duplicate indices
    if predictions.index.duplicated().any():
        print("Handling duplicate timestamps before reindexing...")
        # Save the date column before aggregation
        dates = predictions.index.date
        
        # Only select numeric columns for aggregation
        numeric_cols = predictions.select_dtypes(include=['number']).columns
        
        # Group by index and take the mean of only numeric columns
        predictions = predictions[numeric_cols].groupby(level=0).mean()
        
        # Re-add the date column
        predictions['date'] = pd.Series(dates, index=predictions.index).values
    
    hourly_predictions = predictions.reindex(hourly_index)
    hourly_predictions['date'] = hourly_predictions.index.date
    
    # Create continuous bar positions
    bar_positions = range(len(hourly_index))
    bar_width = 0.8 if has_actual_data else 0.6
    
    # Create gradient color map
    non_na_values = hourly_predictions['kilowatt_hours'].dropna().values
    if len(non_na_values) > 0:
        max_value = max(non_na_values.max(), 0.1)  # Ensure non-zero
    else:
        max_value = 0.1
    norm = plt.Normalize(0, max_value)
    
    # Create colors with same size as data
    colors = plt.cm.YlOrRd(norm(hourly_predictions['kilowatt_hours'].fillna(0).values))
    
    # Plot prediction bars
    pred_bars = ax.bar(
        bar_positions,
        hourly_predictions['kilowatt_hours'].fillna(0),
        bar_width,
        color=colors,
        edgecolor='#FF8C00',
        linewidth=1,
        alpha=0.7 if has_actual_data else 0.9,
        label='Predicted'
    )
    
    # Add actual data visualization if available
    if has_actual_data:
        # Handle duplicates in actuals if any
        if actuals.index.duplicated().any():
            # Save the date column before aggregation
            dates = actuals.index.date
            
            # Only select numeric columns for aggregation
            numeric_cols = actuals.select_dtypes(include=['number']).columns
            
            # Group by index and take the mean of only numeric columns
            actuals = actuals[numeric_cols].groupby(level=0).mean()
            
            # Re-add the date column
            actuals['date'] = pd.Series(dates, index=actuals.index).values
            
        # Reindex actuals to match the hourly index
        hourly_actuals = actuals.reindex(hourly_index)
        
        # Add hatching pattern to actual bars to create overlay effect
        actual_bars = ax.bar(
            bar_positions,
            hourly_actuals['kilowatt_hours'].fillna(0),
            bar_width,
            fill=False,
            hatch='////',
            edgecolor='blue',
            linewidth=1.5,
            label='Actual'
        )
        
        # Add connecting line for the actual values to highlight the trend
        ax.plot(
            bar_positions,
            hourly_actuals['kilowatt_hours'].fillna(0),
            color='blue',
            linewidth=1.5,
            alpha=0.7,
            marker='o',
            markersize=3
        )
    
    # Create day separators and labels
    date_separators = []
    date_labels = []
    day_start_indices = []
    
    prev_date = None
    for i, timestamp in enumerate(hourly_index):
        current_date = timestamp.date()
        if current_date != prev_date:
            date_separators.append(i)
            date_labels.append(current_date.strftime('%a\n%b %d'))
            day_start_indices.append(i)
            prev_date = current_date
    
    # Add vertical lines for day separators
    for i, pos in enumerate(date_separators):
        if i > 0:  # Skip the first day's start
            ax.axvline(x=pos-0.5, color='#888888', linestyle='--', alpha=0.5, zorder=0)
    
    # Add day background colors
    for i in range(len(day_start_indices)):
        start_idx = day_start_indices[i]
        end_idx = day_start_indices[i+1] if i < len(day_start_indices)-1 else len(hourly_index)
        width = end_idx - start_idx
        
        # Alternate background colors for days
        if i % 2 == 0:
            rect = plt.Rectangle((start_idx-0.5, 0), width, -100, 
                              transform=ax.get_xaxis_transform(),
                              color='#f0f0f0', alpha=0.3, zorder=-1)
            ax.add_patch(rect)
    
    # Set x-axis ticks and labels
    # Create major ticks at midnight for each day
    major_ticks = date_separators
    ax.set_xticks(major_ticks)
    ax.set_xticklabels(date_labels, fontsize=10, fontweight='bold')
    
    # Create minor ticks for each 6 hours
    minor_ticks = []
    minor_labels = []
    
    for i, timestamp in enumerate(hourly_index):
        # Add labels for 00:00, 06:00, 12:00, 18:00
        if timestamp.hour in [0, 6, 12, 18]:
            minor_ticks.append(i)
            minor_labels.append(timestamp.strftime('%H:00'))
    
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_xticklabels(minor_labels, minor=True, fontsize=8)
    
    # Set title and labels
    start_date = hourly_index[0].strftime('%b %d, %Y')
    end_date = hourly_index[-1].strftime('%b %d, %Y')
    
    title = f'Solar Energy Production - {start_date} to {end_date}'
    subtitle = 'Hourly Production (kWh)'
    
    ax.set_title(f'{title}\n{subtitle}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Day and Hour', fontsize=12, labelpad=10)
    ax.set_ylabel('Energy Production (kWh)', fontsize=12, labelpad=10)
    
    # Grid lines
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Set background color
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Add total daily production in a table
    daily_pred = predictions.groupby('date')['kilowatt_hours'].sum()
    
    daily_totals_text = "Daily Totals (kWh):\n"
    for date, total in daily_pred.items():
        line = f"{date.strftime('%Y-%m-%d')}: {total:.1f}"
        if has_actual_data:
            # Get actual total for this date if available
            if date in actuals.index.date:
                actual_day = actuals[actuals.index.date == date]
                actual_total = actual_day['kilowatt_hours'].sum()
                line += f" (Actual: {actual_total:.1f})"
        daily_totals_text += line + "\n"
        
    # Add weekly total
    pred_total = daily_pred.sum()
    daily_totals_text += f"\nWeek Total: {pred_total:.1f} kWh"
    
    if has_actual_data:
        # If any actual data is available, calculate and show total
        actual_total = actuals['kilowatt_hours'].sum() if not actuals.empty else 0
        if actual_total > 0:
            daily_totals_text += f" (Actual: {actual_total:.1f} kWh)"
    
    # Place the text in a floating text box
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(
        0.02, 0.97, daily_totals_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=props
    )
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Weekly detailed plot saved as: {output_path}")
    
    plt.show()


def plot_heatmap(data, output_path, period_name="Month"):
    """
    Create a heatmap visualization for monthly data showing energy production by hour and day.
    
    Args:
        data: Dictionary containing prediction and actual data
        output_path: Path to save the plot
        period_name: Name of the period (Week/Month)
    """
    predictions = data['predictions']
    
    # Create pivoted dataframe with days as rows and hours as columns
    predictions['hour'] = predictions.index.hour
    predictions['day'] = predictions.index.day
    predictions['date_str'] = predictions.index.strftime('%b %d')
    
    # Pivot table with day on y-axis, hour on x-axis
    pivot_df = predictions.pivot_table(
        index='date_str', 
        columns='hour', 
        values='kilowatt_hours',
        aggfunc='sum'
    )
    
    # Sort by date
    pivot_df = pivot_df.reindex(
        pd.Series(predictions['date_str'].unique()).sort_values()
    )
    
    # Set up plot (just one figure)
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create the heatmap
    im = ax.imshow(pivot_df, cmap='YlOrRd', aspect='auto')
    
    # Configure axes
    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_yticks(np.arange(len(pivot_df)))
    
    # Label axes
    ax.set_xticklabels([f"{hour}:00" for hour in pivot_df.columns])
    ax.set_yticklabels(pivot_df.index)
    
    # Rotate the x labels
    plt.setp(ax.get_xticklabels(), rotation=0)
    
    # Add annotations for non-zero cells
    threshold = pivot_df.values.max() * 0.4  # Show values above 40% of max
    
    for i in range(len(pivot_df)):
        for j in range(len(pivot_df.columns)):
            value = pivot_df.iloc[i, j]
            if value > threshold:
                text_color = 'black' if value < pivot_df.values.max() * 0.7 else 'white'
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=8)
    
    # Add a color bar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Energy Production (kWh)", rotation=-90, va="bottom")
    
    # Set title and labels
    start_date = predictions.index.min().strftime('%b %d, %Y')
    end_date = predictions.index.max().strftime('%b %d, %Y')
    title = f'Solar Energy Production - {start_date} to {end_date}'
    subtitle = f'Hourly Production Heatmap (kWh)'
    
    ax.set_title(f"{title}\n{subtitle}", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Hour of Day', fontsize=12, labelpad=10)
    ax.set_ylabel('Date', fontsize=12, labelpad=10)
    
    # Add total in text box
    pred_total = predictions['kilowatt_hours'].sum()
    avg_daily = pred_total / len(pivot_df)
    text = f'Total {period_name} Production: {pred_total:.1f} kWh\nAverage Daily: {avg_daily:.1f} kWh'
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(
        0.02, 0.97, text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=props
    )
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved as: {output_path}")
    
    plt.show()


def main():
    try:
        # Get the data directory path
        data_dir = Path(__file__).parent / "data"
        actual_dir = Path(__file__).parent / "actual"
        images_dir = Path(__file__).parent / "images"
        
        # Create main images directory if it doesn't exist
        images_dir.mkdir(parents=True, exist_ok=True)
        
        if not data_dir.exists():
            print(f"Error: Data directory not found at {data_dir}")
            return
            
        # Get the command line argument
        if len(sys.argv) > 1:
            arg = sys.argv[1].lower()
            
            # Handle special commands
            if arg == "last_week":
                # Calculate date range for last week
                end_date = datetime.datetime.now().date()
                start_date = end_date - datetime.timedelta(days=6)
                
                print(f"Plotting data for last week: {start_date} to {end_date}")
                
                # Load data for the range
                data = load_multiple_days(start_date, end_date, data_dir, actual_dir)
                
                if data and len(data['dates']) > 0:
                    # Create date-based subdirectory for images
                    date_str = start_date.strftime('%Y%m%d')
                    date_images_dir = images_dir / date_str
                    date_images_dir.mkdir(exist_ok=True)
                    print(f"Saving images to: {date_images_dir}")
                    
                    # Create detailed hourly plot for the week (primary visualization)
                    hourly_path = date_images_dir / "last_week_hourly.png"
                    plot_weekly_detailed(data, hourly_path)
                    print(f"\nCreated detailed hourly visualization showing all hours for the week")
                    
                    # Also create weekly summary plot
                    output_path = date_images_dir / "last_week_summary.png"
                    plot_daily_summary(data, output_path, "Week")
                    print(f"Created summary bar chart with daily totals")
                    
                    # If more than 3 days are available, create a heatmap too
                    if len(data['dates']) > 3:
                        heatmap_path = date_images_dir / "last_week_heatmap.png"
                        plot_heatmap(data, heatmap_path, "Week")
                        print(f"Created heatmap visualization showing production patterns")
                else:
                    print("No data found for the specified date range.")
                
                return
                
            elif arg == "last_month":
                # Calculate date range for last month (30 days)
                end_date = datetime.datetime.now().date()
                start_date = end_date - datetime.timedelta(days=29)
                
                print(f"Plotting data for last month: {start_date} to {end_date}")
                
                # Load data for the range
                data = load_multiple_days(start_date, end_date, data_dir, actual_dir)
                
                if data and len(data['dates']) > 0:
                    # Create date-based subdirectory for images
                    date_str = start_date.strftime('%Y%m%d')
                    date_images_dir = images_dir / date_str
                    date_images_dir.mkdir(exist_ok=True)
                    print(f"Saving images to: {date_images_dir}")
                    
                    # Create monthly summary plot
                    output_path = date_images_dir / "last_month_summary.png"
                    plot_daily_summary(data, output_path, "Month")
                    print(f"Created summary bar chart with daily totals")
                    
                    # Create a heatmap for the month
                    heatmap_path = date_images_dir / "last_month_heatmap.png"
                    plot_heatmap(data, heatmap_path, "Month")
                    print(f"Created heatmap visualization showing production patterns")
                else:
                    print("No data found for the specified date range.")
                
                return
            
            # Regular date handling
            date_str = arg.replace('-', '')
            try:
                # Validate the date format
                datetime.datetime.strptime(date_str, "%Y%m%d")
            except ValueError:
                print(f"Error: Invalid date format or command. Please use:")
                print(f"  - YYYYMMDD or YYYY-MM-DD format for specific dates")
                print(f"  - 'last_week' to plot the last 7 days")
                print(f"  - 'last_month' to plot the last 30 days")
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
        
        # Create date-based subdirectory for images
        date_images_dir = images_dir / date_str
        date_images_dir.mkdir(exist_ok=True)
        print(f"Saving images to: {date_images_dir}")
        
        # Update the filename to be saved in the date subfolder
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Create images directory if it doesn't exist
        date_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to load actual data for each day in the prediction
        actual_dir = csv_file.parent.parent / "actual"
        
        # Call the plot function with updated path
        filename = f"{csv_file.stem}.png"
        # Update output path to use date subfolder
        output_path = date_images_dir / filename
        
        # Plot the predictions for the single day
        def plot_solar_prediction_with_path(csv_file, output_path):
            """Plot solar prediction and save to specific path"""
            plot_result = plot_solar_prediction(csv_file)
            # Move the file from default location to our date-specific folder if needed
            default_path = images_dir / filename
            if default_path.exists() and default_path != output_path:
                import shutil
                shutil.move(default_path, output_path)
                print(f"Moved plot to date-specific folder: {output_path}")
            return plot_result
            
        plot_solar_prediction_with_path(csv_file, output_path)
        
    except Exception as e:
        print(f"Error plotting solar prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 