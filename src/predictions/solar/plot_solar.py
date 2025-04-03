import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import argparse
from pathlib import Path
import glob
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import sys
import calendar
import time


def load_single_csv(folder_path: Path) -> pd.DataFrame:
    """
    Loads the single relevant CSV file directly from a folder.
    Ignores any subdirectories named 'per_day'.
    Expects one CSV file in the root of the folder.
    Performs unit conversion if actual data (Watts) is detected.
    
    Args:
        folder_path: The Path object for the directory (e.g., actual_data or forecasted_data).
        
    Returns:
        A pandas DataFrame with the loaded data, timestamp index, and 'kilowatt_hours' column, or None.
    """
    if not folder_path.is_dir():
        print(f"Warning: Data folder not found: {folder_path}")
        return None

    # Find CSV files directly in the folder. glob without ** won't enter subdirs.
    csv_files = list(folder_path.glob('*.csv'))

    if len(csv_files) == 0:
        print(f"Info: No CSV file found directly in {folder_path}")
        return None
    elif len(csv_files) > 1:
        # Try to find a file with 'merged' in the name if multiple exist
        merged_files = [f for f in csv_files if 'merged' in f.name.lower()]
        if len(merged_files) == 1:
            print(f"Found multiple CSV files, using '{merged_files[0].name}' based on name.")
            target_file = merged_files[0]
        else:
            print(f"Warning: Found multiple CSV files directly in {folder_path}. Using the first one found: {csv_files[0].name}")
            target_file = csv_files[0]
    else:
        target_file = csv_files[0]

    print(f"Loading data from: {target_file}")

    try:
        df = pd.read_csv(target_file)

        # Robustly find timestamp and value columns
        df.columns = df.columns.str.strip()
        time_col = next((col for col in df.columns if 'timestamp' in col.lower() or 'time' in col.lower() or 'changed' in col.lower()), None)
        # Prioritize 'kilowatt_hours', then 'state', then others
        value_col_options = ['kilowatt_hours', 'kwh', 'state', 'watt_hours', 'value', 'effekt']
        value_col = None
        unit_is_watts = False
        for option in value_col_options:
            found_col = next((col for col in df.columns if option in col.lower()), None)
            if found_col:
                value_col = found_col
                # Check if the found column implies Watts (needs conversion)
                if option in ['state', 'watt_hours', 'effekt']:
                     # Exception: if 'watt_hours' is found but 'kilowatt_hours' also exists, prefer kwh
                     kwh_col_exists = any('kilowatt_hours' in col.lower() or 'kwh' in col.lower() for col in df.columns)
                     if option == 'watt_hours' and kwh_col_exists:
                         continue # Skip watt_hours if kwh exists
                     unit_is_watts = (option != 'watt_hours') # watt_hours column is usually already energy, 'state'/'effekt' are power (Watts)
                break # Stop searching once a primary candidate is found

        if not time_col or not value_col:
            print(f"Error: Could not find required time ('{time_col}') or value ('{value_col}') columns in {target_file}")
            return None
            
        print(f"Using columns: Time='{time_col}', Value='{value_col}' from {target_file.name}. Unit is Watts: {unit_is_watts}")

        # Convert and process
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        if pd.api.types.is_datetime64_any_dtype(df[time_col]) and df[time_col].dt.tz is not None:
            df[time_col] = df[time_col].dt.tz_localize(None)

        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')

        # --- Perform Unit Conversion if necessary ---
        if unit_is_watts:
            print(f"Converting column '{value_col}' from Watts to Kilowatt-hours (dividing by 1000).")
            # Ensure the column is numeric before division
            if pd.api.types.is_numeric_dtype(df[value_col]):
                 df['kilowatt_hours'] = df[value_col] / 1000.0
                 value_col = 'kilowatt_hours' # Update value_col to the new, correct column name
            else:
                 print(f"Warning: Cannot convert column '{value_col}' to kWh as it's not numeric.")
                 # Keep original value_col, potentially problematic later
                 df['kilowatt_hours'] = df[value_col] # Copy potentially non-numeric data
        elif value_col != 'kilowatt_hours':
             # Rename if the original column was already kWh but named differently (e.g., 'kwh')
             print(f"Renaming value column '{value_col}' to 'kilowatt_hours'")
             df.rename(columns={value_col: 'kilowatt_hours'}, inplace=True)
             value_col = 'kilowatt_hours'
        else:
             # Column is already 'kilowatt_hours', no action needed
             pass

        # Drop rows where time or the *final* value column conversion failed
        df = df.dropna(subset=[time_col, value_col])

        # Check if essential columns are present before setting index
        if time_col not in df.columns or value_col not in df.columns:
            print(f"Error: Essential columns ('{time_col}', '{value_col}') not found before setting index in {target_file.name}")
            return None

        df.set_index(time_col, inplace=True)
        df = df.sort_index()

        # Ensure the final 'kilowatt_hours' column exists after all operations
        if 'kilowatt_hours' not in df.columns:
             print(f"Error: Column 'kilowatt_hours' not found after processing in {target_file.name}")
             return None

        # Select only the essential column + add date
        df_processed = df[['kilowatt_hours']].copy()
        df_processed['date'] = df_processed.index.date

        # Handle potential duplicate timestamps (e.g., from merging)
        if df_processed.index.duplicated().any():
            print(f"Found duplicate timestamps in {target_file.name}, aggregating by taking the mean...")
            # Preserve dates associated with the index before grouping
            dates = df_processed['date'].copy()
            # Group by index and aggregate numeric columns (should just be kilowatt_hours)
            df_processed = df_processed.groupby(level=0).mean()
            # Reapply the date information. Use the first date found for duplicates.
            df_processed['date'] = dates.groupby(level=0).first()

        print(f"Successfully loaded {len(df_processed)} rows from {target_file.name}")
        return df_processed
        
    except Exception as e:
        print(f"Error loading or processing file {target_file}: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_data_from_folders(forecast_dir: Path, actual_dir: Path):
    """
    Loads prediction (forecast) and actual data from their respective folders,
    each expected to contain one primary CSV file.
    
    Args:
        forecast_dir: Path to the 'forecasted_data' directory.
        actual_dir: Path to the 'actual_data' directory.
        
    Returns:
        Dictionary {'predictions': df_pred|None, 'actuals': df_actual|None, 'dates': list_of_dates}.
        Returns None if neither prediction nor actual data could be loaded.
    """
    df_pred = load_single_csv(forecast_dir)
    df_actual = load_single_csv(actual_dir)

    if df_pred is None and df_actual is None:
        print("Error: Failed to load any data from forecast or actual folders.")
        return None
        
    # Determine the combined list of unique dates present in the loaded data
    pred_dates = set(df_pred['date']) if df_pred is not None else set()
    actual_dates = set(df_actual['date']) if df_actual is not None else set()
    combined_dates = sorted(list(pred_dates.union(actual_dates)))
    
    return {
        'predictions': df_pred,
        'actuals': df_actual,
        'dates': combined_dates
    }


def plot_hourly_data(data: dict, output_path: Path = None, show_plot: bool = True):
    """
    Plots hourly data comparing predictions against actuals.
    
    Args:
        data: Dictionary with 'predictions', 'actuals', and 'dates' keys
        output_path: Optional path to save the plot
        show_plot: Whether to display the plot interactively
    """
    predictions = data.get('predictions')
    actuals = data.get('actuals')

    if predictions is None and actuals is None:
        print("No data provided to plot_hourly_data.")
        return

    # Determine combined index and date range from available data
    combined_index = pd.Index([])
    if predictions is not None:
        combined_index = combined_index.union(predictions.index)
    if actuals is not None:
        combined_index = combined_index.union(actuals.index)

    if combined_index.empty:
        print("Data is empty, cannot plot.")
        return

    # Ensure the index is sorted before finding min/max
    combined_index = combined_index.sort_values()

    min_timestamp = combined_index.min().floor('h')
    max_timestamp = combined_index.max().ceil('h')
    # If min/max are the same, add an hour to max to create a valid range
    if min_timestamp == max_timestamp:
        max_timestamp += pd.Timedelta(hours=1)

    # Check if range is valid
    if min_timestamp >= max_timestamp:
        print(f"Warning: Invalid timestamp range generated ({min_timestamp} to {max_timestamp}). Skipping plot.")
        return

    hourly_index = pd.date_range(start=min_timestamp, end=max_timestamp, freq='h')

    # Reindex available data to the full hourly range, filling missing hours with 0
    hourly_predictions = None
    if predictions is not None:
         # Ensure no duplicates before reindexing
        if not predictions.index.is_unique:
            print("Warning: Duplicate timestamps found in predictions before reindexing, aggregating...")
            dates = pd.Series(predictions.index.date, index=predictions.index)
            numeric_cols = predictions.select_dtypes(include=['number']).columns
            predictions = predictions[numeric_cols].groupby(level=0).mean()
            predictions['date'] = dates.reindex(predictions.index)
        # Use 'kilowatt_hours' which should be the standardized name
        hourly_predictions = predictions.reindex(hourly_index).fillna({'kilowatt_hours': 0})
        hourly_predictions['date'] = hourly_predictions.index.date

    hourly_actuals = None
    if actuals is not None:
        # Ensure no duplicates before reindexing
        if not actuals.index.is_unique:
            print("Warning: Duplicate timestamps found in actuals before reindexing, aggregating...")
            dates = pd.Series(actuals.index.date, index=actuals.index)
            numeric_cols = actuals.select_dtypes(include=['number']).columns
            actuals = actuals[numeric_cols].groupby(level=0).mean()
            actuals['date'] = dates.reindex(actuals.index)
        # Use 'kilowatt_hours' which should be the standardized name
        hourly_actuals = actuals.reindex(hourly_index).fillna({'kilowatt_hours': 0})
        hourly_actuals['date'] = hourly_actuals.index.date

    
    # Set style for better visualization
    plt.style.use('ggplot')
    # Dynamically adjust figsize based on number of days (approx)
    num_days = (hourly_index.max() - hourly_index.min()).days + 1
    fig_width = max(15, num_days * 1.0) # Increase width more aggressively for more days
    fig_height = 9
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    bar_width = 0.4 # Adjust bar width for potential overlap
    bar_positions = np.arange(len(hourly_index))

    # Track whether we have predictions, actuals, or both for the legend
    has_predictions = hourly_predictions is not None
    has_actuals = hourly_actuals is not None
    
    pred_bars = None
    if has_predictions:
        # RESTORE: Color gradient for prediction bars
        pred_values = hourly_predictions['kilowatt_hours'].values
        # Create gradient color map based on predictions
        max_value_pred = max(pred_values.max(), 0.1) if len(pred_values) > 0 else 0.1
        norm = plt.Normalize(0, max_value_pred)
        colors = plt.cm.YlOrRd(norm(pred_values))
    
    pred_bars = ax.bar(
            bar_positions - bar_width/2 if has_actuals else bar_positions, # Offset if actuals exist
            pred_values,
            width=bar_width if has_actuals else bar_width * 1.5, # Wider if only predictions
        color=colors,
        edgecolor='#FF8C00',
            linewidth=0.8, # Thinner edge
            alpha=0.8,
        label='Predicted'
    )
    
    actual_bars = None
    if has_actuals:
        actual_values = hourly_actuals['kilowatt_hours'].values
        # Plot actual bars (or line)
        actual_bars = ax.bar(
            bar_positions + bar_width/2, # Offset from predictions
            actual_values,
            width=bar_width,
            # Use a solid color for actuals or hatched overlay
            color='lightblue',
            edgecolor='blue',
            linewidth=0.8, # Thinner edge
            # hatch='////', # Optional: use hatching instead of solid color
            label='Actual'
        )
        # Optional: Add connecting line for actual values
        ax.plot(
            bar_positions + bar_width/2, # Align with actual bars
            actual_values,
            color='blue',
            linewidth=1.5,
            alpha=0.6, # Slightly more transparent
            marker='.', # Smaller marker
            markersize=3,
            linestyle=':' # Dotted line
        )

    # Determine title based on data presence
    # Use the original dates from the data dictionary for title range
    available_dates = data.get('dates', [])
    if available_dates:
        start_display = min(available_dates).strftime("%b %d, %Y")
        end_display = max(available_dates).strftime("%b %d, %Y")
        if start_display == end_display:
            title_date = start_display
        else:
            title_date = f'{start_display} to {end_display}'
    else:
        # Fallback if dates list is somehow empty but we have data
        start_display = hourly_index[0].strftime("%b %d, %Y")
        end_display = hourly_index[-1].strftime("%b %d, %Y")
        if start_display == end_display:
            title_date = start_display
        else:
            title_date = f'{start_display} to {end_display}'

    plot_type = []
    if has_predictions: plot_type.append("Predicted")
    if has_actuals: plot_type.append("Actual")
    title_type = " vs ".join(plot_type) if len(plot_type) == 2 else (plot_type[0] if plot_type else "Data")

    # Calculate some simple totals for the title
    overall_pred_total = data['predictions']['kilowatt_hours'].sum() if data.get('predictions') is not None else 0
    overall_actual_total = data['actuals']['kilowatt_hours'].sum() if data.get('actuals') is not None else 0
    
    # Add totals to the title
    title = f'Solar Energy Production - {title_date}'
    subtitle = f'{title_type} (Hourly) | Total: P:{overall_pred_total:.1f} kWh, A:{overall_actual_total:.1f} kWh'
        
    # Customize plot
    ax.set_title(title + '\n' + subtitle, fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel('Date and Hour', fontsize=12, labelpad=15) # Increased labelpad
    ax.set_ylabel('Energy Production (kWh)', fontsize=12, labelpad=10)
    
    # --- X-axis Ticks and Labels ---
    date_separators = []
    date_labels = []
    day_start_indices = []
    hour_ticks = []  # All hour ticks
    hour_labels = []  # All hour labels
    key_hours = [6, 12, 18]  # Keep these as emphasized hours
    prev_date = None

    # Determine tick frequency based on number of days
    num_days_plot = (hourly_index.max() - hourly_index.min()).days + 1
    major_tick_freq = 1 if num_days_plot <= 10 else (2 if num_days_plot <= 20 else (7 if num_days_plot <= 60 else 14))
    
    current_day_count = 0
    day_start_hour_positions = {}  # Track start position of each day for hours
    
    for i, timestamp in enumerate(hourly_index):
        current_date = timestamp.date()
        current_hour = timestamp.hour
        
        # Major ticks and day separators
        if current_date != prev_date:
            current_day_count += 1
            if (current_day_count - 1) % major_tick_freq == 0:
                date_separators.append(i)
                # Show year if major tick frequency is high (e.g., weekly/bi-weekly)
                date_format = '%a\n%b %d' if major_tick_freq < 7 else '%b %d\n%Y'
                date_labels.append(current_date.strftime(date_format))
            day_start_indices.append(i)
            prev_date = current_date
            day_start_hour_positions[current_date] = i  # Record day start position

        # Add ALL hours as ticks
        hour_ticks.append(i)
        hour_labels.append(f"{current_hour:02d}:00")

    # Set major ticks for day starts (or less frequent)
    ax.set_xticks(date_separators)
    ax.set_xticklabels(date_labels, fontsize=11, fontweight='bold')

    # Set hour ticks for ALL hours
    ax.set_xticks(hour_ticks, minor=True)
    
    # For plots with many days, we need to be selective about which labels to show
    # to avoid extreme label density
    if num_days_plot <= 3:
        # For 1-3 days: show all hours
        ax.set_xticklabels(hour_labels, minor=True, fontsize=7, rotation=45, ha='right')
    elif num_days_plot <= 7:
        # For 4-7 days: show every 2 hours
        filtered_labels = []
        for i, label in enumerate(hour_labels):
            if int(label.split(':')[0]) % 2 == 0:  # Every even hour
                filtered_labels.append(label)
        else:
                filtered_labels.append('')  # Empty label for odd hours
        ax.set_xticklabels(filtered_labels, minor=True, fontsize=7, rotation=45, ha='right')
    else:
        # For more days: show every 3 or 4 hours depending on density
        hour_interval = 3 if num_days_plot <= 14 else 4
        filtered_labels = []
        for i, label in enumerate(hour_labels):
            if int(label.split(':')[0]) % hour_interval == 0:
                filtered_labels.append(label)
            else:
                filtered_labels.append('')  # Empty label for skipped hours
        ax.set_xticklabels(filtered_labels, minor=True, fontsize=7, rotation=45, ha='right')
    
    # Add vertical lines and prominent labels for key hours (6, 12, 18)
    for i, timestamp in enumerate(hourly_index):
        current_hour = timestamp.hour
        
        # Add vertical marker lines for all 6-hour intervals
        if current_hour % 6 == 0 and current_hour > 0:  # Skip midnight (already has day separator)
            ax.axvline(x=i, color='#AAAAAA', linestyle=':', linewidth=0.6, alpha=0.4, zorder=1)
            
            # For key hours, add more prominent labels with background
            if current_hour in key_hours and num_days_plot <= 14:
                # Place below the x-axis
                ax.text(
                    i, -0.02, 
                    f"{current_hour:02d}:00", 
                    transform=ax.get_xaxis_transform(),
                    ha='center',
                    va='top', 
                    fontsize=8,
                    fontweight='bold', 
                    color='#444444',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.9, ec='none')
                )
    
    # Adjust padding for overlapping labels
    ax.tick_params(axis='x', which='major', pad=25)  # More padding for day labels
    ax.tick_params(axis='x', which='minor', pad=7)  # Less padding for hour labels
    
    # Add alternating background shading for days
    for i, pos in enumerate(day_start_indices):
        if i < len(day_start_indices) - 1:
            end_pos = day_start_indices[i+1]
        else:
            end_pos = len(hourly_index)
            
        if i % 2 == 0:  # Every other day gets shaded
            ax.axvspan(pos - 0.5, end_pos - 0.5, color='#F5F5F5', alpha=0.5, zorder=0)

    # Add vertical lines for day separators with prominent styling
    for i, pos in enumerate(day_start_indices):
        if i > 0:
            ax.axvline(x=pos - 0.5, color='#444444', linestyle='-', linewidth=1.0, alpha=0.7, zorder=1)

    # Customize grid and background
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_facecolor('#FFFFFF')  # Use white background with shading instead
    fig.patch.set_facecolor('white')
    
    # Add legend
    if has_predictions or has_actuals:
        ax.legend(loc='upper right')
    
    # Adjust subplot to make room for hour labels - increase bottom margin
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.22)
    
    # Save or show the plot
    if output_path:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300)
            print(f"Saved hourly plot to {output_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        
    if show_plot:
        try:
            plt.show()
        except Exception as e:
            print(f"Error showing plot: {e}")
            # If showing fails, try to save as a fallback
            if not output_path:
                try:
                    temp_path = Path("solar_hourly_plot.png")
                    plt.savefig(temp_path)
                    print(f"Plot saved to {temp_path} as fallback")
                except:
                    pass
    plt.close(fig)  # Force close the figure


def plot_daily_summary(data, output_path=None, period_name="Period", show_plot: bool = True):
    """
    Plots daily summary data as bar charts.
    
    Args:
        data: Dictionary with data to plot
        output_path: Optional path to save the plot
        period_name: Name of the period for the title
        show_plot: Whether to display the plot interactively
    """
    predictions = data.get('predictions')
    actuals = data.get('actuals')
    available_dates = data.get('dates', [])

    if not available_dates:
        print("No dates with data found for summary plot.")
        return

    # Use available_dates to create the index for alignment
    summary_index = pd.DatetimeIndex(sorted(available_dates))

    daily_pred = None
    if predictions is not None:
        # Ensure date column exists
        if 'date' not in predictions.columns:
            predictions['date'] = predictions.index.date
        daily_pred = predictions.groupby('date')['kilowatt_hours'].sum().reindex(summary_index).fillna(0)

    daily_actual = None
    if actuals is not None:
        # Ensure date column exists
        if 'date' not in actuals.columns:
            actuals['date'] = actuals.index.date
        daily_actual = actuals.groupby('date')['kilowatt_hours'].sum().reindex(summary_index).fillna(0)


    # Set up the plot
    plt.style.use('ggplot')
    # Dynamically adjust figsize based on number of days
    num_days = len(summary_index)
    fig_width = max(10, num_days * 0.4) # Make wider for more days
    fig_height = 7
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    bar_width = 0.4 # Adjusted for potential side-by-side bars
    x_indices = np.arange(len(summary_index))

    has_predictions = daily_pred is not None
    has_actuals = daily_actual is not None

    # Plot prediction bars if available
    if has_predictions:
        pred_values = daily_pred.values
        # Handle case where all values are zero or negative
        max_pred_val = pred_values.max()
        norm = plt.Normalize(0, max(max_pred_val, 0.1)) if max_pred_val > 0 else plt.Normalize(0, 0.1)
        bar_colors = plt.cm.YlOrRd(norm(pred_values))

        ax.bar(
            x_indices - bar_width/2 if has_actuals else x_indices, # Offset if actuals exist
            pred_values,
            bar_width if has_actuals else bar_width * 1.5,
            color=bar_colors,
        edgecolor='#FF8C00',
            alpha=0.8,
        label='Predicted'
    )
    
    # Plot actual bars if available
    if has_actuals:
        actual_values = daily_actual.values
        ax.bar(
            x_indices + bar_width/2, # Offset from predictions
            actual_values,
            bar_width,
            color='lightblue',
            edgecolor='blue',
            label='Actual'
        )
        # Add value annotations for actuals
        for i, value in enumerate(actual_values):
            if value > 0.05:
                ax.annotate(
                    f'{value:.1f}',
                    xy=(x_indices[i] + bar_width/2, value),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=8,
        color='blue',
                    clip_on=True
                )

    # Add value annotations for predictions (if they exist)
    if has_predictions:
        pred_values = daily_pred.values
        for i, value in enumerate(pred_values):
            if value > 0.05:
                # Determine offset for prediction label based on actual bar presence
                pred_label_x = x_indices[i] - bar_width/2 if has_actuals else x_indices[i]
                ax.annotate(
                    f'{value:.1f}',
                    xy=(pred_label_x, value),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=8,
                    color='#A04000', # Darker color for prediction labels
                    clip_on=True
                )

    # Format date labels for x-axis
    # Show fewer labels if there are many days
    tick_indices = x_indices
    date_labels = [d.strftime('%a\n%b %d') for d in summary_index]
    rotation = 0
    ha = 'center'
    if num_days > 15:
        step = max(1, num_days // 15) # Show ~15 labels max
        tick_indices = x_indices[::step]
        date_labels = [summary_index[i].strftime('%b %d') for i in tick_indices]
        rotation = 45
        ha = 'right'
    elif num_days > 7:
        rotation = 30
        ha = 'right'

    ax.set_xticks(tick_indices)
    ax.set_xticklabels(date_labels, fontsize=9, rotation=rotation, ha=ha)
    
    # Set title and labels
    if available_dates:
        start_date_str = min(available_dates).strftime('%b %d, %Y')
        end_date_str = max(available_dates).strftime('%b %d, %Y')
        title_date = f'{start_date_str} to {end_date_str}'
    else:
        title_date = "Selected Period"
        title = f'Solar Energy Production - {title_date}'
        subtitle = 'Daily Totals (kWh)'
    
    ax.set_title(f'{title}\n{subtitle}', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Energy Production (kWh)', fontsize=12, labelpad=10)
    
    # Add grid and set color
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    # Add total in text box
    pred_total = daily_pred.sum() if daily_pred is not None else 0
    actual_total = daily_actual.sum() if daily_actual is not None else 0

    text = f'Total {period_name} Production:\n'
    if daily_pred is not None:
        # Check if prediction sum is valid number before formatting
        if pd.notna(pred_total):
            text += f'{pred_total:.1f} kWh (Predicted)\n'
        else:
            text += f'N/A kWh (Predicted)\n' # Or handle as appropriate
    if daily_actual is not None:
        # Check if actual sum is valid number
        if pd.notna(actual_total):
            text += f'{actual_total:.1f} kWh (Actual)\n'
        else:
            text += f'N/A kWh (Actual)\n'

    # Add percentage difference if both totals > 0 and are valid numbers
    if pd.notna(pred_total) and pd.notna(actual_total) and pred_total > 0:
        if actual_total >= 0:
            pct_diff = ((actual_total - pred_total) / pred_total) * 100
            text += f'{pct_diff:+.1f}% Diff (Actual vs Pred)' # Added sign
    elif actual_total > 0:
        text += "(No valid prediction total for diff)"

    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(
        0.02, 0.97, text.strip(),
        transform=ax.transAxes,
        fontsize=10, # Reduced font size
        verticalalignment='top',
        bbox=props
    )
    
    # Add legend if there's anything to label
    if has_predictions or has_actuals:
        ax.legend(loc='upper right')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0.03, 0.05, 0.98, 0.95]) # Add slight padding
    # Ensure output directory exists
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved daily summary plot to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_heatmap(data, output_path=None, period_name="Period", show_plot: bool = True):
    """
    Creates a heatmap visualization of the solar prediction data.
    
    Args:
        data: Dictionary with data to plot
        output_path: Optional path to save the plot
        period_name: Name of the period for the title
        show_plot: Whether to display the plot interactively
    """
    predictions = data.get('predictions')

    if predictions is None or predictions.empty:
        print("No prediction data available for heatmap.")
        return

    # Ensure 'date' column exists if coming directly from load
    if 'date' not in predictions.columns:
        predictions['date'] = predictions.index.date
    
    # Create pivoted dataframe with days as rows and hours as columns
    predictions['hour'] = predictions.index.hour
    # Use date_str for index to handle multi-month ranges correctly
    predictions['date_str'] = predictions.index.strftime('%Y-%m-%d') # Use full date for unique index
    
    # Pivot table with date on y-axis, hour on x-axis
    pivot_df = predictions.pivot_table(
        index='date_str', 
        columns='hour', 
        values='kilowatt_hours',
        aggfunc='mean' # Use mean for heatmap, assuming hourly data already
    )

    # Sort index chronologically
    pivot_df.index = pd.to_datetime(pivot_df.index)
    pivot_df = pivot_df.sort_index()

    # Fill potential NaN values resulting from pivot with 0
    pivot_df.fillna(0, inplace=True)

    # Reformat index back to string for display including day abbreviation
    pivot_df_display_index = pivot_df.index.strftime('%Y-%m-%d (%a)') # Format for display

    # Ensure all hours 0-23 are present as columns, filling missing with 0
    all_hours = range(24)
    pivot_df = pivot_df.reindex(columns=all_hours, fill_value=0)


    # Set up plot
    plt.style.use('ggplot')
    # Adjust figsize based on number of days
    num_days = len(pivot_df)
    fig_height = max(6, num_days * 0.25) # Adjust height dynamically
    fig_width = 14
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create the heatmap
    im = ax.imshow(pivot_df, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    
    # Configure axes
    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_yticks(np.arange(len(pivot_df)))
    
    # Label axes
    ax.set_xticklabels([f"{hour:02d}:00" for hour in pivot_df.columns], fontsize=8) # Pad hour
    ax.set_yticklabels(pivot_df_display_index, fontsize=8)

    # Rotate the x labels if needed (optional)
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add annotations for significant values
    max_val = pivot_df.values.max()
    # Only add text if value > 0 and number of days is reasonable
    add_text = num_days <= 31 # Avoid clutter for very long ranges
    if add_text and max_val > 0:
        threshold = max_val * 0.05 # Show values above 5% of max
    for i in range(len(pivot_df)):
        for j in range(len(pivot_df.columns)):
            value = pivot_df.iloc[i, j]
            if value > threshold:
                    # Adjust text color based on background intensity
                    text_color = 'white' if value > max_val * 0.6 else 'black'
                    ax.text(j, i, f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=7, clip_on=True)

    
    # Add a color bar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04) # Adjust size/padding
    cbar.ax.set_ylabel("Predicted Energy (kWh)", rotation=-90, va="bottom")
    
    # Set title and labels
    # Use available dates for title range
    available_dates = data.get('dates', [])
    if available_dates:
        start_date_str = min(available_dates).strftime('%b %d, %Y')
        end_date_str = max(available_dates).strftime('%b %d, %Y')
        title_date_range = f'{start_date_str} to {end_date_str}'
    else:
        title_date_range = "Selected Period"

    title = f'Solar Energy Production Heatmap - {title_date_range}'
    subtitle = f'Predicted Hourly Production (kWh)'
    
    ax.set_title(f"{title}\n{subtitle}", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Hour of Day', fontsize=12, labelpad=10)
    ax.set_ylabel('Date', fontsize=12, labelpad=10)
    
    # Add total in text box
    pred_total = predictions['kilowatt_hours'].sum()
    avg_daily = pred_total / num_days if num_days > 0 else 0
    text = f'Total Predicted: {pred_total:.1f} kWh\nAvg Daily Predicted: {avg_daily:.1f} kWh'
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    # Position text box carefully
    ax.text(
        0.98, 0.98, text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right', # Align to right
        bbox=props
    )
    
    # Save or show the plot
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap plot to {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    """
    Main function to load data from specified folders and run plotting functions.
    """
    # Get the script's directory
    script_dir = Path(__file__).parent
    
    # Try several potential locations for the data folders
    # Option 1: Direct subdirectories of the script directory
    forecast_dir_1 = script_dir / "forecasted_data"
    actual_dir_1 = script_dir / "actual_data"
    
    # Option 2: One level up from script directory
    forecast_dir_2 = script_dir.parent / "forecasted_data"
    actual_dir_2 = script_dir.parent / "actual_data"
    
    # Option 3: Using the 'data' directory for predictions and 'actual' for actuals in 'solar'
    forecast_dir_3 = script_dir / "data"
    actual_dir_3 = script_dir / "actual"
    
    # Option 4: Project root + src/predictions/data
    forecast_dir_4 = script_dir.parent.parent.parent / "src" / "predictions" / "forecasted_data"
    actual_dir_4 = script_dir.parent.parent.parent / "src" / "predictions" / "actual_data"

    parser = argparse.ArgumentParser(
        description="Plot solar forecast vs actual data for a specific date or date range.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Example Usage:
        python plot_solar.py 2025-03-15                   # Plot a single day
        python plot_solar.py 2025-03-01:2025-03-15        # Plot a date range
        python plot_solar.py --forecast_dir /path/to/data  # Specify custom folders'''
    )
    
    # Date selection as a positional argument
    parser.add_argument(
        "date_spec",
        nargs="?",
        help="Date specification: either a single date (YYYY-MM-DD) or a date range (YYYY-MM-DD:YYYY-MM-DD)",
        type=str
    )
    
    # Keep directory options in case needed
    parser.add_argument(
        "--forecast_dir",
        help="Directory containing forecast data CSV files",
        type=str
    )
    parser.add_argument(
        "--actual_dir",
        help="Directory containing actual data CSV files",
        type=str
    )

    args = parser.parse_args()
    
    # Parse date argument
    start_date = None
    end_date = None
    
    if args.date_spec:
        # Check if it's a range (contains a colon)
        if ":" in args.date_spec:
            # Fix: Split the string on the colon to get start and end dates
            date_parts = args.date_spec.split(":")
            if len(date_parts) != 2:
                print(f"Error: Invalid date range format '{args.date_spec}'. Use YYYY-MM-DD:YYYY-MM-DD.")
                return
                
            # Parse start date
            try:
                start_date = datetime.strptime(date_parts[0].strip(), "%Y-%m-%d").date()
            except ValueError:
                print(f"Error: Invalid start date '{date_parts[0]}'. Use YYYY-MM-DD format.")
                return
                
            # Parse end date
            try:
                end_date = datetime.strptime(date_parts[1].strip(), "%Y-%m-%d").date()
            except ValueError:
                print(f"Error: Invalid end date '{date_parts[1]}'. Use YYYY-MM-DD format.")
                return
        else:
            # Single date
            try:
                start_date = datetime.strptime(args.date_spec, "%Y-%m-%d").date()
                end_date = start_date
            except ValueError:
                print(f"Error: Invalid date format '{args.date_spec}'. Use YYYY-MM-DD format.")
                return
    
    # Ensure start date is before end date
    if start_date and end_date and start_date > end_date:
        start_date, end_date = end_date, start_date
        print(f"Note: Swapped dates to ensure correct order. Using {start_date} to {end_date}")
    
    # Print date range info if dates are provided
    if start_date and end_date:
        if start_date == end_date:
            print(f"Plotting data for {start_date}")
        else:
            print(f"Plotting data from {start_date} to {end_date}")
    
    # Always show plots, never save them
    show_plots = True
    
    # Determine forecast and actual data directories
    forecast_dir = None
    if args.forecast_dir:
        forecast_dir = Path(args.forecast_dir)
    else:
        # Try each option in order until one exists
        for option in [forecast_dir_1, forecast_dir_2, forecast_dir_3, forecast_dir_4]:
            if option.exists() and option.is_dir():
                forecast_dir = option
                break
    
    # Actual directory
    actual_dir = None
    if args.actual_dir:
        actual_dir = Path(args.actual_dir)
    else:
        # Try each option in order until one exists
        for option in [actual_dir_1, actual_dir_2, actual_dir_3, actual_dir_4]:
            if option.exists() and option.is_dir():
                actual_dir = option
                break
    
    # If directories weren't found, print helpful message and exit
    if not forecast_dir or not actual_dir:
        print("\nERROR: Could not find data directories. Please specify them with --forecast_dir and --actual_dir.")
        print("\nTried looking in these locations:")
        for i, (f_dir, a_dir) in enumerate(zip(
            [forecast_dir_1, forecast_dir_2, forecast_dir_3, forecast_dir_4],
            [actual_dir_1, actual_dir_2, actual_dir_3, actual_dir_4]
        )):
            print(f"Option {i+1}:")
            print(f"  Forecast dir: {f_dir.resolve()} {'✓' if f_dir.exists() else '✗'}")
            print(f"  Actual dir:   {a_dir.resolve()} {'✓' if a_dir.exists() else '✗'}")
        
        print("\nPlease either:")
        print("1. Create directories at one of the locations above, or")
        print("2. Run this script with --forecast_dir and --actual_dir pointing to your data folders")
        print("\nExample:")
        print(f"  python {Path(__file__).name} 2025-03-15 --forecast_dir /path/to/forecasted_data --actual_dir /path/to/actual_data")
        return
            
    # Load data using the new function
    print(f"Attempting to load data from {forecast_dir.resolve()} and {actual_dir.resolve()}")
    data = load_data_from_folders(forecast_dir, actual_dir)

    if data is None:
        print("Exiting: No data could be loaded.")
        return # Exit if loading failed completely

    available_dates = data.get('dates', [])
    if not available_dates:
        print("Exiting: Data loaded but contains no valid dates or timestamps.")
        return
    
    # Check if dataframes are empty after loading
    if data.get('predictions') is None and data.get('actuals') is None:
        print("Exiting: Both prediction and actual dataframes are empty after loading attempts.")
        return

    print(f"Data loaded. Date range: {min(available_dates)} to {max(available_dates)}. Found {len(available_dates)} unique days with data.")
    
    # Filter data by date if date range was specified
    if start_date and end_date:
        filtered_data = filter_data_by_date_range(data, start_date, end_date)
        if filtered_data.get('dates'):
            data = filtered_data
            print(f"Filtered to date range: {start_date} to {end_date}")
            print(f"Found {len(data['dates'])} days in the specified range.")
        else:
            print(f"Warning: No data found in the specified date range {start_date} to {end_date}.")
            return
            
    if data.get('predictions') is not None: print(f"- Forecast data points: {len(data['predictions'])}")
    if data.get('actuals') is not None: print(f"- Actual data points: {len(data['actuals'])}")

    # Generate plots with error handling and delays between plots
    print(f"Generating plots...")
    
    # Always show all types of plots, but with error handling
    try:
        plot_hourly_data(data, None, show_plot=True)
        # Add a short delay to ensure the plot completes
        time.sleep(1)
    except Exception as e:
        print(f"Error during hourly plot: {e}")
    
    try:
        plot_daily_summary(data, None, show_plot=True)
        # Add a short delay to ensure the plot completes
        time.sleep(1)
    except Exception as e:
        print(f"Error during daily summary plot: {e}")
    
    if 'predictions' in data and data['predictions'] is not None:
        try:
            plot_heatmap(data, None, show_plot=True)
        except Exception as e:
            print(f"Error during heatmap plot: {e}")
        
    print("Plotting complete.")


def filter_data_by_date_range(data, start_date, end_date):
    """
    Filter the data dictionary to include only entries within the specified date range.
    
    Args:
        data: Dictionary with 'predictions', 'actuals', and 'dates' keys
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        
    Returns:
        Filtered data dictionary
    """
    if not data:
        return None
        
    filtered_dates = [date for date in data.get('dates', []) 
                    if start_date <= date <= end_date]
    
    filtered_data = {'dates': filtered_dates}
    
    # Filter prediction data
    if data.get('predictions') is not None:
        mask = data['predictions']['date'].apply(lambda x: start_date <= x <= end_date)
        filtered_data['predictions'] = data['predictions'][mask].copy()
    else:
        filtered_data['predictions'] = None
        
    # Filter actual data
    if data.get('actuals') is not None:
        mask = data['actuals']['date'].apply(lambda x: start_date <= x <= end_date)
        filtered_data['actuals'] = data['actuals'][mask].copy()
    else:
        filtered_data['actuals'] = None
        
    return filtered_data


if __name__ == "__main__":
    main() 