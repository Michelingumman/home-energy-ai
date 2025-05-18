import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys
import datetime
import random
from calendar import monthrange

# Add the project root directory (one level up from 'src') to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.rl import config


def plot_agent_performance(episode_data, model_name=None, save_dir=None):
    """
    Generate a comprehensive plot of agent's performance during an episode.
    
    Args:
        episode_data: A list of episode timestep dictionaries
        model_name: Optional name of the model for the plot title
        save_dir: Optional directory to save the plot
    """
    # Extract data
    timestamps = [data['timestamp'] for data in episode_data]
    rewards = [data['reward'] for data in episode_data]
    soc_values = [data['soc'][0] for data in episode_data]
    prices = [data['current_price'] for data in episode_data]
    battery_powers = [data.get('power_kw', 0) for data in episode_data]
    grid_powers = [data.get('grid_power_kw', 0) for data in episode_data]
    base_demands = [data.get('base_demand_kw', 0) for data in episode_data]
    solar_productions = [data.get('current_solar_production_kw', 0) for data in episode_data]
    night_discounts = [data.get('is_night_discount', False) for data in episode_data]
    actions = [data.get('action', 0) for data in episode_data]
    
    # New metrics
    price_24h_avgs = [data.get('price_24h_avg', None) for data in episode_data]
    price_168h_avgs = [data.get('price_168h_avg', None) for data in episode_data]
    
    # If price averages weren't captured, filter out None values
    if all(avg is None for avg in price_24h_avgs):
        price_24h_avgs = None
    else:
        price_24h_avgs = [avg if avg is not None else 0 for avg in price_24h_avgs]
        
    if all(avg is None for avg in price_168h_avgs):
        price_168h_avgs = None
    else:
        price_168h_avgs = [avg if avg is not None else 0 for avg in price_168h_avgs]
    
    # Get config for night discount factor
    from src.rl.config import get_config_dict
    config = get_config_dict()
    night_capacity_discount = config.get('night_capacity_discount', 0.5)  # Default to 0.5 if not found
    
    # Calculate discounted grid power for night hours
    discounted_grid_powers = []
    for i, power in enumerate(grid_powers):
        if night_discounts[i] and power > 0:  # Only apply discount to positive grid power (imports)
            discounted_grid_powers.append(power * night_capacity_discount)
        else:
            discounted_grid_powers.append(power)
    
    # Calculate cumulative reward
    cum_rewards = np.cumsum(rewards)
    
    # --- Start of HOURLY RESAMPLING for PLOTTING specific series ---
    hourly_timestamps = []
    grid_powers_hourly = []
    discounted_grid_powers_hourly = []
    base_demands_hourly = []
    solar_productions_hourly = []

    if timestamps: # Ensure there is data to process
        # Create a DataFrame with original 15-minute data for resampling
        plot_data_df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps), # Ensure timestamps are datetime objects for indexing
            'grid_power_kw': grid_powers,
            'discounted_grid_power_kw': discounted_grid_powers, # This list is already available
            'base_demand_kw': base_demands,
            'current_solar_production_kw': solar_productions
        }).set_index('timestamp')

        # Resample to hourly mean, fill NaN with 0 (e.g., if an hour has no 15-min data points)
        hourly_resampled_data = plot_data_df.resample('h').mean().fillna(0)
        
        hourly_timestamps = hourly_resampled_data.index.to_list()
        grid_powers_hourly = hourly_resampled_data['grid_power_kw'].tolist()
        discounted_grid_powers_hourly = hourly_resampled_data['discounted_grid_power_kw'].tolist()
        base_demands_hourly = hourly_resampled_data['base_demand_kw'].tolist()
        solar_productions_hourly = hourly_resampled_data['current_solar_production_kw'].tolist()
    # --- End of HOURLY RESAMPLING for PLOTTING ---
    
    # Create figure with subplots
    fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1, 1]})
    
    # Set title based on model name if provided
    if model_name:
        fig.suptitle(f"Agent Performance Analysis - Model: {model_name}", fontsize=10)
    else:
        fig.suptitle("Agent Performance Analysis", fontsize=10)
    
    # Subplot 1: Battery SoC and Electricity Price
    ax1 = axs[0]
    ax1.set_title("Battery SoC and Electricity Price")
    ax1.set_ylabel("SoC")
    
    # Plot SoC
    ax1.step(timestamps, soc_values, 'r-', label="Battery SoC", linewidth=1)
    ax1.set_ylim(0, 1.05)
    
    # Plot Price on secondary y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Electricity Price (öre/kWh)")
    
    # Plot prices
    ax2.step(timestamps, prices, 'b-', label="Electricity Price",linewidth=1)
    
    # # Plot price averages if available
    # if price_24h_avgs:
    #     ax2.step(timestamps, price_24h_avgs, 'g-.', alpha=0.7, label="24h Avg Price",linewidth=1)
    # if price_168h_avgs:
    #     ax2.step(timestamps, price_168h_avgs, 'm-.', alpha=0.7, label="168h Avg Price",linewidth=1)
    
    # Create combined legend for both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    
    # Subplot 2: Agent Actions and Battery Power
    ax3 = axs[1]
    ax3.set_title("Agent Actions and Battery Power (Battery: Positive Power = Discharging, Negative = Charging)")
    ax3.set_ylabel("Action (-1 to 1)")
    ax3.step(timestamps, actions, 'g-', label="Agent Action (Normalized)",linewidth=1)
    ax3.set_ylim(-1.1, 1.1)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Plot Battery Power on secondary y-axis
    ax4 = ax3.twinx()
    ax4.set_ylabel("Battery Power (kW)")
    # Note that in our convention, negative battery power means charging
    ax4.step(timestamps, battery_powers, 'm-', label="Battery Power (kW)",linewidth=1)
    # Create combined legend for both axes
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    lines_4, labels_4 = ax4.get_legend_handles_labels()
    ax3.legend(lines_3 + lines_4, labels_3 + labels_4, loc='upper right')
    
    # --- Start of new calculation for hourly peaks ---
    # Create DataFrame for hourly resampling
    df_episode = pd.DataFrame(episode_data)
    if not df_episode.empty:
        df_episode['timestamp_col'] = pd.to_datetime(df_episode['timestamp']) # Keep original timestamp column for joining
        df_episode_indexed = df_episode.set_index('timestamp_col')

        # Calculate hourly mean grid power, considering only positive values for peaks
        hourly_mean_grid_power = df_episode_indexed['grid_power_kw'].clip(lower=0).resample('h').mean().fillna(0)
        hourly_mean_grid_power_sorted = hourly_mean_grid_power.sort_index()

        # Initialize lists for plotting hourly-based peaks, corresponding to original 15-min steps
        peak1_values_hourly = []
        peak2_values_hourly = []
        peak3_values_hourly = []
        rolling_avg_hourly_values = []

        for current_step_original_timestamp_obj in timestamps: # Iterate using the original 15-min timestamps
            current_step_timestamp = pd.to_datetime(current_step_original_timestamp_obj)
            current_month_start = current_step_timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            # Filter hourly_mean_grid_power for current month and up to current_step_timestamp's hour
            relevant_hourly_means_series = hourly_mean_grid_power_sorted[
                (hourly_mean_grid_power_sorted.index >= current_month_start) &
                (hourly_mean_grid_power_sorted.index <= current_step_timestamp.replace(minute=0,second=0,microsecond=0))
            ]
            
            positive_hourly_means = relevant_hourly_means_series[relevant_hourly_means_series > 0].tolist()
            
            top_3_for_this_step = sorted(positive_hourly_means, reverse=True)[:3]
            while len(top_3_for_this_step) < 3:
                top_3_for_this_step.append(0.0)
            
            peak1_values_hourly.append(top_3_for_this_step[0])
            peak2_values_hourly.append(top_3_for_this_step[1])
            peak3_values_hourly.append(top_3_for_this_step[2])
            rolling_avg_hourly_values.append(sum(top_3_for_this_step) / 3.0 if top_3_for_this_step else 0.0)
    else: # Handle empty episode_data
        peak1_values_hourly = []
        peak2_values_hourly = []
        peak3_values_hourly = []
        rolling_avg_hourly_values = []
        
    # --- End of new calculation for hourly peaks ---

    # Subplot 3: Household Consumption, Grid Power, and Solar Production
    ax5 = axs[2]
    ax5.set_title("Household Consumption, Net Grid Power, and Solar Production (Hourly)")
    ax5.set_ylabel("Power (kW)")
    ax5.step(hourly_timestamps, base_demands_hourly, 'c-', label="Household Consumption (kW)",linewidth=1)
    ax5.step(hourly_timestamps, grid_powers_hourly, 'r-', label="Net Grid Power (kW)",linewidth=1)
    ax5.step(hourly_timestamps, solar_productions_hourly, 'orange', label="Solar Production (kW)",linewidth=1)
    
    ax5.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax5.legend(loc='upper right')
    
    # Subplot 4: Grid Peaks and Capacity Fee with Night Discount Highlight
    ax7 = axs[3]
    ax7.set_ylabel("Power (kW)")
    
    # Add grid power for comparison (hourly)
    ax7.step(hourly_timestamps, grid_powers_hourly, 'black', alpha=0.3, label="Current Grid Power (Hourly)",linewidth=1)
    
    # Add discounted grid power (hourly)
    ax7.step(hourly_timestamps, discounted_grid_powers_hourly, 'purple', alpha=0.6,linewidth=1, label=f"Discounted Grid Power (Hourly, Night Factor: {config.get('night_capacity_discount',0.5)})")
    
    # Find the 3 highest HOURLY peak events from discounted_grid_powers_hourly and mark them
    monthly_capacity_cost_from_env = 0.0 # Default value
    if episode_data:
        last_step_data = episode_data[-1]
        # This is the capacity fee in SEK calculated by the environment based on hourly peaks
        monthly_capacity_cost_from_env = last_step_data.get('current_capacity_fee', 0.0)

    if hourly_timestamps and discounted_grid_powers_hourly:
        # Create a list of (timestamp, discounted_hourly_power) tuples
        hourly_power_events = list(zip(hourly_timestamps, discounted_grid_powers_hourly))

        # Sort by power value in descending order, considering only positive peaks
        sorted_hourly_power_events = sorted(
            [event for event in hourly_power_events if event[1] > 0], 
            key=lambda x: x[1], 
            reverse=True
        )
        
        top_hourly_peak_events = sorted_hourly_power_events[:3]
        
        peak_colors = ['red', 'green', 'blue']
        
        # Calculate capacity cost based on the peaks identified for plotting
        plot_derived_capacity_cost = 0.0
        if top_hourly_peak_events:
            # Ensure we have 3 peak values for averaging, using only positive peaks shown.
            # The top_hourly_peak_events already contains up to 3 positive peaks.
            peak_values_for_avg = [p_val for _, p_val in top_hourly_peak_events]
            while len(peak_values_for_avg) < 3:
                peak_values_for_avg.append(0.0) # Pad if less than 3 peaks found
            
            if peak_values_for_avg: # Should always be true if top_hourly_peak_events was populated
                average_of_plotted_peaks = sum(peak_values_for_avg) / len(peak_values_for_avg) if peak_values_for_avg else 0.0
                capacity_fee_rate = config.get('capacity_fee_sek_per_kw', 81.25) # config is loaded at func start
                plot_derived_capacity_cost = round(average_of_plotted_peaks * capacity_fee_rate, 2)

            for i, (peak_ts, peak_val) in enumerate(top_hourly_peak_events):
                color = peak_colors[i]
                ax7.axvline(x=peak_ts, color=color, linestyle='-', linewidth=1, alpha=0.7)
                
                ax7.annotate(f'Hourly Peak {i+1}: {peak_val:.2f} kW', 
                            xy=(peak_ts, peak_val),
                            xytext=(10, 10 + i*20),
                            textcoords='offset points',
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color=color),
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                            color=color,
                            fontsize=6)

    ax7.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Add a legend with better placement and formatting
    ax7.legend(loc='upper right', fontsize=6)    
    # Use the capacity cost derived from the plotted hourly peaks for the title
    ax7.set_title(f"Grid Power Peaks (Hourly Discounted). Capacity Cost: {plot_derived_capacity_cost:.2f} kr")

    
    # Subplot 5: Cumulative Reward
    ax6 = axs[4]
    ax6.set_title("Cumulative Reward Over Episode")
    ax6.set_ylabel("Reward")
    ax6.step(timestamps, cum_rewards, 'brown', label="Cumulative Reward", linewidth=1)
    ax6.legend(loc='upper left', fontsize=6)
    
    # Add shaded regions for night discount periods to ALL subplots
    night_periods = []
    in_night_period = False
    start_idx = None
    
    for i, is_night in enumerate(night_discounts):
        if is_night and not in_night_period:
            in_night_period = True
            start_idx = i
        elif not is_night and in_night_period:
            in_night_period = False
            night_periods.append((start_idx, i))
    
    # Add the last period if it ends with night time
    if in_night_period:
        night_periods.append((start_idx, len(night_discounts)-1))
    
    # Draw shaded regions for night discount periods on all subplots
    if night_periods:
        # Add night discount shade to each subplot
        for ax in axs:
            for i, (start, end) in enumerate(night_periods):
                if start < len(timestamps) and end < len(timestamps):
                    # No label needed here to avoid cluttering legends
                    ax.axvspan(timestamps[start], timestamps[end], alpha=0.15, color='grey', zorder=0)  # zorder=0 ensures it's drawn behind other elements
    
    # Recreate legend for subplot 1 to include night discount
    lines_1, labels_1 = axs[0].get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    # Add night discount to the legend
    import matplotlib.patches as mpatches
    night_patch = mpatches.Patch(color='grey', alpha=0.15, label='Night (22-06) Discount')
    axs[0].legend(lines_1 + lines_2 + [night_patch], labels_1 + labels_2 + ['Night (22-06) Discount'], loc='upper right', fontsize=6)
    
    # Improved x-axis formatting for all subplots
    # Import required formatter if not already at the top
    import matplotlib.dates as mdates
    
    # Get episode duration to determine appropriate tick spacing
    episode_duration = max(timestamps) - min(timestamps)
    days_in_episode = episode_duration.days + episode_duration.seconds / 86400
    
    # Format the x-axis with better date/time display based on episode length
    for ax in axs:
        if days_in_episode <= 7:  # Short episode (up to a week)
            # Show date every day
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            # Show hours only at noon
            ax.xaxis.set_minor_formatter(mdates.DateFormatter('%h'))
            ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[12]))
        elif days_in_episode <= 14:  # Medium episode (up to two weeks)
            # Show date every two days
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            # No minor ticks for hours
            ax.xaxis.set_minor_locator(plt.NullLocator())
        else:  # Long episode (more than two weeks)
            # Show date every three days
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            # No minor ticks for hours
            ax.xaxis.set_minor_locator(plt.NullLocator())
        
        # Rotate date labels for better readability and prevent overlap
        ax.tick_params(which='major', axis='x', rotation=20, labelsize=10)
        ax.tick_params(which='minor', axis='x', rotation=0, labelsize=8)
        
        # Add grid lines
        ax.grid(True, which='major', axis='x', linestyle='-', alpha=0.7)
        ax.grid(True, which='minor', axis='x', linestyle=':', alpha=0.4)
    
    # Add a clear x-axis label for the bottom subplot
    axs[4].set_xlabel("Date/Time", fontsize=6)
    
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save figure if a directory is specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"agent_performance_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    
    return fig, axs


def plot_reward_components(episode_data, model_name=None, save_dir=None):
    """
    Create a plot showing the different reward components over time, grouped into logical subplots.
    
    Args:
        episode_data: List of dictionaries containing episode data
        model_name: Optional name of the model for the plot title
        save_dir: Optional directory to save the plot
    """
    # Extract data
    timestamps = [data['timestamp'] for data in episode_data]
    rewards = [data['reward'] for data in episode_data]
    
    # Identify all reward components dynamically
    # We'll identify them from the first entry that has all components
    reward_components = {}
    
    # Find all potential reward component keys
    component_keys = []
    for data in episode_data:
        for key in data.keys():
            if key.startswith('reward_') and key not in component_keys:
                component_keys.append(key)
    
    # Extract data for each component
    for key in component_keys:
        reward_components[key] = [data.get(key, 0) for data in episode_data]
    
    # Calculate cumulative total reward
    cum_rewards = np.cumsum(rewards)
    
    # Group components by category
    cost_components = ['reward_grid_cost', 'reward_battery_cost', 'reward_capacity_fee']
    battery_components = ['reward_soc_limit_penalty', 'reward_preferred_soc']
    grid_components = ['reward_peak_penalty', 'reward_arbitrage_bonus', 'reward_export_bonus']
    
    # Colors for different components
    colors = {
        'reward_grid_cost': 'red',
        'reward_battery_cost': 'green',
        'reward_capacity_fee': 'purple',
        'reward_soc_limit_penalty': 'blue',
        'reward_preferred_soc': 'forestgreen',
        'reward_peak_penalty': 'darkgreen',
        'reward_arbitrage_bonus': 'orange',
        'reward_export_bonus': 'deepskyblue'
    }
    
    # Create figure with subplots that share the same x-axis
    fig, axs = plt.subplots(4, 1, figsize=(15, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1]})
    plt.subplots_adjust(hspace=0.3)  # Add spacing between subplots
    
    # Set title
    if model_name:
        plt.suptitle(f"Reward Components - Model: {model_name}", fontsize=10)
    else:
        plt.suptitle("Reward Components", fontsize=10)
    
    # Plot 1: Total Reward and Cumulative Reward
    ax1 = axs[0]
    ax1.set_title("Total Reward and Cumulative Reward")
    ax1.step(timestamps, rewards, 'b-', label='reward', linewidth=1)
    ax1.set_ylabel('Reward', color='blue')
    
    ax1_cum = ax1.twinx()
    ax1_cum.step(timestamps, cum_rewards, 'orange', label='cumulative_reward', linewidth=1)
    ax1_cum.set_ylabel('Cumulative Reward', color='orange')
    
    # Add combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_cum.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=6)
    
    # Plot 2: Cost Components
    ax2 = axs[1]
    ax2.set_title("Cost Components (Grid, Battery, Capacity)")
    
    # Create a twin axis for each cost component
    component_axes = {}
    for i, key in enumerate(cost_components):
        if key in reward_components:
            if i == 0:
                # First component uses the primary axis
                component_axes[key] = ax2
                ax2.set_ylabel(key, color=colors[key])
                ax2.step(timestamps, reward_components[key], color=colors[key], label=key, linewidth=1)
            else:
                # Other components get their own y-axis
                comp_ax = ax2.twinx()
                # Add some spacing between axes
                comp_ax.spines['right'].set_position(('outward', i * 50))
                comp_ax.set_ylabel(key, color=colors[key])
                comp_ax.step(timestamps, reward_components[key], color=colors[key], label=key, linewidth=1)
                component_axes[key] = comp_ax
    
    # Add legend
    all_lines, all_labels = [], []
    for key, ax in component_axes.items():
        lines, labels = ax.get_legend_handles_labels()
        all_lines.extend(lines)
        all_labels.extend(labels)
    if all_lines:
        ax2.legend(all_lines, all_labels, loc='upper left', fontsize=6)
    
    # Plot 3: Battery Management Components
    ax3 = axs[2]
    ax3.set_title("Battery Management Components (SoC Management, Preferred SoC)")
    
    # Create a twin axis for each battery component
    component_axes = {}
    for i, key in enumerate(battery_components):
        if key in reward_components:
            if i == 0:
                # First component uses the primary axis
                component_axes[key] = ax3
                ax3.set_ylabel(key, color=colors[key])
                ax3.step(timestamps, reward_components[key], color=colors[key], label=key, linewidth=1)
            else:
                # Other components get their own y-axis
                comp_ax = ax3.twinx()
                # Add some spacing between axes
                comp_ax.spines['right'].set_position(('outward', i * 50))
                comp_ax.set_ylabel(key, color=colors[key])
                comp_ax.step(timestamps, reward_components[key], color=colors[key], label=key, linewidth=1)
                component_axes[key] = comp_ax
    
    # Add legend
    all_lines, all_labels = [], []
    for key, ax in component_axes.items():
        lines, labels = ax.get_legend_handles_labels()
        all_lines.extend(lines)
        all_labels.extend(labels)
    if all_lines:
        ax3.legend(all_lines, all_labels, loc='upper left', fontsize=6)
    
    # Plot 4: Grid Optimization Components
    ax4 = axs[3]
    ax4.set_title("Grid Optimization Components (Peak Penalty, Arbitrage Bonus, Export Bonus)")
    
    # Create a twin axis for each grid component
    component_axes = {}
    for i, key in enumerate(grid_components):
        if key in reward_components:
            if i == 0:
                # First component uses the primary axis
                component_axes[key] = ax4
                ax4.set_ylabel(key, color=colors[key])
                ax4.step(timestamps, reward_components[key], color=colors[key], label=key, linewidth=1)
            else:
                # Other components get their own y-axis
                comp_ax = ax4.twinx()
                # Add some spacing between axes
                comp_ax.spines['right'].set_position(('outward', i * 50))
                comp_ax.set_ylabel(key, color=colors[key])
                comp_ax.step(timestamps, reward_components[key], color=colors[key], label=key, linewidth=1)
                component_axes[key] = comp_ax
    
    # Add legend
    all_lines, all_labels = [], []
    for key, ax in component_axes.items():
        lines, labels = ax.get_legend_handles_labels()
        all_lines.extend(lines)
        all_labels.extend(labels)
    if all_lines:
        ax4.legend(all_lines, all_labels, loc='upper left', fontsize=6)
    
    # Format x-axis for all subplots
    for ax in axs:
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Get episode duration to determine appropriate tick spacing
        episode_duration = max(timestamps) - min(timestamps)
        days_in_episode = episode_duration.days + episode_duration.seconds / 86400
        
        # Format the x-axis with better date/time display based on episode length
        if days_in_episode <= 7:  # Short episode (up to a week)
            # Show date every day
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            # Show hours only at noon
            ax.xaxis.set_minor_formatter(mdates.DateFormatter('%h'))
            ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[12]))
        elif days_in_episode <= 14:  # Medium episode (up to two weeks)
            # Show date every two days
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            # No minor ticks for hours
            ax.xaxis.set_minor_locator(plt.NullLocator())
        else:  # Long episode (more than two weeks)
            # Show date every three days
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            # No minor ticks for hours
            ax.xaxis.set_minor_locator(plt.NullLocator())
        
        # Rotate date labels for better readability and prevent overlap
        ax.tick_params(which='major', axis='x', rotation=20, labelsize=8)
        ax.tick_params(which='minor', axis='x', rotation=30, labelsize=6)
    
    # Only show x-label on bottom subplot
    ax4.set_xlabel('Date/Time', fontsize=6)
    
    # Add shaded regions for night discount periods to ALL subplots
    night_discounts = [data.get('is_night_discount', False) for data in episode_data]
    night_periods = []
    in_night_period = False
    start_idx = None
    
    for i, is_night in enumerate(night_discounts):
        if is_night and not in_night_period:
            in_night_period = True
            start_idx = i
        elif not is_night and in_night_period:
            in_night_period = False
            night_periods.append((start_idx, i))
    
    # Add the last period if it ends with night time
    if in_night_period:
        night_periods.append((start_idx, len(night_discounts)-1))
    
    # Draw shaded regions for night discount periods on all subplots
    if night_periods:
        # Add night discount shade to each subplot
        for ax in axs:
            for i, (start, end) in enumerate(night_periods):
                if start < len(timestamps) and end < len(timestamps):
                    # No label needed here to avoid cluttering legends
                    ax.axvspan(timestamps[start], timestamps[end], alpha=0.15, color='grey', zorder=0)  # zorder=0 ensures it's drawn behind other elements
    
    # Add night discount to the legend of the first subplot
    import matplotlib.patches as mpatches
    night_patch = mpatches.Patch(color='grey', alpha=0.15, label='Night (22-06) Discount')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_cum.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + [night_patch], labels1 + labels2 + ['Night (22-06) Discount'], loc='upper left', fontsize=6)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Adjust to make room for suptitle
    
    # Add a title above the plot that indicates all series have independent y-scales
    plt.figtext(0.5, 0.97, "All series have independent y-scales", 
                ha="center", fontsize=5, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    # Save figure if a directory is specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"reward_components_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Reward components plot saved to {filename}")
    
    return fig

def calculate_performance_metrics(episode_data, config=None):
    """
    Calculate and print various performance metrics from the episode data.
    
    Args:
        episode_data: List of dictionaries containing episode data
        config: Optional configuration dictionary containing preferred SoC ranges and other settings
    """
    total_reward = sum(data['reward'] for data in episode_data)
    
    # Get time step hours from the episode data if available
    time_step_h = (episode_data[1].get('timestamp') - episode_data[0].get('timestamp')).total_seconds() / 3600.0 if len(episode_data) > 1 else 0.25  # Default to 15 min = 0.25 hours
    
    # Override with config if provided
    if config and 'time_step_minutes' in config:
        time_step_h = config.get('time_step_minutes', 15) / 60.0
    
    # Battery usage metrics
    battery_powers = [abs(data.get('power_kw', 0)) for data in episode_data] # abs() here is for total throughput calculation
    # For correlation and night/day charging, we need the signed power_kw
    signed_battery_powers = [data.get('power_kw', 0) for data in episode_data]

    avg_battery_power = sum(battery_powers) / len(battery_powers) if battery_powers else 0
    max_battery_power = max(battery_powers) if battery_powers else 0
    
    # Grid metrics
    grid_powers = [data.get('grid_power_kw', 0) for data in episode_data]
    peak_grid_import = max(grid_powers) if grid_powers else 0
    peak_grid_export = min(grid_powers) if grid_powers else 0
    
    # Night discount metrics
    night_discounts = [data.get('is_night_discount', False) for data in episode_data]
    night_capacity_discount = config.get('night_capacity_discount', 0.5) if config else 0.5
    
    # Calculate discounted grid power for capacity fee calculation
    discounted_grid_powers = []
    for i, power in enumerate(grid_powers):
        if night_discounts[i] and power > 0:  # Only apply discount to positive grid power (imports)
            discounted_grid_powers.append(power * night_capacity_discount)
        else:
            discounted_grid_powers.append(power)
    
    # Calculate peak metrics with night discount applied
    peak_grid_import_with_discount = max(discounted_grid_powers) if discounted_grid_powers else 0
    
    # Calculate how much the night discount saved from peak grid import
    peak_reduction_due_to_discount = peak_grid_import - peak_grid_import_with_discount
    
    # Calculate fixed grid fee for the episode
    fixed_grid_fee_sek_per_month = config.get('fixed_grid_fee_sek_per_month', 365.0) if config else 365.0
    
    # Count unique months in the episode data
    timestamps = [data['timestamp'] for data in episode_data]
    if timestamps:
        months_in_episode = len(set([(t.year, t.month) for t in timestamps]))
        fixed_grid_fee_total = fixed_grid_fee_sek_per_month * months_in_episode
    else:
        months_in_episode = 0
        fixed_grid_fee_total = 0.0
    
    # Count number of peak events that occurred during night discount periods
    peak_threshold = peak_grid_import * 0.8  # Consider any value over 80% of peak as a significant peak
    night_peaks = sum(1 for i, power in enumerate(grid_powers) if night_discounts[i] and power > peak_threshold)
    day_peaks = sum(1 for i, power in enumerate(grid_powers) if not night_discounts[i] and power > peak_threshold)
    
    # Price response metrics
    prices = [data.get('current_price', 0) for data in episode_data]
    
    # Calculate correlation between price and battery power
    # Positive correlation means discharging more at high prices (good)
    if len(prices) > 1 and len(signed_battery_powers) > 1:
        price_power_correlation = np.corrcoef(prices, signed_battery_powers)[0, 1]
    else:
        price_power_correlation = 0
    
    # Calculate night vs day metrics
    night_prices = [data.get('current_price', 0) for data in episode_data if data.get('is_night_discount', False)]
    day_prices = [data.get('current_price', 0) for data in episode_data if not data.get('is_night_discount', False)]
    
    night_charges_energy_kwh = 0
    day_charges_energy_kwh = 0

    for data in episode_data:
        power_kw = data.get('power_kw', 0)
        # Negative power_kw means charging
        if power_kw < 0: # Charging
            energy_charged_kwh = abs(power_kw) * time_step_h # abs because power_kw is negative
            if data.get('is_night_discount', False):
                night_charges_energy_kwh += energy_charged_kwh
            else:
                day_charges_energy_kwh += energy_charged_kwh
    
    avg_night_price = sum(night_prices) / len(night_prices) if night_prices else 0
    avg_day_price = sum(day_prices) / len(day_prices) if day_prices else 0
    
    # Night vs day charging ratio (higher is better - charging more at night)
    night_day_charge_ratio = night_charges_energy_kwh / day_charges_energy_kwh if day_charges_energy_kwh > 0 else float('inf')
    
    # Calculate SoC metrics
    soc_values = [data['soc'][0] for data in episode_data]
    avg_soc = sum(soc_values) / len(soc_values) if soc_values else 0
    min_soc = min(soc_values) if soc_values else 0
    max_soc = max(soc_values) if soc_values else 0
    
    # New: Calculate capacity fee metrics
    top3_peaks = []
    peak_rolling_average = 0
    current_capacity_fee = 0
    
    # Try to get the latest capacity metrics
    for data in reversed(episode_data):
        if 'top3_peaks' in data:
            top3_peaks = data['top3_peaks']
            break
    
    for data in reversed(episode_data):
        if 'peak_rolling_average' in data:
            peak_rolling_average = data['peak_rolling_average']
            break
    
    for data in reversed(episode_data):
        if 'current_capacity_fee' in data:
            current_capacity_fee = data['current_capacity_fee']
            break
    
    # Calculate estimated savings from night discount on capacity fee
    capacity_fee_sek_per_kw = config.get('capacity_fee_sek_per_kw', 81.25) if config else 81.25
    estimated_savings = peak_reduction_due_to_discount * capacity_fee_sek_per_kw if peak_reduction_due_to_discount > 0 else 0
    
    # Calculate export metrics
    export_energy_kwh = sum(abs(data['grid_power_kw'] * (time_step_h if 'time_step_h' in locals() else 0.25)) 
                           for data in episode_data if data.get('grid_power_kw', 0) < 0)
    export_revenue = sum(data.get('export_bonus', 0) for data in episode_data)
    
    # Identify all reward components dynamically
    reward_components = {}
    for data in episode_data:
        for key in data.keys():
            if key.startswith('reward_') and key not in reward_components:
                reward_components[key] = 0

    # Sum up all reward components
    for key in reward_components.keys():
        reward_components[key] = sum(data.get(key, 0) for data in episode_data)
    
    # Get preferred SoC ranges from config or use defaults
    preferred_soc_min = 0.3
    preferred_soc_max = 0.7
    
    if config:
        preferred_soc_min = config.get('preferred_soc_min_base', 0.3)
        preferred_soc_max = config.get('preferred_soc_max_base', 0.7)
    
    # Calculate percentage of time spent in preferred SoC range
    time_in_preferred_range = sum(1 for soc in soc_values if preferred_soc_min <= soc <= preferred_soc_max)
    pct_time_in_preferred_range = (time_in_preferred_range / len(soc_values) * 100) if soc_values else 0
    
    # Time at high SoC (> 0.8)
    high_soc_threshold = 0.8
    if config:
        high_soc_threshold = config.get('high_soc_threshold', 0.8)
    
    time_at_high_soc = sum(1 for soc in soc_values if soc > high_soc_threshold)
    pct_time_at_high_soc = (time_at_high_soc / len(soc_values) * 100) if soc_values else 0
    
    # Calculate SoC violation percentages
    soc_min_limit = 0.2
    soc_max_limit = 0.8
    
    if config:
        soc_min_limit = config.get('soc_min_limit', 0.2)
        soc_max_limit = config.get('soc_max_limit', 0.8)
    
    soc_violations = sum(1 for soc in soc_values if soc < soc_min_limit or soc > soc_max_limit)
    violation_percentage = (soc_violations / len(soc_values) * 100) if soc_values else 0
    
    # Print results
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average battery power: {avg_battery_power:.2f} kW")
    print(f"Maximum battery power: {max_battery_power:.2f} kW")
    print(f"Peak grid import: {peak_grid_import:.2f} kW")
    print(f"Peak grid export: {abs(peak_grid_export):.2f} kW")
    print(f"Total export energy: {export_energy_kwh:.2f} kWh")
    print(f"Total export revenue: {export_revenue:.2f} öre (approx {export_revenue/100:.2f} SEK)")
    print(f"Price-power correlation: {price_power_correlation:.4f} (higher is better)")
    print(f"Average night price: {avg_night_price:.2f} öre/kWh")
    print(f"Average day price: {avg_day_price:.2f} öre/kWh")
    print(f"Total charging at night: {night_charges_energy_kwh:.2f} kWh")
    print(f"Total charging during day: {day_charges_energy_kwh:.2f} kWh")
    print(f"Night/day charging ratio: {night_day_charge_ratio:.2f} (higher means more night charging)")
    
    # Print SoC metrics
    print(f"\n--- SoC Metrics ---")
    print(f"Average SoC: {avg_soc:.2f}")
    print(f"Min SoC: {min_soc:.2f}")
    print(f"Max SoC: {max_soc:.2f}")
    print(f"Time in preferred SoC range ({preferred_soc_min:.1f}-{preferred_soc_max:.1f}): {pct_time_in_preferred_range:.1f}%")
    print(f"Time at high SoC (>{high_soc_threshold:.1f}): {pct_time_at_high_soc:.1f}%")
    print(f"SoC violations (<{soc_min_limit:.1f} or >{soc_max_limit:.1f}): {violation_percentage:.1f}%")
    
    # Print capacity fee metrics
    print(f"\n--- Capacity Fee Metrics ---")
    if top3_peaks:
        print(f"Top 3 peaks: {[f'{peak:.2f} kW' for peak in top3_peaks]}")
    print(f"Peak rolling average: {peak_rolling_average:.2f} kW")
    print(f"Estimated monthly capacity fee: {current_capacity_fee:.2f} SEK")
    print(f"Fixed grid fee per month: {fixed_grid_fee_sek_per_month:.2f} SEK")
    print(f"Months in episode: {months_in_episode}")
    print(f"Total fixed grid fee: {fixed_grid_fee_total:.2f} SEK")
    
    # Print night discount metrics
    print(f"\n--- Night Discount Metrics ---")
    print(f"Night capacity discount factor: {night_capacity_discount:.2f}")
    print(f"Peak grid import (no discount): {peak_grid_import:.2f} kW")
    print(f"Peak grid import (with discount): {peak_grid_import_with_discount:.2f} kW")
    print(f"Peak reduction due to discount: {peak_reduction_due_to_discount:.2f} kW")
    print(f"Estimated monthly savings: {estimated_savings:.2f} SEK")
    print(f"Number of significant peaks during night hours: {night_peaks}")
    print(f"Number of significant peaks during day hours: {day_peaks}")
    
    # Print reward component metrics
    print(f"\n--- Reward Component Contributions ---")
    for key, value in reward_components.items():
        print(f"{key}: {value:.2f}")
    
    # Group similar metrics together in the return value
    return {
        "total_reward": total_reward,
        "battery_metrics": {
            "avg_power": avg_battery_power,
            "max_power": max_battery_power,
            "night_charges": night_charges_energy_kwh,
            "day_charges": day_charges_energy_kwh,
            "night_day_ratio": night_day_charge_ratio
        },
        "grid_metrics": {
            "peak_import": peak_grid_import,
            "peak_export": peak_grid_export,
            "export_energy_kwh": export_energy_kwh,
            "export_revenue": export_revenue
        },
        "price_metrics": {
            "avg_night_price": avg_night_price,
            "avg_day_price": avg_day_price
        },
        "soc_metrics": {
            "avg": avg_soc,
            "min": min_soc,
            "max": max_soc,
            "in_preferred_range_pct": pct_time_in_preferred_range,
            "at_high_soc_pct": pct_time_at_high_soc,
            "violation_percentage": violation_percentage
        },
        "peak_metrics": {
            "top3_peaks": top3_peaks,
            "peak_rolling_average": peak_rolling_average,
            "capacity_fee": current_capacity_fee,
            "fixed_grid_fee_monthly": fixed_grid_fee_sek_per_month,
            "months_in_episode": months_in_episode,
            "fixed_grid_fee_total": fixed_grid_fee_total
        },
        "night_discount_metrics": {
            "discount_factor": night_capacity_discount,
            "peak_without_discount": peak_grid_import,
            "peak_with_discount": peak_grid_import_with_discount,
            "peak_reduction": peak_reduction_due_to_discount,
            "estimated_savings": estimated_savings,
            "night_peaks": night_peaks,
            "day_peaks": day_peaks
        },
        "price_power_correlation": price_power_correlation,
        "reward_components": reward_components
    }

def load_agent(model_path):
    """
    Load a trained RL agent from the specified path.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        The loaded agent, or None if loading fails
    """
    try:
        print(f"Loading agent from {model_path}")
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        print("Agent loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading agent: {e}")
        return None

def evaluate_episode(agent, config):
    """
    Evaluate a single episode using the provided agent and config.
    
    Args:
        agent: The RL agent to evaluate
        config: Configuration dictionary
        
    Returns:
        A list of dictionaries containing episode data
    """
    from src.rl.custom_env import HomeEnergyEnv
    
    # Create environment
    env = HomeEnergyEnv(config=config)
    
    # Reset environment
    obs, info = env.reset()
    
    episode_data = []
    terminated = False
    truncated = False
    
    # Run episode
    while not (terminated or truncated):
        # Get action from agent
        action, _states = agent.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Extract reward components directly from info['reward_components']
        reward_components = info.get('reward_components', {})
        
        # Store data for this step
        step_data = {
            'timestamp': env.start_datetime + datetime.timedelta(hours=env.current_step * env.time_step_hours),
            'soc': obs['soc'],
            'action': action[0],  # Extract scalar from array
            'reward': reward,
            'current_price': info.get('current_price', 0),
            'power_kw': info.get('power_kw', 0),
            'grid_power_kw': info.get('grid_power_kw', 0),
            'base_demand_kw': info.get('base_demand_kw', 0),
            'current_solar_production_kw': info.get('current_solar_production_kw', 0),
            'is_night_discount': info.get('is_night_discount', False),
        }
        
        # Add all reward components from the info dictionary
        for key, value in reward_components.items():
            step_data[key] = value
        
        episode_data.append(step_data)
    
    print(f"Episode completed with {len(episode_data)} steps")
    return episode_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a trained RL agent for home energy management")
    parser.add_argument("--render", action="store_true", help="Whether to render the environment")
    
    args = parser.parse_args()
    
    # Get configuration from config.py
    from src.rl.config import get_config_dict
    config = get_config_dict()

    # --- Start of new logic to select a random full month ---
    price_predictions_path_str = config.get("price_predictions_path", "src/predictions/prices/plots/predictions/merged")
    # Construct path relative to project root, assuming evaluate_agent.py is in src/rl/
    # PROJECT_ROOT is defined earlier in the script
    price_file_path = Path(PROJECT_ROOT) / price_predictions_path_str

    min_data_date = None
    max_data_date = None

    if price_file_path.exists():
        try:
            price_df = pd.read_csv(price_file_path, index_col='HourSE', parse_dates=True)
            if not price_df.empty:
                if price_df.index.tzinfo is None:
                    try:
                        price_df.index = price_df.index.tz_localize('Europe/Stockholm', ambiguous='NaT', nonexistent='shift_forward')
                        price_df = price_df[~pd.isna(price_df.index)] 
                    except Exception as loc_e:
                        print(f"Warning: Could not localize price data timestamps: {loc_e}. Proceeding with naive timestamps if any.")
                
                if not price_df.empty:
                    min_data_date = price_df.index.min()
                    max_data_date = price_df.index.max()
        except Exception as e:
            print(f"Warning: Could not read price data from {price_file_path} for date range determination: {e}")

    if min_data_date is None or max_data_date is None:
        print("Error: Could not determine data date range from price file. Exiting.")
        sys.exit(1)

    possible_months = []
    # Ensure min_data_date has timezone info if it's going to be used with tz-aware operations
    # For simplicity, if min_data_date became tz-aware, current_month_start should also be.
    # If price_df.index was successfully localized, min_data_date will be tz-aware.
    current_month_start = pd.Timestamp(min_data_date.year, min_data_date.month, 1, tzinfo=min_data_date.tzinfo)

    while current_month_start <= max_data_date:
        month_end = current_month_start + pd.offsets.MonthEnd(0)
        # Ensure the full month is within the data range
        if month_end <= max_data_date and current_month_start >= min_data_date:
            possible_months.append((current_month_start.year, current_month_start.month))
        
        # Move to the start of the next month
        current_month_start += pd.offsets.MonthBegin(1)


    if not possible_months:
        print(f"Error: No full months available in the data range {min_data_date} to {max_data_date}. Exiting.")
        sys.exit(1)

    selected_year, selected_month = random.choice(possible_months)
    days_in_selected_month = monthrange(selected_year, selected_month)[1]

    print(f"Selected random month for evaluation: {selected_year}-{selected_month:02d} ({days_in_selected_month} days)")

    config["simulation_days"] = days_in_selected_month
    config["force_specific_start_month"] = True
    config["start_year"] = selected_year
    config["start_month"] = selected_month
    # --- End of new logic ---

    if args.render:
        config["render_mode"] = "human"
    
    model_name = "short_term_agent_final"
    # Load the agent
    model_path = f"{config['model_dir']}/{model_name}"
    agent = load_agent(model_path)
    
    if agent is None:
        print(f"Failed to load agent from {model_path}")
        sys.exit(1)
    
    # Evaluate the agent
    print(f"Evaluating agent for {selected_year}-{selected_month:02d}...")
    episode_data = evaluate_episode(agent, config)
    
    plot_dir = f"src/rl/simulations/results/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{selected_year}_{selected_month:02d}" # Added year/month to dir name
    # Plot the performance
    fig, axs = plot_agent_performance(episode_data, model_name=model_name, save_dir=plot_dir)
    
    # Plot reward components separately
    plot_reward_components(episode_data, model_name=model_name, save_dir=plot_dir)
    
    # Calculate and print metrics
    print("\n--- Performance Metrics ---")
    calculate_performance_metrics(episode_data, config)
    
    plt.show()  # Show all plots 