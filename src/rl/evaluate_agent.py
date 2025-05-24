"""
Evaluate a trained RL agent in the home energy management environment.

This script contains functions for evaluating a trained agent's performance
and visualizing the results.
"""
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
from datetime import timedelta, datetime as dt
import random
from calendar import monthrange

# Add the project root directory (one level up from 'src') to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.rl.config import get_config_dict
from src.rl.custom_env import HomeEnergyEnv
from src.rl.agent import RecurrentEnergyAgent
from gymnasium.wrappers import FlattenObservation
import gymnasium as gym


def plot_agent_performance(episode_data, model_name=None, save_dir=None):
    """
    Generate a comprehensive plot of agent's performance during an episode.
    Handles different data resolutions properly:
    - Hourly: Prices, solar, household consumption, grid power peaks
    - 15-min: Battery SoC, actions, battery power
    
    Args:
        episode_data: A list of episode timestep dictionaries
        model_name: Optional name of the model for the plot title
        save_dir: Optional directory to save the plot
    """
    # Extract data at original 15-min resolution
    timestamps_15min = [data['timestamp'] for data in episode_data]
    soc_values = []
    for data in episode_data:
        soc = data['soc']
        if isinstance(soc, (list, tuple, np.ndarray)):
            soc_values.append(soc[0])
        else:
            soc_values.append(soc)
    battery_powers = [data.get('power_kw', 0) for data in episode_data]
    original_actions = [data.get('original_action', data.get('action', 0)) for data in episode_data]
    safe_actions = [data.get('safe_action', data.get('action', 0)) for data in episode_data]
    actions_modified = [data.get('action_modified', False) for data in episode_data]
    rewards = [data['reward'] for data in episode_data]
    
    # Create a DataFrame for hourly resampling of naturally hourly data
    df_15min = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps_15min),
        'current_price': [data.get('current_price', 0) for data in episode_data],
        'grid_power_kw': [data.get('grid_power_kw', 0) for data in episode_data],
        'base_demand_kw': [data.get('base_demand_kw', 0) for data in episode_data],
        'current_solar_production_kw': [data.get('current_solar_production_kw', 0) for data in episode_data],
        'is_night_discount': [data.get('is_night_discount', False) for data in episode_data],
    })
    df_15min.set_index('timestamp', inplace=True)
    
    # Resample to hourly data - using appropriate aggregation for each type of data
    hourly_data = df_15min.resample('h').agg({
        'current_price': 'first',  # Prices are hourly, take first value of each hour
        'grid_power_kw': 'mean',   # Average power over the hour
        'base_demand_kw': 'mean',  # Average consumption
        'current_solar_production_kw': 'mean',  # Average solar
        'is_night_discount': 'first'  # Night discount doesn't change within hour
    }).fillna(0)
    
    # Extract hourly timestamps and data
    timestamps_hourly = hourly_data.index.tolist()
    prices_hourly = hourly_data['current_price'].tolist()
    grid_powers_hourly = hourly_data['grid_power_kw'].tolist()
    base_demands_hourly = hourly_data['base_demand_kw'].tolist()
    solar_productions_hourly = hourly_data['current_solar_production_kw'].tolist()
    
    # Calculate discounted grid power for capacity fee
    config = get_config_dict()
    night_capacity_discount = config.get('night_capacity_discount', 0.5)
    
    hourly_data['discounted_grid_power_kw'] = hourly_data.apply(
        lambda row: row['grid_power_kw'] * night_capacity_discount if row['is_night_discount'] and row['grid_power_kw'] > 0 
        else row['grid_power_kw'], axis=1
    )
    discounted_grid_powers_hourly = hourly_data['discounted_grid_power_kw'].tolist()
    
    # Calculate cumulative reward
    cum_rewards = np.cumsum(rewards)
    
    # Create figure with subplots
    fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1, 1]})
    
    # Set title based on model name if provided
    if model_name:
        fig.suptitle(f"Agent Performance Analysis - Model: {model_name}", fontsize=10)
    else:
        fig.suptitle("Agent Performance Analysis", fontsize=10)
    
    # Subplot 1: Battery SoC and Electricity Price (different resolutions)
    ax1 = axs[0]
    ax1.set_title("Battery SoC and Electricity Price")
    ax1.set_ylabel("SoC")
    
    # Plot SoC at 15-min resolution
    ax1.step(timestamps_15min, soc_values, 'r-', label="Battery SoC", linewidth=1)
    ax1.set_ylim(0, 1.05)
    
    # Plot Price on secondary y-axis at HOURLY resolution
    ax2 = ax1.twinx()
    ax2.set_ylabel("Electricity Price (öre/kWh)")
    
    # Plot prices at hourly resolution
    ax2.step(timestamps_hourly, prices_hourly, 'b-', label="Electricity Price", linewidth=1)
    
    # Create combined legend for both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    
    # Subplot 2: Agent Actions and Battery Power (15-min resolution)
    ax3 = axs[1]
    ax3.set_title("Agent Actions and Battery Power (+ = Discharging, - = Charging; Action Masking Shown)")
    ax3.set_ylabel("Action (-1 to 1)")
    
    # Plot original and safe actions - all at 15-min resolution
    ax3.step(timestamps_15min, original_actions, 'g-', alpha=0.7, label="Original Agent Action", linewidth=1)
    ax3.step(timestamps_15min, safe_actions, 'g--', alpha=1.0, label="Safe Action (After Masking)", linewidth=1)
    
    # Highlight points where action was modified
    modified_timestamps = [timestamps_15min[i] for i in range(len(timestamps_15min)) if actions_modified[i]]
    modified_actions = [original_actions[i] for i in range(len(original_actions)) if actions_modified[i]]
    
    if modified_timestamps:
        ax3.scatter(modified_timestamps, modified_actions, c='r', s=10, alpha=0.7, label="Modified Actions")
    
    ax3.set_ylim(-1.1, 1.1)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Plot Battery Power on secondary y-axis at 15-min resolution
    ax4 = ax3.twinx()
    ax4.set_ylabel("Battery Power (kW)")
    ax4.step(timestamps_15min, battery_powers, 'm-', label="Battery Power (kW)", linewidth=1)
    
    # Create combined legend for both axes
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    lines_4, labels_4 = ax4.get_legend_handles_labels()
    ax3.legend(lines_3 + lines_4, labels_3 + labels_4, loc='upper right')
    
    # Subplot 3: Household Consumption, Grid Power, and Solar Production (HOURLY)
    ax5 = axs[2]
    ax5.set_title("Household Consumption and Solar Production (Hourly)")
    ax5.set_ylabel("Power (kW)")
    ax5.step(timestamps_hourly, base_demands_hourly, 'blue', label="Household Consumption (kW)", linewidth=1)
    # ax5.step(timestamps_hourly, grid_powers_hourly, 'r-', label="Net Grid Power (kW)", linewidth=1)
    ax5.step(timestamps_hourly, solar_productions_hourly, 'orange', label="Solar Production (kW)", linewidth=1)
    
    ax5.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax5.legend(loc='upper right')
    
    # Subplot 4: Grid Peaks and Capacity Fee with Night Discount (HOURLY)
    ax7 = axs[3]
    ax7.set_ylabel("Power (kW)")
    
    # Add grid power for comparison (hourly)
    ax7.step(timestamps_hourly, grid_powers_hourly, 'black', alpha=0.3, 
             label="Current Grid Power (Hourly)", linewidth=1)
    
    # Add discounted grid power (hourly)
    ax7.step(timestamps_hourly, discounted_grid_powers_hourly, 'purple', alpha=0.6, linewidth=1, 
             label=f"Discounted Grid Power (Hourly, Night Factor: {night_capacity_discount})")
    
    # Find the 3 highest HOURLY peak events from discounted_grid_powers_hourly
    monthly_capacity_cost_from_env = 0.0  # Default value
    if episode_data:
        last_step_data = episode_data[-1]
        monthly_capacity_cost_from_env = last_step_data.get('current_capacity_fee', 0.0)

    if timestamps_hourly and discounted_grid_powers_hourly:
        # Create a list of (timestamp, discounted_hourly_power) tuples
        hourly_power_events = list(zip(timestamps_hourly, discounted_grid_powers_hourly))

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
            # Ensure we have 3 peak values for averaging, using only positive peaks shown
            peak_values_for_avg = [p_val for _, p_val in top_hourly_peak_events]
            while len(peak_values_for_avg) < 3:
                peak_values_for_avg.append(0.0)  # Pad if less than 3 peaks found
            
            if peak_values_for_avg:  # Should always be true if top_hourly_peak_events was populated
                average_of_plotted_peaks = sum(peak_values_for_avg) / len(peak_values_for_avg)
                capacity_fee_rate = config.get('capacity_fee_sek_per_kw', 81.25)
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
    ax7.legend(loc='upper right', fontsize=6)    
    ax7.set_title(f"Grid Power Peaks (Hourly Discounted). Capacity Cost: {plot_derived_capacity_cost:.2f} kr")
    
    # Subplot 5: Cumulative Reward (15-min to match timestamps)
    ax6 = axs[4]
    ax6.set_title("Cumulative Reward Over Episode")
    ax6.set_ylabel("Reward")
    ax6.step(timestamps_15min, cum_rewards, 'brown', label="Cumulative Reward", linewidth=1)
    ax6.legend(loc='upper left')
    
    # Add shaded regions for night discount periods to ALL subplots
    night_periods = []
    in_night_period = False
    start_idx = None
    
    night_discounts = [data.get('is_night_discount', False) for data in episode_data]
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
        for ax in axs:
            for start, end in night_periods:
                if start < len(timestamps_15min) and end < len(timestamps_15min):
                    ax.axvspan(timestamps_15min[start], timestamps_15min[end], 
                               alpha=0.15, color='grey', zorder=0)
    
    # Add legend for night discount to first subplot
    import matplotlib.patches as mpatches
    night_patch = mpatches.Patch(color='grey', alpha=0.15, label='Night (22-06) Discount')
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2 + [night_patch], labels_1 + labels_2 + ['Night (22-06) Discount'], 
               loc='upper right', fontsize=6)
    
    # Format x-axis with proper date formatting
    for ax in axs:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=20)
    
    # Add bottom x-axis label
    axs[4].set_xlabel("Date", fontsize=6)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if a directory is specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, 
                              f"agent_performance_{dt.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    
    return fig, axs


def plot_reward_components(episode_data, output_path):
    """
    Create a detailed plot of reward components over time with grouped components.
    Components are grouped logically with individual y-axis scaling for better visibility.
    
    Args:
        episode_data: List of dictionaries with episode data
        output_path: Path to save the plot
    """
    timestamps = [data['timestamp'] for data in episode_data]
    total_rewards = [data['reward'] for data in episode_data]
    cumulative_reward = np.cumsum(total_rewards)
    
    # Extract reward components
    grid_costs = [data.get('reward_grid_cost', 0) for data in episode_data]
    capacity_penalties = [data.get('reward_capacity_penalty', 0) for data in episode_data]
    battery_costs = [data.get('reward_battery_cost', 0) for data in episode_data]
    soc_rewards = [data.get('reward_soc_reward', 0) for data in episode_data]
    shaping_rewards = [data.get('reward_shaping', 0) for data in episode_data]
    arbitrage_bonuses = [data.get('reward_arbitrage_bonus', 0) for data in episode_data]
    export_bonuses = [data.get('reward_export_bonus', 0) for data in episode_data]
    night_charging_rewards = [data.get('reward_night_charging', 0) for data in episode_data]
    action_mod_penalties = [data.get('reward_action_mod_penalty', 0) for data in episode_data]
    solar_soc_rewards = [data.get('reward_solar_soc', 0) for data in episode_data]
    night_peak_chain_bonuses = [data.get('reward_night_peak_chain', 0) for data in episode_data]
    
    # Define component groups - now with 5 groups for less cramped visualization
    threshold = 0.001
    component_groups = {
        'Direct Costs': {
            'Grid Cost': (grid_costs, 'red', '-'),
            'Capacity Penalty': (capacity_penalties, 'darkred', '-')
        },
        'System Penalties': {
            'Battery Degradation': (battery_costs, 'orange', '-'),
            'Action Mod Penalty': (action_mod_penalties, 'maroon', '-')
        },
        'Primary Rewards': {
            'SoC Reward': (soc_rewards, 'green', '-'),
            'Arbitrage Bonus': (arbitrage_bonuses, 'purple', '-')
        },
        'Energy Trading': {
            'Export Bonus': (export_bonuses, 'blue', '-'),
            'Night Charging': (night_charging_rewards, 'navy', '-')
        },
        'Advanced Strategies': {
            'Potential Shaping': (shaping_rewards, 'magenta', '-'),
            'Solar SoC': (solar_soc_rewards, 'darkgreen', '-'),
            'Night→Peak Chain': (night_peak_chain_bonuses, 'cyan', '-')
        }
    }
    
    # Filter groups to only include those with meaningful values
    active_groups = {}
    for group_name, components in component_groups.items():
        active_components = {}
        for comp_name, (values, color, linestyle) in components.items():
            if any(abs(v) > threshold for v in values):
                # Calculate stats for the component
                max_val = max(values)
                min_val = min(values)
                avg_val = sum(values) / len(values)
                std_val = np.std(values)
                active_components[comp_name] = {
                    'values': values, 
                    'color': color,
                    'linestyle': linestyle,
                    'max': max_val,
                    'min': min_val, 
                    'avg': avg_val,
                    'std': std_val,
                    'range': max_val - min_val
                }
        
        if active_components:  # Only include groups that have active components
            active_groups[group_name] = active_components
    
    print(f"Active component groups found: {len(active_groups)}")
    for group_name, components in active_groups.items():
        print(f"  {group_name}: {list(components.keys())}")
    
    # Set up the figure - Total reward at top, then grouped components (now 5 component groups max)
    num_group_plots = len(active_groups)
    total_subplots = 1 + num_group_plots  # 1 for total/cumulative, rest for groups
    
    # Create figure with appropriate sizing - more height for additional subplot
    fig_height = min(max(10, total_subplots * 2.5), 20)  # Between 10 and 20 inches, more space per subplot
    fig, axes = plt.subplots(total_subplots, 1, figsize=(16, fig_height), sharex=True)
    
    # Ensure axes is always a list
    if total_subplots == 1:
        axes = [axes]
    
    # Plot 0: Total reward and cumulative reward
    ax0 = axes[0]
    ax0.plot(timestamps, total_rewards, 'b-', label='Step Reward', linewidth=1)
    ax0.set_ylabel('Step Reward', color='blue')
    ax0.tick_params(axis='y', labelcolor='blue')
    ax0.set_title('Total Reward per Step and Cumulative Reward', fontsize=12, fontweight='bold')
    ax0.grid(True, alpha=0.3)
    
    # Add night discount periods as shaded areas
    night_periods = []
    current_night_start = None
    
    for i, ts in enumerate(timestamps):
        is_night = episode_data[i].get('is_night_discount', False)
        if is_night and current_night_start is None:
            current_night_start = ts
        elif not is_night and current_night_start is not None:
            night_periods.append((current_night_start, ts))
            current_night_start = None
    
    # Add last night period if it extends to the end
    if current_night_start is not None:
        night_periods.append((current_night_start, timestamps[-1]))
    
    for start, end in night_periods:
        ax0.axvspan(start, end, alpha=0.15, color='gray', label='Night Discount' if start == night_periods[0][0] else "")
    
    # Add cumulative reward on secondary y-axis
    ax0_twin = ax0.twinx()
    ax0_twin.plot(timestamps, cumulative_reward, 'orange', label='Cumulative Reward', linewidth=1)
    ax0_twin.set_ylabel('Cumulative Reward', color='orange')
    ax0_twin.tick_params(axis='y', labelcolor='orange')
    
    # Add zero line to total reward plot
    ax0.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    # Plot grouped components with individual y-axis scaling
    for i, (group_name, group_components) in enumerate(active_groups.items()):
        ax = axes[i + 1]
        
        # Plot each component in the group
        for comp_name, comp_data in group_components.items():
            values = comp_data['values']
            color = comp_data['color']
            linestyle = comp_data['linestyle']
            
            # Plot the component
            ax.plot(timestamps, values, color=color, linestyle=linestyle, 
                   linewidth=1, label=f"{comp_name} (μ={comp_data['avg']:.3f})")
        
        # Set group title and labels
        ax.set_ylabel(f'{group_name}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        # Individual y-axis scaling - let matplotlib auto-scale each subplot
        # Calculate range for this group
        all_group_values = []
        for comp_data in group_components.values():
            all_group_values.extend(comp_data['values'])
        
        if all_group_values:
            group_min = min(all_group_values)
            group_max = max(all_group_values)
            group_range = group_max - group_min
            
            # Add zero line if range includes both positive and negative values
            if group_min < 0 < group_max:
                ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            
            # Set title with group range statistics
            range_str = f"Range: [{group_min:.3f}, {group_max:.3f}]"
            ax.set_title(f'{group_name} - {range_str}', fontsize=10)
            
            # Individual y-axis scaling with small margin
            if group_range > 0:
                margin = group_range * 0.05  # 5% margin
                ax.set_ylim(group_min - margin, group_max + margin)
            else:
                # Handle case where all values are the same
                ax.set_ylim(group_min - 0.001, group_max + 0.001)
        
        # Add night periods to component plots as well
        for start, end in night_periods:
            ax.axvspan(start, end, alpha=0.1, color='gray')
    
    # Format x-axis with date formatting only on the bottom plot
    date_format = mdates.DateFormatter('%m-%d %H:%M')
    axes[-1].xaxis.set_major_formatter(date_format)
    
    # Adjust locator based on episode length
    episode_hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
    if episode_hours <= 48:  # Less than 2 days - show every 6 hours
        axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=6))
    elif episode_hours <= 168:  # Less than a week - show daily
        axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    else:  # Longer periods - show every 2 days
        axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    
    axes[-1].set_xlabel('Time')
    fig.autofmt_xdate()
    
    # Add overall statistics text box
    total_reward_sum = sum(total_rewards)
    avg_step_reward = total_reward_sum / len(total_rewards)
    
    # Count total active components across all groups
    total_active_components = sum(len(comps) for comps in active_groups.values())
    
    stats_text = f"""Episode Statistics:
Total Reward: {total_reward_sum:.2f}
Avg Step Reward: {avg_step_reward:.4f}
Component Groups: {len(active_groups)}
Active Components: {total_active_components}
Episode Duration: {episode_hours:.1f} hours"""
    
    # Place stats text box on the first subplot
    ax0.text(0.02, 0.98, stats_text, transform=ax0.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Reward components plot with grouped components saved to {output_path}")
    print(f"  - {len(active_groups)} component groups plotted")
    print(f"  - {total_active_components} total active components")
    print(f"  - Episode duration: {episode_hours:.1f} hours")
    print(f"  - Total reward: {total_reward_sum:.2f}")
    print(f"  - Individual y-axis scaling applied to all subplots")

def calculate_performance_metrics(episode_data, config=None):
    """
    Calculate performance metrics for an episode.
    
    Args:
        episode_data: List of dictionaries with episode data
        config: Optional configuration dictionary
        
    Returns:
        Dictionary with performance metrics
    """
    import numpy as np
    
    # Extract data
    timestamps = [data['timestamp'] for data in episode_data]
    
    # Handle SOC values that could be either scalars or arrays
    soc_values = []
    for data in episode_data:
        soc = data['soc']
        if isinstance(soc, (list, tuple, np.ndarray)):
            soc_values.append(soc[0])
        else:
            soc_values.append(soc)
    
    grid_powers = [data.get('grid_power_kw', 0) for data in episode_data]
    battery_powers = [data.get('power_kw', 0) for data in episode_data]
    
    total_reward = sum(data['reward'] for data in episode_data)
    
    # Get time step hours from the episode data if available
    time_step_h = (episode_data[1].get('timestamp') - episode_data[0].get('timestamp')).total_seconds() / 3600.0 if len(episode_data) > 1 else 0.25  # Default to 15 min = 0.25 hours
    
    # Override with config if provided
    if config and 'time_step_minutes' in config:
        time_step_h = config.get('time_step_minutes', 15) / 60.0
    
    # Battery usage metrics
    avg_battery_power = sum(battery_powers) / len(battery_powers) if battery_powers else 0
    max_battery_power = max(battery_powers) if battery_powers else 0
    
    # Grid metrics
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
    if len(prices) > 1 and len(battery_powers) > 1:
        price_power_correlation = np.corrcoef(prices, battery_powers)[0, 1]
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
    
    # Action masking metrics - new
    actions_modified = [data.get('action_modified', False) for data in episode_data]
    num_modified_actions = sum(1 for modified in actions_modified if modified)
    pct_modified_actions = (num_modified_actions / len(actions_modified) * 100) if actions_modified else 0
    
    # Calculate deltas between original and safe actions where modified
    action_modification_deltas = []
    for data in episode_data:
        if data.get('action_modified', False):
            original = data.get('original_action', 0)
            safe = data.get('safe_action', 0)
            action_modification_deltas.append(abs(original - safe))
    
    avg_action_delta = sum(action_modification_deltas) / len(action_modification_deltas) if action_modification_deltas else 0
    max_action_delta = max(action_modification_deltas) if action_modification_deltas else 0
    
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
    
    # Print action masking metrics
    print(f"\n--- Action Masking Metrics ---")
    print(f"Actions modified by safety mask: {num_modified_actions} ({pct_modified_actions:.1f}%)")
    if action_modification_deltas:
        print(f"Average modification magnitude: {avg_action_delta:.4f}")
        print(f"Maximum modification magnitude: {max_action_delta:.4f}")
    
    # Print reward component metrics with only the current active components
    print(f"\n--- Reward Component Contributions ---")
    active_components = [
        'reward_grid_cost',
        'reward_capacity_penalty',
        'reward_battery_cost',
        'reward_soc_reward',
        'reward_shaping',
        'reward_arbitrage_bonus', 
        'reward_export_bonus',
        'reward_night_charging',
        'reward_action_mod_penalty'
    ]
    
    for component in active_components:
        if component in reward_components:
            print(f"{component}: {reward_components[component]:.2f}")
    
    # Add reward component distribution analysis
    print(f"\n--- Reward Component Distribution Analysis ---")
    component_distributions = analyze_reward_component_distributions(episode_data)
    for component, stats in component_distributions.items():
        if abs(stats['total']) > 0:  # Only show components with non-zero totals
            print(f"{component}:")
            print(f"  Range: {stats['min']:.2f} to {stats['max']:.2f}")
            print(f"  Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}, StdDev: {stats['std']:.2f}")
            print(f"  Percentiles: P5={stats['p5']:.2f}, P25={stats['p25']:.2f}, P75={stats['p75']:.2f}, P95={stats['p95']:.2f}")
            print(f"  Zero values: {stats['zero_pct']:.1f}%, Total: {stats['total']:.2f}")
    
    return {
        'total_reward': total_reward,
        'avg_battery_power': avg_battery_power,
        'max_battery_power': max_battery_power,
        'peak_grid_import': peak_grid_import,
        'peak_grid_export': abs(peak_grid_export),
        'export_energy_kwh': export_energy_kwh,
        'export_revenue': export_revenue,
        'price_power_correlation': price_power_correlation,
        'avg_night_price': avg_night_price,
        'avg_day_price': avg_day_price,
        'night_charges_energy_kwh': night_charges_energy_kwh,
        'day_charges_energy_kwh': day_charges_energy_kwh,
        'night_day_charge_ratio': night_day_charge_ratio,
        'avg_soc': avg_soc,
        'min_soc': min_soc,
        'max_soc': max_soc,
        'pct_time_in_preferred_range': pct_time_in_preferred_range,
        'pct_time_at_high_soc': pct_time_at_high_soc,
        'violation_percentage': violation_percentage,
        'top3_peaks': top3_peaks,
        'peak_rolling_average': peak_rolling_average,
        'current_capacity_fee': current_capacity_fee,
        'fixed_grid_fee_total': fixed_grid_fee_total,
        'months_in_episode': months_in_episode,
        'peak_reduction_due_to_discount': peak_reduction_due_to_discount,
        'estimated_savings': estimated_savings,
        'night_peaks': night_peaks,
        'day_peaks': day_peaks,
        'actions_modified': num_modified_actions,
        'pct_modified_actions': pct_modified_actions,
        'avg_action_delta': avg_action_delta,
        'max_action_delta': max_action_delta,
        'reward_components': reward_components,
        'reward_component_distributions': component_distributions
    }

def load_agent(model_path, env, config):
    """
    Load a recurrent agent from a saved model.
    
    Args:
        model_path: Path to the saved model
        env: Environment to use
        config: Configuration dict
        
    Returns:
        Agent: The loaded agent
    """
    # Add debug prints
    print(f"Attempting to load agent from: {model_path}")
    print(f"Model directory exists: {os.path.exists(os.path.dirname(model_path))}")
    print(f"Model file exists: {os.path.exists(model_path)}")
    
    # Always flatten the observation space for RecurrentPPO
    if isinstance(env.observation_space, gym.spaces.Dict):
        print("Flattening dictionary observation space for RecurrentPPO")
        env = FlattenObservation(env)
    
    try:
        agent = RecurrentEnergyAgent(env=env, model_path=model_path, config=config)
        print(f"Loaded recurrent agent from {model_path}")
        return agent
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
    
    # Ensure data augmentation is disabled for evaluation
    # This is important to evaluate the agent on real, non-augmented data
    config["use_data_augmentation"] = False
    
    # Create environment
    base_env = HomeEnergyEnv(config=config)
    
    # Always use FlattenObservation wrapper for recurrent agents
    print("Using FlattenObservation wrapper for recurrent agent")
    env = FlattenObservation(base_env)
    
    # Reset environment
    obs, info = env.reset()
    
    episode_data = []
    terminated = False
    truncated = False
    
    # Track recurrent state
    lstm_state = None
    episode_start = True
    
    # Get the unwrapped environment for accessing attributes
    unwrapped_env = env.unwrapped
    
    # Run episode
    while not (terminated or truncated):
        # Get action from agent (always using recurrent agent predict)
        action, lstm_state = agent.predict(
            obs, 
            state=lstm_state,
            episode_start=episode_start,
            deterministic=True
        )
        episode_start = False
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Extract reward components directly from info['reward_components']
        reward_components = info.get('reward_components', {})
        
        # Store data for this step
        step_data = {
            'timestamp': unwrapped_env.start_datetime + timedelta(hours=unwrapped_env.current_step * unwrapped_env.time_step_hours),
            'soc': info.get('soc', 0),  # For recurrent agents, get from info
            'action': action[0],  # Extract scalar from array
            'reward': reward,
            'current_price': info.get('current_price', 0),
            'power_kw': info.get('power_kw', 0),
            'grid_power_kw': info.get('grid_power_kw', 0),
            'base_demand_kw': info.get('base_demand_kw', 0),
            'current_solar_production_kw': info.get('current_solar_production_kw', 0),
            'is_night_discount': info.get('is_night_discount', False),
            # New fields for action masking information
            'action_modified': info.get('action_modified', False),
            'original_action': info.get('original_action', action[0]),
            'safe_action': info.get('safe_action', action[0]),
        }
        
        # Add capacity related metrics if available
        for capacity_key in ['top3_peaks', 'peak_rolling_average', 'current_capacity_fee']:
            if capacity_key in info:
                step_data[capacity_key] = info[capacity_key]
        
        # Add price average metrics if available
        for price_key in ['price_24h_avg', 'price_168h_avg']:
            if price_key in info:
                step_data[price_key] = info[price_key]
        
        # Add all reward components from the info dictionary
        for key, value in reward_components.items():
            step_data[key] = value
        
        episode_data.append(step_data)
    
    print(f"Episode completed with {len(episode_data)} steps")
    return episode_data

def analyze_reward_component_distributions(episode_data):
    """
    Analyze the statistical distribution of reward components to understand
    their natural ranges before weighting.
    
    Args:
        episode_data: List of dictionaries containing episode data
        
    Returns:
        Dictionary with analysis results for each reward component
    """
    import numpy as np
    
    # Active reward components we want to analyze
    components = [
        'reward_grid_cost',
        'reward_capacity_penalty',
        'reward_battery_cost',
        'reward_soc_reward',
        'reward_shaping',
        'reward_arbitrage_bonus', 
        'reward_export_bonus',
        'reward_night_charging',
        'reward_action_mod_penalty',
        'reward_morning_soc',    
        'reward_night_peak_chain'
    ]
    
    results = {}
    
    # Extract all values for each component
    for component in components:
        values = [data.get(component, 0) for data in episode_data]
        if not values or all(v == 0 for v in values):
            results[component] = {
                'min': 0, 'max': 0, 'mean': 0, 'median': 0, 
                'std': 0, 'p5': 0, 'p25': 0, 'p75': 0, 'p95': 0,
                'zero_pct': 100.0, 'total': 0
            }
            continue
            
        # Calculate distribution statistics
        non_zero_values = [v for v in values if v != 0]
        total = sum(values)
        
        stats = {
            'min': min(values),
            'max': max(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'p5': np.percentile(values, 5),
            'p25': np.percentile(values, 25),
            'p75': np.percentile(values, 75),
            'p95': np.percentile(values, 95),
            'zero_pct': (len(values) - len(non_zero_values)) / len(values) * 100 if values else 0,
            'total': total
        }
        results[component] = stats
    
    return results

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Evaluate a trained RL agent in the home energy management environment")
        
        # Define command-line arguments
        parser.add_argument("--model-path", type=str, 
                           help="Path to the trained model. If not provided, uses the default model path.")
        parser.add_argument("--render", action="store_true", 
                           help="Render the environment during evaluation")
        parser.add_argument("--sanity-check", action="store_true", 
                           help="Run sanity checks on the environment before evaluation")
        parser.add_argument("--sanity-check-only", action="store_true", 
                           help="Run only sanity checks without evaluation")
        parser.add_argument("--sanity-check-steps", type=int, default=100, 
                           help="Number of steps to run for sanity checks")
        parser.add_argument("--num-episodes", type=int, default=1, 
                           help="Number of evaluation episodes to run")
        parser.add_argument("--start-month", type=str, 
                           help="Month to evaluate (format: YYYY-MM)")
        parser.add_argument("--random-start", action="store_true", 
                           help="Use random start dates for evaluation")
        
        # Parse arguments
        args = parser.parse_args()
        
        # Load configuration
        config = get_config_dict()
        
        # Process month selection
        if args.start_month:
            try:
                # Parse YYYY-MM format
                year_month = args.start_month.split('-')
                if len(year_month) != 2:
                    raise ValueError("Invalid format. Use YYYY-MM format.")
                    
                start_year = int(year_month[0])
                start_month = int(year_month[1])
                
                if start_month < 1 or start_month > 12:
                    raise ValueError(f"Invalid month: {start_month}. Month must be between 1 and 12.")
                    
                from calendar import monthrange
                days_in_month = monthrange(start_year, start_month)[1]
                
                print(f"Using specified month for evaluation: {args.start_month} ({days_in_month} days)")
                
                # Set configuration for specific month
                config["simulation_days"] = days_in_month
                config["force_specific_start_month"] = True
                config["start_year"] = start_year
                config["start_month"] = start_month
                
            except Exception as e:
                print(f"Error parsing start-month argument: {e}")
                print("Please use format YYYY-MM (e.g., 2023-06)")
                sys.exit(1)
                
        elif args.random_start:
            print("Using random episode start dates for evaluation")
            # Disable the force_specific_start_month to allow random starts
            config["force_specific_start_month"] = False
            
        else:
            # For now, skip random month selection logic
            print("Using default month selection")
        
        if args.render:
            config["render_mode"] = "human"
        
        # Determine the model path
        if args.model_path:
            model_path = args.model_path
        else:
            # Always use the default recurrent model path
            # grabbign the latest model
            print(f"Using default recurrent model path")
            latest_model = max(os.listdir(config['model_dir']), key=lambda x: os.path.getmtime(os.path.join(config['model_dir'], x)))
            model_path = os.path.join(config['model_dir'], latest_model)
        
        print(f"Model path: {model_path}")
        print(f"Model exists: {os.path.exists(model_path)}")
        
        # Run sanity checks if requested
        if args.sanity_check or args.sanity_check_only:
            from src.rl.train import run_sanity_checks
            print(f"\n{'='*80}\nRunning sanity checks...\n{'='*80}")
            env = HomeEnergyEnv(config=config)
            run_sanity_checks(env, config, num_steps=args.sanity_check_steps)
            
            if args.sanity_check_only:
                print("Sanity checks complete. Exiting as requested.")
                sys.exit(0)
        
        # Create environment first
        env = HomeEnergyEnv(config=config)
        
        # Pass the environment to load_agent
        agent = load_agent(model_path, env, config)
        
        if agent is None:
            print(f"Failed to load agent from {model_path}")
            sys.exit(1)

        # Run multiple evaluation episodes if requested
        all_results = []
        
        for episode in range(args.num_episodes):
            print(f"\n{'='*80}\nRunning evaluation episode {episode+1}/{args.num_episodes}\n{'='*80}")
            
            # Evaluate the agent
            if "start_year" in config and "start_month" in config:
                print(f"Evaluating agent for {config['start_year']}-{config['start_month']:02d}...")
            else:
                print("Evaluating agent with random start date...")
                
            episode_data = evaluate_episode(agent, config)
            
            # Create directory for plots
            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
            date_str = ""
            if "start_year" in config and "start_month" in config:
                date_str = f"_{config['start_year']}_{config['start_month']:02d}"
                
            plot_dir = f"src/rl/simulations/results/{timestamp}{date_str}_episode{episode+1}"
            os.makedirs(plot_dir, exist_ok=True)
            
            # Plot the performance
            fig, axs = plot_agent_performance(episode_data, model_name=os.path.basename(model_path), save_dir=plot_dir)
            
            # Plot reward components separately
            plot_reward_components(episode_data, output_path=os.path.join(plot_dir, "reward_components.png"))
            
            # Calculate and print metrics
            print(f"\n{'='*30} Performance Metrics {'='*30}")
            results = calculate_performance_metrics(episode_data, config)
            all_results.append(results)
            
            # Save the results to a JSON file
            results_file = os.path.join(plot_dir, "metrics.json")
            with open(results_file, 'w') as f:
                import json
                
                # Convert any numpy types to native Python types
                def convert_to_serializable(obj):
                    if isinstance(obj, dict):
                        return {k: convert_to_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_to_serializable(i) for i in obj]
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return convert_to_serializable(obj.tolist())
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    else:
                        return obj
                
                serializable_results = convert_to_serializable(results)
                json.dump(serializable_results, f, indent=4)
                
            print(f"Metrics saved to {results_file}")
            
        # If we ran multiple episodes, print average metrics
        if len(all_results) > 1:
            print(f"\n{'='*30} Average Metrics Across {len(all_results)} Episodes {'='*30}")
            # Calculate and print average metrics
            # TODO: Implement averaging logic for metrics
            
    except Exception as e:
        import traceback
        print(f"Error during evaluation: {e}")
        traceback.print_exc() 