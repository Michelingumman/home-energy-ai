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

# Add the project root directory (one level up from 'src') to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from stable_baselines3 import PPO
from src.rl.custom_env import HomeEnergyEnv # Assumes evaluate_agent.py is in project root
from src.rl import config as rl_config # Import the new config


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
    
    # Calculate cumulative reward
    cum_rewards = np.cumsum(rewards)
    
    # Create figure with subplots
    fig, axs = plt.subplots(4, 1, figsize=(15, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1]})
    
    # Set title based on model name if provided
    if model_name:
        fig.suptitle(f"Agent Performance Analysis - Model: {model_name}", fontsize=16)
    else:
        fig.suptitle("Agent Performance Analysis", fontsize=16)
    
    # Subplot 1: Battery SoC and Electricity Price
    ax1 = axs[0]
    ax1.set_title("Battery SoC and Electricity Price")
    ax1.set_ylabel("SoC")
    
    # Plot SoC
    ax1.plot(timestamps, soc_values, 'r-', label="Battery SoC")
    ax1.set_ylim(0, 1.1)
    
    # Plot Price on secondary y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Electricity Price (öre/kWh)")
    
    # Add shaded regions for night discount periods
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
    
    # Draw shaded regions for night discount periods
    for start, end in night_periods:
        if start < len(timestamps) and end < len(timestamps):
            ax2.axvspan(timestamps[start], timestamps[end], alpha=0.2, color='blue', label='Night Discount' if start == night_periods[0][0] else "")
    
    # Plot prices
    ax2.plot(timestamps, prices, 'b--', label="Electricity Price")
    
    # Create combined legend for both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    
    # Subplot 2: Agent Actions and Battery Power
    ax3 = axs[1]
    ax3.set_title("Agent Actions and Battery Power (Plot: Positive Power = Charging)")
    ax3.set_ylabel("Action (-1 to 1)")
    ax3.plot(timestamps, actions, 'g-', label="Agent Action (Normalized)")
    ax3.set_ylim(-1.1, 1.1)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Plot Battery Power on secondary y-axis
    ax4 = ax3.twinx()
    ax4.set_ylabel("Battery Power (kW)")
    # Note that in our convention, negative battery power means charging
    ax4.plot(timestamps, battery_powers, 'm--', label="Battery Power (kW)")
    
    # Create combined legend for both axes
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    lines_4, labels_4 = ax4.get_legend_handles_labels()
    ax3.legend(lines_3 + lines_4, labels_3 + labels_4, loc='upper right')
    
    # Subplot 3: Household Consumption, Grid Power, and Solar Production
    ax5 = axs[2]
    ax5.set_title("Household Consumption, Net Grid Power, and Solar Production")
    ax5.set_ylabel("Power (kW)")
    ax5.plot(timestamps, base_demands, 'c-', label="Household Consumption (kW)")
    ax5.plot(timestamps, grid_powers, 'r-', label="Net Grid Power (kW)")
    ax5.plot(timestamps, solar_productions, 'orange', label="Solar Production (kW)")
    ax5.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax5.legend(loc='upper right')
    
    # Subplot 4: Cumulative Reward
    ax6 = axs[3]
    ax6.set_title("Cumulative Reward Over Episode")
    ax6.set_ylabel("Reward")
    ax6.plot(timestamps, cum_rewards, 'brown', label="Cumulative Reward")
    ax6.legend(loc='upper left')
    
    # Improved x-axis formatting for all subplots
    # Import required formatter if not already at the top
    import matplotlib.dates as mdates
    
    # Format the x-axis with better date/time display
    for ax in axs:
        # Major ticks - Show date at midnight
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        
        # Minor ticks - Show hours at 6-hour intervals
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
        
        # Ensure both major and minor ticks are visible
        ax.tick_params(which='both', axis='x', rotation=0)
        ax.tick_params(which='major', length=7, labelsize=9)
        ax.tick_params(which='minor', length=4, labelsize=8)
        ax.grid(True, which='major', axis='x', linestyle='-', alpha=0.7)
        ax.grid(True, which='minor', axis='x', linestyle=':', alpha=0.4)
    
    # Add a clear x-axis label for the bottom subplot
    axs[3].set_xlabel("Date/Time", fontsize=12)
    
    # Add vertical lines at midnight for better visual separation of days
    min_date = min(timestamps).replace(hour=0, minute=0, second=0)
    max_date = max(timestamps).replace(hour=0, minute=0, second=0) + pd.Timedelta(days=1)
    current_date = min_date + pd.Timedelta(days=1)
    
    while current_date < max_date:
        for ax in axs:
            ax.axvline(current_date, color='gray', alpha=0.5, linestyle='--', linewidth=0.8)
        current_date += pd.Timedelta(days=1)
    
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
    Create a plot showing the different reward components over time.
    
    Args:
        episode_data: List of dictionaries containing episode data
        model_name: Optional name of the model for the plot title
        save_dir: Optional directory to save the plot
    """
    # Extract data
    timestamps = [data['timestamp'] for data in episode_data]
    rewards = [data['reward'] for data in episode_data]
    grid_cost_terms = [data.get('reward_grid_cost', 0) for data in episode_data]
    peak_penalty_terms = [data.get('reward_peak_penalty', 0) for data in episode_data]
    battery_cost_terms = [data.get('reward_battery_cost', 0) for data in episode_data]
    arbitrage_bonus_terms = [data.get('reward_arbitrage_bonus', 0) for data in episode_data]
    soc_action_penalty_terms = [data.get('reward_soc_action_penalty', 0) for data in episode_data]
    
    # Calculate cumulative total cost
    total_costs = np.cumsum([-r for r in grid_cost_terms])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Set title
    if model_name:
        plt.title(f"Reward Components - Model: {model_name}", fontsize=16)
    else:
        plt.title("Reward Components", fontsize=16)
    
    # Create a plot with different y-scales for each component
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Reward')
    
    # Plot total reward
    ax.plot(timestamps, rewards, 'b-', label='reward', linewidth=2)
    
    # Plot cumulative total cost on a separate axis
    ax_total = ax.twinx()
    ax_total.plot(timestamps, total_costs, 'orange', label='total_cost_cumulative', linewidth=2)
    ax_total.set_ylabel('Cumulative Cost', color='orange')
    
    # Plot grid cost term on another axis
    ax_grid = ax.twinx()
    ax_grid.spines['right'].set_position(('outward', 60))
    ax_grid.plot(timestamps, grid_cost_terms, 'g-', label='grid_cost_term', linewidth=1.5)
    ax_grid.set_ylabel('Grid Cost Term', color='green')
    
    # Plot peak penalty term on another axis
    ax_peak = ax.twinx()
    ax_peak.spines['right'].set_position(('outward', 120))
    ax_peak.plot(timestamps, peak_penalty_terms, 'r-', label='peak_penalty_term', linewidth=1.5)
    ax_peak.set_ylabel('Peak Penalty Term', color='red')
    
    # Plot battery cost term on another axis
    ax_battery = ax.twinx()
    ax_battery.spines['right'].set_position(('outward', 180))
    ax_battery.plot(timestamps, battery_cost_terms, 'c-', label='battery_cost_term', linewidth=1.5)
    ax_battery.set_ylabel('Battery Cost Term', color='cyan')
    
    # Plot arbitrage bonus term on another axis
    ax_arb = ax.twinx()
    ax_arb.spines['right'].set_position(('outward', 240))
    ax_arb.plot(timestamps, arbitrage_bonus_terms, 'm-', label='arbitrage_bonus_term', linewidth=1.5)
    ax_arb.set_ylabel('Arbitrage Bonus Term', color='magenta')
    
    # Plot SoC action penalty term on another axis
    ax_soc = ax.twinx()
    ax_soc.spines['right'].set_position(('outward', 300))
    ax_soc.plot(timestamps, soc_action_penalty_terms, 'y-', label='soc_action_penalty_term', linewidth=1.5)
    ax_soc.set_ylabel('SoC Action Penalty Term', color='yellow')
    
    # Add legend with all lines
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_total.get_legend_handles_labels()
    lines3, labels3 = ax_grid.get_legend_handles_labels()
    lines4, labels4 = ax_peak.get_legend_handles_labels()
    lines5, labels5 = ax_battery.get_legend_handles_labels()
    lines6, labels6 = ax_arb.get_legend_handles_labels()
    lines7, labels7 = ax_soc.get_legend_handles_labels()
    
    ax.legend(lines + lines2 + lines3 + lines4 + lines5 + lines6 + lines7,
              labels + labels2 + labels3 + labels4 + labels5 + labels6 + labels7,
              loc='upper left')
    
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    # Add a title above the plot that indicates all series have independent y-scales
    plt.figtext(0.5, 0.99, "All series, independent y-scales", 
                ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    # Save figure if a directory is specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"reward_components_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Reward components plot saved to {filename}")
    
    return fig

def calculate_performance_metrics(episode_data):
    """
    Calculate and print various performance metrics from the episode data.
    
    Args:
        episode_data: List of dictionaries containing episode data
    """
    total_reward = sum(data['reward'] for data in episode_data)
    
    # Battery usage metrics
    battery_powers = [abs(data.get('power_kw', 0)) for data in episode_data]
    avg_battery_power = sum(battery_powers) / len(battery_powers) if battery_powers else 0
    max_battery_power = max(battery_powers) if battery_powers else 0
    
    # Grid metrics
    grid_powers = [data.get('grid_power_kw', 0) for data in episode_data]
    peak_grid_import = max(grid_powers) if grid_powers else 0
    peak_grid_export = min(grid_powers) if grid_powers else 0
    
    # Price response metrics
    prices = [data.get('current_price', 0) for data in episode_data]
    battery_powers = [data.get('power_kw', 0) for data in episode_data]
    
    # Calculate correlation between price and battery power
    # Positive correlation means discharging more at high prices (good)
    if len(prices) > 1 and len(battery_powers) > 1:
        price_power_correlation = np.corrcoef(prices, battery_powers)[0, 1]
    else:
        price_power_correlation = 0
    
    # Calculate night vs day metrics
    night_prices = [data.get('current_price', 0) for data in episode_data if data.get('is_night_discount', False)]
    day_prices = [data.get('current_price', 0) for data in episode_data if not data.get('is_night_discount', False)]
    
    night_charges = [data.get('power_kw', 0) for data in episode_data if data.get('is_night_discount', False) and data.get('power_kw', 0) < 0]
    day_charges = [data.get('power_kw', 0) for data in episode_data if not data.get('is_night_discount', False) and data.get('power_kw', 0) < 0]
    
    avg_night_price = sum(night_prices) / len(night_prices) if night_prices else 0
    avg_day_price = sum(day_prices) / len(day_prices) if day_prices else 0
    
    total_night_charge = sum([-p for p in night_charges]) * (episode_data[0].get('time_step_hours', 0.25) if episode_data else 0.25)
    total_day_charge = sum([-p for p in day_charges]) * (episode_data[0].get('time_step_hours', 0.25) if episode_data else 0.25)
    
    # Night vs day charging ratio (higher is better - charging more at night)
    night_day_charge_ratio = total_night_charge / total_day_charge if total_day_charge > 0 else float('inf')
    
    # Print results
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average battery power: {avg_battery_power:.2f} kW")
    print(f"Maximum battery power: {max_battery_power:.2f} kW")
    print(f"Peak grid import: {peak_grid_import:.2f} kW")
    print(f"Peak grid export: {abs(peak_grid_export):.2f} kW")
    print(f"Price-power correlation: {price_power_correlation:.4f} (higher is better)")
    print(f"Average night price: {avg_night_price:.2f} öre/kWh")
    print(f"Average day price: {avg_day_price:.2f} öre/kWh")
    print(f"Total charging at night: {total_night_charge:.2f} kWh")
    print(f"Total charging during day: {total_day_charge:.2f} kWh")
    print(f"Night/day charging ratio: {night_day_charge_ratio:.2f} (higher means more night charging)")
    
    return {
        "total_reward": total_reward,
        "avg_battery_power": avg_battery_power,
        "max_battery_power": max_battery_power,
        "peak_grid_import": peak_grid_import,
        "peak_grid_export": peak_grid_export,
        "price_power_correlation": price_power_correlation,
        "avg_night_price": avg_night_price,
        "avg_day_price": avg_day_price,
        "total_night_charge": total_night_charge, 
        "total_day_charge": total_day_charge,
        "night_day_charge_ratio": night_day_charge_ratio
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
        
        # Store data for this step
        step_data = {
            'timestamp': env.start_datetime + datetime.timedelta(hours=env.current_step * env.time_step_hours),
            'soc': obs['soc'],
            'action': action,
            'reward': reward,
            'current_price': info.get('current_price'),
            'power_kw': info.get('power_kw'),
            'grid_power_kw': info.get('grid_power_kw'),
            'base_demand_kw': info.get('base_demand_kw'),
            'current_solar_production_kw': info.get('current_solar_production_kw'),
            'is_night_discount': info.get('is_night_discount', False),
            # Add reward components
            'reward_grid_cost': info.get('reward_grid_cost'),
            'reward_peak_penalty': info.get('reward_peak_penalty'),
            'reward_battery_cost': info.get('reward_battery_cost'),
            'reward_arbitrage_bonus': info.get('reward_arbitrage_bonus'),
            'reward_soc_action_penalty': info.get('reward_soc_action_penalty')
        }
        
        episode_data.append(step_data)
    
    print(f"Episode completed with {len(episode_data)} steps")
    return episode_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a trained RL agent for home energy management")
    parser.add_argument("--model", type=str, default="short_term_agent_final", help="Model name to evaluate (default: short_term_agent_final)")
    parser.add_argument("--days", type=int, default=14, help="Number of days to simulate (default: 14)")
    parser.add_argument("--render", action="store_true", help="Whether to render the environment")
    parser.add_argument("--plot-dir", type=str, default="src/rl/simulations/results", help="Directory to save plots")
    
    args = parser.parse_args()
    
    # Get configuration from config.py
    from src.rl.config import get_config_dict
    config = get_config_dict()
    
    # Override simulation days with command line argument
    config["simulation_days"] = args.days
    if args.render:
        config["render_mode"] = "human"
    
    # Load the agent
    model_path = f"{config['model_dir']}/{args.model}"
    agent = load_agent(model_path)
    
    if agent is None:
        print(f"Failed to load agent from {model_path}")
        sys.exit(1)
    
    # Evaluate the agent
    print(f"Evaluating agent for {args.days} days...")
    episode_data = evaluate_episode(agent, config)
    
    # Plot the performance
    fig, axs = plot_agent_performance(episode_data, model_name=args.model, save_dir=args.plot_dir)
    
    # Plot reward components separately
    plot_reward_components(episode_data, model_name=args.model, save_dir=args.plot_dir)
    
    # Calculate and print metrics
    print("\n--- Performance Metrics ---")
    calculate_performance_metrics(episode_data)
    
    plt.show()  # Show all plots 