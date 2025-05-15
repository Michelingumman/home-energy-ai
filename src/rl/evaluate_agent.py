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


def evaluate_model(model_path: str, config_path: str, num_episodes: int = 1, eval_simulation_days: int = 3):
    """
    Load a trained model and evaluate its performance, collecting data for plotting.

    Args:
        model_path: Path to the saved PPO model (.zip file).
        config_path: Path to the RL configuration JSON file.
        num_episodes: Number of episodes to run for evaluation.
        eval_simulation_days: Number of simulation days for each evaluation episode.
    """
    print(f"Debug: Starting evaluation with model path: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        # Fallback to a default or expected config path if main one is missing
        # For now, we require it.
        return

    print(f"Debug: Loading config from {config_path}")
    config_dict = rl_config.get_config_dict() # New way
    print(f"Debug: Config loaded successfully")

    # Prepare the environment configuration
    env_config = {
        "battery_capacity": config_dict.get("battery_capacity", 22.0),
        "simulation_days": eval_simulation_days, # Use specific eval days
        "peak_penalty_factor": config_dict.get("peak_penalty_factor", 10.0),
        "use_price_predictions": config_dict.get("use_price_predictions_eval", True),
        "price_predictions_path": config_dict.get("price_predictions_path", "data/processed/SE3prices.csv"), # Adjusted default
        "fixed_baseload_kw": config_dict.get("fixed_baseload_kw", 0.5),
        "render_mode": None, # No rendering during data collection for speed
        "time_step_minutes": config_dict.get("time_step_minutes", 15), # Read from main config
        "use_variable_consumption": config_dict.get("use_variable_consumption", False),
        "consumption_data_path": config_dict.get("consumption_data_path", None),
        "battery_degradation_cost_per_kwh": config_dict.get("battery_degradation_cost_per_kwh", 45.0)
    }
    
    # Merge with main config to ensure reward parameters are included
    eval_config = config_dict.copy()
    eval_config.update(env_config)
    
    print(f"Debug: Environment config prepared: {env_config}")
    
    # If price predictions are used, ensure the path is correct relative to project root
    if env_config["use_price_predictions"]:
        if not os.path.isabs(env_config["price_predictions_path"]):
            env_config["price_predictions_path"] = os.path.join(PROJECT_ROOT, env_config["price_predictions_path"])
        
        # Add additional verification that the price predictions file exists
        price_file_path = env_config["price_predictions_path"]
        if not os.path.exists(price_file_path):
            print(f"WARNING: Price predictions file not found at {price_file_path}")
            print(f"Checking if file exists in absolute path: {os.path.isabs(price_file_path)}")
            print(f"Checking if file exists in relative path: {os.path.exists(os.path.join(os.getcwd(), os.path.basename(price_file_path)))}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"PROJECT_ROOT: {PROJECT_ROOT}")
            # Try to list files in the directory to help debugging
            parent_dir = os.path.dirname(price_file_path)
            if os.path.exists(parent_dir):
                print(f"Files in {parent_dir}:")
                for f in os.listdir(parent_dir):
                    print(f"  {f}")
            else:
                print(f"Parent directory {parent_dir} does not exist")

    # If variable consumption is used, ensure the path is correct relative to project root
    if env_config.get("use_variable_consumption") and env_config.get("consumption_data_path") and not os.path.isabs(env_config["consumption_data_path"]):
        env_config["consumption_data_path"] = os.path.join(PROJECT_ROOT, env_config["consumption_data_path"])

    print(f"Creating evaluation environment with {eval_simulation_days} simulation days.")
    eval_env = HomeEnergyEnv(config=eval_config)
    
    # Patch the observation space if needed to match the model's expected format
    # This is necessary if the model was trained with a 2D time_idx format instead of 3D
    # Check if we need to adjust the observation space
    try:
        print(f"Loading model from {model_path}...")
        model = PPO.load(model_path, env=eval_env)
    except ValueError as e:
        if "Observation spaces do not match" in str(e):
            print("Observation spaces mismatch detected. Attempting to fix...")
            # Override the _get_observation method to return the expected format
            original_get_observation = eval_env._get_observation
            
            def patched_get_observation():
                """Patch to return observation compatible with the trained model."""
                obs = original_get_observation()
                # Modify time_idx to match the 2D format expected by the model
                # Keep only hour of day and day of week, drop minute information
                current_dt = eval_env.start_datetime + datetime.timedelta(hours=eval_env.current_step * eval_env.time_step_hours)
                hour_of_day = current_dt.hour
                day_of_week = current_dt.weekday()  # Monday=0, Sunday=6
                
                # Replace time_idx with 2D integer array
                obs["time_idx"] = np.array([hour_of_day, day_of_week], dtype=np.int32)
                return obs
            
            # Replace the method
            eval_env._get_observation = patched_get_observation
            
            # Now try loading the model again
            print("Retrying model loading with patched observation space...")
            model = PPO.load(model_path, env=eval_env)
        else:
            # If it's a different error, re-raise it
            raise

    all_episode_data = []

    for episode in range(num_episodes):
        print(f"\n--- Starting Evaluation Episode {episode + 1}/{num_episodes} ---")
        obs, info = eval_env.reset()
        terminated = False
        truncated = False
        
        episode_data = {
            "timestamps": [],
            "soc": [],
            "price": [],
            "action_normalized": [],
            "battery_power_kw": [],
            "grid_power_kw": [],
            "reward": [],
            "total_cost_cumulative": [],
            "household_consumption_kw": [],
            # Add keys for individual reward components
            "grid_cost_term": [],
            "peak_penalty_term": [],
            "battery_cost_term": [],
            "arbitrage_bonus_term": [],
            "soc_action_penalty_term": [],
            "current_solar_production_kw": [], # Added for solar production
            "is_night_discount": []
        }
        
        total_steps = eval_env.simulation_steps
        for step_num in range(total_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)

            # Collect data
            current_sim_time = eval_env.start_datetime + pd.Timedelta(hours=eval_env.current_step * eval_env.time_step_hours)
            
            episode_data["timestamps"].append(current_sim_time)
            episode_data["soc"].append(obs["soc"][0]) # SoC from new observation
            episode_data["price"].append(info.get("current_price", np.nan))
            episode_data["action_normalized"].append(float(action[0] if isinstance(action, np.ndarray) and action.ndim > 0 else action))
            
            # Store battery power with inverted sign for plotting consistency
            # In the environment: positive = discharge, negative = charge
            # For plotting: positive = charge (SOC increases), negative = discharge (SOC decreases)
            battery_power = info.get("power_kw", np.nan)
            episode_data["battery_power_kw"].append(-battery_power if not np.isnan(battery_power) else np.nan)  # Invert the sign
            
            episode_data["grid_power_kw"].append(info.get("grid_power_kw", np.nan))
            episode_data["reward"].append(reward)
            episode_data["total_cost_cumulative"].append(info.get("total_cost", np.nan))
            episode_data["household_consumption_kw"].append(info.get("base_demand_kw", np.nan)) # base_demand_kw is gross consumption
            episode_data["current_solar_production_kw"].append(info.get("current_solar_production_kw", np.nan))
            episode_data["is_night_discount"].append(info.get("is_night_discount", False))

            episode_data["grid_cost_term"].append(info.get("reward_grid_cost", np.nan))
            episode_data["peak_penalty_term"].append(info.get("reward_peak_penalty", np.nan))
            episode_data["battery_cost_term"].append(info.get("reward_battery_cost", np.nan))
            episode_data["arbitrage_bonus_term"].append(info.get("reward_arbitrage_bonus", np.nan))
            episode_data["soc_action_penalty_term"].append(info.get("reward_soc_action_penalty", np.nan))

            # Calculate steps per day based on time step duration
            steps_per_day = int(24 / eval_env.time_step_hours)
            if (step_num + 1) % steps_per_day == 0:
                print(f"  Simulated day {(step_num + 1) // steps_per_day} complete.")

            if terminated or truncated:
                print(f"Episode finished after {step_num + 1} steps.")
                break
        
        all_episode_data.append(pd.DataFrame(episode_data))
        print(f"--- Finished Evaluation Episode {episode + 1}/{num_episodes} ---")
        print(f"  Total reward: {sum(episode_data['reward']):.2f}")
        print(f"  Final total cost: {episode_data['total_cost_cumulative'][-1]:.2f}")
        print(f"  Peak grid power: {eval_env.peak_power:.2f} kW")


    # Plotting
    if not all_episode_data:
        print("No data collected for plotting.")
        return

    # For simplicity, plot the first episode
    # In a real scenario, you might average or select specific episodes
    df_to_plot = all_episode_data[0]
    
    # Save the detailed data to a CSV for analysis
    csv_dir = os.path.join(PROJECT_ROOT, "src", "rl", "simulations", "results")
    os.makedirs(csv_dir, exist_ok=True)
    csv_filename = os.path.join(csv_dir, f"evaluation_data_{Path(model_path).stem}.csv")
    df_to_plot.to_csv(csv_filename, index=False)
    print(f"\nSaved detailed evaluation data to {csv_filename}")

    # --- Create a single Figure with 4 subplots ---
    fig, axs = plt.subplots(4, 1, figsize=(20, 22), sharex=True)
    fig.suptitle(f'Agent Performance Analysis - Model: {Path(model_path).name}', fontsize=18, weight='bold')

    # Common x-axis date formatter and locator
    date_formatter = mdates.DateFormatter('%Y-%m-%d %H:%M')
    # Place a tick every 6 hours for better readability over multiple days
    hour_locator = mdates.HourLocator(interval=6) 

    # --- Subplot 1: SoC and Price vs. Time ---
    ax1 = axs[0]
    color_soc = 'crimson' # Changed color
    ax1.set_ylabel('SoC', color=color_soc, fontsize=12)
    ax1.step(df_to_plot['timestamps'], df_to_plot['soc'], color=color_soc, label='Battery SoC', linewidth=2.5)
    ax1.tick_params(axis='y', labelcolor=color_soc, labelsize=10)
    ax1.set_ylim(-0.05, 1.05) # Slight padding
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax1_twin = ax1.twinx()
    color_price = 'dodgerblue' # Changed color
    ax1_twin.set_ylabel('Electricity Price (öre/kWh)', color=color_price, fontsize=12)
    ax1_twin.step(df_to_plot['timestamps'], df_to_plot['price'], color=color_price, linestyle='--', label='Electricity Price', linewidth=2)
    ax1_twin.tick_params(axis='y', labelcolor=color_price, labelsize=10)
    ax1.set_title('Battery SoC and Electricity Price', fontsize=14, weight='semibold')
    ax1.axhline(0, color='gray', linewidth=0.8, linestyle='-') # Add a zero line for reference

    
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles1_twin, labels1_twin = ax1_twin.get_legend_handles_labels()
    ax1.legend(handles1 + handles1_twin, labels1 + labels1_twin, loc='upper right', fontsize=10)

    # --- Subplot 2: Agent Action (Normalized) and Actual Battery Power ---
    ax2 = axs[1]
    color_action = 'forestgreen' # Changed color
    ax2.set_ylabel('Normalized Action (-1 to 1)', color=color_action, fontsize=12)
    ax2.step(df_to_plot['timestamps'], df_to_plot['action_normalized'], color=color_action, label='Agent Action (Normalized)', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color_action, labelsize=10)
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.axhline(0, color='gray', linewidth=0.8, linestyle='-') # Add a zero line for reference


    ax2_twin = ax2.twinx()
    color_battery_power = 'darkviolet' # Changed color
    ax2_twin.set_ylabel('Battery Power (kW)', color=color_battery_power, fontsize=12) 
    ax2_twin.step(df_to_plot['timestamps'], df_to_plot['battery_power_kw'], color=color_battery_power, linestyle='--', label='Battery Power (kW)', linewidth=2)
    ax2_twin.tick_params(axis='y', labelcolor=color_battery_power, labelsize=10)
    ax2_twin.axhline(0, color='gray', linewidth=0.8, linestyle='-') # Add a zero line for reference
    ax2.set_title('Agent Actions and Battery Power (Plot: Positive Power = Charging)', fontsize=14, weight='semibold')
    
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles2_twin, labels2_twin = ax2_twin.get_legend_handles_labels()
    ax2.legend(handles2 + handles2_twin, labels2 + labels2_twin, loc='upper right', fontsize=10)

    # --- Subplot 3: Household Consumption, Grid Power, and Solar Production ---
    ax3 = axs[2]
    ax3.step(df_to_plot['timestamps'], df_to_plot['household_consumption_kw'], color='deepskyblue', linestyle='-', label='Household Consumption (kW)', linewidth=2.5) # Changed color & style
    ax3.step(df_to_plot['timestamps'], df_to_plot['grid_power_kw'], color='salmon', label='Net Grid Power (kW)', linewidth=2.5) # Changed color
    ax3.step(df_to_plot['timestamps'], df_to_plot['current_solar_production_kw'], color='darkorange', linestyle='-', label='Solar Production (kW)', linewidth=2.5)
    ax3.set_ylabel('Power (kW)', fontsize=12)
    ax3.set_title('Household Consumption, Net Grid Power, and Solar Production', fontsize=14, weight='semibold')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.axhline(0, color='gray', linewidth=0.8, linestyle='-') # Add a zero line for reference
    
    # --- Subplot 4: Cumulative Reward ---
    ax4 = axs[3]
    cumulative_reward = pd.Series(df_to_plot['reward']).cumsum()
    ax4.plot(df_to_plot['timestamps'], cumulative_reward, color='saddlebrown', label='Cumulative Reward', linewidth=2.5) # Changed color
    ax4.set_ylabel('Cumulative Reward', fontsize=12)
    ax4.set_title('Cumulative Reward Over Episode', fontsize=14, weight='semibold')
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='lower left', fontsize=10)

    # Format x-axis for all subplots
    for ax in axs:
        ax.xaxis.set_major_formatter(date_formatter)
        ax.xaxis.set_major_locator(hour_locator) # Set the locator
        ax.tick_params(axis='x', rotation=30, labelsize=10)
    axs[-1].set_xlabel('Time (Year-Month-Day Hour:Minute)', fontsize=12) # Set x-label only on the last subplot
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to make room for suptitle

    # Save the combined plot
    plot_dir = os.path.join(PROJECT_ROOT, "src", "rl", "simulations", "results")
    os.makedirs(plot_dir, exist_ok=True)
    plot_filename = os.path.join(plot_dir, f"evaluation_combined_plot_{Path(model_path).stem}.png")
    fig.savefig(plot_filename)
    print(f"\nSaved combined analysis plot to {plot_filename}")

    plt.show()


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


# def plot_reward_components(episode_data, model_name=None, save_dir=None):
#     """
#     Create a plot showing the different reward components over time.
    
#     Args:
#         episode_data: List of dictionaries containing episode data
#         model_name: Optional name of the model for the plot title
#         save_dir: Optional directory to save the plot
#     """
#     # Extract data
#     timestamps = [data['timestamp'] for data in episode_data]
#     rewards = [data['reward'] for data in episode_data]
#     grid_cost_terms = [data.get('reward_grid_cost', 0) for data in episode_data]
#     peak_penalty_terms = [data.get('reward_peak_penalty', 0) for data in episode_data]
#     battery_cost_terms = [data.get('reward_battery_cost', 0) for data in episode_data]
#     arbitrage_bonus_terms = [data.get('reward_arbitrage_bonus', 0) for data in episode_data]
#     soc_action_penalty_terms = [data.get('reward_soc_action_penalty', 0) for data in episode_data]
    
#     # Calculate cumulative total cost
#     total_costs = np.cumsum([-r for r in grid_cost_terms])
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=(15, 8))
    
#     # Set title
#     if model_name:
#         plt.title(f"Reward Components - Model: {model_name}", fontsize=16)
#     else:
#         plt.title("Reward Components", fontsize=16)
    
#     # Create a plot with different y-scales for each component
#     ax.set_xlabel('Timestamp')
#     ax.set_ylabel('Reward')
    
#     # Plot total reward
#     ax.plot(timestamps, rewards, 'b-', label='reward', linewidth=2)
    
#     # Plot cumulative total cost on a separate axis
#     ax_total = ax.twinx()
#     ax_total.plot(timestamps, total_costs, 'orange', label='total_cost_cumulative', linewidth=2)
#     ax_total.set_ylabel('Cumulative Cost', color='orange')
    
#     # Plot grid cost term on another axis
#     ax_grid = ax.twinx()
#     ax_grid.spines['right'].set_position(('outward', 60))
#     ax_grid.plot(timestamps, grid_cost_terms, 'g-', label='grid_cost_term', linewidth=1.5)
#     ax_grid.set_ylabel('Grid Cost Term', color='green')
    
#     # Plot peak penalty term on another axis
#     ax_peak = ax.twinx()
#     ax_peak.spines['right'].set_position(('outward', 120))
#     ax_peak.plot(timestamps, peak_penalty_terms, 'r-', label='peak_penalty_term', linewidth=1.5)
#     ax_peak.set_ylabel('Peak Penalty Term', color='red')
    
#     # Plot battery cost term on another axis
#     ax_battery = ax.twinx()
#     ax_battery.spines['right'].set_position(('outward', 180))
#     ax_battery.plot(timestamps, battery_cost_terms, 'c-', label='battery_cost_term', linewidth=1.5)
#     ax_battery.set_ylabel('Battery Cost Term', color='cyan')
    
#     # Plot arbitrage bonus term on another axis
#     ax_arb = ax.twinx()
#     ax_arb.spines['right'].set_position(('outward', 240))
#     ax_arb.plot(timestamps, arbitrage_bonus_terms, 'm-', label='arbitrage_bonus_term', linewidth=1.5)
#     ax_arb.set_ylabel('Arbitrage Bonus Term', color='magenta')
    
#     # Plot SoC action penalty term on another axis
#     ax_soc = ax.twinx()
#     ax_soc.spines['right'].set_position(('outward', 300))
#     ax_soc.plot(timestamps, soc_action_penalty_terms, 'y-', label='soc_action_penalty_term', linewidth=1.5)
#     ax_soc.set_ylabel('SoC Action Penalty Term', color='yellow')
    
#     # Add legend with all lines
#     lines, labels = ax.get_legend_handles_labels()
#     lines2, labels2 = ax_total.get_legend_handles_labels()
#     lines3, labels3 = ax_grid.get_legend_handles_labels()
#     lines4, labels4 = ax_peak.get_legend_handles_labels()
#     lines5, labels5 = ax_battery.get_legend_handles_labels()
#     lines6, labels6 = ax_arb.get_legend_handles_labels()
#     lines7, labels7 = ax_soc.get_legend_handles_labels()
    
#     ax.legend(lines + lines2 + lines3 + lines4 + lines5 + lines6 + lines7,
#               labels + labels2 + labels3 + labels4 + labels5 + labels6 + labels7,
#               loc='upper left')
    
#     plt.tight_layout()
#     plt.grid(True, alpha=0.3)
    
#     # Add a title above the plot that indicates all series have independent y-scales
#     plt.figtext(0.5, 0.99, "All series, independent y-scales", 
#                 ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
#     # Save figure if a directory is specified
#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)
#         filename = os.path.join(save_dir, f"reward_components_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
#         plt.savefig(filename, dpi=300, bbox_inches='tight')
#         print(f"Reward components plot saved to {filename}")
    
#     return fig

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
    print(f"Peak grid export: {peak_grid_export:.2f} kW")
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
    # plot_reward_components(episode_data, model_name=args.model, save_dir=args.plot_dir)
    
    # Calculate and print metrics
    print("\n--- Performance Metrics ---")
    calculate_performance_metrics(episode_data)
    
    plt.show()  # Show all plots 