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
    # If evaluate_agent.py is in the root, this needs to be adjusted
    # Assuming evaluate_agent.py is in the same dir as train.py (e.g. src/rl) for now
    # If it's in project root, then 'src' needs to be added or this logic changed.
    # For now, let's assume it's in src/rl/
    # PROJECT_ROOT_FOR_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # if PROJECT_ROOT_FOR_SRC not in sys.path:
    #     sys.path.insert(0, PROJECT_ROOT_FOR_SRC)
    # Let's assume evaluate_agent.py is in the root of the project.
    sys.path.insert(0, PROJECT_ROOT)


from stable_baselines3 import PPO
from src.rl.custom_env import HomeEnergyEnv # Assumes evaluate_agent.py is in project root
from src.rl.train import load_config # To reuse config loading if needed


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
    config = load_config(config_path)
    print(f"Debug: Config loaded successfully")

    # Override simulation days for evaluation
    env_config = {
        "battery_capacity": config.get("battery_capacity", 22.0),
        "simulation_days": eval_simulation_days, # Use specific eval days
        "peak_penalty_factor": config.get("peak_penalty_factor", 10.0),
        "use_price_predictions": config.get("use_price_predictions_eval", True),
        "price_predictions_path": config.get("price_predictions_path", "data/processed/SE3prices.csv"), # Adjusted default
        "fixed_baseload_kw": config.get("fixed_baseload_kw", 0.5),
        "render_mode": None, # No rendering during data collection for speed
        "time_step_minutes": config.get("time_step_minutes", 15), # Read from main config
        "use_variable_consumption": config.get("use_variable_consumption", False),
        "consumption_data_path": config.get("consumption_data_path", None),
        "battery_degradation_cost_per_kwh": config.get("battery_degradation_cost_per_kwh", 45.0),
        "config": config # Pass the main config dict
    }
    
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
    eval_env = HomeEnergyEnv(**env_config)
    
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
            "soc_action_penalty_term": []
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
            episode_data["household_consumption_kw"].append(info.get("base_demand_kw", np.nan))

            # Collect individual reward components from info
            # These keys must match what's being put into 'info' in custom_env.py's step() or be calculated here
            # For now, let's assume they will be added to info or we extract based on existing info.
            # The environment currently logs them but doesn't explicitly return them all in info.
            # We'll need to ensure custom_env.py puts these into the info dict.
            # For now, we'll add placeholders and then modify custom_env.py.

            # Placeholder - these will be properly populated after custom_env.py is updated
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

    # Create plots
    fig, axs = plt.subplots(4, 1, figsize=(15, 20), sharex=True)
    fig.suptitle(f'Agent Performance Analysis - Model: {Path(model_path).name}', fontsize=16)

    # 1. SoC and Price vs. Time
    ax1 = axs[0]
    color = 'tab:red'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('SoC', color=color)
    ax1.step(df_to_plot['timestamps'], df_to_plot['soc'], color=color, label='Battery SoC')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1)

    ax1_twin = ax1.twinx()
    color = 'tab:blue'
    ax1_twin.set_ylabel('Electricity Price (öre/kWh)', color=color)
    ax1_twin.step(df_to_plot['timestamps'], df_to_plot['price'], color=color, linestyle='--', label='Electricity Price')
    ax1_twin.tick_params(axis='y', labelcolor=color)
    ax1.set_title('Battery SoC and Electricity Price')
    ax1.grid(True)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles1_twin, labels1_twin = ax1_twin.get_legend_handles_labels()
    ax1.legend(handles1 + handles1_twin, labels1 + labels1_twin, loc='upper right')


    # 2. Agent Action (Normalized) and Actual Battery Power
    ax2 = axs[1]
    color = 'tab:green'
    ax2.set_ylabel('Normalized Action (-1 to 1)', color=color)
    ax2.step(df_to_plot['timestamps'], df_to_plot['action_normalized'], color=color, label='Agent Action (Normalized)')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-1.1, 1.1)

    ax2_twin = ax2.twinx()
    color = 'tab:purple'
    ax2_twin.set_ylabel('Battery Power (kW)', color=color) # Positive = Charge, Negative = Discharge
    ax2_twin.step(df_to_plot['timestamps'], df_to_plot['battery_power_kw'], color=color, linestyle='--', label='Battery Power (kW)')
    ax2_twin.tick_params(axis='y', labelcolor=color)
    ax2.set_title('Agent Actions and Battery Power\nBattery Power: Positive = Charging, Negative = Discharging')
    ax2.grid(True)
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles2_twin, labels2_twin = ax2_twin.get_legend_handles_labels()
    ax2.legend(handles2 + handles2_twin, labels2 + labels2_twin, loc='upper right')

    # 3. Grid Power vs. Time
    ax3 = axs[2]
    ax3.step(df_to_plot['timestamps'], df_to_plot['grid_power_kw'], color='tab:orange', label='Grid Power (kW)')
    ax3.step(df_to_plot['timestamps'], df_to_plot['household_consumption_kw'], color='tab:cyan', linestyle=':', label='Household Consumption (kW)')
    ax3.set_ylabel('Power (kW)')
    ax3.set_title('Net Grid Power and Household Consumption')
    ax3.grid(True)
    ax3.legend()
    
    # 4. Cumulative Reward vs. Time
    ax4 = axs[3]
    ax4.step(df_to_plot['timestamps'], pd.Series(df_to_plot['reward']).cumsum(), color='tab:brown', label='Cumulative Reward')
    ax4.set_ylabel('Cumulative Reward')
    ax4.set_title('Cumulative Reward Over Episode')
    ax4.grid(True)
    ax4.legend()

    # Format x-axis for all subplots
    for ax in axs:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.tick_params(axis='x', rotation=45)
        # ax.legend() # Already handled by fig.legend or specific ax.legend

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to make room for suptitle
    
    # Save the plot
    plot_dir = os.path.join(PROJECT_ROOT, "src", "rl", "simulations", "results")
    os.makedirs(plot_dir, exist_ok=True)
    plot_filename = os.path.join(plot_dir, f"evaluation_plot_{Path(model_path).stem}.png")
    plt.savefig(plot_filename)
    print(f"\nSaved analysis plot to {plot_filename}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained RL agent for Home Energy Management.")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="src/rl/saved_models/short_term_agent_final.zip",
        help="Path to the saved PPO model (.zip file), e.g., src/rl/saved_models/short_term_agent_final.zip"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="src/rl/rl_config.json",
        help="Path to the RL configuration file (rl_config.json)."
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=1,
        help="Number of episodes to run for evaluation."
    )
    parser.add_argument(
        "--sim_days",
        type=int,
        default=3, # Default to 3 days for a quick but meaningful evaluation
        help="Number of simulation days for each evaluation episode."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()

    # Set up logging level based on verbose flag
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s')
        print("Verbose logging enabled")
    
    # Adjust model_path and config_path to be absolute or relative to script location if needed
    # For now, assume they are correctly provided (e.g. relative to CWD or absolute)
    
    # Ensure model_path and config_path are correctly interpreted
    # If this script is in the project root:
    model_p = args.model_path
    config_p = args.config

    if not os.path.isabs(model_p):
        model_p = os.path.join(PROJECT_ROOT, model_p)
    if not os.path.isabs(config_p):
        config_p = os.path.join(PROJECT_ROOT, config_p)


    evaluate_model(model_path=model_p, 
                   config_path=config_p, 
                   num_episodes=args.episodes,
                   eval_simulation_days=args.sim_days) 