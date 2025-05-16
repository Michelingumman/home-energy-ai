# """
# Run a simulation with a trained PPO RL agent.

# Loads a PPO model and runs a simulation using HomeEnergyEnv.
# """
# import os
# import json
# import argparse
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# from pathlib import Path
# import datetime
# import pandas as pd
# import sys
# import time
# import logging

# # Import our custom components
# from src.rl.custom_env import HomeEnergyEnv
# from src.rl import config as rl_config # Import the new config

# # Add project root for easier imports if script is moved
# PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

# # Set up logging more robustly
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d: %(message)s')
# logger = logging.getLogger(__name__)

# def run_simulation(
#     model_path: str, 
#     num_episodes: int = 1, 
#     render: bool = False,
#     output_csv_path: str = None,
#     output_plot_path: str = None,
#     override_sim_days: int = None
# ) -> dict:
#     """
#     Run a simulation with trained models.
    
#     Args:
#         model_path: Path to trained model
#         num_episodes: Number of episodes to run
#         render: Whether to render the environment
#         output_csv_path: Path to save CSV results
#         output_plot_path: Path to save plot results
#         override_sim_days: Override the default simulation days
        
#     Returns:
#         dict: Simulation results
#     """
#     logger.info(f"Starting simulation with model: {model_path}")

#     if not Path(model_path).exists():
#         logger.error(f"Model file not found: {model_path}")
#         return

#     # Load configuration using the new Python module
#     config_dict = rl_config.get_config_dict()
#     logger.info(f"Configuration loaded from {rl_config.__file__}")

#     if not config_dict:
#         logger.error("Failed to load configuration dictionary from rl_config. Aborting simulation.")
#         return

#     sim_days = config_dict.get("simulation_days_eval", config_dict.get("simulation_days", 7))
#     if override_sim_days is not None:
#         sim_days = override_sim_days
#         logger.info(f"Overriding simulation days to: {sim_days}")

#     env_config = {
#         "battery_capacity": config_dict.get("battery_capacity"),
#         "simulation_days": sim_days,
#         "peak_penalty_factor": config_dict.get("peak_penalty_factor"),
#         "use_price_predictions": config_dict.get("use_price_predictions_eval"),
#         "price_predictions_path": config_dict.get("price_predictions_path"),
#         "fixed_baseload_kw": config_dict.get("fixed_baseload_kw"),
#         "render_mode": "human" if render else None,
#         "time_step_minutes": config_dict.get("time_step_minutes"),
#         "use_variable_consumption": config_dict.get("use_variable_consumption"),
#         "consumption_data_path": config_dict.get("consumption_data_path"),
#         "battery_degradation_cost_per_kwh": config_dict.get("battery_degradation_cost_per_kwh"),
#         "use_solar_predictions": config_dict.get("use_solar_predictions_eval"),
#         "solar_data_path": config_dict.get("solar_data_path_eval"),
#         "config": config_dict
#     }

#     for key in ["price_predictions_path", "consumption_data_path", "solar_data_path"]:
#         if env_config.get(key) and not os.path.isabs(env_config[key]):
#             abs_path = PROJECT_ROOT / env_config[key]
#             env_config[key] = str(abs_path)
#             logger.info(f"Adjusted path for {key} to absolute: {env_config[key]}")
#             if not abs_path.exists():
#                  logger.warning(f"Warning: Path for {key} does not exist: {abs_path}")

#     try:
#         eval_env = HomeEnergyEnv(**env_config)
#     except Exception as e:
#         logger.error(f"Error creating HomeEnergyEnv: {e}", exc_info=True)
#         return

#     try:
#         model = PPO.load(model_path, env=eval_env) # Use eval_env here
#     except Exception as e:
#         logger.error(f"Error loading PPO model: {e}", exc_info=True)
#         # Attempting to load with a new env instance for diagnostics, if fails above
#         try:
#             logger.info("Attempting to load model with a fresh environment instance for PPO.load().")
#             fresh_env_for_load = HomeEnergyEnv(**env_config) # Create a new instance for loading
#             model = PPO.load(model_path, env=fresh_env_for_load)
#             logger.info("Model loaded successfully with a fresh environment instance.")
#             # If this succeeds, the issue might be with how eval_env was mutated or used before PPO.load
#             # We should still use the original eval_env for running the simulation if this was just for loading.
#             # However, PPO.load might modify the env. Best practice is to pass a non-modified env or allow PPO to create one.
#             # For now, we set the model's env to our main eval_env if loading succeeded this way.
#             model.set_env(eval_env)
#         except Exception as e2:
#             logger.error(f"Still failed to load PPO model with a fresh env: {e2}", exc_info=True)
#             return

#     all_episode_data_dfs = []
#     logger.info(f"Running {num_episodes} simulation episode(s)...")

#     for episode in range(num_episodes):
#         obs, info = eval_env.reset()
#         terminated = False
#         truncated = False
#         episode_reward_sum = 0
        
#         # This data collection needs to match what `evaluate_agent.py` does for consistency if plots are shared
#         episode_data = {
#             "timestamps": [], "soc": [], "price": [], "action_normalized": [],
#             "battery_power_kw": [], "grid_power_kw": [], "reward": [],
#             "total_cost_cumulative": [], "household_consumption_kw": [], 
#             "current_solar_production_kw": [], "reward_grid_cost": [], 
#             "reward_peak_penalty": [], "reward_battery_cost": [],
#             "reward_arbitrage_bonus": [], "reward_soc_action_penalty": []
#         }

#         for step_num in range(eval_env.simulation_steps):
#             action, _states = model.predict(obs, deterministic=True)
#             obs, reward, terminated, truncated, info = eval_env.step(action)
#             episode_reward_sum += reward

#             current_sim_time = eval_env.start_datetime + pd.Timedelta(hours=eval_env.current_step * eval_env.time_step_hours)
            
#             episode_data["timestamps"].append(current_sim_time)
#             episode_data["soc"].append(obs["soc"][0])
#             episode_data["price"].append(info.get("current_price", np.nan))
#             episode_data["action_normalized"].append(float(action[0] if isinstance(action, np.ndarray) and action.ndim > 0 else action))
#             episode_data["battery_power_kw"].append(-info.get("power_kw", np.nan)) # Inverted for plotting
#             episode_data["grid_power_kw"].append(info.get("grid_power_kw", np.nan))
#             episode_data["reward"].append(reward)
#             episode_data["total_cost_cumulative"].append(info.get("total_cost", np.nan))
#             episode_data["household_consumption_kw"].append(info.get("base_demand_kw", np.nan))
#             episode_data["current_solar_production_kw"].append(info.get("current_solar_production_kw", np.nan))
#             episode_data["reward_grid_cost"].append(info.get("reward_grid_cost", np.nan))
#             episode_data["reward_peak_penalty"].append(info.get("reward_peak_penalty", np.nan))
#             episode_data["reward_battery_cost"].append(info.get("reward_battery_cost", np.nan))
#             episode_data["reward_arbitrage_bonus"].append(info.get("reward_arbitrage_bonus", np.nan))
#             episode_data["reward_soc_action_penalty"].append(info.get("reward_soc_action_penalty", np.nan))

#             if render:
#                 eval_env.render()
#                 # time.sleep(0.1) # Optional delay for human viewing
            
#             if terminated or truncated:
#                 break
        
#         all_episode_data_dfs.append(pd.DataFrame(episode_data))
#         logger.info(f"Episode {episode + 1} finished. Total reward: {episode_reward_sum:.2f}, Total cost: {info.get('total_cost', 0):.2f}")

#     if not all_episode_data_dfs:
#         logger.warning("No simulation data collected.")
#         return
    
#     # For simplicity, using data from the first episode for CSV and plot
#     df_to_process = all_episode_data_dfs[0]

#     if output_csv_path:
#         try:
#             csv_path = Path(output_csv_path)
#             csv_path.parent.mkdir(parents=True, exist_ok=True)
#             df_to_process.to_csv(csv_path, index=False)
#             logger.info(f"Simulation data saved to {csv_path}")
#         except Exception as e:
#             logger.error(f"Error saving simulation data to CSV {output_csv_path}: {e}")

#     if output_plot_path:
#         try:
#             plot_path = Path(output_plot_path)
#             plot_path.parent.mkdir(parents=True, exist_ok=True)
#             # Simplified plot - customize as needed, mirroring evaluate_agent.py plotting if desired
#             fig, axs = plt.subplots(4, 1, figsize=(15, 20), sharex=True)
#             axs[0].step(df_to_process['timestamps'], df_to_process['soc'], label='SoC')
#             axs[0].set_ylabel('SoC')
#             ax0_twin = axs[0].twinx()
#             ax0_twin.step(df_to_process['timestamps'], df_to_process['price'], label='Price', color='orange', linestyle='--')
#             ax0_twin.set_ylabel('Price')
#             axs[0].legend(loc='upper left'); ax0_twin.legend(loc='upper right')

#             axs[1].step(df_to_process['timestamps'], df_to_process['battery_power_kw'], label='Battery Power (kW)')
#             axs[1].set_ylabel('Battery Power (kW)')
#             axs[1].legend()

#             axs[2].step(df_to_process['timestamps'], df_to_process['grid_power_kw'], label='Grid Power (kW)')
#             axs[2].step(df_to_process['timestamps'], df_to_process['household_consumption_kw'], label='Demand (kW)', linestyle=':')
#             axs[2].step(df_to_process['timestamps'], df_to_process['current_solar_production_kw'], label='Solar (kW)', linestyle=':')
#             axs[2].set_ylabel('Power (kW)')
#             axs[2].legend()
            
#             axs[3].plot(df_to_process['timestamps'], df_to_process['reward'].cumsum(), label='Cumulative Reward')
#             axs[3].set_ylabel('Cumulative Reward')
#             axs[3].legend()

#             for ax in axs:
#                 ax.grid(True)
#                 ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
#                 plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            
#             fig.tight_layout()
#             plt.savefig(plot_path)
#             logger.info(f"Simulation plot saved to {plot_path}")
#             if render: # Show plot if rendering was also enabled for env steps
#                 plt.show()
#         except Exception as e:
#             logger.error(f"Error saving simulation plot to {output_plot_path}: {e}")
    
#     logger.info("Simulation run completed.")
#     # Return the first episode's dataframe for potential further use
#     return df_to_process 

# def main():
#     parser = argparse.ArgumentParser(description="Run a simulation with a trained RL agent.")
#     parser.add_argument("--model_path", type=str, required=True, help="Path to the trained PPO model (.zip file).")
#     parser.add_argument(
#         "--config", 
#         type=str, 
#         default="src/rl/config.py", 
#         help="Path to the RL Python configuration file (default: src/rl/config.py). This script imports src.rl.config directly."
#     )
#     parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to simulate.")
#     parser.add_argument("--sim_days", type=int, default=None, help="Override simulation days for this run.")
#     parser.add_argument("--render", action="store_true", help="Render the environment during simulation.")
#     parser.add_argument("--output_csv", type=str, default=None, help="Path to save simulation data as CSV.")
#     parser.add_argument("--output_plot", type=str, default=None, help="Path to save simulation plot as PNG.")
#     parser.add_argument("--verbose", action="store_true", help="Enable verbose logging from the environment.")

#     args = parser.parse_args()

#     if args.verbose:
#         # You might need to adjust how logging is configured for the environment if it has its own logger
#         logging.getLogger("src.rl.custom_env").setLevel(logging.DEBUG) # Example
#         logger.info("Verbose logging enabled for custom_env.")

#     # Config path argument handling (for user awareness)
#     default_config_path_str = "src/rl/config.py"
#     if os.path.normpath(args.config) != os.path.normpath(default_config_path_str):
#         logger.warning(
#             f"The --config argument was set to '{args.config}', "
#             f"but this script directly imports configuration from '{default_config_path_str}'. "
#             f"The value of --config is currently not used to load an alternative Python config module."
#         )
    
#     # Check existence of the standard config file we are importing
#     standard_config_module_path = PROJECT_ROOT / default_config_path_str
#     if not standard_config_module_path.exists():
#         logger.error(f"The standard configuration module {standard_config_module_path} was not found. Please ensure it exists.")
#         sys.exit(1)
#     logger.info(f"Using configuration imported from: {rl_config.__file__}")

#     abs_model_path = Path(args.model_path)
#     if not abs_model_path.is_absolute():
#         abs_model_path = PROJECT_ROOT / args.model_path

#     output_csv = args.output_csv
#     if output_csv and not Path(output_csv).is_absolute():
#         output_csv = PROJECT_ROOT / output_csv
        
#     output_plot = args.output_plot
#     if output_plot and not Path(output_plot).is_absolute():
#         output_plot = PROJECT_ROOT / output_plot

#     run_simulation(
#         model_path=str(abs_model_path),
#         num_episodes=args.episodes,
#         render=args.render,
#         output_csv_path=str(output_csv) if output_csv else None,
#         output_plot_path=str(output_plot) if output_plot else None,
#         override_sim_days=args.sim_days
#     )

# if __name__ == "__main__":
#     main() 