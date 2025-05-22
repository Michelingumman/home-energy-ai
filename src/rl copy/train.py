"""
Training script for the hierarchical RL system.

Loads models, settings, and trains agents in a hierarchical fashion.
"""
import os
import json
import argparse
import datetime
from datetime import datetime as dt
from datetime import timedelta
from pathlib import Path
import numpy as np
import tensorflow as tf  # For loading keras models
import sys
import time
import logging
from typing import Optional
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics # For richer per-episode stats
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure as sb3_configure_logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import torch
from gymnasium.wrappers import FlattenObservation

# Add the project root directory (one level up from 'src') to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) 
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
# Import our custom components
from src.rl.custom_env import HomeEnergyEnv
from src.rl.agent import ShortTermAgent, RecurrentEnergyAgent
from src.rl import config as rl_config # Import the new config

# Set up logging - this is the main logging configuration for the training script
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s')
logger = logging.getLogger(__name__)

# Configure the home_energy_env logger to use the same level but avoid duplicate messages
env_logger = logging.getLogger("home_energy_env")
env_logger.setLevel(logging.INFO)  
# No handlers needed as they will inherit from the root logger

def create_env(env_config: dict, seed: Optional[int] = None, env_idx: int = 0):
    """
    Create a HomeEnergyEnv environment.
    
    Args:
        env_config: Environment configuration
        seed: Random seed
        env_idx: Index of the environment
        
    Returns:
        gym.Env: The created environment
    """
    env_config["seed"] = seed
    env_config["env_idx"] = env_idx
    return HomeEnergyEnv(config=env_config)

def setup_callbacks(config: dict) -> dict:
    """
    Set up training callbacks.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        dict: Callbacks for different training phases
    """
    # Create directories
    log_dir = Path(config.get("log_dir", "src/rl/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = Path(config.get("model_dir", "src/rl/saved_models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped folders
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    short_term_log_dir = log_dir / f"short_term_{timestamp}"
    recurrent_log_dir = log_dir / f"recurrent_{timestamp}"
    
    # Configure TensorBoard loggers
    short_term_logger = sb3_configure_logger(str(short_term_log_dir), ["stdout", "tensorboard"])
    recurrent_logger = sb3_configure_logger(str(recurrent_log_dir), ["stdout", "tensorboard"])
    
    # Create callbacks
    checkpoint_freq = config.get("checkpoint_freq", 25000)
    eval_freq = config.get("eval_freq", checkpoint_freq) 

    # Short-term checkpoint callback
    short_term_checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(model_dir / "short_term_checkpoints"),
        name_prefix="short_term_model",
        save_replay_buffer=True, 
        save_vecnormalize=True, 
        verbose=0
    )
    
    # Recurrent checkpoint callback
    recurrent_checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(model_dir / "recurrent_checkpoints"),
        name_prefix="recurrent_model",
        save_replay_buffer=True, 
        save_vecnormalize=True, 
        verbose=0
    )
    
    # Evaluation callback for the short-term agent
    # It will save the best model found during evaluation
    # Create a new HomeEnergyEnv instance for evaluation
    eval_env_config = {
        "battery_capacity": config.get("battery_capacity", 22.0),
        "simulation_days": config.get("simulation_days_eval", 3), # Potentially shorter for eval
        "peak_penalty_factor": config.get("peak_penalty_factor", 10.0),
        "use_price_predictions": config.get("use_price_predictions_eval", False),
        "price_predictions_path": config.get("price_predictions_path", "data/processed/SE3prices.csv"), # Adjusted default
        "fixed_baseload_kw": config.get("fixed_baseload_kw", 0.5),
        "time_step_minutes": config.get("time_step_minutes", 15),
        "use_variable_consumption": config.get("use_variable_consumption", False),
        "consumption_data_path": config.get("consumption_data_path", None),
        "battery_degradation_cost_per_kwh": config.get("battery_degradation_cost_per_kwh", 45.0),
        "log_level": "WARNING",  # Set higher log level for eval env to reduce output noise
        "debug_prints": False    # Disable debug prints for evaluation to prevent duplication
    }
    
    # Merge with main config to ensure reward parameters are included
    eval_config = config.copy()
    eval_config.update(eval_env_config)
    
    short_term_eval_env = Monitor(HomeEnergyEnv(config=eval_config))
    recurrent_eval_env = Monitor(HomeEnergyEnv(config=eval_config))

    short_term_eval_callback = EvalCallback(
        short_term_eval_env,
        best_model_save_path=str(model_dir / "best_short_term_model"),
        log_path=str(short_term_log_dir / "evaluations"),
        eval_freq=eval_freq, 
        n_eval_episodes=config.get("eval_episodes", 5),
        deterministic=True,
        render=False,
        callback_on_new_best=None, 
        callback_after_eval=None 
    )
    
    recurrent_eval_callback = EvalCallback(
        recurrent_eval_env,
        best_model_save_path=str(model_dir / "best_recurrent_model"),
        log_path=str(recurrent_log_dir / "evaluations"),
        eval_freq=eval_freq, 
        n_eval_episodes=config.get("eval_episodes", 5),
        deterministic=True,
        render=False,
        callback_on_new_best=None, 
        callback_after_eval=None 
    )

    short_term_callbacks = [short_term_checkpoint_callback, short_term_eval_callback]
    recurrent_callbacks = [recurrent_checkpoint_callback, recurrent_eval_callback]

    return {
        "short_term": short_term_callbacks,
        "recurrent": recurrent_callbacks,
    }


def create_environments(config: dict) -> HomeEnergyEnv:
    """
    Create base environment for training the ShortTermAgent.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        HomeEnergyEnv: The base environment for the short-term agent, wrapped in Monitor.
    """
    env_config = {
        "battery_capacity": config.get("battery_capacity", 22.0),
        "simulation_days": config.get("simulation_days", 7),
        "peak_penalty_factor": config.get("peak_penalty_factor", 10.0),
        "use_price_predictions": config.get("use_price_predictions_train", True),
        "price_predictions_path": config.get("price_predictions_path", "data/processed/SE3prices.csv"), # Adjusted default
        "fixed_baseload_kw": config.get("fixed_baseload_kw", 0.5),
        "time_step_minutes": config.get("time_step_minutes", 15),
        "use_variable_consumption": config.get("use_variable_consumption", False),
        "consumption_data_path": config.get("consumption_data_path", None),
        "battery_degradation_cost_per_kwh": config.get("battery_degradation_cost_per_kwh", 45.0), # defaults to 45 if not value set in config
        "debug_prints": True  # Enable debug prints for timestamp sanity checks
    }
    
    # Merge with main config to ensure reward parameters are included
    full_config = config.copy()
    full_config.update(env_config)
    
    base_env = HomeEnergyEnv(config=full_config)
    base_env = Monitor(base_env) # Wrap in Monitor for logging
    
    return base_env


def run_sanity_checks(env: HomeEnergyEnv, config: dict, num_steps: int = 100):
    """
    Run sanity checks on the environment to verify the pricing model and rewards.
    
    This function performs comprehensive validation of the reward components and pricing model 
    before starting the actual training, including:
    
    1. Price calculation verification - Tests if spot price with VAT, grid fee, and energy tax
       are calculated correctly
    2. Grid tariff verification - Tests if capacity charges and monthly fees are applied correctly
    3. Night discount verification - Tests if night-time capacity discount (22:00-06:00) works
    4. Reward components analysis - Tests if all reward components are balanced and working
    5. Battery behavior analysis - Tests battery charging/discharging patterns
    6. Peak load tracking - Tests if the top 3 peaks are correctly identified and averaged
    
    This is particularly helpful after implementing changes to the reward function or pricing model
    to catch any issues before spending time on training.
    
    Args:
        env: Environment to test
        config: Configuration dictionary
        num_steps: Number of steps to run in the test episode (default: 100)
    """
    logger.info("\n" + "="*80)
    logger.info("RUNNING PRE-TRAINING SANITY CHECKS")
    logger.info("="*80)
    
    # Reset the environment
    obs, info = env.reset(seed=42)
    
    # Get the unwrapped environment to access internal attributes
    # If env is wrapped in a Monitor, we need to get the unwrapped env
    if hasattr(env, 'env'):
        unwrapped_env = env.env
    else:
        unwrapped_env = env
    
    # Create statistics trackers
    stats = {
        "grid_costs": [],
        "capacity_metrics": [],
        "reward_components": {},
        "total_rewards": [],
        "grid_powers": [],
        "battery_powers": [],
        "prices": [],
        "is_night_discount": [],
        "hour_of_day": []
    }
    
    # Log pricing configuration
    logger.info("\nPRICING MODEL CONFIGURATION:")
    logger.info(f"Energy Tax: {config.get('energy_tax', 54.875)} öre/kWh")
    logger.info(f"VAT Multiplier: {config.get('vat_mult', 1.25)}")
    logger.info(f"Grid Fee: {config.get('grid_fee', 6.25)} öre/kWh")
    logger.info(f"Fixed Grid Fee: {config.get('fixed_grid_fee_sek_per_month', 365.0)} SEK/month")
    logger.info(f"Capacity Fee: {config.get('capacity_fee_sek_per_kw', 81.25)} SEK/kW/month")
    logger.info(f"Night Capacity Discount: {config.get('night_capacity_discount', 0.5)}")
    
    # Define a function to do detailed cost verification at a specific step
    def detailed_cost_verification(step_info, step_num, current_dt):
        """Perform detailed verification of cost calculations for a single step"""
        logger.info("-"*80)
        logger.info(f"DETAILED COST VERIFICATION FOR STEP {step_num} - {current_dt}")
        logger.info("-"*80)
        
        # Get the relevant values
        current_price = step_info.get("current_price", 0)
        grid_power = step_info.get("grid_power_kw", 0)
        is_night = step_info.get("is_night_discount", False)
        time_step_hours = unwrapped_env.time_step_hours
        
        # Get the config values
        energy_tax = config.get("energy_tax", 54.875)
        vat_mult = config.get("vat_mult", 1.25)
        grid_fee = config.get("grid_fee", 6.25)
        
        if grid_power > 0:  # Only for import
            # 1. Calculate spot price with VAT
            spot_with_vat = current_price * vat_mult
            logger.info(f"1. Spot price with VAT: {current_price:.2f} öre/kWh * {vat_mult} = {spot_with_vat:.2f} öre/kWh")
            
            # 2. Add grid fee
            with_grid_fee = spot_with_vat + grid_fee
            logger.info(f"2. With grid fee: {spot_with_vat:.2f} + {grid_fee:.2f} = {with_grid_fee:.2f} öre/kWh")
            
            # 3. Add energy tax
            total_cost_per_kwh = with_grid_fee + energy_tax
            logger.info(f"3. With energy tax: {with_grid_fee:.2f} + {energy_tax:.2f} = {total_cost_per_kwh:.2f} öre/kWh")
            
            # 4. Calculate energy cost for this step
            energy_cost_ore = grid_power * total_cost_per_kwh * time_step_hours
            energy_cost_sek = energy_cost_ore / 100.0
            logger.info(f"4. Energy cost: {grid_power:.2f} kW * {total_cost_per_kwh:.2f} öre/kWh * {time_step_hours:.2f} hours = {energy_cost_ore:.2f} öre = {energy_cost_sek:.2f} SEK")
            
            # 5. Check if night discount applies for capacity tracking
            if is_night:
                night_capacity_discount = config.get("night_capacity_discount", 0.5)
                logger.info(f"5. Night capacity discount: This is a night hour, so this peak will be counted as: {grid_power:.2f} kW * {night_capacity_discount} = {grid_power * night_capacity_discount:.2f} kW for capacity fee calculation")
            else:
                logger.info(f"5. No night discount: This is not a night hour, so this peak will be counted as: {grid_power:.2f} kW for capacity fee calculation")
                
            # 6. Check capacity fee calculation
            capacity_fee_per_kw = config.get("capacity_fee_sek_per_kw", 81.25)
            current_capacity_fee = unwrapped_env.peak_rolling_average * capacity_fee_per_kw
            logger.info(f"6. Current capacity fee (if applied now): {unwrapped_env.peak_rolling_average:.2f} kW * {capacity_fee_per_kw:.2f} SEK/kW = {current_capacity_fee:.2f} SEK/month")
            
            # 7. Top peaks tracking
            logger.info(f"7. Current top 3 peaks: {[f'{p:.2f} kW' for p in unwrapped_env.top3_peaks]}")
            logger.info(f"   Rolling average: {unwrapped_env.peak_rolling_average:.2f} kW")
            
            # 8. Month progress
            month_progress = step_info.get("month_progress", 0)
            logger.info(f"8. Month progress: {month_progress:.2f} ({month_progress*100:.1f}%)")
            
            # 9. Battery metrics
            if "power_kw" in step_info:
                battery_power = step_info.get("power_kw", 0)
                battery_energy = abs(battery_power) * time_step_hours
                battery_degradation_cost = unwrapped_env.battery.degradation_cost_per_kwh * battery_energy / 100.0  # Convert from öre to SEK
                logger.info(f"9. Battery metrics:")
                logger.info(f"   Battery power: {battery_power:.2f} kW ({'charging' if battery_power < 0 else 'discharging' if battery_power > 0 else 'idle'})")
                logger.info(f"   Battery energy processed: {battery_energy:.2f} kWh")
                logger.info(f"   Battery degradation cost: {battery_energy:.2f} kWh * {unwrapped_env.battery.degradation_cost_per_kwh:.2f} öre/kWh = {battery_degradation_cost:.2f} SEK")
        else:
            logger.info(f"Grid power is {grid_power:.2f} kW (negative or zero) - no import cost calculation for this step")
        
        # 10. Reward breakdown
        if "reward_components" in step_info:
            logger.info(f"10. Reward components breakdown:")
            for key, value in step_info["reward_components"].items():
                logger.info(f"    {key}: {value:.2f}")
                
        logger.info("-"*80)
    
    # Run the episode
    logger.info("\nSTEP-BY-STEP TEST RESULTS:")
    
    # Take random actions for the test episode
    detailed_steps = [9, 19, 29]  # Steps for detailed analysis
    max_steps = min(num_steps, unwrapped_env.simulation_steps)
    logger.info(f"Running sanity check for {max_steps} steps (out of {unwrapped_env.simulation_steps} total steps in a full episode)")
    
    for i in range(max_steps):
        action = np.array([np.random.uniform(-1.0, 1.0)])
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Store data for analysis
        if "reward_components" in info:
            for key, value in info["reward_components"].items():
                if key not in stats["reward_components"]:
                    stats["reward_components"][key] = []
                stats["reward_components"][key].append(value)
        
        # Store current datetime for hour analysis
        current_dt = unwrapped_env.start_datetime + timedelta(hours=unwrapped_env.current_step * unwrapped_env.time_step_hours)
        stats["hour_of_day"].append(current_dt.hour)
        
        stats["total_rewards"].append(reward)
        stats["grid_powers"].append(info.get("grid_power_kw", 0))
        stats["battery_powers"].append(info.get("power_kw", 0))
        stats["prices"].append(info.get("current_price", 0))
        stats["is_night_discount"].append(info.get("is_night_discount", False))
        
        # Every 10 steps, log detailed information for that step
        if i % 10 == 0:
            logger.info(f"\nStep {i} - Time: {current_dt}")
            logger.info(f"  Price: {info.get('current_price', 0):.2f} öre/kWh")
            logger.info(f"  Grid Power: {info.get('grid_power_kw', 0):.2f} kW")
            logger.info(f"  Battery Power: {info.get('power_kw', 0):.2f} kW")
            logger.info(f"  Is Night Discount: {info.get('is_night_discount', False)}")
            
            # Log reward components
            if "reward_components" in info:
                logger.info("  Reward Components:")
                for key, value in info["reward_components"].items():
                    logger.info(f"    {key}: {value:.2f}")
            
            # Log capacity metrics
            logger.info(f"  Top 3 Peaks: {unwrapped_env.top3_peaks}")
            logger.info(f"  Peak Rolling Average: {unwrapped_env.peak_rolling_average:.2f} kW")
            logger.info(f"  Current Capacity Fee: {info.get('current_capacity_fee', 0):.2f} SEK")
            logger.info(f"  Month Progress: {info.get('month_progress', 0):.2f}")
        
        # Perform detailed cost verification for specific steps
        if i in detailed_steps:
            detailed_cost_verification(info, i, current_dt)
        
        if terminated or truncated:
            break
    
    # Calculate and log summary statistics
    logger.info("="*80)
    logger.info("SANITY CHECK SUMMARY STATISTICS")
    logger.info("="*80)
    
    # Reward components analysis
    logger.info("\nREWARD COMPONENT STATISTICS:")
    for component, values in stats["reward_components"].items():
        if values:
            logger.info(f"{component}:")
            logger.info(f"  Mean: {np.mean(values):.2f}")
            logger.info(f"  Min: {np.min(values):.2f}")
            logger.info(f"  Max: {np.max(values):.2f}")
            logger.info(f"  Count of Non-Zero: {sum(1 for v in values if v != 0)}/{len(values)}")
    
    # Price statistics
    logger.info("\nPRICE STATISTICS:")
    prices = stats["prices"]
    if prices:
        logger.info(f"  Mean Price: {np.mean(prices):.2f} öre/kWh")
        logger.info(f"  Min Price: {np.min(prices):.2f} öre/kWh")
        logger.info(f"  Max Price: {np.max(prices):.2f} öre/kWh")
    
    # Night discount statistics
    night_discounts = stats["is_night_discount"]
    hours = stats["hour_of_day"]
    if night_discounts and hours:
        # Check if night discount hours match 22:00-06:00
        night_hours_expected = [22, 23, 0, 1, 2, 3, 4, 5]
        night_count = sum(1 for nd in night_discounts if nd)
        match_count = sum(1 for nd, hr in zip(night_discounts, hours) 
                           if (nd and hr in night_hours_expected) or 
                              (not nd and hr not in night_hours_expected))
        match_percent = (match_count / len(night_discounts)) * 100 if night_discounts else 0
                
        logger.info(f"\nNIGHT DISCOUNT STATISTICS:")
        logger.info(f"  Night Hours: {night_count}/{len(night_discounts)} ({night_count/len(night_discounts)*100:.1f}%)")
        logger.info(f"  Time Window Match: {match_count}/{len(night_discounts)} ({match_percent:.1f}%)")
        logger.info(f"  Expected Night Hours: {night_hours_expected}")
        
        # Log any discrepancies
        if match_percent < 100:
            logger.warning("  Night discount time window discrepancies detected!")
            for i, (nd, hr) in enumerate(zip(night_discounts, hours)):
                if (nd and hr not in night_hours_expected) or (not nd and hr in night_hours_expected):
                    current_dt = unwrapped_env.start_datetime + timedelta(hours=i * unwrapped_env.time_step_hours)
                    # Include both timestamp hour and actual hour in log to make it clearer
                    logger.warning(f"  Step {i} - {current_dt}: Timestamp is in hour {current_dt.hour}, has night_discount={nd} (expected {current_dt.hour in night_hours_expected})")
    
    # Grid power statistics
    grid_powers = stats["grid_powers"]
    if grid_powers:
        logger.info(f"\nGRID POWER STATISTICS:")
        logger.info(f"  Mean Grid Power: {np.mean(grid_powers):.2f} kW")
        logger.info(f"  Min Grid Power: {np.min(grid_powers):.2f} kW")
        logger.info(f"  Max Grid Power: {np.max(grid_powers):.2f} kW")
        # Count positive (import) and negative (export) grid power
        import_count = sum(1 for p in grid_powers if p > 0)
        export_count = sum(1 for p in grid_powers if p < 0)
        logger.info(f"  Import Count: {import_count}/{len(grid_powers)} ({import_count/len(grid_powers)*100:.1f}%)")
        logger.info(f"  Export Count: {export_count}/{len(grid_powers)} ({export_count/len(grid_powers)*100:.1f}%)")
    
    # Battery power statistics
    battery_powers = stats["battery_powers"]
    if battery_powers:
        logger.info(f"\nBATTERY POWER STATISTICS:")
        logger.info(f"  Mean Battery Power: {np.mean(battery_powers):.2f} kW")
        logger.info(f"  Min Battery Power: {np.min(battery_powers):.2f} kW")
        logger.info(f"  Max Battery Power: {np.max(battery_powers):.2f} kW")
        # Count charging and discharging
        charging_count = sum(1 for p in battery_powers if p < 0)
        discharging_count = sum(1 for p in battery_powers if p > 0)
        logger.info(f"  Charging Count: {charging_count}/{len(battery_powers)} ({charging_count/len(battery_powers)*100:.1f}%)")
        logger.info(f"  Discharging Count: {discharging_count}/{len(battery_powers)} ({discharging_count/len(battery_powers)*100:.1f}%)")
    
    logger.info("\n" + "="*80)
    logger.info("SANITY CHECKS COMPLETED - Check the logs for any anomalies")
    logger.info("="*80 + "\n")


def main():
    """
    Main entry point for the training script.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train RL agents for home energy management')
    parser.add_argument('--load-model', type=str, help='Path to load a pre-trained model')
    parser.add_argument('--test-only', action='store_true', help='Only test the model, no training')
    parser.add_argument('--timesteps', type=int, help='Override number of timesteps to train')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--recurrent', action='store_true', help='Use RecurrentPPO agent instead of standard PPO')
    
    # Add date range options
    parser.add_argument('--start-date', type=str, help='Start date for training data (format: YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for training data (format: YYYY-MM-DD)')
    
    # Add sanity check options
    parser.add_argument('--skip-sanity-check', action='store_true', help='Skip sanity checks before training')
    parser.add_argument('--sanity-check-only', action='store_true', help='Only run sanity checks, then exit')
    parser.add_argument('--sanity-check-steps', type=int, default=100, help='Number of steps to check during sanity check')
    
    # Add data augmentation option
    parser.add_argument('--augment-data', action='store_true', help='Enable data augmentation for solar and consumption data')
    
    args = parser.parse_args()
    
    # Load configuration
    config = rl_config.get_config_dict()
    
    # Update config with command-line arguments
    if args.load_model:
        config['load_model'] = True
        config['load_model_path'] = args.load_model
    
    if args.timesteps:
        config['short_term_timesteps'] = args.timesteps
    
    if args.seed:
        config['seed'] = args.seed
    
    # Process date range if specified
    if args.start_date and args.end_date:
        try:
            start_date = dt.strptime(args.start_date, "%Y-%m-%d")
            end_date = dt.strptime(args.end_date, "%Y-%m-%d")
            
            if start_date > end_date:
                raise ValueError("Start date must be before end date")
                
            print(f"Restricting training data to range: {args.start_date} to {args.end_date}")
            config["start_date"] = args.start_date
            config["end_date"] = args.end_date
            
        except Exception as e:
            print(f"Error processing date range: {e}")
            print("Please use format YYYY-MM-DD (e.g., 2023-06-01)")
            sys.exit(1)
    
    # Set data augmentation
    if args.augment_data:
        print("Enabling data augmentation for solar and consumption")
        config["use_data_augmentation"] = True
        config["augment_solar_data"] = True
        config["augment_consumption_data"] = True
    
    # Set up logging
    log_dir = Path(config.get("log_dir", "src/rl/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"short_term_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(asctime)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Log key configuration parameters
    for key, value in config.items():
        if key.startswith('short_term_') or key in ['seed', 'peak_penalty_factor']:
            logging.info(f"  {key}: {value}")
    
    # Set random seed if specified
    if 'seed' in config:
        seed = config['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    # Run sanity checks if not skipped
    if not args.skip_sanity_check or args.sanity_check_only:
        print(f"\n{'='*80}\nRunning sanity checks...\n{'='*80}")
        env = HomeEnergyEnv(config=config)
        run_sanity_checks(env, config, num_steps=args.sanity_check_steps)
        
        if args.sanity_check_only:
            print("Sanity checks complete. Exiting as requested.")
            sys.exit(0)
    
    # Set up callbacks
    callbacks = setup_callbacks(config)
    
    # Train or test based on arguments
    if args.recurrent:
        # Train with RecurrentPPO
        agent = train_recurrent_agent(config)
    else:
        # Train with standard PPO
        agent = train_short_term_agent(config)
    
    # Test the agent
    if not args.test_only:
        test_agent(agent, config)
    
    return agent


def train_recurrent_agent(config: dict):
    """
    Train the recurrent agent for short-term control.
    
    Args:
        config: Configuration dictionary
    """
    print("Setting up recurrent agent training...")
    
    # Get hyperparameters
    timesteps = config.get("short_term_timesteps", 50000)
    
    # Create and configure the environment
    env = HomeEnergyEnv(config=config)
    
    # Set up callbacks
    callbacks = setup_callbacks(config)
    
    # Create the agent
    print(f"Creating recurrent agent...")
    
    # Check if we should load an existing model
    model_path = None
    if config.get("load_recurrent_model", False):
        model_dir = config.get("model_dir", "src/rl/saved_models")
        model_name = config.get("load_recurrent_model_name", "best_recurrent_model")
        model_path = f"{model_dir}/{model_name}"
        if not os.path.exists(model_path + ".zip"):
            print(f"WARNING: Model {model_path}.zip not found, training new model")
            model_path = None
    
    # Create and train the agent
    agent = RecurrentEnergyAgent(env=env, model_path=model_path, config=config)
    
    # Start training
    print(f"Starting recurrent agent training for {timesteps} timesteps")
    agent.train(
        total_timesteps=timesteps,
        callback=callbacks.get("recurrent_callbacks", None)
    )
    
    # Save the final model
    model_dir = config.get("model_dir", "src/rl/saved_models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the final model
    final_model_path = f"{model_dir}/final_recurrent_model"
    agent.save(final_model_path)
    
    return agent


def test_agent(agent, config: dict):
    """
    Test a trained agent on a new episode.
    
    Args:
        agent: The trained agent
        config: Configuration dictionary
    """
    print("Testing trained agent...")
    
    # Create a test environment
    test_config = config.copy()
    test_config.update({
        "simulation_days": config.get("test_simulation_days", 7),
        "log_level": "INFO"
    })
    
    # Create test environment
    if isinstance(agent, RecurrentEnergyAgent):
        # For recurrent agents, we need to use the same wrapper
        env = HomeEnergyEnv(config=test_config)
        if isinstance(env.observation_space, gym.spaces.Dict):
            env = FlattenObservation(env)
    else:
        # For standard agents
        env = HomeEnergyEnv(config=test_config)
    
    # Run a test episode
    obs, info = env.reset()
    done = False
    total_reward = 0
    episode_steps = 0
    
    while not done:
        if isinstance(agent, RecurrentEnergyAgent):
            action, _ = agent.predict(obs, deterministic=True)
        else:
            action, _ = agent.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        episode_steps += 1
        
        if episode_steps % 100 == 0:
            print(f"Step {episode_steps}, Current reward: {reward:.2f}, Total reward: {total_reward:.2f}")
    
    print(f"Test completed: {episode_steps} steps, Total reward: {total_reward:.2f}")
    
    return total_reward


def train_short_term_agent(config: dict):
    """
    Train the short term agent for home energy management.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ShortTermAgent: The trained agent
    """
    print("Setting up short term agent training...")
    
    # Get hyperparameters
    timesteps = config.get("short_term_timesteps", 100000)
    
    # Create and configure the environment
    env = create_environments(config)
    
    # Set up callbacks
    callbacks = setup_callbacks(config)
    
    # Create the agent
    print(f"Creating short term agent...")
    
    # Check if we should load an existing model
    model_path = None
    if config.get("load_model", False):
        model_path = config.get("load_model_path", None)
        if model_path and not os.path.exists(model_path):
            print(f"WARNING: Model {model_path} not found, training new model")
            model_path = None
    
    # Create and train the agent
    agent = ShortTermAgent(env=env, model_path=model_path, config=config)
    
    # Start training
    print(f"Starting short term agent training for {timesteps} timesteps")
    agent.train(
        total_timesteps=timesteps,
        callback=callbacks.get("short_term", None)
    )
    
    # Save the final model
    model_dir = config.get("model_dir", "src/rl/saved_models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the final model
    final_model_path = f"{model_dir}/short_term_agent_final"
    agent.save(final_model_path)
    
    return agent


if __name__ == "__main__":
    main() 