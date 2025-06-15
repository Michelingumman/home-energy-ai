"""
Training script for the RL energy management system.

Enhanced with improvements including:
- Curriculum learning schedule for progressive training
- Adaptive exploration with entropy scheduling
- Reward component analysis and balance monitoring
- Training issue detection and recommendations
- Better PPO hyperparameters
- Advanced callbacks for monitoring

examples:
python train.py --config config.json --test-only
python train.py --config config.json --skip-sanity-check
python train.py --config config.json --sanity-check-only
python train.py --config config.json --start-date 2023-01-01 --end-date 2023-01-31

Loads models, settings, and trains agents.
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
from typing import Optional, Dict, Any, List
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics # For richer per-episode stats
from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
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
from src.rl.agent import (
    RecurrentEnergyAgent,
    analyze_reward_components, 
    detect_training_issues,
    adaptive_exploration_schedule,
    create_curriculum_schedule
)
from src.rl import config as rl_config # Import the new config

# Set up logging - reduced noise during training
logging.basicConfig(
    level=logging.WARNING,  # Only warnings and errors to console
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Set up file logging for detailed logs
file_handler = logging.FileHandler('src/rl/logs/training.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Configure the home_energy_env logger to log to file only
env_logger = logging.getLogger("home_energy_env")
env_logger.setLevel(logging.WARNING)  # Reduce console noise


class EnhancedTrainingCallback(BaseCallback):
    """Enhanced callback with reward analysis and training issue detection."""
    
    def __init__(self, curriculum_schedule=None, verbose=0):
        super().__init__(verbose)
        self.reward_history = []
        self.metrics_history = []
        self.last_analysis_step = 0
        self.analysis_interval = 100000  # Analyze every 100k steps (reduced frequency)
        self.curriculum_schedule = curriculum_schedule or []
        self.current_phase = 0
        
    def _on_step(self) -> bool:
        # Handle curriculum learning
        if self.curriculum_schedule:
            self._update_curriculum()
        
        # Collect reward components from info
        if 'reward_components' in self.locals.get('infos', [{}])[0]:
            self.reward_history.append(self.locals['infos'][0]['reward_components'])
        
        # Collect training metrics
        if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
            metrics = dict(self.model.logger.name_to_value)
            metrics['step'] = self.num_timesteps
            self.metrics_history.append(metrics)
        
        # Periodic analysis
        if self.num_timesteps - self.last_analysis_step >= self.analysis_interval:
            self._perform_analysis()
            self.last_analysis_step = self.num_timesteps
        
        return True
    
    def _update_curriculum(self):
        """Update curriculum phase if needed."""
        if self.current_phase < len(self.curriculum_schedule):
            phase = self.curriculum_schedule[self.current_phase]
            if self.num_timesteps >= phase['end_step']:
                self.current_phase += 1
                if self.current_phase < len(self.curriculum_schedule):
                    next_phase = self.curriculum_schedule[self.current_phase]
                    logger.info(f"\n[CURRICULUM] Advancing to phase {self.current_phase + 1}: {next_phase['name']}")
                    logger.info(f"New phase parameters: {next_phase['config_overrides']}")
    
    def _perform_analysis(self):
        """Perform reward and training analysis."""
        logger.info(f"\n{'='*60}")
        logger.info(f"[TRAINING ANALYSIS] AT STEP {self.num_timesteps}")
        logger.info(f"{'='*60}")
        
        # Analyze reward components
        if len(self.reward_history) > 100:
            recent_rewards = self.reward_history[-1000:]  # Last 1000 episodes
            analysis = analyze_reward_components(recent_rewards)
            
            logger.info("[REWARD ANALYSIS] Component Analysis:")
            logger.info(f"Balance Score: {analysis.get('balance_score', 0):.3f}")
            
            if 'recommendations' in analysis:
                for rec in analysis['recommendations']:
                    logger.warning(f"  WARNING: {rec}")
        
        # Detect training issues
        if len(self.metrics_history) > 50:
            issues = detect_training_issues(self.metrics_history)
            logger.info("[TRAINING ISSUES] Detection Results:")
            for issue in issues:
                if "No major" in issue:
                    logger.info(f"  OK: {issue}")
                else:
                    logger.warning(f"  WARNING: {issue}")
        
        logger.info(f"{'='*60}\n")


class AdaptiveEntropyCallback(BaseCallback):
    """Callback to adaptively adjust entropy coefficient during training."""
    
    def __init__(self, total_timesteps: int, base_entropy: float = 0.005, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.base_entropy = base_entropy
        
    def _on_step(self) -> bool:
        # Update entropy coefficient
        new_entropy = adaptive_exploration_schedule(
            self.num_timesteps, 
            self.total_timesteps, 
            self.base_entropy
        )
        
        # Update the model's entropy coefficient
        if hasattr(self.model, 'ent_coef'):
            self.model.ent_coef = new_entropy
            
        # Log entropy changes periodically
        if self.num_timesteps % 50000 == 0:  # Much less frequent
            print(f"Step {self.num_timesteps}: Entropy coefficient = {new_entropy:.4f}")
        
        return True


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
    Set up enhanced training callbacks with all improvements.
    
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
    recurrent_log_dir = log_dir / f"recurrent_{timestamp}"
    
    # Configure TensorBoard loggers
    recurrent_logger = sb3_configure_logger(str(recurrent_log_dir), ["stdout", "tensorboard"])
    
    # Create callbacks
    checkpoint_freq = config.get("checkpoint_freq", 25000)
    eval_freq = config.get("eval_freq", checkpoint_freq) 

    # Create curriculum schedule
    total_timesteps = config.get("total_timesteps", 500000)
    curriculum_schedule = create_curriculum_schedule(total_timesteps)
    
    # Enhanced training callback with all improvements
    enhanced_callback = EnhancedTrainingCallback(
        curriculum_schedule=curriculum_schedule,
        verbose=1
    )
    
    # Adaptive entropy callback
    entropy_callback = AdaptiveEntropyCallback(
        total_timesteps=total_timesteps,
        base_entropy=config.get("ent_coef", 0.005),
        verbose=1
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
    
    # Evaluation callback for the recurrent agent
    # Create a new HomeEnergyEnv instance for evaluation
    eval_env_config = {
        "battery_capacity": config.get("battery_capacity", 22.0),
        "simulation_days": config.get("simulation_days_eval", 3), # Potentially shorter for eval
        "peak_penalty_factor": config.get("peak_penalty_factor", 10.0),
        "use_price_predictions": config.get("use_price_predictions_eval", False),
        "price_predictions_path": config.get("price_predictions_path", "data/processed/SE3prices.csv"), 
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
    
    recurrent_eval_env = Monitor(HomeEnergyEnv(config=eval_config))
    # Wrap with FlattenObservation for RecurrentPPO
    recurrent_eval_env = FlattenObservation(recurrent_eval_env)

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

    return [enhanced_callback, entropy_callback, recurrent_checkpoint_callback, recurrent_eval_callback]


def create_environments(config: dict) -> HomeEnergyEnv:
    """
    Create enhanced base environment for training.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        HomeEnergyEnv: The base environment, wrapped in Monitor.
    """
    env_config = {
        "battery_capacity": config.get("battery_capacity", 22.0),
        "simulation_days": config.get("simulation_days", 7),
        "peak_penalty_factor": config.get("peak_penalty_factor", 10.0),
        "use_price_predictions": config.get("use_price_predictions_train", True),
        "price_predictions_path": config.get("price_predictions_path", "data/processed/SE3prices.csv"),
        "fixed_baseload_kw": config.get("fixed_baseload_kw", 0.5),
        "time_step_minutes": config.get("time_step_minutes", 15),
        "use_variable_consumption": config.get("use_variable_consumption", False),
        "consumption_data_path": config.get("consumption_data_path", None),
        "battery_degradation_cost_per_kwh": config.get("battery_degradation_cost_per_kwh", 45.0),
        "debug_prints": False,  # Reduce debug output during training
        "log_level": "WARNING"  # Reduce log noise
    }
    
    # Merge with main config to ensure reward parameters are included
    full_config = config.copy()
    full_config.update(env_config)
    
    logger.info("Creating training environment with enhanced configuration...")
    logger.info(f"Key parameters:")
    logger.info(f"  - Simulation days: {full_config['simulation_days']}")
    logger.info(f"  - Battery capacity: {full_config['battery_capacity']} kWh")
    logger.info(f"  - Reward scaling: {full_config.get('reward_scaling_factor', 'default')}")
    logger.info(f"  - SoC penalty factor: {full_config.get('soc_limit_penalty_factor', 'default')}")
    
    # Create environment
    env = HomeEnergyEnv(config=full_config)
    # Wrap with FlattenObservation for RecurrentPPO
    env = FlattenObservation(env)
    
    return env


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
        if i % 100 == 0:
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


def train_recurrent_agent(config: dict, callbacks=None):
    """
    Train the recurrent agent for energy management with all improvements.
    
    Args:
        config: Configuration dictionary
        callbacks: Optional callbacks for training
    """
    print("Setting up enhanced recurrent agent training...")
    
    # Get hyperparameters with improved defaults
    timesteps = config.get("short_term_timesteps", 500000)
    
    # Print key hyperparameters (to console, won't interrupt progress bar)
    print(f"Model configuration:")
    print(f"  - Total timesteps: {timesteps:,}")
    print(f"  - Learning rate: {config.get('learning_rate', 3e-4)}")
    print(f"  - Batch size: {config.get('batch_size', 64)}")
    print(f"  - N steps: {config.get('n_steps', 2048)}")
    print(f"  - Gamma: {config.get('gamma', 0.99)}")
    
    # Create and configure the environment
    env = create_environments(config)
    
    # Create the agent
    print(f"Creating enhanced recurrent agent...")
    
    # Check if we should load an existing model
    model_path = None
    if config.get("load_recurrent_model", False):
        model_dir = config.get("model_dir", "src/rl/saved_models")
        model_name = config.get("load_recurrent_model_name", "best_recurrent_model")
        model_path = f"{model_dir}/{model_name}"
        if not os.path.exists(model_path + ".zip"):
            print(f"WARNING: Model {model_path}.zip not found, training new model")
            model_path = None
        else:
            print(f"Loading pre-trained model from {model_path}")
    
    # Enhanced agent configuration with improved hyperparameters
    enhanced_config = config.copy()
    
    # Apply improved hyperparameters if not already set
    improved_defaults = {
        'learning_rate': 2e-4,  # Slightly lower learning rate for stability
        'n_steps': 4096,        # Larger rollout buffer
        'batch_size': 128,      # Larger batch size
        'gamma': 0.995,         # Slightly higher discount factor
        'gae_lambda': 0.98,     # Higher GAE lambda
        'ent_coef': 0.005,      # Balanced entropy
        'n_epochs': 8,          # More training epochs per rollout
    }
    
    for key, value in improved_defaults.items():
        if key not in enhanced_config:
            enhanced_config[key] = value
    
    # Create and train the agent
    agent = RecurrentEnergyAgent(env=env, model_path=model_path, config=enhanced_config)
    
    # Start training with simple print message
    print(f"Starting enhanced training for {timesteps:,} timesteps...")
    print("Training features: Curriculum learning, Adaptive exploration, Reward analysis")
    print("="*80)
    
    try:
        agent.train(
            total_timesteps=timesteps,
            callback=callbacks
        )
        
        # Save the final model with timestamp
        model_dir = config.get("model_dir", "src/rl/saved_models")
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = f"{model_dir}/final_recurrent_model_{timestamp}"
        agent.save(final_model_path)
        
        print(f"\nTraining completed successfully!")
        print(f"Final model saved to: {final_model_path}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        env.close()
    
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
    env = HomeEnergyEnv(config=test_config)
    # For recurrent agents, we need to use the FlattenObservation wrapper
    env = FlattenObservation(env)
    
    # Run a test episode
    obs, info = env.reset()
    done = False
    total_reward = 0
    episode_steps = 0
    
    # For recurrent agents, we need to track state
    lstm_state = None
    episode_start = True
    
    while not done:
        action, lstm_state = agent.predict(
            obs, 
            state=lstm_state,
            episode_start=episode_start,
            deterministic=True
        )
        episode_start = False
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        episode_steps += 1
        
        if episode_steps % 100 == 0:
            print(f"Step {episode_steps}, Current reward: {reward:.2f}, Total reward: {total_reward:.2f}")
    
    print(f"Test completed: {episode_steps} steps, Total reward: {total_reward:.2f}")
    
    return total_reward





def main():
    """Main entry point for training."""
    
    # Command line arguments
    parser = argparse.ArgumentParser(description="Train an RL agent for home energy management")
    parser.add_argument("--test-only", action="store_true", help="Only run testing, no training")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--skip-sanity-check", action="store_true", help="Skip sanity checks")
    parser.add_argument("--sanity-check-only", action="store_true", help="Only run sanity checks, then exit")
    parser.add_argument("--sanity-check-steps", type=int, default=100, help="Number of steps to run for sanity checks")
    parser.add_argument("--start-date", type=str, help="Start date for training data (format: YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for training data (format: YYYY-MM-DD)")
    parser.add_argument("--total-timesteps", type=int, help="Override total timesteps for training")
    
    args = parser.parse_args()
    

    # Load configuration, with priority order:
    # 1. CLI config path
    # 2. Default config from config.py
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Use our standard configuration
        config = rl_config.get_config_dict()

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
            
    # Override timesteps if specified
    if args.total_timesteps:
        config["short_term_timesteps"] = args.total_timesteps
        print(f"Using {args.total_timesteps} timesteps for training")

    # Print configuration overview
    print(f"\n{'='*80}\nStarting training with configuration\n{'='*80}")
    
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
    agent = train_recurrent_agent(config, callbacks)
    
    # Test the agent
    if not args.test_only:
        test_agent(agent, config)
    
    return agent




if __name__ == "__main__":
    main() 