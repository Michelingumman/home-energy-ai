"""
Training script for the hierarchical RL system.

Loads models, settings, and trains agents in a hierarchical fashion.
"""
import os
import json
import argparse
import datetime
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

# Add the project root directory (one level up from 'src') to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) 
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
# Import our custom components
from src.rl.custom_env import HomeEnergyEnv
from src.rl.agent import ShortTermAgent
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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_term_log_dir = log_dir / f"short_term_{timestamp}"
    
    # Configure TensorBoard loggers
    short_term_logger = sb3_configure_logger(str(short_term_log_dir), ["stdout", "tensorboard"])
    
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

    short_term_callbacks = [short_term_checkpoint_callback, short_term_eval_callback]

    return {
        "short_term": short_term_callbacks,
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


def main():
    """
    Main training function.
    """
    # Load configuration
    config_dict = rl_config.get_config_dict() # New way
    
    # Override/set training_mode to short_term_only
    config_dict["training_mode"] = "short_term_only" 
    print(f"Forcing training mode to: {config_dict['training_mode']}")

    if args.short_term_model:
        config_dict["short_term_model_path"] = args.short_term_model
    
    print("\n===== Training Configuration =====")
    print(f"Training mode: {config_dict.get('training_mode')}")
    print(f"Short-term timesteps: {config_dict.get('short_term_timesteps', 500000)}")
    print("==================================\n")
    
    base_env = create_environments(config_dict)
    
    callbacks_dict = setup_callbacks(config_dict)
    
    short_term_agent = ShortTermAgent(
        env=base_env,
        model_path=config_dict.get("short_term_model_path"),
        config=config_dict
    )
    
    print("\n===== Training Short-Term Agent Only =====")
    short_term_agent.train(
        total_timesteps=config_dict.get("short_term_timesteps", 500000),
        callback=callbacks_dict["short_term"] # Pass the list of callbacks
    )
    
    model_dir = Path(config_dict.get("model_dir", "src/rl/saved_models"))
    short_term_agent.save(str(model_dir / "short_term_agent_final"))
        
    # Hierarchical/Long-term agent training and evaluation sections are removed.
    # The EvalCallback within short_term_callbacks will handle evaluation of the short-term agent.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the RL system (Simplified: Short-Term Agent Only)")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="src/rl/config.py", # Changed default path
        help="Path to the RL Python configuration file (e.g., src/rl/config.py)." # Updated help text
    )
    
    # parser.add_argument( # Commented out mode, as it's forced to short_term_only
    #     "--mode", 
    #     type=str, 
    #     choices=["hierarchical", "short_term_only", "long_term_only"],
    #     help="Training mode (currently forced to short_term_only)"
    # )
    
    parser.add_argument(
        "--short_term_model", 
        type=str, 
        help="Path to pre-trained short-term model to continue training"
    )
    
    # parser.add_argument( # Commented out long_term_model argument
    #     "--long_term_model", 
    #     type=str, 
    #     help="Path to pre-trained long-term model"
    # )
    
    # parser.add_argument( # Commented out evaluate, as EvalCallback handles it
    #     "--evaluate", 
    #     action="store_true",
    #     help="Evaluate the trained model after training (handled by EvalCallback)"
    # )
    
    args = parser.parse_args()
    main() 