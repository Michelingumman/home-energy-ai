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
# Import stable-baselines3 components
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor


# Add the project root directory (one level up from 'src') to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) 
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
# Import our custom components
from src.rl.custom_env import HomeEnergyEnv
# from src.rl.wrappers import LongTermEnv # Commented out for simplification
from src.rl.agent import ShortTermAgent # Removed LongTermAgent, HierarchicalController


def load_config(config_path: str = "src/rl/rl_config.json") -> dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to the config file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            print(f"Loaded configuration from {config_path}")
            return config
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default configuration")
        return {}


def load_prediction_models(config: dict) -> tuple:
    """
    Load demand and price prediction models.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        tuple: (demand_model, price_model)
    """
    demand_model_path = config.get("demand_model_path", "")
    price_model_path = config.get("price_model_path", "")
    
    demand_model = None
    price_model = None
    
    # Load demand model if path is provided and file exists
    if demand_model_path and os.path.exists(demand_model_path):
        try:
            demand_model = tf.keras.models.load_model(demand_model_path)
            print(f"Loaded demand model from {demand_model_path}")
        except Exception as e:
            print(f"Error loading demand model: {e}")
    else:
        print("No demand model provided or file not found, will use simple forecasts")
    
    # Load price model if path is provided and file exists
    if price_model_path and os.path.exists(price_model_path):
        try:
            price_model = tf.keras.models.load_model(price_model_path)
            print(f"Loaded price model from {price_model_path}")
        except Exception as e:
            print(f"Error loading price model: {e}")
    else:
        print("No price model provided or file not found, will use simple forecasts")
    
    return demand_model, price_model


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
    # long_term_log_dir = log_dir / f"long_term_{timestamp}" # Commented out
    
    # Configure TensorBoard loggers
    short_term_logger = configure(str(short_term_log_dir), ["stdout", "tensorboard"])
    # long_term_logger = configure(str(long_term_log_dir), [\\"stdout\\", \\"tensorboard\\"]) # Commented out
    
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
        "config": config # Pass the main config dict
    }
    # Check if price_model should be passed to eval_env
    # For simplicity, let's assume eval_env uses simple prices or CSV if use_price_predictions_eval is True
    # and doesn't rely on a passed price_model object for now.
    short_term_eval_env = Monitor(HomeEnergyEnv(**eval_env_config))


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

    # long_term_callback = CheckpointCallback( # Commented out
    #     save_freq=checkpoint_freq // 2,
    #     save_path=str(model_dir / \\"long_term_checkpoints\\"),
    #     name_prefix=\\"long_term_model\\",
    #     verbose=1
    # )
    
    return {
        "short_term": short_term_callbacks,
        # "long_term": long_term_callback, # Commented out
        # "joint": None # Commented out
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
        "battery_degradation_cost_per_kwh": config.get("battery_degradation_cost_per_kwh", 45.0),
        "config": config # Pass the main config dict
    }
    # Ensure time_step_minutes is present, add if not in config for safety, though it should be.
    if "time_step_minutes" not in env_config:
        env_config["time_step_minutes"] = config.get("time_step_minutes", 15) # Default if not in config for some reason

    base_env = HomeEnergyEnv(**env_config)
    base_env = Monitor(base_env) # Wrap in Monitor for logging
    
    # long_term_env = LongTermEnv( # Commented out
    #     env=base_env,
    #     steps_per_action=config.get(\\"steps_per_action\\", 4),
    #     corridor_width=config.get(\\"corridor_width\\\", 0.1),
    #     planning_horizon=config.get(\\"planning_horizon\\\", 42)
    # )
    
    return base_env


def main(args):
    """
    Main training function.
    """
    config = load_config(args.config)
    
    # Override/set training_mode to short_term_only
    config["training_mode"] = "short_term_only" 
    print(f"Forcing training mode to: {config['training_mode']}")

    if args.short_term_model:
        config["short_term_model_path"] = args.short_term_model
    
    # if args.long_term_model: # Commented out
    #     config["long_term_model_path"] = args.long_term_model
    
    print("\n===== Training Configuration =====")
    print(f"Training mode: {config.get('training_mode')}")
    print(f"Short-term timesteps: {config.get('short_term_timesteps', 500000)}")
    print("==================================\n")
    
    # Load prediction models (only price_model might be used) - REMOVED
    # demand_model, price_model = load_prediction_models(config) # demand_model is loaded but not used by simplified env
    
    base_env = create_environments(config)
    
    callbacks_dict = setup_callbacks(config)
    
    short_term_agent = ShortTermAgent(
        env=base_env,
        model_path=config.get("short_term_model_path"),
        config=config
    )
    
    print("\n===== Training Short-Term Agent Only =====")
    short_term_agent.train(
        total_timesteps=config.get("short_term_timesteps", 500000),
        callback=callbacks_dict["short_term"] # Pass the list of callbacks
    )
    
    model_dir = Path(config.get("model_dir", "src/rl/saved_models"))
    short_term_agent.save(str(model_dir / "short_term_agent_final"))
        
    # Hierarchical/Long-term agent training and evaluation sections are removed.
    # The EvalCallback within short_term_callbacks will handle evaluation of the short-term agent.

    # if args.evaluate: # Commented out, EvalCallback handles this
    #     print("\n===== Evaluating Trained System =====")
    #     # Evaluation logic for hierarchical controller was here.
    #     # For short-term agent, evaluation is part of the EvalCallback.
    #     # We can print a message about where to find best model.
    #     print(f"Evaluation performed during training. Best model saved to: {model_dir / 'best_short_term_model'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the RL system (Simplified: Short-Term Agent Only)")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="src/rl/rl_config.json",
        help="Path to configuration file"
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
    main(args) 