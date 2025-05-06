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

# Import stable-baselines3 components
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

# Import our custom components
from src.rl.custom_env import HomeEnergyEnv
from src.rl.wrappers import LongTermEnv
from src.rl.agent import ShortTermAgent, LongTermAgent, HierarchicalController


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
    long_term_log_dir = log_dir / f"long_term_{timestamp}"
    
    # Configure TensorBoard loggers
    short_term_logger = configure(str(short_term_log_dir), ["stdout", "tensorboard"])
    long_term_logger = configure(str(long_term_log_dir), ["stdout", "tensorboard"])
    
    # Create callbacks
    checkpoint_freq = config.get("checkpoint_freq", 25000)
    
    # Short-term checkpoint callback
    short_term_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(model_dir / "short_term_checkpoints"),
        name_prefix="short_term_model",
        verbose=1
    )
    
    # Long-term checkpoint callback
    long_term_callback = CheckpointCallback(
        save_freq=checkpoint_freq // 2,  # More frequent checkpoints for long-term
        save_path=str(model_dir / "long_term_checkpoints"),
        name_prefix="long_term_model",
        verbose=1
    )
    
    # Placeholder for joint training callback (if needed)
    joint_callback = None
    
    return {
        "short_term": short_term_callback,
        "long_term": long_term_callback,
        "joint": joint_callback
    }


def create_environments(config: dict, demand_model=None, price_model=None) -> tuple:
    """
    Create environments for training.
    
    Args:
        config: Configuration dictionary
        demand_model: Pre-trained demand prediction model
        price_model: Pre-trained price prediction model
        
    Returns:
        tuple: (base_env, long_term_env)
    """
    # Create base environment
    base_env = HomeEnergyEnv(
        price_model=price_model,
        demand_model=demand_model,
        use_price_model=price_model is not None,
        use_demand_model=demand_model is not None,
        battery_capacity=config.get("battery_capacity", 22.0),
        simulation_days=config.get("simulation_days", 7),
        peak_penalty_factor=config.get("peak_penalty_factor", 10.0),
        comfort_bonus_factor=config.get("comfort_bonus_factor", 2.0),
        random_weather=config.get("random_weather", True)
    )
    
    # Wrap in Monitor for logging
    base_env = Monitor(base_env)
    
    # Create long-term environment
    long_term_env = LongTermEnv(
        env=base_env,
        steps_per_action=config.get("steps_per_action", 4),
        corridor_width=config.get("corridor_width", 0.1),
        planning_horizon=config.get("planning_horizon", 42)
    )
    
    return base_env, long_term_env


def main(args):
    """
    Main training function.
    
    Args:
        args: Command-line arguments
    """
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.mode:
        config["training_mode"] = args.mode
    
    if args.short_term_model:
        config["short_term_model_path"] = args.short_term_model
    
    if args.long_term_model:
        config["long_term_model_path"] = args.long_term_model
    
    # Print training configuration
    print("\n===== Training Configuration =====")
    print(f"Training mode: {config.get('training_mode', 'hierarchical')}")
    print(f"Short-term timesteps: {config.get('short_term_timesteps', 500000)}")
    print(f"Long-term timesteps: {config.get('long_term_timesteps', 200000)}")
    print(f"Joint timesteps: {config.get('joint_timesteps', 100000)}")
    print("==================================\n")
    
    # Load prediction models
    demand_model, price_model = load_prediction_models(config)
    
    # Create environments
    base_env, long_term_env = create_environments(config, demand_model, price_model)
    
    # Set up callbacks
    callbacks = setup_callbacks(config)
    
    # Determine training mode
    training_mode = config.get("training_mode", "hierarchical")
    
    if training_mode == "short_term_only":
        # Train only the short-term agent
        short_term_agent = ShortTermAgent(
            env=base_env,
            model_path=config.get("short_term_model_path"),
            config=config
        )
        
        print("\n===== Training Short-Term Agent Only =====")
        short_term_agent.train(
            total_timesteps=config.get("short_term_timesteps", 500000),
            callback=callbacks["short_term"]
        )
        
        # Save the final model
        model_dir = Path(config.get("model_dir", "src/rl/saved_models"))
        short_term_agent.save(str(model_dir / "short_term_agent_final"))
        
    elif training_mode == "long_term_only":
        # Train only the long-term agent
        # Load a pre-trained short-term agent if available
        short_term_model_path = config.get("short_term_model_path")
        
        if short_term_model_path and os.path.exists(short_term_model_path):
            print(f"Using pre-trained short-term agent from {short_term_model_path}")
            # In a full implementation, we'd use this agent inside the LongTermEnv
        else:
            print("Warning: Training long-term agent without a pre-trained short-term agent")
        
        long_term_agent = LongTermAgent(
            env=long_term_env,
            model_path=config.get("long_term_model_path"),
            config=config
        )
        
        print("\n===== Training Long-Term Agent Only =====")
        long_term_agent.train(
            total_timesteps=config.get("long_term_timesteps", 200000),
            callback=callbacks["long_term"]
        )
        
        # Save the final model
        model_dir = Path(config.get("model_dir", "src/rl/saved_models"))
        long_term_agent.save(str(model_dir / "long_term_agent_final"))
        
    else:  # hierarchical (default)
        # Train the hierarchical system
        controller = HierarchicalController(
            config=config,
            base_env=base_env,
            short_term_model_path=config.get("short_term_model_path"),
            long_term_model_path=config.get("long_term_model_path")
        )
        
        print("\n===== Training Hierarchical System =====")
        controller.train(callbacks=callbacks)
        
        # Save the final models
        controller.save()
        
        # Evaluate the trained system
        if args.evaluate:
            print("\n===== Evaluating Trained System =====")
            results = controller.evaluate(num_episodes=5)
            
            print("\nEvaluation Results:")
            for key, value in results.items():
                print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the hierarchical RL system")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="src/rl/rl_config.json",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["hierarchical", "short_term_only", "long_term_only"],
        help="Training mode"
    )
    
    parser.add_argument(
        "--short_term_model", 
        type=str, 
        help="Path to pre-trained short-term model"
    )
    
    parser.add_argument(
        "--long_term_model", 
        type=str, 
        help="Path to pre-trained long-term model"
    )
    
    parser.add_argument(
        "--evaluate", 
        action="store_true",
        help="Evaluate the trained model after training"
    )
    
    args = parser.parse_args()
    main(args) 