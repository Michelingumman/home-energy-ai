#!/usr/bin/env python3
"""
Run Hyperparameter Optimization and then Train with Best Parameters

This script:
1. Runs hyperparameter optimization using Optuna
2. Takes the best parameters and trains a model with them
3. Evaluates the optimized model and generates performance plots
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add the project root directory to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import our modules
from src.rl.hyperparameter_optimization import run_hpo
from src.rl.config import get_config_dict
from src.rl.custom_env import HomeEnergyEnv
from src.rl.agent import ShortTermAgent
from src.rl.evaluate_agent import evaluate_episode, calculate_performance_metrics, plot_agent_performance, plot_reward_components

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_optimization")


def apply_params_to_config(best_params, config):
    """
    Apply the best parameters from HPO to the configuration dictionary.
    
    Args:
        best_params: Dictionary of best parameters from Optuna
        config: Configuration dictionary to update
        
    Returns:
        Updated configuration dictionary
    """
    logger.info("Applying best parameters to config:")
    for key, value in best_params.items():
        if key in config:
            logger.info(f"  {key}: {config[key]} -> {value}")
            config[key] = value
        else:
            logger.warning(f"  Parameter '{key}' not found in config, adding it with value {value}")
            config[key] = value
            
    return config


def train_with_best_params(config, model_dir="src/rl/saved_models", eval_dir="src/rl/simulations/results"):
    """
    Train a model with the best parameters from optimization.
    
    Args:
        config: Configuration dictionary with best parameters
        model_dir: Directory to save the trained model
        eval_dir: Directory to save evaluation results
        
    Returns:
        Path to the saved model
    """
    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Configure the environment with best parameters
    env = HomeEnergyEnv(config=config)
    
    # Create the agent
    agent = ShortTermAgent(
        env=env,
        config=config
    )
    
    # Train the agent (with full timesteps from config)
    logger.info(f"Training agent with optimized parameters for {config.get('short_term_timesteps', 500000)} timesteps")
    agent.train(total_timesteps=config.get("short_term_timesteps", 500000))
    
    # Save the trained model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"optimized_model_{timestamp}")
    agent.save(model_path)
    
    # Evaluate the trained model
    logger.info("Evaluating trained model...")
    episode_data = evaluate_episode(agent.model, config)
    metrics = calculate_performance_metrics(episode_data)
    
    # Log evaluation metrics
    logger.info(f"Evaluation metrics:")
    logger.info(f"  Total reward: {metrics['total_reward']:.2f}")
    logger.info(f"  Price-power correlation: {metrics['price_power_correlation']:.4f}")
    if 'peak_metrics' in metrics:
        logger.info(f"  Peak average: {metrics['peak_metrics'].get('peak_rolling_average', 0):.2f} kW")
    if 'soc_metrics' in metrics:
        logger.info(f"  SoC violations: {metrics['soc_metrics'].get('violation_percentage', 0):.2f}%")
    
    # Generate and save performance plots
    logger.info("Generating performance plots...")
    plot_save_path = os.path.join(eval_dir, f"optimized_performance_{timestamp}")
    plot_agent_performance(episode_data, model_name="Optimized Model", save_dir=plot_save_path)
    plot_reward_components(episode_data, model_name="Optimized Model", save_dir=plot_save_path)
    
    # Save evaluation metrics to file
    with open(os.path.join(eval_dir, f"optimized_metrics_{timestamp}.json"), "w") as f:
        # Convert metrics to a serializable format
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                serializable_metrics[k] = {sk: float(sv) if hasattr(sv, 'item') else sv 
                                          for sk, sv in v.items()}
            else:
                serializable_metrics[k] = float(v) if hasattr(v, 'item') else v
        json.dump(serializable_metrics, f, indent=2, default=str)
    
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization and train with best parameters")
    parser.add_argument("--trials", type=int, default=25, help="Number of optimization trials to run")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds for optimization")
    parser.add_argument("--skip-optimization", action="store_true", help="Skip optimization and use existing parameters file")
    parser.add_argument("--params-file", type=str, help="JSON file with parameters to use (if skipping optimization)")
    parser.add_argument("--timesteps", type=int, default=None, help="Override the number of timesteps for training")
    parser.add_argument("--simulation-days", type=int, default=None, help="Override the number of simulation days")
    
    args = parser.parse_args()
    
    # Get base configuration
    config = get_config_dict()
    
    # Apply overrides from command-line arguments
    if args.timesteps:
        config["short_term_timesteps"] = args.timesteps
        logger.info(f"Overriding short_term_timesteps to {args.timesteps}")
    
    if args.simulation_days:
        config["simulation_days"] = args.simulation_days
        logger.info(f"Overriding simulation_days to {args.simulation_days}")
    
    if not args.skip_optimization:
        # Run hyperparameter optimization
        logger.info(f"Starting hyperparameter optimization with {args.trials} trials")
        best_params = run_hpo(n_trials=args.trials, timeout=args.timeout)
        logger.info(f"Optimization complete. Best parameters: {best_params}")
    else:
        # Load parameters from file
        if not args.params_file:
            logger.error("Must specify --params-file when using --skip-optimization")
            sys.exit(1)
            
        logger.info(f"Loading parameters from {args.params_file}")
        with open(args.params_file, "r") as f:
            best_params = json.load(f)
    
    # Apply best parameters to config
    updated_config = apply_params_to_config(best_params, config)
    
    # Train with best parameters
    model_path = train_with_best_params(updated_config)
    logger.info(f"Training complete. Model saved to {model_path}")


if __name__ == "__main__":
    main() 