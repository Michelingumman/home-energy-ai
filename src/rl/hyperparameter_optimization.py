#!/usr/bin/env python3
"""
Hyperparameter Optimization for the RL agent using Optuna.

This script implements Bayesian optimization to find optimal values for:
1. Reward component scaling factors
2. RL algorithm hyperparameters
"""

import os
import sys
import json
import traceback
import numpy as np
import logging
import optuna
from optuna.samplers import TPESampler
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root directory to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from src.rl.custom_env import HomeEnergyEnv
from src.rl.config import get_config_dict
from src.rl.evaluate_agent import evaluate_episode, calculate_performance_metrics


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("hpo")


def objective(trial):
    """
    Optuna objective function to minimize.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        float: Score based on reward and price-power correlation
    """
    # Load base configuration
    config = get_config_dict()
    
    # Sample hyperparameters to optimize
    
    # 1. Reward Scaling Parameters
    config["soc_violation_scale"] = trial.suggest_float("soc_violation_scale", 0.1, 5.0, log=True)
    config["soc_action_limit_scale"] = trial.suggest_float("soc_action_limit_scale", 0.1, 2.0, log=True)
    config["preferred_soc_reward_scale"] = trial.suggest_float("preferred_soc_reward_scale", 0.5, 5.0)
    config["high_soc_penalty_scale"] = trial.suggest_float("high_soc_penalty_scale", 0.1, 2.0)
    config["peak_penalty_scale"] = trial.suggest_float("peak_penalty_scale", 0.1, 3.0)
    config["arbitrage_reward_scale"] = trial.suggest_float("arbitrage_reward_scale", 0.5, 5.0)
    config["reward_scaling_factor"] = trial.suggest_float("reward_scaling_factor", 0.01, 1.0, log=True)
    
    # Additional reward parameters
    config["charge_at_low_price_reward_factor"] = trial.suggest_float("charge_at_low_price_reward_factor", 1.0, 10.0)
    config["discharge_at_high_price_reward_factor"] = trial.suggest_float("discharge_at_high_price_reward_factor", 1.0, 15.0)
    config["soc_limit_penalty_factor"] = trial.suggest_float("soc_limit_penalty_factor", 5.0, 30.0)
    config["preferred_soc_reward_factor"] = trial.suggest_float("preferred_soc_reward_factor", 1.0, 10.0)
    config["high_soc_penalty_factor"] = trial.suggest_float("high_soc_penalty_factor", 5.0, 20.0)
    config["peak_penalty_factor"] = trial.suggest_float("peak_penalty_factor", 1.0, 10.0)
    
    # 2. SoC target parameters
    config["preferred_soc_min_base"] = trial.suggest_float("preferred_soc_min_base", 0.2, 0.4)
    config["preferred_soc_max_base"] = trial.suggest_float("preferred_soc_max_base", 0.6, 0.8)
    config["high_soc_threshold"] = trial.suggest_float("high_soc_threshold", 0.7, 0.9)
    
    # 3. Price thresholds
    if config["use_percentile_price_thresholds"]:
        config["low_price_percentile"] = trial.suggest_float("low_price_percentile", 10.0, 40.0)
        config["high_price_percentile"] = trial.suggest_float("high_price_percentile", 60.0, 90.0)
    
    # 4. RL Algorithm Parameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_int("n_steps", 1024, 4096)
    batch_size = trial.suggest_int("batch_size", 64, 512)
    n_epochs = trial.suggest_int("n_epochs", 5, 30)
    gamma = trial.suggest_float("gamma", 0.85, 0.99)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.1, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    
    # Override training config params
    config["short_term_learning_rate"] = learning_rate
    config["short_term_n_steps"] = n_steps 
    config["short_term_batch_size"] = batch_size
    config["short_term_n_epochs"] = n_epochs
    config["short_term_gamma"] = gamma
    config["short_term_ent_coeff"] = ent_coef
    config["short_term_gae_lambda"] = gae_lambda
    
    # Use shorter simulation for optimization efficiency
    config["simulation_days"] = 7  # Even shorter for initial trials
    
    # Create training environment
    env = HomeEnergyEnv(config=config)
    
    # Create agent with sampled hyperparameters
    model = PPO(
        "MultiInputPolicy", 
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        verbose=0
    )
    
    # Limited training for HPO efficiency
    try:
        total_timesteps = min(config.get("short_term_timesteps", 100_000), 50_000)  # Even shorter for initial testing
        logger.info(f"Training model for {total_timesteps} timesteps")
        model.learn(total_timesteps=total_timesteps)
        
        # Run evaluation episodes
        num_eval_episodes = 2  # Reduced for efficiency
        rewards = []
        correlation_scores = []
        peak_scores = []
        soc_scores = []
        
        for i in range(num_eval_episodes):
            logger.info(f"Running evaluation episode {i+1}/{num_eval_episodes}")
            # Using the updated evaluate_episode function
            try:
                episode_data = evaluate_episode(model, config)
                metrics = calculate_performance_metrics(episode_data, config)
                
                rewards.append(metrics["total_reward"])
                correlation_scores.append(metrics["price_power_correlation"])
                
                # Get peak score (lower is better)
                peak_score = metrics.get("peak_metrics", {}).get("peak_rolling_average", 0)
                peak_scores.append(peak_score)
                
                # Get SoC management score
                soc_violations = metrics.get("soc_metrics", {}).get("violation_percentage", 0)
                soc_scores.append(100 - soc_violations)  # Higher is better (100% = no violations)
            except Exception as e:
                logger.error(f"Error in evaluation episode {i+1}: {e}")
                logger.error(traceback.format_exc())
                # Provide default values if evaluation fails
                rewards.append(0)
                correlation_scores.append(0)
                peak_scores.append(10)  # High default peak (worse)
                soc_scores.append(0)  # Low default SoC score (worse)
            
        # Multi-objective optimization: 
        # 1. Maximize reward (minimize negative reward)
        # 2. Maximize price-power correlation
        # 3. Minimize peak average
        # 4. Maximize SoC management (minimize violations)
        mean_reward = np.mean(rewards)
        mean_correlation = np.mean(correlation_scores)
        mean_peak = np.mean(peak_scores)
        mean_soc_score = np.mean(soc_scores)
        
        # Weighted score based on multiple objectives
        # We scale components to make them comparable and give different weights
        # Higher score is better
        score = (
            mean_reward + 
            (mean_correlation * 1000) +
            (-mean_peak * 10) +   # Negative because lower peaks are better
            (mean_soc_score * 10)
        )
        
        # Log the results
        trial.set_user_attr("mean_reward", mean_reward)
        trial.set_user_attr("mean_correlation", mean_correlation)
        trial.set_user_attr("mean_peak", mean_peak)
        trial.set_user_attr("mean_soc_score", mean_soc_score)
        
        logger.info(f"Trial {trial.number}: mean_reward={mean_reward:.2f}, "
                   f"correlation={mean_correlation:.4f}, peak={mean_peak:.2f}, "
                   f"soc_score={mean_soc_score:.2f}")
                   
        return score
        
    except Exception as e:
        logger.error(f"Error during trial {trial.number}: {e}")
        logger.error(traceback.format_exc())
        return float('-inf')  # Return worst score on error


def plot_optimization_history(study, save_dir="src/rl/simulations/results"):
    """Plot the optimization history."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot optimization history
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig1.write_image(os.path.join(save_dir, "hpo_history.png"))
    
    # Plot parameter importances
    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.write_image(os.path.join(save_dir, "hpo_param_importance.png"))
    
    # Plot parallel coordinate
    fig3 = optuna.visualization.plot_parallel_coordinate(study)
    fig3.write_image(os.path.join(save_dir, "hpo_parallel_coordinate.png"))


def run_hpo(n_trials=50, timeout=None):
    """Run the hyperparameter optimization process."""
    logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
    
    # Create Optuna study
    study_name = f"home_energy_rl_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage_name = f"sqlite:///src/rl/hpo_{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        sampler=TPESampler(),
        direction="maximize",  # We want to maximize our score
        load_if_exists=True
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    # Print results
    logger.info(f"Best parameters: {study.best_params}")
    logger.info(f"Best value: {study.best_value}")
    logger.info(f"Best trial: {study.best_trial.number}")
    
    # Log additional metrics from best trial
    logger.info(f"Best trial mean reward: {study.best_trial.user_attrs['mean_reward']}")
    logger.info(f"Best trial correlation: {study.best_trial.user_attrs['mean_correlation']}")
    logger.info(f"Best trial peak average: {study.best_trial.user_attrs['mean_peak']}")
    logger.info(f"Best trial SoC score: {study.best_trial.user_attrs['mean_soc_score']}")
    
    # Save best parameters to a file
    with open(f"src/rl/best_hpo_params_{study_name}.txt", "w") as f:
        f.write("Best parameters:\n")
        for param, value in study.best_params.items():
            f.write(f"{param}: {value}\n")
        f.write(f"\nBest value: {study.best_value}\n")
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Mean reward: {study.best_trial.user_attrs['mean_reward']}\n")
        f.write(f"Correlation: {study.best_trial.user_attrs['mean_correlation']}\n")
        f.write(f"Peak average: {study.best_trial.user_attrs['mean_peak']}\n")
        f.write(f"SoC score: {study.best_trial.user_attrs['mean_soc_score']}\n")
    
    # Apply best parameters to config
    best_config = get_config_dict()
    for param, value in study.best_params.items():
        best_config[param] = value
    
    # Save best config as JSON for easier loading
    with open(f"src/rl/best_hpo_config_{study_name}.json", "w") as f:
        # Convert best config to a serializable format (some numpy values might not be directly serializable)
        serializable_config = {k: float(v) if isinstance(v, np.float32) else v for k, v in best_config.items() 
                               if isinstance(v, (int, float, bool, str)) or v is None}
        json.dump(serializable_config, f, indent=2)
        
    # Create visualization plots
    try:
        plot_optimization_history(study)
    except Exception as e:
        logger.error(f"Error plotting optimization history: {e}")
    
    return study.best_params


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization for RL agent")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials to run")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds (None = no timeout)")
    parser.add_argument("--apply-best", action="store_true", help="Apply the best parameters from the study to config.py")
    
    args = parser.parse_args()
    
    try:
        best_params = run_hpo(n_trials=args.trials, timeout=args.timeout)
        print(f"Optimization complete. Best parameters: {best_params}")
        
        if args.apply_best:
            print("Applying best parameters to config.py...")
            # Implementation would need to modify the config.py file
            # This is complex and requires careful handling of Python code
            print("Note: Manual application of parameters is recommended for safety.")
            print("Please copy values from the best_hpo_params_*.txt file to config.py")
    except Exception as e:
        logger.error(f"Error in optimization process: {e}")
        logger.error(traceback.format_exc()) 