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
    
    # 1. Reward Component Weights
    config["w_grid"] = trial.suggest_float("w_grid", 0.5, 2.0)
    config["w_cap"] = trial.suggest_float("w_cap", 0.1, 1.5) # Adjusted range slightly
    config["w_deg"] = trial.suggest_float("w_deg", 0.1, 1.0) # Adjusted range slightly
    config["w_soc"] = trial.suggest_float("w_soc", 1.0, 3.0) # Adjusted range slightly
    config["w_shape"] = trial.suggest_float("w_shape", 0.5, 1.5) # Adjusted range slightly
    config["w_night"] = trial.suggest_float("w_night", 0.5, 1.5) # Adjusted range slightly
    config["w_arbitrage"] = trial.suggest_float("w_arbitrage", 0.5, 1.5)
    config["w_export"] = trial.suggest_float("w_export", 0.1, 1.0) # Adjusted range slightly
    config["w_action_mod"] = trial.suggest_float("w_action_mod", 1.0, 5.0) # Adjusted range slightly
    
    # 2. Base Reward Factors (no secondary scaling factors anymore)
    config["soc_limit_penalty_factor"] = trial.suggest_float("soc_limit_penalty_factor", 50.0, 200.0) 
    config["preferred_soc_reward_factor"] = trial.suggest_float("preferred_soc_reward_factor", 10.0, 50.0)
    config["peak_penalty_factor"] = trial.suggest_float("peak_penalty_factor", 20.0, 100.0)
    config["action_modification_penalty"] = trial.suggest_float("action_modification_penalty", 5.0, 20.0)
    # grid_cost_scaling_factor is a direct config, not typically HPO'd if w_grid is HPO'd

    # Additional arbitrage parameters (base factors)
    if config["enable_explicit_arbitrage_reward"]:
        config["charge_at_low_price_reward_factor"] = trial.suggest_float("charge_at_low_price_reward_factor", 5.0, 20.0)
        config["discharge_at_high_price_reward_factor"] = trial.suggest_float("discharge_at_high_price_reward_factor", 10.0, 40.0)
        
        if config["use_percentile_price_thresholds"]:
            config["low_price_percentile"] = trial.suggest_float("low_price_percentile", 20.0, 40.0)
            config["high_price_percentile"] = trial.suggest_float("high_price_percentile", 60.0, 80.0)
    
    # Export parameters (base factor)
    config["export_reward_bonus_ore_kwh"] = trial.suggest_float("export_reward_bonus_ore_kwh", 30.0, 100.0)
    
    # 3. SoC target parameters
    config["preferred_soc_min_base"] = trial.suggest_float("preferred_soc_min_base", 0.25, 0.35)
    config["preferred_soc_max_base"] = trial.suggest_float("preferred_soc_max_base", 0.65, 0.75)
    
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
    config["simulation_days"] = 7
    
    # Ensure data augmentation is OFF for HPO evaluation consistency
    config["use_data_augmentation"] = False

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
        total_timesteps = config.get("short_term_timesteps", 10000) # Reduced for HPO trial speed
        logger.info(f"Trial {trial.number}: Training model for {total_timesteps} timesteps with config: {json.dumps({k: v for k, v in config.items() if k.startswith('w_') or k.endswith('_factor') or k.endswith('_scale')}, indent=2)}")
        model.learn(total_timesteps=total_timesteps)
        
        # Run evaluation episodes
        num_eval_episodes = config.get("eval_episodes_hpo", 2) # Reduced for efficiency, allow config override
        rewards = []
        price_power_correlations = []
        peak_rolling_averages = []
        soc_violation_percentages = []
        grid_costs_total = [] # Total cost for the episode
        battery_costs_total = [] # Total cost for the episode
        arbitrage_rewards_total = []
        export_rewards_total = []

        for i in range(num_eval_episodes):
            logger.info(f"Trial {trial.number}: Running evaluation episode {i+1}/{num_eval_episodes}")
            try:
                episode_data = evaluate_episode(model, config) # evaluate_episode uses its own env with same config
                metrics = calculate_performance_metrics(episode_data, config)
                
                rewards.append(metrics.get("total_reward", 0))
                price_power_correlations.append(metrics.get("price_power_correlation", 0))
                peak_rolling_averages.append(metrics.get("peak_rolling_average", 0))
                soc_violation_percentages.append(metrics.get("violation_percentage", 100)) # Default to 100% violations if not found
                
                # Sum of cost components (these are usually negative or zero)
                grid_costs_total.append(metrics.get("reward_components", {}).get("reward_grid_cost", 0) * -1) # Make it positive cost
                battery_costs_total.append(metrics.get("reward_components", {}).get("reward_battery_cost", 0) * -1) # Make it positive cost
                arbitrage_rewards_total.append(metrics.get("reward_components", {}).get("reward_arbitrage_bonus", 0))
                export_rewards_total.append(metrics.get("reward_components", {}).get("reward_export_bonus", 0))

            except Exception as e:
                logger.error(f"Error in evaluation episode {i+1} for trial {trial.number}: {e}")
                logger.error(traceback.format_exc())
                # Provide default/worst-case values if evaluation fails
                rewards.append(float('-inf'))
                price_power_correlations.append(-1) # Worst correlation
                peak_rolling_averages.append(999)  # High default peak (worse)
                soc_violation_percentages.append(100)    # Max violations
                grid_costs_total.append(99999) 
                battery_costs_total.append(9999)
                arbitrage_rewards_total.append(float('-inf'))
                export_rewards_total.append(float('-inf'))
            
        # Aggregate metrics
        mean_reward = np.mean(rewards) if rewards else float('-inf')
        mean_correlation = np.mean(price_power_correlations) if price_power_correlations else -1
        mean_peak_avg = np.mean(peak_rolling_averages) if peak_rolling_averages else 999
        mean_soc_violations_pct = np.mean(soc_violation_percentages) if soc_violation_percentages else 100
        mean_grid_cost = np.mean(grid_costs_total) if grid_costs_total else 99999
        mean_battery_cost = np.mean(battery_costs_total) if battery_costs_total else 9999
        mean_arbitrage_reward = np.mean(arbitrage_rewards_total) if arbitrage_rewards_total else float('-inf')
        mean_export_reward = np.mean(export_rewards_total) if export_rewards_total else float('-inf')
        
        # Define the objective score to maximize
        # Components should be positive if good, negative if bad before weighting.
        # We aim to MAXIMIZE this score.
        score = (
            (mean_reward * 1.0) +               # Primary objective
            (mean_correlation * 10.0) +         # Encourage good price correlation (range -1 to 1)
            (-mean_peak_avg * 0.1) +            # Penalize high peak average (lower is better)
            (-mean_soc_violations_pct * 0.2)+ # Penalize SoC violations (lower is better)
            (-mean_grid_cost * 0.001) +         # Penalize grid cost (lower is better)
            (-mean_battery_cost * 0.01) +       # Penalize battery cost (lower is better)
            (mean_arbitrage_reward * 0.05) +    # Encourage arbitrage
            (mean_export_reward * 0.02)         # Encourage export
        )

        if np.isnan(score) or np.isinf(score):
            logger.warning(f"Trial {trial.number} resulted in NaN or Inf score. Setting to -1e9.")
            score = -1e9 # A very bad score
        
        # Log the results for Optuna
        trial.set_user_attr("mean_reward", mean_reward)
        trial.set_user_attr("mean_correlation", mean_correlation)
        trial.set_user_attr("mean_peak_avg", mean_peak_avg)
        trial.set_user_attr("mean_soc_violations_pct", mean_soc_violations_pct)
        trial.set_user_attr("mean_grid_cost", mean_grid_cost)
        trial.set_user_attr("mean_battery_cost", mean_battery_cost)
        trial.set_user_attr("mean_arbitrage_reward", mean_arbitrage_reward)
        trial.set_user_attr("mean_export_reward", mean_export_reward)
        trial.set_user_attr("final_score", score)
        
        logger.info(f"Trial {trial.number} finished. Score: {score:.4f}, Mean Reward: {mean_reward:.2f}, Correlation: {mean_correlation:.3f}, PeakAvg: {mean_peak_avg:.2f}, GridCost: {mean_grid_cost:.2f}")
                   
        return score
        
    except Exception as e:
        logger.error(f"Unhandled error during trial {trial.number}: {e}")
        logger.error(traceback.format_exc())
        return float('-inf')  # Return worst score on error


def plot_optimization_history(study, save_dir="src/rl/simulations/results"):
    """Plot the optimization history."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot optimization history
    try:
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_image(os.path.join(save_dir, "hpo_history.png"))
    except Exception as e:
        logger.warning(f"Could not plot optimization history: {e}")
    
    # Plot parameter importances
    try:
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_image(os.path.join(save_dir, "hpo_param_importance.png"))
    except Exception as e:
        logger.warning(f"Could not plot param importances: {e}")
    
    # Plot parallel coordinate
    try:
        fig3 = optuna.visualization.plot_parallel_coordinate(study)
        fig3.write_image(os.path.join(save_dir, "hpo_parallel_coordinate.png"))
    except Exception as e:
        logger.warning(f"Could not plot parallel coordinate: {e}")


def save_best_params(study, save_path="src/rl/best_hpo_params.json"):
    """Save the best hyperparameters to a JSON file."""
    best_params = study.best_trial.params
    # Add user attributes (metrics) to the saved file for context
    best_params_with_metrics = {
        "params": best_params,
        "value": study.best_trial.value,
        "user_attrs": study.best_trial.user_attrs
    }
    with open(save_path, 'w') as f:
        json.dump(best_params_with_metrics, f, indent=4)
    logger.info(f"Best hyperparameters saved to {save_path}")

def run_hpo(n_trials_override=None, study_name_suffix=None, storage_db_template=None, output_dir_template=None, timeout_hours_override=None):
    """
    Runs the Hyperparameter Optimization process.

    Args:
        n_trials_override (int, optional): Number of Optuna trials. Defaults to N_TRIALS_DEFAULT.
        study_name_suffix (str, optional): Suffix for the study name. Defaults to "v3_reward_rework".
        storage_db_template (str, optional): Template for the storage DB path. Defaults to "sqlite:///src/rl/{STUDY_NAME}.db".
        output_dir_template (str, optional): Template for the output directory. Defaults to "src/rl/logs/hpo_results/{STUDY_NAME}_{TIMESTAMP}".
        timeout_hours_override (int, optional): Timeout for the optimization in hours. Defaults to TIMEOUT_HOURS_DEFAULT.
    """
    # --- Configuration ---
    N_TRIALS_DEFAULT = 50
    STUDY_NAME_BASE = "home_energy_rl_hpo"
    TIMEOUT_HOURS_DEFAULT = 6

    n_trials = n_trials_override if n_trials_override is not None else N_TRIALS_DEFAULT
    current_study_name_suffix = study_name_suffix if study_name_suffix is not None else "v3_reward_rework"
    study_name = f"{STUDY_NAME_BASE}_{current_study_name_suffix}"
    
    current_storage_db_template = storage_db_template if storage_db_template is not None else "sqlite:///src/rl/{STUDY_NAME}.db"
    storage_db = current_storage_db_template.format(STUDY_NAME=study_name.replace(' ', '_'))
    
    current_output_dir_template = output_dir_template if output_dir_template is not None else "src/rl/logs/hpo_results/{STUDY_NAME}_{TIMESTAMP}"
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = current_output_dir_template.format(STUDY_NAME=study_name.replace(' ', '_'), TIMESTAMP=timestamp_str)
    
    timeout_hours = timeout_hours_override if timeout_hours_override is not None else TIMEOUT_HOURS_DEFAULT

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting HPO: Trials={n_trials}, Study Name='{study_name}', DB='{storage_db}', Output='{output_dir}', Timeout={timeout_hours}h")

    # --- Setup Study ---
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_db,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True
    )

    # --- Run Optimization ---
    try:
        study.optimize(objective, n_trials=n_trials, timeout=timeout_hours * 3600 if timeout_hours else None)
    except KeyboardInterrupt:
        logger.info("HPO interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred during HPO: {e}")
        logger.error(traceback.format_exc())
    finally:
        # --- Log and Save Results ---
        logger.info(f"Number of finished trials: {len(study.trials)}")
        if study.best_trial:
            logger.info(f"Best trial: {study.best_trial.number}")
            logger.info(f"  Value (Score): {study.best_trial.value}")
            logger.info("  Params: ")
            for key, value in study.best_trial.params.items():
                logger.info(f"    {key}: {value}")
            
            # Save best parameters
            save_best_params(study, save_path=os.path.join(output_dir, "best_hpo_params.json"))
            
            # Plot results
            plot_optimization_history(study, save_dir=output_dir)
            logger.info(f"HPO plots saved to {output_dir}")
        else:
            logger.info("No trials were completed or no best trial found.")

    logger.info(f"HPO for study '{study_name}' finished.")
    return study


if __name__ == "__main__":
    # Example of running directly, can be configured with argparse if needed
    run_hpo(n_trials_override=2, timeout_hours_override=0.1) # Short run for direct testing
    logger.info("HPO script finished (direct run).") 