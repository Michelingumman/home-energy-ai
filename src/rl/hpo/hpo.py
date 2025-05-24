#!/usr/bin/env python3
"""
Hyperparameter Optimization for the RL agent using Optuna.

This script implements Bayesian optimization to find optimal values for:
1. Reward component scaling factors (primarily w_* weights)
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
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sb3_contrib.ppo_recurrent import RecurrentPPO
from src.rl.custom_env import HomeEnergyEnv
from src.rl.config import get_config_dict
from src.rl.evaluate_agent import evaluate_episode, calculate_performance_metrics


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("hpo")


def objective(trial):
    """
    Optuna objective function.
    Focuses on optimizing w_* weights and RL hyperparameters.
    Base reward factors are mostly taken from config.py.
    """
    # Load base configuration
    config = get_config_dict()
    
    # Sample hyperparameters to optimize
    
    # 1. Reward Component Weights (Primary focus of HPO)
    config["w_grid"] = trial.suggest_float("w_grid", 0.5, 2.5) # Expanded slightly
    config["w_cap"] = trial.suggest_float("w_cap", 0.1, 2.0)  # Expanded slightly
    config["w_deg"] = trial.suggest_float("w_deg", 0.1, 1.5)  # Expanded slightly
    config["w_soc"] = trial.suggest_float("w_soc", 1.0, 4.0) # Adjusted range based on previous results
    config["w_shape"] = trial.suggest_float("w_shape", 0.5, 2.0) # Expanded slightly
    config["w_night"] = trial.suggest_float("w_night", 0.2, 1.5) 
    config["w_arbitrage"] = trial.suggest_float("w_arbitrage", 0.2, 2.0) # Expanded slightly
    config["w_export"] = trial.suggest_float("w_export", 0.1, 1.5)   # Expanded slightly
    config["w_action_mod"] = trial.suggest_float("w_action_mod", 0.5, 5.0) # Adjusted range based on previous results
    config["w_chain"] = trial.suggest_float("w_chain", 0.5, 3.0) 
    config["w_morning"] = trial.suggest_float("w_morning", 0.2, 2.0)

    # 2. Base Reward Factors (Mostly use values from config.py, HPO for a few if needed)
    # config["soc_limit_penalty_factor"] = trial.suggest_float("soc_limit_penalty_factor", 50.0, 200.0) 
    # config["preferred_soc_reward_factor"] = trial.suggest_float("preferred_soc_reward_factor", 10.0, 50.0) # Keep if uncertain
    # config["peak_penalty_factor"] = trial.suggest_float("peak_penalty_factor", 20.0, 100.0)
    config["action_modification_penalty"] = trial.suggest_float("action_modification_penalty", 1.0, 20.0) # Keep HPO for this
    # grid_cost_scaling_factor is a direct config, not typically HPO'd if w_grid is HPO'd

    # Additional arbitrage parameters (base factors)
    # if config["enable_explicit_arbitrage_reward"]:
        # config["charge_at_low_price_reward_factor"] = trial.suggest_float("charge_at_low_price_reward_factor", 5.0, 20.0)
        # config["discharge_at_high_price_reward_factor"] = trial.suggest_float("discharge_at_high_price_reward_factor", 10.0, 40.0)
        
        # if config["use_percentile_price_thresholds"]:
            # config["low_price_percentile"] = trial.suggest_float("low_price_percentile", 20.0, 40.0)
            # config["high_price_percentile"] = trial.suggest_float("high_price_percentile", 60.0, 80.0)
    
    # Export parameters (base factor)
    # config["export_reward_bonus_ore_kwh"] = trial.suggest_float("export_reward_bonus_ore_kwh", 30.0, 100.0)
    
    # 3. SoC target parameters (Mostly use values from config.py)
    # config["preferred_soc_min_base"] = trial.suggest_float("preferred_soc_min_base", 0.25, 0.35)
    # config["preferred_soc_max_base"] = trial.suggest_float("preferred_soc_max_base", 0.65, 0.75)
    config["high_soc_penalty_multiplier"] = trial.suggest_float("high_soc_penalty_multiplier", 1.0, 3.0) # Adjusted range
    
    # 4. RL Algorithm Parameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096]) # Using categorical for common values
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512]) # Using categorical
    n_epochs = trial.suggest_int("n_epochs", 5, 20) # Reduced upper range
    gamma = trial.suggest_float("gamma", 0.90, 0.995) # Slightly adjusted
    ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.05, log=True) # Adjusted range
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    
    # 5. Recurrent-specific parameters
    lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [32, 64, 128]) # Using categorical
    n_lstm_layers = trial.suggest_int("n_lstm_layers", 1, 2)
    
    # Override training config params from PPO section in config.py
    config["short_term_learning_rate"] = learning_rate
    config["short_term_n_steps"] = n_steps 
    config["short_term_batch_size"] = batch_size
    config["short_term_n_epochs"] = n_epochs
    config["short_term_gamma"] = gamma
    config["short_term_ent_coeff"] = ent_coef
    config["short_term_gae_lambda"] = gae_lambda
    
    # Override recurrent PPO specific parameters in config.py
    config["lstm_hidden_size"] = lstm_hidden_size
    config["n_lstm_layers"] = n_lstm_layers
    
    # Use shorter simulation for optimization efficiency
    config["simulation_days"] = 7 # Kept short for HPO speed
    config["simulation_days_eval"] = 7 # Align eval days for HPO trial
    
    # Ensure data augmentation is OFF for HPO evaluation consistency
    config["use_data_augmentation"] = False

    # Create training environment
    # Ensure the env uses a flattened observation space if it's Dict, as RecurrentPPO expects that.
    # HomeEnergyEnv should internally handle or provide a way to get a flattened observation if needed,
    # or we wrap it here. For now, assume HomeEnergyEnv is compatible or RecurrentPPO handles it.
    env = HomeEnergyEnv(config=config)
    
    # Create agent with sampled hyperparameters using RecurrentPPO
    policy_kwargs = dict(
        lstm_hidden_size=lstm_hidden_size,
        n_lstm_layers=n_lstm_layers, # Make sure this is passed if agent uses it
        # net_arch is often defined within the policy class or taken from defaults.
        # If you need to customize net_arch (e.g., pi=[64,64], vf=[64,64]), ensure it's
        # correctly structured for MlpLstmPolicy.
        # Example: net_arch=[dict(pi=[64, 64], vf=[64, 64])] or net_arch=[64, dict(vf=[64])] etc.
        # Check sb3-contrib docs for MlpLstmPolicy's net_arch structure if customizing.
        # For now, let's assume default net_arch or simple [64,64] for pi/vf if that's intended.
        net_arch=dict(pi=[64, 64], vf=[64, 64]) # A common simple architecture
    )
    
    model = RecurrentPPO(
        "MlpLstmPolicy", 
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=0 # Keep verbose low for HPO
    )
    
    # Limited training for HPO efficiency
    try:
        # Use a much shorter total_timesteps for HPO trials
        hpo_trial_timesteps = config.get("hpo_trial_timesteps", 20000) # e.g., 20k-50k per trial
        
        logger.info(f"Trial {trial.number}: Training model for {hpo_trial_timesteps} timesteps...")
        # Log only a subset of params for brevity
        loggable_params = {
            k: v for k, v in config.items() 
            if k.startswith('w_') or k in [
                "learning_rate", "n_steps", "batch_size", "lstm_hidden_size",
                "action_modification_penalty", "high_soc_penalty_multiplier" # Example of kept base factors
            ]
        }
        logger.debug(f"Trial {trial.number} Config (subset): {json.dumps(loggable_params, indent=2)}")

        model.learn(total_timesteps=hpo_trial_timesteps, progress_bar=False) # Disable progress bar for cleaner logs
        
        # Run evaluation episodes
        num_eval_episodes = config.get("eval_episodes_hpo", 2) # Reduced for efficiency
        rewards = []
        price_power_correlations = []
        peak_rolling_averages = []
        soc_violation_percentages = []
        grid_costs_total = [] 
        battery_costs_total = []
        arbitrage_rewards_total = [] # Ensure this is initialized
        export_rewards_total = []    # Ensure this is initialized


        for i in range(num_eval_episodes):
            logger.info(f"Trial {trial.number}: Running evaluation episode {i+1}/{num_eval_episodes}")
            try:
                # Important: evaluate_episode should also use RecurrentPPO's state handling
                episode_data = evaluate_episode(model, config, is_recurrent=True) 
                metrics = calculate_performance_metrics(episode_data, config)
                
                rewards.append(metrics.get("total_reward", 0))
                price_power_correlations.append(metrics.get("price_power_correlation", 0))
                peak_rolling_averages.append(metrics.get("peak_rolling_average", 0))
                soc_violation_percentages.append(metrics.get("violation_percentage", 100))
                
                grid_costs_total.append(metrics.get("reward_components", {}).get("reward_grid_cost", 0) * -1) 
                battery_costs_total.append(metrics.get("reward_components", {}).get("reward_battery_cost", 0) * -1)
                arbitrage_rewards_total.append(metrics.get("reward_components", {}).get("reward_arbitrage_bonus", 0))
                export_rewards_total.append(metrics.get("reward_components", {}).get("reward_export_bonus", 0))


            except Exception as e:
                logger.error(f"Error in evaluation episode {i+1} for trial {trial.number}: {e}")
                logger.error(traceback.format_exc())
                rewards.append(float('-inf'))
                price_power_correlations.append(-1) 
                peak_rolling_averages.append(999) 
                soc_violation_percentages.append(100)
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
        mean_arbitrage_reward = np.mean(arbitrage_rewards_total) if arbitrage_rewards_total else float('-inf') # Default for maximization
        mean_export_reward = np.mean(export_rewards_total) if export_rewards_total else float('-inf') # Default for maximization

        
        # Define the objective score to maximize
        # This score needs to be carefully balanced.
        score = (
            (mean_reward * config.get("hpo_score_w_reward", 1.0)) +              
            (mean_correlation * config.get("hpo_score_w_correlation", 10.0)) +    
            (-mean_peak_avg * config.get("hpo_score_w_peak", 0.1)) +           
            (-mean_soc_violations_pct * config.get("hpo_score_w_soc_viol", 0.2))+ 
            (-mean_grid_cost * config.get("hpo_score_w_grid_cost", 0.001)) +       
            (-mean_battery_cost * config.get("hpo_score_w_battery_cost", 0.01)) +  
            (mean_arbitrage_reward * config.get("hpo_score_w_arbitrage", 0.05)) +   
            (mean_export_reward * config.get("hpo_score_w_export", 0.02))          
        )

        if np.isnan(score) or np.isinf(score):
            logger.warning(f"Trial {trial.number} resulted in NaN or Inf score. Setting to -1e9.")
            score = -1e9 
        
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
        return float('-inf')


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
    best_params_with_metrics = {
        "params": best_params,
        "value": study.best_trial.value,
        "user_attrs": study.best_trial.user_attrs
    }
    # Ensure the output directory for best_hpo_params.json exists
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(best_params_with_metrics, f, indent=4)
    logger.info(f"Best hyperparameters saved to {save_path}")

def run_hpo(n_trials_override=None, study_name_suffix=None, storage_db_template=None, output_dir_template=None, timeout_hours_override=None):
    """
    Runs the Hyperparameter Optimization process.
    """
    N_TRIALS_DEFAULT = 50 # Default HPO trials
    STUDY_NAME_BASE = "home_energy_rl_hpo"
    TIMEOUT_HOURS_DEFAULT = 6 # Default timeout

    n_trials = n_trials_override if n_trials_override is not None else N_TRIALS_DEFAULT
    current_study_name_suffix = study_name_suffix if study_name_suffix is not None else "v4_recurrent_weights" # New suffix
    study_name = f"{STUDY_NAME_BASE}_{current_study_name_suffix}"
    
    # Default DB path to be relative to the script or a logs directory
    default_db_path = f"sqlite:///src/rl/logs/{study_name}.db"
    storage_db = storage_db_template.format(STUDY_NAME=study_name.replace(' ', '_')) if storage_db_template else default_db_path
    Path(os.path.dirname(storage_db.replace("sqlite:///", ""))).mkdir(parents=True, exist_ok=True)


    default_output_dir = f"src/rl/logs/hpo_results/{study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = output_dir_template.format(STUDY_NAME=study_name.replace(' ', '_'), TIMESTAMP=datetime.now().strftime('%Y%m%d_%H%M%S')) if output_dir_template else default_output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timeout_hours = timeout_hours_override if timeout_hours_override is not None else TIMEOUT_HOURS_DEFAULT

    logger.info(f"Starting HPO: Trials={n_trials}, Study Name='{study_name}', DB='{storage_db}', Output='{output_dir}', Timeout={timeout_hours}h")

    sampler = TPESampler(seed=42) # Or other samplers like CmaEsSampler
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_db,
        direction="maximize", # We want to maximize the score
        sampler=sampler,
        load_if_exists=True # Load existing study to continue
    )

    try:
        study.optimize(objective, n_trials=n_trials, timeout=timeout_hours * 3600 if timeout_hours else None)
    except KeyboardInterrupt:
        logger.info("HPO interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred during HPO: {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info(f"Number of finished trials: {len(study.trials)}")
        if study.trials: # Check if there are any trials
            # Sort trials by value to find the best one if study.best_trial is None
            # (can happen if all trials error out or return NaN/inf)
            valid_trials = [t for t in study.trials if t.value is not None and not (isinstance(t.value, float) and (np.isnan(t.value) or np.isinf(t.value)))]
            if valid_trials:
                best_trial_overall = max(valid_trials, key=lambda t: t.value)
                logger.info(f"Best trial overall (manually checked): {best_trial_overall.number}")
                logger.info(f"  Value (Score): {best_trial_overall.value}")
                logger.info("  Params: ")
                for key, value in best_trial_overall.params.items():
                    logger.info(f"    {key}: {value}")
                
                # Save best parameters using the overall best trial
                save_best_params(study, save_path=os.path.join(output_dir, "best_hpo_params.json"))
                 # Pass the best_trial_overall to plotting functions if Optuna's best_trial is problematic
                plot_optimization_history(study, save_dir=output_dir) # Optuna's plot functions usually use study.best_trial
                logger.info(f"HPO plots saved to {output_dir}")

            elif study.best_trial: # Fallback to Optuna's best_trial if it exists
                logger.info(f"Best trial (Optuna's): {study.best_trial.number}")
                logger.info(f"  Value (Score): {study.best_trial.value}")
                # ... log params and save ...
                save_best_params(study, save_path=os.path.join(output_dir, "best_hpo_params.json"))
                plot_optimization_history(study, save_dir=output_dir)
                logger.info(f"HPO plots saved to {output_dir}")
            else:
                logger.info("No valid trials were completed or no best trial found.")
        else:
            logger.info("No trials were run in this HPO session.")


    logger.info(f"HPO for study '{study_name}' finished.")
    return study


if __name__ == "__main__":
    # Example of running directly
    # You can use argparse here for more flexible command-line configuration
    run_hpo(n_trials_override=5, timeout_hours_override=0.5, study_name_suffix="test_run") # Short run for testing
    logger.info("HPO script finished (direct run).") 