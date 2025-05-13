"""
This script trains a simple RL agent on the home energy environment
using real predictions for prices and solar production.
"""
import os
import sys
import argparse
import logging
import numpy as np
import traceback
from typing import Dict, Any, Optional

# Add the project root to the Python path to ensure imports work properly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Import stable-baselines3 
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.evaluation import evaluate_policy
    
    # Import our custom modules
    from src.rl.custom_env import HomeEnergyEnv
    
    import json
    
    logger.info("Successfully imported all required modules")
except ImportError as e:
    logger.error(f"Error importing modules: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)

def load_config(config_path="src/rl/rl_config.json"):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config
    
def create_environment(config, use_predictions=True):
    """Create the environment with the given configuration."""
    # Create base environment 
    env_kwargs = {
        "battery_capacity": config.get("battery_capacity", 22.0),
        "simulation_days": config.get("simulation_days", 7),
        "peak_penalty_factor": config.get("peak_penalty_factor", 10.0),
        "comfort_bonus_factor": config.get("comfort_bonus_factor", 2.0),
        "random_weather": config.get("random_weather", True),
        "use_price_predictions": use_predictions,
        "use_solar_predictions": use_predictions
    }
    
    base_env = HomeEnergyEnv(**env_kwargs)
    
    # Wrap with Monitor for logging
    monitored_env = Monitor(base_env)
    
    return monitored_env

def train_agent(config, env, total_timesteps=50000):
    """Train a PPO agent on the environment."""
    # Create model directory
    model_dir = "src/rl/models/single_agent"
    os.makedirs(model_dir, exist_ok=True)
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="ppo_agent"
    )
    
    # Set up model parameters
    model_params = {
        "learning_rate": config.get("short_term_learning_rate", 3e-4),
        "gamma": config.get("short_term_gamma", 0.99),
        "n_steps": config.get("short_term_n_steps", 2048),
        "batch_size": config.get("short_term_batch_size", 64),
        "n_epochs": config.get("short_term_n_epochs", 10),
        "verbose": 1
    }
    
    # Create the agent
    logger.info("Creating PPO agent")
    agent = PPO("MultiInputPolicy", env, **model_params)
    
    # Train the agent
    logger.info(f"Training agent for {total_timesteps} timesteps")
    agent.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save the final model
    final_model_path = os.path.join(model_dir, "final_model")
    agent.save(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    return agent

def evaluate_agent(agent, env, n_eval_episodes=5):
    """Evaluate an agent on the environment."""
    logger.info(f"Evaluating agent for {n_eval_episodes} episodes")
    
    mean_reward, std_reward = evaluate_policy(
        agent,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True
    )
    
    logger.info(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward

def run_simulation(agent, env, render=False):
    """Run a simulation with the trained agent."""
    logger.info("Running simulation with trained agent...")
    
    # Reset environment
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    
    # Run one episode
    while not done:
        # Get action from agent
        action, _ = agent.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        
        # Print info
        logger.info(f"Hour: {info['current_hour']}, Reward: {reward:.2f}, SoC: {info['battery_soc']:.2f}")
        logger.info(f"  Solar: {info['solar_production']:.2f} kWh, Price: {info['current_price']:.2f}, Net consumption: {info['net_consumption']:.2f} kWh")
        
        if render:
            env.render()
    
    logger.info(f"Simulation complete. Total reward: {episode_reward:.2f}")
    return episode_reward

def main(args):
    """Main function to run training or evaluation."""
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Successfully loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        logger.info("Using default parameters")
        config = {}
    
    # Create environment
    try:
        env = create_environment(config, use_predictions=args.use_predictions)
        logger.info("Created environment with predictions" if args.use_predictions else "Created environment without predictions")
    except Exception as e:
        logger.error(f"Error creating environment: {str(e)}")
        raise
    
    if args.train:
        try:
            # Train agent
            logger.info(f"Starting training for {args.timesteps} timesteps")
            agent = train_agent(config, env, total_timesteps=args.timesteps)
            
            # Evaluate agent
            logger.info(f"Evaluating agent for {args.eval_episodes} episodes")
            evaluate_agent(agent, env, n_eval_episodes=args.eval_episodes)
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    else:
        try:
            # Load the saved model
            model_path = args.model or "src/rl/models/single_agent/final_model"
            logger.info(f"Loading model from {model_path}")
            
            agent = PPO.load(model_path, env=env)
            
            # Run simulation
            logger.info("Running simulation with trained agent")
            run_simulation(agent, env, render=args.render)
        except Exception as e:
            logger.error(f"Error during simulation: {str(e)}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or run RL agent with real predictions")
    parser.add_argument("--train", action="store_true", help="Train a new agent")
    parser.add_argument("--model", type=str, help="Path to pre-trained model (for evaluation)")
    parser.add_argument("--config", type=str, default="src/rl/rl_config.json", help="Configuration file path")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total timesteps for training")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of episodes for evaluation")
    parser.add_argument("--render", action="store_true", help="Render the environment during evaluation")
    parser.add_argument("--use-predictions", action="store_true", default=True, help="Use real price and solar predictions")
    
    args = parser.parse_args()
    main(args) 