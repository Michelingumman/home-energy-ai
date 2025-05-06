"""
RL agents for hierarchical home energy management.

Contains:
- ShortTermAgent: Reacts to immediate conditions and operates hourly
- LongTermAgent: Plans ahead for the week and operates on 4-hour intervals
- HierarchicalController: Orchestrates both agents during training and inference
"""
import os
import numpy as np
import gymnasium as gym
from typing import Dict, List, Tuple, Optional, Any, Union, Type, Callable
import datetime
from pathlib import Path

# Import Stable Baselines 3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv

# Import our custom environment and wrapper
from src.rl.custom_env import HomeEnergyEnv
from src.rl.wrappers import LongTermEnv


class ShortTermAgent:
    """
    Short-term agent that operates on 1-hour timesteps.
    
    Handles immediate control decisions based on current state and forecasts.
    """
    
    def __init__(
        self, 
        env: gym.Env,
        model_path: Optional[str] = None,
        config: Dict = None
    ):
        """
        Initialize the short-term agent.
        
        Args:
            env: Environment to operate in
            model_path: Path to a saved model to load
            config: Configuration dictionary with hyperparameters
        """
        self.env = env
        self.config = config or {}
        
        # Default hyperparameters
        self.learning_rate = self.config.get("short_term_learning_rate", 3e-4)
        self.gamma = self.config.get("short_term_gamma", 0.99)
        self.n_steps = self.config.get("short_term_n_steps", 2048)
        self.batch_size = self.config.get("short_term_batch_size", 64)
        self.n_epochs = self.config.get("short_term_n_epochs", 10)
        
        # Initialize the model
        self.model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            verbose=1
        )
        
        # Load a pre-trained model if provided
        if model_path and os.path.exists(model_path):
            self.model = PPO.load(model_path, env=env)
            print(f"Loaded short-term model from {model_path}")
    
    def train(
        self, 
        total_timesteps: int, 
        callback: Optional[BaseCallback] = None
    ) -> None:
        """
        Train the agent.
        
        Args:
            total_timesteps: Total number of timesteps to train
            callback: Optional callback for monitoring training
        """
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
    
    def predict(
        self, 
        observation: Dict, 
        state: Optional[Any] = None, 
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Predict action based on observation.
        
        Args:
            observation: Environment observation
            state: RNN hidden state (if applicable)
            deterministic: Whether to use deterministic policy
            
        Returns:
            tuple: (action, state)
        """
        return self.model.predict(observation, state, deterministic)
    
    def save(self, path: str) -> None:
        """
        Save the model.
        
        Args:
            path: Path to save the model
        """
        self.model.save(path)
        print(f"Saved short-term model to {path}")


class LongTermAgent:
    """
    Long-term agent that plans ahead for multiple days.
    
    Operates on 4-hour timesteps and produces SoC corridors and 
    appliance activation windows.
    """
    
    def __init__(
        self, 
        env: gym.Env,
        model_path: Optional[str] = None,
        config: Dict = None
    ):
        """
        Initialize the long-term agent.
        
        Args:
            env: Environment to operate in (should be wrapped in LongTermEnv)
            model_path: Path to a saved model to load
            config: Configuration dictionary with hyperparameters
        """
        self.env = env
        self.config = config or {}
        
        # Default hyperparameters
        self.learning_rate = self.config.get("long_term_learning_rate", 1e-4)
        self.gamma = self.config.get("long_term_gamma", 0.997)  # Higher gamma for long-term planning
        self.n_steps = self.config.get("long_term_n_steps", 1024)
        self.batch_size = self.config.get("long_term_batch_size", 64)
        self.n_epochs = self.config.get("long_term_n_epochs", 10)
        
        # Initialize the model
        self.model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            verbose=1
        )
        
        # Load a pre-trained model if provided
        if model_path and os.path.exists(model_path):
            self.model = PPO.load(model_path, env=env)
            print(f"Loaded long-term model from {model_path}")
    
    def train(
        self, 
        total_timesteps: int, 
        callback: Optional[BaseCallback] = None
    ) -> None:
        """
        Train the agent.
        
        Args:
            total_timesteps: Total number of timesteps to train
            callback: Optional callback for monitoring training
        """
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
    
    def predict(
        self, 
        observation: Dict, 
        state: Optional[Any] = None, 
        deterministic: bool = True
    ) -> Tuple[Dict, Optional[Any]]:
        """
        Predict action based on observation.
        
        Args:
            observation: Environment observation
            state: RNN hidden state (if applicable)
            deterministic: Whether to use deterministic policy
            
        Returns:
            tuple: (action, state)
        """
        return self.model.predict(observation, state, deterministic)
    
    def save(self, path: str) -> None:
        """
        Save the model.
        
        Args:
            path: Path to save the model
        """
        self.model.save(path)
        print(f"Saved long-term model to {path}")


class HierarchicalController:
    """
    Hierarchical controller that coordinates short-term and long-term agents.
    
    Handles:
    - Training both agents in a staged approach
    - Inference for the complete hierarchical system
    - Communication between the two levels of control
    """
    
    def __init__(
        self,
        config: Dict,
        base_env: Optional[gym.Env] = None,
        short_term_model_path: Optional[str] = None,
        long_term_model_path: Optional[str] = None
    ):
        """
        Initialize the hierarchical controller.
        
        Args:
            config: Configuration dictionary with all hyperparameters
            base_env: Base environment (HomeEnergyEnv) if already created
            short_term_model_path: Path to a saved short-term model
            long_term_model_path: Path to a saved long-term model
        """
        self.config = config
        
        # Create environments if not provided
        if base_env is None:
            # Create base environment
            self.base_env = HomeEnergyEnv(
                battery_capacity=config.get("battery_capacity", 22.0),
                simulation_days=config.get("simulation_days", 7),
                peak_penalty_factor=config.get("peak_penalty_factor", 10.0),
                comfort_bonus_factor=config.get("comfort_bonus_factor", 2.0),
                random_weather=config.get("random_weather", True)
            )
        else:
            self.base_env = base_env
        
        # Create long-term environment by wrapping base environment
        self.long_term_env = LongTermEnv(
            env=self.base_env,
            steps_per_action=config.get("steps_per_action", 4),
            corridor_width=config.get("corridor_width", 0.1),
            planning_horizon=config.get("planning_horizon", 42)
        )
        
        # Create agents
        self.short_term_agent = ShortTermAgent(
            env=self.base_env,
            model_path=short_term_model_path,
            config=config
        )
        
        self.long_term_agent = LongTermAgent(
            env=self.long_term_env,
            model_path=long_term_model_path,
            config=config
        )
        
        # Training parameters
        self.staged_training = config.get("staged_training", True)
        self.short_term_timesteps = config.get("short_term_timesteps", 500000)
        self.long_term_timesteps = config.get("long_term_timesteps", 200000)
        self.joint_timesteps = config.get("joint_timesteps", 100000)
        
        # Directory for saving models and logs - FIX PATH REFERENCES
        self.model_dir = Path(config.get("model_dir", "src/rl/saved_models"))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config.get("log_dir", "src/rl/logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self, callbacks: Dict[str, BaseCallback] = None) -> None:
        """
        Train the hierarchical agent system.
        
        Args:
            callbacks: Optional callbacks for monitoring training
                (keys: 'short_term', 'long_term', 'joint')
        """
        callbacks = callbacks or {}
        
        if self.staged_training:
            # Phase 1: Train short-term agent
            print("\n===== Training Short-Term Agent =====")
            self.short_term_agent.train(
                total_timesteps=self.short_term_timesteps,
                callback=callbacks.get('short_term')
            )
            self.short_term_agent.save(str(self.model_dir / "short_term_agent"))
            
            # Phase 2: Train long-term agent with fixed short-term agent
            print("\n===== Training Long-Term Agent =====")
            self._prepare_long_term_training()
            self.long_term_agent.train(
                total_timesteps=self.long_term_timesteps,
                callback=callbacks.get('long_term')
            )
            self.long_term_agent.save(str(self.model_dir / "long_term_agent"))
            
            # Optional Phase 3: Joint fine-tuning with slower learning rate for short-term
            if self.joint_timesteps > 0:
                print("\n===== Joint Fine-Tuning =====")
                # Reduce learning rate for short-term agent
                original_lr = self.short_term_agent.model.learning_rate
                self.short_term_agent.model.learning_rate = original_lr * 0.1
                
                # TODO: Implement joint training
                print("Joint training not fully implemented yet")
                
                # Restore original learning rate
                self.short_term_agent.model.learning_rate = original_lr
        else:
            # Simultaneous training (more complex, not fully implemented)
            print("Simultaneous training not fully implemented yet")
            # TODO: Implement simultaneous training
    
    def _prepare_long_term_training(self) -> None:
        """Prepare for long-term agent training by configuring the environment."""
        # Here we would typically set up the environment to use the trained short-term
        # agent for the lower-level control during long-term training
        
        # This is a simplified implementation
        print("Preparing long-term environment with trained short-term agent")
        # The LongTermEnv wrapper should ideally be configured to use the short-term agent
        # for hourly control based on the long-term plan
        
        # TODO: Implement proper integration between long-term and short-term agents
        pass
    
    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        """
        Evaluate the hierarchical agent system.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            dict: Evaluation metrics
        """
        total_rewards = []
        total_costs = []
        peak_powers = []
        corridor_violations = []
        
        for episode in range(num_episodes):
            # Reset environments
            short_term_obs, _ = self.base_env.reset()
            long_term_obs, _ = self.long_term_env.reset()
            
            episode_reward = 0
            done = False
            
            while not done:
                # Get long-term plan
                long_term_action, _ = self.long_term_agent.predict(long_term_obs)
                
                # Execute long-term step (which handles 4 hours internally)
                long_term_obs, long_term_reward, done, _, long_term_info = self.long_term_env.step(long_term_action)
                
                episode_reward += long_term_reward
            
            # Record metrics
            total_rewards.append(episode_reward)
            total_costs.append(self.base_env.total_cost)
            peak_powers.append(self.base_env.peak_power)
            corridor_violations.append(long_term_info.get("corridor_violations", 0))
            
            print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Cost = {self.base_env.total_cost:.2f}")
            
        # Calculate summary statistics
        results = {
            "mean_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "mean_cost": np.mean(total_costs),
            "mean_peak_power": np.mean(peak_powers),
            "mean_corridor_violations": np.mean(corridor_violations)
        }
        
        return results
    
    def save(self) -> None:
        """Save both agents."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.short_term_agent.save(str(self.model_dir / f"short_term_agent_{timestamp}"))
        self.long_term_agent.save(str(self.model_dir / f"long_term_agent_{timestamp}"))
        
        # Also save the latest versions
        self.short_term_agent.save(str(self.model_dir / "short_term_agent_latest"))
        self.long_term_agent.save(str(self.model_dir / "long_term_agent_latest"))
    
    def load(self, short_term_path: str, long_term_path: str) -> None:
        """
        Load saved models for both agents.
        
        Args:
            short_term_path: Path to saved short-term model
            long_term_path: Path to saved long-term model
        """
        self.short_term_agent = ShortTermAgent(
            env=self.base_env,
            model_path=short_term_path,
            config=self.config
        )
        
        self.long_term_agent = LongTermAgent(
            env=self.long_term_env,
            model_path=long_term_path,
            config=self.config
        ) 