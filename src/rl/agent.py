"""
RL agents for hierarchical home energy management.

Contains:
- ShortTermAgent: Reacts to immediate conditions and operates hourly
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
            verbose=0
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

