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
import torch

# Import Stable Baselines 3
from stable_baselines3 import PPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.ppo_recurrent import RecurrentPPO
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
        self.ent_coeff = self.config.get("short_term_ent_coeff", 0.01)
        self.gae_lambda = self.config.get("short_term_gae_lambda", 0.98)
        # Initialize the model
        self.model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            verbose=0,
            ent_coef=self.ent_coeff,
            gae_lambda=self.gae_lambda,
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


class RecurrentEnergyAgent:
    """
    Recurrent agent using PPO with LSTM that maintains memory across timesteps.
    
    This allows the agent to learn patterns over time, ideal for:
    - Remembering night-time charging to use during day peaks
    - Maintaining awareness of solar patterns and price trends
    - Learning smooth control strategies that span multiple timesteps
    """
    
    def __init__(
        self, 
        env: gym.Env,
        model_path: Optional[str] = None,
        config: Dict = None
    ):
        """
        Initialize a RecurrentEnergyAgent.
        
        Args:
            env: The gym environment
            model_path: Optional path to load a pre-trained model
            config: Configuration dictionary
        """
        # Store config
        self.config = config or {}
        
        # Extract parameters
        self.learning_rate = self.config.get("learning_rate", 3e-4)
        self.gamma = self.config.get("gamma", 0.99)
        self.n_steps = self.config.get("n_steps", 2048)
        self.batch_size = self.config.get("batch_size", 64)
        self.n_epochs = self.config.get("n_epochs", 10)
        self.ent_coeff = self.config.get("ent_coef", 0.0)
        self.gae_lambda = self.config.get("gae_lambda", 0.95)
        
        # RNN-specific parameters
        self.n_lstm_layers = self.config.get("n_lstm_layers", 1)
        self.lstm_hidden_size = self.config.get("lstm_hidden_size", 64)
        
        # Wrap the environment to handle dictionary observations for RecurrentPPO
        from gymnasium.wrappers import FlattenObservation
        
        # Check if the observation space is a Dict
        if isinstance(env.observation_space, gym.spaces.Dict):
            print("Flattening dictionary observation space for RecurrentPPO")
            self.env = FlattenObservation(env)
        else:
            self.env = env
            
        # Initialize the model with recurrent policy
        policy_kwargs = dict(
            lstm_hidden_size=self.lstm_hidden_size,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        )
        
        if model_path:
            print(f"Loading pre-trained model from {model_path}")
            self.model = RecurrentPPO.load(
                model_path,
                env=self.env,
                verbose=1
            )
        else:
            self.model = RecurrentPPO(
                "MlpLstmPolicy",
                self.env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                verbose=0,
                ent_coef=self.ent_coeff,
                gae_lambda=self.gae_lambda,
                policy_kwargs=policy_kwargs
            )
    
    def train(
        self, 
        total_timesteps: int,
        callback: Optional[BaseCallback] = None
    ) -> None:
        """
        Train the model.
        
        Args:
            total_timesteps: Total timesteps to train for
            callback: Optional callback
        """
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
    
    def predict(
        self, 
        observation: np.ndarray,
        state: Optional[Tuple] = None,
        episode_start: Optional[bool] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[Tuple]]:
        """
        Predict action and next state given current observation.
        
        Args:
            observation: Current observation
            state: Current RNN state
            episode_start: Whether this is the start of an episode
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (action, next_state)
        """
        return self.model.predict(
            observation, 
            state=state,
            episode_start=episode_start,
            deterministic=deterministic
        )
    
    def save(self, path: str) -> None:
        """
        Save the model.
        
        Args:
            path: Path to save the model
        """
        self.model.save(path)
        print(f"Saved recurrent model to {path}")


class RuleBasedAgent:
    """
    Rule-based agent that operates on 1-hour timesteps.
    
    Handles immediate control decisions based on current state and forecasts.
    
    The general ideas is to charge when the price is below 20 öre/kWh and 
    discharge when the price is above 100 öre/kWh.
    """
