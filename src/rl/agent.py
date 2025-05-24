"""
Enhanced RL agents for home energy management.

Contains:
- RecurrentEnergyAgent: LSTM-based agent that maintains memory across timesteps
- Utility functions for smart action selection and training improvements
- Analysis tools for reward components and training issues
"""
import os
import numpy as np
import pandas as pd
import gymnasium as gym
from typing import Dict, List, Tuple, Optional, Any, Union, Type, Callable
import datetime
from pathlib import Path
import torch
import logging

# Import Stable Baselines 3
from stable_baselines3 import PPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv

# Import our custom environment and wrapper
from src.rl.custom_env import HomeEnergyEnv

logger = logging.getLogger("agent")


def smart_action_selection(
    current_soc: float,
    price_forecast: List[float], 
    solar_forecast: List[float],
    current_hour: int,
    battery_capacity: float,
    config: Dict
) -> float:
    """
    Provides intelligent action suggestions based on current state and forecasts.
    This can be used as a baseline or for action masking improvements.
    
    Args:
        current_soc: Current battery state of charge (0-1)
        price_forecast: Price forecast for next 24 hours
        solar_forecast: Solar forecast for next 24 hours  
        current_hour: Current hour of day (0-23)
        battery_capacity: Battery capacity in kWh
        config: Configuration dictionary
        
    Returns:
        float: Suggested action (-1 to 1, where -1 is max charge, 1 is max discharge)
    """
    # Calculate price percentiles
    if len(price_forecast) >= 24:
        current_price = price_forecast[0]
        price_25th = np.percentile(price_forecast[:24], 25)
        price_75th = np.percentile(price_forecast[:24], 75)
    else:
        return 0.0  # No action if insufficient data
    
    # Calculate expected solar for next 12 hours
    next_12h_solar = sum(solar_forecast[:48]) if len(solar_forecast) >= 48 else 0  # 48 * 15min = 12h
    
    # Emergency SoC management
    if current_soc < 0.15:  # Very low SoC
        return -0.8  # Strong charge
    elif current_soc > 0.85:  # Very high SoC  
        return 0.6   # Moderate discharge
    
    # Night hours (22:00-06:00) - charge at low prices
    if current_hour >= 22 or current_hour <= 6:
        if current_price <= price_25th and current_soc < 0.7:
            return -0.5  # Charge at low prices
        elif next_12h_solar > 3.0 and current_soc > 0.6:
            return 0.3   # Make room for solar if significant production expected
    
    # Morning hours (06:00-10:00) - prepare for solar
    elif 6 <= current_hour <= 10:
        if next_12h_solar > 2.0 and current_soc > 0.6:
            return 0.4   # Discharge to make room for solar
    
    # Peak hours (16:00-20:00) - discharge at high prices
    elif 16 <= current_hour <= 20:
        if current_price >= price_75th and current_soc > 0.3:
            return 0.6   # Discharge at high prices
    
    # Default: small actions based on price
    if current_price <= price_25th and current_soc < 0.6:
        return -0.2  # Light charging at low prices
    elif current_price >= price_75th and current_soc > 0.4:
        return 0.2   # Light discharging at high prices
    
    return 0.0  # No action


def analyze_reward_components(reward_history: List[Dict]) -> Dict:
    """
    Analyzes reward component history to identify balance issues.
    
    Args:
        reward_history: List of reward component dictionaries from training
        
    Returns:
        Dict: Analysis results with statistics and recommendations
    """
    if not reward_history:
        return {"error": "No reward history provided"}
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(reward_history)
    
    # Calculate statistics for each component
    stats = {}
    for col in df.columns:
        if col.startswith('reward_'):
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'range': df[col].max() - df[col].min(),
                'contribution_pct': abs(df[col].mean()) / sum(abs(df[col2].mean()) for col2 in df.columns if col2.startswith('reward_')) * 100
            }
    
    # Identify problematic components
    recommendations = []
    
    for component, stat in stats.items():
        # Check for extreme ranges
        if stat['range'] > 100:
            recommendations.append(f"{component}: Very large range ({stat['range']:.1f}), consider scaling down")
        
        # Check for dominance
        if stat['contribution_pct'] > 40:
            recommendations.append(f"{component}: Dominates reward ({stat['contribution_pct']:.1f}%), reduce weight")
        
        # Check for high variance
        if stat['std'] > abs(stat['mean']) * 3:
            recommendations.append(f"{component}: High variance, consider smoothing")
    
    return {
        'statistics': stats,
        'recommendations': recommendations,
        'balance_score': 1.0 - max(stat['contribution_pct'] for stat in stats.values()) / 100.0
    }


def create_curriculum_schedule(total_timesteps: int) -> List[Dict]:
    """
    Creates a curriculum learning schedule for progressive training.
    
    Args:
        total_timesteps: Total training timesteps
        
    Returns:
        List[Dict]: Curriculum schedule with different phases
    """
    schedule = []
    
    # Phase 1: Basic battery management (20% of training)
    phase1_steps = int(total_timesteps * 0.2)
    schedule.append({
        'name': 'basic_battery',
        'start_step': 0,
        'end_step': phase1_steps,
        'config_overrides': {
            'simulation_days': 3,  # Shorter episodes
            'w_soc': 3.0,         # Higher SoC weight
            'w_arbitrage': 0.5,   # Lower arbitrage weight
            'use_solar_predictions': False,  # No solar complexity
            'enable_night_peak_chain': False  # No advanced features
        }
    })
    
    # Phase 2: Price arbitrage (30% of training)
    phase2_steps = int(total_timesteps * 0.3)
    schedule.append({
        'name': 'price_arbitrage',
        'start_step': phase1_steps,
        'end_step': phase1_steps + phase2_steps,
        'config_overrides': {
            'simulation_days': 7,  # Medium episodes
            'w_soc': 2.0,         # Reduced SoC weight
            'w_arbitrage': 2.0,   # Increased arbitrage weight
            'use_solar_predictions': False,  # Still no solar
            'enable_night_peak_chain': True   # Enable chain bonuses
        }
    })
    
    # Phase 3: Full complexity (50% of training)
    schedule.append({
        'name': 'full_complexity',
        'start_step': phase1_steps + phase2_steps,
        'end_step': total_timesteps,
        'config_overrides': {
            'simulation_days': 30,  # Full episodes
            'use_solar_predictions': True,  # Full solar complexity
            # Use default weights from config
        }
    })
    
    return schedule


def adaptive_exploration_schedule(current_step: int, total_steps: int, base_entropy: float = 0.01) -> float:
    """
    Creates an adaptive exploration schedule that reduces entropy over time.
    
    Args:
        current_step: Current training step
        total_steps: Total training steps
        base_entropy: Base entropy coefficient
        
    Returns:
        float: Adjusted entropy coefficient
    """
    progress = current_step / total_steps
    
    # Start with higher exploration, reduce exponentially
    if progress < 0.2:  # First 20% - high exploration
        return base_entropy * 4.0
    elif progress < 0.5:  # Next 30% - medium exploration  
        return base_entropy * 2.0
    elif progress < 0.8:  # Next 30% - reduced exploration
        return base_entropy * 1.5
    else:  # Final 20% - minimal exploration
        return base_entropy * 0.5


def detect_training_issues(metrics_history: List[Dict]) -> List[str]:
    """
    Detects common training issues from metrics history.
    
    Args:
        metrics_history: List of training metrics dictionaries
        
    Returns:
        List[str]: List of detected issues and suggestions
    """
    if len(metrics_history) < 10:
        return ["Insufficient training history for analysis"]
    
    issues = []
    recent_metrics = metrics_history[-100:]  # Last 100 updates
    
    # Convert to DataFrame
    df = pd.DataFrame(recent_metrics)
    
    # Check for common issues
    if 'episode_reward' in df.columns:
        reward_trend = np.polyfit(range(len(df)), df['episode_reward'], 1)[0]
        if reward_trend < -0.01:
            issues.append("Episode rewards are declining - check reward scaling or learning rate")
        elif abs(reward_trend) < 0.001:
            issues.append("Episode rewards are stagnating - consider curriculum learning")
    
    if 'value_loss' in df.columns:
        if df['value_loss'].mean() > 10.0:
            issues.append("Value loss is very high - reduce learning rate or check reward scaling")
        elif df['value_loss'].std() > df['value_loss'].mean():
            issues.append("Value loss is very unstable - check batch size or network architecture")
    
    if 'policy_loss' in df.columns:
        if df['policy_loss'].std() > abs(df['policy_loss'].mean()) * 2:
            issues.append("Policy loss is unstable - consider gradient clipping or lower learning rate")
    
    if 'entropy_loss' in df.columns:
        if df['entropy_loss'].mean() < 0.001:
            issues.append("Entropy loss is very low - agent may be getting too deterministic too quickly")
    
    return issues if issues else ["No major training issues detected"]


def create_action_mask(
    current_soc: float,
    min_soc: float,
    max_soc: float,
    battery_capacity: float,
    max_charge_power: float,
    max_discharge_power: float,
    time_step_hours: float
) -> Tuple[float, float]:
    """
    Creates intelligent action bounds based on current SoC and constraints.
    
    Args:
        current_soc: Current battery state of charge (0-1)
        min_soc: Minimum allowed SoC
        max_soc: Maximum allowed SoC
        battery_capacity: Battery capacity in kWh
        max_charge_power: Maximum charging power in kW
        max_discharge_power: Maximum discharging power in kW
        time_step_hours: Time step duration in hours
        
    Returns:
        Tuple[float, float]: (min_action, max_action) bounds for safe actions
    """
    # Calculate energy headroom
    energy_to_min = (current_soc - min_soc) * battery_capacity
    energy_to_max = (max_soc - current_soc) * battery_capacity
    
    # Convert to power limits
    max_safe_discharge = min(max_discharge_power, energy_to_min / time_step_hours)
    max_safe_charge = min(max_charge_power, energy_to_max / time_step_hours)
    
    # Convert to normalized action space (-1 to 1)
    min_action = -max_safe_charge / max_charge_power  # Most negative (charging)
    max_action = max_safe_discharge / max_discharge_power  # Most positive (discharging)
    
    return max(-1.0, min_action), min(1.0, max_action)


class RecurrentEnergyAgent:
    """
    Enhanced recurrent agent using PPO with LSTM that maintains memory across timesteps.
    
    This allows the agent to learn patterns over time, ideal for:
    - Remembering night-time charging to use during day peaks
    - Maintaining awareness of solar patterns and price trends
    - Learning smooth control strategies that span multiple timesteps
    
    Enhanced with improvements including:
    - Better hyperparameter defaults
    - Improved network architecture
    - Smart action selection utilities
    - Enhanced logging and monitoring
    """
    
    def __init__(
        self, 
        env: gym.Env,
        model_path: Optional[str] = None,
        config: Dict = None
    ):
        """
        Initialize an enhanced RecurrentEnergyAgent.
        
        Args:
            env: The gym environment
            model_path: Optional path to load a pre-trained model
            config: Configuration dictionary
        """
        # Store config
        self.config = config or {}
        
        # Extract parameters with improved defaults
        self.learning_rate = self.config.get("learning_rate", 2e-4)  # Lower LR for stability
        self.gamma = self.config.get("gamma", 0.995)               # Higher discount
        self.n_steps = self.config.get("n_steps", 4096)            # Larger rollout
        self.batch_size = self.config.get("batch_size", 128)       # Larger batch
        self.n_epochs = self.config.get("n_epochs", 8)             # More epochs
        self.ent_coeff = self.config.get("ent_coef", 0.005)        # Balanced entropy
        self.gae_lambda = self.config.get("gae_lambda", 0.98)      # Higher GAE lambda
        
        # RNN-specific parameters
        self.n_lstm_layers = self.config.get("n_lstm_layers", 1)
        self.lstm_hidden_size = self.config.get("lstm_hidden_size", 128)  # Larger hidden size
        
        logger.info(f"Initializing RecurrentEnergyAgent with enhanced configuration:")
        logger.info(f"  - Learning rate: {self.learning_rate}")
        logger.info(f"  - LSTM hidden size: {self.lstm_hidden_size}")
        logger.info(f"  - Rollout steps: {self.n_steps}")
        logger.info(f"  - Batch size: {self.batch_size}")
        
        # Wrap the environment to handle dictionary observations for RecurrentPPO
        from gymnasium.wrappers import FlattenObservation
        
        # Check if the observation space is a Dict
        if isinstance(env.observation_space, gym.spaces.Dict):
            logger.info("Flattening dictionary observation space for RecurrentPPO")
            self.env = FlattenObservation(env)
        else:
            self.env = env
            
        # Initialize the model with enhanced recurrent policy
        policy_kwargs = dict(
            lstm_hidden_size=self.lstm_hidden_size,
            net_arch=dict(pi=[128, 128], vf=[128, 128]),  # Larger networks
            activation_fn=torch.nn.ReLU,
        )
        
        if model_path:
            logger.info(f"Loading pre-trained model from {model_path}")
            self.model = RecurrentPPO.load(
                model_path,
                env=self.env,
                verbose=1
            )
        else:
            logger.info("Creating new RecurrentPPO model with enhanced configuration")
            self.model = RecurrentPPO(
                "MlpLstmPolicy",
                self.env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                verbose=1,
                ent_coef=self.ent_coeff,
                gae_lambda=self.gae_lambda,
                clip_range=0.2,      # Standard PPO clipping
                max_grad_norm=0.5,   # Gradient clipping for stability
                policy_kwargs=policy_kwargs,
                device='auto',
                tensorboard_log='src/rl/logs/tensorboard/'
            )
    
    def get_smart_action_suggestion(
        self,
        current_soc: float,
        price_forecast: List[float],
        solar_forecast: List[float],
        current_hour: int
    ) -> float:
        """
        Get intelligent action suggestion based on current state.
        
        Args:
            current_soc: Current battery state of charge (0-1)
            price_forecast: Price forecast for next 24 hours
            solar_forecast: Solar forecast for next 24 hours
            current_hour: Current hour of day (0-23)
            
        Returns:
            float: Suggested action (-1 to 1)
        """
        battery_capacity = self.config.get("battery_capacity", 22.0)
        return smart_action_selection(
            current_soc=current_soc,
            price_forecast=price_forecast,
            solar_forecast=solar_forecast,
            current_hour=current_hour,
            battery_capacity=battery_capacity,
            config=self.config
        )
    
    def train(
        self, 
        total_timesteps: int,
        callback: Optional[BaseCallback] = None
    ) -> None:
        """
        Train the model with enhanced logging.
        
        Args:
            total_timesteps: Total timesteps to train for
            callback: Optional callback
        """
        # Log to file only, don't interrupt progress bar
        logger.info(f"Starting training for {total_timesteps:,} timesteps")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
            tb_log_name=f"enhanced_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        logger.info("Training completed!")
    
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
        logger.info(f"Saved enhanced recurrent model to {path}")


class RuleBasedAgent:
    """
    Enhanced rule-based agent that operates on 1-hour timesteps.
    
    Handles immediate control decisions based on current state and forecasts.
    
    The general idea is to charge when the price is below 20 öre/kWh and 
    discharge when the price is above 100 öre/kWh, with enhanced logic for
    solar production and SoC management.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the enhanced rule-based agent.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.battery_capacity = self.config.get("battery_capacity", 22.0)
        self.min_soc = self.config.get("min_soc", 0.1)
        self.max_soc = self.config.get("max_soc", 0.9)
        
        # Enhanced thresholds
        self.low_price_threshold = self.config.get("rule_low_price_threshold", 20.0)  # öre/kWh
        self.high_price_threshold = self.config.get("rule_high_price_threshold", 100.0)  # öre/kWh
        self.emergency_soc_threshold = self.config.get("rule_emergency_soc", 0.15)
        
        logger.info(f"🎯 Initialized enhanced rule-based agent")
        logger.info(f"  - Low price threshold: {self.low_price_threshold} öre/kWh")
        logger.info(f"  - High price threshold: {self.high_price_threshold} öre/kWh")
        logger.info(f"  - Emergency SoC threshold: {self.emergency_soc_threshold}")
    
    def predict(self, observation: Dict, **kwargs) -> Tuple[float, None]:
        """
        Predict action based on enhanced rule-based logic.
        
        Args:
            observation: Dictionary containing state information
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Tuple of (action, None) where action is between -1 and 1
        """
        current_soc = observation.get('battery_soc', 0.5)
        current_price = observation.get('current_price', 50.0)
        current_hour = observation.get('hour_of_day', 12)
        
        # Get forecasts if available
        price_forecast = observation.get('price_forecast', [current_price] * 24)
        solar_forecast = observation.get('solar_forecast', [0.0] * 48)
        
        # Use smart action selection
        action = smart_action_selection(
            current_soc=current_soc,
            price_forecast=price_forecast,
            solar_forecast=solar_forecast,
            current_hour=current_hour,
            battery_capacity=self.battery_capacity,
            config=self.config
        )
        
        return action, None
