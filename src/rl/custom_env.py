"""
Custom RL environment for home energy system control.

This environment simulates a home energy system with:
- Battery storage
- Appliance management
- Solar production
- Grid connection
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, List, Optional, Any, Union
import datetime
import pandas as pd
import os
import logging
from pathlib import Path
import time # Added for profiling
import sys # Added for sys.exit

from src.rl.components import Battery
from src.rl.safety_buffer import ensure_soc_limits  # Import the safety buffer function

# Get a logger without re-configuring basicConfig
logger = logging.getLogger("home_energy_env")

# Default paths for prediction files
PRICE_PREDICTIONS_PATH = "src/predictions/prices/plots/predictions/merged"

# Utility functions for the new reward structure

def compute_max_charge_rate(soc: float, max_soc: float, max_charge_power_kw: float, 
                           battery_capacity_kwh: float, time_step_hours: float) -> float:
    """
    Computes maximum allowed charging rate (in normalized -1 to 0 space) based on current SoC.
    Returns a value between 0 (no charging allowed) and max_charge_power_kw.
    
    Args:
        soc: Current battery state of charge (0 to 1)
        max_soc: Maximum allowed SoC (e.g., 0.8)
        max_charge_power_kw: Maximum battery charging power
        battery_capacity_kwh: Battery capacity in kWh
        time_step_hours: Time step duration in hours
        
    Returns:
        float: Maximum normalized charging rate (0 to max_charge_power_kw)
    """
    # If already at or above max_soc, no charging allowed
    if soc >= max_soc:
        return 0.0
    
    # Calculate energy headroom and convert to power
    energy_headroom_kwh = (max_soc - soc) * battery_capacity_kwh
    max_safe_charge_power = min(max_charge_power_kw, energy_headroom_kwh / time_step_hours)
    
    return max_safe_charge_power


def compute_max_discharge_rate(soc: float, min_soc: float, max_discharge_power_kw: float,
                              battery_capacity_kwh: float, time_step_hours: float) -> float:
    """
    Computes maximum allowed discharging rate (in normalized 0 to 1 space) based on current SoC.
    Returns a value between 0 (no discharging allowed) and max_discharge_power_kw.
    
    Args:
        soc: Current battery state of charge (0 to 1)
        min_soc: Minimum allowed SoC (e.g., 0.2)
        max_discharge_power_kw: Maximum battery discharging power
        battery_capacity_kwh: Battery capacity in kWh
        time_step_hours: Time step duration in hours
        
    Returns:
        float: Maximum normalized discharging rate (0 to max_discharge_power_kw)
    """
    # If already at or below min_soc, no discharging allowed
    if soc <= min_soc:
        return 0.0
    
    # Calculate energy available and convert to power
    energy_available_kwh = (soc - min_soc) * battery_capacity_kwh
    max_safe_discharge_power = min(max_discharge_power_kw, energy_available_kwh / time_step_hours)
    
    return max_safe_discharge_power


def safe_action_mask(raw_action: float, soc: float, min_soc: float, max_soc: float, 
                    max_charge_power_kw: float, max_discharge_power_kw: float,
                    battery_capacity_kwh: float, time_step_hours: float) -> float:
    """
    Projects raw_action into the feasible action set such that
    resulting SoC remains within [min_soc, max_soc].
    
    Args:
        raw_action: Normalized action from agent (-1 to 1)
        soc: Current battery state of charge (0 to 1)
        min_soc: Minimum allowed SoC (e.g., 0.2)
        max_soc: Maximum allowed SoC (e.g., 0.8)
        max_charge_power_kw: Maximum battery charging power
        max_discharge_power_kw: Maximum battery discharging power
        battery_capacity_kwh: Battery capacity in kWh
        time_step_hours: Time step duration in hours
        
    Returns:
        float: Safe action that won't violate SoC constraints
    """
    # Compute the max allowable charge/discharge for current SoC
    max_charge = compute_max_charge_rate(soc, max_soc, max_charge_power_kw, 
                                        battery_capacity_kwh, time_step_hours)
    max_discharge = compute_max_discharge_rate(soc, min_soc, max_discharge_power_kw,
                                             battery_capacity_kwh, time_step_hours)
    
    # Convert from raw normalized (-1 to 1) to actual power values
    requested_power = raw_action
    if requested_power < 0:  # Charging request
        # Scale the -1 to 0 range to -max_charge_power_kw to 0
        requested_charge_power = -requested_power * max_charge_power_kw
        # Limit to safe value
        safe_charge_power = min(requested_charge_power, max_charge)
        # Convert back to normalized space
        safe_action = -safe_charge_power / max_charge_power_kw if max_charge_power_kw > 0 else 0
    else:  # Discharging request
        # Scale the 0 to 1 range to 0 to max_discharge_power_kw  
        requested_discharge_power = requested_power * max_discharge_power_kw
        # Limit to safe value
        safe_discharge_power = min(requested_discharge_power, max_discharge)
        # Convert back to normalized space
        safe_action = safe_discharge_power / max_discharge_power_kw if max_discharge_power_kw > 0 else 0
        
    return safe_action


def soc_potential(soc: float, min_soc: float, low_pref: float, high_pref: float, max_soc: float) -> float:
    """
    Creates a potential function that guides the agent toward preferred SoC ranges.
    Uses finite values with smoother gradients to avoid extreme reward spikes.
    
    Args:
        soc: Current battery state of charge (0 to 1)
        min_soc: Minimum allowed SoC (hard constraint, e.g., 0.2)
        low_pref: Lower bound of preferred SoC range (soft constraint, e.g., 0.3)
        high_pref: Upper bound of preferred SoC range (soft constraint, e.g., 0.7)
        max_soc: Maximum allowed SoC (hard constraint, e.g., 0.8)
        
    Returns:
        float: Potential value for shaping reward
    """
    # Use configurable finite values instead of -1e6
    from src.rl.config import get_config_dict
    config = get_config_dict()
    potential_min_value = config.get("soc_potential_min_value", -50.0)
    
    if soc <= min_soc or soc >= max_soc:
        return potential_min_value  # Finite negative value
    
    # Calculate center and width of preferred range
    mid = 0.5 * (low_pref + high_pref)
    width = max(0.001, 0.5 * (high_pref - low_pref))  # Prevent division by zero
    
    # Smooth transition outside preferred range
    if soc < low_pref:
        # Normalize to [0,1] between min_soc and low_pref
        normalized_dist = (soc - min_soc) / max(0.001, (low_pref - min_soc))
        return normalized_dist - 0.5  # Range from -0.5 to 0.5
    elif soc > high_pref:
        # Normalize to [0,1] between high_pref and max_soc
        normalized_dist = (max_soc - soc) / max(0.001, (max_soc - high_pref))
        return normalized_dist - 0.5  # Range from -0.5 to 0.5
    else:
        # Inside preferred band - normalized distance from center
        normalized_dist = 1.0 - (abs(soc - mid) / width)
        return normalized_dist  # Range from 0 to 1


def shaping_reward(soc_t: float, soc_tp1: float, gamma: float, 
                  min_soc: float, low_pref: float, high_pref: float, max_soc: float) -> float:
    """
    Calculates potential-based shaping reward that guides agent behavior without
    changing the optimal policy.
    
    Args:
        soc_t: SoC at current time step
        soc_tp1: SoC at next time step
        gamma: Discount factor for future rewards
        min_soc: Minimum allowed SoC (hard constraint)
        low_pref: Lower bound of preferred SoC range (soft constraint)
        high_pref: Upper bound of preferred SoC range (soft constraint)
        max_soc: Maximum allowed SoC (hard constraint)
        
    Returns:
        float: Shaping reward component
    """
    pot_t = soc_potential(soc_t, min_soc, low_pref, high_pref, max_soc)
    pot_tp1 = soc_potential(soc_tp1, min_soc, low_pref, high_pref, max_soc)
    
    return gamma * pot_tp1 - pot_t


def soc_reward(soc: float, min_soc: float, low_pref: float, high_pref: float, max_soc: float,
              soc_limit_penalty_factor: float, preferred_soc_reward_factor: float) -> float:
    """
    Unified SoC reward function that provides graduated rewards/penalties based on SoC position.
    Includes extra penalties for very high SoC to discourage constant high-SoC operation.
    
    Args:
        soc: Current battery state of charge (0 to 1)
        min_soc: Minimum allowed SoC (hard constraint, e.g., 0.2)
        low_pref: Lower bound of preferred SoC range (soft constraint, e.g., 0.3)
        high_pref: Upper bound of preferred SoC range (soft constraint, e.g., 0.7)
        max_soc: Maximum allowed SoC (hard constraint, e.g., 0.8)
        soc_limit_penalty_factor: Base penalty for hard limit violations
        preferred_soc_reward_factor: Base reward for being in preferred range
        
    Returns:
        float: SoC-related reward component
    """
    # Get high SoC penalty parameters from config
    from src.rl.config import get_config_dict
    config = get_config_dict()
    high_soc_multiplier = config.get("high_soc_penalty_multiplier", 2.0)
    very_high_threshold = config.get("very_high_soc_threshold", 0.75)
    
    # Hard limit violations (severe penalties)
    if soc < min_soc:
        return -(min_soc - soc) * soc_limit_penalty_factor
    if soc > max_soc:
        return -(soc - max_soc) * soc_limit_penalty_factor * 2.0  # Double penalty for exceeding max SoC
    
    # Soft limit violations with progressive penalties
    if soc < low_pref:
        return -(low_pref - soc) * soc_limit_penalty_factor * 0.1  # 10% of hard limit penalty
    if soc > high_pref:
        # Apply stronger penalty for high SoC
        base_penalty = -(soc - high_pref) * soc_limit_penalty_factor * 0.15  # 15% of hard limit penalty
        
        # Additional penalty for very high SoC that increases exponentially with SoC
        if soc > very_high_threshold:
            # Add extra penalty that scales based on how far above very_high_threshold
            very_high_factor = (soc - very_high_threshold) / (max_soc - very_high_threshold)
            very_high_penalty = -soc_limit_penalty_factor * high_soc_multiplier * very_high_factor * 0.1
            return base_penalty + very_high_penalty
        
        return base_penalty
    
    # Inside preferred band (rewards)
    mid = 0.5 * (low_pref + high_pref)
    width = 0.5 * (high_pref - low_pref)
    # Reward is highest at center of range and decreases toward edges
    return preferred_soc_reward_factor * (1.0 - abs(soc - mid) / width)


def total_reward(components: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Aggregates all reward components using weighted sum approach.
    
    Args:
        components: Dictionary containing reward components
        weights: Dictionary of weight factors for each component
        
    Returns:
        float: Total aggregated reward
    """
    total = 0.0
    
    # Grid cost (negative penalty)
    if 'grid_cost' in components and 'w_grid' in weights:
        total += weights['w_grid'] * (-components['grid_cost'])
    
    # Capacity (peak demand)
    if 'capacity_penalty' in components and 'w_cap' in weights:
        total += weights['w_cap'] * (-components['capacity_penalty'])
    
    # Battery degradation
    if 'degradation_cost' in components and 'w_deg' in weights:
        total += weights['w_deg'] * (-components['degradation_cost'])
    
    # SoC management
    if 'soc_reward' in components and 'w_soc' in weights:
        total += weights['w_soc'] * components['soc_reward']
    
    # Potential-based shaping
    if 'shaping_reward' in components and 'w_shape' in weights:
        total += weights['w_shape'] * components['shaping_reward']
    
    # Night-time charging reward - only applied when actually charging the battery
    if all(k in components for k in ['night_discount', 'battery_charging_power']) and 'w_night' in weights:
        # Only add night charging reward if actually charging (battery_charging_power > 0)
        total += weights['w_night'] * components['night_discount'] * components['battery_charging_power']
    
    # Arbitrage bonus
    if 'arbitrage_bonus' in components and 'w_arbitrage' in weights:
        total += weights['w_arbitrage'] * components['arbitrage_bonus']
    
    # Export bonus
    if 'export_bonus' in components and 'w_export' in weights:
        total += weights['w_export'] * components['export_bonus']
    
    # Morning SoC target reward (new)
    if 'solar_soc_reward' in components and 'w_solar' in weights:
        total += weights['w_solar'] * components['solar_soc_reward']
    
    # Night-to-peak chain bonus (new)
    if 'night_peak_chain_bonus' in components and 'w_chain' in weights:
        total += weights['w_chain'] * components['night_peak_chain_bonus']
    
    # Action modification penalty - apply directly as a negative value
    if 'action_mod_penalty' in components and 'w_action_mod' in weights:
        # FIXED: Apply as a subtraction with the original positive penalty value
        total -= weights['w_action_mod'] * components['action_mod_penalty']
    
    return total

def adaptive_soc_targets(current_hour: int, price_forecast: List[float], solar_forecast: List[float], 
                        base_min: float, base_max: float) -> Tuple[float, float]:
    """
    Dynamically adjusts preferred SoC range based on price and solar forecasts.
    
    Args:
        current_hour: Current hour of day (0-23)
        price_forecast: Next 24h price forecast
        solar_forecast: Next 24h solar forecast
        base_min: Base minimum preferred SoC
        base_max: Base maximum preferred SoC
        
    Returns:
        Tuple[float, float]: Adjusted (min_pref, max_pref) SoC targets
    """
    # Calculate price percentiles for next 24 hours
    if len(price_forecast) >= 24:
        next_24h_prices = price_forecast[:24]
        price_25th = np.percentile(next_24h_prices, 25)
        price_75th = np.percentile(next_24h_prices, 75)
        current_price = price_forecast[0] if len(price_forecast) > 0 else price_25th
    else:
        # Fallback if insufficient data
        return base_min, base_max
    
    # Calculate expected solar for next 12 hours
    next_12h_solar = sum(solar_forecast[:12]) if len(solar_forecast) >= 12 else 0
    
    # Adjust targets based on time of day and forecasts
    adjusted_min = base_min
    adjusted_max = base_max
    
    # Night hours (22:00-06:00) - prepare for next day
    if current_hour >= 22 or current_hour <= 6:
        if current_price <= price_25th:  # Very low prices
            adjusted_min = max(0.15, base_min - 0.1)  # Allow lower SoC to charge more
            adjusted_max = min(0.85, base_max + 0.1)  # Encourage charging
        elif next_12h_solar > 3.0:  # Significant solar expected
            adjusted_max = min(0.7, base_max - 0.05)  # Leave room for solar
    
    # Morning hours (06:00-10:00) - prepare for solar
    elif 6 <= current_hour <= 10:
        if next_12h_solar > 2.0:  # Solar expected
            adjusted_max = min(0.7, base_max - 0.1)  # Make room for solar charging
    
    # Peak hours (16:00-20:00) - prepare for high prices
    elif 16 <= current_hour <= 20:
        if current_price >= price_75th:  # High prices
            adjusted_min = max(0.3, base_min + 0.1)  # Encourage higher SoC for discharge
    
    return adjusted_min, adjusted_max

class HomeEnergyEnv(gym.Env):
    """
    Simplified environment for controlling a home battery system
    reacting to grid prices.
    
    Observation space contains:
    - Battery state of charge
    - Time index (hour of day, minute of hour, day of week)
    - Price forecast for next 24 hours (from historical data)
    - Solar forecast for the next 3 days (from solar data)
    - Capacity metrics
    - Price averages
    - Is night discount flag
    
    Action space includes:
    - Battery charging/discharging rate (-1 to 1)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        config: Optional[Dict] = None
    ):
        """
        Initialize the environment.
        
        Args:
            config: Configuration dictionary containing all parameters.
                   If None, default values will be used.
        """
        super().__init__()
        self.config = config if config is not None else {}
        
        # Set up logger level if specified in config
        if self.config.get("log_level"):
            log_level = self.config.get("log_level")
            # Convert string level to actual logging level
            numeric_level = getattr(logging, log_level.upper(), None)
            if isinstance(numeric_level, int):
                logger.setLevel(numeric_level)
            else:
                logger.warning(f"Invalid log level: {log_level}, using default")
                
        # Flag to enable/disable timestamp sanity check prints
        self.debug_prints = self.config.get("debug_prints", False)
        
        # Counter for consecutive invalid actions
        self.consecutive_invalid_actions = 0
        self.max_consecutive_penalty_multiplier = self.config.get("max_consecutive_penalty_multiplier", 5.0)
        
        # SoC violation tracking with memory
        self.soc_violation_memory = 0.0  # Tracks cumulative violation severity
        self.soc_violation_memory_factor = self.config.get("soc_violation_memory_factor", 0.95)
        self.soc_violation_escalation_factor = self.config.get("soc_violation_escalation_factor", 1.5)
        
        # Variables for night charge tracking
        self.night_charge_pool = 0.0  # Energy charged during night hours (kWh)
        self.night_charge_timestamp = None  # When the night charging occurred
        self.night_charge_window_hours = self.config.get("night_charge_window_hours", 24.0)
                
        logger.info("\n{:=^80}".format(" Initializing HomeEnergyEnv "))

        # Load parameters from config with defaults
        self.battery_capacity = self.config.get("battery_capacity", 22.0)
        self.simulation_days = self.config.get("simulation_days", 7)
        self.render_mode = self.config.get("render_mode", None)
        self.use_price_predictions = self.config.get("use_price_predictions", True)
        self.price_predictions_path = self.config.get("price_predictions_path", PRICE_PREDICTIONS_PATH)
        self.fixed_baseload_kw = self.config.get("fixed_baseload_kw", 0.5)
        self.time_step_minutes = self.config.get("time_step_minutes", 15)
        self.use_variable_consumption = self.config.get("use_variable_consumption", False)
        self.consumption_data_path = self.config.get("consumption_data_path", None)
        self.battery_degradation_cost_per_kwh = self.config.get("battery_degradation_cost_per_kwh", 45.0)
        self.use_solar_predictions = self.config.get("use_solar_predictions", True)
        self.solar_data_path = self.config.get("solar_data_path", None)

        # New parameters from config for battery and rewards
        self.battery_initial_soc = self.config.get("battery_initial_soc", 0.5)
        self.battery_max_charge_power_kw = self.config.get("battery_max_charge_power_kw", self.battery_capacity / 2)
        self.battery_max_discharge_power_kw = self.config.get("battery_max_discharge_power_kw", self.battery_capacity / 2)
        self.battery_charge_efficiency = self.config.get("battery_charge_efficiency", 0.95)
        self.battery_discharge_efficiency = self.config.get("battery_discharge_efficiency", 0.95)
        self.soc_limit_penalty_factor = self.config.get("soc_limit_penalty_factor", 20.0)
        self.peak_power_threshold_kw = self.config.get("peak_power_threshold_kw", 5.0)
        self.peak_penalty_factor = self.config.get("peak_penalty_factor", 2.0)
        self.battery_degradation_reward_scaling_factor = self.config.get("battery_degradation_reward_scaling_factor", 1.0)
        
        # Arbitrage reward parameters
        self.enable_explicit_arbitrage_reward = self.config.get("enable_explicit_arbitrage_reward", False)
        self.low_price_threshold_ore_kwh = self.config.get("low_price_threshold_ore_kwh", 30.0)
        self.high_price_threshold_ore_kwh = self.config.get("high_price_threshold_ore_kwh", 80.0)
        self.charge_at_low_price_reward_factor = self.config.get("charge_at_low_price_reward_factor", 1.0)
        self.discharge_at_high_price_reward_factor = self.config.get("discharge_at_high_price_reward_factor", 1.0)
        self.use_percentile_price_thresholds = self.config.get("use_percentile_price_thresholds", False)
        self.low_price_percentile = self.config.get("low_price_percentile", 25.0)
        self.high_price_percentile = self.config.get("high_price_percentile", 75.0)
        
        # Price threshold values (will be set after price data is loaded if using percentiles)
        self.low_price_threshold_actual = self.low_price_threshold_ore_kwh  # Default to fixed threshold
        self.high_price_threshold_actual = self.high_price_threshold_ore_kwh  # Default to fixed threshold

        # Additional variables setup
        self.all_consumption_data_kw = None
        self.episode_consumption_kw = None
        self.all_solar_data_hourly_kw = None
        self.solar_forecast_actual = None
        self.solar_forecast_observed = None
        self.min_solar_data_date = None
        self.max_solar_data_date = None
        self.time_step_hours = self.time_step_minutes / 60.0
        
        # Optional: restrict training to a specific date range
        self.start_date_str = self.config.get("start_date", None)
        self.end_date_str = self.config.get("end_date", None)
        self.start_date = None
        self.end_date = None
        # Will parse these in _set_random_start_datetime
        logger.info(f"Time step configured: {self.time_step_minutes} minutes ({self.time_step_hours:.2f} hours).")
        
        # Log key parameters in a more consolidated way
        logger.info(
            f"Env Params: Sim Days: {self.simulation_days}, Battery: {self.battery_capacity} kWh, "
            f"Fixed Baseload: {self.fixed_baseload_kw} kW, Peak Penalty: {self.peak_penalty_factor}"
        )
        logger.info(
            f"Price Data: Use Predictions (CSV): {self.use_price_predictions}, Path: {self.price_predictions_path}"
        )
        if self.use_variable_consumption:
            logger.info(f"Consumption Data: Variable, Path: {self.consumption_data_path}")
        else:
            logger.info("Consumption Data: Using fixed baseload.")
        if self.use_solar_predictions:
            logger.info(f"Solar Data: Use Predictions: {self.use_solar_predictions}, Path: {self.solar_data_path}")
        else:
            logger.info("Solar Data: Not using predictions.")

        if not self.use_price_predictions:
            logger.critical(
                "CRITICAL: `use_price_predictions` is set to False, but this simplified environment "
                "is designed to exclusively use historical price data. Ensure `use_price_predictions` is True "
                "and a valid `price_predictions_path` is provided in the configuration. EXITING."
            )
            sys.exit(1)
            
        self.price_predictions_df = None # Loaded in _load_price_predictions
        
        # Time parameters
        self.simulation_hours = self.simulation_days * 24
        self.simulation_steps = int(self.simulation_hours / self.time_step_hours)
        self.current_step = 0
        self.start_datetime = None
        self.last_action = None # For rendering
        
        self.min_price_data_date = None
        self.max_price_data_date = None
        
        self.battery = Battery(
            capacity_kwh=self.battery_capacity, 
            degradation_cost_per_kwh=self.battery_degradation_cost_per_kwh,
            initial_soc=self.battery_initial_soc,
            max_charge_power_kw=self.battery_max_charge_power_kw,
            max_discharge_power_kw=self.battery_max_discharge_power_kw,
            charge_efficiency=self.battery_charge_efficiency,
            discharge_efficiency=self.battery_discharge_efficiency
        )
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        self.observation_space = spaces.Dict({
            "soc": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "time_idx": spaces.Box(
                low=np.array([0, 0, 0]),  # hour of day, minute of hour, day of week
                high=np.array([23, 1, 6]),  # minute_of_hour is normalized (0 to <1)
                dtype=np.float32
            ),
            "price_forecast": spaces.Box(
                low=0.0, high=10.0, shape=(24,), dtype=np.float32 # Placeholder range, actual range depends on data
            ),
            "solar_forecast": spaces.Box( # 3 days * 24 hours
                low=0.0, high=10.0, shape=(3 * 24,), dtype=np.float32 # Placeholder high, actual solar capacity
            ),
            "capacity_metrics": spaces.Box(  # New metrics for capacity tariff
                low=0.0, 
                high=np.array([20.0, 20.0, 20.0, 20.0, 1.0]), # [top1, top2, top3, rolling_avg, month_progress]
                shape=(5,), 
                dtype=np.float32
            ),
            "price_averages": spaces.Box(  # Price averages for better context
                low=0.0, high=1000.0, shape=(2,), dtype=np.float32  # [24h_avg, 168h_avg]
            ),
            "is_night_discount": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "load_forecast": spaces.Box( # 3 days * 24 hours for household consumption
                low=0.0, high=20.0, shape=(3 * 24,), dtype=np.float32 # Placeholder high, adjust with actual data
            )
        })
        
        # State variables for tracking
        self.price_history = []
        self.grid_cost_history = []
        self.battery_cost_history = []
        self.total_cost = 0.0
        self.peak_power = 0.0
        self.total_reward = 0.0 # Reset total reward for the episode
        
        # Capacity fee related attributes, reset per episode/month
        self.top3_peaks: List[float] = [0.0, 0.0, 0.0]
        self.peak_rolling_average: float = 0.0
        # Stores (timestamp, effective_15min_peak_power) tuples for the current month's calculation basis
        self.current_month_peak_data: List[Tuple[datetime.datetime, float]] = [] 

        if self.use_price_predictions:
            self._load_price_predictions()
            if self.price_predictions_df is not None and not self.price_predictions_df.empty:
                self.min_price_data_date = self.price_predictions_df.index.min()
                self.max_price_data_date = self.price_predictions_df.index.max()
            else:
                logger.critical(
                    "Price predictions CSV data is None or empty after attempting to load, "
                    "but `use_price_predictions` is True. Historical data is essential. EXITING."
                )
                sys.exit(1)
        
        # Initialize consumption data dates to None or to actuals if not using price predictions (for fallback range)
        min_consum_date = None
        max_consum_date = None

        if self.use_variable_consumption:
            self._load_consumption_data()
            if self.all_consumption_data_kw is None or self.all_consumption_data_kw.empty:
                logger.critical(
                    "CRITICAL: Variable consumption is enabled, but consumption data is missing or empty. "
                    "Ensure `consumption_data_path` is correct and the file is valid. EXITING."
                )
                sys.exit(1)
            else:
                min_consum_date = self.all_consumption_data_kw.index.min()
                max_consum_date = self.all_consumption_data_kw.index.max()

        if self.use_solar_predictions:
            self._load_solar_data()
            if self.all_solar_data_hourly_kw is None or self.all_solar_data_hourly_kw.empty:
                logger.critical(
                    "CRITICAL: Solar predictions are enabled, but solar data is missing or empty. "
                    "Ensure `solar_data_path` is correct and the file is valid. EXITING."
                )
                sys.exit(1)
            else:
                self.min_solar_data_date = self.all_solar_data_hourly_kw.index.min()
                self.max_solar_data_date = self.all_solar_data_hourly_kw.index.max()
                # Adjust observation space for solar_forecast high value if possible
                if 'solar_production_kw' in self.all_solar_data_hourly_kw.columns:
                    max_solar_val = self.all_solar_data_hourly_kw['solar_production_kw'].max()
                    if pd.notna(max_solar_val) and max_solar_val > 0:
                        self.observation_space.spaces['solar_forecast'] = spaces.Box(
                            low=0.0, high=float(max_solar_val * 1.1), # Add 10% buffer
                            shape=(3*24,), dtype=np.float32 # Ensure shape is 72 here too
                        )
                        logger.info(f"Adjusted solar_forecast observation space high to: {max_solar_val * 1.1:.2f} kW for 72 hours")

        # Adjust observation space for load_forecast high value if using variable consumption
        if self.use_variable_consumption and self.all_consumption_data_kw is not None and not self.all_consumption_data_kw.empty:
            if 'consumption_kw' in self.all_consumption_data_kw.columns:
                max_load_val = self.all_consumption_data_kw['consumption_kw'].max() # Assuming all_consumption_data_kw is already at simulation frequency
                # If all_consumption_data_kw is hourly, this max might be an hourly peak.
                # We might need to consider the resampling to time_step_minutes when estimating a general peak.
                # For now, using the max of the loaded (potentially resampled) data.
                if pd.notna(max_load_val) and max_load_val > 0:
                    self.observation_space.spaces['load_forecast'] = spaces.Box(
                        low=0.0, high=float(max_load_val * 1.2), # Add 20% buffer for load, can be more variable
                        shape=(3*24,), dtype=np.float32
                    )
                    logger.info(f"Adjusted load_forecast observation space high to: {max_load_val * 1.2:.2f} kW for 72 hours")


        logger.info("{:-^80}".format(" Environment Initialized "))
        
    def _set_random_start_datetime(self) -> None:
        """Sets self.start_datetime to a random valid point based on available data ranges."""
        min_consum_date = None
        max_consum_date = None
        if self.use_variable_consumption:
            if self.all_consumption_data_kw is not None and not self.all_consumption_data_kw.empty:
                min_consum_date = self.all_consumption_data_kw.index.min()
                max_consum_date = self.all_consumption_data_kw.index.max()
            else:
                logger.warning("Variable consumption enabled, but all_consumption_data_kw is not available. "
                               "This might impact date ranging for random start.")

        effective_min_data_date = self.min_price_data_date 
        effective_max_data_date = self.max_price_data_date

        if self.use_variable_consumption and min_consum_date is not None and max_consum_date is not None:
            if effective_min_data_date is None:
                effective_min_data_date = min_consum_date
                effective_max_data_date = max_consum_date 
            else:
                effective_min_data_date = max(effective_min_data_date, min_consum_date)
                if effective_max_data_date is not None:
                     effective_max_data_date = min(effective_max_data_date, max_consum_date)
                else:
                     effective_max_data_date = max_consum_date
        
        if self.use_solar_predictions and self.min_solar_data_date is not None and self.max_solar_data_date is not None:
            if effective_min_data_date is None:
                effective_min_data_date = self.min_solar_data_date
                effective_max_data_date = self.max_solar_data_date
            else:
                effective_min_data_date = max(effective_min_data_date, self.min_solar_data_date)
                if effective_max_data_date is not None:
                    effective_max_data_date = min(effective_max_data_date, self.max_solar_data_date)
                else:
                    effective_max_data_date = self.max_solar_data_date

        # Restrict by user-specified start_date and end_date if provided
        tz = None
        # Try to use the timezone of the data if available
        if hasattr(effective_min_data_date, 'tzinfo') and effective_min_data_date.tzinfo is not None:
            tz = effective_min_data_date.tzinfo
        if self.start_date_str:
            try:
                self.start_date = pd.Timestamp(self.start_date_str)
                if tz is not None and self.start_date.tzinfo is None:
                    self.start_date = self.start_date.tz_localize(tz)
                effective_min_data_date = max(effective_min_data_date, self.start_date)
                logger.info(f"Restricting minimum episode start date to {self.start_date} (user-specified)")
            except Exception as e:
                logger.error(f"Could not parse start_date '{self.start_date_str}': {e}")
                raise
        if self.end_date_str:
            try:
                self.end_date = pd.Timestamp(self.end_date_str)
                if tz is not None and self.end_date.tzinfo is None:
                    self.end_date = self.end_date.tz_localize(tz)
                effective_max_data_date = min(effective_max_data_date, self.end_date)
                logger.info(f"Restricting maximum episode start date to {self.end_date} (user-specified)")
            except Exception as e:
                logger.error(f"Could not parse end_date '{self.end_date_str}': {e}")
                raise
        if effective_min_data_date > effective_max_data_date:
            logger.critical(f"No valid data range after applying start_date ({self.start_date}) and end_date ({self.end_date}). Exiting.")
            raise ValueError(f"No valid data range after applying start_date ({self.start_date}) and end_date ({self.end_date})")

        max_possible_start_date = effective_max_data_date - pd.Timedelta(days=self.simulation_days) # simulation_days is from config

        if effective_min_data_date > max_possible_start_date:
            logger.warning(
                f"Overall data range {effective_min_data_date} to {effective_max_data_date} is too short for simulation_days ({self.simulation_days}). "
                f"Attempting to start at earliest possible: {effective_min_data_date}"
            )
            self.start_datetime = effective_min_data_date
        else:
            time_delta_seconds = (max_possible_start_date - effective_min_data_date).total_seconds()
            if time_delta_seconds < 0:
                 logger.warning(f"max_possible_start_date ({max_possible_start_date}) is before effective_min_data_date ({effective_min_data_date}). Defaulting to effective_min_data_date.")
                 self.start_datetime = effective_min_data_date
            else:
                # Ensure start_datetime is at the beginning of an hour, or consistent with time_step_minutes
                # Normalizing to day and then adding random offset of days is safer.
                min_day_norm = effective_min_data_date.normalize()
                max_day_norm = max_possible_start_date.normalize()
                num_sampleable_days = (max_day_norm - min_day_norm).days + 1
                if num_sampleable_days <= 0:
                    logger.warning(f"Calculated num_sampleable_days is {num_sampleable_days} based on effective range. Defaulting to effective_min_data_date.")
                    self.start_datetime = effective_min_data_date # Fallback, could be non-normalized.
                else:
                    random_day_offset = np.random.randint(0, num_sampleable_days)
                    self.start_datetime = min_day_norm + pd.Timedelta(days=random_day_offset)
        
        # Ensure self.start_datetime is timezone-aware if other data is (e.g., price data)
        # This should align with how price_predictions_df.index is handled (localized to Europe/Stockholm)
        if self.price_predictions_df is not None and self.price_predictions_df.index.tzinfo is not None:
            if self.start_datetime.tzinfo is None:
                self.start_datetime = self.start_datetime.tz_localize(self.price_predictions_df.index.tzinfo, ambiguous='raise', nonexistent='raise')
            elif self.start_datetime.tzinfo != self.price_predictions_df.index.tzinfo:
                self.start_datetime = self.start_datetime.tz_convert(self.price_predictions_df.index.tzinfo)
        logger.info(f"Random simulation start datetime set to: {self.start_datetime}")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            logger.info(f"Random seed set to: {seed}")
        
        self.current_step = 0
        self.consecutive_invalid_actions = 0  # Reset consecutive invalid actions counter
        self.soc_violation_memory = 0.0  # Reset SoC violation memory

        # Log effective data ranges for all sources
        logger.info("--- DATA RANGE SUMMARY ---")
        logger.info(f"Price data: {self.min_price_data_date} to {self.max_price_data_date}")
        if self.use_variable_consumption and self.all_consumption_data_kw is not None:
            logger.info(f"Consumption data: {self.all_consumption_data_kw.index.min()} to {self.all_consumption_data_kw.index.max()}")
        if self.use_solar_predictions and self.all_solar_data_hourly_kw is not None:
            logger.info(f"Solar data: {self.all_solar_data_hourly_kw.index.min()} to {self.all_solar_data_hourly_kw.index.max()}")
        logger.info("--------------------------")
        
        # Determine start_datetime for the episode
        if self.config.get("force_specific_date_range", False):
            start_date = self.config.get("start_date")
            end_date = self.config.get("end_date")
            
            if start_date is not None and end_date is not None:
                default_tz_str = 'Europe/Stockholm'  # Consistent with price loading
                tz_info_to_use = None
                
                # Try to infer timezone from loaded price data if available
                if self.price_predictions_df is not None and self.price_predictions_df.index.tzinfo is not None:
                    tz_info_to_use = self.price_predictions_df.index.tzinfo
                elif self.min_price_data_date is not None and self.min_price_data_date.tzinfo is not None:
                    tz_info_to_use = self.min_price_data_date.tzinfo
                
                try:
                    # Create a naive datetime from start_date string, then localize
                    start_date_parts = start_date.split('-')
                    if len(start_date_parts) != 3:
                        raise ValueError(f"Invalid start date format: {start_date}. Expected YYYY-MM-DD.")
                    
                    year, month, day = map(int, start_date_parts)
                    naive_dt = datetime.datetime(year, month, day, 0, 0)
                    
                    if tz_info_to_use is not None:
                        self.start_datetime = pd.Timestamp(naive_dt).tz_localize(tz_info_to_use, ambiguous='raise', nonexistent='raise')
                    else:  # Fallback: localize to default_tz_str if no other info
                        self.start_datetime = pd.Timestamp(naive_dt).tz_localize(default_tz_str, ambiguous='raise', nonexistent='raise')
                    
                    logger.info(f"Forcing simulation to start at: {self.start_datetime} (date range from {start_date} to {end_date})")
                
                except Exception as e:
                    logger.error(f"Could not set forced start datetime for date range {start_date} to {end_date} with tz {tz_info_to_use or default_tz_str}: {e}. Falling back to random start.")
                    self._set_random_start_datetime()
            else:
                logger.warning("force_specific_date_range is True, but start_date or end_date not in config. Falling back to random start.")
                self._set_random_start_datetime()
        
        elif self.config.get("force_specific_start_month", False):
            start_year = self.config.get("start_year")
            start_month = self.config.get("start_month")
            if start_year is not None and start_month is not None:
                default_tz_str = 'Europe/Stockholm' # Consistent with price loading
                tz_info_to_use = None
                # Try to infer timezone from loaded price data if available
                if self.price_predictions_df is not None and self.price_predictions_df.index.tzinfo is not None:
                    tz_info_to_use = self.price_predictions_df.index.tzinfo
                elif self.min_price_data_date is not None and self.min_price_data_date.tzinfo is not None:
                    tz_info_to_use = self.min_price_data_date.tzinfo
                
                try:
                    # Create a naive datetime first, then localize if tz_info_to_use is known, else default
                    naive_dt = datetime.datetime(start_year, start_month, 1, 0, 0)
                    if tz_info_to_use is not None:
                        self.start_datetime = pd.Timestamp(naive_dt).tz_localize(tz_info_to_use, ambiguous='raise', nonexistent='raise')
                    else: # Fallback: localize to default_tz_str if no other info
                        self.start_datetime = pd.Timestamp(naive_dt).tz_localize(default_tz_str, ambiguous='raise', nonexistent='raise')
                    
                    logger.info(f"Forcing simulation to start at: {self.start_datetime} (forced month)")
                except Exception as e:
                    logger.error(f"Could not set forced start datetime for {start_year}-{start_month} with tz {tz_info_to_use or default_tz_str}: {e}. Falling back to random start.")
                    self._set_random_start_datetime()
            else:
                logger.warning("force_specific_start_month is True, but start_year or start_month not in config. Falling back to random start.")
                self._set_random_start_datetime()
        else:
            self._set_random_start_datetime()

        # Initialize month_start_date and days_in_month based on the determined self.start_datetime
        self.month_start_date = self.start_datetime.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        next_month_start_date = (self.month_start_date + pd.DateOffset(months=1))
        self.days_in_month = (next_month_start_date - self.month_start_date).days
        logger.info(f"Current month for capacity fee: {self.month_start_date.strftime('%Y-%m')}, days: {self.days_in_month}")

        self.battery.reset()
        self.peak_power = 0.0  # Reset overall peak power for the episode
        self.total_reward = 0.0 # Reset total reward for the episode

        # Reset capacity metrics for the new episode
        self.top3_peaks = [0.0, 0.0, 0.0]
        self.peak_rolling_average = 0.0
        self.current_month_peak_data = []
        
        # Reset night charge tracking
        self.night_charge_pool = 0.0
        self.night_charge_timestamp = None

        # Initialize forecasts for the new episode start
        self._initialize_forecasts()
        if self.use_solar_predictions:
            self._initialize_solar_forecasts()
        
        # Initialize load forecasts for the new episode start if using variable consumption
        if self.use_variable_consumption:
            self._initialize_load_forecasts()

        # Calculate price thresholds based on percentiles if enabled
        if self.enable_explicit_arbitrage_reward and self.use_percentile_price_thresholds and self.price_forecast_actual is not None:
            # Calculate percentile-based price thresholds from this episode's price data
            try:
                self.low_price_threshold_actual = np.percentile(self.price_forecast_actual, self.low_price_percentile)
                self.high_price_threshold_actual = np.percentile(self.price_forecast_actual, self.high_price_percentile)
                logger.info(f"Using percentile-based price thresholds: low={self.low_price_threshold_actual:.2f} öre/kWh (p{self.low_price_percentile}), "
                           f"high={self.high_price_threshold_actual:.2f} öre/kWh (p{self.high_price_percentile})")
            except Exception as e:
                logger.warning(f"Failed to calculate percentile-based price thresholds: {e}. Using fixed thresholds.")
                self.low_price_threshold_actual = self.low_price_threshold_ore_kwh
                self.high_price_threshold_actual = self.high_price_threshold_ore_kwh
        else:
            # Use fixed thresholds from config
            self.low_price_threshold_actual = self.low_price_threshold_ore_kwh
            self.high_price_threshold_actual = self.high_price_threshold_ore_kwh
            if self.enable_explicit_arbitrage_reward:
                logger.info(f"Using fixed price thresholds: low={self.low_price_threshold_actual:.2f} öre/kWh, "
                           f"high={self.high_price_threshold_actual:.2f} öre/kWh")
        
        self.price_history = []
        self.grid_cost_history = []
        self.battery_cost_history = []
        
        if self.use_variable_consumption:
            if self.all_consumption_data_kw is None or self.all_consumption_data_kw.empty:
                logger.critical("Variable consumption enabled but no data loaded. Exiting.")
                sys.exit(1)
            
            # Ensure self.start_datetime is timezone-aware (UTC) for proper indexing
            start_dt_utc = pd.Timestamp(self.start_datetime).tz_localize('UTC') if pd.Timestamp(self.start_datetime).tzinfo is None else pd.Timestamp(self.start_datetime).tz_convert('UTC')
            
            end_dt_utc = start_dt_utc + pd.Timedelta(hours=self.simulation_hours) - pd.Timedelta(minutes=int(self.time_step_hours*60)) # Ensure end is inclusive for reindex

            # Filter the consumption data for the simulation period
            # We need to align the consumption data's timestamps with our simulation steps
            
            # Create the target index for the simulation period
            sim_timestamps_utc = pd.date_range(start=start_dt_utc, periods=self.simulation_steps, freq=f"{int(self.time_step_hours*60)}min")

            try:
                # Align all_consumption_data_kw to UTC if it's not already (should be from loading)
                if self.all_consumption_data_kw.index.tzinfo is None:
                     self.all_consumption_data_kw.index = self.all_consumption_data_kw.index.tz_localize('UTC')
                else:
                     self.all_consumption_data_kw.index = self.all_consumption_data_kw.index.tz_convert('UTC')

                # Reindex and fill to match simulation steps exactly
                episode_consumption_df = self.all_consumption_data_kw.reindex(sim_timestamps_utc, method='ffill').bfill()
                
                if episode_consumption_df.isnull().values.any() or len(episode_consumption_df) < self.simulation_steps:
                    logger.error(f"Could not retrieve a full consumption profile for the simulation period starting {start_dt_utc}. "
                                 f"Needed {self.simulation_steps} steps, got {len(episode_consumption_df.dropna())}. "
                                 "Check data range and resampling. Defaulting to fixed baseload for this episode.")
                    self.episode_consumption_kw = np.full(self.simulation_steps, self.fixed_baseload_kw)
                else:
                    self.episode_consumption_kw = episode_consumption_df['consumption_kw'].values[:self.simulation_steps]
                    # Apply data augmentation if enabled
                    if self.config.get("use_data_augmentation", False) and self.config.get("augment_consumption_data", True):
                        augmentation_factor = self.config.get("consumption_augmentation_factor", 0.15)
                        
                        # Generate random scaling factor for the episode
                        consumption_scale = np.random.normal(1.0, augmentation_factor)
                        consumption_scale = max(0.7, min(1.3, consumption_scale))  # Limit scaling to reasonable range
                        
                        # Apply augmentation
                        self.episode_consumption_kw = self.episode_consumption_kw * consumption_scale
                        logger.info(f"Applied consumption data augmentation with scaling factor: {consumption_scale:.2f}")

            except Exception as e:
                logger.error(f"Error selecting episode consumption data: {e}. Defaulting to fixed baseload.")
                self.episode_consumption_kw = np.full(self.simulation_steps, self.fixed_baseload_kw) # Fallback
        else:
            self.episode_consumption_kw = np.full(self.simulation_steps, self.fixed_baseload_kw) # Fallback if not using variable
            logger.info(f"Using fixed baseload of {self.fixed_baseload_kw} kW for this episode.")

        observation = self._get_observation()
        info = self._get_info()
        return observation, info
    
    def _map_action_to_battery_power(self, action: float) -> float:
        """
        Maps the normalized action value (-1 to 1) to an appropriate battery power command,
        taking into account the asymmetric charge/discharge capabilities.
        
        Args:
            action (float): Normalized action from the agent in range [-1, 1]
            
        Returns:
            float: Target battery power in kW (negative for charging, positive for discharging)
        """
        if action >= 0:  # Discharging (0 to 1 maps to 0 to max_discharge)
            return action * self.battery_max_discharge_power_kw
        else:  # Charging (negative) (-1 to 0 maps to -max_charge to 0)
            return action * self.battery_max_charge_power_kw
            
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one time step within the environment.

        Args:
            action: A_step (np.ndarray): Agent's action, a 1D array with a single value in [-1, 1].
                    -1 indicates maximum charging rate, 1 indicates maximum discharging rate.

        Returns:
            observation (Dict): Agent's observation of the current environment.
            reward (float): Amount of reward returned after previous action.
            terminated (bool): Whether the episode has ended (e.g., time limit reached).
            truncated (bool): Whether the episode was truncated (not used here, use terminated).
            info (Dict): Contains auxiliary diagnostic information useful for debugging, logging, and learning.
        """
        # 1. Apply Safe Action Masking
        # Store the original action for logging/debugging
        original_action = action[0]
        
        # Apply safety constraints to avoid SoC violations
        soc_min_limit = self.config.get("soc_min_limit", 0.2)
        soc_max_limit = self.config.get("soc_max_limit", 0.8)
        
        # Apply the safety mask to constrain actions that would violate SoC limits
        safe_action = safe_action_mask(
            raw_action=action[0],
            soc=self.battery.soc,
            min_soc=soc_min_limit,
            max_soc=soc_max_limit,
            max_charge_power_kw=self.battery_max_charge_power_kw,
            max_discharge_power_kw=self.battery_max_discharge_power_kw,
            battery_capacity_kwh=self.battery_capacity,
            time_step_hours=self.time_step_hours
        )
        
        # Additional safety check using our safety buffer
        safe_action = ensure_soc_limits(
            soc=self.battery.soc,
            action=safe_action,
            min_soc=soc_min_limit,
            max_soc=soc_max_limit,
            max_charge_power_kw=self.battery_max_charge_power_kw,
            max_discharge_power_kw=self.battery_max_discharge_power_kw,
            capacity_kwh=self.battery_capacity,
            time_step_hours=self.time_step_hours
        )
        
        # Record if action was modified
        action_modified = (original_action != safe_action)
        if action_modified and self.debug_prints:
            logger.debug(f"Step {self.current_step}: Action modified from {original_action:.4f} to {safe_action:.4f} to protect SoC limits")
        
        # Convert the safe normalized action to actual power
        target_battery_terminal_power_kw = self._map_action_to_battery_power(safe_action)

        # 2. Get Current Actuals for this step
        current_dt = self.start_datetime + datetime.timedelta(hours=self.current_step * self.time_step_hours)
        current_price_ore_per_kwh = self.price_forecast_actual[self.current_step]
        
        current_solar_kw = 0.0
        if self.use_solar_predictions and self.solar_forecast_actual is not None and self.current_step < len(self.solar_forecast_actual):
            current_solar_kw = self.solar_forecast_actual[self.current_step]
        
        current_consumption_kw = self.fixed_baseload_kw # Default to fixed baseload
        if self.use_variable_consumption and self.episode_consumption_kw is not None and self.current_step < len(self.episode_consumption_kw):
            current_consumption_kw = self.episode_consumption_kw[self.current_step]

        # 3. Battery Operation
        # Store current SoC for reward shaping calculation
        soc_before = self.battery.soc
        
        # Apply action to battery
        actual_battery_power_kw_at_terminals, energy_change_in_storage_kwh, limited_by_soc = self.battery.step(
            target_power_kw=target_battery_terminal_power_kw,
            duration_hours=self.time_step_hours,
            min_soc=soc_min_limit,
            max_soc=soc_max_limit
        )
        
        # Get the SoC after action
        soc_after = self.battery.soc
        
        # Track battery operations for debugging
        if self.debug_prints and (self.current_step % 20 == 0 or action_modified or limited_by_soc):
            logger.debug(f"Battery operation - Step {self.current_step}: "
                        f"Target power: {target_battery_terminal_power_kw:.2f} kW, "
                        f"Actual power: {actual_battery_power_kw_at_terminals:.2f} kW, "
                        f"SoC: {soc_before:.2f} → {soc_after:.2f}, "
                        f"Limited by SoC: {limited_by_soc}, "
                        f"Modified: {action_modified}, "
                        f"Original action: {original_action:.2f}, "
                        f"Safe action: {safe_action:.2f}")

        # 4. Net Power Calculation
        # Net consumption by the house after its own solar production
        net_house_demand_kw = current_consumption_kw - current_solar_kw
        
        # Grid power is what the house needs minus what the battery provides (or plus what battery consumes)
        # If battery is discharging (positive actual_battery_power_kw_at_terminals), it reduces grid import.
        # If battery is charging (negative actual_battery_power_kw_at_terminals), it increases grid import.
        grid_power_kw = net_house_demand_kw - actual_battery_power_kw_at_terminals
        
        # Positive grid_power_kw: drawing from grid (import)
        # Negative grid_power_kw: exporting to grid (export)
        
        # Track peak power for capacity fee calculation if importing from grid
        if grid_power_kw > 0:
            # Check if current power is one of the top 3 peaks this month
            is_night = False
            if hasattr(self, 'is_night_discount') and self.current_step < len(self.is_night_discount):
                is_night = self.is_night_discount[self.current_step]
            
            # Apply night discount to peak value if applicable
            night_capacity_discount = self.config.get("night_capacity_discount", 0.5)
            effective_grid_power = grid_power_kw * (night_capacity_discount if is_night else 1.0)
            
            # Use the new method that enforces same-day constraint (Swedish regulation)
            self._update_capacity_peaks_with_same_day_constraint(current_dt, effective_grid_power)
                
            # Also track overall peak power for the episode
            if grid_power_kw > self.peak_power:
                self.peak_power = grid_power_kw

        # 5. Reward Calculation using the new component-based approach
        
        # Time-dependent parameters
        gamma = self.config.get("short_term_gamma", 0.98)  # Discount factor for shaping
        current_hour = current_dt.hour
        is_night = False
        if hasattr(self, 'is_night_discount') and self.current_step < len(self.is_night_discount):
            is_night = self.is_night_discount[self.current_step]
            
        # Log pricing and energy flows for debugging
        if self.debug_prints and self.current_step % 20 == 0:
            logger.debug(f"Energy flows - Step {self.current_step}: "
                        f"Time: {current_dt.strftime('%Y-%m-%d %H:%M')} {'(Night)' if is_night else ''}, "
                        f"Price: {current_price_ore_per_kwh:.2f} öre/kWh, "
                        f"Grid Power: {grid_power_kw:.2f} kW, "
                        f"Battery Power: {actual_battery_power_kw_at_terminals:.2f} kW, "
                        f"Consumption: {current_consumption_kw:.2f} kW, "
                        f"Solar: {current_solar_kw:.2f} kW")
        
        # Get base SoC parameters and adapt them based on forecasts
        base_preferred_soc_min = self.config.get("preferred_soc_min_base", 0.25)
        base_preferred_soc_max = self.config.get("preferred_soc_max_base", 0.75)
        
        # Get price and solar forecasts for adaptive SoC management
        price_forecast = []
        solar_forecast = []
        
        # Extract price forecast for next 24 hours (hourly)
        remaining_steps = len(self.price_forecast_observed) - self.current_step
        if remaining_steps > 0:
            price_forecast = self.price_forecast_observed[self.current_step][:24]  # Next 24h hourly prices
        
        # Extract solar forecast for next 24 hours (hourly)
        if hasattr(self, 'solar_forecast_observed') and self.solar_forecast_observed is not None:
            remaining_solar_steps = len(self.solar_forecast_observed) - self.current_step
            if remaining_solar_steps > 0:
                solar_forecast = self.solar_forecast_observed[self.current_step][:24]  # Next 24h hourly solar
        
        # Apply adaptive SoC targeting
        preferred_soc_min, preferred_soc_max = adaptive_soc_targets(
            current_hour=current_hour,
            price_forecast=price_forecast,
            solar_forecast=solar_forecast,
            base_min=base_preferred_soc_min,
            base_max=base_preferred_soc_max
        )
        
        soc_limit_penalty_factor = self.config.get("soc_limit_penalty_factor", 100.0)
        preferred_soc_reward_factor = self.config.get("preferred_soc_reward_factor", 20.0)
        
        # 5.1 Calculate each reward component separately
        reward_components = {}
        
        # Grid cost component (always present)
        grid_energy_kwh = grid_power_kw * self.time_step_hours
        grid_cost = 0.0
        
        if grid_power_kw > 0:  # Importing from grid
            energy_tax = self.config.get("energy_tax", 54.875)
            vat_mult = self.config.get("vat_mult", 1.25)
            grid_fee_val = self.config.get("grid_fee", 6.25)
            
            spot_with_vat = current_price_ore_per_kwh * vat_mult
            cost_ore = spot_with_vat + grid_fee_val + energy_tax
            
            grid_cost = (grid_power_kw * cost_ore * self.time_step_hours)  # in öre
            grid_cost = grid_cost * self.config.get("grid_cost_scaling_factor", 1.0)
            
            # Update tracking metrics
            self.total_cost += grid_cost / 100.0  # Convert öre to SEK
            self.grid_cost_history.append(grid_cost)
        
        reward_components['grid_cost'] = grid_cost
        
        # Capacity penalty component (Peak shaving incentive)
        capacity_penalty = 0.0
        peak_power_threshold_kw = self.config.get("peak_power_threshold_kw", 5.0)
        
        if grid_power_kw > peak_power_threshold_kw:
            # Penalty for exceeding threshold
            excess_power = grid_power_kw - peak_power_threshold_kw
            peak_penalty_factor = self.config.get("peak_penalty_factor", 50.0)
            capacity_penalty = excess_power * peak_penalty_factor
        
        reward_components['capacity_penalty'] = capacity_penalty
        
        # Battery degradation component
        degradation_cost = 0.0
        if abs(energy_change_in_storage_kwh) > 0:
            # Calculate the battery degradation cost in öre
            energy_throughput = abs(energy_change_in_storage_kwh)
            degradation_cost = energy_throughput * self.battery_degradation_cost_per_kwh
            degradation_cost *= self.battery_degradation_reward_scaling_factor
            
            self.battery_cost_history.append(degradation_cost)
        
        reward_components['degradation_cost'] = degradation_cost
        
        # Update SoC violation memory (decay previous violations, add current ones)
        self.soc_violation_memory *= self.soc_violation_memory_factor
        
        # Check for current SoC violations and add to memory
        current_violation_severity = 0.0
        if soc_after < soc_min_limit:
            current_violation_severity = (soc_min_limit - soc_after) * 10.0  # Amplify violation
        elif soc_after > soc_max_limit:
            current_violation_severity = (soc_after - soc_max_limit) * 15.0  # Higher penalty for over-charging
        
        self.soc_violation_memory += current_violation_severity
        
        # Calculate escalated penalty factor based on violation memory
        violation_escalation = 1.0 + (self.soc_violation_memory * self.soc_violation_escalation_factor)
        
        # Unified SoC reward component with violation memory
        soc_rew = soc_reward(
            soc=soc_after, 
            min_soc=soc_min_limit,
            low_pref=preferred_soc_min,
            high_pref=preferred_soc_max,
            max_soc=soc_max_limit,
            soc_limit_penalty_factor=soc_limit_penalty_factor * violation_escalation,  # Apply escalation
            preferred_soc_reward_factor=preferred_soc_reward_factor
        )
        reward_components['soc_reward'] = soc_rew
        
        # Potential-based shaping reward
        shape_rew = shaping_reward(
            soc_t=soc_before,
            soc_tp1=soc_after,
            gamma=gamma,
            min_soc=soc_min_limit,
            low_pref=preferred_soc_min,
            high_pref=preferred_soc_max,
            max_soc=soc_max_limit
        )
        reward_components['shaping_reward'] = shape_rew
        
        # Night-time discount incentive (encourage night charging)
        reward_components['night_discount'] = float(is_night)
        reward_components['grid_import'] = max(0, grid_power_kw)  # Only positive for import
        
        # Export bonus component
        export_bonus = 0.0
        if grid_power_kw < 0:  # Exporting to grid
            export_energy_kwh = abs(grid_energy_kwh)
            export_reward_bonus_ore_kwh = self.config.get("export_reward_bonus_ore_kwh", 60.0)
            export_reward_scaling_factor = self.config.get("export_reward_scaling_factor", 0.1)
            export_revenue_ore = (current_price_ore_per_kwh + export_reward_bonus_ore_kwh) * export_energy_kwh
            export_revenue_ore = export_revenue_ore * export_reward_scaling_factor # Scaling factor to get a better range for the agent training
            export_bonus = export_revenue_ore
        reward_components['export_bonus'] = export_bonus
        
        # Arbitrage bonus component
        arbitrage_bonus = 0.0
        if self.enable_explicit_arbitrage_reward:
            charge_at_low_price_reward_factor = self.config.get("charge_at_low_price_reward_factor", 1.0)
            discharge_at_high_price_reward_factor = self.config.get("discharge_at_high_price_reward_factor", 2.0)
            
            # Charging at low prices
            if actual_battery_power_kw_at_terminals < 0 and current_price_ore_per_kwh < self.low_price_threshold_actual:
                energy_charged_kwh = abs(actual_battery_power_kw_at_terminals) * self.time_step_hours
                arbitrage_bonus += energy_charged_kwh * charge_at_low_price_reward_factor
            # Discharging at high prices
            elif actual_battery_power_kw_at_terminals > 0 and current_price_ore_per_kwh > self.high_price_threshold_actual:
                energy_discharged_kwh = actual_battery_power_kw_at_terminals * self.time_step_hours
                arbitrage_bonus += energy_discharged_kwh * discharge_at_high_price_reward_factor
        reward_components['arbitrage_bonus'] = arbitrage_bonus
        
        # Calculate battery charging power for night charging reward
        # This should be positive only when the battery is actually charging
        battery_charging_power = 0.0
        if actual_battery_power_kw_at_terminals < 0:  # Negative power means charging
            battery_charging_power = abs(actual_battery_power_kw_at_terminals)
        reward_components['battery_charging_power'] = battery_charging_power
        
        # Solar-aware SoC management reward (replaces morning emptying)
        solar_soc_reward = 0.0
        if self.use_solar_predictions and hasattr(self, 'solar_forecast_observed') and self.current_step < len(self.solar_forecast_observed):
            solar_fc = self.solar_forecast_observed[self.current_step]
            
            # Look ahead for significant solar production in next 6-12 hours
            next_6h_solar = sum(solar_fc[:24]) if solar_fc is not None else 0.0  # Next 6 hours (24 * 15min steps)
            next_12h_solar = sum(solar_fc[:48]) if solar_fc is not None else 0.0  # Next 12 hours
            
            solar_threshold = self.config.get("morning_solar_threshold_kwh", 2.0)
            has_significant_solar_coming = next_6h_solar > solar_threshold
            
            # If significant solar is coming AND battery is quite full, reward discharge
            if has_significant_solar_coming and soc_after > 0.6:
                # Calculate how much room we need for solar charging
                max_solar_input_kwh = next_6h_solar * 0.8  # Assume 80% will be available for battery
                current_storage_kwh = soc_after * self.battery_capacity
                max_storage_kwh = soc_max_limit * self.battery_capacity
                available_space_kwh = max_storage_kwh - current_storage_kwh
                
                # If we don't have enough space for the incoming solar, reward discharge
                if available_space_kwh < max_solar_input_kwh * 0.5:  # Need at least 50% capacity for solar
                    # Reward discharging (positive battery power) when space is needed
                    if actual_battery_power_kw_at_terminals > 0:
                        energy_discharged_kwh = actual_battery_power_kw_at_terminals * self.time_step_hours
                        solar_soc_reward = energy_discharged_kwh * 1.0  # Moderate reward for making space
                        
                        if self.debug_prints and solar_soc_reward > 0.1:
                            logger.debug(f"Solar-aware discharge: +{solar_soc_reward:.2f} reward for making space " +
                                        f"(next 6h solar: {next_6h_solar:.1f} kWh, available space: {available_space_kwh:.1f} kWh)")
            
        reward_components['solar_soc_reward'] = solar_soc_reward
        
        # Night-to-peak chain bonus
        night_peak_chain_bonus = 0.0
        if self.config.get("enable_night_peak_chain", False):
            # Night charge accumulation
            if is_night and actual_battery_power_kw_at_terminals < 0:  # Charging at night
                energy_charged_kwh = abs(actual_battery_power_kw_at_terminals) * self.time_step_hours
                self.night_charge_pool += energy_charged_kwh
                self.night_charge_timestamp = current_dt
                
                if self.debug_prints and energy_charged_kwh > 0.1:  # Only log significant charging
                    logger.debug(f"Night charging: +{energy_charged_kwh:.2f} kWh, pool now: {self.night_charge_pool:.2f} kWh")
            
            # Peak discharge bonus
            if not is_night and actual_battery_power_kw_at_terminals > 0 and self.night_charge_pool > 0:
                # Check if this is a peak hour (high price or high demand)
                is_peak_hour = (current_price_ore_per_kwh > self.high_price_threshold_actual) or \
                              (grid_power_kw > self.peak_power_threshold_kw * 0.8)
                
                # Check if discharge is within the time window of night charging
                is_within_discharge_window = False
                if self.night_charge_timestamp is not None:
                    time_since_charge_hours = (current_dt - self.night_charge_timestamp).total_seconds() / 3600
                    is_within_discharge_window = time_since_charge_hours < self.night_charge_window_hours
                
                if is_peak_hour and is_within_discharge_window:
                    energy_discharged_kwh = actual_battery_power_kw_at_terminals * self.time_step_hours
                    usable_night_energy = min(energy_discharged_kwh, self.night_charge_pool)
                    
                    # Apply the chain bonus - higher than regular arbitrage
                    night_to_peak_bonus_factor = self.config.get("night_to_peak_bonus_factor", 2.0)
                    night_peak_chain_bonus = usable_night_energy * night_to_peak_bonus_factor
                    
                    # Deplete the night charge pool
                    self.night_charge_pool -= usable_night_energy
                    
                    if self.debug_prints and night_peak_chain_bonus > 0.1:  # Only log significant bonuses
                        logger.debug(f"Night-peak chain: {night_peak_chain_bonus:.2f} bonus for using " +
                                    f"{usable_night_energy:.2f} kWh of night energy at peak. " +
                                    f"Pool remaining: {self.night_charge_pool:.2f} kWh")
        
        reward_components['night_peak_chain_bonus'] = night_peak_chain_bonus

        # Add action modification penalty if safety mask was applied
        if action_modified:
            # Update consecutive invalid actions counter
            self.consecutive_invalid_actions += 1
            # Apply escalating penalty for repeated invalid actions
            escalation_factor = min(
                self.consecutive_invalid_actions,
                self.max_consecutive_penalty_multiplier
            )
            # Store as a positive value since it's a penalty magnitude
            action_mod_penalty = abs(original_action - safe_action) * self.config.get("action_modification_penalty", 10.0) * escalation_factor
            reward_components['action_mod_penalty'] = action_mod_penalty
            if self.debug_prints:
                logger.debug(f"Applied escalating penalty (x{escalation_factor}) for {self.consecutive_invalid_actions} consecutive invalid actions")
        else:
            # Reset consecutive counter when action is valid
            self.consecutive_invalid_actions = 0
            reward_components['action_mod_penalty'] = 0.0
        
        # 5.2 Aggregate all components with weights
        # Define component weights (customizable through config)
        weights = {
            'w_grid': self.config.get("w_grid", 1.0),
            'w_cap': self.config.get("w_cap", 0.5),
            'w_deg': self.config.get("w_deg", 0.5),
            'w_soc': self.config.get("w_soc", 2.0),
            'w_shape': self.config.get("w_shape", 1.0),
            'w_night': self.config.get("w_night", 1.0),
            'w_arbitrage': self.config.get("w_arbitrage", 1.0),
            'w_export': self.config.get("w_export", 0.2),
            'w_action_mod': self.config.get("w_action_mod", 3.0),
            'w_solar': self.config.get("w_solar", 1.0),
            'w_chain': self.config.get("w_chain", 1.0)
        }
        
        # Log reward components before scaling for debugging the ranges
        if self.debug_prints and self.current_step % 20 == 0:
            # Capture the raw values before any scaling factors
            logger.info(f"Step {self.current_step} RAW REWARD COMPONENTS (before w_* weighting):")
            logger.info(f"  grid_cost (raw, as penalty): {-reward_components['grid_cost']:.2f}")
            logger.info(f"  capacity_penalty (raw, as penalty): {-reward_components['capacity_penalty']:.2f}")
            logger.info(f"  degradation_cost (raw, as penalty): {-reward_components['degradation_cost']:.2f}")
            logger.info(f"  soc_reward (raw): {reward_components['soc_reward']:.2f}")
            logger.info(f"  shaping_reward (raw): {reward_components['shaping_reward']:.2f}")
            logger.info(f"  arbitrage_bonus (raw): {reward_components['arbitrage_bonus']:.2f}")
            logger.info(f"  export_bonus (raw): {reward_components['export_bonus']:.2f}")
            raw_night_incentive = reward_components['night_discount'] * reward_components['battery_charging_power']
            logger.info(f"  night_charging_incentive (raw): {raw_night_incentive:.2f}")
            logger.info(f"  action_mod_penalty (raw, as penalty): {-reward_components['action_mod_penalty']:.2f}")
        
        # Calculate total reward using weighted aggregation
        reward = total_reward(reward_components, weights)
        
        # The night_charging_reward for logging should be the raw incentive, not the weighted one.
        # The weights are applied in total_reward.
        raw_night_charging_incentive = reward_components['night_discount'] * reward_components['battery_charging_power'] * self.config.get("night_charging_scaling_factor", 1.0)
        
        # Apply overall reward scaling factor if configured
        reward_scaling_factor = self.config.get("reward_scaling_factor", 0.1)
        reward = reward * reward_scaling_factor
        
        # Log detailed reward breakdown periodically or when debugging
        if self.debug_prints or self.current_step % 100 == 0:
            # For this log, night_charging should be the weighted value that contributes to total reward
            weighted_night_charging = raw_night_charging_incentive * weights['w_night']
            logger.debug(f"Step {self.current_step} reward breakdown (after weights, before global scale):")
            logger.debug(f"  grid_cost_contrib={-reward_components['grid_cost'] * weights['w_grid']:.2f}, "
                         f"capacity_penalty_contrib={-reward_components['capacity_penalty'] * weights['w_cap']:.2f}, "
                         f"battery_cost_contrib={-reward_components['degradation_cost'] * weights['w_deg']:.2f}, "
                         f"soc_reward_contrib={reward_components['soc_reward'] * weights['w_soc']:.2f}, "
                         f"shaping_contrib={reward_components['shaping_reward'] * weights['w_shape']:.2f}, "
                         f"arbitrage_contrib={reward_components['arbitrage_bonus'] * weights['w_arbitrage']:.2f}, "
                         f"export_contrib={reward_components['export_bonus'] * weights['w_export']:.2f}, "
                         f"night_charging_contrib={weighted_night_charging:.2f}, "
                         f"action_mod_penalty_contrib={-reward_components['action_mod_penalty'] * weights['w_action_mod']:.2f}")
            logger.debug(f"  TOTAL (before global scale) = {reward / reward_scaling_factor:.2f}, TOTAL (after global scale) = {reward:.2f}")
        
        # Create a formatted reward_components dict for info tracking
        # This is to maintain compatibility with existing evaluation code
        formatted_reward_components = {
            'reward_grid_cost': -reward_components['grid_cost'],
            'reward_capacity_penalty': -reward_components['capacity_penalty'],
            'reward_battery_cost': -reward_components['degradation_cost'],
            'reward_soc_reward': reward_components['soc_reward'],
            'reward_shaping': reward_components['shaping_reward'],
            'reward_arbitrage_bonus': reward_components['arbitrage_bonus'],
            'reward_export_bonus': reward_components['export_bonus'],
            'reward_night_charging': raw_night_charging_incentive,  # Storing the raw incentive here
            'reward_action_mod_penalty': -reward_components['action_mod_penalty'],
            'reward_solar_soc': reward_components['solar_soc_reward'],
            'reward_night_peak_chain': reward_components['night_peak_chain_bonus'],
            'battery_charging_power': reward_components['battery_charging_power'],
            'night_charge_pool': self.night_charge_pool
        }

        # 6. Update State & History
        self.current_step += 1
        self.price_history.append(current_price_ore_per_kwh)
        self.last_action = original_action  # Store the original action for rendering
        self.total_reward += reward

        # 7. Termination/Truncation
        terminated = False
        if self.current_step >= self.simulation_steps:
            terminated = True
        truncated = False  # Not using truncation for now

        # 8. Observation & Info
        observation = self._get_observation()
        info = self._get_info()
        
        # Add additional info fields
        info["current_price"] = current_price_ore_per_kwh
        info["power_kw"] = actual_battery_power_kw_at_terminals
        info["grid_power_kw"] = grid_power_kw
        info["base_demand_kw"] = current_consumption_kw
        info["current_solar_production_kw"] = current_solar_kw
        info["action_modified"] = action_modified
        info["original_action"] = original_action
        info["safe_action"] = safe_action
        info["export_bonus"] = export_bonus
        info["is_night_discount"] = is_night
        info["reward_components"] = formatted_reward_components
        info["total_reward_episode"] = self.total_reward
        info["soc"] = self.battery.soc  # Add current battery SoC to info dictionary
        
        # For compatibility with evaluate_agent.py
        info.update(formatted_reward_components)

        return observation, reward, terminated, truncated, info

    def _initialize_forecasts(self) -> None:
        num_steps = self.simulation_steps
        
        if self.use_price_predictions:
            if self.price_predictions_df is not None:
                self._initialize_price_forecasts_from_predictions()
            else:
                logger.critical(
                    "CRITICAL: `use_price_predictions` is True, but no price prediction data "
                    "loaded (price_predictions_df is None). Cannot generate factual day-ahead "
                    "forecasts. Defaulting to a zero price forecast for actuals and observed. "
                    "This will likely lead to poor agent performance. EXITING INSTEAD OF DEFAULTING."
                )
                sys.exit(1)
        else: # This case implies configuration error if historical data is solely expected
            logger.critical(
                "CRITICAL: `use_price_predictions` is False. This environment requires historical "
                "price data. Ensure `use_price_predictions` is True and data is available. EXITING."
            )
            sys.exit(1)
            
    def _initialize_price_forecasts_from_predictions(self) -> None:
        if self.price_predictions_df is None:
            logger.critical(
                "CRITICAL: _initialize_price_forecasts_from_predictions called but "
                "price_predictions_df is None. This indicates an issue in data loading or configuration. EXITING."
            )
            sys.exit(1)
            return # Should not be reached due to sys.exit()

        num_steps = self.simulation_steps
        sim_timestamps = pd.to_datetime([self.start_datetime + datetime.timedelta(hours=i * self.time_step_hours) for i in range(num_steps)])
        
        if not isinstance(self.price_predictions_df.index, pd.DatetimeIndex):
            self.price_predictions_df.index = pd.to_datetime(self.price_predictions_df.index)

        self.price_predictions_df = self.price_predictions_df.sort_index()
        
        # Important change: use 'ffill' method instead of 'interpolate' to properly handle hourly data
        # This will forward-fill hourly price data instead of linearly interpolating between hours
        resampled_prices = self.price_predictions_df['price'].resample(f'{int(self.time_step_hours*60)}min').ffill()
        
        aligned_prices = resampled_prices.reindex(sim_timestamps, method='nearest').ffill().bfill()
        aligned_prices_np = aligned_prices.values

        if len(aligned_prices_np) >= num_steps:
            self.price_forecast_actual = aligned_prices_np[:num_steps]
        else:
            self.price_forecast_actual = np.zeros(num_steps)
            if len(aligned_prices_np) > 0:
                logger.critical(
                    f"CRITICAL: Not enough price prediction data for full simulation "
                    f"({num_steps} steps required, {len(aligned_prices_np)} available) "
                    f"when `use_price_predictions` is True. Using available data and padding with the last known value. "
                    f"EXITING AS THIS IS NOT IDEAL."
                )
                sys.exit(1)
            else:
                logger.critical(
                    "CRITICAL: No price prediction data available after alignment when `use_price_predictions` is True. "
                    "Cannot proceed without price data. EXITING."
                )
                sys.exit(1)
        
        # Apply Swedish night-time discount (50% reduction from 22:00 to 06:00)
        self._mark_night_hours(sim_timestamps)
        
        self.price_forecast_observed = np.zeros((num_steps, 24))
        steps_per_hour = int(1 / self.time_step_hours)
        for i in range(num_steps):
            for j in range(24):
                hour_start_idx = i + (j * steps_per_hour)
                if hour_start_idx < num_steps:
                    self.price_forecast_observed[i, j] = self.price_forecast_actual[hour_start_idx]
                else:
                    self.price_forecast_observed[i, j] = self.price_forecast_actual[num_steps - 1]
        
        # Process price averages using efficient pandas operations
        vat_mult = self.config.get("vat_mult", 1.25)
        
        # Check if price_24h_avg and price_168h_avg columns exist
        if 'price_24h_avg' in self.price_predictions_df.columns and 'price_168h_avg' in self.price_predictions_df.columns:
            logger.info("Found price_24h_avg and price_168h_avg columns in price predictions. Using these values.")
            
            # Build a DataFrame of simulation times 
            sim_df = pd.DataFrame(index=sim_timestamps)
            
            # Merge in the price averages via nearest join
            merged = sim_df.merge(
                self.price_predictions_df[['price_24h_avg','price_168h_avg']],
                left_index=True, right_index=True,
                how='left', 
                sort=True
            ).interpolate(method='time').ffill().bfill()
            
            # Apply VAT in one go
            merged[['price_24h_avg','price_168h_avg']] *= vat_mult
            
            # Then pull into numpy arrays
            self.price_24h_avg = merged['price_24h_avg'].to_numpy()
            self.price_168h_avg = merged['price_168h_avg'].to_numpy()
        else:
            logger.warning("price_24h_avg and/or price_168h_avg columns not found in price predictions. Calculating on the fly.")
            
            # Calculate 24h rolling mean
            rolling_24h = pd.Series(self.price_forecast_actual, index=sim_timestamps).rolling('24H', min_periods=1).mean()
            
            # Calculate 168h (7-day) rolling mean
            rolling_168h = pd.Series(self.price_forecast_actual, index=sim_timestamps).rolling('168H', min_periods=1).mean()
            
            # Apply VAT and convert to numpy arrays
            self.price_24h_avg = (rolling_24h.values * vat_mult)
            self.price_168h_avg = (rolling_168h.values * vat_mult)
    
    def _mark_night_hours(self, timestamps):
        """Mark night hours (22:00 to 06:00) for capacity fee discount"""
        # This method only marks night hours for capacity fee discount tracking
        logger.info(f"Setting up night-time capacity discount flags (22:00-06:00)")
        
        # Initialize night discount tracking array
        self.is_night_discount = np.zeros(len(timestamps), dtype=bool)
        
        # Log timestamps at day boundaries to debug night discount issues
        boundary_timestamps = []
        
        for i, timestamp in enumerate(timestamps):
            hour = timestamp.hour
            minute = timestamp.minute
            
            # Check if time is between 22:00 and 06:00 (exclusive)
            is_night = (hour >= 22 or hour < 6)
            
            # Check boundary conditions specifically
            if hour == 5 and minute >= 45:
                # logger.info(f"Boundary case: {timestamp} (Hour {hour}:{minute}) marked as night={is_night}")
                boundary_timestamps.append((i, timestamp, is_night))
            elif hour == 6 and minute < 15:
                # logger.info(f"Boundary case: {timestamp} (Hour {hour}:{minute}) marked as night={is_night}")
                boundary_timestamps.append((i, timestamp, is_night))
            elif hour == 21 and minute >= 45:
                # logger.info(f"Boundary case: {timestamp} (Hour {hour}:{minute}) marked as night={is_night}")
                boundary_timestamps.append((i, timestamp, is_night))
            elif hour == 22 and minute < 15:
                # logger.info(f"Boundary case: {timestamp} (Hour {hour}:{minute}) marked as night={is_night}")
                boundary_timestamps.append((i, timestamp, is_night))
                
            self.is_night_discount[i] = is_night
        
        # Log a summary of all boundary cases
        if boundary_timestamps:
            logger.info(f"Found {len(boundary_timestamps)} boundary timestamps:")
            for idx, ts, night in boundary_timestamps:
                everyfour = (idx % 4 == 0) # True if index is divisible by 4 (i.e. every 4th)
                if everyfour:# reduces clutter
                    logger.info(f"  Index {idx}: {ts} - Night discount: {night}")
        else:
            logger.info("No boundary timestamps found in simulation period")
    
    def _initialize_solar_forecasts(self) -> None:
        """Initializes actual solar production for each simulation step and the 4-day hourly forecast."""
        num_steps = self.simulation_steps
        sim_timestamps = pd.to_datetime([self.start_datetime + datetime.timedelta(hours=i * self.time_step_hours) for i in range(num_steps)])

        if not self.use_solar_predictions or self.all_solar_data_hourly_kw is None or self.all_solar_data_hourly_kw.empty:
            logger.warning("Solar predictions disabled or no solar data loaded. Solar production will be zero.")
            self.solar_forecast_actual = np.zeros(num_steps)
            self.solar_forecast_observed = np.zeros((num_steps, 3 * 24)) # 3 days * 24 hours
            return

        # 1. Initialize solar_forecast_actual (solar production at simulation step frequency)
        # Reindex the hourly solar data to simulation timestamps, forward-filling
        aligned_solar_kw_actual_df = self.all_solar_data_hourly_kw['solar_production_kw'].reindex(
            sim_timestamps, method='ffill'
        )
        # Backward fill any remaining NaNs at the beginning
        aligned_solar_kw_actual_df = aligned_solar_kw_actual_df.bfill()
        
        # If there are still NaNs (e.g., simulation range is outside data range entirely), fill with 0
        aligned_solar_kw_actual_df = aligned_solar_kw_actual_df.fillna(0)
        
        self.solar_forecast_actual = aligned_solar_kw_actual_df.values
        if len(self.solar_forecast_actual) < num_steps: # Should not happen if fillna(0) is used
             logger.error(f"Solar actual forecast length ({len(self.solar_forecast_actual)}) is less than num_steps ({num_steps}). Padding with zeros.")
             self.solar_forecast_actual = np.pad(self.solar_forecast_actual, (0, num_steps - len(self.solar_forecast_actual)), 'constant')


        # 2. Initialize solar_forecast_observed (3-day hourly forecast for the agent)
        # Each row `i` in solar_forecast_observed contains 72 hourly values for the 3 days
        # starting from the hour of sim_timestamps[i].
        self.solar_forecast_observed = np.zeros((num_steps, 3 * 24)) # 3 days * 24 hours
        
        for i in range(num_steps):
            forecast_start_time = sim_timestamps[i]
            # Create 72 hourly timestamps for the forecast
            hourly_forecast_timestamps = pd.date_range(start=forecast_start_time, periods=(3 * 24), freq='h') # 3 days * 24 hours
            
            # Reindex the original hourly solar data to these forecast timestamps
            observed_forecast_series = self.all_solar_data_hourly_kw['solar_production_kw'].reindex(
                hourly_forecast_timestamps, method='ffill' # Use ffill to carry last known value if forecast extends beyond data
            )
            # Fill any remaining NaNs (e.g., if forecast starts before data or extends far beyond) with 0
            observed_forecast_series = observed_forecast_series.fillna(0)
            
            self.solar_forecast_observed[i, :] = observed_forecast_series.values
        
        # Apply data augmentation if enabled
        if self.config.get("use_data_augmentation", False) and self.config.get("augment_solar_data", True):
            augmentation_factor = self.config.get("solar_augmentation_factor", 0.2)
            
            # Generate random scaling factors for the entire episode
            # Use a normal distribution centered at 1.0 with a standard deviation based on the augmentation factor
            solar_scale = np.random.normal(1.0, augmentation_factor)
            solar_scale = max(0.5, min(1.5, solar_scale))  # Limit scaling to reasonable range
            
            # Apply augmentation to both actual and observed solar forecasts
            self.solar_forecast_actual = self.solar_forecast_actual * solar_scale
            self.solar_forecast_observed = self.solar_forecast_observed * solar_scale
            
            logger.info(f"Applied solar data augmentation with scaling factor: {solar_scale:.2f}")

    def _initialize_load_forecasts(self) -> None:
        """Initializes the 4-day hourly load forecast for the agent."""
        num_steps = self.simulation_steps
        sim_timestamps = pd.to_datetime([self.start_datetime + datetime.timedelta(hours=i * self.time_step_hours) for i in range(num_steps)])

        if not self.use_variable_consumption or self.all_consumption_data_kw is None or self.all_consumption_data_kw.empty:
            logger.warning("Variable consumption disabled or no consumption data loaded. Load forecast will be fixed baseload or zeros.")
            # self.episode_consumption_kw is already set to fixed_baseload_kw in reset() if not use_variable_consumption
            # So, load_forecast_observed should reflect this.
            self.load_forecast_observed = np.full((num_steps, 3 * 24), self.fixed_baseload_kw, dtype=np.float32)
            return

        # self.episode_consumption_kw already contains the actual load for each simulation step (potentially augmented)
        # We need to generate the 4-day *hourly* forecast for the agent observation
        
        self.load_forecast_observed = np.zeros((num_steps, 3 * 24)) # 3 days * 24 hours
        
        # The source for the forecast is self.all_consumption_data_kw, which is at self.time_step_minutes frequency.
        # For an hourly forecast, we need to ensure we're sampling/aggregating this correctly to hourly points.
        # Let's re-sample all_consumption_data_kw to hourly if it's not already.
        # Note: _load_consumption_data already resamples to time_step_minutes.
        # If time_step_minutes is less than 60, we should average up to hourly for the forecast.
        # If time_step_minutes is 60, it's already hourly.
        
        hourly_consumption_data = self.all_consumption_data_kw['consumption_kw'].resample('h').mean()


        for i in range(num_steps):
            forecast_start_time = sim_timestamps[i] # This is the start time of the current simulation step
            # Create 96 hourly timestamps for the forecast starting from forecast_start_time
            hourly_forecast_timestamps = pd.date_range(start=forecast_start_time, periods=(3 * 24), freq='h')
            
            # Reindex the hourly consumption data to these forecast timestamps
            # Use ffill to carry last known value if forecast extends beyond data
            observed_forecast_series = hourly_consumption_data.reindex(
                hourly_forecast_timestamps, method='ffill' 
            )
            # Fill any remaining NaNs (e.g., if forecast starts before data or extends far beyond)
            # Try bfill first for leading NaNs, then fill remaining with a reasonable default (e.g., fixed_baseload_kw or mean)
            observed_forecast_series = observed_forecast_series.bfill().fillna(self.fixed_baseload_kw) # Fallback to fixed baseload
            
            self.load_forecast_observed[i, :] = observed_forecast_series.values

        # Apply data augmentation to the load forecast if configured
        # This augmentation should mirror what's done to self.episode_consumption_kw in reset()
        # For simplicity, we assume if augmentation was applied to episode_consumption_kw,
        # the forecast should reflect a similarly scaled version.
        # However, self.episode_consumption_kw might already be augmented.
        # The cleanest way is to augment self.all_consumption_data_kw *before* creating episode_consumption_kw
        # and *before* creating load_forecast_observed.
        # Let's assume for now that if data augmentation for consumption is on, it's handled
        # when all_consumption_data_kw is initially processed or when episode_consumption_kw is derived.
        # If self.config.get("use_data_augmentation", False) and self.config.get("augment_consumption_data", True):
        #     augmentation_factor = self.config.get("consumption_augmentation_factor", 0.15)
        #     # This needs to be consistent with the augmentation applied to self.episode_consumption_kw
        #     # For now, we are forecasting the *actual* loaded (and potentially pre-augmented) consumption data.
        #     # If self.episode_consumption_kw was scaled, this forecast might not match its scale unless
        #     # the scaling factor is applied here too, or the source (hourly_consumption_data) was scaled.
        #     logger.info(f"Load forecast based on (potentially augmented) consumption data. Ensure consistency if augment_consumption_data is True.")

    def _get_observation(self) -> Dict:
        current_dt = self.start_datetime + datetime.timedelta(hours=self.current_step * self.time_step_hours)
        hour_of_day = current_dt.hour
        minute_of_hour = current_dt.minute / 60.0 # Normalized
        day_of_week = current_dt.weekday()
        
        time_idx = np.array([hour_of_day, minute_of_hour, day_of_week], dtype=np.float32)
        
        current_step_idx = min(self.current_step, self.price_forecast_observed.shape[0] - 1)
        price_fc = self.price_forecast_observed[current_step_idx, :]

        solar_fc = np.zeros(3 * 24) # Default to zeros, CHANGED to 72 hours
        if self.use_solar_predictions and self.solar_forecast_observed is not None:
            current_solar_step_idx = min(self.current_step, self.solar_forecast_observed.shape[0] - 1)
            if current_solar_step_idx < self.solar_forecast_observed.shape[0]:
                 solar_fc = self.solar_forecast_observed[current_solar_step_idx, :]
            else:
                 logger.warning(f"Current step {self.current_step} is out of bounds for solar_forecast_observed shape {self.solar_forecast_observed.shape}. Using zeros.")

        load_fc = np.zeros(3 * 24) # Default to zeros
        if self.use_variable_consumption and hasattr(self, 'load_forecast_observed') and self.load_forecast_observed is not None:
            current_load_step_idx = min(self.current_step, self.load_forecast_observed.shape[0] - 1)
            if current_load_step_idx < self.load_forecast_observed.shape[0]:
                load_fc = self.load_forecast_observed[current_load_step_idx, :]
            else:
                logger.warning(f"Current step {self.current_step} is out of bounds for load_forecast_observed shape {self.load_forecast_observed.shape}. Using zeros.")
        elif not self.use_variable_consumption: # If not using variable consumption, fill with fixed baseload
            load_fc = np.full(3 * 24, self.fixed_baseload_kw, dtype=np.float32)

        # Use the pre-calculated night discount flag instead of recalculating
        is_night = float(self.is_night_discount[self.current_step]) if self.current_step < len(self.is_night_discount) else float(hour_of_day >= 22 or hour_of_day < 6)
        
        # Calculate month progress
        days_elapsed = (current_dt - self.month_start_date).days
        hours_elapsed = (current_dt - self.month_start_date).seconds / 3600
        total_elapsed = days_elapsed + (hours_elapsed / 24)
        month_progress = min(1.0, total_elapsed / self.days_in_month)
        
        # Get price averages if they exist, otherwise calculate them
        price_24h_avg = 0.0
        price_168h_avg = 0.0
        
        if hasattr(self, 'price_24h_avg') and hasattr(self, 'price_168h_avg'):
            # Use pre-calculated averages
            price_24h_avg = self.price_24h_avg[current_step_idx] if current_step_idx < len(self.price_24h_avg) else 0.0
            price_168h_avg = self.price_168h_avg[current_step_idx] if current_step_idx < len(self.price_168h_avg) else 0.0
        elif self.price_forecast_actual is not None and len(self.price_forecast_actual) > 0:
            # Calculate on the fly if needed
            start_idx = max(0, current_step_idx - 24)
            end_idx = current_step_idx + 1
            if end_idx <= len(self.price_forecast_actual):
                price_24h_avg = np.mean(self.price_forecast_actual[start_idx:end_idx])
            
            start_idx = max(0, current_step_idx - 168)
            if end_idx <= len(self.price_forecast_actual):
                price_168h_avg = np.mean(self.price_forecast_actual[start_idx:end_idx])

        return {
            "soc": np.array([self.battery.soc], dtype=np.float32),
            "time_idx": time_idx,
            "price_forecast": price_fc.astype(np.float32),
            "solar_forecast": solar_fc.astype(np.float32),
            "capacity_metrics": np.array([
                self.top3_peaks[0],
                self.top3_peaks[1], 
                self.top3_peaks[2],
                self.peak_rolling_average,
                month_progress
            ], dtype=np.float32),
            "price_averages": np.array([
                price_24h_avg,
                price_168h_avg
            ], dtype=np.float32),
            "is_night_discount": np.array([is_night], dtype=np.float32),
            "load_forecast": load_fc.astype(np.float32)
        }
    
    def _get_info(self) -> Dict:
        # Calculate current capacity fee
        current_capacity_fee = self.peak_rolling_average * self.config.get("capacity_fee_sek_per_kw", 81.25)
        
        # Calculate month progress for monitoring
        current_dt = self.start_datetime + datetime.timedelta(hours=self.current_step * self.time_step_hours)
        days_elapsed = (current_dt - self.month_start_date).days
        hours_elapsed = (current_dt - self.month_start_date).seconds / 3600
        total_elapsed = days_elapsed + (hours_elapsed / 24)
        month_progress = min(1.0, total_elapsed / self.days_in_month)
        
        # Get current price averages
        current_step_idx = min(self.current_step, len(self.price_forecast_actual) - 1) if len(self.price_forecast_actual) > 0 else 0
        price_24h_avg = self.price_24h_avg[current_step_idx] if hasattr(self, 'price_24h_avg') and current_step_idx < len(self.price_24h_avg) else 0.0
        price_168h_avg = self.price_168h_avg[current_step_idx] if hasattr(self, 'price_168h_avg') and current_step_idx < len(self.price_168h_avg) else 0.0
        
        return {
            "current_step": self.current_step,
            "total_cost": self.total_cost,
            "peak_power": self.peak_power,
            "total_reward_episode": self.total_reward,
            "top3_peaks": self.top3_peaks,
            "peak_rolling_average": self.peak_rolling_average,
            "current_capacity_fee": current_capacity_fee,
            "month_progress": month_progress,
            "price_24h_avg": price_24h_avg,
            "price_168h_avg": price_168h_avg
        }
    
    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "human":
            obs = self._get_observation()
            price_fc_snippet = obs['price_forecast'][:3]
            action_display = f"{self.last_action:.2f}" if self.last_action is not None else "N/A"
            current_price_display = f"{self.price_history[-1]:.2f}" if self.price_history else "N/A"
            
            # Simplified grid_power_display logic as info dict is more direct now
            # last_grid_power = self.info.get("grid_power_kw", "N/A") # info is not attribute
            # For render, it's better to show current step's info if possible, but it's computed after call.
            # We will rely on history or direct calculation for render if needed.

            print(f"--- Step: {self.current_step}/{self.simulation_steps} ---")
            print(f"  SoC: {obs['soc'][0]:.2f}, Time: H{int(obs['time_idx'][0])} M{int(obs['time_idx'][1]*60)} D{int(obs['time_idx'][2])}")
            print(f"  Price Forecast (next 3h): {price_fc_snippet}")
            print(f"  Action Taken: {action_display}")
            print(f"  Current Price: {current_price_display}")
            # To get current step's battery_power_kw and grid_power_kw for render, they'd need to be stored from step()
            # Or recalculate based on last_action, which might be complex.
            # Simplest is to rely on history or just not show live power values in this simple render.
            # However, we can get them from the info dict if step() was just called.
            # For a standalone render call, this might not be available.
            print(f"  Total Cost: {self.total_cost:.2f}")
            print(f"  Total Reward: {self.total_reward:.2f}")
            print("------------------------------------")
        return None # RGB array rendering not implemented
    
    def close(self) -> None:
        pass
    
    def _load_price_predictions(self) -> None:
        if not self.price_predictions_path:
            logger.warning("Price predictions path not set. Historical price data cannot be loaded.")
            self.price_predictions_df = None
            return

        file_path_to_load = self.price_predictions_path
        if not os.path.isabs(file_path_to_load):
            project_root = Path(__file__).resolve().parent.parent.parent 
            file_path_to_load = project_root / file_path_to_load
        else:
            file_path_to_load = Path(file_path_to_load)

        # Enhanced file existence checking with detailed logging
        logger.info(f"Attempting to load price data from: {file_path_to_load}")

        try:
            # Try to load the file with various possible formats
            try:
                self.price_predictions_df = pd.read_csv(
                    file_path_to_load, 
                    index_col='HourSE', 
                    parse_dates=True
                )
            except Exception as e:
                logger.error(f"Error loading price predictions from {file_path_to_load}: {e}")
                sys.exit(1)

            # Print timestamp sanity check if debug_prints is enabled
            if self.debug_prints and self.price_predictions_df is not None:
                logger.info("---PRICE DATA BEFORE PROCESSING---")
                logger.info(self.price_predictions_df.iloc[:, 0].tail(1)) # Show only the last timestamp, not all columns
                logger.info("-----------------------")

            # If the dataframe was successfully loaded, process it
            if self.price_predictions_df is not None and not self.price_predictions_df.empty:
                self.price_predictions_df['price'] = self.price_predictions_df['SE3_price_ore']

                # 1. Ensure we have a DatetimeIndex (should be true from parse_dates, but double-check)
                if not isinstance(self.price_predictions_df.index, pd.DatetimeIndex):
                    self.price_predictions_df.index = pd.to_datetime(self.price_predictions_df.index)
                
                # 2. Handle the problematic DST date/hour before attaching timezone
                if self.price_predictions_df.index.tzinfo is None:
                    try:
                        # Use ambiguous='NaT' to mark ambiguous times as Not-a-Time rather than failing
                        self.price_predictions_df.index = self.price_predictions_df.index.tz_localize('Europe/Stockholm', ambiguous='NaT', nonexistent='shift_forward')
                        # Check if we have any NaT values after localization
                        nat_mask = pd.isna(self.price_predictions_df.index)
                        if nat_mask.any():
                            nat_count = nat_mask.sum()
                            logger.warning(f"Found {nat_count} ambiguous timestamps during DST transitions that were set to NaT. Removing these rows.")
                            # Remove rows with NaT timestamps
                            self.price_predictions_df = self.price_predictions_df[~nat_mask]
                            
                        # Verify we have valid data after filtering
                        if self.price_predictions_df.empty:
                            logger.critical("All timestamps were ambiguous! This is unexpected.")
                            self.price_predictions_df = None
                    except Exception as e:
                        logger.error(f"Error localizing timestamps: {e}")
                        # Try an alternative approach - use 'infer' to let pandas determine the correct offset
                        try:
                            logger.info("Trying alternative localization with ambiguous='infer'")
                            self.price_predictions_df.index = self.price_predictions_df.index.tz_localize('Europe/Stockholm', ambiguous='infer')
                        except Exception as e2:
                            logger.error(f"Alternative localization also failed: {e2}")
                            self.price_predictions_df = None
                else:
                    logger.warning(f"Price timestamps already had timezone info ({self.price_predictions_df.index.tzinfo}). Not expected from CSV data.")
                
                if self.price_predictions_df is not None and not self.price_predictions_df.empty:
                    logger.info(f"Successfully loaded {len(self.price_predictions_df)} price data points. Path: {file_path_to_load}")
                    
                    # Print timestamp sanity check after processing if debug_prints is enabled
                    if self.debug_prints:
                        logger.info("---PRICE DATA AFTER PROCESSING---")
                        logger.info(self.price_predictions_df.iloc[:, 0].tail(1)) # Show only the last timestamp
                        logger.info("-----------------------")
            else:
                logger.error(f"Failed to load price predictions data from {file_path_to_load}")
                self.price_predictions_df = None
        except Exception as e:
            logger.error(f"Error loading price predictions from {file_path_to_load}: {e}. Historical data will be unavailable.")
            self.price_predictions_df = None

    def _load_solar_data(self) -> None:
        """Loads solar production data from the specified CSV file."""
        if not self.solar_data_path:
            logger.error("`use_solar_predictions` is True, but `solar_data_path` is not set.")
            self.all_solar_data_hourly_kw = None
            return

        file_path_to_load = self.solar_data_path
        if not os.path.isabs(file_path_to_load):
            project_root = Path(__file__).resolve().parent.parent.parent 
            file_path_to_load = project_root / file_path_to_load
        else:
            file_path_to_load = Path(file_path_to_load)

        logger.info(f"Attempting to load solar production data from: {file_path_to_load}")
        if not file_path_to_load.exists() or not file_path_to_load.is_file():
            logger.error(f"Solar data file does not exist or is not a file: {file_path_to_load}")
            self.all_solar_data_hourly_kw = None
            return

        try:
            df = pd.read_csv(file_path_to_load, parse_dates=['Timestamp'])
            if 'Timestamp' not in df.columns or 'solar_production_kwh' not in df.columns:
                logger.error(f"Solar data CSV must contain 'Timestamp' and 'solar_production_kwh' columns. Found: {df.columns.tolist()}")
                self.all_solar_data_hourly_kw = None
                return

            df.set_index('Timestamp', inplace=True)
            
            # Print timestamp sanity check if debug_prints is enabled
            if self.debug_prints:
                logger.info("---SOLAR DATA BEFORE PROCESSING---")
                logger.info(df.iloc[:, 0].tail(1))
                logger.info("-----------------------")
                
            # Timestamps are UTC (due to 'Z' in example "2023-03-13T10:00:00.000Z")
            # If index is not timezone-aware, make it UTC. If it is, ensure it's UTC.
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
            
            # Convert to 'Europe/Stockholm'
            df.index = df.index.tz_convert('Europe/Stockholm')
            
            # 'solar_production_kwh' for an hour is numerically equal to average kW for that hour
            df.rename(columns={'solar_production_kwh': 'solar_production_kw'}, inplace=True)
            
            # Ensure no negative solar production (can happen with bad data)
            df['solar_production_kw'] = df['solar_production_kw'].clip(lower=0)

            self.all_solar_data_hourly_kw = df[['solar_production_kw']]
            
            # Sort index to be sure
            self.all_solar_data_hourly_kw.sort_index(inplace=True)

            # Remove duplicate indices, keeping the first entry
            if self.all_solar_data_hourly_kw.index.duplicated().any():
                logger.warning("Duplicate timestamps found in solar data. Keeping first entry.")
                self.all_solar_data_hourly_kw = self.all_solar_data_hourly_kw[~self.all_solar_data_hourly_kw.index.duplicated(keep='first')]

            logger.info(f"Successfully loaded {self.all_solar_data_hourly_kw.shape[0]} solar data points. Path: {file_path_to_load}")

            # Print timestamp sanity check after processing if debug_prints is enabled
            if self.debug_prints:
                logger.info("---SOLAR DATA AFTER PROCESSING---")
                logger.info(self.all_solar_data_hourly_kw.iloc[:,0].tail(1))
                logger.info("-----------------------")

        except Exception as e:
            logger.error(f"Error loading or processing solar data from {file_path_to_load}: {e}")
            self.all_solar_data_hourly_kw = None

    def _load_consumption_data(self) -> None:
        if not self.consumption_data_path:
            logger.error("`use_variable_consumption` is True, but `consumption_data_path` is not set.")
            self.all_consumption_data_kw = None
            return

        file_path_to_load = self.consumption_data_path
        if not os.path.isabs(file_path_to_load):
            project_root = Path(__file__).resolve().parent.parent.parent 
            file_path_to_load = project_root / file_path_to_load
        else:
            file_path_to_load = Path(file_path_to_load)

        if file_path_to_load.exists() and file_path_to_load.is_file():
            logger.info(f"Attempting to load consumption data from: {file_path_to_load}") # Added log
            try:
                df = pd.read_csv(file_path_to_load, parse_dates=['timestamp'], index_col='timestamp')
                
                # Print timestamp sanity check if debug_prints is enabled
                if self.debug_prints:
                    logger.info("---CONSUMPTION DATA BEFORE PROCESSING---")
                    logger.info(df.iloc[:,0].tail(1))
                    logger.info("-----------------------")
                    
                # Drop extra columns
                df.drop(columns=['consumption_cost', 'consumption_unit_price', 
                                    'consumption_unit_price_vat', 'consumption_unit', 
                                    'production', 'production_profit', 'production_unit_price', 
                                    'production_unit'], 
                                    inplace=True)

                # Force conversion to DatetimeIndex if needed
                if not isinstance(df.index, pd.DatetimeIndex):
                    logger.warning("Consumption data index is not a DatetimeIndex after loading. Converting explicitly.")
                    # Use pd.to_datetime with explicit handling for timezones
                    df.index = pd.to_datetime(df.index, utc=False)
                
                # Now check if the conversion worked and handle timezone
                if not isinstance(df.index, pd.DatetimeIndex):
                    logger.error("Consumption data index could not be converted to DatetimeIndex. This is unexpected.")
                    self.all_consumption_data_kw = None
                    return # Cannot proceed
                
                # Handle timezone
                if df.index.tz is None:
                    logger.warning("Consumption data index is timezone-naive after loading. This is unexpected if CSV has timezone offsets.")
                    # Try to preserve the original timezone information if possible
                    # Use a fallback approach - assume the timestamps were intended as 'Europe/Stockholm'
                    try:
                        # Use 'NaT' for ambiguous times and handle them
                        df.index = df.index.tz_localize('Europe/Stockholm', ambiguous='NaT', nonexistent='shift_forward')
                        
                        # Check if we have any NaT values after localization
                        nat_mask = pd.isna(df.index)
                        if nat_mask.any():
                            nat_count = nat_mask.sum()
                            logger.warning(f"Found {nat_count} ambiguous timestamps in consumption data. Removing these rows.")
                            # Remove rows with NaT timestamps
                            df = df[~nat_mask]
                    except Exception as e:
                        logger.error(f"Error localizing consumption timestamps with 'NaT': {e}")
                        # Try alternative approach 
                        try:
                            logger.info("Trying alternative consumption data localization with ambiguous='infer'")
                            df.index = df.index.tz_localize('Europe/Stockholm', ambiguous='infer', nonexistent='shift_forward')
                        except Exception as e2:
                            logger.error(f"Alternative consumption localization also failed: {e2}")
                            self.all_consumption_data_kw = None
                            return
                elif str(df.index.tz) != 'Europe/Stockholm':
                    logger.info(f"Converting consumption data index from {df.index.tz} to 'Europe/Stockholm'.")
                    # Convert from whatever timezone to Stockholm time
                    try:
                        df.index = df.index.tz_convert('Europe/Stockholm')
                        logger.info(f"Consumption data index after tz_convert to Europe/Stockholm: {df.index.min()} to {df.index.max()} (tz: {df.index.tzinfo})")
                    except Exception as e:
                        logger.error(f"Error converting consumption timestamps to Stockholm time: {e}")
                        self.all_consumption_data_kw = None
                
                # Handle duplicates after getting timezone right
                if df.index.duplicated().any():
                    logger.info("Duplicated indices found in consumption data after timezone conversion. Dropping duplicates, keeping first instance.")
                    df = df[~df.index.duplicated(keep='first')] 
                
                logger.info(f"Final consumption data index range (Europe/Stockholm, duplicates dropped): {df.index.min()} to {df.index.max()}")
                
                # Print timestamp sanity check after processing if debug_prints is enabled
                if self.debug_prints:
                    logger.info("---CONSUMPTION DATA AFTER PROCESSING---")
                    logger.info(df.iloc[:,0].tail(1))
                    logger.info("-----------------------")    
                    
                # Consumption is in kWh per hour. This value is directly average kW for that hour.
                # Resample this to our time_step_minutes.
                resample_freq = f"{int(self.time_step_hours * 60)}min"
                
                if 'consumption' not in df.columns:
                    logger.error(f"'consumption' column not found in {file_path_to_load}. Available columns: {df.columns.tolist()}")
                    self.all_consumption_data_kw = None
                    return
                
                # Rename column and resample
                self.all_consumption_data_kw = df[['consumption']].rename(columns={'consumption': 'consumption_kw'})
                self.all_consumption_data_kw = self.all_consumption_data_kw.resample(resample_freq).ffill().bfill()

                logger.info(f"Successfully loaded and resampled {len(self.all_consumption_data_kw)} consumption data points. Path: {file_path_to_load}")
                if self.all_consumption_data_kw.isnull().any().any():
                    logger.warning(f"NaN values found in resampled consumption data. Check resampling or source data. Filling with 0 for safety.")
                    self.all_consumption_data_kw = self.all_consumption_data_kw.fillna(0)

            except Exception as e:
                logger.error(f"Error loading or processing consumption data from {file_path_to_load}: {e}")
                self.all_consumption_data_kw = None
        else:
            logger.warning(f"Consumption data file {file_path_to_load} not found or is not a file.")
            self.all_consumption_data_kw = None

    def _update_capacity_peaks_with_same_day_constraint(self, current_dt: datetime.datetime, effective_grid_power: float) -> None:
        """
        Update top 3 capacity peaks enforcing the Swedish regulation:
        Only the 3 highest hourly IMPORT peaks per month that do NOT lie on the same day.
        
        Args:
            current_dt: Current timestamp
            effective_grid_power: Grid power adjusted for night discount
        """
        # Check if same-day constraint is enabled (default: True for Swedish regulation compliance)
        enforce_same_day_constraint = self.config.get("enforce_capacity_same_day_constraint", True)
        
        # Record this peak with its timestamp
        self.current_month_peak_data.append((current_dt, effective_grid_power))
        
        if enforce_same_day_constraint:
            # Group all peaks by date to enforce same-day constraint
            peaks_by_date = {}
            for timestamp, power in self.current_month_peak_data:
                date_key = timestamp.date()
                if date_key not in peaks_by_date:
                    peaks_by_date[date_key] = []
                peaks_by_date[date_key].append((timestamp, power))
            
            # For each date, keep only the highest peak
            daily_max_peaks = []
            for date_key, day_peaks in peaks_by_date.items():
                # Find the highest peak for this day
                max_peak = max(day_peaks, key=lambda x: x[1])
                daily_max_peaks.append(max_peak)
            
            # Sort all daily max peaks by power (descending) and take top 3
            daily_max_peaks.sort(key=lambda x: x[1], reverse=True)
            top_3_daily_peaks = daily_max_peaks[:3]
            
            # Update the top3_peaks list with the properly constrained values
            old_peaks = self.top3_peaks.copy()
            self.top3_peaks = [0.0, 0.0, 0.0]
            for i, (_, power) in enumerate(top_3_daily_peaks):
                if i < 3:
                    self.top3_peaks[i] = power
            
            # Debug logging when peaks change significantly
            if self.debug_prints and old_peaks != self.top3_peaks:
                peak_dates = [timestamp.strftime('%Y-%m-%d') for timestamp, _ in top_3_daily_peaks]
                logger.debug(f"Capacity peaks updated (same-day constraint): {[f'{p:.2f}kW' for p in self.top3_peaks]} " +
                           f"from dates: {peak_dates}")
        else:
            # Original logic without same-day constraint (for comparison/testing)
            if effective_grid_power > min(self.top3_peaks) or 0.0 in self.top3_peaks:
                self.top3_peaks[self.top3_peaks.index(min(self.top3_peaks))] = effective_grid_power
                self.top3_peaks.sort(reverse=True)
        
        # Calculate rolling average of actual peaks (excluding zeros)
        non_zero_peaks = [p for p in self.top3_peaks if p > 0]
        self.peak_rolling_average = sum(non_zero_peaks) / len(non_zero_peaks) if non_zero_peaks else 0.0

# Example usage (for testing the simplified environment)
if __name__ == "__main__":
    # Test with settings that should use historical data
    # Ensure a valid price_predictions_path is available or expect critical errors.
    # For this example, we assume use_price_predictions=True is the norm.
    # If your PRICE_PREDICTIONS_PATH is not valid, it will log errors and use zero forecasts.
    env = HomeEnergyEnv(
        simulation_days=3, 
        fixed_baseload_kw=1.0, 
        use_price_predictions=True, # Explicitly True
        # price_predictions_path="path/to/your/actual/price_data.csv" # Replace if needed
    )
    obs, info = env.reset()
    print("Initial Observation:", obs)
    
    terminated = False
    total_reward_sum = 0
    print(f"Simulating for {env.simulation_steps} steps...")
    for step_num in range(env.simulation_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward_sum += reward
        if (step_num + 1) % (24 * int(1/env.time_step_hours)) == 0 or step_num == 0 : # Log daily or first step
            print(f"--- Step {info.get('current_step', step_num + 1)} ---")
            print(f"  SoC: {obs['soc'][0]:.2f}, Price: {info.get('current_price',0):.2f}, Grid kW: {info.get('grid_power_kw',0):.2f}, Reward: {reward:.2f}")
        if terminated or truncated:
            print("Episode finished.")
            break
    print(f"Total reward over episode: {total_reward_sum:.2f}")
    print(f"Final Total Cost: {env.total_cost:.2f}")
    print(f"Peak Power Seen: {env.peak_power:.2f} kW")
    env.close() 