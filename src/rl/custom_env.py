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

# Get a logger without re-configuring basicConfig
logger = logging.getLogger("home_energy_env")

# Default paths for prediction files
PRICE_PREDICTIONS_PATH = "src/predictions/prices/plots/predictions/merged"

# Default path for actual solar data
ACTUAL_SOLAR_DATA_PATH = "src/predictions/solar/actual_data/ActualSolarProductionData.csv"

class HomeEnergyEnv(gym.Env):
    """
    Simplified environment for controlling a home battery system
    reacting to grid prices.
    
    Observation space contains:
    - Battery state of charge
    - Time index (hour of day, minute of hour, day of week)
    - Price forecast for next 24 hours (from historical data)
    - Solar forecast for the next 4 days (from solar data)
    
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
                
        logger.info("\n{:=^80}".format(" Initializing HomeEnergyEnv "))

        # Load parameters from config with defaults
        self.battery_capacity = self.config.get("battery_capacity", 22.0)
        self.simulation_days = self.config.get("simulation_days", 7)
        self.peak_penalty_factor = self.config.get("peak_penalty_factor", 5.0)
        self.render_mode = self.config.get("render_mode", None)
        self.use_price_predictions = self.config.get("use_price_predictions", True)
        self.price_predictions_path = self.config.get("price_predictions_path", PRICE_PREDICTIONS_PATH)
        self.fixed_baseload_kw = self.config.get("fixed_baseload_kw", 0.5)
        self.time_step_minutes = self.config.get("time_step_minutes", 15)
        self.use_variable_consumption = self.config.get("use_variable_consumption", False)
        self.consumption_data_path = self.config.get("consumption_data_path", None)
        self.battery_degradation_cost_per_kwh = self.config.get("battery_degradation_cost_per_kwh", 45.0)
        self.use_solar_predictions = self.config.get("use_solar_predictions", True)
        self.solar_data_path = self.config.get("solar_data_path", ACTUAL_SOLAR_DATA_PATH)

        # Additional variables setup
        self.all_consumption_data_kw = None
        self.episode_consumption_kw = None
        self.all_solar_data_hourly_kw = None
        self.solar_forecast_actual = None
        self.solar_forecast_observed = None
        self.min_solar_data_date = None
        self.max_solar_data_date = None
        self.time_step_hours = self.time_step_minutes / 60.0
        
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
            degradation_cost_per_kwh=self.battery_degradation_cost_per_kwh
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
            "solar_forecast": spaces.Box( # 4 days * 24 hours
                low=0.0, high=10.0, shape=(4 * 24,), dtype=np.float32 # Placeholder high, actual solar capacity
            )
        })
        
        # State variables for tracking
        self.price_history = []
        self.grid_cost_history = []
        self.battery_cost_history = []
        self.total_cost = 0.0
        self.peak_power = 0.0
        
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
                            shape=(4*24,), dtype=np.float32
                        )
                        logger.info(f"Adjusted solar_forecast observation space high to: {max_solar_val * 1.1:.2f} kW")

        logger.info("{:-^80}".format(" Environment Initialized "))
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            logger.info(f"Random seed set to: {seed}")
        
        self.current_step = 0

        # Initialize consumption data dates
        min_consum_date = None
        max_consum_date = None

        if self.use_variable_consumption:
            # _load_consumption_data() is called in __init__ if use_variable_consumption is True.
            # We just need to check if it was successful.
            if self.all_consumption_data_kw is not None and not self.all_consumption_data_kw.empty:
                min_consum_date = self.all_consumption_data_kw.index.min()
                max_consum_date = self.all_consumption_data_kw.index.max()
            else:
                logger.warning("Variable consumption enabled, but all_consumption_data_kw is not available at reset. "
                               "This might indicate an issue during __init__ or data loading. "
                               "Falling back to non-variable consumption behavior for date ranging.")
                # min_consum_date and max_consum_date remain None

        # Determine the overall valid date range for starting simulations
        effective_min_data_date = self.min_price_data_date 
        effective_max_data_date = self.max_price_data_date

        # Factor in consumption data date range
        if self.use_variable_consumption and min_consum_date is not None and max_consum_date is not None:
            if effective_min_data_date is None: # If no price data, use consumption range
                effective_min_data_date = min_consum_date
                effective_max_data_date = max_consum_date # Also set max based on consumption if min was None
            else: # Intersect with price data
                effective_min_data_date = max(effective_min_data_date, min_consum_date)
                if effective_max_data_date is not None: # Ensure effective_max_data_date from prices is not None
                     effective_max_data_date = min(effective_max_data_date, max_consum_date)
                else: # If price max date was None, use consumption max date
                     effective_max_data_date = max_consum_date
        
        # Factor in solar data date range
        if self.use_solar_predictions and self.min_solar_data_date is not None and self.max_solar_data_date is not None:
            if effective_min_data_date is None: # If no price/consumption data, use solar range
                effective_min_data_date = self.min_solar_data_date
                effective_max_data_date = self.max_solar_data_date
            else: # Intersect with existing effective range
                effective_min_data_date = max(effective_min_data_date, self.min_solar_data_date)
                if effective_max_data_date is not None:
                    effective_max_data_date = min(effective_max_data_date, self.max_solar_data_date)
                else: # If previous effective max date was None (e.g. only price min was set)
                    effective_max_data_date = self.max_solar_data_date

        if effective_min_data_date is None or effective_max_data_date is None:
            logger.critical("CRITICAL: No valid date range found from price/consumption/solar data. Cannot set start_datetime. Exiting.")
            sys.exit(1)

        # Ensure dates are not None before proceeding
        if effective_min_data_date and effective_max_data_date:
            max_possible_start_date = effective_max_data_date - pd.Timedelta(days=self.simulation_hours // 24)
            
            if effective_min_data_date > max_possible_start_date:
                logger.warning(
                    f"Overall data range (Price: {self.min_price_data_date}-{self.max_price_data_date}, "
                    f"Consumption: {min_consum_date}-{max_consum_date}) resulting in effective range "
                    f"{effective_min_data_date} to {effective_max_data_date} is too short for simulation_days ({self.simulation_hours // 24}). "
                    f"Attempting to start at earliest possible: {effective_min_data_date}"
                )
                self.start_datetime = effective_min_data_date
            else:
                time_delta_seconds = (max_possible_start_date - effective_min_data_date).total_seconds()
                if time_delta_seconds < 0:
                     logger.warning(f"max_possible_start_date ({max_possible_start_date}) is before effective_min_data_date ({effective_min_data_date}). Defaulting to effective_min_data_date.")
                     self.start_datetime = effective_min_data_date
                else:
                    num_sampleable_days = (max_possible_start_date.normalize() - effective_min_data_date.normalize()).days + 1
                    if num_sampleable_days <= 0:
                        logger.warning(f"Calculated num_sampleable_days is {num_sampleable_days} based on effective range. Defaulting to effective_min_data_date.")
                        self.start_datetime = effective_min_data_date
                    else:
                        random_day_offset = np.random.randint(0, num_sampleable_days)
                        self.start_datetime = effective_min_data_date.normalize() + pd.Timedelta(days=random_day_offset)
        else:
            logger.critical(
                "Fallback: effective_min_data_date or effective_max_data_date is None. "
                "This should not happen if data loading was successful. Using system time fallback. This indicates a problem."
            )
            # Fallback to system time if all data loading/range logic fails (should be rare)
            current_year = datetime.datetime.now().year
            max_start_day_offset = 365 - (self.simulation_hours // 24)
            start_day_offset = np.random.randint(1, max_start_day_offset + 1) if max_start_day_offset > 0 else 0
            self.start_datetime = datetime.datetime(current_year, 1, 1) + datetime.timedelta(days=start_day_offset)

        self.battery.reset()
        self.peak_power = 0.0
        self.total_cost = 0.0
        self.last_action = None

        self._initialize_forecasts()
        if self.use_solar_predictions:
            self._initialize_solar_forecasts()
        
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
                    # logger.info(f"Variable consumption profile for episode loaded. Length: {len(self.episode_consumption_kw)}")

            except Exception as e:
                logger.error(f"Error selecting episode consumption data: {e}. Defaulting to fixed baseload.")
                self.episode_consumption_kw = np.full(self.simulation_steps, self.fixed_baseload_kw) # Fallback
        else:
            self.episode_consumption_kw = np.full(self.simulation_steps, self.fixed_baseload_kw) # Fallback if not using variable
            logger.info(f"Using fixed baseload of {self.fixed_baseload_kw} kW for this episode.")

        observation = self._get_observation()
        info = self._get_info()
        return observation, info
    
    def step(self, action: Union[np.ndarray, Dict]) -> Tuple[Dict, float, bool, bool, Dict]:
        current_timestamp = self.start_datetime + datetime.timedelta(hours=self.current_step * self.time_step_hours)
        logger.debug(f"\n--- Step {self.current_step} --- Timestamp: {current_timestamp} ---")
        
        battery_action = float(action[0] if isinstance(action, np.ndarray) and action.ndim > 0 else action)
        self.last_action = battery_action
        logger.debug(f"  Action received: {battery_action:.4f}")
        
        # SoC action penalties are no longer needed as we've implemented hard constraints
        # in the _handle_battery_action method
        soc_action_penalty = 0.0

        battery_info = self._handle_battery_action(battery_action)
        # battery_info already logs its details
        
        # Re-implement penalty for attempted constraint violations
        # Add the penalty if the agent attempted an invalid action (that required constraint enforcement)
        if battery_info.get("constraint_applied", False):
            soc_action_penalty_factor = self.config.get("soc_action_penalty_val", -500.0)
            soc_action_penalty = soc_action_penalty_factor * abs(battery_action)
            logger.debug(f"  Applied SoC action penalty: {soc_action_penalty:.4f} for attempted constraint violation")
        else:
            soc_action_penalty = 0.0

        current_price = self.price_forecast_actual[self.current_step]
        logger.debug(f"  Current Price: {current_price:.2f} öre/kWh")
        
        current_solar_production_kw = 0.0
        if self.use_solar_predictions and self.solar_forecast_actual is not None:
            current_solar_production_kw = self.solar_forecast_actual[self.current_step]
            # Ensure solar production is not negative (can happen with faulty data or ffill at start)
            current_solar_production_kw = max(0.0, current_solar_production_kw)
        logger.debug(f"  Current Solar Production: {current_solar_production_kw:.2f} kW")
        
        if self.use_variable_consumption and self.episode_consumption_kw is not None:
            current_demand_kw = self.episode_consumption_kw[self.current_step]
        else: # Fallback or fixed mode
            current_demand_kw = self.fixed_baseload_kw
        
        base_demand_kw = current_demand_kw # For logging consistency and if needed elsewhere
        net_load_before_battery = current_demand_kw - current_solar_production_kw # Subtract solar
        logger.debug(f"  Current Demand (Gross): {current_demand_kw:.2f} kW")
        logger.debug(f"  Net Load Before Battery (Demand - Solar): {net_load_before_battery:.2f} kW")

        battery_power_kw = battery_info["power_kw"]
        # Logged in _handle_battery_action, but good to see in context too
        logger.debug(f"  Battery Power: {battery_power_kw:.2f} kW (Neg=Charge, Pos=Discharge)")
        
        grid_power_kw = net_load_before_battery - battery_power_kw
        logger.debug(f"  Grid Power: {grid_power_kw:.2f} kW (Neg=Export, Pos=Import)")
        
        grid_energy_kwh = grid_power_kw * self.time_step_hours
        grid_cost = grid_energy_kwh * current_price
        logger.debug(f"  Grid Cost for step: {grid_cost:.4f}")
        
        battery_cost = battery_info.get("degradation_cost", 0.0)
        # Logged in _handle_battery_action
        
        # Get peak_threshold_kw from config, default to 5.0
        peak_thresh_kw = self.config.get("peak_threshold_kw", 5.0)
        peak_penalty_factor = self.config.get("peak_penalty_factor", 5.0)
        peak_penalty = 0
        if grid_power_kw > peak_thresh_kw:
             peak_penalty = -peak_penalty_factor * (grid_power_kw - peak_thresh_kw) * self.time_step_hours
        
        if grid_power_kw > self.peak_power:
            self.peak_power = grid_power_kw

        # Get scaling factors from config
        battery_cost_scale = self.config.get("battery_cost_scale_factor", 1.0)
        peak_penalty_scale = self.config.get("peak_penalty_scale_factor", 0.5)

        scaled_battery_cost = battery_cost * battery_cost_scale
        scaled_peak_penalty = peak_penalty * peak_penalty_scale
        
        price_arbitrage_bonus = 0
        
        # Get arbitrage parameters from config
        charge_thresh_price = self.config.get("charge_bonus_threshold_price", 20.0)
        charge_multiplier = self.config.get("charge_bonus_multiplier", 5.0)
        discharge_thresh_moderate = self.config.get("discharge_bonus_threshold_price_moderate", 50.0)
        discharge_multiplier_moderate = self.config.get("discharge_bonus_multiplier_moderate", 10.0)
        discharge_thresh_high = self.config.get("discharge_bonus_threshold_price_high", 100.0)
        discharge_multiplier_high = self.config.get("discharge_bonus_multiplier_high", 25.0)

        # === START DETAILED LOGGING FOR REWARD DEBUGGING ===
        logger.debug(f"  REWARD_DEBUG: --- Config Values Used This Step ---")
        logger.debug(f"  REWARD_DEBUG: charge_thresh_price: {charge_thresh_price}")
        logger.debug(f"  REWARD_DEBUG: charge_multiplier: {charge_multiplier}")
        logger.debug(f"  REWARD_DEBUG: discharge_thresh_moderate: {discharge_thresh_moderate}")
        logger.debug(f"  REWARD_DEBUG: discharge_multiplier_moderate: {discharge_multiplier_moderate}")
        logger.debug(f"  REWARD_DEBUG: discharge_thresh_high: {discharge_thresh_high}")
        logger.debug(f"  REWARD_DEBUG: discharge_multiplier_high: {discharge_multiplier_high}")
        logger.debug(f"  REWARD_DEBUG: battery_cost_scale_factor: {battery_cost_scale}")
        logger.debug(f"  REWARD_DEBUG: peak_penalty_scale_factor: {peak_penalty_scale}")
        logger.debug(f"  REWARD_DEBUG: peak_threshold_kw (config): {peak_thresh_kw}")
        logger.debug(f"  REWARD_DEBUG: peak_penalty_factor (config): {peak_penalty_factor}")
        logger.debug(f"  REWARD_DEBUG: battery_degradation_cost_per_kwh (battery init): {self.battery.degradation_cost_per_kwh}")
        logger.debug(f"  REWARD_DEBUG: --- Intermediate Values ---")
        logger.debug(f"  REWARD_DEBUG: current_price: {current_price:.4f}")
        logger.debug(f"  REWARD_DEBUG: battery_power_kw: {battery_power_kw:.4f}")
        logger.debug(f"  REWARD_DEBUG: grid_power_kw: {grid_power_kw:.4f}")
        logger.debug(f"  REWARD_DEBUG: grid_energy_kwh: {grid_energy_kwh:.4f}")
        logger.debug(f"  REWARD_DEBUG: raw_grid_cost: {grid_cost:.4f}")
        logger.debug(f"  REWARD_DEBUG: raw_battery_cost (degradation): {battery_cost:.4f}")
        logger.debug(f"  REWARD_DEBUG: raw_peak_penalty: {peak_penalty:.4f}")
        # === END DETAILED LOGGING FOR REWARD DEBUGGING ===

        if battery_power_kw < 0 and current_price < charge_thresh_price:
            price_arbitrage_bonus = charge_multiplier * abs(battery_power_kw) * self.time_step_hours
        elif battery_power_kw > 0 and current_price > discharge_thresh_moderate:
             price_arbitrage_bonus = discharge_multiplier_moderate * battery_power_kw * self.time_step_hours
        if battery_power_kw > 0 and current_price > discharge_thresh_high: # Additional bonus for very high prices
             price_arbitrage_bonus += discharge_multiplier_high * battery_power_kw * self.time_step_hours # Note: This logic means high bonus can stack if > moderate & > high
        
        # === DETAILED LOGGING FOR REWARD COMPONENTS ===
        logger.debug(f"  REWARD_DEBUG: --- Final Reward Component Values ---")
        logger.debug(f"  REWARD_DEBUG: Term (-grid_cost): {-grid_cost:.4f}")
        logger.debug(f"  REWARD_DEBUG: Term (scaled_peak_penalty): {scaled_peak_penalty:.4f}")
        logger.debug(f"  REWARD_DEBUG: Term (-scaled_battery_cost): {-scaled_battery_cost:.4f}")
        logger.debug(f"  REWARD_DEBUG: Term (price_arbitrage_bonus): {price_arbitrage_bonus:.4f}")
        logger.debug(f"  REWARD_DEBUG: Term (soc_action_penalty): {soc_action_penalty:.4f}")
        # === END DETAILED LOGGING FOR REWARD COMPONENTS ===

        reward = -grid_cost + scaled_peak_penalty - scaled_battery_cost + price_arbitrage_bonus + soc_action_penalty
        logger.debug(f"  Reward Components: GridCostTerm ({-grid_cost:.4f}), PeakPenaltyTerm ({scaled_peak_penalty:.4f}), ScaledBatteryCostTerm ({-scaled_battery_cost:.4f}), ArbitrageBonus ({price_arbitrage_bonus:.4f}), SoCActionPenalty ({soc_action_penalty:.4f})")
        logger.debug(f"  Total Reward for step: {reward:.4f}")
        
        self.price_history.append(current_price)
        self.grid_cost_history.append(grid_cost)
        self.battery_cost_history.append(battery_cost)
        self.total_cost += grid_cost
        
        self.current_step += 1
        terminated = self.current_step >= self.simulation_steps
        truncated = False
        
        observation = self._get_observation()
        info = self._get_info()
        info.update(battery_info)
        info["current_price"] = current_price
        info["grid_power_kw"] = grid_power_kw
        info["base_demand_kw"] = base_demand_kw
        info["battery_cost"] = battery_cost
        info["current_solar_production_kw"] = current_solar_production_kw
        
        # Add night discount information for visualization
        if hasattr(self, 'is_night_discount') and self.current_step < len(self.is_night_discount):
            info["is_night_discount"] = self.is_night_discount[self.current_step]
        else:
            info["is_night_discount"] = False
        
        # Store individual reward components in info dict for logging/evaluation
        info["reward_grid_cost"] = -grid_cost
        info["reward_peak_penalty"] = scaled_peak_penalty
        info["reward_battery_cost"] = -scaled_battery_cost # Note: battery_cost itself is positive, term is negative
        info["reward_arbitrage_bonus"] = price_arbitrage_bonus
        info["reward_soc_action_penalty"] = soc_action_penalty
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def _handle_battery_action(self, battery_action: float) -> Dict:
        duration_hours = self.time_step_hours
        actual_power_kw = 0.0
        degradation_cost = 0.0
        constraint_applied = False

        logger.debug(f"  _handle_battery_action: Input action: {battery_action:.4f}")

        # Get current demand and solar production to calculate export constraints
        if self.use_variable_consumption and self.episode_consumption_kw is not None:
            current_demand_kw = self.episode_consumption_kw[self.current_step]
        else:
            current_demand_kw = self.fixed_baseload_kw
        
        if self.use_solar_predictions and self.solar_forecast_actual is not None:
            current_solar_kw = max(0.0, self.solar_forecast_actual[self.current_step])
        else:
            current_solar_kw = 0.0
        
        net_load = current_demand_kw - current_solar_kw
        
        # Get export threshold from config or use default (use absolute value for clarity)
        export_peak_thresh_kw = abs(self.config.get("export_peak_threshold_kw", 20.0))
        
        # Calculate maximum safe discharge to avoid overloading grid export
        # If net_load is -10 (exporting 10kW solar) and threshold is 20kW, 
        # we can safely discharge max 10kW more from battery
        max_safe_discharge_kw = max(0.0, export_peak_thresh_kw + net_load)
        
        if battery_action < 0:  # Charge
            # Skip charging if battery is already at max_soc (hard constraint)
            if self.battery.soc >= self.battery.max_soc:
                logger.debug(f"    Battery already at max SoC ({self.battery.soc:.4f}). Charging request ignored.")
                constraint_applied = True
                return {
                    "power_kw": 0.0,
                    "degradation_cost": 0.0,
                    "soc": self.battery.soc,
                    "constraint_applied": constraint_applied
                }
                
            requested_charge_power_kw = -battery_action * self.battery.max_charge_rate
            logger.debug(f"    Attempting Charge: Requested Power = {requested_charge_power_kw:.2f} kW")
            requested_charge_energy_kwh = requested_charge_power_kw * duration_hours
            actual_charged_kwh, degradation_cost = self.battery.charge(
                amount_kwh=requested_charge_energy_kwh, hours=duration_hours
            )
            actual_power_kw = -actual_charged_kwh / duration_hours
            
        elif battery_action > 0:  # Discharge
            # Skip discharging if battery is already at min_soc (hard constraint)
            if self.battery.soc <= self.battery.min_soc:
                logger.debug(f"    Battery already at min SoC ({self.battery.soc:.4f}). Discharge request ignored.")
                constraint_applied = True
                return {
                    "power_kw": 0.0,
                    "degradation_cost": 0.0,
                    "soc": self.battery.soc,
                    "constraint_applied": constraint_applied
                }
            
            # Limit discharge based on grid export threshold
            requested_discharge_power_kw = battery_action * self.battery.max_discharge_rate
            # Apply grid export constraint - limit discharge to avoid overloading
            constrained_discharge_power_kw = min(requested_discharge_power_kw, max_safe_discharge_kw)
            
            if constrained_discharge_power_kw < requested_discharge_power_kw:
                logger.debug(f"    Grid export constraint applied: Reduced discharge from {requested_discharge_power_kw:.2f} kW to {constrained_discharge_power_kw:.2f} kW")
                constraint_applied = True
            
            logger.debug(f"    Attempting Discharge: Requested Power = {constrained_discharge_power_kw:.2f} kW (original: {requested_discharge_power_kw:.2f} kW)")
            requested_discharge_energy_kwh = constrained_discharge_power_kw * duration_hours
            actual_discharged_kwh, degradation_cost = self.battery.discharge(
                amount_kwh=requested_discharge_energy_kwh, hours=duration_hours
            )
            actual_power_kw = actual_discharged_kwh / duration_hours
        else:
            logger.debug(f"    Battery Idle Action (0.0)")

        logger.debug(f"    _handle_battery_action Result: Actual Power = {actual_power_kw:.2f} kW, Degradation Cost = {degradation_cost:.4f}, New SoC = {self.battery.soc:.4f}")
        return {
            "power_kw": actual_power_kw,
            "degradation_cost": degradation_cost,
            "soc": self.battery.soc,
            "constraint_applied": constraint_applied
        }
    
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
        resampled_prices = self.price_predictions_df['price'].resample(f'{int(self.time_step_hours*60)}min').interpolate(method='linear')
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
        self._apply_swedish_night_discount(sim_timestamps)
        
        self.price_forecast_observed = np.zeros((num_steps, 24))
        steps_per_hour = int(1 / self.time_step_hours)
        for i in range(num_steps):
            for j in range(24):
                hour_start_idx = i + (j * steps_per_hour)
                if hour_start_idx < num_steps:
                    self.price_forecast_observed[i, j] = self.price_forecast_actual[hour_start_idx]
                else:
                    self.price_forecast_observed[i, j] = self.price_forecast_actual[num_steps - 1]
    
    def _apply_swedish_night_discount(self, timestamps):
        """Apply Swedish night-time electricity discount (50% reduction from 22:00 to 06:00)"""
        # Get whether to apply the discount from config, default to True
        apply_night_discount = self.config.get("apply_swedish_night_discount", True)
        
        if not apply_night_discount:
            logger.info("Swedish night-time discount is disabled in config.")
            return
            
        night_discount_factor = self.config.get("night_discount_factor", 0.5)  # 50% discount by default
        logger.info(f"Applying Swedish night-time discount: {(1-night_discount_factor)*100}% off between 22:00-06:00")
        
        for i, timestamp in enumerate(timestamps):
            hour = timestamp.hour
            # Check if time is between 22:00 and 06:00
            if hour >= 22 or hour < 6:
                self.price_forecast_actual[i] *= night_discount_factor
                # Add a flag to indicate this is a discounted price (useful for visualization)
                if not hasattr(self, 'is_night_discount'):
                    self.is_night_discount = np.zeros(len(self.price_forecast_actual), dtype=bool)
                self.is_night_discount[i] = True
    
    def _initialize_solar_forecasts(self) -> None:
        """Initializes actual solar production for each simulation step and the 4-day hourly forecast."""
        num_steps = self.simulation_steps
        sim_timestamps = pd.to_datetime([self.start_datetime + datetime.timedelta(hours=i * self.time_step_hours) for i in range(num_steps)])

        if not self.use_solar_predictions or self.all_solar_data_hourly_kw is None or self.all_solar_data_hourly_kw.empty:
            logger.warning("Solar predictions disabled or no solar data loaded. Solar production will be zero.")
            self.solar_forecast_actual = np.zeros(num_steps)
            self.solar_forecast_observed = np.zeros((num_steps, 4 * 24)) # 4 days * 24 hours
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


        # 2. Initialize solar_forecast_observed (4-day hourly forecast for the agent)
        # Each row `i` in solar_forecast_observed contains 96 hourly values for the 4 days
        # starting from the hour of sim_timestamps[i].
        self.solar_forecast_observed = np.zeros((num_steps, 4 * 24))
        
        for i in range(num_steps):
            forecast_start_time = sim_timestamps[i]
            # Create 96 hourly timestamps for the forecast
            hourly_forecast_timestamps = pd.date_range(start=forecast_start_time, periods=(4 * 24), freq='h')
            
            # Reindex the original hourly solar data to these forecast timestamps
            observed_forecast_series = self.all_solar_data_hourly_kw['solar_production_kw'].reindex(
                hourly_forecast_timestamps, method='ffill' # Use ffill to carry last known value if forecast extends beyond data
            )
            # Fill any remaining NaNs (e.g., if forecast starts before data or extends far beyond) with 0
            observed_forecast_series = observed_forecast_series.fillna(0)
            
            self.solar_forecast_observed[i, :] = observed_forecast_series.values
        

    def _get_observation(self) -> Dict:
        current_dt = self.start_datetime + datetime.timedelta(hours=self.current_step * self.time_step_hours)
        hour_of_day = current_dt.hour
        minute_of_hour = current_dt.minute / 60.0 # Normalized
        day_of_week = current_dt.weekday()
        
        time_idx = np.array([hour_of_day, minute_of_hour, day_of_week], dtype=np.float32)
        
        current_step_idx = min(self.current_step, self.price_forecast_observed.shape[0] - 1)
        price_fc = self.price_forecast_observed[current_step_idx, :]

        solar_fc = np.zeros(4 * 24) # Default to zeros
        if self.use_solar_predictions and self.solar_forecast_observed is not None:
            current_solar_step_idx = min(self.current_step, self.solar_forecast_observed.shape[0] - 1)
            if current_solar_step_idx < self.solar_forecast_observed.shape[0]:
                 solar_fc = self.solar_forecast_observed[current_solar_step_idx, :]
            else:
                 logger.warning(f"Current step {self.current_step} is out of bounds for solar_forecast_observed shape {self.solar_forecast_observed.shape}. Using zeros.")


        return {
            "soc": np.array([self.battery.soc], dtype=np.float32),
            "time_idx": time_idx,
            "price_forecast": price_fc.astype(np.float32),
            "solar_forecast": solar_fc.astype(np.float32)
        }
    
    def _get_info(self) -> Dict:
        return {
            "current_step": self.current_step,
            "total_cost": self.total_cost,
            "peak_power": self.peak_power,
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
            print(f"  Total Cost: {self.total_cost:.2f}")
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
                print("---PRICE DATA BEFORE PROCESSING---")
                print(self.price_predictions_df.iloc[:, 0].tail(1)) # Show only the last timestamp, not all columns
                print("-----------------------")

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
                        print("---PRICE DATA AFTER PROCESSING---")
                        print(self.price_predictions_df.iloc[:, 0].tail(1)) # Show only the last timestamp
                        print("-----------------------")
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
                print("---SOLAR DATA BEFORE PROCESSING---")
                print(df.iloc[:, 0].tail(1))
                print("-----------------------")
                
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
                print("---SOLAR DATA AFTER PROCESSING---")
                print(self.all_solar_data_hourly_kw.iloc[:,0].tail(1))
                print("-----------------------")

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
                    print("---CONSUMPTION DATA BEFORE PROCESSING---")
                    print(df.iloc[:,0].tail(1))
                    print("-----------------------")
                    
                # Drop extra columns
                df.drop(columns=['cost', 'unit_price', 'unit_price_vat', 'unit'], inplace=True)

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
                    print("---CONSUMPTION DATA AFTER PROCESSING---")
                    print(df.iloc[:,0].tail(1))
                    print("-----------------------")    
                    
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