"""
Simplified Custom RL environment for home energy system control (battery only).
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import datetime
import logging
import os
from pathlib import Path
import sys
from typing import Dict, Tuple, List, Optional, Any

# Adjust sys.path to allow imports from the project root (e.g., src.rl.components)
# This is typically needed when running a script directly from a subdirectory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.rl.components import Battery
from src.rl.config import get_config_dict

logger = logging.getLogger(__name__)

class SimpleBatteryEnv(gym.Env):
    """
    Simplified environment for controlling a home battery system.
    Focuses on battery interaction with grid prices.

    Observation space (simplified):
    - Battery state of charge (SoC)
    - Time index (hour of day, minute of hour, day of week)
    - Price forecast for the next N hours (e.g., 24)

    Action space (simplified):
    - Battery charging/discharging rate (normalized: -1 to 1)
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, env_config: Optional[Dict] = None):
        super().__init__()

        base_config = get_config_dict()
        if env_config:
            base_config.update(env_config)
        self.config = base_config

        # Setup logger
        log_level_str = self.config.get("log_level", "INFO").upper()
        numeric_log_level = getattr(logging, log_level_str, logging.INFO)
        logging.basicConfig(level=numeric_log_level)
        logger.setLevel(numeric_log_level)

        logger.info("{:=^80}".format(" Initializing SimpleBatteryEnv "))

        # Time parameters
        self.time_step_minutes = self.config["time_step_minutes"]
        self.time_step_hours = self.time_step_minutes / 60.0
        self.simulation_days = self.config["simulation_days"]
        self.simulation_hours = self.simulation_days * 24
        self.simulation_steps = int(self.simulation_hours / self.time_step_hours)

        # Data paths & loading flags
        self.price_predictions_path = self.config["price_predictions_path"]
        self.use_variable_consumption = self.config["use_variable_consumption"]
        self.consumption_data_path = self.config["consumption_data_path"]
        self.use_solar_predictions = self.config["use_solar_predictions"]
        self.solar_data_path = self.config["solar_data_path"]
        self.fixed_baseload_kw = self.config["fixed_baseload_kw"]

        # Battery component
        self.battery = Battery(
            capacity_kwh=self.config["battery_capacity_kwh"],
            initial_soc=self.config["battery_initial_soc"],
            max_charge_power_kw=self.config["battery_max_charge_power_kw"],
            max_discharge_power_kw=self.config["battery_max_discharge_power_kw"],
            charge_efficiency=self.config["battery_charge_efficiency"],
            discharge_efficiency=self.config["battery_discharge_efficiency"],
            min_soc_limit=self.config["battery_min_soc_limit"],
            max_soc_limit=self.config["battery_max_soc_limit"]
        )

        # Action space: normalized power (-1 for max charge, 1 for max discharge)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation space
        self.forecast_horizon_hours = 24 # Agent sees 24 hours of price forecast
        obs_space_dict = {
            "soc": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "time_idx": spaces.Box(
                low=np.array([0, 0, 0]),  # hour_of_day, minute_of_hour (0-1), day_of_week (0-6)
                high=np.array([23, 1.0, 6]),
                dtype=np.float32
            ),
            "price_forecast": spaces.Box(
                low=0.0, high=1000.0, shape=(self.forecast_horizon_hours,), dtype=np.float32 # öre/kWh
            )
        }
        if self.use_solar_predictions:
            obs_space_dict["solar_forecast"] = spaces.Box(
                low=0.0, high=15.0, shape=(self.forecast_horizon_hours,), dtype=np.float32 # kW, placeholder high
            )
        # Later, we can add consumption forecasts here if enabled
        self.observation_space = spaces.Dict(obs_space_dict)

        # Dataframes for episode data
        self.price_predictions_df = None
        self.all_price_data = None # Full loaded price data
        self.episode_price_actual = None # Actual prices for the current episode steps
        self.episode_consumption_kw = None # Consumption for current episode steps
        self.episode_solar_kw = None # Solar for current episode steps (future)

        self.min_data_date = None
        self.max_data_date = None
        
        self._load_data()

        self.current_step = 0
        self.start_datetime = None
        self.total_reward_episode = 0.0
        self.total_grid_cost_episode = 0.0 # in öre

        logger.info(f"Time step: {self.time_step_minutes} min ({self.time_step_hours:.2f} hr). Sim steps: {self.simulation_steps}")
        logger.info("{:-^80}".format(" Environment Initialized "))

    def _load_data(self) -> None:
        """Loads all necessary data from CSV files."""
        self._load_price_predictions()
        self._load_consumption_data()
        self._load_solar_data()
        
        if self.all_price_data is None or self.all_price_data.empty:
            logger.critical("Price data could not be loaded. Environment cannot operate. Exiting.")
            sys.exit(1)
        
        self.min_data_date = self.all_price_data.index.min()
        self.max_data_date = self.all_price_data.index.max()
        logger.info(f"Price data loaded. Range: {self.min_data_date} to {self.max_data_date}")

    def _load_price_predictions(self) -> None:
        """Loads price prediction data from CSV."""
        path_str = self.price_predictions_path
        project_root = Path(__file__).resolve().parent.parent.parent
        file_path = project_root / path_str if not Path(path_str).is_absolute() else Path(path_str)

        if not file_path.exists():
            logger.error(f"Price predictions file not found at {file_path}")
            return
        try:
            df = pd.read_csv(file_path, index_col='HourSE', parse_dates=True)
            logger.info(f"Loaded price data from {file_path}. Columns: {df.columns.tolist()}")
            
            price_col = self.config["spot_price_source_column"]
            if price_col not in df.columns:
                logger.error(f"Spot price column '{price_col}' not found in {file_path}. Available: {df.columns.tolist()}")
                sys.exit(1)
                
            self.all_price_data = df[[price_col]].rename(columns={price_col: 'price_ore_kwh'})

            if not isinstance(self.all_price_data.index, pd.DatetimeIndex):
                self.all_price_data.index = pd.to_datetime(self.all_price_data.index)

            if self.all_price_data.index.tzinfo is None:
                self.all_price_data.index = self.all_price_data.index.tz_localize(
                    'Europe/Stockholm', ambiguous='NaT', nonexistent='shift_forward'
                )
                if self.all_price_data.index.hasnans:
                    logger.warning(f"Removed {self.all_price_data.index.isna().sum()} NaT timestamps from price data after tz_localize.")
                    self.all_price_data = self.all_price_data.dropna(axis=0)
            else:
                self.all_price_data.index = self.all_price_data.index.tz_convert('Europe/Stockholm')
            
            self.all_price_data.sort_index(inplace=True)
            if self.all_price_data.index.duplicated().any():
                logger.warning("Duplicate timestamps found in price data. Keeping first entry.")
                self.all_price_data = self.all_price_data[~self.all_price_data.index.duplicated(keep='first')]

            logger.info(f"Processed price data. Shape: {self.all_price_data.shape}")

        except Exception as e:
            logger.error(f"Error loading price predictions from {file_path}: {e}")
            self.all_price_data = None

    def _load_consumption_data(self) -> None:
        """Loads variable consumption data from CSV if enabled."""
        if not self.use_variable_consumption:
            logger.info("Variable consumption is not enabled. Using fixed baseload.")
            self.all_consumption_data_kw = None
            return

        path_str = self.consumption_data_path
        if not path_str:
            logger.error("`use_variable_consumption` is True, but `consumption_data_path` is not set.")
            self.all_consumption_data_kw = None
            return
            
        project_root = Path(__file__).resolve().parent.parent.parent
        file_path = project_root / path_str if not Path(path_str).is_absolute() else Path(path_str)

        if not file_path.exists():
            logger.error(f"Consumption data file not found at {file_path}")
            self.all_consumption_data_kw = None
            return
        
        try:
            df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
            logger.info(f"Loaded consumption data from {file_path}. Columns: {df.columns.tolist()}")

            # Expected columns from VillamichelinConsumption.csv: 'timestamp','cost','unit_price','unit_price_vat','unit','consumption'
            if 'consumption' not in df.columns:
                logger.error(f"'consumption' column not found in {file_path}. Available: {df.columns.tolist()}")
                self.all_consumption_data_kw = None
                return

            self.all_consumption_data_kw = df[['consumption']].rename(columns={'consumption': 'consumption_kw'})

            if not isinstance(self.all_consumption_data_kw.index, pd.DatetimeIndex):
                self.all_consumption_data_kw.index = pd.to_datetime(self.all_consumption_data_kw.index)

            # Handle timezone - VillamichelinConsumption.csv timestamps are likely Europe/Stockholm based on previous env
            if self.all_consumption_data_kw.index.tzinfo is None:
                self.all_consumption_data_kw.index = self.all_consumption_data_kw.index.tz_localize(
                    'Europe/Stockholm', ambiguous='NaT', nonexistent='shift_forward'
                )
                if self.all_consumption_data_kw.index.hasnans:
                    logger.warning(f"Removed {self.all_consumption_data_kw.index.isna().sum()} NaT timestamps from consumption data after tz_localize.")
                    self.all_consumption_data_kw = self.all_consumption_data_kw.dropna(axis=0)
            else:
                self.all_consumption_data_kw.index = self.all_consumption_data_kw.index.tz_convert('Europe/Stockholm')
            
            self.all_consumption_data_kw.sort_index(inplace=True)
            if self.all_consumption_data_kw.index.duplicated().any():
                logger.warning("Duplicate timestamps found in consumption data. Keeping first entry.")
                self.all_consumption_data_kw = self.all_consumption_data_kw[~self.all_consumption_data_kw.index.duplicated(keep='first')]

            # The data is hourly kWh, which is average kW for that hour. 
            # No resampling needed here as it will be resampled during episode initialization if timestep is different.
            logger.info(f"Processed consumption data. Shape: {self.all_consumption_data_kw.shape}")

        except Exception as e:
            logger.error(f"Error loading consumption data from {file_path}: {e}")
            self.all_consumption_data_kw = None

    def _load_solar_data(self) -> None:
        """Loads solar production data from CSV if enabled."""
        if not self.use_solar_predictions:
            logger.info("Solar predictions are not enabled.")
            self.all_solar_data_kw = None # Ensure attribute exists
            return

        path_str = self.solar_data_path
        if not path_str:
            logger.error("`use_solar_predictions` is True, but `solar_data_path` is not set.")
            self.all_solar_data_kw = None
            return
            
        project_root = Path(__file__).resolve().parent.parent.parent
        file_path = project_root / path_str if not Path(path_str).is_absolute() else Path(path_str)

        if not file_path.exists():
            logger.error(f"Solar data file not found at {file_path}")
            self.all_solar_data_kw = None
            return
        
        try:
            # ActualSolarProduction.csv has columns: Timestamp, solar_production_kwh, location, source
            # We expect Timestamps to be in UTC as per the previous environment's handling.
            df = pd.read_csv(file_path, parse_dates=['Timestamp'], index_col='Timestamp')
            logger.info(f"Loaded solar data from {file_path}. Columns: {df.columns.tolist()}")

            if 'solar_production_kwh' not in df.columns:
                logger.error(f"'solar_production_kwh' column not found in {file_path}. Available: {df.columns.tolist()}")
                self.all_solar_data_kw = None
                return
            
            # solar_production_kwh for an hour is average kW for that hour
            self.all_solar_data_kw = df[['solar_production_kwh']].rename(columns={'solar_production_kwh': 'solar_kw'})
            self.all_solar_data_kw['solar_kw'] = self.all_solar_data_kw['solar_kw'].clip(lower=0) # Ensure no negative production

            if not isinstance(self.all_solar_data_kw.index, pd.DatetimeIndex):
                self.all_solar_data_kw.index = pd.to_datetime(self.all_solar_data_kw.index)

            # Handle timezone - Solar data from custom_env was localized to UTC then Stockholm
            if self.all_solar_data_kw.index.tzinfo is None:
                logger.info("Solar data timestamps are naive. Assuming UTC and converting to Europe/Stockholm.")
                self.all_solar_data_kw.index = self.all_solar_data_kw.index.tz_localize('UTC', ambiguous='NaT', nonexistent='shift_forward').tz_convert('Europe/Stockholm')
            else:
                self.all_solar_data_kw.index = self.all_solar_data_kw.index.tz_convert('Europe/Stockholm')

            if self.all_solar_data_kw.index.hasnans:
                logger.warning(f"Removed {self.all_solar_data_kw.index.isna().sum()} NaT timestamps from solar data after tz_localize/convert.")
                self.all_solar_data_kw = self.all_solar_data_kw.dropna(axis=0)
            
            self.all_solar_data_kw.sort_index(inplace=True)
            if self.all_solar_data_kw.index.duplicated().any():
                logger.warning("Duplicate timestamps found in solar data. Keeping first entry.")
                self.all_solar_data_kw = self.all_solar_data_kw[~self.all_solar_data_kw.index.duplicated(keep='first')]
            
            logger.info(f"Processed solar data. Shape: {self.all_solar_data_kw.shape}")

        except Exception as e:
            logger.error(f"Error loading solar data from {file_path}: {e}")
            self.all_solar_data_kw = None

    def _set_random_start_datetime(self) -> None:
        """Sets a random start datetime for an episode based on available data."""
        if self.all_price_data is None or self.all_price_data.empty:
            logger.critical("Price data not loaded. Cannot determine episode start.")
            sys.exit(1)

        # Start with price data range
        min_available_date = self.all_price_data.index.min()
        max_available_date = self.all_price_data.index.max()

        # If using variable consumption, intersect data ranges
        if self.use_variable_consumption:
            if self.all_consumption_data_kw is None or self.all_consumption_data_kw.empty:
                logger.critical("Variable consumption enabled, but data not loaded. Cannot determine episode start.")
                sys.exit(1)
            min_available_date = max(min_available_date, self.all_consumption_data_kw.index.min())
            max_available_date = min(max_available_date, self.all_consumption_data_kw.index.max())

        # If using solar, intersect data ranges further
        if self.use_solar_predictions:
            if self.all_solar_data_kw is None or self.all_solar_data_kw.empty:
                logger.critical("Solar predictions enabled, but data not loaded. Cannot determine episode start.")
                sys.exit(1)
            min_available_date = max(min_available_date, self.all_solar_data_kw.index.min())
            max_available_date = min(max_available_date, self.all_solar_data_kw.index.max())
        
        sim_duration_timedelta = pd.Timedelta(days=self.simulation_days)
        
        # User-defined start/end dates for sampling range
        user_start_date_str = self.config.get("start_date")
        user_end_date_str = self.config.get("end_date")

        effective_min_date = min_available_date
        if user_start_date_str:
            user_start_dt = pd.Timestamp(user_start_date_str, tz='Europe/Stockholm')
            effective_min_date = max(min_available_date, user_start_dt)

        latest_possible_data_needed = max_available_date 
        max_episode_start_date = latest_possible_data_needed - sim_duration_timedelta + pd.Timedelta(hours=self.time_step_hours)

        if user_end_date_str:
            user_end_dt = pd.Timestamp(user_end_date_str, tz='Europe/Stockholm')
            max_episode_start_date = min(max_episode_start_date, user_end_dt - sim_duration_timedelta + pd.Timedelta(hours=self.time_step_hours))

        if effective_min_date >= max_episode_start_date:
            logger.error(
                f"Overall data range too short or restrictive. Cannot find a valid start date. "
                f"Effective Min Available: {effective_min_date}, Latest Possible Episode Start: {max_episode_start_date}. "
                f"Simulation Days: {self.simulation_days}. Check data files and date range configs. Exiting."
            )
            sys.exit(1) # Critical error, cannot proceed
        
        min_day_norm = effective_min_date.normalize()
        max_day_norm = max_episode_start_date.normalize()
        
        if min_day_norm > max_day_norm:
            logger.warning(f"Normalized min_day ({min_day_norm}) is after max_day ({max_day_norm}). Using effective_min_date {effective_min_date} as fallback.")
            self.start_datetime = effective_min_date
        else:
            num_sampleable_days = (max_day_norm - min_day_norm).days + 1
            random_day_offset = np.random.randint(0, num_sampleable_days)
            self.start_datetime = min_day_norm + pd.Timedelta(days=random_day_offset)
        
        # Ensure start_datetime respects the timezone of the price data (which should be the reference)
        if self.all_price_data.index.tzinfo is not None:
            if self.start_datetime.tzinfo is None:
                self.start_datetime = self.start_datetime.tz_localize(self.all_price_data.index.tzinfo, ambiguous='raise', nonexistent='raise')
            elif self.start_datetime.tzinfo != self.all_price_data.index.tzinfo:
                self.start_datetime = self.start_datetime.tz_convert(self.all_price_data.index.tzinfo)

        logger.info(f"Episode start datetime set to: {self.start_datetime} (from range {effective_min_date} to {max_episode_start_date})")

    def _initialize_episode_data(self) -> None:
        """Prepares data (prices, consumption) for the current episode."""
        sim_end_datetime = self.start_datetime + pd.Timedelta(hours=self.simulation_hours) - pd.Timedelta(hours=self.time_step_hours)
        episode_timestamps = pd.date_range(
            start=self.start_datetime,
            periods=self.simulation_steps,
            freq=f"{self.time_step_minutes}min"
        )
        
        # Price data
        # Resample hourly prices to the environment's time_step_minutes, forward-filling
        resampled_prices = self.all_price_data['price_ore_kwh'].resample(f"{self.time_step_minutes}min").ffill()
        self.episode_price_actual = resampled_prices.reindex(episode_timestamps, method='ffill').bfill()
        if len(self.episode_price_actual) < self.simulation_steps:
            logger.error("Failed to get complete price data for the episode. Check data range and resampling.")
            # Fallback: pad with last known value or zeros
            padding_needed = self.simulation_steps - len(self.episode_price_actual)
            if not self.episode_price_actual.empty:
                padding_value = self.episode_price_actual.iloc[-1]
            else:
                padding_value = 0
            padding = pd.Series([padding_value] * padding_needed, index=episode_timestamps[-padding_needed:])
            self.episode_price_actual = pd.concat([self.episode_price_actual, padding])
        
        self.episode_price_actual = self.episode_price_actual.values[:self.simulation_steps]

        # Consumption data
        if self.use_variable_consumption and self.all_consumption_data_kw is not None:
            resampled_consumption = self.all_consumption_data_kw['consumption_kw'].resample(f"{self.time_step_minutes}min").ffill()
            self.episode_consumption_kw = resampled_consumption.reindex(episode_timestamps, method='ffill').bfill()
            if len(self.episode_consumption_kw) < self.simulation_steps:
                logger.error("Failed to get complete consumption data for the episode. Padding with fixed baseload.")
                padding_needed = self.simulation_steps - len(self.episode_consumption_kw)
                padding_value = self.fixed_baseload_kw # Fallback to fixed baseload
                padding = pd.Series([padding_value] * padding_needed, index=episode_timestamps[-padding_needed:])
                self.episode_consumption_kw = pd.concat([self.episode_consumption_kw, padding])
            self.episode_consumption_kw = self.episode_consumption_kw.values[:self.simulation_steps]
        else:
            self.episode_consumption_kw = np.full(self.simulation_steps, self.fixed_baseload_kw)
        
        # Solar data
        self.episode_solar_kw = np.zeros(self.simulation_steps) # Default to no solar
        if self.use_solar_predictions and self.all_solar_data_kw is not None:
            resampled_solar = self.all_solar_data_kw['solar_kw'].resample(f"{self.time_step_minutes}min").ffill()
            self.episode_solar_kw = resampled_solar.reindex(episode_timestamps, method='ffill').bfill()
            if len(self.episode_solar_kw) < self.simulation_steps:
                logger.error("Failed to get complete solar data for the episode. Padding with zeros.")
                padding_needed = self.simulation_steps - len(self.episode_solar_kw)
                padding_value = 0.0 # Fallback to zero solar
                padding = pd.Series([padding_value] * padding_needed, index=episode_timestamps[-padding_needed:])
                self.episode_solar_kw = pd.concat([self.episode_solar_kw, padding])
            self.episode_solar_kw = self.episode_solar_kw.values[:self.simulation_steps]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            # Note: Global random state is affected by this. For more isolated seeding, use
            # self.np_random, _ = gymnasium.utils.seeding.np_random(seed)
            logger.info(f"Random seed set to: {seed}")

        self.current_step = 0
        self.battery.reset(initial_soc=self.config["battery_initial_soc"]) 
        self.total_reward_episode = 0.0
        self.total_grid_cost_episode = 0.0

        self._set_random_start_datetime()
        self._initialize_episode_data()
        
        if self.episode_price_actual is None or len(self.episode_price_actual) < self.simulation_steps:
             logger.critical("Episode price data is not properly initialized. Env cannot proceed.")
             # This case should ideally be caught earlier or handled with a more robust fallback.
             # For now, returning a dummy observation and exiting might be an option, or raising an error.
             # Returning a valid observation structure filled with zeros/defaults:
             dummy_obs = self._get_dummy_observation()
             return dummy_obs, {}

        observation = self._get_observation()
        info = self._get_info()
        return observation, info
    
    def _get_dummy_observation(self) -> Dict:
        """ Returns a valid observation structure with default values, for error cases. """
        return {
            "soc": np.array([self.config["battery_initial_soc"]], dtype=np.float32),
            "time_idx": np.array([0,0,0], dtype=np.float32),
            "price_forecast": np.zeros(self.forecast_horizon_hours, dtype=np.float32)
        }

    def _map_action_to_battery_power(self, normalized_action: float) -> float:
        """Maps normalized action (-1 to 1) to battery power (kW)."""
        if normalized_action >= 0: # Discharge
            return normalized_action * self.battery.max_discharge_power_kw
        else: # Charge
            return normalized_action * self.battery.max_charge_power_kw # action is negative

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        normalized_action = float(action[0])
        target_battery_terminal_power_kw = self._map_action_to_battery_power(normalized_action)

        # Battery step
        actual_battery_terminal_power_kw, _, _ = self.battery.step(
            target_terminal_power_kw,
            self.time_step_hours
        )

        # Get current values for this step
        current_consumption_kw = self.episode_consumption_kw[self.current_step]
        current_price_ore_per_kwh = self.episode_price_actual[self.current_step]
        current_solar_kw = self.episode_solar_kw[self.current_step]

        # Net power calculation
        # Positive grid_power_kw means import, negative means export
        # grid_power = house_demand - battery_discharge_power + battery_charge_power
        # house_demand = consumption - solar
        net_house_demand_kw = current_consumption_kw - current_solar_kw
        grid_power_kw = net_house_demand_kw - actual_battery_terminal_power_kw
        
        # Cost/Revenue calculation (in öre for this step)
        step_cost_ore = 0.0
        grid_energy_kwh_step = grid_power_kw * self.time_step_hours

        # Full cost calculation (including taxes and fees if importing)
        if grid_power_kw > 0: # Importing
            spot_price_with_vat = current_price_ore_per_kwh * self.config["vat_mult"]
            total_price_ore_kwh = spot_price_with_vat + self.config["energy_tax_ore_per_kwh"] + self.config["grid_fee_ore_per_kwh"]
            step_cost_ore = grid_energy_kwh_step * total_price_ore_kwh
        elif grid_power_kw < 0: # Exporting (assume export price is spot price for simplicity initially)
            # Revenue is negative cost
            step_cost_ore = grid_energy_kwh_step * current_price_ore_per_kwh 
        
        self.total_grid_cost_episode += step_cost_ore

        # Reward: For now, simple negative cost (so agent minimizes cost)
        # We scale it to be somewhat reasonable, 1 öre = 0.01 reward points roughly
        reward = (-step_cost_ore / 100.0) * self.config["reward_grid_cost_scale"]

        # Simple SoC comfort zone penalty (can be expanded)
        soc_penalty = 0.0
        if self.battery.soc < self.config["preferred_soc_min"]: 
            soc_penalty = (self.config["preferred_soc_min"] - self.battery.soc) * 10 # Arbitrary scale for penalty
        elif self.battery.soc > self.config["preferred_soc_max"]: 
            soc_penalty = (self.battery.soc - self.config["preferred_soc_max"]) * 10
        
        reward -= soc_penalty * self.config["reward_soc_penalty_scale"]
        self.total_reward_episode += reward

        # Update state
        self.current_step += 1
        terminated = self.current_step >= self.simulation_steps
        truncated = False # Not using truncation for now

        observation = self._get_observation()
        info = self._get_info()
        info["actual_battery_terminal_power_kw"] = actual_battery_terminal_power_kw
        info["grid_power_kw"] = grid_power_kw
        info["step_cost_ore"] = step_cost_ore
        info["current_consumption_kw"] = current_consumption_kw
        info["current_solar_kw"] = current_solar_kw

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> Dict:
        current_dt = self.start_datetime + pd.Timedelta(hours=self.current_step * self.time_step_hours)
        
        time_idx = np.array([
            current_dt.hour,
            current_dt.minute / 59.0,  # Normalize minute (0 to ~1)
            current_dt.weekday()       # Monday=0 to Sunday=6
        ], dtype=np.float32)

        # Price forecast for the next N hours from the current step
        # The forecast is hourly, so we need to select the correct hourly price
        price_fc = np.zeros(self.forecast_horizon_hours, dtype=np.float32)
        if self.episode_price_actual is not None:
            num_steps_per_hour = int(60 / self.time_step_minutes)
            for i in range(self.forecast_horizon_hours):
                forecast_step_idx = self.current_step + (i * num_steps_per_hour)
                if forecast_step_idx < len(self.episode_price_actual):
                    price_fc[i] = self.episode_price_actual[forecast_step_idx]
                elif len(self.episode_price_actual) > 0: 
                    price_fc[i] = self.episode_price_actual[-1]
                else: 
                    price_fc[i] = 0 
        
        obs_data = {
            "soc": np.array([self.battery.soc], dtype=np.float32),
            "time_idx": time_idx,
            "price_forecast": price_fc
        }

        if self.use_solar_predictions:
            solar_fc = np.zeros(self.forecast_horizon_hours, dtype=np.float32)
            if self.episode_solar_kw is not None:
                num_steps_per_hour = int(60 / self.time_step_minutes)
                for i in range(self.forecast_horizon_hours):
                    forecast_step_idx = self.current_step + (i * num_steps_per_hour)
                    if forecast_step_idx < len(self.episode_solar_kw):
                        solar_fc[i] = self.episode_solar_kw[forecast_step_idx]
                    elif len(self.episode_solar_kw) > 0:
                        solar_fc[i] = self.episode_solar_kw[-1]
                    else:
                        solar_fc[i] = 0
            obs_data["solar_forecast"] = solar_fc

        return obs_data

    def _get_info(self) -> Dict:
        current_dt = self.start_datetime + pd.Timedelta(hours=self.current_step * self.time_step_hours)
        return {
            "current_step": self.current_step,
            "current_datetime": current_dt.isoformat(),
            "current_price_ore_per_kwh": self.episode_price_actual[self.current_step] if self.episode_price_actual is not None and self.current_step < len(self.episode_price_actual) else -1,
            "battery_soc": self.battery.soc,
            "total_reward_episode": self.total_reward_episode,
            "total_grid_cost_episode_ore": self.total_grid_cost_episode
        }

    def render(self, mode='human'):
        if mode == 'human':
            obs = self._get_observation()
            info = self._get_info()
            print(f"--- Step: {info['current_step']}/{self.simulation_steps} ---")
            print(f"  Time: {info['current_datetime']}, SoC: {obs['soc'][0]:.3f}")
            print(f"  Price: {info['current_price_ore_per_kwh']:.2f} öre/kWh")
            # Add more printouts as needed, e.g., action taken, reward
            if "actual_battery_terminal_power_kw" in info:
                 print(f"  Battery Power: {info['actual_battery_terminal_power_kw']:.2f} kW, Grid Power: {info['grid_power_kw']:.2f} kW")
            print(f"  Step Reward: N/A (not passed to render), Episodic Reward: {info['total_reward_episode']:.2f}")
            print(f"  Price Forecast (next 3h): {obs['price_forecast'][:3]}")
            print("-" * 30)

    def close(self) -> None:
        logger.info("SimpleBatteryEnv closed.")

# Example usage for testing the environment
if __name__ == '''__main__''':
    env_params = {
        "simulation_days": 2,
        "log_level": "DEBUG",
        "debug_prints": True,
        # "start_date": "2023-03-01", # Optional: for specific start date testing
        # "end_date": "2023-03-10"      # Optional: for specific end date testing
    }
    env = SimpleBatteryEnv(env_config=env_params)
    obs, info = env.reset()
    print("Initial Observation:", obs)
    print("Initial Info:", info)

    terminated = False
    total_reward_sum = 0
    num_test_steps = env.simulation_steps # Run for full episode

    for step_num in range(num_test_steps):
        action = env.action_space.sample()  # Sample a random action
        # action = np.array([-0.5], dtype=np.float32) # Example: force charging
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward_sum += reward
        if (step_num + 1) % (24 * int(60 / env.time_step_minutes)) == 0 or step_num == 0 or (step_num + 1) == num_test_steps:
            print(f"\n--- Step {info.get('current_step', step_num + 1)} --- Action: {action[0]:.2f}")
            print(f"  Time: {info.get('current_datetime')}, SoC: {obs['soc'][0]:.3f}")
            print(f"  Price: {info.get('current_price_ore_per_kwh',0):.2f} öre/kWh, Actual Bat Power: {info.get('actual_battery_terminal_power_kw',0):.2f} kW")
            print(f"  Grid Power: {info.get('grid_power_kw',0):.2f} kW, Step Cost: {info.get('step_cost_ore',0):.2f} öre")
            print(f"  Reward: {reward:.3f}, Total Reward: {total_reward_sum:.2f}")
            print(f"  Price Forecast (next 3h): {obs['price_forecast'][:3]}")
            if info.get('debug_prints'):
                env.render()

        if terminated or truncated:
            print(f"\nEpisode finished after {step_num + 1} steps.")
            break
    
    print(f"\nTotal reward over episode: {total_reward_sum:.2f}")
    print(f"Total grid cost over episode: {info.get('total_grid_cost_episode_ore', 0):.2f} öre")
    env.close() 