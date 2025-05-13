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

from src.rl.components import Battery, ApplianceManager, SolarSystem

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default paths for prediction files
PRICE_PREDICTIONS_PATH = "src/predictions/prices/plots/predictions/merged"
SOLAR_PREDICTIONS_PATH = "src/predictions/solar/forecasted_data/merged_predictions.csv"

class HomeEnergyEnv(gym.Env):
    """
    Environment for controlling a home energy system on an hourly basis.
    
    Observation space contains:
    - Battery state of charge
    - Time index (hour of day, day of week)
    - Price forecast for next 24 hours
    - Demand forecast for next 24 hours
    - Appliance states
    
    Action space includes:
    - Battery charging/discharging rate
    - Appliance on/off toggles
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        price_model: Any = None,
        demand_model: Any = None,
        use_price_model: bool = False,
        use_demand_model: bool = False,
        battery_capacity: float = 22.0,  # kWh
        simulation_days: int = 7,
        peak_penalty_factor: float = 10.0,
        comfort_bonus_factor: float = 2.0,
        random_weather: bool = True,
        render_mode: Optional[str] = None,
        use_price_predictions: bool = False,
        use_solar_predictions: bool = False,
        price_predictions_path: str = PRICE_PREDICTIONS_PATH,
        solar_predictions_path: str = SOLAR_PREDICTIONS_PATH
    ):
        """
        Initialize the environment.
        
        Args:
            price_model: Optional model to predict electricity prices
            demand_model: Optional model to predict household demand
            use_price_model: Whether to use the price model or fixed prices
            use_demand_model: Whether to use the demand model or fixed demand
            battery_capacity: Battery capacity in kWh
            simulation_days: Length of simulation in days
            peak_penalty_factor: Penalty factor for peak loads
            comfort_bonus_factor: Bonus factor for satisfying appliance requests
            random_weather: Whether to use random weather patterns
            render_mode: Rendering mode
            use_price_predictions: Whether to use price predictions from CSV
            use_solar_predictions: Whether to use solar predictions from CSV
            price_predictions_path: Path to price predictions directory
            solar_predictions_path: Path to solar predictions file
        """
        super().__init__()
        
        self.price_model = price_model
        self.demand_model = demand_model
        self.use_price_model = use_price_model
        self.use_demand_model = use_demand_model
        self.peak_penalty_factor = peak_penalty_factor
        self.comfort_bonus_factor = comfort_bonus_factor
        self.random_weather = random_weather
        self.render_mode = render_mode
        
        # Prediction settings
        self.use_price_predictions = use_price_predictions
        self.use_solar_predictions = use_solar_predictions
        self.price_predictions_path = price_predictions_path
        self.solar_predictions_path = solar_predictions_path
        
        # Prediction data
        self.price_predictions_df = None
        self.solar_predictions_df = None
        
        # Time parameters
        self.simulation_hours = simulation_days * 24
        self.current_hour = 0
        self.start_datetime = None
        
        # Initialize components
        self.battery = Battery(capacity_kwh=battery_capacity)
        self.appliance_manager = ApplianceManager()
        self.solar_system = SolarSystem()
        
        # Action space: Battery control + appliance toggles
        num_appliances = len(self.appliance_manager.appliances)
        
        # Option 1: Continuous action space (battery) + discrete (appliances)
        # self.action_space = spaces.Dict({
        #     "battery": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),  # -1 to 1 for discharge/charge
        #     "appliances": spaces.MultiBinary(num_appliances)
        # })
        
        # Option 2: Combined Box
        # First value is battery control, rest are appliance toggles
        self.action_space = spaces.Box(
            low=np.array([-1.0] + [0.0] * num_appliances),
            high=np.array([1.0] + [1.0] * num_appliances),
            dtype=np.float32
        )
        
        # Observation space
        self.observation_space = spaces.Dict({
            "soc": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "time_idx": spaces.Box(
                low=np.array([0, 0]),  # hour of day, day of week
                high=np.array([23, 6]),  
                dtype=np.int32
            ),
            "price_forecast": spaces.Box(
                low=0.0, high=10.0, shape=(24,), dtype=np.float32
            ),
            "demand_forecast": spaces.Box(
                low=0.0, high=30.0, shape=(24,), dtype=np.float32
            ),
            "solar_forecast": spaces.Box(
                low=0.0, high=30.0, shape=(24,), dtype=np.float32
            ),
            "appliance_states": spaces.MultiBinary(num_appliances),
            "requested_appliances": spaces.MultiBinary(num_appliances)
        })
        
        # State variables
        self.price_history = []
        self.demand_history = []
        self.solar_history = []
        self.grid_cost_history = []
        self.battery_cost_history = []
        self.total_cost = 0.0
        self.peak_power = 0.0
        
        # Remember our appliance order
        self.appliance_names = sorted(self.appliance_manager.appliances.keys())
        
        # Load predictions if enabled
        if self.use_price_predictions:
            self._load_price_predictions()
        if self.use_solar_predictions:
            self._load_solar_predictions()
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Optional parameters
            
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        
        # Set start date to a random day or fixed day based on seed
        if seed is not None:
            np.random.seed(seed)
        
        # Reset time
        self.current_hour = 0
        current_year = datetime.datetime.now().year
        start_day = np.random.randint(1, 365 - self.simulation_hours // 24)
        self.start_datetime = datetime.datetime(current_year, 1, 1) + datetime.timedelta(days=start_day)
        
        # Reset components
        initial_soc = 0.5  # Start at 50% charge
        self.battery.reset(initial_soc=initial_soc)
        self.appliance_manager.reset()
        self.solar_system.reset()
        
        # Initialize price and demand forecasts for simulation period
        self._initialize_forecasts()
        
        # Reset history
        self.price_history = []
        self.demand_history = []
        self.solar_history = []
        self.grid_cost_history = []
        self.battery_cost_history = []
        self.total_cost = 0.0
        self.peak_power = 0.0
        
        # Generate initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: Union[np.ndarray, Dict]) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (battery control + appliance toggles)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Extract actions
        if isinstance(action, dict):
            battery_action = action["battery"][0]
            appliance_actions = action["appliances"]
        else:
            battery_action = action[0]
            appliance_actions = action[1:]
        
        # Process battery action (-1 to 1 -> actual charge/discharge rate)
        battery_result = self._handle_battery_action(battery_action)
        
        # Process appliance actions
        comfort_bonus = self._handle_appliance_actions(appliance_actions)
        
        # Current time features
        current_datetime = self.start_datetime + datetime.timedelta(hours=self.current_hour)
        
        # Get solar production for current hour
        if self.use_solar_predictions and len(self.solar_history) > self.current_hour:
            # Use pre-loaded solar predictions for this hour
            solar_production = self.solar_history[self.current_hour]
        else:
            # Use the simple model as a fallback
            current_hour_of_day = current_datetime.hour
            current_month = current_datetime.month
            weather_factor = np.random.uniform(0.3, 1.0) if self.random_weather else 0.8
            solar_production = self.solar_system.current_production(
                hour=current_hour_of_day,
                month=current_month,
                weather_factor=weather_factor
            )
        
        # Calculate energy balance
        appliance_consumption = self.appliance_manager.get_power_consumption()
        
        # Current electricity price (SEK/kWh)
        current_price = self.price_history[self.current_hour] 
        
        # Calculate grid exchange
        net_consumption = appliance_consumption - solar_production + battery_result["grid_charge"]
        
        # We pay for imports, get less for exports
        if net_consumption > 0:  # Importing from grid
            grid_cost = net_consumption * current_price
        else:  # Exporting to grid
            grid_cost = net_consumption * (current_price * 0.8)  # 80% of import price
            
        # Calculate peak power (for penalties)
        self.peak_power = max(self.peak_power, net_consumption)
        
        # Apply peak penalty if load is high
        peak_threshold = 10.0  # kW
        peak_penalty = 0.0
        if net_consumption > peak_threshold:
            peak_penalty = (net_consumption - peak_threshold) * current_price * self.peak_penalty_factor
            
        # Update cost histories
        self.grid_cost_history.append(grid_cost)
        self.battery_cost_history.append(battery_result["degradation_cost"])
        self.total_cost += grid_cost + battery_result["degradation_cost"] + peak_penalty - comfort_bonus
        
        # Update time
        self.current_hour += 1
        
        # Check if done
        done = self.current_hour >= self.simulation_hours
        
        # Update appliance state (automatic shutdowns etc.)
        self.appliance_manager.update(hours=1.0)
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward (negative cost is reward)
        reward = -(grid_cost + battery_result["degradation_cost"] + peak_penalty - comfort_bonus)
        
        # Get info
        info = self._get_info()
        info.update({
            "grid_cost": grid_cost,
            "battery_cost": battery_result["degradation_cost"],
            "peak_penalty": peak_penalty,
            "comfort_bonus": comfort_bonus,
            "net_consumption": net_consumption,
            "solar_production": solar_production,
            "appliance_consumption": appliance_consumption,
            "current_price": current_price
        })
        
        return observation, reward, done, False, info
    
    def _handle_battery_action(self, battery_action: float) -> Dict:
        """
        Process the battery action.
        
        Args:
            battery_action: Value between -1 and 1
            
        Returns:
            dict: Results of battery operation
        """
        # Convert action (-1 to 1) to charge/discharge rate
        charge_rate = self.battery.max_charge_rate
        discharge_rate = self.battery.max_discharge_rate
        
        if battery_action >= 0:  # Charging
            charge_amount = battery_action * charge_rate
            discharge_amount = 0.0
        else:  # Discharging
            charge_amount = 0.0
            discharge_amount = -battery_action * discharge_rate
        
        # Calculate actual charge/discharge
        actual_charged, charge_degradation = 0.0, 0.0
        actual_discharged, discharge_degradation = 0.0, 0.0
        
        if charge_amount > 0:
            actual_charged, charge_degradation = self.battery.charge(charge_amount)
        elif discharge_amount > 0:
            actual_discharged, discharge_degradation = self.battery.discharge(discharge_amount)
        
        # For grid calculation, charging is positive (consumption), discharging is negative (generation)
        grid_charge = actual_charged
        grid_discharge = -actual_discharged
        net_grid = grid_charge + grid_discharge
        
        # Total degradation cost
        degradation_cost = charge_degradation + discharge_degradation
        
        return {
            "grid_charge": net_grid,
            "actual_charged": actual_charged,
            "actual_discharged": actual_discharged,
            "degradation_cost": degradation_cost
        }
    
    def _handle_appliance_actions(self, appliance_actions: np.ndarray) -> float:
        """
        Process the appliance actions.
        
        Args:
            appliance_actions: Action values for each appliance
            
        Returns:
            float: Comfort bonus for satisfying user requests
        """
        comfort_bonus = 0.0
        
        # Process each appliance
        for i, name in enumerate(self.appliance_names):
            appliance = self.appliance_manager.appliances[name]
            desired_state = bool(appliance_actions[i] > 0.5)  # Convert to binary
            
            # Apply state change
            success = self.appliance_manager.set_state(name, desired_state)
            
            # Award comfort bonus if successfully activating a requested appliance
            if success and desired_state and appliance.pending_request:
                comfort_bonus += self.comfort_bonus_factor * appliance.priority
                appliance.pending_request = False  # Reset request
                
        return comfort_bonus
    
    def _initialize_forecasts(self) -> None:
        """Initialize price and demand forecasts for the simulation period."""
        # Initialize price forecasts
        self.price_history = []
        self.solar_history = []
        
        if self.use_price_predictions and self.price_predictions_df is not None:
            # If we have price predictions, use them
            self._initialize_price_forecasts_from_predictions()
        else:
            # Otherwise use the simple model
            self._initialize_price_forecasts_simple()
            
        # Initialize solar production forecasts
        if self.use_solar_predictions and self.solar_predictions_df is not None:
            # If we have solar predictions, use them
            self._initialize_solar_forecasts_from_predictions()
        else:
            # Otherwise use a simple model
            self._initialize_solar_forecasts_simple()
        
        # Initialize demand forecasts (could be extended to use real data)
        self._initialize_demand_forecasts_simple()
        
    def _initialize_price_forecasts_from_predictions(self) -> None:
        """Initialize price forecasts using the loaded predictions."""
        # Start with an empty list
        self.price_history = []
        
        # For each hour in the simulation
        for h in range(self.simulation_hours):
            # Calculate the timestamp for this hour
            timestamp = self.start_datetime + datetime.timedelta(hours=h)
            
            # Find the closest prediction in our dataframe
            if 'timestamp' in self.price_predictions_df.columns and 'price_prediction' in self.price_predictions_df.columns:
                # Convert timestamp to the same format as in the dataframe
                timestamp_str = timestamp.strftime('%Y-%m-%d %H:00:00')
                
                # Find exact match or closest time
                closest_idx = self.price_predictions_df['timestamp'].searchsorted(timestamp_str)
                if closest_idx >= len(self.price_predictions_df):
                    closest_idx = len(self.price_predictions_df) - 1
                
                # Get the price prediction
                price = self.price_predictions_df.iloc[closest_idx]['price_prediction']
                
                # Add some small random noise for variability
                price = price * (1 + np.random.normal(0, 0.05))
                
                # Append to history
                self.price_history.append(max(0.1, price))
            else:
                # Fallback to simple model if dataframe doesn't have expected columns
                self._initialize_price_forecasts_simple()
                return
    
    def _initialize_price_forecasts_simple(self) -> None:
        """Initialize price forecasts using a simple model."""
        # Simple price model with time-of-day and weekday/weekend patterns
        self.price_history = []
        for h in range(self.simulation_hours):
            datetime_h = self.start_datetime + datetime.timedelta(hours=h)
            hour_of_day = datetime_h.hour
            is_weekend = datetime_h.weekday() >= 5
            
            # Base price with time-of-day pattern
            if 6 <= hour_of_day <= 9 or 17 <= hour_of_day <= 20:  # Peak hours
                base_price = np.random.uniform(1.5, 2.5)
            else:
                base_price = np.random.uniform(0.8, 1.2)
                
            # Weekend discount
            if is_weekend:
                base_price *= 0.9
                
            # Add some noise
            price = base_price * (1 + np.random.normal(0, 0.1))
            self.price_history.append(max(0.1, price))
            
    def _initialize_solar_forecasts_from_predictions(self) -> None:
        """Initialize solar forecasts using the loaded predictions."""
        # Start with an empty list
        self.solar_history = []
        
        # For each hour in the simulation
        for h in range(self.simulation_hours):
            # Calculate the timestamp for this hour
            timestamp = self.start_datetime + datetime.timedelta(hours=h)
            
            # Find the closest prediction in our dataframe
            if 'timestamp' in self.solar_predictions_df.columns:
                # Convert timestamp to the same format as in the dataframe
                timestamp_str = timestamp.strftime('%Y-%m-%d %H:00:00')
                
                # Find exact match or closest time
                closest_idx = self.solar_predictions_df['timestamp'].searchsorted(timestamp_str)
                if closest_idx >= len(self.solar_predictions_df):
                    closest_idx = len(self.solar_predictions_df) - 1
                
                # Get the solar production (in kWh)
                if 'kilowatt_hours' in self.solar_predictions_df.columns:
                    solar = self.solar_predictions_df.iloc[closest_idx]['kilowatt_hours']
                elif 'watt_hours' in self.solar_predictions_df.columns:
                    solar = self.solar_predictions_df.iloc[closest_idx]['watt_hours'] / 1000  # Convert to kWh
                else:
                    # Fallback if no production column
                    solar = self._get_simple_solar_production(timestamp)
                
                # Add some small random noise for variability
                solar = solar * (1 + np.random.normal(0, 0.1))
                
                # Append to history
                self.solar_history.append(max(0.0, solar))
            else:
                # Fallback to simple model if dataframe doesn't have expected columns
                self._initialize_solar_forecasts_simple()
                return
    
    def _initialize_solar_forecasts_simple(self) -> None:
        """Initialize solar forecasts using a simple model."""
        self.solar_history = []
        for h in range(self.simulation_hours):
            timestamp = self.start_datetime + datetime.timedelta(hours=h)
            solar = self._get_simple_solar_production(timestamp)
            self.solar_history.append(solar)
    
    def _get_simple_solar_production(self, timestamp: datetime.datetime) -> float:
        """Get a simple estimate of solar production for a given timestamp."""
        hour = timestamp.hour
        month = timestamp.month
        
        # No production at night
        if hour < 6 or hour > 20:
            return 0.0
        
        # Seasonal factor (more production in summer)
        if 3 <= month <= 9:  # Spring and summer
            season_factor = np.random.uniform(0.7, 1.0)
        else:  # Fall and winter
            season_factor = np.random.uniform(0.2, 0.5)
        
        # Time of day factor (peak at noon)
        time_factor = 1.0 - abs(hour - 13) / 7.0
        
        # Base production with random weather impact
        base_production = 5.0  # kWh
        weather_factor = np.random.uniform(0.5, 1.0) if self.random_weather else 0.8
        
        return base_production * season_factor * time_factor * weather_factor
        
    def _initialize_demand_forecasts_simple(self) -> None:
        """Initialize demand forecasts using a simple model."""
        # Simple demand model
        self.demand_history = []
        for h in range(self.simulation_hours):
            datetime_h = self.start_datetime + datetime.timedelta(hours=h)
            hour_of_day = datetime_h.hour
            
            # Base demand with time-of-day pattern
            if 6 <= hour_of_day <= 9:  # Morning peak
                base_demand = np.random.uniform(3.0, 6.0)
            elif 17 <= hour_of_day <= 22:  # Evening peak
                base_demand = np.random.uniform(4.0, 7.0)
            elif 23 <= hour_of_day or hour_of_day <= 5:  # Night (low demand)
                base_demand = np.random.uniform(0.5, 2.0)
            else:  # Day (medium demand)
                base_demand = np.random.uniform(2.0, 4.0)
                
            # Add some noise
            demand = base_demand * (1 + np.random.normal(0, 0.2))
            self.demand_history.append(max(0.1, demand))
    
    def _get_observation(self) -> Dict:
        """
        Get the current observation.
        
        Returns:
            dict: Observation dictionary
        """
        # Current time features
        current_datetime = self.start_datetime + datetime.timedelta(hours=self.current_hour)
        hour_of_day = current_datetime.hour
        day_of_week = current_datetime.weekday()
        
        # Get price forecast for next 24 hours
        if self.current_hour + 24 <= len(self.price_history):
            price_forecast = self.price_history[self.current_hour:self.current_hour + 24]
        else:
            # Pad with last values if at end of simulation
            available = len(self.price_history) - self.current_hour
            price_forecast = self.price_history[self.current_hour:] + [self.price_history[-1]] * (24 - available)
        
        # Get demand forecast for next 24 hours
        if self.current_hour + 24 <= len(self.demand_history):
            demand_forecast = self.demand_history[self.current_hour:self.current_hour + 24]
        else:
            available = len(self.demand_history) - self.current_hour
            demand_forecast = self.demand_history[self.current_hour:] + [self.demand_history[-1]] * (24 - available)
            
        # Get solar forecast for next 24 hours
        if self.current_hour + 24 <= len(self.solar_history):
            solar_forecast = self.solar_history[self.current_hour:self.current_hour + 24]
        else:
            available = len(self.solar_history) - self.current_hour
            solar_forecast = self.solar_history[self.current_hour:] + [self.solar_history[-1]] * (24 - available)
        
        # Get appliance states
        appliance_states = self.appliance_manager.get_state_vector()
        
        # Simulate user requests (some random appliances are requested)
        requested_appliances = np.zeros_like(appliance_states)
        
        # Randomly set pending requests for some appliances
        if np.random.random() < 0.2:  # 20% chance to have a request in any given hour
            appliance_idx = np.random.randint(0, len(self.appliance_names))
            name = self.appliance_names[appliance_idx]
            self.appliance_manager.appliances[name].pending_request = True
        
        # Create requested_appliances vector
        for i, name in enumerate(self.appliance_names):
            requested_appliances[i] = float(self.appliance_manager.appliances[name].pending_request)
        
        return {
            "soc": np.array([self.battery.soc], dtype=np.float32),
            "time_idx": np.array([hour_of_day, day_of_week], dtype=np.int32),
            "price_forecast": np.array(price_forecast, dtype=np.float32),
            "demand_forecast": np.array(demand_forecast, dtype=np.float32),
            "solar_forecast": np.array(solar_forecast, dtype=np.float32),
            "appliance_states": appliance_states.astype(np.int8),
            "requested_appliances": requested_appliances.astype(np.int8)
        }
    
    def _get_info(self) -> Dict:
        """
        Get environment information.
        
        Returns:
            dict: Information dictionary
        """
        return {
            "total_cost": self.total_cost,
            "current_hour": self.current_hour,
            "battery_soc": self.battery.soc,
            "peak_power": self.peak_power
        }
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Returns:
            Optional[np.ndarray]: Rendered image (if render_mode is "rgb_array")
        """
        # TODO: Implement rendering
        if self.render_mode == "rgb_array":
            return np.zeros((400, 600, 3), dtype=np.uint8)
        return None
    
    def close(self) -> None:
        """Close the environment."""
        pass
    
    def _load_price_predictions(self) -> None:
        """Load price predictions from the most recent file."""
        try:
            # Find the most recent prediction directory
            prediction_dirs = sorted([d for d in os.listdir(self.price_predictions_path) 
                                     if os.path.isdir(os.path.join(self.price_predictions_path, d))])
            
            if not prediction_dirs:
                logger.warning(f"No prediction directories found in {self.price_predictions_path}")
                return
                
            latest_dir = os.path.join(self.price_predictions_path, prediction_dirs[-1])
            prediction_files = [f for f in os.listdir(latest_dir) if f.endswith('.csv')]
            
            if not prediction_files:
                logger.warning(f"No prediction files found in {latest_dir}")
                return
                
            # Load the most recent prediction file
            latest_file = sorted(prediction_files)[-1]
            file_path = os.path.join(latest_dir, latest_file)
            
            logger.info(f"Loading price predictions from {file_path}")
            self.price_predictions_df = pd.read_csv(file_path)
            
            # Ensure we have a timestamp column and convert to datetime
            if 'timestamp' in self.price_predictions_df.columns:
                self.price_predictions_df['timestamp'] = pd.to_datetime(self.price_predictions_df['timestamp'])
                
                # Look for the price prediction column
                if 'price_prediction' in self.price_predictions_df.columns:
                    # Successfully loaded
                    logger.info(f"Successfully loaded {len(self.price_predictions_df)} price predictions")
                else:
                    logger.warning("Price prediction column not found in file")
            else:
                logger.warning("Timestamp column not found in price predictions file")
                
        except Exception as e:
            logger.error(f"Error loading price predictions: {str(e)}")
            self.price_predictions_df = None
    
    def _load_solar_predictions(self) -> None:
        """Load solar predictions from file."""
        try:
            if not os.path.exists(self.solar_predictions_path):
                logger.warning(f"Solar predictions file not found at {self.solar_predictions_path}")
                return
                
            logger.info(f"Loading solar predictions from {self.solar_predictions_path}")
            self.solar_predictions_df = pd.read_csv(self.solar_predictions_path)
            
            # Ensure we have timestamp and energy production columns
            if 'timestamp' in self.solar_predictions_df.columns:
                self.solar_predictions_df['timestamp'] = pd.to_datetime(self.solar_predictions_df['timestamp'])
                
                # Check for watt_hours or kilowatt_hours column
                if 'kilowatt_hours' in self.solar_predictions_df.columns:
                    # Successfully loaded
                    logger.info(f"Successfully loaded {len(self.solar_predictions_df)} solar predictions")
                elif 'watt_hours' in self.solar_predictions_df.columns:
                    # Convert to kilowatt_hours
                    self.solar_predictions_df['kilowatt_hours'] = self.solar_predictions_df['watt_hours'] / 1000
                    logger.info(f"Successfully loaded {len(self.solar_predictions_df)} solar predictions (converted from watt_hours)")
                else:
                    logger.warning("Energy production column not found in solar predictions file")
            else:
                logger.warning("Timestamp column not found in solar predictions file")
                
        except Exception as e:
            logger.error(f"Error loading solar predictions: {str(e)}")
            self.solar_predictions_df = None 