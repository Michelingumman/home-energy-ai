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

from src.rl.components import Battery, ApplianceManager, SolarSystem


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
        render_mode: Optional[str] = None
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
        current_hour_of_day = current_datetime.hour
        current_month = current_datetime.month
        
        # Simulate solar generation
        weather_factor = np.random.uniform(0.3, 1.0) if self.random_weather else 0.8
        solar_production = self.solar_system.current_production(
            hour=current_hour_of_day,
            month=current_month,
            weather_factor=weather_factor
        )
        self.solar_history.append(solar_production)
        
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
            "appliance_consumption": appliance_consumption
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
        # TODO: Use actual models if provided
        
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