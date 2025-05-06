"""
Environment wrappers for the RL system.

Contains:
- LongTermEnv: Wrapper for HomeEnergyEnv that aggregates 4 hourly steps into one step
"""
import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, List, Optional, Any, Union
import datetime


class LongTermEnv(gym.Wrapper):
    """
    Wrapper for HomeEnergyEnv that groups 4 hourly steps into a single step.
    
    Designed for the long-term planner that operates on a 4-hour time scale.
    It handles:
    - State aggregation
    - Action translation (setting corridors/setpoints)
    - Reward aggregation
    """
    
    def __init__(
        self, 
        env: gym.Env,
        steps_per_action: int = 4,
        corridor_width: float = 0.1,  # SoC corridor width
        planning_horizon: int = 42  # 7 days * 6 steps per day
    ):
        """
        Initialize the wrapper.
        
        Args:
            env: Environment to wrap
            steps_per_action: Number of hourly steps per action (default: 4)
            corridor_width: Width of the SoC corridor as fraction of capacity
            planning_horizon: Number of future timesteps to plan for
        """
        super().__init__(env)
        self.steps_per_action = steps_per_action
        self.corridor_width = corridor_width
        self.planning_horizon = planning_horizon
        
        # Define new observation and action spaces
        self._setup_spaces()
        
        # State tracking
        self.current_corridor = {
            "lower": 0.3,  # Default corridor bounds
            "upper": 0.8
        }
        self.current_appliance_windows = {}
        
    def _setup_spaces(self) -> None:
        """Set up the observation and action spaces for the long-term planner."""
        # Original environment observation keys
        orig_obs_space = self.env.observation_space
        
        # Define the planning observation space
        # We aggregate multiple timesteps and add planning-specific observations
        self.observation_space = gym.spaces.Dict({
            # Original observation elements
            "soc": orig_obs_space["soc"],
            "time_idx": orig_obs_space["time_idx"],
            
            # Extended forecasts
            "price_forecast": gym.spaces.Box(
                low=0.0, high=10.0, shape=(self.planning_horizon,), dtype=np.float32
            ),
            "demand_forecast": gym.spaces.Box(
                low=0.0, high=30.0, shape=(self.planning_horizon,), dtype=np.float32
            ),
            
            # Current state of appliances
            "appliance_states": orig_obs_space["appliance_states"],
            
            # Summary of hourly observations within this step
            "hourly_data": gym.spaces.Box(
                low=-float('inf'), 
                high=float('inf'), 
                shape=(self.steps_per_action, 5),  # hour, price, demand, solar, net load
                dtype=np.float32
            )
        })
        
        # Number of appliances
        num_appliances = self.env.observation_space["appliance_states"].shape[0]
        
        # Action space for long-term planner:
        # - SoC corridors for each time step in the planning horizon
        # - Appliance activation windows
        self.action_space = gym.spaces.Dict({
            # SoC corridors (lower and upper bounds) for each planning timestep
            "soc_corridors": gym.spaces.Box(
                low=np.array([[0.2, 0.4]] * self.planning_horizon),  # [lower, upper] bounds
                high=np.array([[0.6, 0.9]] * self.planning_horizon),
                dtype=np.float32
            ),
            
            # Appliance windows (when each appliance can be turned on)
            # 1 = allowed, 0 = not allowed
            "appliance_windows": gym.spaces.Box(
                low=0, 
                high=1,
                shape=(self.planning_horizon, num_appliances), 
                dtype=np.int8
            )
        })
    
    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        """
        Reset the environment.
        
        Returns:
            tuple: (observation, info)
        """
        observation, info = self.env.reset(**kwargs)
        
        # Reset internal state tracking
        self.current_corridor = {"lower": 0.3, "upper": 0.8}
        
        # Convert the hourly observation to a 4-hour observation
        long_term_obs = self._aggregate_observations(observation)
        
        return long_term_obs, info
    
    def step(self, action: Dict) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Long-term planner action (corridors and windows)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Extract corridor settings for the current timestep
        # The action contains plans for the entire horizon, but we only use the first one now
        current_corridor = {
            "lower": action["soc_corridors"][0][0],  # First timestep, lower bound
            "upper": action["soc_corridors"][0][1]   # First timestep, upper bound
        }
        
        # Extract appliance windows for the current timestep
        current_appliance_windows = action["appliance_windows"][0]
        
        # Save for internal tracking
        self.current_corridor = current_corridor
        
        # Run multiple hourly steps and accumulate results
        total_reward = 0
        step_observations = []
        step_infos = []
        done = False
        truncated = False
        
        for i in range(self.steps_per_action):
            # Convert long-term plan to hourly action
            hourly_action = self._convert_to_hourly_action(
                current_corridor, 
                current_appliance_windows, 
                step_idx=i
            )
            
            # Execute hourly action
            obs, reward, term, trunc, info = self.env.step(hourly_action)
            
            # Store results
            step_observations.append(obs)
            step_infos.append(info)
            total_reward += reward
            
            # Check if environment terminated or truncated
            if term or trunc:
                done = term
                truncated = trunc
                break
        
        # Create the aggregated observation
        long_term_obs = self._aggregate_observations(step_observations[-1], step_observations, step_infos)
        
        # Aggregate info with additional planning-related information
        aggregated_info = self._aggregate_info(step_infos)
        
        return long_term_obs, total_reward, done, truncated, aggregated_info
    
    def _convert_to_hourly_action(self, corridor: Dict[str, float], 
                                  appliance_windows: np.ndarray,
                                  step_idx: int) -> np.ndarray:
        """
        Convert long-term plan to hourly action.
        
        Args:
            corridor: SoC corridor {"lower": float, "upper": float}
            appliance_windows: Allowed appliance activations
            step_idx: Current step index within the planning window
            
        Returns:
            np.ndarray: Hourly action for the base environment
        """
        # Get current SoC
        current_soc = self.env.unwrapped.battery.soc
        
        # Determine battery action based on corridor
        # If below corridor, charge; if above corridor, discharge; otherwise maintain
        if current_soc < corridor["lower"]:
            # Charge to reach corridor
            battery_action = 0.8  # Strong charging
        elif current_soc > corridor["upper"]:
            # Discharge to reach corridor
            battery_action = -0.8  # Strong discharging
        else:
            # Within corridor, small adjustment to stay centered
            target = (corridor["lower"] + corridor["upper"]) / 2
            # Proportional control
            battery_action = (target - current_soc) * 2.0  # Scale factor
        
        # Clip to valid range
        battery_action = max(-1.0, min(1.0, battery_action))
        
        # Get number of appliances
        num_appliances = len(self.env.unwrapped.appliance_names)
        
        # Determine appliance actions based on windows and current requests
        appliance_actions = np.zeros(num_appliances)
        
        # Get information about current appliance requests
        current_obs = self._get_current_observation()
        requested_appliances = current_obs.get("requested_appliances", 
                                              np.zeros(num_appliances))
        
        # Set appliance actions based on windows and requests
        for i in range(num_appliances):
            # If the appliance is allowed in this window and requested, turn it on
            if appliance_windows[i] > 0.5 and requested_appliances[i] > 0.5:
                appliance_actions[i] = 1.0
                
            # Apply additional logic for high-priority appliances
            appliance_name = self.env.unwrapped.appliance_names[i]
            appliance = self.env.unwrapped.appliance_manager.appliances[appliance_name]
            
            # Always allow high-priority appliances if requested
            if appliance.priority >= 8 and requested_appliances[i] > 0.5:
                appliance_actions[i] = 1.0
        
        # Combine battery and appliance actions
        hourly_action = np.concatenate([[battery_action], appliance_actions])
        
        return hourly_action
    
    def _get_current_observation(self) -> Dict:
        """
        Get the current observation from the base environment.
        
        Returns:
            dict: Current observation
        """
        # For simplicity, we generate a new observation
        # In a real implementation, you might want to store the most recent observation
        return self.env.unwrapped._get_observation()
    
    def _aggregate_observations(
        self, 
        last_obs: Dict, 
        all_observations: Optional[List[Dict]] = None,
        all_infos: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Aggregate hourly observations into a long-term observation.
        
        Args:
            last_obs: Last observation from the hourly environment
            all_observations: List of all observations in this step
            all_infos: List of all infos in this step
            
        Returns:
            dict: Aggregated observation
        """
        # If no observations list is provided, create a single-item list
        if all_observations is None:
            all_observations = [last_obs]
            
        if all_infos is None:
            all_infos = [{}] * len(all_observations)
            
        # Extract base information from the last observation
        aggregated = {
            "soc": last_obs["soc"],
            "time_idx": last_obs["time_idx"],
            "appliance_states": last_obs["appliance_states"],
        }
        
        # Extend forecasts to cover the planning horizon
        # We need to expand the 24-hour forecast to cover our planning horizon
        
        # For price forecast
        price_forecast = list(last_obs["price_forecast"])
        if len(price_forecast) < self.planning_horizon:
            # Simple extension with repeating pattern
            while len(price_forecast) < self.planning_horizon:
                # Add forecasted prices with a slight upward trend and noise
                last_day = price_forecast[-24:]
                trend_factor = 1.01  # Slight upward trend
                for p in last_day:
                    price_forecast.append(p * trend_factor * (1 + np.random.normal(0, 0.05)))
                    
        aggregated["price_forecast"] = np.array(price_forecast[:self.planning_horizon], dtype=np.float32)
        
        # For demand forecast
        demand_forecast = list(last_obs["demand_forecast"])
        if len(demand_forecast) < self.planning_horizon:
            # Simple extension with repeating pattern
            while len(demand_forecast) < self.planning_horizon:
                # Add forecasted demand with a slight variation
                last_day = demand_forecast[-24:]
                for d in last_day:
                    demand_forecast.append(d * (1 + np.random.normal(0, 0.1)))
                    
        aggregated["demand_forecast"] = np.array(demand_forecast[:self.planning_horizon], dtype=np.float32)
        
        # Create summary of hourly data within this step
        hourly_data = np.zeros((self.steps_per_action, 5), dtype=np.float32)
        
        for i, (obs, info) in enumerate(zip(all_observations, all_infos)):
            if i >= self.steps_per_action:
                break
                
            # Fill in hourly data with available information
            current_datetime = (self.env.unwrapped.start_datetime + 
                               datetime.timedelta(hours=self.env.unwrapped.current_hour - (len(all_observations) - i)))
            
            hourly_data[i, 0] = current_datetime.hour  # Hour of day
            
            # Add data from observation and info if available
            if i < len(all_observations) and "price_forecast" in obs:
                hourly_data[i, 1] = obs["price_forecast"][0]  # Current price
                
            if i < len(all_observations) and "demand_forecast" in obs:
                hourly_data[i, 2] = obs["demand_forecast"][0]  # Current demand
                
            if i < len(all_infos) and "solar_production" in info:
                hourly_data[i, 3] = info["solar_production"]  # Solar production
                
            if i < len(all_infos) and "net_consumption" in info:
                hourly_data[i, 4] = info["net_consumption"]  # Net load
        
        aggregated["hourly_data"] = hourly_data
        
        return aggregated
    
    def _aggregate_info(self, infos: List[Dict]) -> Dict:
        """
        Aggregate hourly infos into a long-term info dict.
        
        Args:
            infos: List of info dicts from hourly steps
            
        Returns:
            dict: Aggregated info
        """
        if not infos:
            return {}
            
        # Start with the last info dict
        aggregated_info = dict(infos[-1])
        
        # Add aggregated metrics
        aggregated_info["step_grid_costs"] = [info.get("grid_cost", 0.0) for info in infos]
        aggregated_info["step_battery_costs"] = [info.get("battery_cost", 0.0) for info in infos]
        aggregated_info["step_peak_penalties"] = [info.get("peak_penalty", 0.0) for info in infos]
        aggregated_info["step_comfort_bonuses"] = [info.get("comfort_bonus", 0.0) for info in infos]
        
        # Add aggregated totals
        aggregated_info["total_grid_cost"] = sum(aggregated_info["step_grid_costs"])
        aggregated_info["total_battery_cost"] = sum(aggregated_info["step_battery_costs"])
        aggregated_info["total_peak_penalty"] = sum(aggregated_info["step_peak_penalties"])
        aggregated_info["total_comfort_bonus"] = sum(aggregated_info["step_comfort_bonuses"])
        
        # Track corridor violations
        soc_values = [info.get("battery_soc", 0.5) for info in infos]
        corridor_lower = self.current_corridor["lower"]
        corridor_upper = self.current_corridor["upper"]
        
        lower_violations = sum(1 for soc in soc_values if soc < corridor_lower)
        upper_violations = sum(1 for soc in soc_values if soc > corridor_upper)
        
        aggregated_info["corridor_violations"] = lower_violations + upper_violations
        aggregated_info["corridor"] = self.current_corridor
        
        return aggregated_info 