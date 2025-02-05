import gymnasium as gym
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple
from datetime import datetime, timedelta
from components import ComponentManager

class HomeEnergyEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        
        # Load TensorFlow models
        self.demand_model = tf.keras.models.load_model(config['demand_model_path'])
        self.price_model = tf.keras.models.load_model(config['price_model_path'])
        
        # Initialize components
        self.components = ComponentManager(config)
        
        # Action Space (Modified for flexibility)
        self.action_space = gym.spaces.Dict({
            'battery_mode': gym.spaces.Discrete(3),  # 0=charge, 1=discharge, 2=hold
            'appliance_control': gym.spaces.MultiBinary(
                len(self.components.appliances.appliances)
            ),
            'grid_sell': gym.spaces.Box(low=0.0, high=1.0, shape=(1,))
        })
        
        # Observation Space (Unified interface)
        self.observation_space = self.components.get_observation_space()

    def step(self, action):
        # Execute component actions
        self._execute_actions(action)
        
        # Get new state
        obs = self.components.get_state()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination
        done = self._check_done()
        
        return obs, reward, done, {}

    def _handle_battery_action(self, action):
        battery_action = action["battery"]
        if battery_action == 0:  # Charge from grid
            self.battery.charge(self.battery.capacity * 0.1)  # Charge 10% capacity
        elif battery_action == 1:  # Discharge to home
            self.battery.discharge(self.appliance_manager.total_power_demand())
        elif battery_action == 3:  # Sell to grid
            sell_amount = action["grid_export"][0] * self.battery.capacity
            self.battery.discharge(sell_amount)

    def _handle_appliance_action(self, action):
        appliance_actions = {
            name: bool(action["appliances"][i])
            for i, name in enumerate(self.appliance_manager.appliances)
        }
        self.appliance_manager.set_state(appliance_actions)

    def _handle_grid_export(self, action):
        if action["battery"] == 3:
            sell_amount = action["grid_export"][0] * self.battery.capacity
            self.grid.calculate_export_value(sell_amount, self.current_time.hour)

    def _calculate_reward(self):
        return (
            self.components.grid.sales_profit
            - self.components.battery.degradation_cost
            - self.components.grid.import_cost
            - self._peak_penalty()
        )

    def reset(self):
        self.battery.reset()
        self.appliance_manager.reset()
        self.current_time = self.start_time
        self.grid.update_prices()
        return self._get_observation()

    def _get_observation(self):
        hour = self.current_time.hour
        weekday = self.current_time.weekday()
        
        return {
            "battery_soc": np.array([self.battery.soc]),
            "appliance_states": np.array([
                int(app.active) 
                for app in self.appliance_manager.appliances.values()
            ]),
            "solar_production": np.array([self.solar.current_production(
                self.current_time, 0.2  # Assume 20% cloud cover
            )]),
            "grid_price": np.array([self.grid.get_current_price(hour)]),
            "export_tariff": np.array([self.grid.export_tariff]),
            "time_encoding": np.array([
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * weekday / 7),
                np.cos(2 * np.pi * weekday / 7)
            ])
        }

    def render(self, mode='human'):
        state = self.components.get_state()
        print(f"\nTime: {self.components.current_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"Battery SOC: {state['battery_soc'][0]:.1%}")
        print(f"Solar Production: {state['solar_production'][0]:.1f} kW")
        print(f"Grid Price: {state['grid_price'][0]:.2f} öre/kWh")
        print(f"Active Appliances: {self.components.appliances.active_set}")