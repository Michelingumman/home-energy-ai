import numpy as np
import tensorflow as tf
import requests
from typing import Dict, List
from datetime import datetime, timedelta
import gymnasium as gym


# components.py
class ComponentManager:
    def __init__(self, config):
        self.battery = Battery(**config['battery'])
        self.solar = SolarSystem(**config['solar'])
        self.grid = GridInterface(**config['grid'])
        self.appliances = ApplianceManager(**config['appliances'])
        self._time = datetime.now()
        self.time_step = timedelta(minutes=15)
        
    def get_observation_space(self):
        return gym.spaces.Dict({
            "battery_soc": gym.spaces.Box(0.0, 1.0, (1,)),
            "appliance_states": gym.spaces.MultiBinary(
                len(self.appliances.appliances)
            ),
            "solar_production": gym.spaces.Box(
                0.0, self.solar.capacity, (1,)
            ),
            "grid_price": gym.spaces.Box(0.0, 100.0, (1,)),
            "export_tariff": gym.spaces.Box(0.0, 1.0, (1,)),
            "time_encoding": gym.spaces.Box(-1.0, 1.0, (4,))
        })
        
    def execute_actions(self, action):
        # Battery handling
        if action['battery_mode'] == 0:
            self.battery.charge(self.grid.get_current_price(self._time.hour))
        elif action['battery_mode'] == 1:
            self.battery.discharge(
                self.appliances.total_power_demand()
            )
        elif action['battery_mode'] == 2:  # Sell to grid
            sell_energy = min(
                action['grid_sell'][0] * self.battery.capacity,
                self.battery.available_energy
            )
            self.battery.discharge(sell_energy)
            self.grid.log_sale(sell_energy, self._time.hour)
            
        # Appliance handling
        self.appliances.set_state({
            name: bool(action['appliance_control'][i])
            for i, name in enumerate(self.appliances.appliances)
        })
    
        # Time progression
        self._time += self.time_step
        self.grid.update_prices()
        
    def get_state(self):
        return {
            "battery_soc": np.array([self.battery.soc]),
            "appliance_states": np.array([
                int(app.active) 
                for app in self.appliances.appliances.values()
            ]),
            "solar_production": np.array([
                self.solar.current_production(
                    self._time, 
                    self.weather_service.get_cloud_cover()
                )
            ]),
            "grid_price": np.array([
                self.grid.get_current_price(self._time.hour)
            ]),
            "export_tariff": np.array([self.grid.export_tariff]),
            "time_encoding": np.array([
                np.sin(2 * np.pi * self._time.hour / 24),
                np.cos(2 * np.pi * self._time.hour / 24),
                np.sin(2 * np.pi * self._time.weekday() / 7),
                np.cos(2 * np.pi * self._time.weekday() / 7)
            ])
        }
    
    def reset(self):
        self._time = datetime.now()
        self.battery.reset()
        self.appliances.reset()
        self.grid.reset()
        return self.get_state()
        
        
class Appliance:
    def __init__(self, name, power_usage, priority, conflicts):
        self.name = name
        self.power = power_usage
        self.priority = priority
        self.conflicts = conflicts
        self.active = False

class ApplianceManager:
    def __init__(self, appliances_config: Dict):
        self.appliances = {
            name: Appliance(name, **config)
            for name, config in appliances_config.items()
        }
        self.active_set = set()

    def set_state(self, actions: Dict[str, bool]):
        # Sort by priority (descending)
        sorted_appliances = sorted(
            self.appliances.values(),
            key=lambda x: -x.priority
        )

        for appliance in sorted_appliances:
            if appliance.name in actions:
                desired_state = actions[appliance.name]
                
                if desired_state:
                    # Check conflicts
                    conflicts = any(
                        conflict in self.active_set
                        for conflict in appliance.conflicts
                    )
                    
                    if not conflicts:
                        appliance.active = True
                        self.active_set.add(appliance.name)
                else:
                    appliance.active = False
                    self.active_set.discard(appliance.name)

    def total_power_demand(self):
        return sum(
            app.power for app in self.appliances.values() 
            if app.active
        )

    def reset(self):
        for app in self.appliances.values():
            app.active = False
        self.active_set.clear()

    @property
    def state(self):
        return {name: app.active for name, app in self.appliances.items()}
                    
                    



class Battery:
    def __init__(self, capacity, soc_min, soc_max, degradation_factor=0.0001):
        self.capacity = capacity  # kWh
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.degradation_factor = degradation_factor
        self.soc = 0.5  # Initial state of charge
        self.cycle_count = 0
        self.total_degradation_cost = 0.0

    def charge(self, energy):
        available_space = (self.soc_max - self.soc) * self.capacity
        charge_energy = min(energy, available_space)
        self.soc += charge_energy / self.capacity
        self._update_degradation(charge_energy)
        return charge_energy

    def discharge(self, energy):
        available_energy = (self.soc - self.soc_min) * self.capacity
        discharge_energy = min(energy, available_energy)
        self.soc -= discharge_energy / self.capacity
        self._update_degradation(discharge_energy)
        return discharge_energy

    def _update_degradation(self, delta):
        cycle_depth = abs(delta) / self.capacity
        self.cycle_count += cycle_depth ** 1.5
        self.total_degradation_cost = self.cycle_count * self.capacity * self.degradation_factor

    def reset(self):
        self.soc = 0.5
        self.cycle_count = 0
        self.total_degradation_cost = 0.0

    @property
    def state(self):
        return {
            "soc": self.soc,
            "cycle_count": self.cycle_count,
            "degradation_cost": self.total_degradation_cost
        }
        
        
        


class SolarSystem:
    def __init__(self, capacity, efficiency=0.85, model_path=None):
        self.capacity = capacity  # kWp
        self.efficiency = efficiency
        self.model = tf.keras.models.load_model(model_path) if model_path else None

    def predict_production(self, weather_data):
        if self.model:
            return self.model.predict(np.expand_dims(weather_data, 0))[0]
        return np.zeros(24)

    def current_production(self, datetime, cloud_cover):
        hour_angle = (datetime.hour + datetime.minute/60) * 15 - 180
        solar_irradiance = max(0, np.sin(np.radians(hour_angle)) * 1000)
        return self.capacity * self.efficiency * solar_irradiance * (1 - cloud_cover) / 1000

    @property
    def state(self):
        return {"capacity": self.capacity, "efficiency": self.efficiency}
    
    


class GridInterface:
    def __init__(self, import_price_api, export_tariff):
        self.import_price_api = import_price_api
        self.export_tariff = export_tariff
        self.prices = np.zeros(24)
        
    def update_prices(self):
        try:
            response = requests.get(self.import_price_api)
            self.prices = np.array(response.json()["prices"])
        except:
            self.prices = np.zeros(24)

    def get_current_price(self, hour):
        return self.prices[hour % 24]

    def calculate_export_value(self, energy, hour):
        return energy * self.get_current_price(hour) * self.export_tariff

    @property
    def state(self):
        return {
            "current_price": self.get_current_price(0),
            "export_tariff": self.export_tariff
        }