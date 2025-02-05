
import numpy as np
import tensorflow as tf
import requests
from typing import Dict, List



# components.py
class ComponentManager:
    def __init__(self, config):
        self.battery = Battery(**config['battery'])
        self.solar = SolarSystem(**config['solar'])
        self.grid = GridInterface(**config['grid'])
        self.appliances = ApplianceManager(**config['appliances'])
        
    def get_state(self):
        return {
            "battery": self.battery.state,
            "solar": self.solar.state,
            "grid": self.grid.state,
            "appliances": self.appliances.state
        }
        
        
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