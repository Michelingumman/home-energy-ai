"""
Components for the home energy system simulation.

Contains classes for:
- Battery: Battery storage system with charge/discharge capabilities
- Appliance: Individual appliances with power requirements
- ApplianceManager: Manages all household appliances
- SolarSystem: Solar panel system with production capabilities
"""
from typing import Dict, List, Tuple, Optional, Set
import numpy as np


class Battery:
    """Battery energy storage system with degradation modeling."""
    
    def __init__(
        self, 
        capacity_kwh: float = 22.0,
        min_soc: float = 0.1,
        max_soc: float = 0.9,
        max_cycles: int = 10000,
        degradation_cost_per_kwh: float = 45.0,  # Was 45.0, originally 0.45 (öre/kWh)
        max_charge_rate: float = 5.0,  # kW
        max_discharge_rate: float = 5.0,  # kW
        efficiency: float = 0.95
    ):
        """Initialize battery with given parameters.
        
        Args:
            capacity_kwh: Total battery capacity in kWh
            min_soc: Minimum state of charge (0.0-1.0)
            max_soc: Maximum state of charge (0.0-1.0)
            max_cycles: Expected battery lifetime in cycles
            degradation_cost_per_kwh: Cost in SEK per kWh for degradation
            max_charge_rate: Maximum charging power in kW
            max_discharge_rate: Maximum discharging power in kW
            efficiency: Battery round-trip efficiency
        """
        self.capacity_kwh = capacity_kwh
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.max_cycles = max_cycles
        self.degradation_cost_per_kwh = degradation_cost_per_kwh
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate
        self.efficiency = efficiency
        
        # State variables
        self.soc: float = (min_soc + max_soc) / 2  # Start at middle of allowed range
        self.cycle_count: float = 0.0
        self.energy_throughput: float = 0.0
        
    def charge(self, amount_kwh: float, hours: float = 1.0) -> Tuple[float, float]:
        """Charge the battery with the given amount of energy.
        
        Args:
            amount_kwh: Amount of energy to charge (kWh)
            hours: Time period in hours
            
        Returns:
            tuple: (actual_charged, degradation_cost)
        """
        # Validate against max charge rate
        max_possible = self.max_charge_rate * hours
        amount_kwh = min(amount_kwh, max_possible)
        
        # Calculate how much energy the battery can accept
        current_energy = self.soc * self.capacity_kwh
        max_energy = self.max_soc * self.capacity_kwh
        energy_headroom = max_energy - current_energy
        
        # Apply efficiency loss during charging
        effective_charge = min(amount_kwh * self.efficiency, energy_headroom)
        
        if effective_charge > 0:
            self.soc = (current_energy + effective_charge) / self.capacity_kwh
            self.energy_throughput += effective_charge
            degradation_cost = self.calc_degrade_cost(effective_charge)
            return effective_charge / self.efficiency, degradation_cost
        
        return 0.0, 0.0
    
    def discharge(self, amount_kwh: float, hours: float = 1.0) -> Tuple[float, float]:
        """Discharge the battery by the given amount of energy.
        
        Args:
            amount_kwh: Amount of energy to discharge (kWh)
            hours: Time period in hours
            
        Returns:
            tuple: (actual_discharged, degradation_cost)
        """
        # Validate against max discharge rate
        max_possible = self.max_discharge_rate * hours
        amount_kwh = min(amount_kwh, max_possible)
        
        # Calculate how much energy the battery can provide
        current_energy = self.soc * self.capacity_kwh
        min_energy = self.min_soc * self.capacity_kwh
        available_energy = current_energy - min_energy
        
        # Apply efficiency loss during discharging
        effective_discharge = min(amount_kwh, available_energy * self.efficiency)
        
        if effective_discharge > 0:
            energy_removed = effective_discharge / self.efficiency
            self.soc = (current_energy - energy_removed) / self.capacity_kwh
            self.energy_throughput += energy_removed
            degradation_cost = self.calc_degrade_cost(energy_removed)
            return effective_discharge, degradation_cost
        
        return 0.0, 0.0
    
    def calc_degrade_cost(self, energy_kwh: float) -> float:
        """Calculate the degradation cost for a given energy throughput.
        
        Args:
            energy_kwh: Energy throughput in kWh
            
        Returns:
            float: Degradation cost in SEK
        """
        # One full cycle is charging and discharging the full battery capacity
        cycle_fraction = energy_kwh / self.capacity_kwh
        self.cycle_count += cycle_fraction / 2  # Divided by 2 since we count separately for charge/discharge
        
        # Calculate degradation cost
        # Cost per cycle = total battery cost / max cycles
        # Cost per kWh = cost per cycle / capacity
        degradation_cost = energy_kwh * self.degradation_cost_per_kwh
        
        return degradation_cost
    
    def reset(self, initial_soc: Optional[float] = None) -> None:
        """Reset the battery to initial state.
        
        Args:
            initial_soc: Initial state of charge (default: middle of allowed range)
        """
        self.soc = initial_soc if initial_soc is not None else (self.min_soc + self.max_soc) / 2
        # Note: We don't reset cycle_count and energy_throughput as those are cumulative


class Appliance:
    """Represents a household appliance with power requirements and constraints."""
    
    def __init__(
        self, 
        name: str,
        power_kw: float,
        priority: int,
        min_runtime: float = 0.0,  # hours
        max_runtime: Optional[float] = None,  # hours
        conflicts: Optional[Set[str]] = None
    ):
        """Initialize an appliance.
        
        Args:
            name: Appliance name
            power_kw: Power consumption in kW
            priority: Priority level (higher = more important)
            min_runtime: Minimum runtime in hours once started
            max_runtime: Maximum runtime in hours
            conflicts: Set of appliance names that cannot run simultaneously
        """
        self.name = name
        self.power_kw = power_kw
        self.priority = priority
        self.min_runtime = min_runtime
        self.max_runtime = max_runtime
        self.conflicts = conflicts or set()
        
        # State variables
        self.is_on = False
        self.runtime = 0.0  # hours
        self.pending_request = False  # User request to start
    
    def turn_on(self) -> bool:
        """Turn on the appliance if possible.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_on:
            self.is_on = True
            return True
        return False
    
    def turn_off(self) -> bool:
        """Turn off the appliance if possible.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.is_on and self.runtime >= self.min_runtime:
            self.is_on = False
            self.runtime = 0.0
            return True
        return False
    
    def update(self, hours: float = 1.0) -> None:
        """Update appliance state for the given time period.
        
        Args:
            hours: Time period in hours
        """
        if self.is_on:
            self.runtime += hours
            
            # Auto-shutdown if max runtime exceeded
            if self.max_runtime is not None and self.runtime >= self.max_runtime:
                self.is_on = False
                self.runtime = 0.0


class ApplianceManager:
    """Manages all household appliances and their interactions."""
    
    def __init__(self):
        """Initialize the appliance manager."""
        # Simplified: No appliances
        self.appliances: Dict[str, Appliance] = {}
        # self.setup_appliances() # Commented out to have no appliances by default
    
    def setup_appliances(self) -> None:
        """Set up default appliances for the household."""
        # Commented out all appliance creation for simplification
        # self.add_appliance(Appliance(
        #     name="heat_pump", 
        #     power_kw=3.0, 
        #     priority=9,
        #     min_runtime=0.5
        # ))
        # 
        # self.add_appliance(Appliance(
        #     name="floor_heating", 
        #     power_kw=4.5, 
        #     priority=8,
        #     conflicts={"heat_pump"}
        # ))
        # 
        # self.add_appliance(Appliance(
        #     name="ev_charger", 
        #     power_kw=11.0, 
        #     priority=5,
        #     conflicts={"sauna"}
        # ))
        # 
        # self.add_appliance(Appliance(
        #     name="sauna", 
        #     power_kw=9.0, 
        #     priority=3,
        #     max_runtime=3.0,
        #     conflicts={"floor_heating", "ev_charger", "heat_pump"}
        # ))
        # 
        # # Add more common appliances with lower power consumption
        # self.add_appliance(Appliance(name="dishwasher", power_kw=1.2, priority=4, min_runtime=1.0))
        # self.add_appliance(Appliance(name="washing_machine", power_kw=1.5, priority=4, min_runtime=1.0))
        # self.add_appliance(Appliance(name="dryer", power_kw=2.5, priority=2, min_runtime=0.5))
        pass # No appliances in simplified version
    
    def add_appliance(self, appliance: Appliance) -> None:
        """Add an appliance to the manager.
        
        Args:
            appliance: Appliance object to add
        """
        self.appliances[appliance.name] = appliance
    
    def set_state(self, appliance_name: str, state: bool) -> bool:
        """Set the state of an appliance.
        
        Args:
            appliance_name: Name of the appliance to control
            state: True to turn on, False to turn off
            
        Returns:
            bool: True if successful, False if constraints prevent the change
        """
        # Simplified: Always return False as there are no appliances to control
        # if appliance_name in self.appliances:
        #     app = self.appliances[appliance_name]
        #     if state:
        #         # Check for conflicts before turning on
        #         if self._has_conflicts(appliance_name):
        #             return False
        #         return app.turn_on()
        #     else:
        #         return app.turn_off()
        return False
    
    def _has_conflicts(self, appliance_name: str) -> bool:
        """Check if turning on an appliance would create conflicts.
        
        Args:
            appliance_name: Name of the appliance to check
            
        Returns:
            bool: True if there are conflicts, False otherwise
        """
        # Simplified: Always return False as there are no appliances to conflict with
        # if appliance_name not in self.appliances or not self.appliances[appliance_name].conflicts:
        #     return False
        # 
        # for conflict_name in self.appliances[appliance_name].conflicts:
        #     if conflict_name in self.appliances and self.appliances[conflict_name].is_on:
        #         return True
        return False
    
    def get_state_vector(self) -> np.ndarray:
        """Get the state of all appliances as a vector.
        
        Returns:
            np.ndarray: Vector of appliance states (0 = off, 1 = on)
        """
        # Simplified: Return empty array as there are no appliances
        # states = [int(app.is_on) for app in self.appliances.values()]
        # return np.array(states, dtype=np.int8)
        return np.array([], dtype=np.int8)
    
    def get_power_consumption(self) -> float:
        """Calculate the total power consumption of all running appliances.
        
        Returns:
            float: Total power consumption in kW
        """
        # Simplified: Always return 0 as there are no appliances
        # total_power = sum(app.power_kw for app in self.appliances.values() if app.is_on)
        # return total_power
        return 0.0
    
    def reset(self) -> None:
        """Reset all appliances to their default state."""
        # Simplified: If self.appliances is empty, this loop does nothing.
        for app in self.appliances.values():
            app.is_on = False
            app.runtime = 0.0
            app.pending_request = False
        # Or simply: pass
    
    def update(self, hours: float = 1.0) -> None:
        """Update all appliances for the given time period.
        
        Args:
            hours: Time period in hours
        """
        # Simplified: If self.appliances is empty, this loop does nothing.
        for app in self.appliances.values():
            app.update(hours)
        # Or simply: pass


class SolarSystem:
    """Represents the solar panel system."""
    
    def __init__(self, peak_capacity_kw: float = 5.0, efficiency: float = 0.9):
        self.peak_capacity_kw = peak_capacity_kw
        self.efficiency = efficiency
        # self.current_weather_factor = 1.0 # Commented out, not used in simplified version

    def reset(self):
        # self.current_weather_factor = np.random.uniform(0.7, 1.0) # Commented out
        pass # No specific state to reset for always-zero production

    def current_production(
        self,
        hour: int = 0,          # Default values added for simplified interface
        month: int = 1,         # Default values added for simplified interface
        weather_factor: Optional[float] = None # Kept for interface compatibility
    ) -> float:
        """Calculate current solar production in kW."""
        # Simplified to always return 0 for the basic environment
        return 0.0
        
        # # More complex model (commented out for simplification):
        # if weather_factor is None:
        #     # If no specific weather factor is given, use a random one for variability
        #     weather_factor = np.random.uniform(0.3, 1.0) # Simulate cloud cover, etc.
        
        # # Basic sinusoidal model for daily production
        # # No sun at night (adjust hours for your location/season)
        # if not (6 <= hour <= 18):
        #     return 0.0
        
        # # Normalized time from sunrise (e.g., 6 AM) to sunset (e.g., 6 PM)
        # t_norm = (hour - 6) / 12.0 # Assuming a 12-hour solar day for simplicity
        
        # # Sinusoidal production curve peaking at noon
        # base_production = self.peak_capacity_kw * np.sin(t_norm * np.pi)
        
        # # Seasonal adjustment (very rough example)
        # seasonal_factor = 1.0
        # if month in [12, 1, 2]: # Winter
        #     seasonal_factor = 0.4
        # elif month in [6, 7, 8]: # Summer
        #     seasonal_factor = 1.0 # Peak production
        # elif month in [3, 4, 5]: # Spring
        #     seasonal_factor = 0.8
        # else: # Autumn
        #     seasonal_factor = 0.6
            
        # production = base_production * seasonal_factor * weather_factor * self.efficiency
        # return max(0, production) # Ensure production is not negative

# Remove duplicate class definitions that start here (IF THEY EXISTED - this is a safeguard)
# # class Appliance:
# #     \"\"\"Represents a household appliance with power requirements and constraints.\"\"\"
# #     ...
# # 
# # class ApplianceManager:
# #     \"\"\"Manages multiple appliances, their states, and power consumption.\"\"\"
# #     ... 