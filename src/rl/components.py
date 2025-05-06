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
        min_soc: float = 0.2,
        max_soc: float = 0.9,
        max_cycles: int = 10000,
        degradation_cost_per_kwh: float = 0.45,
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
        self.appliances: Dict[str, Appliance] = {}
        self.setup_appliances()
    
    def setup_appliances(self) -> None:
        """Set up default appliances for the household."""
        # Create common household appliances with realistic power values
        self.add_appliance(Appliance(
            name="heat_pump", 
            power_kw=3.0, 
            priority=9,
            min_runtime=0.5
        ))
        
        self.add_appliance(Appliance(
            name="floor_heating", 
            power_kw=4.5, 
            priority=8,
            conflicts={"sauna"}
        ))
        
        self.add_appliance(Appliance(
            name="ev_charger", 
            power_kw=11.0, 
            priority=5,
            conflicts={"sauna"}
        ))
        
        self.add_appliance(Appliance(
            name="sauna", 
            power_kw=9.0, 
            priority=3,
            max_runtime=3.0,
            conflicts={"floor_heating", "ev_charger"}
        ))
        
        # Add more common appliances with lower power consumption
        self.add_appliance(Appliance(name="dishwasher", power_kw=1.2, priority=4, min_runtime=1.0))
        self.add_appliance(Appliance(name="washing_machine", power_kw=1.5, priority=4, min_runtime=1.0))
        self.add_appliance(Appliance(name="dryer", power_kw=2.5, priority=2, min_runtime=0.5))
    
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
        if appliance_name not in self.appliances:
            return False
        
        appliance = self.appliances[appliance_name]
        
        if state:  # Turning on
            # Check if turning on would create conflicts
            if self._has_conflicts(appliance_name):
                return False
            
            return appliance.turn_on()
        else:  # Turning off
            return appliance.turn_off()
    
    def _has_conflicts(self, appliance_name: str) -> bool:
        """Check if turning on an appliance would create conflicts.
        
        Args:
            appliance_name: Name of the appliance to check
            
        Returns:
            bool: True if there are conflicts, False otherwise
        """
        appliance = self.appliances[appliance_name]
        
        for conflict in appliance.conflicts:
            if conflict in self.appliances and self.appliances[conflict].is_on:
                return True
                
        # Check if other running appliances conflict with this one
        for name, other in self.appliances.items():
            if other.is_on and appliance_name in other.conflicts:
                return True
                
        return False
    
    def get_state_vector(self) -> np.ndarray:
        """Get the state of all appliances as a vector.
        
        Returns:
            np.ndarray: Vector of appliance states (0 = off, 1 = on)
        """
        # Sort by name for consistent ordering
        names = sorted(self.appliances.keys())
        return np.array([float(self.appliances[name].is_on) for name in names])
    
    def get_power_consumption(self) -> float:
        """Calculate the total power consumption of all running appliances.
        
        Returns:
            float: Total power consumption in kW
        """
        return sum(app.power_kw for app in self.appliances.values() if app.is_on)
    
    def reset(self) -> None:
        """Reset all appliances to their default state."""
        for appliance in self.appliances.values():
            appliance.is_on = False
            appliance.runtime = 0.0
            appliance.pending_request = False
    
    def update(self, hours: float = 1.0) -> None:
        """Update all appliances for the given time period.
        
        Args:
            hours: Time period in hours
        """
        for appliance in self.appliances.values():
            appliance.update(hours)


class SolarSystem:
    """Represents a solar panel system with production capabilities."""
    
    def __init__(
        self, 
        capacity_kw: float = 10.0,
        latitude: float = 59.33,  # Stockholm latitude
        efficiency: float = 0.85
    ):
        """Initialize the solar system.
        
        Args:
            capacity_kw: Maximum capacity in kW
            latitude: Geographical latitude for solar calculations
            efficiency: System efficiency
        """
        self.capacity_kw = capacity_kw
        self.latitude = latitude
        self.efficiency = efficiency
        
        # Simple solar model parameters
        self._hour_weights = np.zeros(24)
        self._setup_hour_weights()
        
        # Season weights (approximate for Northern Europe)
        self._month_weights = {
            1: 0.2,   # January
            2: 0.3,   # February
            3: 0.5,   # March
            4: 0.7,   # April
            5: 0.85,  # May
            6: 0.95,  # June
            7: 0.9,   # July
            8: 0.8,   # August
            9: 0.6,   # September
            10: 0.4,  # October
            11: 0.25, # November
            12: 0.15  # December
        }
        
        # Internal state for weather variations
        self._weather_factor = 1.0  # Will vary to simulate weather
        
    def _setup_hour_weights(self) -> None:
        """Set up hourly production weights based on time of day."""
        # Simple bell curve centered at noon
        for h in range(24):
            if 5 <= h <= 21:  # Daylight hours (simplified)
                # Use a bell curve centered at 13 (1 PM)
                self._hour_weights[h] = np.exp(-0.5 * ((h - 13) / 3.5) ** 2)
            else:
                self._hour_weights[h] = 0.0
    
    def current_production(self, hour: int, month: int, weather_factor: Optional[float] = None) -> float:
        """Calculate the current solar production.
        
        Args:
            hour: Hour of the day (0-23)
            month: Month of the year (1-12)
            weather_factor: Weather influence factor (0.0-1.0)
            
        Returns:
            float: Current production in kW
        """
        if weather_factor is not None:
            self._weather_factor = max(0.0, min(1.0, weather_factor))
        
        hour_factor = self._hour_weights[hour % 24]
        month_factor = self._month_weights.get(month, 0.5)
        
        production = self.capacity_kw * hour_factor * month_factor * self._weather_factor
        return production * self.efficiency
    
    def predict_production(self, hours: List[int], months: List[int], 
                           weather_factors: Optional[List[float]] = None) -> List[float]:
        """Predict solar production for multiple timepoints.
        
        Args:
            hours: List of hours (0-23)
            months: List of months (1-12)
            weather_factors: Optional list of weather factors
            
        Returns:
            List[float]: Predicted production in kW for each timepoint
        """
        if weather_factors is None:
            # Generate random weather factors with some correlation
            base = np.random.normal(0.8, 0.2)
            weather_factors = [max(0.1, min(1.0, base + np.random.normal(0, 0.1))) 
                              for _ in range(len(hours))]
        
        return [self.current_production(h, m, w) 
                for h, m, w in zip(hours, months, weather_factors)]
    
    def reset(self) -> None:
        """Reset the solar system's internal state."""
        self._weather_factor = 1.0 