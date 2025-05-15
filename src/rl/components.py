"""
Components for the home energy system simulation.

Contains classes for:
- Battery: Battery storage system with charge/discharge capabilities
- Appliance: Individual appliances with power requirements
- ApplianceManager: Manages all household appliances
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
        degradation_cost_per_kwh: float = 45.0,  # Cost in öre per kWh 
        max_charge_rate: float = 5.0,  # kW
        max_discharge_rate: float = 10.0,  # kW
        efficiency: float = 0.95
    ):
        """Initialize battery with given parameters.
        
        Args:
            capacity_kwh: Total battery capacity in kWh
            min_soc: Minimum state of charge (0.0-1.0)
            max_soc: Maximum state of charge (0.0-1.0)
            max_cycles: Expected battery lifetime in cycles
            degradation_cost_per_kwh: Cost in öre per kWh (e.g., 45.0 for 45 öre/kWh)
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
        degradation_cost = energy_kwh * self.degradation_cost_per_kwh # This is now energy_kwh * öre_cost_per_kwh
        
        return degradation_cost
    
    def reset(self, initial_soc: Optional[float] = None) -> None:
        """Reset the battery to initial state.
        
        Args:
            initial_soc: Initial state of charge (default: middle of allowed range)
        """
        self.soc = initial_soc if initial_soc is not None else (self.min_soc + self.max_soc) / 2
        # Note: We don't reset cycle_count and energy_throughput as those are cumulative

# Remove duplicate class definitions that start here (IF THEY EXISTED - this is a safeguard)
# # class Appliance:
# #     \"\"\"Represents a household appliance with power requirements and constraints.\"\"\"
# #     ...
# # 
# # class ApplianceManager:
# #     \"\"\"Manages multiple appliances, their states, and power consumption.\"\"\"
# #     ... 