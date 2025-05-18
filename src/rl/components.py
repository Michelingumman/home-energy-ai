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
        capacity_kwh: float,
        degradation_cost_per_kwh: float,
        initial_soc: float = 0.5,
        max_charge_power_kw: Optional[float] = None,
        max_discharge_power_kw: Optional[float] = None,
        charge_efficiency: float = 0.95,
        discharge_efficiency: float = 0.95
    ):
        """Initialize battery with given parameters.
        
        Args:
            capacity_kwh: Total battery capacity in kWh
            degradation_cost_per_kwh: Cost in öre per kWh (e.g., 45.0 for 45 öre/kWh)
            initial_soc: Initial state of charge (0.0-1.0)
            max_charge_power_kw: Maximum charging power in kW
            max_discharge_power_kw: Maximum discharging power in kW
            charge_efficiency: Battery charging efficiency
            discharge_efficiency: Battery discharging efficiency
        """
        self.capacity_kwh = capacity_kwh
        self.degradation_cost_per_kwh = degradation_cost_per_kwh  # öre/kWh
        
        # If max_charge_power_kw or max_discharge_power_kw are not provided, default to C/2 rate (full charge/discharge in 2 hours)
        self.max_charge_power_kw = max_charge_power_kw if max_charge_power_kw is not None else self.capacity_kwh / 2
        self.max_discharge_power_kw = max_discharge_power_kw if max_discharge_power_kw is not None else self.capacity_kwh / 2
        
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency

        if not (0.0 <= initial_soc <= 1.0):
            raise ValueError(f"Initial SoC must be between 0.0 and 1.0, got {initial_soc}")
        self.initial_soc = initial_soc
        self.soc = self.initial_soc
        self.current_kwh = self.capacity_kwh * self.soc

    def reset(self):
        self.soc = self.initial_soc
        self.current_kwh = self.capacity_kwh * self.soc

    def step(self, target_power_kw: float, duration_hours: float) -> Tuple[float, float, bool]:
        """
        Charges or discharges the battery based on target_power_kw.
        Args:
            target_power_kw: Power at the battery terminals. 
                             Positive to discharge (power flowing out of battery), 
                             negative to charge (power flowing into battery).
            duration_hours: Duration of the step.
        Returns:
            Tuple[float, float, bool]: 
                1. actual_power_kw_at_terminals: Actual power at battery terminals (kW).
                   Positive if discharging, negative if charging.
                2. energy_change_in_storage_kwh: Actual energy change in battery storage (kWh).
                   Positive if energy was added (charged), negative if energy was removed (discharged).
                3. limited_by_soc: Boolean flag indicating if the action was limited by SoC bounds.
        """
        energy_change_in_storage_kwh = 0.0
        actual_power_kw_at_terminals = 0.0
        limited_by_soc = False  # New flag to indicate if action was limited by SoC bounds

        if target_power_kw < 0:  # Attempting to charge
            # Power requested to flow INTO the battery (absolute value)
            requested_charge_power_kw = abs(target_power_kw)
            # Limit by max_charge_power_kw
            effective_charge_power_kw = min(requested_charge_power_kw, self.max_charge_power_kw)
            
            # Energy that would be added to storage if no SoC limit and 100% efficiency
            potential_energy_to_storage_kwh = effective_charge_power_kw * duration_hours
            
            # Account for charging efficiency: energy drawn from source > energy stored
            # If we want to store X kWh, we need to draw X / efficiency kWh from the source.
            # Here, `effective_charge_power_kw` is power at terminals, so energy to storage is `* efficiency`
            energy_added_to_storage_with_eff_kwh = potential_energy_to_storage_kwh * self.charge_efficiency

            # How much can we actually add due to SoC limit?
            available_capacity_kwh = self.capacity_kwh - self.current_kwh
            actual_energy_added_to_storage_kwh = min(energy_added_to_storage_with_eff_kwh, available_capacity_kwh)
            
            # Check if we're being limited by SoC (battery is near full)
            if actual_energy_added_to_storage_kwh < energy_added_to_storage_with_eff_kwh:
                limited_by_soc = True
            
            # Update SoC
            self.current_kwh += actual_energy_added_to_storage_kwh
            self.soc = self.current_kwh / self.capacity_kwh
            
            energy_change_in_storage_kwh = actual_energy_added_to_storage_kwh
            
            # Actual power at terminals (negative because charging)
            # This is the power that was effectively pushed into the battery from the outside.
            if duration_hours > 0 and self.charge_efficiency > 0:
                actual_power_kw_at_terminals = -(actual_energy_added_to_storage_kwh / self.charge_efficiency) / duration_hours
            elif duration_hours > 0: # Avoid division by zero if efficiency is zero
                actual_power_kw_at_terminals = -effective_charge_power_kw # Fallback, assumes perfect conversion if efficiency is 0
            else:
                actual_power_kw_at_terminals = 0.0

        elif target_power_kw > 0:  # Attempting to discharge
            # Power requested to flow OUT of the battery
            requested_discharge_power_kw = target_power_kw
            # Limit by max_discharge_power_kw
            effective_discharge_power_kw = min(requested_discharge_power_kw, self.max_discharge_power_kw)
            
            # Energy that would be delivered if no SoC limit and 100% efficiency
            potential_energy_from_storage_kwh = effective_discharge_power_kw * duration_hours
            
            # Account for discharging efficiency: energy drawn from storage > energy delivered
            # To deliver Y kWh, we need to draw Y / efficiency kWh from storage.
            # Here, `effective_discharge_power_kw` is power at terminals, so energy drawn from storage is ` / efficiency`
            if self.discharge_efficiency > 0:
                 energy_drawn_from_storage_for_eff_kwh = potential_energy_from_storage_kwh / self.discharge_efficiency
            else: # Avoid division by zero
                 energy_drawn_from_storage_for_eff_kwh = potential_energy_from_storage_kwh # Fallback

            # How much can we actually draw due to SoC limit?
            available_energy_in_storage_kwh = self.current_kwh
            actual_energy_drawn_from_storage_kwh = min(energy_drawn_from_storage_for_eff_kwh, available_energy_in_storage_kwh)
            
            # Check if we're being limited by SoC (battery is near empty)
            if actual_energy_drawn_from_storage_kwh < energy_drawn_from_storage_for_eff_kwh:
                limited_by_soc = True
            
            # Update SoC
            self.current_kwh -= actual_energy_drawn_from_storage_kwh
            self.soc = self.current_kwh / self.capacity_kwh
            
            energy_change_in_storage_kwh = -actual_energy_drawn_from_storage_kwh # Negative as it's removed from storage
            
            # Actual power at terminals (positive because discharging)
            # This is the power that was effectively delivered by the battery to the outside.
            if duration_hours > 0:
                actual_power_kw_at_terminals = (actual_energy_drawn_from_storage_kwh * self.discharge_efficiency) / duration_hours
            else:
                actual_power_kw_at_terminals = 0.0
        
        # Ensure SoC is not numerically unstable
        self.soc = np.clip(self.soc, 0.0, 1.0)
        self.current_kwh = np.clip(self.current_kwh, 0.0, self.capacity_kwh)

        return actual_power_kw_at_terminals, energy_change_in_storage_kwh, limited_by_soc

    # def charge(self, energy_kwh: float) -> float:
    #     """DEPRECATED: Use step method. Kept for compatibility if used elsewhere but not recommended.
    #     Charges the battery with a given amount of energy.
    #     Args:
    #         energy_kwh: Energy to charge in kWh.
    #     Returns:
    #         float: Actual energy charged in kWh.
    #     """
    #     # This method does not consider power limits or efficiency directly.
    #     # It's a simplified version. For proper simulation, use the step method.
    #     possible_charge = self.capacity_kwh - self.current_kwh
    #     actual_charge = min(energy_kwh, possible_charge)
    #     self.current_kwh += actual_charge
    #     self.soc = self.current_kwh / self.capacity_kwh
    #     return actual_charge

    # def discharge(self, energy_kwh: float) -> float:
    #     """DEPRECATED: Use step method. Kept for compatibility if used elsewhere but not recommended.
    #     Discharges the battery by a given amount of energy.
    #     Args:
    #         energy_kwh: Energy to discharge in kWh.
    #     Returns:
    #         float: Actual energy discharged in kWh.
    #     """
    #     # This method does not consider power limits or efficiency directly.
    #     # It's a simplified version. For proper simulation, use the step method.
    #     possible_discharge = self.current_kwh
    #     actual_discharge = min(energy_kwh, possible_discharge)
    #     self.current_kwh -= actual_discharge
    #     self.soc = self.current_kwh / self.capacity_kwh
    #     return actual_discharge

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



