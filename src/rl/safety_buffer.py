"""
Safety buffer implementation for home energy control.

This module contains functions that help ensure the battery state of charge
stays within safe operational limits.
"""

from typing import Tuple, Dict, List, Optional
import numpy as np

def ensure_soc_limits(
    soc: float, 
    action: float, 
    min_soc: float = 0.2, 
    max_soc: float = 0.8,
    max_charge_power_kw: float = 5.0,
    max_discharge_power_kw: float = 5.0,
    capacity_kwh: float = 22.0,
    time_step_hours: float = 0.25
) -> float:
    """
    Adjust action to ensure SoC stays within safe limits.
    
    Args:
        soc: Current battery state of charge (0-1)
        action: Raw action from agent (-1 to 1)
        min_soc: Minimum allowed SoC (e.g., 0.2)
        max_soc: Maximum allowed SoC (e.g., 0.8)
        max_charge_power_kw: Maximum battery charging power in kW
        max_discharge_power_kw: Maximum battery discharging power in kW
        capacity_kwh: Battery capacity in kWh
        time_step_hours: Time step duration in hours
        
    Returns:
        float: Safe action that won't violate SoC constraints
    """
    # If already at or below min_soc, prevent further discharge
    if soc <= min_soc and action > 0:  # Discharging
        return 0.0
    
    # If already at or above max_soc, prevent further charging
    if soc >= max_soc and action < 0:  # Charging
        return 0.0
    
    # For actions in safe range, compute exact limits
    if action < 0:  # Charging (negative value)
        # Calculate energy headroom and max safe charge power
        energy_headroom_kwh = (max_soc - soc) * capacity_kwh
        max_safe_charge_kw = min(max_charge_power_kw, energy_headroom_kwh / time_step_hours)
        
        # Scale action if needed
        requested_charge_kw = -action * max_charge_power_kw
        if requested_charge_kw > max_safe_charge_kw:
            return -max_safe_charge_kw / max_charge_power_kw
    
    elif action > 0:  # Discharging (positive value)
        # Calculate available energy and max safe discharge power
        available_energy_kwh = (soc - min_soc) * capacity_kwh
        max_safe_discharge_kw = min(max_discharge_power_kw, available_energy_kwh / time_step_hours)
        
        # Scale action if needed
        requested_discharge_kw = action * max_discharge_power_kw
        if requested_discharge_kw > max_safe_discharge_kw:
            return max_safe_discharge_kw / max_discharge_power_kw
    
    # If no adjustments needed, return original action
    return action 