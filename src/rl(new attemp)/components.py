"""
Battery component for the Home Energy RL Environment.
"""
import logging

logger = logging.getLogger(__name__)

class Battery:
    """
    Simulates a battery with charging/discharging dynamics and efficiencies.
    """
    def __init__(
        self,
        capacity_kwh: float,
        initial_soc: float,
        max_charge_power_kw: float,
        max_discharge_power_kw: float,
        charge_efficiency: float,
        discharge_efficiency: float,
        min_soc_limit: float = 0.0, # Min SoC based on battery chemistry/safety
        max_soc_limit: float = 1.0  # Max SoC based on battery chemistry/safety
    ):
        self.capacity_kwh = capacity_kwh
        self.initial_soc = initial_soc
        self.max_charge_power_kw = abs(max_charge_power_kw)  # Ensure positive
        self.max_discharge_power_kw = abs(max_discharge_power_kw) # Ensure positive
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.min_soc_limit = min_soc_limit
        self.max_soc_limit = max_soc_limit

        if not (0.0 <= self.initial_soc <= 1.0):
            raise ValueError(f"Initial SoC must be between 0 and 1, got {self.initial_soc}")
        if not (0.0 <= self.min_soc_limit < self.max_soc_limit <= 1.0):
            raise ValueError(
                f"SoC limits invalid: min_soc_limit={min_soc_limit}, max_soc_limit={max_soc_limit}"
            )
        
        self.soc = self.initial_soc
        self.current_energy_kwh = self.soc * self.capacity_kwh

        logger.info(
            f"Battery initialized: Capacity={capacity_kwh:.2f} kWh, Initial SoC={initial_soc:.2f}, "
            f"Max Charge={self.max_charge_power_kw:.2f} kW, Max Discharge={self.max_discharge_power_kw:.2f} kW, "
            f"Eff_charge={charge_efficiency:.2f}, Eff_discharge={discharge_efficiency:.2f}, "
            f"SoC Limits=[{self.min_soc_limit:.2f}, {self.max_soc_limit:.2f}]"
        )

    def reset(self, initial_soc: float = None) -> None:
        """Resets the battery to its initial or a specified state."""
        if initial_soc is not None:
            if not (self.min_soc_limit <= initial_soc <= self.max_soc_limit):
                 logger.warning(f"Provided initial_soc {initial_soc} is outside effective limits [{self.min_soc_limit}, {self.max_soc_limit}]. Clipping.")
                 initial_soc = max(self.min_soc_limit, min(initial_soc, self.max_soc_limit))
            self.soc = initial_soc
        else:
            self.soc = self.initial_soc
        self.current_energy_kwh = self.soc * self.capacity_kwh
        logger.debug(f"Battery reset. New SoC: {self.soc:.3f}")

    def step(self, target_terminal_power_kw: float, duration_hours: float) -> tuple[float, float, bool]:
        """
        Processes a charge/discharge request for a given duration.

        Args:
            target_terminal_power_kw: Desired power at the battery terminals.
                                     Negative for charging, positive for discharging.
            duration_hours: Duration of the power application in hours.

        Returns:
            A tuple containing:
            - actual_terminal_power_kw (float): The actual power at the battery terminals (kW).
                                               Negative for charging, positive for discharging.
            - energy_change_in_storage_kwh (float): The actual change in stored energy (kWh).
                                                    Positive for energy increase (charging),
                                                    negative for energy decrease (discharging).
            - limited_by_soc_or_power (bool): True if the operation was constrained by SoC limits
                                              or power limits.
        """
        limited_by_soc_or_power = False

        # Determine actual power based on charge/discharge limits
        if target_terminal_power_kw < 0:  # Charging request
            # Power at terminals is limited by max_charge_power_kw
            actual_terminal_power_kw = max(target_terminal_power_kw, -self.max_charge_power_kw)
            if actual_terminal_power_kw != target_terminal_power_kw: 
                limited_by_soc_or_power = True
            
            # Energy delivered to storage (accounts for efficiency)
            energy_to_storage_kwh = -actual_terminal_power_kw * self.charge_efficiency * duration_hours
        elif target_terminal_power_kw > 0:  # Discharging request
            # Power at terminals is limited by max_discharge_power_kw
            actual_terminal_power_kw = min(target_terminal_power_kw, self.max_discharge_power_kw)
            if actual_terminal_power_kw != target_terminal_power_kw:
                limited_by_soc_or_power = True

            # Energy drawn from storage (accounts for efficiency)
            # To get actual_terminal_power_kw out, we need to draw more from storage
            energy_from_storage_kwh = actual_terminal_power_kw / self.discharge_efficiency * duration_hours
            energy_to_storage_kwh = -energy_from_storage_kwh # Negative as it's leaving storage
        else: # No power requested
            actual_terminal_power_kw = 0.0
            energy_to_storage_kwh = 0.0

        # Calculate potential new energy and SoC
        potential_energy_kwh = self.current_energy_kwh + energy_to_storage_kwh
        
        # Enforce SoC limits (min_soc_limit and max_soc_limit)
        min_energy_kwh = self.min_soc_limit * self.capacity_kwh
        max_energy_kwh = self.max_soc_limit * self.capacity_kwh

        previous_energy_kwh = self.current_energy_kwh

        if potential_energy_kwh > max_energy_kwh:
            self.current_energy_kwh = max_energy_kwh
            limited_by_soc_or_power = True
        elif potential_energy_kwh < min_energy_kwh:
            self.current_energy_kwh = min_energy_kwh
            limited_by_soc_or_power = True
        else:
            self.current_energy_kwh = potential_energy_kwh
        
        # Actual energy change that occurred in storage
        energy_change_in_storage_kwh = self.current_energy_kwh - previous_energy_kwh

        # If energy_change_in_storage was different than energy_to_storage_kwh due to SoC limits,
        # we need to recalculate the actual_terminal_power_kw
        if energy_change_in_storage_kwh != energy_to_storage_kwh:
            limited_by_soc_or_power = True # Operation was definitely limited by SoC
            if target_terminal_power_kw < 0: # Charging
                # If we stored less than intended due to max_soc_limit
                actual_terminal_power_kw = -(energy_change_in_storage_kwh / self.charge_efficiency / duration_hours) if duration_hours > 0 else 0
            elif target_terminal_power_kw > 0: # Discharging
                # If we drew less than intended due to min_soc_limit (energy_change_in_storage_kwh is negative)
                actual_terminal_power_kw = (-energy_change_in_storage_kwh * self.discharge_efficiency / duration_hours) if duration_hours > 0 else 0
            else: # No target power, but somehow energy changed (should not happen if SoC limits were hit properly initially)
                 actual_terminal_power_kw = 0 # Should remain 0

        self.soc = self.current_energy_kwh / self.capacity_kwh if self.capacity_kwh > 0 else 0
        
        # Ensure SoC is clamped as a final safety, though logic above should handle it.
        self.soc = max(self.min_soc_limit, min(self.soc, self.max_soc_limit))
        self.current_energy_kwh = self.soc * self.capacity_kwh

        return actual_terminal_power_kw, energy_change_in_storage_kwh, limited_by_soc_or_power

    @property
    def is_full(self) -> bool:
        """Returns True if the battery is at its maximum SoC limit."""
        return self.soc >= self.max_soc_limit - 1e-6 # Using a small epsilon for float comparison

    @property
    def is_empty(self) -> bool:
        """Returns True if the battery is at its minimum SoC limit."""
        return self.soc <= self.min_soc_limit + 1e-6 # Using a small epsilon

if __name__ == '''__main__''':
    # Example Usage
    logging.basicConfig(level=logging.DEBUG)
    battery = Battery(
        capacity_kwh=10.0,
        initial_soc=0.5,
        max_charge_power_kw=5.0,
        max_discharge_power_kw=5.0,
        charge_efficiency=0.9,
        discharge_efficiency=0.9,
        min_soc_limit=0.1,
        max_soc_limit=0.9
    )
    print(f"Initial SoC: {battery.soc:.2f}, Energy: {battery.current_energy_kwh:.2f} kWh")

    # Test charging
    print("\n--- Test Charging ---")
    power, energy_change, limited = battery.step(target_terminal_power_kw=-5.0, duration_hours=1.0)
    print(f"Target: -5.0 kW, Actual: {power:.2f} kW, Stored Change: {energy_change:.2f} kWh, SoC: {battery.soc:.2f}, Limited: {limited}")
    power, energy_change, limited = battery.step(target_terminal_power_kw=-5.0, duration_hours=1.0) # Charge more
    print(f"Target: -5.0 kW, Actual: {power:.2f} kW, Stored Change: {energy_change:.2f} kWh, SoC: {battery.soc:.2f}, Limited: {limited}") # Should be limited by max_soc
    power, energy_change, limited = battery.step(target_terminal_power_kw=-5.0, duration_hours=1.0) # Try to charge when full
    print(f"Target: -5.0 kW, Actual: {power:.2f} kW, Stored Change: {energy_change:.2f} kWh, SoC: {battery.soc:.2f}, Limited: {limited}") # Should be 0 power

    # Test discharging
    battery.reset(initial_soc=0.9)
    print(f"\n--- Test Discharging (Reset to SoC: {battery.soc:.2f}) ---")
    power, energy_change, limited = battery.step(target_terminal_power_kw=5.0, duration_hours=1.0)
    print(f"Target: +5.0 kW, Actual: {power:.2f} kW, Stored Change: {energy_change:.2f} kWh, SoC: {battery.soc:.2f}, Limited: {limited}")
    power, energy_change, limited = battery.step(target_terminal_power_kw=5.0, duration_hours=1.0) # Discharge more
    print(f"Target: +5.0 kW, Actual: {power:.2f} kW, Stored Change: {energy_change:.2f} kWh, SoC: {battery.soc:.2f}, Limited: {limited}") # Should be limited by min_soc
    power, energy_change, limited = battery.step(target_terminal_power_kw=5.0, duration_hours=1.0) # Try to discharge when empty
    print(f"Target: +5.0 kW, Actual: {power:.2f} kW, Stored Change: {energy_change:.2f} kWh, SoC: {battery.soc:.2f}, Limited: {limited}")

    # Test efficiency
    battery.reset(initial_soc=0.5)
    print(f"\n--- Test Efficiency (Reset to SoC: {battery.soc:.2f}) ---")
    # Charge 1 kWh at terminals (should store 0.9 kWh)
    # Power = Energy / Time = 1 kWh / 1h = 1kW. To get 1kWh into terminals, request -1kW.
    power, energy_change, limited = battery.step(target_terminal_power_kw=-1.0, duration_hours=1.0)
    print(f"Charge Target: -1.0 kW, Actual: {power:.2f} kW, Stored Change: {energy_change:.2f} kWh, SoC: {battery.soc:.2f}")
    # Expected SoC = 0.5 + (1.0 * 0.9 / 10.0) = 0.5 + 0.09 = 0.59
    
    # Discharge 1 kWh from terminals (should take 1/0.9 = 1.11 kWh from storage)
    # To get 1kWh from terminals, request 1kW.
    power, energy_change, limited = battery.step(target_terminal_power_kw=1.0, duration_hours=1.0)
    print(f"Discharge Target: +1.0 kW, Actual: {power:.2f} kW, Stored Change: {energy_change:.2f} kWh, SoC: {battery.soc:.2f}")
    # Expected SoC = 0.59 - (1.0 / 0.9 / 10.0) = 0.59 - 0.111 = 0.479

    # Test power limits
    battery.reset(initial_soc=0.5)
    print(f"\n--- Test Power Limits (Reset to SoC: {battery.soc:.2f}) ---")
    power, energy_change, limited = battery.step(target_terminal_power_kw=-10.0, duration_hours=1.0) # Exceed max charge power
    print(f"Charge Target: -10.0 kW, Actual: {power:.2f} kW (Expected -5.0), Limited: {limited}")
    power, energy_change, limited = battery.step(target_terminal_power_kw=10.0, duration_hours=1.0) # Exceed max discharge power
    print(f"Discharge Target: +10.0 kW, Actual: {power:.2f} kW (Expected +5.0), Limited: {limited}")

    # Test zero power
    print("\n--- Test Zero Power ---")
    current_soc_before_zero_op = battery.soc
    power, energy_change, limited = battery.step(target_terminal_power_kw=0.0, duration_hours=1.0)
    print(f"Target: 0.0 kW, Actual: {power:.2f} kW, Stored Change: {energy_change:.2f} kWh, SoC: {battery.soc:.2f}, Limited: {limited}")
    assert abs(battery.soc - current_soc_before_zero_op) < 1e-6, "SoC should not change with zero power target" 