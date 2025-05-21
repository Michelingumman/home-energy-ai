"""
Configuration for the RL Home Energy Environment
"""

# Default paths for prediction files (can be overridden in env config)
# Ensure these paths are relative to the project root or absolute.
# Example: project_root/src/predictions/...

DEFAULT_PRICE_PREDICTIONS_PATH = "data/processed/SE3prices.csv" # From old custom_env.py
DEFAULT_CONSUMPTION_DATA_PATH = "data/processed/villamichelin/synthetic/VillamichelinConsumption.csv" # Example path, replace with actual if available
DEFAULT_SOLAR_DATA_PATH = "data/processed/villamichelin/synthetic/ActualSolarProduction.csv" # Example path, replace with actual if available


def get_config_dict() -> dict:
    """
    Returns a dictionary of default configuration parameters.
    These can be overridden when instantiating the environment.
    """
    config = {
        # Environment settings
        "simulation_days": 7,
        "time_step_minutes": 15,
        "log_level": "INFO",  # "DEBUG", "INFO", "WARNING", "ERROR"
        "debug_prints": False, # Enable verbose debug prints in the environment

        # Data paths
        "price_predictions_path": DEFAULT_PRICE_PREDICTIONS_PATH,
        "use_variable_consumption": False, # Start with fixed baseload
        "consumption_data_path": DEFAULT_CONSUMPTION_DATA_PATH,
        "use_solar_predictions": False, # Start without solar
        "solar_data_path": DEFAULT_SOLAR_DATA_PATH,

        # Fixed baseload (if use_variable_consumption is False)
        "fixed_baseload_kw": 0.5,

        # Battery parameters
        "battery_capacity_kwh": 22.0,
        "battery_initial_soc": 0.5,
        "battery_max_charge_power_kw": 11.0,  # capacity / 2
        "battery_max_discharge_power_kw": 11.0, # capacity / 2
        "battery_charge_efficiency": 0.95,
        "battery_discharge_efficiency": 0.95,
        "battery_min_soc_limit": 0.1, # Hardware/safety minimum SoC
        "battery_max_soc_limit": 0.9, # Hardware/safety maximum SoC (e.g. 80% of nominal for longevity)

        # Reward parameters (to be expanded later)
        "reward_grid_cost_scale": 1.0, # Scales the cost of importing from grid
        "reward_export_revenue_scale": 1.0, # Scales the revenue from exporting to grid
        "reward_soc_penalty_scale": 0.1, # Simple penalty for being outside preferred SoC

        # For SoC reward (can be made more complex later)
        "preferred_soc_min": 0.3,
        "preferred_soc_max": 0.7,

        # Date range for training/evaluation (optional, None means use full data range)
        "start_date": None, # e.g., "2023-01-01"
        "end_date": None,   # e.g., "2023-12-31"

        # Technical parameters
        "vat_mult": 1.25, # VAT multiplier
        "energy_tax_ore_per_kwh": 54.875, # Energy tax in öre per kWh
        "grid_fee_ore_per_kwh": 6.25, # Grid transfer fee in öre per kWh
        "spot_price_source_column": "SE3_price_ore", # Column name for spot price in price_predictions_df
    }
    return config

if __name__ == '''__main__''':
    # Example of how to get the config
    default_config = get_config_dict()
    print("Default Configuration:")
    for key, value in default_config.items():
        print(f"  {key}: {value}")

    # Example of overriding a config value
    custom_config_env = {"simulation_days": 3, "log_level": "DEBUG"}
    # In the environment, you would typically do:
    # self.config = get_config_dict()
    # self.config.update(custom_config_env_params_passed_to_init)
    # print("\nCustomized Configuration (example):")
    # print(f"  Simulation Days: {self.config['simulation_days']}")
    # print(f"  Log Level: {self.config['log_level']}") 