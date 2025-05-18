"""
Home Energy AI Configuration

Time wasted here is insane...
/sad-Adam :)


This file centralizes all configuration parameters for the RL agent, 
the simulation environment, and training processes.
"""



# ==============================================================================
#                           SYSTEM PATHS & BASIC SETTINGS
# ==============================================================================

# Model and data paths
model_dir: str = "src/rl/saved_models"                # Directory to save trained models
log_dir: str = "src/rl/logs"                          # Directory for logs
price_predictions_path: str = "data/processed/SE3prices.csv"  # Price predictions/historical data
consumption_data_path: str = "data/processed/villamichelin/VillamichelinConsumption.csv"  # Household consumption
solar_data_path_train: str = "src/predictions/solar/actual_data/ActualSolarProductionData.csv"  # Solar production data
solar_data_path_eval: str = "src/predictions/solar/actual_data/ActualSolarProductionData.csv"  # Solar data for evaluation

# ML model paths (for demand/price forecasting)
demand_model_path: str = "src/models/demand_model.keras"  # Demand prediction model
price_model_path: str = "src/models/price_model.keras"    # Price prediction model

# Training control settings
random_seed: int = 42               # For reproducibility
training_mode: str = "short_term_only"  # Options: "short_term_only", "long_term_only", "hierarchical"
checkpoint_freq: int = 1_000_000    # Save checkpoint every X timesteps
eval_freq: int = 50_000             # Evaluate model every X timesteps
eval_episodes: int = 10             # Number of episodes per evaluation







# ==============================================================================
#                           SIMULATION PARAMETERS
# ==============================================================================

# Time settings
simulation_days: int = 30           # Length of a training episode in days
simulation_days_eval: int = 14      # Length of evaluation episodes in days

# Baseline energy parameters
fixed_baseload_kw: float = 0.5      # Default household consumption if variable data not used

# Data source flags
use_variable_consumption: bool = True       # Use actual consumption data
use_price_model_train: bool = False         # Use ML price model vs historical data
use_price_predictions_train: bool = True    # Use pre-generated price predictions
use_price_predictions_eval: bool = True     # Use price predictions for evaluation
use_solar_predictions_train: bool = True    # Use solar production data for training
use_solar_predictions_eval: bool = True     # Use solar production data for evaluation
random_weather_eval: bool = False           # Use randomized weather for evaluation







# ==============================================================================
#                           BATTERY CONFIGURATION
# ==============================================================================

# Battery physical properties
battery_capacity: float = 22.0              # Total capacity in kWh
battery_initial_soc: float = 0.5            # Initial State of Charge (0.0 to 1.0)
battery_max_charge_power_kw: float = 5      # Maximum charging power (kW)
battery_max_discharge_power_kw: float = 10  # Maximum discharging power (kW)
battery_charge_efficiency: float = 0.95     # Efficiency when charging (0-1)
battery_discharge_efficiency: float = 0.95  # Efficiency when discharging (0-1)

# Battery operational limits
soc_min_limit: float = 0.2                  # Minimum allowable SoC
soc_max_limit: float = 0.8                  # Maximum allowable SoC

# Battery economics
battery_degradation_cost_per_kwh: float = 45.0  # Cost in öre/kWh for battery usage







# ==============================================================================
#                           SWEDISH ELECTRICITY PRICING MODEL
# ==============================================================================

# Energy costs
energy_tax: float = 54.875                   # Energy tax in öre/kWh including VAT
vat_mult: float = 1.25                       # VAT multiplier (25%)
grid_fee: float = 6.25                       # Grid fee in öre/kWh including VAT

# Grid connection fees
fixed_grid_fee_sek_per_month: float = 365.0  # Monthly subscription fee in SEK
capacity_fee_sek_per_kw: float = 81.25       # Fee per kW of peak demand per month

# Special rates
night_capacity_discount: float = 0.5         # Discount for peaks during 22:00-06:00







# ==============================================================================
#                           REWARD FUNCTION - SOC MANAGEMENT
# ==============================================================================

# Physical limit violation penalties
soc_limit_penalty_factor: float = 1000.0       # Base penalty for SoC outside allowed range
soc_violation_scale: float = 0.0             # Scaling for physical limit violations (will be multiplied by 50 in code)
# soc_action_limit_scale: float = 5.0          # Scaling for actions constrained by SoC

# Preferred SoC range (soft constraints)
preferred_soc_min_base: float = 0.3          # Lower bound of preferred SoC range
preferred_soc_max_base: float = 0.7          # Upper bound of preferred SoC range
preferred_soc_reward_factor: float = 100.0   # Reward for staying in preferred range (doubled to make it stronger)
preferred_soc_reward_scale: float = 1.0      # Scaling for preferred range reward






# ==============================================================================
#                           REWARD FUNCTION - GRID MANAGEMENT
# ==============================================================================

# Peak shaving incentives
peak_power_threshold_kw: float = 5.0         # Target maximum grid import
peak_penalty_factor: float = 300.0             # Penalty per kW above threshold
peak_penalty_scale: float = 1.0              # Scaling for peak penalties

# Price arbitrage incentives
enable_explicit_arbitrage_reward: bool = True  # Enable additional arbitrage rewards
arbitrage_reward_scale: float = 1.0            # Scaling for arbitrage rewards

# Low price charging incentives
low_price_threshold_ore_kwh: float = 20.0      # Fixed low price threshold
charge_at_low_price_reward_factor: float = 100.0 # Reward for charging at low prices

# High price discharging incentives
high_price_threshold_ore_kwh: float = 100.0    # Fixed high price threshold
discharge_at_high_price_reward_factor: float = 1000.0 # Reward for discharging at high prices

# Export reward
export_reward_bonus_ore_kwh: float = 60.0      # Bonus in öre/kWh for exported electricity on top of spot price

# Dynamic price threshold calculation
use_percentile_price_thresholds: bool = False  # Use percentiles instead of fixed values
low_price_percentile: float = 25.0             # Bottom percentile for "low" prices
high_price_percentile: float = 75.0            # Top percentile for "high" prices

# Battery degradation cost in reward
battery_degradation_reward_scaling_factor: float = 1.0  # Scale degradation cost in reward

# Global reward scaling
reward_scaling_factor: float = 0.03            # Global multiplier for all rewards






# ==============================================================================
#                           PPO ALGORITHM HYPERPARAMETERS
# ==============================================================================

# Core PPO parameters
short_term_learning_rate: float = 3e-4        # Learning rate for the optimizer
short_term_gamma: float = 0.95               # Discount factor for future rewards
short_term_n_steps: int = 2048                # Steps per update batch
short_term_batch_size: int = 256              # Minibatch size for updates
short_term_n_epochs: int = 15                 # Number of epochs per update
short_term_ent_coeff: float = 0.01           # Entropy coefficient (exploration)
short_term_gae_lambda: float = 0.97           # GAE lambda parameter
short_term_timesteps: int = 500000            # Total timesteps for training

# ==============================================================================
#                           HELPER FUNCTION
# ==============================================================================

def get_config_dict() -> dict:
    """Returns the configuration parameters as a dictionary.
    
    This function collects all global variables defined in this module (excluding
    dunder methods and this function itself) and returns them as a dictionary.
    This allows other scripts to easily access these configuration values.
    
    Returns:
        dict: All configuration parameters as a dictionary
    """
    return {
        key: value for key, value in globals().items() 
        if not key.startswith("__") and key != "get_config_dict"
    } 