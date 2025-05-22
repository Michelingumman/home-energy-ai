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
log_dir: str = "src/rl/logs"  
# Directory for logs
price_predictions_path: str = "data/processed/SE3prices.csv"  # Price predictions/historical data
# consumption_data_path: str = "data/processed/villamichelin/VillamichelinEnergyData.csv"  # Household consumption
consumption_data_path: str = "data/processed/villamichelin/VillamichelinEnergyData.csv"  # Synthetic Household consumption


# solar_data_path: str = "src/predictions/solar/actual_data/ActualSolarProductionData.csv"  # Solar production data
solar_data_path: str = "data/processed/villamichelin/synthetic/ActualSolarProductionData.csv"  # Synthetic Solar production data

# ML model paths (for demand/price forecasting)
demand_model_path: str = "src/models/demand_model.keras"  # Demand prediction model
price_model_path: str = "src/models/price_model.keras"    # Price prediction model

# Training control settings
random_seed: int = 42               # For reproducibility
training_mode: str = "short_term_only"  # Options: "short_term_only", "long_term_only", "hierarchical"
checkpoint_freq: int = 1_000_000    # Save checkpoint every X timesteps
eval_freq: int = 50_000             # Evaluate model every X timesteps
eval_episodes: int = 10             # Number of episodes per evaluation


debug_prints: bool = True          # Print debug information  

# Optional: Restrict training to a specific date range (YYYY-MM-DD)
start_date: str = None  # Earliest allowed date for training episodes (inclusive)
end_date: str = None    # Latest allowed date for training episodes (inclusive)






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
# Battery degradation cost in reward
battery_degradation_reward_scaling_factor: float = 0.05  # Scale degradation cost in reward







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
soc_limit_penalty_factor: float = 10.0      # Base penalty for SoC outside allowed range

# Preferred SoC range (soft constraints)
preferred_soc_min_base: float = 0.3          # Lower bound of preferred SoC range
preferred_soc_max_base: float = 0.7          # Upper bound of preferred SoC range
preferred_soc_reward_factor: float = 5.0    # Reward for staying in preferred range

# High SoC specific penalties
high_soc_penalty_multiplier: float = 1.0     # Multiplier for penalties above very_high_soc_threshold
very_high_soc_threshold: float = 0.75        # Threshold for applying extra penalties

# Action modification penalty
action_modification_penalty: float = 0.5    # Penalty for actions requiring safety modification
max_consecutive_penalty_multiplier: float = 6.0  # Maximum escalation factor for repeated invalid actions

# Potential function parameters
soc_potential_min_value: float = -7.0       # Finite min value replacing -1e6
soc_potential_max_value: float = 7.0       # Finite max value replacing -1e6







# ==============================================================================
#                           REWARD FUNCTION - GRID MANAGEMENT
# ==============================================================================

#consumption * price = grid cost scaling factor
grid_cost_scaling_factor: float = 0.01

# Peak shaving incentives
peak_power_threshold_kw: float = 5.0        # Target maximum grid import
peak_penalty_factor: float = 1.0           # Penalty per kW above threshold

# Price arbitrage incentives
enable_explicit_arbitrage_reward: bool = False    # Whether to include arbitrage bonus rewards

# Low price charging incentives
low_price_threshold_ore_kwh: float = 30.0        # Fixed threshold for low prices (for charging)
charge_at_low_price_reward_factor: float = 2.0  # Reward for charging at low prices

# High price discharging incentives
high_price_threshold_ore_kwh: float = 100.0      # Fixed threshold for high prices (for discharging)
discharge_at_high_price_reward_factor: float = 5.0  # Reward for discharging at high prices

# Dynamic price threshold calculation
use_percentile_price_thresholds: bool = True     # Whether to use percentiles instead of fixed thresholds
low_price_percentile: float = 30.0               # Percentile for low price (increased)
high_price_percentile: float = 70.0              # Percentile for high price (decreased)

# Export reward
export_reward_bonus_ore_kwh: float = 60     # Bonus in öre/kWh for exported electricity on top of spot price
export_reward_scaling_factor: float = 0.01    # Scaling factor for export reward
# the division is to get a better range for the agent training

# Morning SoC targeting before solar production
enable_morning_soc_target: bool = True      # Enable morning SoC targeting
morning_hours_start: int = 5               # Start of morning hours (e.g., 5 AM)
morning_hours_end: int = 8                # End of morning hours (e.g., 8 AM)
morning_target_soc: float = 0.25          # Target SoC in morning before solar production
morning_solar_threshold_kwh: float = 2.0  # Minimum expected solar production to activate
morning_soc_reward_factor: float = 5.0    # Reward multiplier for morning SoC targeting

# Night-to-peak chain bonus
enable_night_peak_chain: bool = True       # Enable night-to-peak chain bonus
night_to_peak_bonus_factor: float = 15.0   # Bonus multiplier for using night energy at peak
night_charge_window_hours: float = 24.0    # How long night energy is valid (hours)

# RecurrentPPO parameters
n_lstm_layers: int = 1           # Number of LSTM layers in the recurrent policy
lstm_hidden_size: int = 64      # Hidden size of LSTM layers


# Global reward scaling
reward_scaling_factor: float = 0.1            # Global multiplier for all rewards

# Multi-Objective Reward Component Weights
w_grid: float = 1.0
w_cap: float = 2.0           # Increased from 0.1 to make peak penalties more visible
w_deg: float = 0.8
w_soc: float = 1.1              # soc limit penalty
w_shape: float = 1.5            # potential shaping, should guide into the preferred range
w_night: float = 0.2
w_arbitrage: float = 2.0
w_export: float = 1.5
w_action_mod: float = 1.0
w_chain: float = 2.5                      # Weight for night-to-peak chain bonus in total reward
w_morning: float = 1.2                    # Weight for morning SoC reward in total reward


# ==============================================================================
#                           PPO ALGORITHM HYPERPARAMETERS
# ==============================================================================

# Core PPO parameters
short_term_learning_rate: float = 1e-4        # Learning rate for the optimizer
short_term_gamma: float = 0.98               # Discount factor for future rewards, higher is more future rewards
short_term_n_steps: int = 1024                # Steps per update batch
short_term_batch_size: int = 64              # Minibatch size for updates
short_term_n_epochs: int = 15                 # Number of epochs per update
short_term_ent_coeff: float = 0.01           # Entropy coefficient (exploration)
short_term_gae_lambda: float = 0.96           # GAE lambda parameter, higher means more credit is given to future rewards
short_term_timesteps: int = 100_000            # Total timesteps for training

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

# Data augmentation settings (only used during training)
use_data_augmentation: bool = False  # Master switch for data augmentation
augment_solar_data: bool = True      # Apply random scaling to solar data
solar_augmentation_factor: float = 0.05  # Random variation factor for solar data (±20%)
augment_consumption_data: bool = True  # Apply random scaling to consumption data
consumption_augmentation_factor: float = 0.15  # Random variation factor for consumption (±15%) 