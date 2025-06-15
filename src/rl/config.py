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
soc_min_limit: float = 0.1                  # Minimum allowable SoC
soc_max_limit: float = 0.9                  # Maximum allowable SoC

# Battery economics
battery_degradation_cost_per_kwh: float = 45.0  # Cost in öre/kWh for battery usage
# Battery degradation cost in reward
battery_degradation_reward_scaling_factor: float = 0.05  # Reduced from 0.3 - Scale degradation cost in reward







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
enforce_capacity_same_day_constraint: bool = True  # Enforce Swedish regulation: max 1 peak per day in top 3
night_charging_scaling_factor : float = 0.8  # Significantly reduced from 2.0 - was creating excessive night charging






# ==============================================================================
#                           REWARD FUNCTION - SOC MANAGEMENT
# ==============================================================================

# Physical limit violation penalties
soc_limit_penalty_factor: float = 2.0       # Reduced from 5.0 to be less punitive

# Preferred SoC range (soft constraints)
preferred_soc_min_base: float = 0.3          # Keep same - working well
preferred_soc_max_base: float = 0.8          # Keep same - working well
preferred_soc_reward_factor: float = 1.5    # Reduced from 1.2 - SoC rewards too dominant

# High SoC specific penalties
high_soc_penalty_multiplier: float = 1.5    # Reduced from 1.05 - much less pressure to discharge
very_high_soc_threshold: float = 0.80        # Increased from 0.85 - only penalize truly excessive SoC

# Action modification penalty
action_modification_penalty: float = 0.8    # Reduced from 1.0 to be less punitive
max_consecutive_penalty_multiplier: float = 1.5  # Reduced from 2.0

# New: SoC limit violation tracking
soc_violation_memory_factor: float = 0.95   # Exponential decay for violation memory
soc_violation_escalation_factor: float = 1.3  # Reduced from 1.5

# Potential function parameters
soc_potential_min_value: float = -2.0       # Reduced magnitude from -3.0
soc_potential_max_value: float = 2.0       # Reduced magnitude from 3.0







# ==============================================================================
#                           REWARD FUNCTION - GRID MANAGEMENT
# ==============================================================================

#consumption * price = grid cost scaling factor
grid_cost_scaling_factor: float = 0.004  # Reduced from 0.01 - bring ranges down

# Peak shaving incentives
peak_power_threshold_kw: float = 5.0        # Target maximum grid import
peak_penalty_factor: float = 0.5           # Reduced from 1.0 to minimize capacity penalties

# Price arbitrage incentives
enable_explicit_arbitrage_reward: bool = True    # Whether to include arbitrage bonus rewards (changed from False)

# Low price charging incentives
low_price_threshold_ore_kwh: float = 30.0        # Fixed threshold for low prices (for charging)
charge_at_low_price_reward_factor: float = 2.0  # Increased from 1.0 - better incentive for low price charging

# High price discharging incentives
high_price_threshold_ore_kwh: float = 100.0       # Increased from 70.0 - trigger less often
discharge_at_high_price_reward_factor: float = 3.5  # Reduced from 4.0 - less aggressive discharge incentive

# Dynamic price threshold calculation
use_percentile_price_thresholds: bool = True     # Whether to use percentiles instead of fixed thresholds
low_price_percentile: float = 30.0               # Increased from 25.0 - be more selective about "low" prices
high_price_percentile: float = 75.0              # Reduced from 85.0 - more opportunities for discharge

# Export reward
export_reward_bonus_ore_kwh: float = 60     # In Sweden this is called 60öringen :)
export_reward_scaling_factor: float = 0.004    # Reduced from 0.002 - fine-tune scaling

# Morning SoC targeting before solar production
enable_morning_soc_target: bool = False      # Disable simplistic morning emptying strategy
morning_hours_start: int = 5               # Start of morning hours (e.g., 5 AM)
morning_hours_end: int = 8                # End of morning hours (e.g., 8 AM)
morning_target_soc: float = 0.25          # Target SoC in morning before solar production
morning_solar_threshold_kwh: float = 2.0  # Minimum expected solar production to activate
morning_soc_reward_factor: float = 2.0    # Reduced from 7.0 - reward multiplier for morning SoC targeting

# Night-to-peak chain bonus
enable_night_peak_chain: bool = False       # Temporarily disable - was causing problems
night_to_peak_bonus_factor: float = 1.0   # Keep same for when re-enabled
night_charge_window_hours: float = 24.0    # How long night energy is valid (hours)

# RecurrentPPO parameters
n_lstm_layers: int = 1           # Number of LSTM layers in the recurrent policy
lstm_hidden_size: int = 64      # Hidden size of LSTM layers


# Global reward scaling
reward_scaling_factor: float = 0.1            # Reduced from 0.2 - bring overall ranges down

# Multi-Objective Reward Component Weights

w_grid: float = 0.3              # Keep same - working well
w_cap: float = 1.0               # Keep same - capacity management is good
w_deg: float = 0.1               # Keep same - battery cost is reasonable
w_soc: float = 0.5               # Reduced from 1.1 - reduce SoC dominance slightly
w_shape: float = 0.5             # Keep same - shaping is balanced
w_night: float = 0.0             # Keep same - night charging is working
w_arbitrage: float = 0.7         # Increased from 3.0 - encourage more strategic discharge
w_export: float = 0.1            # Keep same - export is working well
w_action_mod: float = 1.0        # Reduced from 0.6 - reduce action masking penalty
w_chain: float = 0.0             #
w_solar: float = 0.0             # Keep same - solar component is minor

# ==============================================================================
#                           PPO ALGORITHM HYPERPARAMETERS
# ==============================================================================

# Core PPO parameters
short_term_learning_rate: float = 2e-4        # Reduced from 3e-4 for more stable learning
short_term_gamma: float = 0.995               # Increased from 0.99 for better long-term planning
short_term_n_steps: int = 4096                # Increased from 2048 for better sample efficiency
short_term_batch_size: int = 256              # Increased from 64 for more stable updates
short_term_n_epochs: int = 8                  # Reduced from 10 to prevent overfitting
short_term_ent_coeff: float = 0.01           # Reduced from 0.01 for more focused exploration
short_term_gae_lambda: float = 0.95           # Increased from 0.95 for better advantage estimation
short_term_timesteps: int = 3_000_000         # Increased from 500_000 for more thorough training

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