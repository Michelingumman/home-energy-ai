"""
RL Agent and Environment Configuration

This file centralizes all configuration parameters for the RL agent, 
the simulation environment, and training processes.
"""

# ==============================================================================
# GENERAL SETTINGS
# ==============================================================================
# Paths and general behavior for the RL system

demand_model_path: str = "src/models/demand_model.keras" # Path to the pre-trained demand prediction model
price_model_path: str = "src/models/price_model.keras"   # Path to the pre-trained price prediction model
model_dir: str = "src/rl/saved_models"                # Directory to save trained RL models
log_dir: str = "src/rl/logs"                          # Directory for TensorBoard and other logs

training_mode: str = "short_term_only"  # Options: "short_term_only", "long_term_only", "hierarchical"
random_seed: int = 42                   # Seed for random number generators for reproducibility

# Training process control
checkpoint_freq: int = 100_000  # Timesteps: How often to save a model checkpoint during training
eval_freq: int = 50_000     # Timesteps: How often to run evaluation during training
eval_episodes: int = 10       # Number of episodes to run for each evaluation cycle


# ==============================================================================
# ENVIRONMENT SETTINGS (TRAINING)
# ==============================================================================
# Configuration for the simulation environment when training the agent

# --- Battery Configuration ---
battery_capacity: float = 22.0  # kWh: Total capacity of the battery
battery_degradation_cost_per_kwh: float = 45.0  # Cost units per kWh of battery charge/discharge throughput (NOW IN ÖRE)

# --- Simulation Time & Conditions ---
simulation_days: int = 60      # Days: Length of a single training episode
random_weather: bool = True    # Whether to use randomized weather data during training
fixed_baseload_kw: float = 0.5 # kW: Constant household demand if variable consumption is not used

# --- Price Data Configuration (Training) ---
use_price_model_train: bool = False         # If True, use the ML price model; if False, use historical/predicted prices from CSV
use_price_predictions_train: bool = True    # If True (and use_price_model_train is False), use pre-generated price predictions from CSV
price_predictions_path: str = "data/processed/SE3prices.csv"  # Path to the CSV file containing price predictions or historical prices

# --- Consumption Data Configuration (Training) ---
use_variable_consumption: bool = True     # If True, use a variable consumption profile from a CSV file
consumption_data_path: str = "data/processed/villamichelin/VillamichelinEnergyData.csv" # Path to consumption data

# --- Solar Data Configuration (Training) ---
use_solar_predictions_train: bool = True # If True, use solar production forecasts/actuals from a CSV file
solar_data_path_train: str = "src/predictions/solar/actual_data/ActualSolarProductionData.csv" # Path to solar data for training


# ==============================================================================
# ENVIRONMENT SETTINGS (EVALUATION)
# ==============================================================================
# Configuration for the simulation environment when evaluating the agent

simulation_days_eval: int = 14       # Days: Length of a single evaluation episode
random_weather_eval: bool = False    # Whether to use randomized weather data during evaluation

# --- Price Data Configuration (Evaluation) ---
use_price_predictions_eval: bool = True # If True, use pre-generated price predictions from CSV for evaluation
# price_predictions_path is shared with training, defined in GENERAL SETTINGS

# --- Solar Data Configuration (Evaluation) ---
use_solar_predictions_eval: bool = True # If True, use solar production forecasts/actuals from a CSV file for evaluation
solar_data_path_eval: str = "src/predictions/solar/actual_data/ActualSolarProductionData.csv" # Path to solar data for evaluation
# consumption_data_path is shared with training


# ==============================================================================
# REWARD FUNCTION PARAMETERS
# ==============================================================================
# Tunable parameters that define the agent's reward signal

soc_action_penalty_val: float = -150.0  # Penalty for illegal State of Charge actions (e.g., charging a full battery)

# --- Peak Demand Penalties ---
peak_threshold_kw: float = 6.0          # kW: Threshold above which peak demand penalty applies
peak_penalty_factor: float = 20.0         # Multiplier for the peak penalty component (original value in file was 5.0, but was 10.0 in other places, using 10.0 from eval script for consistency example)
peak_penalty_scale_factor: float = 10.0  # Scaling factor for the peak penalty term in the total reward calculation

# --- Export Limitation ---
export_peak_threshold_kw: float = 20.0  # kW: Maximum total power (solar + battery) that can be exported to the grid (25A * 230V * 3 phases ≈ 17.25kW)

# --- Swedish Night Tariff Discount ---
apply_swedish_night_discount: bool = True  # Whether to apply the Swedish night-time discount (22:00 to 06:00)
night_discount_factor: float = 0.5  # Price is multiplied by this factor during night hours (default: 0.5 = 50% discount)

# --- Battery Degradation Cost ---
battery_cost_scale_factor: float = 1.0  # Scaling factor for battery degradation cost in the total reward calculation

# --- Price Arbitrage Bonuses (Charging) ---
charge_bonus_threshold_price: float = 20.0  # öre/kWh: Price below which charging the battery receives a bonus
charge_bonus_multiplier: float = 200.0       # Multiplier for the charging bonus

# --- Price Arbitrage Bonuses (Discharging) ---
discharge_bonus_threshold_price_moderate: float = 100.0  # öre/kWh: Moderate price above which discharging receives a bonus
discharge_bonus_multiplier_moderate: float = 150.0        # Multiplier for the moderate discharge bonus

discharge_bonus_threshold_price_high: float = 150.0  # öre/kWh: High price above which discharging receives a larger bonus
discharge_bonus_multiplier_high: float = 300.0        # Multiplier for the high discharge bonus


# ==============================================================================
# SHORT-TERM AGENT SETTINGS (PPO)
# ==============================================================================
# Configuration specific to the short-term Proximal Policy Optimization (PPO) agent

short_term_learning_rate: float = 1e-5   # Learning rate for the PPO optimizer
short_term_gamma: float = 0.998          # Discount factor for future rewards
short_term_n_steps: int = 2048           # Number of steps to run for each environment per update (collects data for this many steps)
short_term_batch_size: int = 256          # Minibatch size for PPO updates
short_term_n_epochs: int = 20            # Number of epochs when optimizing the PPO surrogate loss
short_term_timesteps: int = 5000000        # Total number of timesteps to train the short-term agent


# ==============================================================================
# LONG-TERM ENVIRONMENT SETTINGS (Commented out - for future use)
# ==============================================================================
# These would be used if training_mode includes "long_term" or "hierarchical"
# lt_simulation_days: int = 365
# lt_time_step_days: int = 1


# ==============================================================================
# LONG-TERM AGENT SETTINGS (Commented out - for future use)
# ==============================================================================
# These would be used if training_mode includes "long_term" or "hierarchical"
# long_term_learning_rate: float = 1e-3
# long_term_gamma: float = 0.99
# long_term_n_steps: int = 30
# long_term_batch_size: int = 5
# long_term_n_epochs: int = 10
# long_term_timesteps: int = 1000


# ==============================================================================
# HIERARCHICAL CONTROLLER SETTINGS (Commented out - for future use)
# ==============================================================================
# These would be used if training_mode is "hierarchical"
# hc_planning_horizon_days: int = 7 # How far the long-term agent plans for the short-term


# ==============================================================================
# HELPER FUNCTION
# ==============================================================================

def get_config_dict() -> dict:
    """Returns the configuration parameters as a dictionary.
    
    This function collects all global variables defined in this module (excluding
    dunder methods and this function itself) and returns them as a dictionary.
    This allows other scripts to easily access these configuration values.
    """
    return {
        key: value for key, value in globals().items() 
        if not key.startswith("__") and key != "get_config_dict"
    } 