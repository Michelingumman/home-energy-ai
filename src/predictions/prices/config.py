"""
Configuration file for the electricity price forecasting models.
This unified configuration is used by SARIMAX trend, peak, and valley models.
"""
from pathlib import Path
import os
import sys

#############################################################################
#                              CORE SETTINGS                                #
#############################################################################

# ----- Model Names -----
MODEL_NAME = "trend_model"  # Name for the trend model

# ----- Target Variable -----
TARGET_VARIABLE = "SE3_price_ore"  # Target variable to predict

# ----- Paths & Directories -----
# Get the directory of the current file
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = Path(os.path.abspath(os.path.join(BASE_DIR, "../../../data/processed")))
PLOTS_DIR = BASE_DIR / "plots"
LOGS_DIR = BASE_DIR / "logs"
EVAL_DIR = PLOTS_DIR / "evaluation"

# Model subdirectories
TREND_MODEL_DIR = MODELS_DIR / "trend_model"
PEAK_MODEL_DIR = MODELS_DIR / "peak_model"
VALLEY_MODEL_DIR = MODELS_DIR / "valley_model"
TREND_EVAL_DIR = EVAL_DIR / "trend"

# ----- Data Files -----
SE3_PRICES_FILE = DATA_DIR / "SE3prices.csv"
SWEDEN_GRID_FILE = DATA_DIR / "SwedenGrid.csv"
TIME_FEATURES_FILE = DATA_DIR / "time_features.csv"
HOLIDAYS_FILE = DATA_DIR / "holidays.csv"
WEATHER_DATA_FILE = DATA_DIR / "weather_data.csv"

#############################################################################
#                          TRAINING PARAMETERS                              #
#############################################################################

# ----- Sequence Configuration -----
LOOKBACK_WINDOW = 168  # 7 days of hourly data as input
PREDICTION_HORIZON = 24  # Predict 24 hours ahead

# ----- Dataset Split Configuration -----
VALIDATION_SPLIT = 0.15  # 15% for validation
TEST_SPLIT = 0.15  # 15% for testing

# ----- Shared Model Architecture Parameters -----
DROPOUT_RATE = 0.3
L1_REG = 1e-6
L2_REG = 1e-4

# ----- Loss Function Configuration -----
# Options: 'mae', 'mse', 'huber', 'log_cosh', 'custom_weighted', 'smape_loss', 'spike_weighted'
LOSS_FUNCTION = 'custom_weighted'

# Weighted loss parameters - give higher weight to price spikes and unusual values
WEIGHTED_LOSS_PARAMS = {
    "baseline_weight": 1.0,       # Base weight for normal prices
    "spike_threshold": 100.0,     # Threshold above which to consider a price a spike
    "spike_weight": 3.0,          # Weight multiplier for price spikes
    "negative_weight": 2.0,       # Weight multiplier for negative prices
}

#############################################################################
#                           TREND MODEL SETTINGS                            #
#############################################################################

# ----- XGBoost Model Parameters -----
# For Exogenous variables in SARIMAX model
TREND_EXOG_FEATURES = [
    # Grid data - key electricity system factors
    "powerConsumptionTotal",  # Electricity demand is crucial
    "powerProductionTotal",   # Supply side
    # "hydro",                  # Major production source in Sweden
    # "nuclear",                # Nuclear power generation
    # "wind",                   # Wind power generation
    # "powerImportTotal",       # Power imports
    # "powerExportTotal",       # Power exports
    
    # Price derived features
    "price_168h_avg",         # Weekly average price (stable)
    "hour_avg_price",         # Average price for this hour (stable)
    "price_24h_avg",          # Recent price level (with higher weight in smoothing)
]

#############################################################################
#                            PEAK MODEL SETTINGS                            #
#############################################################################

# ----- Peak Model Specific Parameters -----
PEAK_BATCH_SIZE = 128         # Larger batch size for handling class imbalance
PEAK_EPOCHS = 30              # Fewer epochs for faster training
PEAK_EARLY_STOPPING_PATIENCE = 5
PEAK_LEARNING_RATE = 2e-3     # Higher learning rate for faster convergence

# TCN architecture parameters
PEAK_TCN_FILTERS = 64         # Reduced from 128 for faster training
PEAK_TCN_KERNEL_SIZE = 3
PEAK_TCN_DILATIONS = [1, 2, 4, 8]  # Fewer dilations for faster training
PEAK_TCN_NB_STACKS = 2        # Single stack for reduced complexity

#############################################################################
#                           VALLEY MODEL SETTINGS                           #
#############################################################################

# ----- Valley Model Specific Parameters -----
VALLEY_BATCH_SIZE = 128        # Smaller batch size for more gradient updates
VALLEY_EPOCHS = 30            # Increased epochs for better learning with recall loss
VALLEY_EARLY_STOPPING_PATIENCE = 5  # Shorter patience for quicker training
VALLEY_LEARNING_RATE = 5e-4   # Lower learning rate for more stable training

# TCN architecture parameters - optimized for valley detection
VALLEY_TCN_FILTERS = 64      # Increased from 16 for better capacity 
VALLEY_TCN_KERNEL_SIZE = 3    # Smaller kernel for faster computation
VALLEY_TCN_DILATIONS = [1, 2, 4, 8]  # Extended dilations for better temporal context
VALLEY_TCN_NB_STACKS = 1      # Single stack is sufficient for production

# ----- Loss function and optimization parameters -----
# Higher value will make the model more aggressive in finding valleys
VALLEY_CLASS_WEIGHT_MULTIPLIER = 8.0  # Increased from 5.0 for more aggressive valley detection
FALSE_NEG_WEIGHT = 7.0 # Weight for false negatives (missed valleys) - higher means better recall
FALSE_POS_WEIGHT = 1.5 # Weight for false positives (false alarms) - lower means more permissive predictions

#############################################################################
#                            FEATURE DEFINITIONS                            #
#############################################################################

# ----- Core Feature Sets -----
# Core features that are essential for training the trend model
CORE_FEATURES = [
    TARGET_VARIABLE,        # Target price itself (historical)
    "hour_sin",             # Sine of hour 
    "hour_cos",             # Cosine of hour
    "day_of_week_sin",      # Sine of day of week
    "day_of_week_cos",      # Cosine of day of week
    "is_weekend",           # Weekend indicator
    "temperature_2m",       # Temperature data
    "powerConsumptionTotal",# Power demand/load
]

# Extended features to add incrementally for trend model experimentation
EXTENDED_FEATURES = [
    # Market factors
    "Gas_Price",
    "Coal_Price",
    "CO2_Price",
    
    # Grid data
    "wind",
    "hydro",
    "solar",
    "powerImportTotal",
    "powerExportTotal",
    
    # Weather data
    "wind_speed_100m",
    "cloud_cover",
    
    # Time and calendar features
    "is_holiday",
    "is_morning_peak",
    "is_evening_peak",
    "season",
    
    # Price features
    "price_24h_avg",
    "price_168h_avg",
    "price_24h_std",
]

# ----- Feature Category Definitions -----
# Core features from SE3prices.csv
PRICE_FEATURES = [
    TARGET_VARIABLE,      # Target price itself (historical)
    "price_24h_avg",      # 24-hour average price
    "price_168h_avg",     # Week average price
    "price_24h_std",      # Price standard deviation (volatility)
    "hour_avg_price",     # Average price for this hour
    "price_vs_hour_avg",  # Deviation from hourly average
]

# Features from external market factors
MARKET_FEATURES = [
    "Gas_Price",     # Natural gas price
    "Coal_Price",    # Coal price
    "CO2_Price",     # Carbon price
]

# Features from SwedenGrid.csv
GRID_FEATURES = [
    "powerConsumptionTotal",   # Total power consumption
    "powerProductionTotal",    # Total power production
    "powerImportTotal",        # Total power imports
    "powerExportTotal",        # Total power exports
    "nuclear",                 # Nuclear power generation
    "wind",                    # Wind power generation
    "hydro",                   # Hydro power generation
    "solar",                   # Solar power generation
    "renewablePercentage",     # Percentage of renewable energy
    "fossilFreePercentage",    # Percentage of fossil-free energy
]

# Features from time_features.csv
TIME_FEATURES = [
    "hour_sin",           # Sine of hour (cyclical encoding)
    "hour_cos",           # Cosine of hour (cyclical encoding)
    "day_of_week_sin",    # Sine of day of week (cyclical encoding)
    "day_of_week_cos",    # Cosine of day of week (cyclical encoding)
    "month_sin",          # Sine of month (cyclical encoding)
    "month_cos",          # Cosine of month (cyclical encoding)
    "is_morning_peak",    # Morning peak hours indicator
    "is_evening_peak",    # Evening peak hours indicator
    "is_weekend",         # Weekend indicator
    "season",             # Season indicator
]

# Features from holidays.csv
HOLIDAY_FEATURES = [
    "is_holiday",            # Holiday indicator
    "is_holiday_eve",        # Holiday eve indicator
    "days_to_next_holiday",  # Days to next holiday
    "days_from_last_holiday",# Days since last holiday
]

# Features from weather_data.csv
WEATHER_FEATURES = [
    "temperature_2m",         # Temperature
    "cloud_cover",            # Cloud cover
    "relative_humidity_2m",   # Humidity
    "wind_speed_100m",        # Wind speed
    "wind_direction_100m",    # Wind direction
    "shortwave_radiation_sum",# Solar radiation
]

# ----- Peak/Valley Model Specific Features -----
# Core features specifically selected for spike/peak detection
PEAK_CORE_FEATURES = [
    TARGET_VARIABLE,          # Target price itself (historical)
    "price_24h_avg",          # 24-hour average price
    "price_168h_avg",         # Week average price
    "price_24h_std",          # Price standard deviation (volatility)
    "price_vs_hour_avg",      # Deviation from hourly average
    "hour_sin",               # Sine of hour (cyclical encoding)
    "hour_cos",               # Cosine of hour (cyclical encoding)
    "day_of_week_sin",        # Sine of day of week
    "day_of_week_cos",        # Cosine of day of week
    "month_sin",              # Sine of month
    "month_cos",              # Cosine of month
    "is_morning_peak",        # Morning peak hours indicator
    "is_evening_peak",        # Evening peak hours indicator
    "is_weekend",             # Weekend indicator
    "powerConsumptionTotal",  # Total power consumption
    "powerProductionTotal",   # Total power production
    "nuclear",                # Nuclear power (baseload)
    "hydro",                  # Hydro power generation
    "powerImportTotal",       # Total power imports
    "temperature_2m",         # Temperature (affects heating/cooling demand)
    "Gas_Price",              # Natural gas price
    "CO2_Price",              # Carbon price
    "Coal_Price",             # Coal price
]

# Core features specifically selected for valley detection
VALLEY_CORE_FEATURES = [
    TARGET_VARIABLE,          # Target price itself (historical)
    "price_24h_avg",          # 24-hour average price
    "price_168h_avg",         # Week average price
    "price_24h_std",          # Price standard deviation (volatility)
    "price_vs_hour_avg",      # Deviation from hourly average
    # Time features
    "hour_sin",               # Sine of hour (cyclical encoding)
    "hour_cos",               # Cosine of hour (cyclical encoding)
    "day_of_week_sin",        # Sine of day of week
    "day_of_week_cos",        # Cosine of day of week
    "month_sin",              # Sine of month
    "month_cos",              # Cosine of month
    "is_weekend",             # Weekend indicator
    "is_holiday",             # Holiday indicator (low demand)
    "is_holiday_eve",         # Holiday eve indicator (low demand)
    # Detrended price features - NEW
    "price_diff_1h",          # Price change from 1 hour ago (momentum)
    "price_diff_3h",          # Price change from 3 hours ago (trend)
    "price_diff_6h",          # Price change from 6 hours ago (trend)
    "price_detrended",        # Detrended price (deviation from longer trend)
    "price_momentum",         # Rate of change in prices (acceleration)
    # Demand/supply indicators  
    "powerConsumptionTotal",  # Total power consumption
    "wind",                   # Wind power generation (excess causes low prices)
    "solar",                  # Solar power generation
    "powerExportTotal",       # Total power exports
    "powerImportTotal",       # Total power imports
    "wind_speed_100m",        # Wind speed (affects wind generation)
    "cloud_cover",            # Cloud cover (affects solar generation)
    "temperature_2m",         # Temperature
]

# For backward compatibility
SPIKE_CORE_FEATURES = PEAK_CORE_FEATURES.copy()

#############################################################################
#                       PEAK/VALLEY DETECTION SETTINGS                      #
#############################################################################

# Define constants specific to spike/valley detection
SPIKE_THRESHOLD_PERCENTILE = 90  # Top 10% of prices are considered spikes
VALLEY_THRESHOLD_PERCENTILE = 10  # Bottom 10% of prices are considered valleys

# Valley detection parameters - added for better daily valley detection
VALLEY_DETECTION_PARAMS = {
    "window": 5,                    # Reduced from 8 to focus on even more local patterns
    "slope_threshold": 0.03,        # Reduced from 0.05 for higher sensitivity
    "curvature_threshold": 0.01,    # Reduced from 0.03 for higher sensitivity
    "distance": 4,                  # Reduced from 3 to detect valleys that are even closer together
    "smoothing_window": 3,          # Reduced from 3 for finer detail
    "detect_daily_valleys": True,   # Keep daily valley detection enabled
    "daily_lookback": 4,            # Reduced from 5 to focus on more local patterns
    "daily_lookahead": 4,           # Reduced from 5 to focus on more local patterns
    "detect_relative_valleys": True, # Enable relative valley detection
    "relative_depth_threshold": 0.8 # Reduced from 0.10 to capture even more subtle valleys
}

# New robust valley detection parameters
ROBUST_VALLEY_DETECTION_PARAMS = {
    "min_prominence": 0.03,       # Reduced from 0.04 to detect more subtle valleys
    "min_width": 1,               # Reduced from 2 to detect narrower valleys
    "distance": 2,                # Reduced from 3 to allow closer valleys
    "depth_percentile": 15,       # Increased from 10 to include more valleys
    "smoothing_window": 3         # Keep smoothing window the same
}

#############################################################################
#                       FEATURE ENGINEERING SETTINGS                         #
#############################################################################

# ----- Lag Features -----
# Lag features to create (in hours)
PRICE_LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 72, 168]  # 1h, 2h, 3h, 6h, 12h, 1d, 2d, 3d, 7d

# ----- Rolling Window Features -----
# Rolling window features to create
ROLLING_WINDOWS = [
    {"window": 6, "features": ["mean", "std", "min", "max"]},    # 6h stats
    {"window": 24, "features": ["mean", "std", "min", "max"]},   # 24h stats
    {"window": 168, "features": ["mean", "std", "min", "max"]}   # 7d stats
]

# ----- Data Scaling Configuration -----
# Options: 'standard', 'robust', 'minmax', 'log_transform', 'custom'
SCALING_METHOD = 'log_transform'

# Parameters for log transform scaling
# For negative values, we'll use a shift to make all values positive before log transform
LOG_TRANSFORM_PARAMS = {
    "offset": 100,  # Add this value to make all prices positive
    "base": 10      # Base of the logarithm (10 for log10, e for ln)
}

# Custom scaling bounds for MinMax scaling if used
CUSTOM_SCALING_BOUNDS = {
    "price_min": -100,  # Set a reasonable minimum price floor
    "price_max": 1000   # Set a reasonable maximum price ceiling
}

#############################################################################
#                         EVALUATION CONFIGURATION                          #
#############################################################################

# ----- Metrics to calculate -----
METRICS = ["mae", "mse", "rmse", "mape", "smape", "median_ae", "direction_accuracy"]

# ----- Evaluation Settings -----
# Number of random start points for sequential prediction evaluation
EVALUATION_START_POINTS = 10

# ----- Prediction Configuration -----
# Default settings for predictions
DEFAULT_PREDICTION_HORIZON = 24   # Default prediction horizon in hours
DEFAULT_CONFIDENCE_LEVEL = 0.95   # Default confidence level for prediction intervals
ROLLING_PREDICTION_WINDOW = 6     # Window size for rolling predictions

#############################################################################
#                       DATA COLLECTION CONFIGURATION                       #
#############################################################################

# ----- Feature Groups for Data Collection Scripts -----
FEATURE_GROUPS = {
    "grid_cols": [
        "fossilFreePercentage", "renewablePercentage", 
        "powerConsumptionTotal", "powerProductionTotal", 
        "powerImportTotal", "powerExportTotal",
        "nuclear", "wind", "hydro", "solar", "unknown",
        "import_SE-SE2", "export_SE-SE4", 
        "import_NO-NO1", "export_NO-NO1",
        "import_DK-DK1", "export_DK-DK1",
        "import_FI", "export_FI"
    ],
    "price_cols": PRICE_FEATURES,
    "time_cols": TIME_FEATURES,
    "holiday_cols": HOLIDAY_FEATURES,
    "weather_cols": WEATHER_FEATURES,
    "market_cols": MARKET_FEATURES
}
