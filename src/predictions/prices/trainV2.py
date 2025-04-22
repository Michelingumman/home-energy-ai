import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import joblib
import argparse
from datetime import datetime, timedelta
import json
import os
import logging
from pathlib import Path
import sys
import re # Import regex for weighting

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Paths
DATA_DIR = PROJECT_ROOT / "data/processed"
MODEL_SAVE_DIR = Path(__file__).resolve().parent / "models_v2_price_weather_grid" # NEW directory name
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --- Feature Configuration (Price + Weather + Grid) ---
TARGET_VARIABLE = "SE3_price_ore"

# Define weather features to load
WEATHER_FEATURES_TO_LOAD = [
    "temperature_2m", "wind_speed_100m", "shortwave_radiation_sum"
]

# Define grid features to load
GRID_FEATURES_TO_LOAD = [
    "powerConsumptionTotal", "hydro", "wind", "powerImportTotal", "powerExportTotal"
]

# Lags to apply
LAGS = [24, 72, 168] # Keep reduced lags: 1d, 3d, 7d
# Features to lag: ONLY the target price for now
FEATURES_TO_LAG = [TARGET_VARIABLE]

# Feature Weights Configuration
FEATURE_WEIGHTS = {
    TARGET_VARIABLE: 1.0,           # Target is unscaled, weight must be 1
    "_lag": 1.5,                    # Lagged price features
    # Weather features
    "temperature_2m": 0.1,          
    "wind_speed_100m": 0.1,
    "shortwave_radiation_sum": 0.01,
    # Grid features - typically highly relevant for price prediction
    "powerConsumptionTotal": 0.1,
    "hydro": 0.1,
    "wind": 0.1,
    "powerImportTotal": 0.02,
    "powerExportTotal": 0.02,
    # Add other features here later with appropriate weights
}

# Data Split
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Model & Training Params (Keep structure from baseline)
SEQUENCE_LENGTH = 72
HORIZON = 24
LSTM_UNITS = [256, 128]
DROPOUT_RATE = 0.2
L1_REG = 0.001
L2_REG = 0.001
LEARNING_RATE = 0.01
EPOCHS = 100
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 15
LOSS_FUNCTION = 'mae'

# --- Helper Functions ---

def load_data():
    """Loads Price, Weather, and Grid data files."""
    logging.info("Loading Price + Weather + Grid datasets...")
    df = None
    # Load Price data
    try:
        price_df = pd.read_csv(DATA_DIR / "SE3prices.csv", index_col=0, parse_dates=True)[[TARGET_VARIABLE]]
        df = price_df
        logging.info(f"Loaded Price data: {price_df.shape}")
    except FileNotFoundError as e:
        logging.error(f"CRITICAL: Price data file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading Price data: {e}")
        sys.exit(1)

    # Load Weather data 
    try:
        weather_df = pd.read_csv(DATA_DIR / "weather_data.csv", index_col=0, parse_dates=True)
        
        # Convert weather_df index to timezone-naive
        weather_df.index = weather_df.index.tz_localize(None)

        missing_cols = [col for col in WEATHER_FEATURES_TO_LOAD if col not in weather_df.columns]
        if missing_cols:
            logging.warning(f"Weather data missing columns: {missing_cols}")
        cols_to_use = [col for col in WEATHER_FEATURES_TO_LOAD if col in weather_df.columns]
        if cols_to_use:
            df = df.join(weather_df[cols_to_use], how='left')
            logging.info(f"Joined Weather data. Current shape: {df.shape}")
        else:
            logging.warning("No usable weather columns found.")
    except FileNotFoundError:
        logging.warning("Weather data file not found: weather_data.csv. Skipping weather features.")
    except Exception as e:
        logging.error(f"Error loading/joining Weather data: {e}")

    # Load Grid data
    try:
        grid_df = pd.read_csv(DATA_DIR / "SwedenGrid.csv", index_col=0, parse_dates=True)
        
        # Convert grid_df index to timezone-naive if needed
        if grid_df.index.tzinfo is not None:
            grid_df.index = grid_df.index.tz_localize(None)

        missing_cols = [col for col in GRID_FEATURES_TO_LOAD if col not in grid_df.columns]
        if missing_cols:
            logging.warning(f"Grid data missing columns: {missing_cols}")
        cols_to_use = [col for col in GRID_FEATURES_TO_LOAD if col in grid_df.columns]
        if cols_to_use:
            df = df.join(grid_df[cols_to_use], how='left')
            logging.info(f"Joined Grid data. Current shape: {df.shape}")
        else:
            logging.warning("No usable grid columns found.")
    except FileNotFoundError:
        logging.warning("Grid data file not found: grid_data.csv. Skipping grid features.")
    except Exception as e:
        logging.error(f"Error loading/joining Grid data: {e}")

    # Fill NaNs after join (crucial for left join)
    cols_before_fill = df.columns[df.isnull().any()].tolist()
    if cols_before_fill:
        logging.warning(f"NaNs found after joins in columns: {cols_before_fill}. Applying ffill then bfill.")
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        cols_after_fill = df.columns[df.isnull().any()].tolist()
        if cols_after_fill:
             logging.error(f"NaN values REMAIN after fill in columns: {cols_after_fill}. Cannot proceed.")
             raise ValueError("Unfillable NaN values detected in DataFrame")
        else:
             logging.info("NaN values handled successfully.")
    else:
        logging.info("No NaNs found after joining data sources.")

    logging.info(f"Final loaded data shape: {df.shape}. Date range: {df.index.min()} to {df.index.max()}")
    return df

def add_lagged_features(df, features_to_lag, lags):
    """Adds lagged versions of specified features."""
    logging.info(f"Adding lags ({lags}) for features: {features_to_lag}")
    df_lagged = df.copy()
    for feature in features_to_lag:
        if feature not in df.columns:
            logging.warning(f"Feature '{feature}' not found for lagging. Skipping.")
            continue
        for lag in lags:
            df_lagged[f'{feature}_lag{lag}h'] = df_lagged[feature].shift(lag)

    # Drop rows with NaNs introduced by lagging
    original_len = len(df_lagged)
    df_lagged.dropna(inplace=True)
    new_len = len(df_lagged)
    logging.info(f"Dropped {original_len - new_len} rows due to NaNs from lagging.")
    return df_lagged

def train_validate_test_split(df, val_split, test_split):
    """Performs sequential train/validation/test split."""
    n = len(df)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_test - n_val

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    logging.info(f"Data split: Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df

def fit_scaler(train_df, target_variable):
    """Fits StandardScaler on training data input features (excluding target)."""
    input_features = [col for col in train_df.columns if col != target_variable]
    if not input_features:
        logging.warning("No input features found to scale. Skipping scaler fitting.")
        return None, []

    scaler = StandardScaler()
    scaler.fit(train_df[input_features])

    # Store scaler
    scaler_path = MODEL_SAVE_DIR / "standard_scaler_price_weather_grid.save"
    joblib.dump(scaler, scaler_path)
    logging.info(f"StandardScaler for input features saved to {scaler_path}")

    # Return scaler and the list of features it was fitted on
    return scaler, input_features

def scale_dataframe(df, scaler, input_feature_names, target_variable):
    """Applies a pre-fitted scaler to input features of a DataFrame."""
    df_scaled = df.copy()
    # Scale only the input features (if scaler exists)
    if scaler and input_feature_names:
        df_scaled[input_feature_names] = scaler.transform(df[input_feature_names])
        logging.info(f"Scaled {len(input_feature_names)} input features. Target '{target_variable}' remains unscaled.")
    else:
        logging.info("Skipping scaling as no input features/scaler provided.")
    return df_scaled

def apply_feature_weights(df_scaled, weights_config, feature_names, lags_list, target_variable):
    """Applies weights to the scaled features (excluding target)."""
    logging.info("Applying feature weights (excluding target)...")
    df_weighted = df_scaled.copy()
    weight_vector = np.ones(len(feature_names))
    applied_weights_log = {}

    for i, name in enumerate(feature_names):
        if name == target_variable:
             weight_vector[i] = 1.0
             applied_weights_log[name] = "N/A (Unscaled Target)"
             continue

        matched = False
        weight_applied = 1.0 # Default
        if name in weights_config:
             weight_applied = weights_config[name]
             matched = True
        elif name.endswith(tuple([f"_lag{l}h" for l in lags_list])) and "_lag" in weights_config:
             weight_applied = weights_config["_lag"]
             matched = True
        # Add more pattern matching if needed for future feature groups

        weight_vector[i] = weight_applied
        applied_weights_log[name] = weight_applied

    default_weighted_features = [k for k, v in applied_weights_log.items() if v == 1.0 and k != target_variable]
    logging.info(f"Feature weights applied. Sample: { {k: v for k, v in list(applied_weights_log.items())[:15]} } ... ")
    if default_weighted_features:
        logging.warning(f"Features using default weight (1.0): {default_weighted_features}")

    df_weighted = df_weighted * weight_vector
    return df_weighted

def create_sequences(data, sequence_length, horizon, target_col_index):
    """Creates input sequences (X) and corresponding unscaled target sequences (y)."""
    X, y = [], []
    # data is expected to be a numpy array here
    for i in range(len(data) - sequence_length - horizon + 1):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length : i + sequence_length + horizon, target_col_index])
    return np.array(X), np.array(y)

def build_model(input_shape, horizon, lstm_units, dropout_rate, l1_reg, l2_reg):
    """Builds the LSTM model (with regularization)."""
    model = Sequential(name="LSTM_Price_Predictor_PriceWeatherGrid") # Updated name
    model.add(Input(shape=input_shape, name="Input_Sequence"))

    # LSTM Layers with Regularization
    regularizer = tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1)
        model.add(LSTM(units, return_sequences=return_sequences,
                       kernel_regularizer=regularizer,
                       recurrent_regularizer=regularizer,
                       name=f"LSTM_{i+1}"))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate, name=f"Dropout_{i+1}"))

    # Output Layer
    model.add(Dense(horizon, name="Output_Prediction"))

    model.summary(print_fn=logging.info)
    return model

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data (Price + Weather + Grid)
    df_raw = load_data()

    # 2. Add Lagged Features (Price Only)
    df_lagged = add_lagged_features(df_raw, FEATURES_TO_LAG, LAGS)

    # Define final feature list
    feature_names = df_lagged.columns.tolist()
    logging.info(f"Final features ({len(feature_names)}): {feature_names}")

    # Save feature list
    feature_list_path = MODEL_SAVE_DIR / "feature_list_price_weather_grid.json"
    with open(feature_list_path, 'w') as f:
        json.dump(feature_names, f, indent=4)
    logging.info(f"Feature list saved to {feature_list_path}")

    # 3. Split Data
    train_df, val_df, test_df = train_validate_test_split(df_lagged, VAL_SPLIT, TEST_SPLIT)

    # --- Save Test DataFrame --- #
    test_df_path = MODEL_SAVE_DIR / "test_df_price_weather_grid.csv"
    test_df.to_csv(test_df_path)
    logging.info(f"Unscaled test dataframe saved to {test_df_path}")
    # ------------------------- #

    # 4. Fit Scaler (on Train Input Features)
    scaler, input_feature_names = fit_scaler(train_df, TARGET_VARIABLE)
    if set(input_feature_names) != (set(feature_names) - {TARGET_VARIABLE}):
         logging.error("Input feature mismatch after scaler fitting!")
         sys.exit(1)

    # 5. Scale DataFrames (Input Features Only)
    train_scaled_inputs_df = scale_dataframe(train_df, scaler, input_feature_names, TARGET_VARIABLE)
    val_scaled_inputs_df = scale_dataframe(val_df, scaler, input_feature_names, TARGET_VARIABLE)
    test_scaled_inputs_df = scale_dataframe(test_df, scaler, input_feature_names, TARGET_VARIABLE)

    # 6. Apply Feature Weights
    train_weighted_df = apply_feature_weights(train_scaled_inputs_df, FEATURE_WEIGHTS, feature_names, LAGS, TARGET_VARIABLE)
    val_weighted_df = apply_feature_weights(val_scaled_inputs_df, FEATURE_WEIGHTS, feature_names, LAGS, TARGET_VARIABLE)
    test_weighted_df = apply_feature_weights(test_scaled_inputs_df, FEATURE_WEIGHTS, feature_names, LAGS, TARGET_VARIABLE)

    # 7. Get Target Column Index
    try:
        target_col_index = feature_names.index(TARGET_VARIABLE)
    except ValueError:
         logging.error(f"Target variable '{TARGET_VARIABLE}' not found in final feature list!")
         sys.exit(1)

    # 8. Create Sequences (Input X is scaled & weighted, Target Y is UNscaled)
    X_train, y_train = create_sequences(train_weighted_df[feature_names].values, SEQUENCE_LENGTH, HORIZON, target_col_index)
    X_val, y_val = create_sequences(val_weighted_df[feature_names].values, SEQUENCE_LENGTH, HORIZON, target_col_index)
    X_test, y_test = create_sequences(test_weighted_df[feature_names].values, SEQUENCE_LENGTH, HORIZON, target_col_index)

    logging.info(f"Sequence shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    logging.info(f"Sequence shapes: X_val={X_val.shape}, y_val={y_val.shape}")
    logging.info(f"Sequence shapes: X_test={X_test.shape}, y_test={y_test.shape}")

    # --- Save Test Sequences --- #
    X_test_path = MODEL_SAVE_DIR / "X_test_price_weather_grid.npy"
    y_test_unscaled_path = MODEL_SAVE_DIR / "y_test_unscaled_price_weather_grid.npy"
    np.save(X_test_path, X_test)
    np.save(y_test_unscaled_path, y_test)
    logging.info(f"Test sequences saved: X_test={X_test_path}, y_test_unscaled={y_test_unscaled_path}")
    # ----------------------------------------------------------------- #

    # 9. Build Model
    input_shape = (SEQUENCE_LENGTH, X_train.shape[2])
    model = build_model(input_shape, HORIZON, LSTM_UNITS, DROPOUT_RATE, L1_REG, L2_REG)

    # 10. Compile Model
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=LOSS_FUNCTION, metrics=['mae'])

    # 11. Train Model
    model_path = MODEL_SAVE_DIR / "best_model_price_weather_grid.keras"
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    logging.info("Starting Price+Weather+Grid model training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    logging.info(f"Training finished. Price+Weather+Grid model saved to {model_path}")
