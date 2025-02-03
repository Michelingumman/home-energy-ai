import os
import json
import tensorflow as tf
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

# Import preprocessing from the data folder if available,
# or adjust these calls if you merged them into utils.
from src.data.preprocess import load_raw_data, clean_data, resample_data
from src.models.lstm.utils import df_to_X_y
from src.models.lstm.model import build_model

# Load configuration from JSON
config_path = os.path.join(os.path.dirname(__file__), "lstm_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

csv_paths         = config["csv_paths"]         # List of CSV file paths
input_window      = config["input_window"]        # e.g., 168 hours of history
forecast_horizon  = config["forecast_horizon"]    # e.g., 168 hours forecast
train_split_frac  = config["train_split"]         # e.g., 0.75
val_split_frac    = config["val_split"]           # e.g., 0.90
lstm_units        = config["lstm_units"]
learning_rate     = config["learning_rate"]
epochs            = config["epochs"]
batch_size        = config["batch_size"]
model_save_path   = config["model_save_path"]

# Load, clean, and combine data from multiple CSV files.
dfs = [load_raw_data(path) for path in csv_paths]
df_combined = pd.concat(dfs).sort_index()
df_clean = clean_data(df_combined, missing_method='ffill')
# Resample data to an hourly frequency
df_resampled = resample_data(df_clean, rule='1H', agg='mean')

# Extract the series and create windowed data.
power_series = df_resampled['state']
X, y = df_to_X_y(power_series, input_window, forecast_horizon)

# Determine split indices based on fractions.
total_samples = X.shape[0]
train_end = int(total_samples * train_split_frac)
val_end   = int(total_samples * val_split_frac)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val     = X[train_end:val_end], y[train_end:val_end]
X_test, y_test   = X[val_end:], y[val_end:]

print("Total samples:", total_samples)
print("Training samples:", X_train.shape, y_train.shape)
print("Validation samples:", X_val.shape, y_val.shape)
print("Test samples:", X_test.shape, y_test.shape)

# Build and compile the model.
model = build_model(input_window, forecast_horizon, lstm_units)
model.summary()

model.compile(
    loss=MeanSquaredError(),
    optimizer=Adam(learning_rate=learning_rate),
    metrics=[RootMeanSquaredError()]
)

# Setup callbacks.
checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, verbose=1)
early_stop = EarlyStopping(patience=3, restore_best_weights=True)

# Train the model.
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[checkpoint, early_stop]
)

print("Training complete. Model saved at:", model_save_path)
