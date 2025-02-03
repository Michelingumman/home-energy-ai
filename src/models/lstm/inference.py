import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from src.data.preprocess import load_raw_data, clean_data, resample_data
from src.models.lstm.utils import df_to_X_y

# Load configuration.
config_path = os.path.join(os.path.dirname(__file__), "lstm_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

csv_paths         = config["csv_paths"]
input_window      = config["input_window"]
forecast_horizon  = config["forecast_horizon"]
train_split_frac  = config["train_split"]
val_split_frac    = config["val_split"]
model_save_path   = config["model_save_path"]

# Load and combine data.
dfs = [load_raw_data(path) for path in csv_paths]
df_combined = pd.concat(dfs).sort_index()
df_clean = clean_data(df_combined, missing_method='ffill')
df_resampled = resample_data(df_clean, rule='1H', agg='mean')

# Extract series and create windowed data.
power_series = df_resampled['state']
X, y = df_to_X_y(power_series, input_window, forecast_horizon)

# Determine split indices.
total_samples = X.shape[0]
train_end = int(total_samples * train_split_frac)
val_end   = int(total_samples * val_split_frac)

# Here, we use the test set for inference.
X_test, y_test = X[val_end:], y[val_end:]

# Load the trained model.
model = tf.keras.models.load_model(model_save_path)

# Generate forecasts on the test set.
predictions = model.predict(X_test)

# For visualization, we plot the first sample’s forecast vs. actual.
plt.figure(figsize=(12, 6))
plt.plot(range(forecast_horizon), predictions[0].flatten(), label='Predicted')
plt.plot(range(forecast_horizon), y_test[0].flatten(), label='Actual')
plt.title("One Week Ahead Forecast (First Test Sample)")
plt.xlabel("Hour")
plt.ylabel("Power Consumption")
plt.legend()
plt.show()
