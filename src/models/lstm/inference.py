import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.data.preprocess import load_raw_data, clean_data, resample_data, normalize_series
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
# Use lowercase '1h' for hourly resampling.
df_resampled = resample_data(df_clean, rule='1h', agg='mean')

# Extract the 'state' series and normalize it.
power_series = df_resampled['state']
power_series, scaler = normalize_series(power_series)

# Create windowed data.
X, y = df_to_X_y(power_series, input_window, forecast_horizon)

# Determine split indices.
total_samples = X.shape[0]
train_end = int(total_samples * train_split_frac)
val_end   = int(total_samples * val_split_frac)

# Here, we use the test set for inference.
X_test, y_test = X[val_end:], y[val_end:]
print("Total samples:", total_samples)
print("Test samples:", X_test.shape, y_test.shape)

# Load the trained model.
model = tf.keras.models.load_model(model_save_path)

# Generate forecasts on the test set.
predictions = model.predict(X_test)

# Invert normalization.
def inverse_transform_samples(samples, scaler):
    """
    Given an array of shape (n_samples, forecast_horizon, 1),
    apply the scaler's inverse_transform to each sample.
    """
    samples_orig = []
    for sample in samples:
        # sample has shape (forecast_horizon, 1)
        sample_orig = scaler.inverse_transform(sample)
        samples_orig.append(sample_orig)
    return np.array(samples_orig)

predictions_orig = inverse_transform_samples(predictions, scaler)
y_test_orig = inverse_transform_samples(y_test, scaler)

# For visualization, plot the first test sample (in original scale).
plt.figure(figsize=(12, 6))
plt.plot(range(forecast_horizon), predictions_orig[0].flatten(), label='Predicted', marker='o')
plt.plot(range(forecast_horizon), y_test_orig[0].flatten(), label='Actual', marker='x')
plt.title("One Week Ahead Forecast (First Test Sample) in Original Scale")
plt.xlabel("Hour")
plt.ylabel("Power Consumption")
plt.legend()
plt.grid(True)
plt.show()
