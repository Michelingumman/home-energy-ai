import os
import pandas as pd
import matplotlib.pyplot as plt

# Import your preprocessing functions.
# Adjust the import paths based on your project structure.
from src.data.preprocess import load_raw_data, clean_data, resample_data, normalize_series

# Path to your CSV file.
csv_path = os.path.join("data", "raw", "Tibber_villamichelin_PowerConsumption.csv")

# Load the raw data (with header assumed in the CSV).
df = load_raw_data(csv_path)

# Clean the data by converting non-numeric entries and filling missing values.
df_clean = clean_data(df, missing_method='ffill')

# Resample the data to an hourly frequency.
# Using '1h' (lowercase) to avoid deprecation warnings.
df_resampled = resample_data(df_clean, rule='1h', agg='mean')

# Extract the raw series.
raw_series = df_resampled['state']

# Normalize the series.
normalized_series, scaler = normalize_series(raw_series)

# Plot both the raw and normalized series.
plt.figure(figsize=(14, 8))

# Plot the raw data.
plt.subplot(2, 1, 1)
plt.plot(raw_series.index, raw_series.values, color='blue', label='Raw Data')
plt.title("Raw Power Consumption Data")
plt.xlabel("Time")
plt.ylabel("Power Consumption")
plt.legend()

# Plot the normalized data.
plt.subplot(2, 1, 2)
plt.plot(normalized_series.index, normalized_series.values, color='orange', label='Normalized Data')
plt.title("Normalized Power Consumption Data")
plt.xlabel("Time")
plt.ylabel("Normalized Value (0-1)")
plt.legend()

plt.tight_layout()
plt.show()
