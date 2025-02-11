import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

plt.style.use('fivethirtyeight')

# -------------------------------
# Helper Function: Create sequences for LSTM
# -------------------------------
def create_sequences(data, window_size):
    """
    Given a 1D array (data), create sequences of length `window_size`
    to predict the next value.
    Returns arrays X (of shape [samples, window_size, 1]) and y.
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

# -------------------------------
# 1. Process Old Data (2015-2020)
# -------------------------------
old_file_path = "C:/_Projects/home-energy-ai/data/raw/Elspotprices/sweden_elspot_prices.csv"
df_old = pd.read_csv(old_file_path, parse_dates=["cet_cest_timestamp"], index_col="cet_cest_timestamp")
df_old = df_old[["SE3"]].ffill().bfill()
df_old.index = pd.to_datetime(df_old.index, utc=True)
df_old.index = df_old.index.tz_localize(None)
df_old.rename(columns={"SE3": "SE3_old"}, inplace=True)

# Scale the target values
scaler_old = MinMaxScaler(feature_range=(0,1))
old_values = df_old["SE3_old"].values.reshape(-1, 1)
scaled_old = scaler_old.fit_transform(old_values)

# Choose a window size (e.g., past 24 hours to predict the next hour)
window_size = 24
X_old, y_old = create_sequences(scaled_old, window_size)
# The corresponding target timestamps (starting at position window_size)
timestamps_old = df_old.index[window_size:]

# Split into train and test based on date
cutoff_old = pd.to_datetime('2019-01-01')
train_idx_old = [i for i, t in enumerate(timestamps_old) if t < cutoff_old]
test_idx_old  = [i for i, t in enumerate(timestamps_old) if t >= cutoff_old]

X_old_train = X_old[train_idx_old]
y_old_train = y_old[train_idx_old]
X_old_test  = X_old[test_idx_old]
y_old_test  = y_old[test_idx_old]
test_timestamps_old = np.array(timestamps_old)[test_idx_old]

# Build the LSTM model for old data
model_old = Sequential([
    LSTM(50, activation='relu', input_shape=(window_size, 1)),
    Dense(1)
])
model_old.compile(optimizer='adam', loss='mse')
model_old.summary()

# Train the model (adjust epochs and batch_size as needed)
history_old = model_old.fit(X_old_train, y_old_train, epochs=20, batch_size=32, 
                            validation_split=0.1, verbose=1)

# Predict on test set and invert scaling
y_old_pred = model_old.predict(X_old_test)
y_old_pred_inv = scaler_old.inverse_transform(y_old_pred)
y_old_test_inv  = scaler_old.inverse_transform(y_old_test)

# Plot predictions vs actual for old data
plt.figure(figsize=(15,5))
plt.plot(test_timestamps_old, y_old_test_inv, label="Actual Prices")
plt.plot(test_timestamps_old, y_old_pred_inv, label="Predicted Prices", linestyle='dashed')
plt.title("LSTM Forecasting on Old Data (SE3_old)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

# -------------------------------
# 2. Process New Data (2015-2024)
# -------------------------------
new_file_path = r"C:\_Projects\home-energy-ai\data\raw\Elspotprices\Elspotprices 2015- 2024.csv"
df_new = pd.read_csv(new_file_path, sep=';', decimal=',')
df_new = df_new.ffill().bfill()

# Check for missing values
if df_new.isnull().values.any():
    print("There are missing values in the new dataset")
else:
    print("There are no missing values in the new dataset")

# Filter for SE3 rows
df_new = df_new[df_new['PriceArea'] == 'SE3'].copy()

# Parse HourUTC as datetime and set as index
df_new['HourUTC'] = pd.to_datetime(df_new['HourUTC'])
df_new.set_index('HourUTC', inplace=True)

# Convert SpotPriceEUR to öre/kWh.
# Example: 1 EUR = 10 SEK and 1 SEK = 100 öre.
# (The given code divides by 1000, adjust as needed.)
exchange_rate = 10  
df_new['SE3_new'] = (df_new['SpotPriceEUR'] * exchange_rate * 100) / 1000

# Scale the new target values
scaler_new = MinMaxScaler(feature_range=(0,1))
new_values = df_new['SE3_new'].values.reshape(-1, 1)
scaled_new = scaler_new.fit_transform(new_values)

# Create sequences from new data
X_new, y_new = create_sequences(scaled_new, window_size)
timestamps_new = df_new.index[window_size:]

# Split into train and test using a cutoff date (e.g., 2022-01-01)
cutoff_new = pd.to_datetime('2022-01-01')
train_idx_new = [i for i, t in enumerate(timestamps_new) if t < cutoff_new]
test_idx_new  = [i for i, t in enumerate(timestamps_new) if t >= cutoff_new]

X_new_train = X_new[train_idx_new]
y_new_train = y_new[train_idx_new]
X_new_test  = X_new[test_idx_new]
y_new_test  = y_new[test_idx_new]
test_timestamps_new = np.array(timestamps_new)[test_idx_new]

# Build the LSTM model for new data (using the same architecture)
model_new = Sequential([
    LSTM(50, activation='relu', input_shape=(window_size, 1)),
    Dense(1)
])
model_new.compile(optimizer='adam', loss='mse')
model_new.summary()

# Train the new data model
history_new = model_new.fit(X_new_train, y_new_train, epochs=20, batch_size=32, 
                            validation_split=0.1, verbose=1)

# Predict on test set and invert scaling
y_new_pred = model_new.predict(X_new_test)
y_new_pred_inv = scaler_new.inverse_transform(y_new_pred)
y_new_test_inv  = scaler_new.inverse_transform(y_new_test)

# Plot predictions vs actual for new data
plt.figure(figsize=(15,5))
plt.plot(test_timestamps_new, y_new_test_inv, label="Actual Prices")
plt.plot(test_timestamps_new, y_new_pred_inv, label="Predicted Prices", linestyle='dashed')
plt.title("LSTM Forecasting on New Data (SE3_new)")
plt.xlabel("Time")
plt.ylabel("Price (Öre/kWh)")
plt.legend()
plt.show()
