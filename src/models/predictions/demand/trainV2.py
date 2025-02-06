# import os
# import json
# import tensorflow as tf
# import pandas as pd
# import numpy as np
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# from tensorflow.keras.losses import MeanSquaredError
# from tensorflow.keras.metrics import RootMeanSquaredError
# from tensorflow.keras.optimizers import Adam

# from src.data.preprocess import normalize_series

# from matplotlib import pyplot as plt

# # Import preprocessing from the data folder if available,
# # or adjust these calls if you merged them into utils.
# from src.data.preprocess import load_raw_data, clean_data, resample_data
# from src.models.lstm.utils import df_to_X_y
# from src.models.lstm.model import build_cnn_lstm_model


# # Load configuration from JSON
# config_path = os.path.join(os.path.dirname(__file__), "lstm_config.json")
# with open(config_path, "r") as f:
#     config = json.load(f)

# csv_paths         = config["csv_paths"]         # List of CSV file paths
# input_window      = config["input_window"]        # e.g., 168 hours of history
# forecast_horizon  = config["forecast_horizon"]    # e.g., 168 hours forecast
# train_split_frac  = config["train_split"]         # e.g., 0.75
# val_split_frac    = config["val_split"]           # e.g., 0.90
# lstm_units        = config["lstm_units"]
# learning_rate     = config["learning_rate"]
# epochs            = config["epochs"]
# batch_size        = config["batch_size"]
# model_save_path   = config["model_save_path"]

# dfs = [load_raw_data(path) for path in csv_paths]


# def df_to_X_y(df, window_size=5):
#     df_as_np = df.to_numpy()
#     X = []
#     y = []
#     for i in range(len(df_as_np)-window_size):
#         row = [[a] for a in df_as_np[i:i+window_size]]
#         X.append(row)
#         label = df_as_np[i+window_size]
#         y.append(label)
#     return np.array(X), np.array(y)


# WINDOW_SIZE = 5
# X1, y1 = df_to_X_y(temp, WINDOW_SIZE)
# X1.shape, y1.shape


# X_train1, y_train1 = X1[:60000], y1[:60000]
# X_val1, y_val1 = X1[60000:65000], y1[60000:65000]
# X_test1, y_test1 = X1[65000:], y1[65000:]
# X_train1.shape, y_train1.shape, X_val1.shape, y_val1.shape, X_test1.shape, y_test1.shape


# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import *
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.losses import MeanSquaredError
# from tensorflow.keras.metrics import RootMeanSquaredError
# from tensorflow.keras.optimizers import Adam

# model1 = Sequential()
# model1.add(InputLayer((5, 1)))
# model1.add(LSTM(64))
# model1.add(Dense(8, 'relu'))
# model1.add(Dense(1, 'linear'))

# model1.summary()



# cp1 = ModelCheckpoint('model1/', save_best_only=True)
# model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])


# model1.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=10, callbacks=[cp1])

# from tensorflow.keras.models import load_model
# model1 = load_model('model1/')


# train_predictions = model1.predict(X_train1).flatten()
# train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train1})
# train_results


# import matplotlib.pyplot as plt
# plt.plot(train_results['Train Predictions'][50:100])
# plt.plot(train_results['Actuals'][50:100])
# plt,show()



import os
import json
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# -------------------------------
# Configuration (can be read from JSON too)
# -------------------------------
config = {
    "csv_paths": ["data/raw/synthetic_power.csv"],
    "window_size": 5,
    "train_split": 0.8,
    "model_save_path": "src/models/lstm/modelv2.keras"
}

# -------------------------------
# Data Loading and Preprocessing Functions
# -------------------------------
def load_data(csv_path):
    """
    Load a CSV file and return the 'state' column as a pandas Series.
    Assumes the CSV has a header with a 'state' column.
    """
    df = pd.read_csv(csv_path)
    # Convert 'state' to numeric and fill missing values
    df["state"] = pd.to_numeric(df["state"], errors="coerce")
    df["state"] = df["state"].ffill().bfill()
    return df["state"]

def create_sliding_windows(series, window_size=5):
    """
    Create sliding windows from a pandas Series.
    
    For each window, the target is the value immediately following the window.
    
    Returns:
        X: numpy array of shape (n_samples, window_size, 1)
        y: numpy array of shape (n_samples,)
    """
    data = series.to_numpy()
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size].reshape(window_size, 1))
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

# -------------------------------
# Main Script
# -------------------------------

# Use the first CSV from the config
csv_path = config["csv_paths"][0]
series = load_data(csv_path)

# Create sliding windows
window_size = config["window_size"]
X, y = create_sliding_windows(series, window_size)

# Split data into training and test sets (80/20 split)
split_idx = int(len(X) * config["train_split"])
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

print("Train samples:", X_train.shape, y_train.shape)
print("Test samples:", X_test.shape, y_test.shape)

# -------------------------------
# Build a Simple LSTM Model
# -------------------------------
model = Sequential()
model.add(InputLayer(shape=(window_size, 1)))
model.add(LSTM(64))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="linear"))
model.summary()

# Compile the model
model.compile(loss="mean_squared_error",
            optimizer=Adam(learning_rate=0.0001),
            metrics=[tf.keras.metrics.RootMeanSquaredError()])

# Setup a checkpoint callback to save the best model
checkpoint = ModelCheckpoint(config["model_save_path"], save_best_only=True, verbose=1)

# Train the model
model.fit(X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        callbacks=[checkpoint])

# -------------------------------
# Inference: Load the Best Model and Predict
# -------------------------------
model = load_model(config["model_save_path"])
predictions = model.predict(X_train).flatten()

# Create a DataFrame for a sample of predictions vs. actual values
sample_start = 50
sample_end = 100
sample_df = pd.DataFrame({
    "Predicted": predictions[sample_start:sample_end],
    "Actual": y_train[sample_start:sample_end]
})
print(sample_df)

# Plot the predictions vs. actual values
plt.figure(figsize=(10, 5))
plt.plot(range(sample_start, sample_end), predictions[sample_start:sample_end], label="Predicted")
plt.plot(range(sample_start, sample_end), y_train[sample_start:sample_end], label="Actual")
plt.title("Predictions vs Actual (Sample from Training Set)")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.legend()
plt.show()
