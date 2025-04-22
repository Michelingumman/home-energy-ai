import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
from sklearn.preprocessing import StandardScaler # Only needed for type hint
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import joblib
import json
import logging
from pathlib import Path
import sys
import argparse
from math import ceil

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Evaluate a trained price prediction model.')
parser.add_argument('--model', type=str, default="price_weather_grid", 
                    choices=["baseline", "price_weather", "price_weather_grid"],
                    help='Model type to evaluate (default: price_weather_grid)')
parser.add_argument('--max_weeks', type=int, default=4,
                    help='Maximum number of weeks to plot in weekly comparisons (default: 4)')
args = parser.parse_args()

MODEL_NAME = args.model
MAX_WEEKS_TO_PLOT = args.max_weeks
logging.info(f"Evaluating model: {MODEL_NAME} with up to {MAX_WEEKS_TO_PLOT} weeks of plots")

# Paths to load model artifacts from
MODEL_DIR = Path(__file__).resolve().parent / f"models_v2_{MODEL_NAME}"
MODEL_PATH = MODEL_DIR / f"best_model_{MODEL_NAME}.keras"
SCALER_PATH = MODEL_DIR / f"standard_scaler_{MODEL_NAME}.save"
FEATURE_LIST_PATH = MODEL_DIR / f"feature_list_{MODEL_NAME}.json"
TEST_DF_PATH = MODEL_DIR / f"test_df_{MODEL_NAME}.csv"
X_TEST_PATH = MODEL_DIR / f"X_test_{MODEL_NAME}.npy"
# Load unscaled y_test directly
Y_TEST_UNSCALED_PATH = MODEL_DIR / f"y_test_unscaled_{MODEL_NAME}.npy"

# Constants derived during training (should match trainV2.py)
TARGET_VARIABLE = "SE3_price_ore"
SEQUENCE_LENGTH = 72
HORIZON = 24

# --- Helper Functions (Copied/adapted from trainV2.py) ---

def create_sequences(data, sequence_length, horizon, target_col_index):
    """Creates input sequences and corresponding target sequences.
       Used here to recreate the unscaled y_test.
    """
    X, y = [], []
    timestamps = [] # Store start timestamp of the target sequence y
    # Ensure data is numpy array, keep index if DataFrame
    index = None
    if isinstance(data, pd.DataFrame):
        index = data.index
        data = data.values

    for i in range(len(data) - sequence_length - horizon + 1):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length : i + sequence_length + horizon, target_col_index])
        if index is not None:
            timestamps.append(index[i + sequence_length]) # Timestamp of the first hour in y

    if index is not None:
         return np.array(X), np.array(y), timestamps
    else:
         # If no index (e.g., called on numpy array), return None for timestamps
         # This case might occur if we call it on X_test which is already numpy
         return np.array(X), np.array(y), None

def calculate_metrics(y_true, y_pred, save_prefix=f"overall"):
    """Calculates evaluation metrics for multi-output predictions."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch between y_true ({y_true.shape}) and y_pred ({y_pred.shape})")

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Calculate overall MAPE carefully
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    mask = y_true_flat != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
    else:
        mape = np.nan

    logging.info(f"Overall Metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.4f}%")
    metrics = {"mae": mae, "rmse": rmse, "mape": mape}

    # Save overall metrics
    metrics_path = MODEL_DIR / f"{save_prefix}_metrics_{MODEL_NAME}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Metrics saved to {metrics_path}")
    return metrics

def plot_results(y_true, y_pred, target_variable, num_plots=5):
    """Plots actual vs predicted results for a few examples."""
    num_samples = y_true.shape[0]
    plot_indices = np.random.choice(num_samples, size=min(num_plots, num_samples), replace=False)

    fig, axes = plt.subplots(len(plot_indices), 1, figsize=(12, 3 * len(plot_indices)), sharex=True)
    if len(plot_indices) == 1: # Handle single subplot case
        axes = [axes]

    for i, idx in enumerate(plot_indices):
        ax = axes[i]
        ax.plot(y_true[idx, :], label='Actual', marker='.')
        ax.plot(y_pred[idx, :], label='Predicted', linestyle='--', marker='x')
        ax.set_ylabel(f"{target_variable}")
        ax.set_title(f"Test Sample Index {idx} (from created sequences)")
        ax.grid(alpha=0.5)
        ax.legend()

    axes[-1].set_xlabel("Prediction Horizon (Hours)")
    plt.tight_layout()
    plot_path = MODEL_DIR / f"test_predictions_{MODEL_NAME}.png"
    plt.savefig(plot_path)
    logging.info(f"Prediction plots saved to {plot_path}")
    plt.close()

# --- New Plotting Functions ---

def determine_available_weeks(timestamps):
    """Determine how many complete weeks are available in the test dataset."""
    if not timestamps or len(timestamps) < 24*7:  # Need at least 7 days of hourly data
        return 0
    
    first_timestamp = timestamps[0]
    last_timestamp = timestamps[-1]
    total_days = (last_timestamp - first_timestamp).days
    
    # Calculate how many complete weeks are available
    complete_weeks = total_days // 7
    
    logging.info(f"Test data spans {total_days} days, containing {complete_weeks} complete weeks")
    return complete_weeks

def get_week_start_indices(timestamps, num_weeks):
    """Get starting indices for each week to plot."""
    if not timestamps or len(timestamps) < 24*7 or num_weeks <= 0:
        return []
    
    # If requesting more weeks than we have, limit to available weeks
    available_weeks = determine_available_weeks(timestamps)
    num_weeks = min(num_weeks, available_weeks)
    
    if num_weeks <= 0:
        return []
    
    # Calculate indices that would give an even spread across the test period
    indices = []
    if num_weeks == 1:
        indices = [0]  # Just the first week
    else:
        step = available_weeks / (num_weeks - 1) if num_weeks > 1 else 1
        for i in range(num_weeks):
            week_number = int(i * step)
            # Each week is 7 days * 24 hours
            index = min(week_number * 7 * 24, len(timestamps) - 7 * 24)
            indices.append(max(0, index))
    
    # Ensure indices are unique and in ascending order
    indices = sorted(list(set(indices)))
    logging.info(f"Selected {len(indices)} week starting indices: {indices}")
    return indices

def plot_weekly_comparison(timestamps, y_true, y_pred, target_variable, week_index=0):
    """Plots one week of actual vs predicted values."""
    logging.info(f"Generating weekly comparison plot for week starting around index {week_index} ({timestamps[week_index]})")

    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    timestamps = np.asarray(timestamps)

    # Select the week data based on index
    start_idx = week_index
    # Find the end index approx 7 days later
    end_timestamp_approx = timestamps[start_idx] + pd.Timedelta(days=7)
    # Find the actual index closest to 7 days later, ensure it doesn't exceed bounds
    end_idx = np.searchsorted(timestamps, end_timestamp_approx)
    end_idx = min(end_idx, len(timestamps))

    if start_idx >= end_idx or (end_idx - start_idx) < HORIZON: # Need at least one full prediction cycle
        logging.warning(f"Not enough data points ({end_idx - start_idx}) to plot a full week starting at index {week_index}. Skipping weekly plot.")
        return

    # Extract data for the selected week
    week_timestamps = timestamps[start_idx:end_idx]
    # y_true/y_pred have shape [n_samples, horizon]
    # We need the actual hourly values for the week
    # The true value for hour `t` is the first element of the y_true sequence starting at `t`
    week_y_true_hourly = y_true[start_idx:end_idx, 0]
    # The prediction for hour `t` is the first element of the y_pred sequence starting at `t`
    week_y_pred_hourly = y_pred[start_idx:end_idx, 0]

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(week_timestamps, week_y_true_hourly, label='Actual', marker='.', linestyle='-', markersize=4)
    ax.plot(week_timestamps, week_y_pred_hourly, label=f'Predicted (1-hr ahead)', linestyle='--', marker='x', markersize=4)

    ax.set_title(f"Weekly Price Comparison (Starting: {week_timestamps[0].strftime('%Y-%m-%d')})")
    ax.set_ylabel(target_variable)
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.5)

    # Format x-axis for dates
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plot_path = MODEL_DIR / f"weekly_comparison_{MODEL_NAME}_{week_timestamps[0].strftime('%Y%m%d')}.png"
    plt.savefig(plot_path)
    logging.info(f"Weekly comparison plot saved to {plot_path}")
    plt.close()

def plot_multi_week_comparisons(timestamps, y_true, y_pred, target_variable, max_weeks=4):
    """Generates multiple weekly comparison plots across the test period."""
    if not timestamps:
        logging.warning("No timestamps available for weekly plots.")
        return
    
    # Get indices for week starts
    week_indices = get_week_start_indices(timestamps, max_weeks)
    
    if not week_indices:
        logging.warning("No suitable week start indices found. Skipping weekly plots.")
        return
    
    logging.info(f"Generating {len(week_indices)} weekly comparison plots...")
    
    for idx in week_indices:
        plot_weekly_comparison(timestamps, y_true, y_pred, target_variable, week_index=idx)
    
    logging.info(f"Completed generating {len(week_indices)} weekly comparison plots.")

def plot_error_distribution(y_true, y_pred):
    """Plots the distribution of prediction errors."""
    errors = (y_true - y_pred).flatten()
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, density=True, alpha=0.75)
    plt.title("Distribution of Prediction Errors (All Horizon Steps)")
    plt.xlabel("Error (Actual - Predicted)")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.5)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    plt.axvline(mean_error, color='r', linestyle='dashed', linewidth=1, label=f'Mean Error: {mean_error:.2f}')
    plt.legend()
    plt.tight_layout()
    plot_path = MODEL_DIR / f"error_distribution_{MODEL_NAME}.png"
    plt.savefig(plot_path)
    logging.info(f"Error distribution plot saved to {plot_path}")
    plt.close()

def plot_metrics_by_horizon(y_true, y_pred):
    """Calculates and plots MAE, RMSE, MAPE for each forecast horizon step."""
    mae_per_step = []
    rmse_per_step = []
    mape_per_step = []
    horizon_steps = range(1, y_true.shape[1] + 1)

    for i in range(y_true.shape[1]): # Iterate through horizon steps 0 to 23
        y_true_step = y_true[:, i]
        y_pred_step = y_pred[:, i]

        mae_per_step.append(mean_absolute_error(y_true_step, y_pred_step))
        rmse_per_step.append(np.sqrt(mean_squared_error(y_true_step, y_pred_step)))

        mask = y_true_step != 0
        if np.any(mask):
            step_mape = np.mean(np.abs((y_true_step[mask] - y_pred_step[mask]) / y_true_step[mask])) * 100
            mape_per_step.append(step_mape)
        else:
            mape_per_step.append(np.nan)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(horizon_steps, mae_per_step, marker='.')
    axes[0].set_title("MAE per Forecast Horizon Step")
    axes[0].set_ylabel("MAE")
    axes[0].grid(True, alpha=0.5)

    axes[1].plot(horizon_steps, rmse_per_step, marker='.')
    axes[1].set_title("RMSE per Forecast Horizon Step")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(True, alpha=0.5)

    axes[2].plot(horizon_steps, mape_per_step, marker='.')
    axes[2].set_title("MAPE per Forecast Horizon Step")
    axes[2].set_ylabel("MAPE (%)")
    axes[2].set_xlabel("Forecast Horizon (Hours)")
    axes[2].grid(True, alpha=0.5)

    plt.tight_layout()
    plot_path = MODEL_DIR / f"metrics_by_horizon_{MODEL_NAME}.png"
    plt.savefig(plot_path)
    logging.info(f"Metrics by horizon plot saved to {plot_path}")
    plt.close()

# --- Main Evaluation Script ---
if __name__ == "__main__":
    logging.info(f"Starting evaluation script for {MODEL_NAME} model...")

    # 1. Load Artifacts
    logging.info(f"Loading {MODEL_NAME} evaluation artifacts...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        # Scaler might be None if only target was present during training
        scaler = None
        if SCALER_PATH.exists():
            scaler = joblib.load(SCALER_PATH)
            logging.info("Loaded scaler.")
        else:
            logging.warning("Scaler file not found. Assuming no input features were scaled.")

        with open(FEATURE_LIST_PATH, 'r') as f:
            feature_names = json.load(f)
        # Load test_df with datetime index
        test_df = pd.read_csv(TEST_DF_PATH, index_col=0, parse_dates=True)
        X_test = np.load(X_TEST_PATH)
        y_test_unscaled = np.load(Y_TEST_UNSCALED_PATH)
        logging.info(f"Loaded model, features ({len(feature_names)}), test_df ({test_df.shape}), X_test ({X_test.shape}), y_test_unscaled ({y_test_unscaled.shape})")
    except FileNotFoundError as e:
        logging.error(f"Error loading artifact: {e}. Ensure {MODEL_NAME} trainV2.py ran successfully.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred loading artifacts: {e}")
        sys.exit(1)

    # 2. Predict
    logging.info("Generating predictions on X_test...")
    y_pred = model.predict(X_test)

    # 3. Recreate Timestamps
    logging.info("Recreating timestamps from test_df for alignment...")
    if list(test_df.columns) != feature_names:
        logging.warning("Columns in loaded test_df do not match loaded feature list. Reordering test_df.")
        try:
            test_df = test_df[feature_names]
        except KeyError as e:
            logging.error(f"Cannot reorder test_df columns. Missing feature: {e}")
            sys.exit(1)

    try:
         target_col_index_for_ts = feature_names.index(TARGET_VARIABLE)
         _, _, test_timestamps = create_sequences(test_df, SEQUENCE_LENGTH, HORIZON, target_col_index_for_ts)
    except Exception as e:
         logging.error(f"Error recreating timestamps from test_df: {e}")
         test_timestamps = None

    # Align lengths
    num_predictions = len(y_pred)
    if num_predictions != len(y_test_unscaled):
        logging.warning(f"Length mismatch: Predictions={num_predictions}, Loaded True Targets={len(y_test_unscaled)}. Adjusting...")
        min_len = min(num_predictions, len(y_test_unscaled))
        y_pred = y_pred[:min_len]
        y_test_unscaled = y_test_unscaled[:min_len]
        if test_timestamps:
             test_timestamps = test_timestamps[:min_len]
        if len(X_test) != num_predictions:
             logging.warning(f"Original X_test length ({len(X_test)}) differs from prediction length ({num_predictions}).")

    if not test_timestamps:
        logging.error("Failed to retrieve timestamps for test data. Cannot generate weekly plot.")

    logging.info(f"Aligned shapes: y_pred={y_pred.shape}, y_test_unscaled={y_test_unscaled.shape}, timestamps={len(test_timestamps) if test_timestamps else 'N/A'}")

    # 4. Calculate Overall Metrics
    logging.info("Calculating overall metrics...")
    metrics = calculate_metrics(y_test_unscaled, y_pred, save_prefix="overall")

    # 5. Generate Enhanced Plots
    logging.info("Generating evaluation plots...")
    if test_timestamps:
        plot_multi_week_comparisons(test_timestamps, y_test_unscaled, y_pred, TARGET_VARIABLE, max_weeks=MAX_WEEKS_TO_PLOT)
    else:
        logging.warning("Skipping weekly comparison plots due to missing timestamps.")
    plot_results(y_test_unscaled, y_pred, TARGET_VARIABLE)
    plot_error_distribution(y_test_unscaled, y_pred)
    plot_metrics_by_horizon(y_test_unscaled, y_pred)

    logging.info(f"Evaluation for {MODEL_NAME} model finished.")
