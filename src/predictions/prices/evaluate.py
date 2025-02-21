import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from datetime import datetime

def predict_single_day(target_date):
    # Load the saved model and scaler
    model = tf.keras.models.load_model("C:/_Projects/home-energy-ai/models/saved/large_delta_price_model.keras")
    price_scaler = joblib.load("C:/_Projects/home-energy-ai/models/saved/price_scaler.save")

    # Load test data and timestamps
    X_test = np.load("C:/_Projects/home-energy-ai/models/test_data/X_test.npy")
    y_test = np.load("C:/_Projects/home-energy-ai/models/test_data/y_test.npy")
    timestamps = np.load("C:/_Projects/home-energy-ai/models/test_data/test_timestamps.npy", allow_pickle=True)
    timestamps = pd.to_datetime(timestamps)

    # Print available date range
    print(f"\nAvailable date range for predictions:")
    print(f"From: {timestamps.min().strftime('%Y-%m-%d')}")
    print(f"To: {timestamps.max().strftime('%Y-%m-%d')}\n")

    # Convert user input to a date and find matching samples
    target_date = pd.to_datetime(target_date)
    print(f"Evaluating predictions for: {target_date.strftime('%Y-%m-%d')}")
    
    # Find samples that fall on the chosen date
    target_day_start = pd.Timestamp(target_date.date())
    target_day_end = target_day_start + pd.Timedelta(days=1)
    matching_mask = (timestamps >= target_day_start) & (timestamps < target_day_end)
    matching_indices = np.where(matching_mask)[0]
    
    if len(matching_indices) == 0:
        print(f"\nNo predictions found for {target_date.strftime('%Y-%m-%d')}.")
        print("Please choose a date within the available range.")
        return

    # Among all matches, try to pick the one starting exactly at midnight (hour == 0).
    # If none start at midnight, just pick the earliest one of that date.
    day_timestamps = timestamps[matching_indices]
    midnight_mask = day_timestamps.hour == 0
    if midnight_mask.any():
        sample_index = matching_indices[day_timestamps.hour == 0][0]
    else:
        sample_index = matching_indices[0]

    # Create figure for plotting
    plt.figure(figsize=(15, 6))

    # Get the sample and make prediction
    X_sample = X_test[sample_index:sample_index+1]
    y_true = y_test[sample_index]            # shape: (24,)
    y_pred = model.predict(X_sample, verbose=0)[0]  # shape: (24,)

    # Inverse transform
    num_features = len(price_scaler.scale_)
    dummy_pred = np.zeros((len(y_pred), num_features))
    dummy_true = np.zeros((len(y_true), num_features))
    dummy_pred[:, 0] = y_pred
    dummy_true[:, 0] = y_true

    y_pred_inv = price_scaler.inverse_transform(dummy_pred)[:, 0]
    y_true_inv = price_scaler.inverse_transform(dummy_true)[:, 0]

    # Create hour labels for x-axis
    start_time = timestamps[sample_index]
    hours = pd.date_range(start_time, periods=24, freq='h')

    # Plot predictions vs. actual using a step plot
    plt.step(hours, y_true_inv, 'b-', label='Actual')
    plt.step(hours, y_pred_inv, 'r--', label='Predicted')
    plt.title(f'24-Hour Price Prediction Starting {start_time.strftime("%Y-%m-%d %H:%M")}')
    plt.xlabel('Time')
    plt.ylabel('Price (öre/kWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # Calculate and print summary statistics for the day's prediction
    error = y_pred_inv - y_true_inv
    mape = np.mean(np.abs(error / y_true_inv)) * 100
    rmse = np.sqrt(np.mean(error**2))
    print("\nPrediction Summary:")
    print(f"Date: {start_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"MAPE: {mape:.2f}%")
    print(f"RMSE: {rmse:.2f} öre/kWh")

def predict_month(target_month):
    """
    Predict and display 24-hour forecasts for each day in the given month.
    target_month should be in "YYYY-MM" format (e.g., "2023-08").
    """
    # Load model, scaler, test data and timestamps
    model = tf.keras.models.load_model("C:/_Projects/home-energy-ai/models/saved/large_delta_price_model.keras")
    price_scaler = joblib.load("C:/_Projects/home-energy-ai/models/saved/price_scaler.save")
    X_test = np.load("C:/_Projects/home-energy-ai/models/test_data/X_test.npy")
    y_test = np.load("C:/_Projects/home-energy-ai/models/test_data/y_test.npy")
    timestamps = np.load("C:/_Projects/home-energy-ai/models/test_data/test_timestamps.npy", allow_pickle=True)
    timestamps = pd.to_datetime(timestamps)

    # Print available date range
    print(f"\nAvailable date range for predictions:")
    print(f"From: {timestamps.min().strftime('%Y-%m-%d')}")
    print(f"To: {timestamps.max().strftime('%Y-%m-%d')}\n")

    # Parse target month (e.g., "2023-08")
    try:
        target_month_start = pd.to_datetime(target_month + "-01")
    except Exception as e:
        print("Invalid month format. Please use YYYY-MM")
        return
    target_month_end = target_month_start + pd.DateOffset(months=1)

    # Filter indices to those within the target month
    month_mask = (timestamps >= target_month_start) & (timestamps < target_month_end)
    month_indices = np.where(month_mask)[0]
    if len(month_indices) == 0:
        print(f"No predictions found for {target_month}.")
        return

    # Group the indices by day using the date part of the timestamp
    df_indices = pd.DataFrame({'index': month_indices, 'timestamp': timestamps[month_indices]})
    df_indices['date'] = df_indices['timestamp'].dt.date
    unique_days = df_indices['date'].unique()

    all_times = []
    all_actual = []
    all_pred = []
    error_list = []

    plt.figure(figsize=(15, 6))
    
    for i, day in enumerate(unique_days):
        # Get indices for the current day
        day_indices = df_indices[df_indices['date'] == day]['index'].values
        day_timestamps = timestamps[day_indices]
        
        # Try to pick the sample starting at midnight; if not, use the earliest sample of that day
        midnight_mask = day_timestamps.hour == 0
        if midnight_mask.any():
            sample_index = day_indices[day_timestamps.hour == 0][0]
        else:
            sample_index = day_indices[0]

        # Get prediction for the sample
        X_sample = X_test[sample_index:sample_index+1]
        y_true = y_test[sample_index]    # shape: (24,)
        y_pred = model.predict(X_sample, verbose=0)[0]  # shape: (24,)

        num_features = len(price_scaler.scale_)
        dummy_pred = np.zeros((len(y_pred), num_features))
        dummy_true = np.zeros((len(y_true), num_features))
        dummy_pred[:, 0] = y_pred
        dummy_true[:, 0] = y_true

        y_pred_inv = price_scaler.inverse_transform(dummy_pred)[:, 0]
        y_true_inv = price_scaler.inverse_transform(dummy_true)[:, 0]

        # Create hourly timestamps for this 24-hour prediction
        start_time = timestamps[sample_index]
        hours = pd.date_range(start_time, periods=24, freq='h')

        # Accumulate data for overall error metrics
        all_times.extend(hours)
        all_actual.extend(y_true_inv)
        all_pred.extend(y_pred_inv)
        error = y_pred_inv - y_true_inv
        error_list.extend(error)

        # Plot this day's forecast. Only label the first day's lines.
        if i == 0:
            plt.step(hours, y_true_inv, 'b-', label='Actual')
            plt.step(hours, y_pred_inv, 'r--', label='Predicted')
        else:
            plt.step(hours, y_true_inv, 'b-', alpha=0.7)
            plt.step(hours, y_pred_inv, 'r--', alpha=0.7)
    
    plt.title(f'24-Hour Price Predictions for Each Day in {target_month}')
    plt.xlabel('Time')
    plt.ylabel('Price (öre/kWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)


    plt.tight_layout()
    plt.show()

    # Calculate and print overall summary statistics
    all_actual = np.array(all_actual)
    all_pred = np.array(all_pred)
    mape = np.mean(np.abs((all_actual - all_pred) / all_actual)) * 100
    rmse = np.sqrt(np.mean((all_actual - all_pred)**2))
    print("\nPrediction Summary for the Month:")
    print(f"MAPE: {mape:.2f}%")
    print(f"RMSE: {rmse:.2f} öre/kWh")


def predict_rolling_forecast(target_date, horizon=24):
    # Load model, scaler, test data, and timestamps
    model = tf.keras.models.load_model("C:/_Projects/home-energy-ai/models/saved/large_delta_price_model.keras")
    price_scaler = joblib.load("C:/_Projects/home-energy-ai/models/saved/price_scaler.save")
    X_test = np.load("C:/_Projects/home-energy-ai/models/test_data/X_test.npy")
    y_test = np.load("C:/_Projects/home-energy-ai/models/test_data/y_test.npy")
    timestamps = np.load("C:/_Projects/home-energy-ai/models/test_data/test_timestamps.npy", allow_pickle=True)
    timestamps = pd.to_datetime(timestamps)
    
    # Find the sample for the target date (preferably one starting at midnight)
    target_date = pd.to_datetime(target_date)
    target_day_start = pd.Timestamp(target_date.date())
    target_day_end = target_day_start + pd.Timedelta(days=1)
    matching_mask = (timestamps >= target_day_start) & (timestamps < target_day_end)
    matching_indices = np.where(matching_mask)[0]
    if len(matching_indices) == 0:
        print(f"No sample found for {target_date.strftime('%Y-%m-%d')}.")
        return
    
    day_timestamps = timestamps[matching_indices]
    midnight_mask = day_timestamps.hour == 0
    if midnight_mask.any():
        sample_index = matching_indices[day_timestamps.hour == 0][0]
    else:
        sample_index = matching_indices[0]
    
    # Get the initial input window (shape: 1 x window_size x num_features)
    current_window = X_test[sample_index:sample_index+1].copy()
    
    rolling_preds = []
    
    # Iteratively forecast one hour ahead until reaching the desired horizon.
    # The model is trained to predict 24 hours, so we take only the first predicted hour.
    for i in range(horizon):
        pred_block = model.predict(current_window, verbose=0)[0]  # shape: (24,)
        next_pred_scaled = pred_block[0]  # forecast for the next hour
        rolling_preds.append(next_pred_scaled)
        
        # Update the window:
        # Drop the oldest hour and append a new row.
        # For the new row, use the last row of current_window as baseline,
        # but update its target (first feature) with the predicted value.
        new_row = current_window[0, -1, :].copy()
        new_row[0] = next_pred_scaled
        updated_window = np.concatenate([current_window[0, 1:, :], new_row[np.newaxis, :]], axis=0)
        current_window = updated_window[np.newaxis, :, :]  # reshape back to (1, window_size, num_features)
    
    # Inverse transform the forecasted target values
    num_features = len(price_scaler.scale_)
    dummy = np.zeros((horizon, num_features))
    dummy[:, 0] = np.array(rolling_preds)
    forecast_inv = price_scaler.inverse_transform(dummy)[:, 0]
    
    # Get the actual values for the same forecast horizon from y_test.
    # y_test[sample_index] is a 24-hour block (or longer) so we slice the first 'horizon' hours.
    actual_scaled = y_test[sample_index][:horizon]
    dummy_actual = np.zeros((horizon, num_features))
    dummy_actual[:, 0] = actual_scaled
    actual_inv = price_scaler.inverse_transform(dummy_actual)[:, 0]
    
    # Prepare time axis for plotting (forecast starting at the chosen sample time)
    start_time = timestamps[sample_index]
    forecast_times = pd.date_range(start_time, periods=horizon, freq='H')
    
    # Plot the rolling forecast together with the actual data
    plt.figure(figsize=(15, 6))
    plt.step(forecast_times, actual_inv, 'b-', label='Actual')
    plt.step(forecast_times, forecast_inv, 'r--', label='Rolling Forecast')
    plt.title(f'Iterative 1-Hour Rolling Forecast for {target_date.strftime("%Y-%m-%d")}')
    plt.xlabel('Time')
    plt.ylabel('Price (öre/kWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Print the forecasted values alongside actual values
    print("Time\t\tActual\tForecast")
    for t, act, pred in zip(forecast_times, actual_inv, forecast_inv):
        print(f"{t}: {act:.2f} öre/kWh vs {pred:.2f} öre/kWh")



# Example usage:
predict_single_day("2024-01-20")

predict_month("2024-01")

predict_rolling_forecast("2023-10-22", horizon=24)