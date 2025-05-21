# Predict file for demand predictions
import pandas as pd
import numpy as np
import logging
import pickle
import os
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import subprocess
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants - Mirroring train.py for consistency where applicable
MODEL_LOAD_PATH = 'src/predictions/demand/models/villamichelin_demand_model.pkl'
HMM_MODEL_LOAD_PATH = 'src/predictions/demand/models/villamichelin_hmm_model.pkl'
CONSUMPTION_DATA_PATH = 'data/processed/villamichelin/VillamichelinEnergyData.csv'
HEAT_PUMP_DATA_PATH = 'data/processed/villamichelin/Thermia/HeatPumpPower.csv'
WEATHER_DATA_PATH = 'data/processed/weather_data.csv'

# Weather forecast paths
WEATHER_FORECAST_DIR = 'data/processed/forecasts/weather'
WEATHER_FORECAST_PATH = f'{WEATHER_FORECAST_DIR}/{datetime.now().strftime("%Y-%m-%d_4days")}.csv'
FETCH_WEATHER_SCRIPT = 'src/predictions/demand/FetchWeatherData.py'

# Predictions paths
PREDICTIONS_DIR = 'src/predictions/demand/predictions'
PREDICTIONS_SAVE_PATH = f'{PREDICTIONS_DIR}/demand_predictions.csv'
PLOTS_SAVE_DIR = 'src/predictions/demand/plots/predictions/'

# Ensure directories exist
os.makedirs(WEATHER_FORECAST_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(PLOTS_SAVE_DIR, exist_ok=True)

# Set nicer plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.figsize'] = (14, 7)

# Add project root to path to import from train.py
ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Import necessary functions from train.py (similar to evaluate.py)
from src.predictions.demand.train import (
    run_data_pipeline, 
    fit_hmm, 
    decode_states, 
    get_state_posteriors,
    add_hmm_features,
    add_lagged_features,
    add_calendar_features,
    add_weather_transforms,
    add_interaction_terms,
    engineer_features,
    TARGET_COL,
    N_HMM_STATES,
    FORECAST_HORIZON
)

def ensure_weather_forecast():
    """Check if weather forecast file exists, if not run FetchWeatherData.py to create it."""
    if not os.path.exists(WEATHER_FORECAST_PATH):
        logging.info(f"Weather forecast file not found: {WEATHER_FORECAST_PATH}")
        logging.info("Running FetchWeatherData.py to generate forecast...")
        
        # Construct the command to run FetchWeatherData.py
        python_executable = sys.executable
        fetch_weather_cmd = [python_executable, FETCH_WEATHER_SCRIPT, "--forecast", "--days", "4"]
        
        try:
            # Run the command
            subprocess.run(fetch_weather_cmd, check=True)
            logging.info("Weather forecast generated successfully")
            
            # Verify the file was created
            if not os.path.exists(WEATHER_FORECAST_PATH):
                raise FileNotFoundError(f"Weather forecast file still not found after running FetchWeatherData.py: {WEATHER_FORECAST_PATH}")
                
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running FetchWeatherData.py: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error ensuring weather forecast: {e}")
            raise
    else:
        logging.info(f"Weather forecast file found: {WEATHER_FORECAST_PATH}")

def load_model(model_path: str):
    """Load a previously trained model from disk."""
    logging.info(f"Loading model from {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise
        
def load_hmm_model(model_path: str):
    """Load a previously trained HMM model from disk."""
    logging.info(f"Loading HMM model from {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logging.info("HMM model loaded successfully.")
        return model
    except Exception as e:
        logging.warning(f"Could not load HMM model: {e}. Will fit a new one if needed.")
        return None

def load_weather_forecast(file_path: str, start_time=None, horizon_hours=24) -> pd.DataFrame:
    """Load weather forecast data for the prediction period."""
    logging.info(f"Loading weather forecast data from {file_path}")
    try:
        df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
        df.index = df.index.tz_convert('Europe/Stockholm')
        
        # Select only required forecast data based on start time and horizon
        if start_time is not None:
            end_time = start_time + pd.Timedelta(hours=horizon_hours)
            df = df.loc[start_time:end_time].copy()
            
        # Ensure we have the same columns as in training
        required_cols = ['temperature_2m', 'cloud_cover', 'relative_humidity_2m', 
                         'wind_speed_100m', 'shortwave_radiation_sum']
        for col in required_cols:
            if col not in df.columns:
                logging.warning(f"Missing required weather column: {col}")
                
        logging.info(f"Weather forecast data loaded. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading weather forecast data: {e}")
        raise

def prepare_data_for_prediction(consumption_df, heat_pump_df, weather_df, horizon_hours=24):
    """Prepare the data for prediction by creating a future dataframe with weather forecasts."""
    # Get the latest timestamp from consumption data
    latest_timestamp = consumption_df.index.max()
    
    # Create a date range for future predictions
    future_index = pd.date_range(
        start=latest_timestamp + pd.Timedelta(hours=1),
        periods=horizon_hours,
        freq='h'
    )
    
    # Create future dataframe with NaNs for consumption and heat pump
    future_df = pd.DataFrame(index=future_index)
    future_df[TARGET_COL] = np.nan
    future_df['power_input_kwh'] = np.nan
    
    # Join with weather forecast for the future period
    future_df = future_df.join(weather_df)
    
    # Combine historical data with future dataframe
    combined_df = pd.concat([consumption_df.join(heat_pump_df), future_df])
    
    return combined_df

def generate_future_features(df, historical_only=False, country_code='SE'):
    """Generate features for future prediction periods with enhanced handling for forecasting."""
    logging.info("Generating features for prediction")
    
    # Apply HMM to the historical data
    historical_df = df[df[TARGET_COL].notna()].copy()
    
    # Try to load pretrained HMM model first
    hmm_model = load_hmm_model(HMM_MODEL_LOAD_PATH)
    
    if hmm_model is None:
        # Fit HMM model on historical data
        logging.info("Fitting new HMM model on historical data")
        hmm_model = fit_hmm(historical_df[TARGET_COL], N_HMM_STATES)
    
    # Get states and posteriors
    states = decode_states(hmm_model, historical_df[TARGET_COL])
    posteriors = get_state_posteriors(hmm_model, historical_df[TARGET_COL])
    
    # Map states by consumption level
    mean_consumptions = hmm_model.means_.flatten()
    state_order = np.argsort(mean_consumptions)
    state_map = {old_label: new_label for new_label, old_label in enumerate(state_order)}

    # Add HMM features to historical part
    historical_df['hmm_state'] = np.array([state_map[s] for s in states])
    posterior_probs_sorted = posteriors[:, state_order]
    for i in range(N_HMM_STATES):
        historical_df[f'hmm_state_posterior_{i}'] = posterior_probs_sorted[:, i]
    
    # Save the last state for future prediction
    last_state = historical_df['hmm_state'].iloc[-1]
    last_posteriors = {f'hmm_state_posterior_{i}': historical_df[f'hmm_state_posterior_{i}'].iloc[-1] 
                      for i in range(N_HMM_STATES)}
    
    # For prediction, compute historical averages that will be useful for imputation
    if not historical_only:
        # Calculate average consumption by hour of day
        historical_df['hour'] = historical_df.index.hour
        hour_avg = historical_df.groupby('hour')[TARGET_COL].mean().to_dict()
        
        # Calculate average consumption by day of week
        historical_df['dayofweek'] = historical_df.index.dayofweek
        dow_avg = historical_df.groupby('dayofweek')[TARGET_COL].mean().to_dict()
        
        # Calculate overall average consumption
        overall_avg = historical_df[TARGET_COL].mean()
    
    # Create complete dataframe with HMM features
    if historical_only:
        df_features = historical_df.copy()
    else:
        # Forward fill HMM states to future periods
        full_df = df.copy()
        future_mask = full_df[TARGET_COL].isna()
        
        # Add computed averages for future reference during imputation
        full_df['consumption_avg'] = overall_avg
        full_df['hour'] = full_df.index.hour
        full_df['dayofweek'] = full_df.index.dayofweek
        
        full_df['consumption_hour_avg'] = full_df['hour'].map(hour_avg).fillna(overall_avg)
        full_df['consumption_dow_avg'] = full_df['dayofweek'].map(dow_avg).fillna(overall_avg)
        
        # Add HMM features with last known state
        full_df['hmm_state'] = last_state  # Use last known state
        for i in range(N_HMM_STATES):
            full_df[f'hmm_state_posterior_{i}'] = last_posteriors[f'hmm_state_posterior_{i}']
        
        # Copy HMM features from historical data
        full_df.loc[~future_mask, 'hmm_state'] = historical_df['hmm_state']
        for i in range(N_HMM_STATES):
            full_df.loc[~future_mask, f'hmm_state_posterior_{i}'] = historical_df[f'hmm_state_posterior_{i}']
        
        df_features = full_df
    
    # Add standard features
    logging.info("Adding calendar features")
    df_features = add_calendar_features(df_features, country_code)
    
    logging.info("Adding weather transforms")
    df_features = add_weather_transforms(df_features)
    
    logging.info("Adding lagged features")
    df_features = add_lagged_features(df_features, TARGET_COL)
    
    logging.info("Adding interaction features")
    df_features = add_interaction_terms(df_features)
    
    # For future periods, pre-fill problematic features with sensible defaults
    if not historical_only:
        future_mask = df_features[TARGET_COL].isna()
        future_df = df_features[future_mask]
        
        if not future_df.empty:
            # Add simple time-based consumption estimates for future periods
            # These can help with lag features that might be missing
            for idx in future_df.index:
                hour = idx.hour
                day = idx.dayofweek
                df_features.loc[idx, TARGET_COL] = hour_avg.get(hour, overall_avg) * 0.8  # 80% of hour average as initial guess
        
            # Pre-fill any _diff features (that compare to 24h ago) with zeros
            diff_cols = [col for col in df_features.columns if '_diff' in col]
            for col in diff_cols:
                df_features.loc[future_mask, col] = df_features.loc[future_mask, col].fillna(0)
    
    logging.info(f"Features generated. Shape: {df_features.shape}")
    return df_features

def make_predictions(model, df_features, forecast_horizon=24):
    """Make predictions using the trained model with enhanced handling of missing features."""
    logging.info(f"Making predictions for {forecast_horizon} hours")
    
    # Initialize prediction dataframe
    df_pred = df_features.copy()
    df_pred['predictions'] = np.nan
    
    # Identify future periods where we need to make predictions
    future_mask = df_pred[TARGET_COL].isna()
    future_indices = df_pred.index[future_mask]
    
    if len(future_indices) == 0:
        logging.warning("No future periods to predict")
        return df_pred
    
    # Get model feature names
    feature_names = model.feature_names_in_
    
    # Create a buffer for holding prediction results
    predictions = []
    prediction_indices = []
    
    # For each future period, make predictions iteratively to account for lag features
    for i, idx in enumerate(future_indices):
        # Extract features for this timestamp
        X = df_pred.loc[idx:idx, feature_names].copy()
        
        # If any features are missing, handle them more aggressively
        if X.isna().any().any():
            missing_cols = X.columns[X.isna().any()]
            logging.warning(f"Missing features for prediction at {idx}: {len(missing_cols)} features")
            
            # Split missing features by type for better handling
            missing_consumption = [col for col in missing_cols if col.startswith('consumption')]
            missing_power = [col for col in missing_cols if col.startswith('power_input')]
            missing_interactions = [col for col in missing_cols if '_x_' in col]
            missing_diff = [col for col in missing_cols if col.endswith('_diff')]
            missing_other = [col for col in missing_cols if col not in missing_consumption + 
                            missing_power + missing_interactions + missing_diff]
            
            # 1. Forward fill from previous known values
            X = X.ffill().bfill()
            
            # 2. Apply special handling for different feature types
            # For consumption features, use zero or historical averages
            for col in missing_consumption:
                if 'min' in col or 'max' in col:
                    # For min/max features, use the average consumption if available
                    if 'consumption_avg' in df_pred.columns and not df_pred['consumption_avg'].isna().all():
                        X[col] = df_pred['consumption_avg'].iloc[-1] if len(df_pred['consumption_avg']) > 0 else 0
                    else:
                        X[col] = 0
                elif 'ratio' in col:
                    # For ratio features, use 1.0 (neutral value)
                    X[col] = 1.0
                else:
                    # For other consumption features, use zero
                    X[col] = 0
            
            # For power features
            for col in missing_power:
                X[col] = 0  # Default to zero power input
            
            # For interaction features
            for col in missing_interactions:
                X[col] = 0  # Default to zero for interactions
            
            # For difference features
            for col in missing_diff:
                X[col] = 0  # Default to zero difference

            # For remaining features, use mean of non-NaN values or zero
            for col in missing_other:
                if not df_pred[col].isna().all():
                    X[col] = df_pred[col].dropna().mean()
                else:
                    X[col] = 0
        
        # Final check for any remaining NaNs
        if X.isna().any().any():
            still_missing = X.columns[X.isna().any()].tolist()
            logging.warning(f"Still missing {len(still_missing)} features after imputation, defaulting to zeros")
            X = X.fillna(0)  # Last resort: fill any remaining NaNs with zeros
        
        # Make prediction
        try:
            pred = model.predict(X)[0]
            predictions.append(pred)
            prediction_indices.append(idx)
            
            # Update the target column with prediction for next-step forecasting
            df_pred.loc[idx, 'predictions'] = pred
            df_pred.loc[idx, TARGET_COL] = pred
            
            # Recalculate lag features if we have more predictions to make
            if i < len(future_indices) - 1:
                # Only update lag features that would be affected by this new prediction
                lag_cols = [col for col in feature_names if '_lag_' in col and col.split('_lag_')[0] == TARGET_COL]
                for lag_col in lag_cols:
                    lag = int(lag_col.split('_lag_')[1])
                    lag_idx = idx + pd.Timedelta(hours=lag)
                    if lag_idx in df_pred.index:
                        df_pred.loc[lag_idx, lag_col] = pred
        except Exception as e:
            logging.error(f"Error making prediction at {idx}: {e}")
    
    logging.info(f"Successfully made {len(predictions)} predictions out of {len(future_indices)} requested")
    
    return df_pred

def plot_predictions(df_pred, historical_window=48, save_path=None):
    """Plot the predictions along with historical data."""
    logging.info("Plotting predictions")
    
    # Check if there are any predictions
    has_valid_predictions = df_pred['predictions'].notna().sum() > 0
    
    # Find the boundary between historical and prediction data
    historical_data = df_pred[df_pred[TARGET_COL].notna() & (df_pred['predictions'].isna())]
    future_data = df_pred[df_pred[TARGET_COL].isna()]
    
    if len(historical_data) == 0 and len(future_data) == 0:
        logging.error("No data to plot")
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.text(0.5, 0.5, 'No data to display', ha='center', va='center', fontsize=16)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig, ax
    
    # If we have future data but no predictions, create a baseline prediction
    if not has_valid_predictions and len(future_data) > 0:
        logging.warning("No valid predictions to plot. Creating baseline prediction from historical patterns.")
        
        # Create a baseline prediction based on historical patterns if available
        if 'consumption_hour_avg' in df_pred.columns:
            df_pred.loc[future_data.index, 'baseline_pred'] = df_pred.loc[future_data.index, 'consumption_hour_avg']
        elif 'consumption_avg' in df_pred.columns:
            df_pred.loc[future_data.index, 'baseline_pred'] = df_pred.loc[future_data.index, 'consumption_avg']
        else:
            # If we don't have even historical averages, use a simple average of historical data
            historical_mean = df_pred[df_pred[TARGET_COL].notna()][TARGET_COL].mean()
            df_pred.loc[future_data.index, 'baseline_pred'] = historical_mean
    
    # Determine the prediction start point
    if has_valid_predictions:
        pred_start = df_pred.index[df_pred['predictions'].notna()][0]
    elif len(future_data) > 0:
        pred_start = future_data.index[0]
    else:
        pred_start = df_pred.index[-1]
    
    # Calculate historical start point
    hist_start = pred_start - pd.Timedelta(hours=historical_window)
    if hist_start < df_pred.index[0]:
        hist_start = df_pred.index[0]
    
    # Get the data to plot
    plot_df = df_pred.loc[hist_start:].copy()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot historical consumption
    historical_mask = plot_df.index < pred_start
    if historical_mask.any():
        ax.plot(plot_df.index[historical_mask], 
                plot_df.loc[historical_mask, TARGET_COL], 
                label='Historical Consumption', 
                color='#2C3E50', 
                linewidth=2)
    
    # Plot predictions or baseline
    pred_mask = plot_df.index >= pred_start
    if has_valid_predictions and pred_mask.any():
        ax.plot(plot_df.index[pred_mask], 
                plot_df.loc[pred_mask, 'predictions'], 
                label='Predicted Consumption', 
                color='#E74C3C', 
                linewidth=2)
    elif 'baseline_pred' in plot_df.columns and pred_mask.any():
        ax.plot(plot_df.index[pred_mask],
                plot_df.loc[pred_mask, 'baseline_pred'],
                label='Baseline Prediction (Historical Average)',
                color='#F39C12',
                linestyle='--',
                linewidth=2)
    
    # Add vertical line to mark prediction start
    ax.axvline(x=pred_start, color='gray', linestyle='--', alpha=0.7)
    
    # Get a reasonable y-value for the text
    if historical_mask.any() and not plot_df.loc[historical_mask, TARGET_COL].empty:
        text_y = plot_df.loc[historical_mask, TARGET_COL].max() * 0.95
    elif 'baseline_pred' in plot_df.columns:
        text_y = plot_df['baseline_pred'].max() * 0.95
    else:
        text_y = 1.0
    
    ax.text(pred_start, text_y, 
            'Prediction Start', 
            rotation=90, 
            verticalalignment='top')
    
    # Formatting
    ax.set_title('Energy Consumption Forecast', fontsize=16)
    ax.set_ylabel('Energy Consumption', fontsize=12)
    ax.set_xlabel('Time', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Plot saved to {save_path}")
    
    return fig, ax

def save_predictions(df_pred, save_path):
    """Save the predictions to a CSV file."""
    logging.info(f"Saving predictions to {save_path}")
    
    # Extract only relevant columns
    save_df = df_pred[['predictions']].copy()
    save_df.loc[~save_df['predictions'].isna()].to_csv(save_path)
    
    logging.info(f"Predictions saved to {save_path}")
    return save_df

def main():
    parser = argparse.ArgumentParser(description="Predict energy consumption for future periods")
    parser.add_argument('--horizon', type=int, default=24,
                        help="Number of hours to predict ahead. Default: 24")
    parser.add_argument('--plot', action='store_true', 
                        help="Generate and show plots of the predictions")
    parser.add_argument('--save-plot', type=str, default=None,
                        help="Save plot to specified path")
    parser.add_argument('--save-pred', type=str, default=PREDICTIONS_SAVE_PATH,
                        help=f"Save predictions to CSV file. Default: {PREDICTIONS_SAVE_PATH}")
    args = parser.parse_args()
    
    # Ensure weather forecast data is available
    ensure_weather_forecast()
    
    # Load model
    model = load_model(MODEL_LOAD_PATH)
    
    # Load historical data
    historical_df = run_data_pipeline(
        CONSUMPTION_DATA_PATH,
        HEAT_PUMP_DATA_PATH,
        WEATHER_DATA_PATH
    )
    
    # Get latest timestamp from historical data
    latest_timestamp = historical_df.index.max()
    logging.info(f"Latest timestamp in historical data: {latest_timestamp}")
    
    # Set prediction start time
    prediction_start = latest_timestamp + pd.Timedelta(hours=1)
    prediction_end = prediction_start + pd.Timedelta(hours=args.horizon-1)
    logging.info(f"Making predictions from {prediction_start} to {prediction_end} ({args.horizon} hours)")
    
    # Load weather forecast
    weather_forecast = load_weather_forecast(
        WEATHER_FORECAST_PATH, 
        start_time=prediction_start, 
        horizon_hours=args.horizon
    )
    
    # Prepare data for prediction
    combined_df = prepare_data_for_prediction(
        historical_df[[TARGET_COL]], 
        historical_df[['power_input_kwh']], 
        weather_forecast, 
        horizon_hours=args.horizon
    )
    
    # Generate features for prediction
    df_features = generate_future_features(combined_df)
    
    # Make predictions
    df_pred = make_predictions(model, df_features, forecast_horizon=args.horizon)
    
    # Save predictions if requested
    if args.save_pred:
        save_predictions(df_pred, args.save_pred)
    
    # Plot predictions if requested
    if args.plot or args.save_plot:
        # If save_plot is None, use default path with timestamp
        save_path = args.save_plot
        if args.plot and not save_path:
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(PLOTS_SAVE_DIR, f'demand_forecast_{current_time}.png')
            
        fig, ax = plot_predictions(df_pred, 
                                   historical_window=48,
                                   save_path=save_path)
        if args.plot:
            plt.show()
    
    logging.info("Prediction complete")
    
    return df_pred

if __name__ == "__main__":
    main()