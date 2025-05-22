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
AGENT_MODEL_LOAD_PATH = 'src/predictions/demand/models/baseload/villamichelin_baseload_model.pkl'
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
AGENT_PREDICTIONS_SAVE_PATH = f'{PREDICTIONS_DIR}/baseload/baseload_predictions.csv'
PLOTS_SAVE_DIR = 'src/predictions/demand/plots/predictions/'

# Default target column name
TARGET_COL = 'consumption'
BASELOAD_TARGET_COL = 'baseload'

# Ensure directories exist
os.makedirs(WEATHER_FORECAST_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(os.path.join(PREDICTIONS_DIR, "baseload"), exist_ok=True)
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
    load_heat_pump_data,
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

def load_consumption_data(file_path: str, for_agent: bool = False) -> pd.DataFrame:
    """Load consumption data from CSV file."""
    logging.info(f"Loading consumption data from {file_path}")
    try:
        df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
        df.index = df.index.tz_localize('Europe/Stockholm', ambiguous=True, nonexistent='shift_forward')
        
        if for_agent:
            # For agent predictions, we need both consumption and baseload if available
            # If baseload column doesn't exist yet, we'll calculate it
            if 'production' in df.columns:
                if 'baseload' not in df.columns:
                    df['baseload'] = df['consumption'] + df['production']
                df = df[['baseload', 'consumption', 'production']]
            else:
                logging.warning("Production data not found. Using consumption as baseload.")
                df['baseload'] = df['consumption']
                df = df[['baseload', 'consumption']]
        else:
            # For regular predictions, just use consumption
            df = df[['consumption']]
            
        logging.info(f"Consumption data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading consumption data: {e}")
        raise

def generate_future_features(df, hmm_model=None, historical_only=False, country_code='SE', target_column='consumption'):
    """Generate features for future prediction periods with enhanced handling for forecasting."""
    logging.info("Generating features for prediction")
    
    # Apply HMM to the historical data
    historical_df = df[df[target_column].notna()].copy()
    
    # If hmm_model is not provided, try to fit a new one
    if hmm_model is None:
        logging.info("No pre-trained HMM model provided, fitting a new one on historical data")
        from src.predictions.demand.train import fit_hmm, N_HMM_STATES
        try:
            hmm_model = fit_hmm(historical_df[target_column], N_HMM_STATES)
        except Exception as e:
            logging.warning(f"Error fitting HMM model: {e}. Will use dummy HMM features.")
            # Create dummy HMM features
            historical_df['hmm_state'] = 0
            for i in range(N_HMM_STATES):
                historical_df[f'hmm_state_posterior_{i}'] = 1.0 / N_HMM_STATES
            
            # Create dummy HMM features for future prediction
            future_df = df[df[target_column].isna()].copy()
            future_df['hmm_state'] = 0
            for i in range(N_HMM_STATES):
                future_df[f'hmm_state_posterior_{i}'] = 1.0 / N_HMM_STATES
                
            combined_df = pd.concat([historical_df, future_df])
            
            # Add remaining features
            featured_df = add_calendar_features(combined_df)
            featured_df = add_lagged_features(featured_df, target_column)
            featured_df = add_weather_transforms(featured_df)
            featured_df = add_interaction_terms(featured_df)
            
            return featured_df
    
    # Get states and posteriors using the HMM model
    from src.predictions.demand.train import decode_states, get_state_posteriors, N_HMM_STATES
    
    states = decode_states(hmm_model, historical_df[target_column])
    posteriors = get_state_posteriors(hmm_model, historical_df[target_column])
    
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

    # Create future dataframe with the same HMM state as the last known state
    future_df = df[df[target_column].isna()].copy()
    future_df['hmm_state'] = last_state
    for i in range(N_HMM_STATES):
        future_df[f'hmm_state_posterior_{i}'] = last_posteriors[f'hmm_state_posterior_{i}']
    
    # Combine historical and future data
    combined_df = pd.concat([historical_df, future_df])
    
    # Add all other required features
    # Import necessary functions from train.py
    from src.predictions.demand.train import (
        add_calendar_features, 
        add_lagged_features, 
        add_weather_transforms, 
        add_interaction_terms
    )
    
    # Generate all features needed for prediction
    featured_df = add_calendar_features(combined_df, country_code)
    featured_df = add_lagged_features(featured_df, target_column)
    featured_df = add_weather_transforms(featured_df)
    featured_df = add_interaction_terms(featured_df)
    
    return featured_df

def make_predictions(model, df_features, forecast_horizon=24, for_agent=False):
    """
    Make predictions for the future using the trained model.
    
    Args:
        model: Trained model (XGBoost)
        df_features: DataFrame with features prepared for prediction
        forecast_horizon: Number of hours to predict
        for_agent: Whether this is for agent mode (baseload vs consumption)
        
    Returns:
        DataFrame with original data and predictions column added
    """
    logging.info(f"Making predictions for {forecast_horizon} hours")
    
    # Find the latest non-NaN consumption time (boundary between historical and future)
    target_col = 'baseload' if for_agent else 'consumption'
    latest_data_time = df_features[df_features[target_col].notna()].index.max()
    
    # Create a copy to avoid modifying the original
    pred_df = df_features.copy()
    
    # Initialize a predictions column
    pred_df['predictions'] = np.nan
    
    # Loop through each future timestamp and make a prediction
    future_indices = pred_df[pred_df.index > latest_data_time].index
    future_indices = future_indices[:forecast_horizon]  # Limit to requested horizon
    
    # Store the features that our model expects
    model_features = model.feature_names_in_
    n_predictions_made = 0
    
    for i, timestamp in enumerate(future_indices):
        # Get feature values for this timestamp
        row = pred_df.loc[timestamp]
        
        # Check if we have all required features
        missing_features = [f for f in model_features if f not in row.index or pd.isna(row[f])]
        if missing_features:
            logging.warning(f"Missing features for prediction at {timestamp}: {len(missing_features)} features")
            
        # Prepare feature vector for prediction (handling missing features)
        X = np.zeros((1, len(model_features)))
        for j, feat in enumerate(model_features):
            if feat in row.index and not pd.isna(row[feat]):
                X[0, j] = row[feat]
        
        # Make prediction
        try:
            pred_value = model.predict(X)[0]
            pred_df.loc[timestamp, 'predictions'] = pred_value
            n_predictions_made += 1
            
            # Add confidence bounds if the model supports it
            if hasattr(model, 'predict_quantile'):
                lower_bound = model.predict_quantile(X, quantile=0.025)[0]
                upper_bound = model.predict_quantile(X, quantile=0.975)[0]
                pred_df.loc[timestamp, 'lower_bound'] = lower_bound
                pred_df.loc[timestamp, 'upper_bound'] = upper_bound
                
        except Exception as e:
            logging.error(f"Error making prediction for {timestamp}: {e}")
    
    logging.info(f"Successfully made {n_predictions_made} predictions out of {len(future_indices)} requested")
    
    return pred_df

def plot_predictions(df_pred, historical_window=48, save_path=None, title=None):
    """
    Plot the predictions along with historical data.
    
    Args:
        df_pred: DataFrame with historical data and predictions
        historical_window: Number of historical hours to display
        save_path: Path to save the plot, if provided
        title: Custom title for the plot
    """
    logging.info("Plotting predictions")
    
    # Debug the dataframe
    logging.info(f"DataFrame shape: {df_pred.shape}")
    logging.info(f"DataFrame columns: {df_pred.columns.tolist()}")
    logging.info(f"First few rows of predictions:\n{df_pred.head().to_string()}")
    
    # Create a copy of the dataframe to avoid modifying the original
    df = df_pred.copy()
    
    # Identify which rows have predictions (not NaN)
    has_predictions = ~df['predictions'].isna()
    
    if not has_predictions.any():
        logging.error("No valid predictions found in the data")
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.text(0.5, 0.5, "No valid predictions to display", 
                ha='center', va='center', fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi=100)
            logging.info(f"Empty plot saved to {save_path}")
        return fig, ax
    
    # Get the first timestamp that has a prediction
    forecast_start = df.loc[has_predictions].index[0]
    logging.info(f"Forecast start time: {forecast_start}")
    
    # Filter to show only the last [historical_window] hours + forecast
    historical_start = forecast_start - pd.Timedelta(hours=historical_window)
    df_plot = df[df.index >= historical_start].copy()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot historical data (if we have both consumption and baseload columns, use the right one based on title)
    historical_mask = df_plot.index < forecast_start
    
    # Determine which column to use for historical data
    target_col = 'baseload' if 'baseload' in df_plot.columns and 'Baseload' in (title or '') else 'consumption'
    if target_col not in df_plot.columns:
        target_col = df_plot.columns[0]  # Fallback to first column
        
    if historical_mask.any() and not df_plot.loc[historical_mask, target_col].empty:
        ax.plot(df_plot.index[historical_mask], 
                df_plot.loc[historical_mask, target_col], 
                label=f'Historical {target_col.capitalize()}', 
                color='#1f77b4', 
                linewidth=2)
    
    # Plot predictions
    forecast_mask = df_plot.index >= forecast_start
    if forecast_mask.any() and not df_plot.loc[forecast_mask, 'predictions'].empty:
        ax.plot(df_plot.index[forecast_mask], 
                df_plot.loc[forecast_mask, 'predictions'], 
                label=f'Predicted {target_col.capitalize()}', 
                color='#ff7f0e', 
                linewidth=2, 
                linestyle='-', 
                marker='o', 
                markersize=5)
    
    # Add vertical line at forecast start
    ax.axvline(x=forecast_start, color='red', linestyle='--', alpha=0.7, 
              label='Forecast Start')
    
    # Add confidence intervals if available
    if 'lower_bound' in df_plot.columns and 'upper_bound' in df_plot.columns:
        if forecast_mask.any():
            ax.fill_between(df_plot.index[forecast_mask], 
                           df_plot.loc[forecast_mask, 'lower_bound'], 
                           df_plot.loc[forecast_mask, 'upper_bound'], 
                           color='#ff7f0e', 
                           alpha=0.2, 
                           label='95% Confidence Interval')
    
    # Formatting
    ax.set_title(title or f'Energy {target_col.capitalize()} Forecast', fontsize=16)
    ax.set_ylabel(f'Energy {target_col.capitalize()} (kWh)', fontsize=12)
    ax.set_xlabel('Time', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format dates on x-axis
    fig.autofmt_xdate()
    
    # Add extra information
    forecast_horizon = sum(forecast_mask)
    ax.text(0.02, 0.95, f'Forecast Horizon: {forecast_horizon} hours', 
           transform=ax.transAxes, 
           bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logging.info(f"Plot saved to {save_path}")
    
    return fig, ax

def save_predictions(df_pred, save_path):
    """Save the predictions to a CSV file."""
    logging.info(f"Saving predictions to {save_path}")
    
    # Extract only relevant columns
    save_df = df_pred[['predictions']].copy()
    save_df.index.name = 'timestamp'
    save_df.loc[~save_df['predictions'].isna()].to_csv(save_path)
    
    logging.info(f"Predictions saved to {save_path}")
    return save_df

def main():
    """Main function to generate and save demand predictions."""
    parser = argparse.ArgumentParser(description='Generate demand predictions')
    parser.add_argument('--horizon', type=int, default=24, 
                        help='Number of hours to predict into the future')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to a specific model to use for prediction (optional)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate and display prediction plots')
    parser.add_argument('--save', action='store_true',
                        help='Save predictions to CSV file')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Custom path to save predictions CSV')
    parser.add_argument('--for-agent', action='store_true',
                        help='Use agent-specific baseload model instead of standard demand model')
    args = parser.parse_args()
    
    # Set paths based on agent mode
    model_path = args.model_path
    if model_path is None:
        model_path = AGENT_MODEL_LOAD_PATH if args.for_agent else MODEL_LOAD_PATH
    
    save_path = args.save_path
    if save_path is None:
        save_path = AGENT_PREDICTIONS_SAVE_PATH if args.for_agent else PREDICTIONS_SAVE_PATH
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Set target column based on agent mode
    target_column = BASELOAD_TARGET_COL if args.for_agent else TARGET_COL
    logging.info(f"Using {'agent-specific baseload' if args.for_agent else 'standard demand'} model")
    logging.info(f"Target column: {target_column}")
    
    try:
        # Ensure we have a weather forecast file
        ensure_weather_forecast()
        
        # Load the model
        model = load_model(model_path)
        
        # Try to load the HMM model, will be None if not found
        hmm_model = load_hmm_model(HMM_MODEL_LOAD_PATH)
        
        # Get current time for prediction start
        prediction_start_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        prediction_start_time = pd.Timestamp(prediction_start_time).tz_localize('Europe/Stockholm')
        
        # Load historical data for context
        consumption_df = load_consumption_data(CONSUMPTION_DATA_PATH, for_agent=args.for_agent)
        heat_pump_df = load_heat_pump_data(HEAT_PUMP_DATA_PATH)
        heat_pump_df['power_input_kwh'] = heat_pump_df['power_input_kw'] * 0.25  # Convert to kWh
        heat_pump_df = heat_pump_df.drop(columns=['power_input_kw'])
        heat_pump_df = heat_pump_df.resample('h').sum()
        
        # Load weather forecast for prediction period
        weather_forecast_df = load_weather_forecast(WEATHER_FORECAST_PATH, 
                                                   start_time=prediction_start_time,
                                                   horizon_hours=args.horizon)
        
        # Prepare data with historical + forecast
        combined_df = prepare_data_for_prediction(consumption_df, heat_pump_df, 
                                                 weather_forecast_df, args.horizon)
        
        # Generate features for the entire dataset
        featured_df = generate_future_features(combined_df, hmm_model, 
                                              target_column=target_column)
        
        # Make predictions
        predictions_df = make_predictions(model, featured_df, args.horizon, args.for_agent)
        
        # Save predictions if requested
        if args.save:
            save_predictions(predictions_df, save_path)
            logging.info(f"Predictions saved to {save_path}")
        
        # Plot predictions if requested
        if args.plot:
            title = f"{'Baseload' if args.for_agent else 'Energy Demand'} Forecast"
            plot_path = None
            if args.save:
                plot_file = f"{'baseload' if args.for_agent else 'demand'}_forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
                plot_path = os.path.join(PLOTS_SAVE_DIR, plot_file)
            
            plot_predictions(predictions_df, historical_window=48, save_path=plot_path, title=title)
            
            if plot_path is None:  # Only show if not saving
                plt.show()
        
        logging.info("Prediction process completed successfully.")
        return predictions_df
        
    except Exception as e:
        logging.error(f"Error during prediction process: {e}")
        raise

if __name__ == "__main__":
    main()