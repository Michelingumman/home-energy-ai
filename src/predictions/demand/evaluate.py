# evaluate file for demand predictions

import logging
# silence TensorFlow-emitted WARNING/INFO logs
logging.getLogger('pandas').setLevel(logging.ERROR)

import warnings
# suppress all Python warnings (FutureWarning, DeprecationWarning, etc.)
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
import logging
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from holidays import country_holidays # For feature engineering consistency
from hmmlearn import hmm # For feature engineering consistency
import sys
import argparse
from pathlib import Path
from datetime import datetime
from matplotlib.dates import DayLocator, DateFormatter
import matplotlib.dates as mdates

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set nicer plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.figsize'] = (14, 7)

# Constants - Mirroring train.py for consistency
MODEL_LOAD_PATH = 'src/predictions/demand/models/villamichelin_demand_model.pkl'
MODEL_SPLITS_INFO_PATH = 'src/predictions/demand/models/villamichelin_demand_model_splits_info.pkl'
PREDICTED_TARGET_COL = 'y' # Name of the target column model was trained to predict (e.g., consumption.shift(-1))
PLOTS_SAVE_DIR = 'src/predictions/demand/plots/demand_model_evaluation/'

# Paths to data for evaluation (typically a hold-out set)
# For this example, we'll re-use the main data paths and simulate a hold-out split.
CONSUMPTION_DATA_PATH = 'data/processed/villamichelin/VillamichelinEnergyData.csv'
HAT_PUMP_DATA_PATH = 'data/processed/villamichelin/Thermia/HeatPumpPower.csv'
WEATHER_DATA_PATH = 'data/processed/weather_data.csv'

# Ensure plots directory exists
os.makedirs(PLOTS_SAVE_DIR, exist_ok=True)

ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Import all necessary functions from train.py
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

# --- Evaluation Functions ---
def load_model(model_path: str):
    logging.info(f"Loading model from {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def load_splits_info(splits_info_path: str):
    logging.info(f"Loading model splits information from {splits_info_path}")
    try:
        with open(splits_info_path, 'rb') as f:
            splits_info = pickle.load(f)
        logging.info("Model splits information loaded successfully.")
        return splits_info
    except Exception as e:
        logging.warning(f"Could not load model splits information: {e}")
        return None

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero for rows where y_true is 0
    mask = y_true != 0
    if not np.any(mask):
        return np.nan # Or some other indicator like np.inf if all true values are zero
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_metrics(y_true, y_pred, model_name: str):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    logging.info(f"Metrics for {model_name}:")
    logging.info(f"  MAE:  {mae:.4f}")
    logging.info(f"  RMSE: {rmse:.4f}")
    logging.info(f"  MAPE: {mape:.2f}%")
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def persistence_model_forecast(series: pd.Series, lag: int = 24) -> pd.Series:
    """Generates persistence forecast (value from `lag` hours ago)."""
    return series.shift(lag)

def _apply_hmm_outputs_to_df(df, hmm_model, series, n_states):
    """Helper function to apply a pre-trained HMM model to data and extract states and posteriors"""
    mean_consumptions = hmm_model.means_.flatten()
    state_order = np.argsort(mean_consumptions)
    state_map = {old_label: new_label for new_label, old_label in enumerate(state_order)}
    
    # Apply the HMM model to get states and posteriors
    decoded_states_raw = hmm_model.predict(series.values.reshape(-1, 1))
    posterior_probs_raw = hmm_model.predict_proba(series.values.reshape(-1, 1))
    
    # Map states and add to dataframe
    df_result = df.copy()
    df_result['hmm_state'] = np.array([state_map[s] for s in decoded_states_raw])
    
    posterior_probs_sorted = posterior_probs_raw[:, state_order]
    for i in range(n_states):
        df_result[f'hmm_state_posterior_{i}'] = posterior_probs_sorted[:, i]
    
    return df_result

# --- Enhanced Visualization Functions ---

def plot_demand_with_hmm(eval_df, title="Energy Demand with HMM States", save_path=None):
    """
    Plot demand predictions along with HMM states in a two-panel figure.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Upper plot: Demand
    ax1.plot(eval_df.index, eval_df[PREDICTED_TARGET_COL], label='Actual', color='#2C3E50', linewidth=1.5)
    ax1.plot(eval_df.index, eval_df['predictions'], label='Predicted', color='#E74C3C', linewidth=1.5)
    
    if 'persistence_pred' in eval_df.columns:
        ax1.plot(eval_df.index, eval_df['persistence_pred'], label='Persistence (t-24h)', 
                 linestyle='--', color='#27AE60', alpha=0.6, linewidth=1)
    
    # Format top plot
    ax1.set_title(title, fontsize=16)
    ax1.set_ylabel('Energy Consumption', fontsize=12)
    ax1.legend(frameon=True, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Set daily x-axis ticks with angled date labels for the top plot
    ax1.xaxis.set_major_locator(DayLocator())
    ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    
    # Rotate tick labels for better readability
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Bottom plot: HMM states
    if 'hmm_state' in eval_df.columns:
        # Use the newer recommended approach for getting colormaps
        cmap = plt.colormaps['viridis'].resampled(N_HMM_STATES)
        scatter = ax2.scatter(eval_df.index, [0.5] * len(eval_df), 
                             c=eval_df['hmm_state'], cmap=cmap, 
                             s=50, alpha=0.8, marker='s')
        
        # Format bottom plot
        ax2.set_ylim(0, 1)
        ax2.set_yticks([])
        ax2.set_ylabel('HMM State', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        cbar = plt.colorbar(scatter, ax=ax2, ticks=range(N_HMM_STATES))
        cbar.set_label('Occupancy State')
        
        # Set the same x-axis ticks with angled date labels for the bottom plot
        ax2.xaxis.set_major_locator(DayLocator())
        ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        # Add labels to identify peak demand periods
        peak_threshold = eval_df[PREDICTED_TARGET_COL].quantile(0.9)
        peak_periods = eval_df[eval_df[PREDICTED_TARGET_COL] > peak_threshold]
        for idx, row in peak_periods.iterrows():
            ax1.axvspan(idx-pd.Timedelta(hours=0.5), idx+pd.Timedelta(hours=0.5), 
                      color='grey', alpha=0.15)
    
    # Add metrics annotation
    if len(eval_df) > 0:
        actual = eval_df[PREDICTED_TARGET_COL].dropna()
        predicted = eval_df['predictions'].loc[actual.index]
        if len(actual) > 0 and len(predicted) > 0:
            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            
            ax1.annotate(f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}',
                        xy=(0.02, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", 
                                ec="gray", alpha=0.8))
    
    # Adjust layout to accommodate the rotated tick labels
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, (ax1, ax2)

def create_analysis_dashboard(eval_df, title="Demand Prediction Analysis", save_path=None):
    """
    Create a comprehensive dashboard with multiple panels analyzing the predictions.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2)
    
    # Panel 1: Time series comparison
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(eval_df.index, eval_df[PREDICTED_TARGET_COL], label='Actual', color='#2980B9')
    ax1.plot(eval_df.index, eval_df['predictions'], label='Predicted', color='#E74C3C')
    if 'persistence_pred' in eval_df.columns:
        ax1.plot(eval_df.index, eval_df['persistence_pred'], label='Persistence (t-24h)', 
                 linestyle='--', color='#27AE60', alpha=0.6)
    ax1.set_title(title, fontsize=16)
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Energy Consumption')
    
    # Panel 2: Residuals by hour
    ax2 = fig.add_subplot(gs[1, 0])
    if 'residuals' in eval_df.columns:
        eval_df['hour'] = eval_df.index.hour
        sns.boxplot(x='hour', y='residuals', data=eval_df, ax=ax2, palette='Blues')
        ax2.set_title('Prediction Error by Hour')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Error (Actual - Predicted)')
        ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Panel 3: HMM state distribution over time
    ax3 = fig.add_subplot(gs[1, 1])
    if 'hmm_state' in eval_df.columns:
        eval_df['day_of_month'] = eval_df.index.day
        pivot_df = eval_df.pivot_table(
            index='day_of_month', 
            columns='hour', 
            values='hmm_state',
            aggfunc='mean'
        )
        # Use matplotlib colormaps directly instead of string name
        sns.heatmap(pivot_df, cmap=plt.colormaps['viridis'], ax=ax3, cbar_kws={'label': 'HMM State'})
        ax3.set_title('HMM State Distribution (Day × Hour)')
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Day of Month')
    
    # Panel 4: Error vs. Temperature
    ax4 = fig.add_subplot(gs[2, 0])
    if all(col in eval_df.columns for col in ['temperature_2m', 'residuals', 'hmm_state']):
        sns.scatterplot(x='temperature_2m', y='residuals', 
                       hue='hmm_state', data=eval_df, ax=ax4, palette='viridis')
        ax4.set_title('Error vs. Temperature by HMM State')
        ax4.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Panel 5: Error distribution by HMM state
    ax5 = fig.add_subplot(gs[2, 1])
    if all(col in eval_df.columns for col in ['residuals', 'hmm_state']):
        order = sorted(eval_df['hmm_state'].unique())
        sns.violinplot(x='hmm_state', y='residuals', data=eval_df, ax=ax5, 
                      palette='viridis', order=order)
        ax5.set_title('Error Distribution by HMM State')
        ax5.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, (ax1, ax2, ax3, ax4, ax5)

def plot_month_data(period_arg, show_hmm=True, show_dashboard=False, save_dir=None):
    """
    Plot predictions for a specific month with enhanced visualizations.
    
    Parameters:
    -----------
    period_arg : str
        Month to plot in YYYY-MM format
    show_hmm : bool
        Whether to include HMM state visualization
    show_dashboard : bool
        Whether to create the comprehensive dashboard
    save_dir : str, optional
        If provided, save plots to this directory
    """
    # Parse the month argument
    try:
        parts = period_arg.split('-')
        if len(parts) == 2 and len(parts[0]) == 4:  # Format: YYYY-MM
            year = int(parts[0])
            month = int(parts[1])
            
            # Create start and end dates for the month
            start_date = pd.Timestamp(year=year, month=month, day=1, tz='Europe/Stockholm')
            if month == 12:
                end_date = pd.Timestamp(year=year+1, month=1, day=1, tz='Europe/Stockholm') - pd.Timedelta(days=1)
            else:
                end_date = pd.Timestamp(year=year, month=month+1, day=1, tz='Europe/Stockholm') - pd.Timedelta(days=1)
            
            title = f'Demand Prediction for {start_date.strftime("%B %Y")}'
        else:
            logging.error("Invalid month format. Use YYYY-MM (e.g., 2025-05)")
            return
    except (ValueError, IndexError) as e:
        logging.error(f"Error parsing month format: {e}")
        return
        
    logging.info(f"Plotting data from {start_date.date()} to {end_date.date()}")
    
    # Load model
    try:
        model = load_model(MODEL_LOAD_PATH)
        model_feature_names = model.get_booster().feature_names
        if not model_feature_names:
            logging.error("Could not retrieve feature names from the model.")
            return
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return
    
    # Load data for the period
    full_data = run_data_pipeline(CONSUMPTION_DATA_PATH, HAT_PUMP_DATA_PATH, WEATHER_DATA_PATH)
    
    # Filter data for the specified month with some margin for feature engineering
    # Add 14 days before to have enough data for lagged features
    data_start = start_date - pd.Timedelta(days=14)
    period_data = full_data[full_data.index >= data_start]
    
    if period_data.empty:
        logging.error(f"No data found for period {period_arg}")
        return
    
    # Add HMM features
    hmm_model_for_period = fit_hmm(period_data[TARGET_COL].dropna(), n_states=N_HMM_STATES)
    data_with_hmm = add_hmm_features(period_data, consumption_col=TARGET_COL, n_states=N_HMM_STATES)
    
    # Generate features for the period data
    featured_data = engineer_features(data_with_hmm, target_col=TARGET_COL)
    
    # Filter to the exact period after feature engineering
    period_featured_data = featured_data[(featured_data.index >= start_date) & 
                                       (featured_data.index <= end_date)].copy()
    
    if period_featured_data.empty or PREDICTED_TARGET_COL not in period_featured_data.columns:
        logging.error(f"No featured data available for period {period_arg} after processing")
        return
    
    # Ensure all necessary features are present
    if set(model_feature_names).issubset(period_featured_data.columns):
        # More efficient approach - use only the columns we need in one operation
        X = period_featured_data[model_feature_names].copy()
    else:
        # Handle missing features efficiently by building a dictionary first
        feature_dict = {}
        for col in model_feature_names:
            if col in period_featured_data.columns:
                feature_dict[col] = period_featured_data[col]
            else:
                feature_dict[col] = pd.Series(0, index=period_featured_data.index)
                logging.warning(f"Feature '{col}' not found in data. Using 0.")
        
        # Create DataFrame from dictionary all at once to avoid fragmentation
        X = pd.DataFrame(feature_dict, index=period_featured_data.index)
    
    # Handle NaNs
    if X.isnull().any().any():
        logging.warning("NaNs found in features. Using 0 for predictions.")
        X = X.fillna(0)
    
    # Make predictions
    predictions = model.predict(X)
    period_featured_data['predictions'] = predictions
    
    # Add persistence prediction (t-24h) and residuals
    period_featured_data['persistence_pred'] = period_featured_data[TARGET_COL].shift(24 - FORECAST_HORIZON)
    period_featured_data['residuals'] = period_featured_data[PREDICTED_TARGET_COL] - period_featured_data['predictions']
    
    # Calculate metrics
    actuals = period_featured_data[PREDICTED_TARGET_COL].dropna()
    preds = period_featured_data['predictions'].loc[actuals.index]
    metrics = calculate_metrics(actuals, preds, "XGBoost")
    
    if 'persistence_pred' in period_featured_data.columns:
        persistence_preds = period_featured_data['persistence_pred'].loc[actuals.index].dropna()
        shared_idx = actuals.index.intersection(persistence_preds.index)
        if len(shared_idx) > 0:
            persistence_metrics = calculate_metrics(
                actuals.loc[shared_idx], 
                persistence_preds.loc[shared_idx], 
                "Persistence"
            )
    
    # Create plots
    if show_hmm:
        if save_dir:
            hmm_save_path = os.path.join(save_dir, f'demand_with_hmm_{period_arg}.png')
        else:
            hmm_save_path = None
        plot_demand_with_hmm(period_featured_data, title=title, save_path=hmm_save_path)
        plt.show()
    
    if show_dashboard:
        if save_dir:
            dashboard_save_path = os.path.join(save_dir, f'dashboard_{period_arg}.png')
        else:
            dashboard_save_path = None
        create_analysis_dashboard(period_featured_data, title=title, save_path=dashboard_save_path)
        plt.show()
        
    return period_featured_data, metrics

def evaluate_model_on_holdout(hold_out_df_featured: pd.DataFrame, model, model_feature_names: list):
    # Renamed from evaluate_model to avoid confusion if evaluate_model is a general concept
    logging.info(f"Evaluating model on hold-out set of shape {hold_out_df_featured.shape}")
    
    # Ensure PREDICTED_TARGET_COL ('y') is present
    if PREDICTED_TARGET_COL not in hold_out_df_featured.columns:
        logging.error(f"Target column '{PREDICTED_TARGET_COL}' not found in featured hold-out data. Cannot evaluate.")
        return None, None, pd.DataFrame() # Return empty/None for metrics and df
        
    eval_X = hold_out_df_featured[model_feature_names]
    eval_y_true = hold_out_df_featured[PREDICTED_TARGET_COL]

    if eval_X.isnull().any().any():
        logging.warning("NaNs found in evaluation features. Imputing with 0. Evaluation might be affected.")
        eval_X = eval_X.fillna(0)

    predictions = model.predict(eval_X)
    
    eval_results_df = hold_out_df_featured.copy()
    eval_results_df['predictions'] = predictions
    eval_results_df['residuals'] = eval_results_df[PREDICTED_TARGET_COL] - eval_results_df['predictions']

    model_metrics = calculate_metrics(eval_y_true, predictions, "XGBoost")

    # Persistence model (t-24h relative to target time)
    # TARGET_COL here refers to the original consumption column (imported from train)
    # PREDICTED_TARGET_COL is 'y', which is TARGET_COL.shift(-FORECAST_HORIZON)
    # So, persistence prediction for PREDICTED_TARGET_COL at time t (which is actual consumption at t+FH)
    # should be TARGET_COL at time (t+FH) - 24h.
    # If FH=1, this is TARGET_COL at t-23h.
    # eval_results_df[TARGET_COL] is consumption at time t.
    # So, .shift(24 - FORECAST_HORIZON) on TARGET_COL is correct.
    eval_results_df['persistence_pred'] = eval_results_df[TARGET_COL].shift(24 - FORECAST_HORIZON)
    eval_results_df.dropna(subset=['persistence_pred', PREDICTED_TARGET_COL], inplace=True) # Ensure target and pred are non-NaN
    
    if not eval_results_df.empty:
        y_true_for_persistence = eval_results_df[PREDICTED_TARGET_COL]
        persistence_predictions_series = eval_results_df['persistence_pred']
        logging.info(f"Persistence benchmark using lag of 24 hours relative to target time.")
        persistence_metrics = calculate_metrics(y_true_for_persistence, persistence_predictions_series, "Persistence (t-24h)")
    else:
        logging.warning("DataFrame became empty after dropping NaNs for persistence benchmark. Skipping persistence metrics.")
        persistence_metrics = {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}
    
    return model_metrics, persistence_metrics, eval_results_df

def main():
    # Simplified command line arguments
    parser = argparse.ArgumentParser(description='Evaluate demand prediction model and visualize results')
    parser.add_argument('--plot', type=str, help='Plot a specific month (YYYY-MM format)')
    parser.add_argument('--hmm', action='store_true', help='Show HMM states with the plot')
    parser.add_argument('--dashboard', action='store_true', help='Show comprehensive analysis dashboard')
    parser.add_argument('--eval-ratio', type=float, default=0.2, help='Proportion of data to use for evaluation')
    parser.add_argument('--save', action='store_true', help='Save plots instead of displaying them')
    
    args = parser.parse_args()
    
    # Create save directory if needed
    save_dir = PLOTS_SAVE_DIR if args.save else None
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # If plot argument is specified, plot that month and exit
    if args.plot:
        plot_month_data(
            args.plot, 
            show_hmm=True,  # Always show HMM by default 
            show_dashboard=args.dashboard,
            save_dir=save_dir
        )
        return
    
    # Original evaluation functionality
    logging.info("Starting Villamichelin demand model evaluation script")

    # Try to load splits information if available
    model_splits_info = load_splits_info(MODEL_SPLITS_INFO_PATH)
    if model_splits_info:
        logging.info("\n=== MODEL TRAINING DATA SPLITS ===")
        logging.info(f"Full dataset: {model_splits_info['full_dataset']['start_date']} to {model_splits_info['full_dataset']['end_date']} ({model_splits_info['full_dataset']['n_samples']} samples)")
        logging.info(f"Training set: {model_splits_info['final_train']['start_date']} to {model_splits_info['final_train']['end_date']} ({model_splits_info['final_train']['n_samples']} samples)")
        logging.info(f"Validation set: {model_splits_info['final_val']['start_date']} to {model_splits_info['final_val']['end_date']} ({model_splits_info['final_val']['n_samples']} samples)")
        logging.info("===================================\n")
    
    # 1. Load and prepare data using run_data_pipeline from train.py
    full_data = run_data_pipeline(CONSUMPTION_DATA_PATH, HAT_PUMP_DATA_PATH, WEATHER_DATA_PATH)
    if full_data.empty:
        logging.error("Data pipeline for evaluation resulted in an empty dataframe. Exiting.")
        return

    # Log information about the full data
    if isinstance(full_data.index, pd.DatetimeIndex):
        data_start_date = full_data.index.min().strftime('%Y-%m-%d %H:%M:%S')
        data_end_date = full_data.index.max().strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"Full data available for evaluation: {data_start_date} to {data_end_date} ({len(full_data)} samples)")

    # Split data into train and test sets
    evaluation_period_ratio = args.eval_ratio
    split_point = int(len(full_data) * (1 - evaluation_period_ratio))
    hmm_fit_data_portion = full_data.iloc[:split_point]
    eval_data_raw_portion = full_data.iloc[split_point:]

    # Log information about the evaluation split
    if isinstance(full_data.index, pd.DatetimeIndex):
        hmm_start_date = hmm_fit_data_portion.index.min().strftime('%Y-%m-%d %H:%M:%S')
        hmm_end_date = hmm_fit_data_portion.index.max().strftime('%Y-%m-%d %H:%M:%S')
        eval_start_date = eval_data_raw_portion.index.min().strftime('%Y-%m-%d %H:%M:%S')
        eval_end_date = eval_data_raw_portion.index.max().strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"Evaluation split ({evaluation_period_ratio*100:.1f}% of data):")
        logging.info(f"  HMM fit data: {hmm_start_date} to {hmm_end_date} ({len(hmm_fit_data_portion)} samples)")
        logging.info(f"  Evaluation data: {eval_start_date} to {eval_end_date} ({len(eval_data_raw_portion)} samples)")
        
        # Print a warning if the evaluation data is already included in the training data
        if model_splits_info:
            try:
                # Parse model end date string to datetime with timezone info
                model_end_date_str = model_splits_info['final_val']['end_date']
                model_end_date = datetime.strptime(model_end_date_str, '%Y-%m-%d %H:%M:%S')
                # Make eval_start_datetime timezone-naive for comparison
                eval_start_datetime = eval_data_raw_portion.index.min()
                eval_start_naive = eval_start_datetime.tz_localize(None)
                
                if eval_start_naive <= model_end_date:
                    logging.warning(f"WARNING: Evaluation data overlaps with model training data!")
                    logging.warning(f"Model was trained on data up to: {model_splits_info['final_val']['end_date']}")
                    logging.warning(f"Evaluation starts from: {eval_start_date}")
                    logging.warning("This may lead to overly optimistic evaluation metrics.")
                else:
                    logging.info(f"Evaluation data begins after model training data ended - testing on unseen data.")
            except Exception as e:
                logging.warning(f"Could not compare evaluation dates due to: {e}")
                logging.warning(f"Model training end date: {model_splits_info['final_val']['end_date']}")
                logging.warning(f"Evaluation start date: {eval_start_date}")

    if eval_data_raw_portion.empty:
        logging.error("Hold-out evaluation set is empty after split. Exiting.")
        return
    
    # Prepare HMM features for the evaluation set
    hmm_input_series_for_fit = hmm_fit_data_portion[TARGET_COL].dropna()
    eval_target_series_for_decode = eval_data_raw_portion[TARGET_COL] 

    eval_data_with_hmm = eval_data_raw_portion.copy()
    if not hmm_input_series_for_fit.empty and len(hmm_input_series_for_fit) >= N_HMM_STATES:
        logging.info(f"Fitting HMM on historical data (length {len(hmm_input_series_for_fit)}) for evaluation set.")
        hmm_model_for_eval = fit_hmm(hmm_input_series_for_fit, n_states=N_HMM_STATES)
        eval_data_with_hmm = _apply_hmm_outputs_to_df(eval_data_raw_portion, 
                                                      hmm_model_for_eval, 
                                                      eval_target_series_for_decode, 
                                                      N_HMM_STATES)
    else:
        logging.warning("Not enough historical data to fit HMM for evaluation. Adding default HMM features.")
        eval_data_with_hmm['hmm_state'] = 0
        for i in range(N_HMM_STATES):
            eval_data_with_hmm[f'hmm_state_posterior_{i}'] = 1.0 / N_HMM_STATES

    # 2. Feature Engineering for evaluation set
    eval_df_featured = engineer_features(eval_data_with_hmm, target_col=TARGET_COL)

    if eval_df_featured.empty or PREDICTED_TARGET_COL not in eval_df_featured.columns:
        logging.error("Feature engineering for evaluation set failed or target column missing. Exiting.")
        return

    # Log the final evaluation dataset range after all processing
    if isinstance(eval_df_featured.index, pd.DatetimeIndex):
        final_eval_start_date = eval_df_featured.index.min().strftime('%Y-%m-%d %H:%M:%S')
        final_eval_end_date = eval_df_featured.index.max().strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"Final evaluation dataset after feature engineering: {final_eval_start_date} to {final_eval_end_date} ({len(eval_df_featured)} samples)")

    # 3. Load Trained Model
    try:
        model = load_model(MODEL_LOAD_PATH)
        model_feature_names = model.get_booster().feature_names 
    except Exception as e:
        logging.error(f"Failed to load model or get feature names: {e}. Cannot evaluate.")
        return
        
    if not model_feature_names:
        logging.error("Could not retrieve feature names from the loaded model. Cannot evaluate.")
        return

    # 4. Prepare evaluation data
    aligned_eval_X_df = pd.DataFrame(columns=model_feature_names, index=eval_df_featured.index)
    for col in model_feature_names:
        if col in eval_df_featured:
            aligned_eval_X_df[col] = eval_df_featured[col]
            
    other_cols_to_keep = [PREDICTED_TARGET_COL, TARGET_COL]
    if 'hmm_state' in eval_df_featured.columns: other_cols_to_keep.append('hmm_state')
    if 'temperature_2m' in eval_df_featured.columns: other_cols_to_keep.append('temperature_2m')
    
    for col in other_cols_to_keep:
        if col in eval_df_featured:
             aligned_eval_X_df[col] = eval_df_featured[col]
        elif col == PREDICTED_TARGET_COL and PREDICTED_TARGET_COL not in aligned_eval_X_df: 
             aligned_eval_X_df[PREDICTED_TARGET_COL] = np.nan

    # Ensure target is present
    aligned_eval_X_df.dropna(subset=[PREDICTED_TARGET_COL], inplace=True)
    if aligned_eval_X_df.empty:
        logging.error("Evaluation dataframe is empty after aligning features and ensuring target. Cannot evaluate.")
        return

    # 5. Evaluate Model
    model_metrics, persistence_metrics, eval_results_with_preds = evaluate_model_on_holdout(
        aligned_eval_X_df, model, model_feature_names
    )

    if model_metrics and persistence_metrics:
        logging.info("\n--- Evaluation Summary ---")
        logging.info(f"XGBoost Model: MAE={model_metrics['MAE']:.4f}, RMSE={model_metrics['RMSE']:.4f}, MAPE={model_metrics['MAPE']:.2f}%")
        logging.info(f"Persistence (t-24h): MAE={persistence_metrics['MAE']:.4f}, RMSE={persistence_metrics['RMSE']:.4f}, MAPE={persistence_metrics['MAPE']:.2f}%")
    
        # Add evaluation date range to the summary
        if isinstance(aligned_eval_X_df.index, pd.DatetimeIndex):
            eval_start = aligned_eval_X_df.index.min().strftime('%Y-%m-%d %H:%M:%S')
            eval_end = aligned_eval_X_df.index.max().strftime('%Y-%m-%d %H:%M:%S')
            logging.info(f"Evaluation period: {eval_start} to {eval_end} ({len(aligned_eval_X_df)} samples)")
    
    # 6. Display visualizations
    if not args.save:
        plot_demand_with_hmm(eval_results_with_preds)
        plt.show()
        
        if args.dashboard:
            create_analysis_dashboard(eval_results_with_preds)
            plt.show()
    else:
        hmm_plot_path = os.path.join(save_dir, 'demand_with_hmm_evaluation.png')
        plot_demand_with_hmm(eval_results_with_preds, save_path=hmm_plot_path)
        logging.info(f"Saved HMM plot to {hmm_plot_path}")
        
        if args.dashboard:
            dashboard_path = os.path.join(save_dir, 'dashboard_evaluation.png')
            create_analysis_dashboard(eval_results_with_preds, save_path=dashboard_path)
            logging.info(f"Saved dashboard to {dashboard_path}")
    
    logging.info("Demand model evaluation script finished.")

if __name__ == '__main__':
    main()