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
AGENT_MODEL_LOAD_PATH = 'src/predictions/demand/models/baseload/villamichelin_baseload_model.pkl'
AGENT_MODEL_SPLITS_INFO_PATH = 'src/predictions/demand/models/baseload/villamichelin_baseload_model_splits_info.pkl'
HMM_MODEL_SAVE_PATH = 'src/predictions/demand/models/villamichelin_hmm_model.pkl'
PREDICTED_TARGET_COL = 'y' # Name of the target column model was trained to predict (e.g., consumption.shift(-1))
PLOTS_SAVE_DIR = 'src/predictions/demand/plots/demand_model_evaluation/'
AGENT_PLOTS_SAVE_DIR = 'src/predictions/demand/plots/baseload_model_evaluation/'

# Paths to data for evaluation (typically a hold-out set)
# For this example, we'll re-use the main data paths and simulate a hold-out split.
CONSUMPTION_DATA_PATH = 'data/processed/villamichelin/VillamichelinEnergyData.csv'
HAT_PUMP_DATA_PATH = 'data/processed/villamichelin/Thermia/HeatPumpPower.csv'
WEATHER_DATA_PATH = 'data/processed/weather_data.csv'

# Ensure plots directory exists
os.makedirs(PLOTS_SAVE_DIR, exist_ok=True)
os.makedirs(AGENT_PLOTS_SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(HMM_MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(AGENT_MODEL_LOAD_PATH), exist_ok=True)

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
    BASELOAD_TARGET_COL,
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
        # Make sure we're using aligned data for metrics
        valid_mask = ~eval_df[PREDICTED_TARGET_COL].isna() & ~eval_df['predictions'].isna()
        if valid_mask.sum() > 0:
            actual = eval_df.loc[valid_mask, PREDICTED_TARGET_COL]
            predicted = eval_df.loc[valid_mask, 'predictions']
            
            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            
            # Calculate R²
            r2 = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - actual.mean()) ** 2))
            
            # Create metrics text
            metrics_text = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}'
            bbox_props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
            ax1.annotate(metrics_text, xy=(0.02, 0.98), xycoords='axes fraction', 
                       va='top', ha='left', fontsize=10, bbox=bbox_props)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logging.info(f"Plot saved to {save_path}")
    
    return fig

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
    
    # Create a cleaned DataFrame for plotting, handling NaNs
    plot_df = eval_df.copy()
    
    # Add residuals column if it doesn't exist
    if 'residuals' not in plot_df.columns:
        valid_mask = ~plot_df[PREDICTED_TARGET_COL].isna() & ~plot_df['predictions'].isna()
        plot_df.loc[valid_mask, 'residuals'] = plot_df.loc[valid_mask, PREDICTED_TARGET_COL] - plot_df.loc[valid_mask, 'predictions']
    
    # Panel 2: Residuals by hour
    ax2 = fig.add_subplot(gs[1, 0])
    if 'residuals' in plot_df.columns:
        plot_df['hour'] = plot_df.index.hour
        valid_data = plot_df.dropna(subset=['residuals'])
        if not valid_data.empty:
            sns.boxplot(x='hour', y='residuals', data=valid_data, ax=ax2, palette='Blues')
            ax2.set_title('Prediction Error by Hour')
            ax2.set_xlabel('Hour of Day')
            ax2.set_ylabel('Error (Actual - Predicted)')
            ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Panel 3: HMM state distribution over time
    ax3 = fig.add_subplot(gs[1, 1])
    if 'hmm_state' in plot_df.columns:
        plot_df['day_of_month'] = plot_df.index.day
        pivot_data = plot_df.dropna(subset=['hmm_state'])
        if not pivot_data.empty:
            pivot_df = pivot_data.pivot_table(
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
    if all(col in plot_df.columns for col in ['temperature_2m', 'residuals', 'hmm_state']):
        valid_data = plot_df.dropna(subset=['temperature_2m', 'residuals', 'hmm_state'])
        if not valid_data.empty:
            sns.scatterplot(x='temperature_2m', y='residuals', 
                           hue='hmm_state', data=valid_data, ax=ax4, palette='viridis')
            ax4.set_title('Error vs. Temperature by HMM State')
            ax4.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Panel 5: Error distribution by HMM state
    ax5 = fig.add_subplot(gs[2, 1])
    if all(col in plot_df.columns for col in ['residuals', 'hmm_state']):
        valid_data = plot_df.dropna(subset=['residuals', 'hmm_state'])
        if not valid_data.empty:
            order = sorted(valid_data['hmm_state'].unique())
            sns.violinplot(x='hmm_state', y='residuals', data=valid_data, ax=ax5, 
                          palette='viridis', order=order)
            ax5.set_title('Error Distribution by HMM State')
            ax5.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logging.info(f"Dashboard saved to {save_path}")
    
    return fig

def plot_month_data(period_arg, show_hmm=True, show_dashboard=False, save_dir=None, for_agent=False):
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
    for_agent : bool
        Whether to evaluate agent-specific baseload model instead of standard demand model
    """
    # Parse the month argument
    try:
        parts = period_arg.split('-')
        if len(parts) == 2 and len(parts[0]) == 4:  # Format YYYY-MM
            year = int(parts[0])
            month = int(parts[1])
            start_date = pd.Timestamp(year=year, month=month, day=1, tz='Europe/Stockholm')
            if month == 12:
                end_date = pd.Timestamp(year=year+1, month=1, day=1, tz='Europe/Stockholm') - pd.Timedelta(days=1)
            else:
                end_date = pd.Timestamp(year=year, month=month+1, day=1, tz='Europe/Stockholm') - pd.Timedelta(days=1)
            end_date = end_date.replace(hour=23, minute=59, second=59)
        else:
            raise ValueError("Invalid period format. Use YYYY-MM.")
    except Exception as e:
        logging.error(f"Error parsing period argument: {e}")
        return
    
    logging.info(f"Plotting data from {start_date.date()} to {end_date.date()}")
    
    # Load the model based on whether we're evaluating the agent model
    model_path = AGENT_MODEL_LOAD_PATH if for_agent else MODEL_LOAD_PATH
    target_col = BASELOAD_TARGET_COL if for_agent else TARGET_COL
    plots_dir = AGENT_PLOTS_SAVE_DIR if for_agent else PLOTS_SAVE_DIR
    
    # Load the trained model
    model = load_model(model_path)
    
    # Run data pipeline to get the data
    df = run_data_pipeline(CONSUMPTION_DATA_PATH, HAT_PUMP_DATA_PATH, WEATHER_DATA_PATH, for_agent=for_agent)
    
    # HMM state detection
    logging.info("Fitting HMM with 3 states.")
    hmm_model = fit_hmm(df[target_col], N_HMM_STATES)
    
    # Save the HMM model
    with open(HMM_MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(hmm_model, f)
    logging.info(f"HMM model saved to {HMM_MODEL_SAVE_PATH}")
    
    # Decode states and get posteriors
    states = decode_states(hmm_model, df[target_col])
    posteriors = get_state_posteriors(hmm_model, df[target_col])
    
    # Add HMM features to dataframe
    df_with_hmm = df.copy()
    df_with_hmm['hmm_state'] = states
    for i in range(N_HMM_STATES):
        df_with_hmm[f'hmm_state_posterior_{i}'] = posteriors[:, i]
    
    logging.info("HMM features added to DataFrame.")
    print("\nHMM features added to DataFrame:")
    print(df_with_hmm.head(), "\n", df_with_hmm.tail())
    
    # Engineer features
    df_featured = engineer_features(df_with_hmm, target_col=target_col, create_target=True)
    
    # Filter to just the month we want
    month_df = df_featured[(df_featured.index >= start_date) & (df_featured.index <= end_date)]
    if month_df.empty:
        logging.error(f"No data available for period {period_arg}")
        return
    
    # Get the feature names from the model
    feature_names = model.feature_names_in_
    
    # Handle missing features
    X_month = pd.DataFrame(index=month_df.index)
    for feature in feature_names:
        if feature in month_df.columns:
            X_month[feature] = month_df[feature]
        else:
            logging.warning(f"Feature '{feature}' not found in data. Using 0.")
            X_month[feature] = 0
    
    # Make predictions
    preds = model.predict(X_month)
    
    # Create a DataFrame with actuals and predictions
    plot_df = pd.DataFrame(index=month_df.index)
    plot_df['y'] = month_df[PREDICTED_TARGET_COL]
    plot_df['predictions'] = preds
    plot_df['hmm_state'] = month_df['hmm_state'] if 'hmm_state' in month_df.columns else 0
    
    # Add t-24h persistence model
    plot_df['persistence_pred'] = month_df[target_col].shift(24)
    
    # Align actuals and predictions for metrics calculation
    valid_mask = ~plot_df['y'].isna() & ~plot_df['predictions'].isna() & ~plot_df['persistence_pred'].isna()
    actuals = plot_df.loc[valid_mask, 'y']
    model_preds = plot_df.loc[valid_mask, 'predictions']
    persistence_preds = plot_df.loc[valid_mask, 'persistence_pred']
    
    # Calculate metrics
    if len(actuals) > 0:
        metrics = calculate_metrics(actuals, model_preds, "XGBoost")
        persistence_metrics = calculate_metrics(actuals, persistence_preds, "Persistence (t-24h)")
        
        # Log improvement over persistence
        mae_improvement = (persistence_metrics['MAE'] - metrics['MAE']) / persistence_metrics['MAE'] * 100
        rmse_improvement = (persistence_metrics['RMSE'] - metrics['RMSE']) / persistence_metrics['RMSE'] * 100
        logging.info(f"Model improvement over persistence: MAE: {mae_improvement:.2f}%, RMSE: {rmse_improvement:.2f}%")
    
    # Visualize results
    title = f"{'Baseload' if for_agent else 'Energy Demand'} Prediction - {period_arg}"
    
    if show_hmm:
        hmm_plot_path = os.path.join(plots_dir, f"{'baseload' if for_agent else 'demand'}_hmm_{period_arg}.png") if save_dir else None
        plot_demand_with_hmm(plot_df, title=title, save_path=hmm_plot_path)
        
        if not save_dir:
            plt.show()
    
    if show_dashboard:
        dashboard_path = os.path.join(plots_dir, f"{'baseload' if for_agent else 'demand'}_dashboard_{period_arg}.png") if save_dir else None
        create_analysis_dashboard(plot_df, title=title, save_path=dashboard_path)
        
        if not save_dir:
            plt.show()

def evaluate_model_on_holdout(hold_out_df_featured: pd.DataFrame, model, model_feature_names: list, target_col: str):
    """
    Evaluate model on hold-out dataset and return metrics and results.
    
    Parameters:
    -----------
    hold_out_df_featured : pd.DataFrame
        DataFrame with features and target column
    model : xgboost.XGBRegressor
        Trained model
    model_feature_names : list
        List of feature names expected by the model
    target_col : str
        Name of the target column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with actual values, predictions, and errors
    """
    logging.info("Evaluating model on holdout set")
    
    # Create a copy to avoid modifying the original
    eval_df = hold_out_df_featured.copy()
    
    # Make sure all needed features are available
    X_eval = pd.DataFrame(index=eval_df.index)
    for feature in model_feature_names:
        if feature in eval_df.columns:
            X_eval[feature] = eval_df[feature]
        else:
            logging.warning(f"Feature {feature} not found in evaluation data")
            X_eval[feature] = 0  # Default value
    
    # Ensure target column exists
    if target_col not in eval_df.columns:
        logging.error(f"Target column {target_col} not found in evaluation data")
        return None
    
    # Make predictions
    try:
        predictions = model.predict(X_eval)
        eval_df['predictions'] = predictions
    except Exception as e:
        logging.error(f"Error making predictions: {e}")
        return None
    
    # Create persistence model prediction (t-24h)
    eval_df['persistence_pred'] = eval_df[target_col].shift(24)
    
    # Calculate errors
    eval_df['model_error'] = eval_df[target_col] - eval_df['predictions']
    eval_df['persistence_error'] = eval_df[target_col] - eval_df['persistence_pred']
    
    # Drop rows with missing values in error calculations
    results = eval_df.dropna(subset=['model_error', 'persistence_error'])
    
    # Calculate metrics
    if len(results) > 0:
        model_metrics = calculate_metrics(results[target_col], results['predictions'], "XGBoost Model")
        persistence_metrics = calculate_metrics(results[target_col], results['persistence_pred'], "Persistence Model (t-24h)")
        
        # Log improvement over persistence
        mae_improvement = (persistence_metrics['MAE'] - model_metrics['MAE']) / persistence_metrics['MAE'] * 100
        rmse_improvement = (persistence_metrics['RMSE'] - model_metrics['RMSE']) / persistence_metrics['RMSE'] * 100
        logging.info(f"Model improvement over persistence: MAE: {mae_improvement:.2f}%, RMSE: {rmse_improvement:.2f}%")
    else:
        logging.warning("No valid rows for evaluation after calculating errors")
        return None
    
    return results

def perform_deep_baseload_analysis(df_orig, model, feature_names):
    """
    Performs deep analysis on why the baseload model might be performing poorly.
    
    Parameters:
    -----------
    df_orig : pd.DataFrame
        Original dataframe with all data
    model : object
        Trained model to evaluate
    feature_names : list
        Feature names the model was trained on
    """
    logging.info("="*60)
    logging.info("DEEP BASELOAD MODEL ANALYSIS")
    logging.info("="*60)
    
    # 1. Data Examination
    # -------------------
    # Create a copy to avoid modifying original data
    df = df_orig.copy()
    
    # Check for data quality issues
    logging.info("Data Quality Analysis:")
    logging.info(f"  Total samples: {len(df)}")
    logging.info(f"  Date range: {df.index.min()} to {df.index.max()}")
    
    # Check for missing values
    missing_values = df.isna().sum()
    if missing_values.sum() > 0:
        logging.info("  Missing values detected:")
        for col in missing_values[missing_values > 0].index:
            logging.info(f"    {col}: {missing_values[col]} missing values")
    else:
        logging.info("  No missing values detected.")
    
    # 2. Examine Target Variable Distribution
    # ---------------------------------------
    if 'baseload' in df.columns and 'consumption' in df.columns and 'production' in df.columns:
        logging.info("\nTarget Variable Analysis:")
        
        # Basic statistics
        logging.info("\nBasic Statistics:")
        logging.info(f"  Baseload mean: {df['baseload'].mean():.4f}, std: {df['baseload'].std():.4f}")
        logging.info(f"  Consumption mean: {df['consumption'].mean():.4f}, std: {df['consumption'].std():.4f}")
        logging.info(f"  Production mean: {df['production'].mean():.4f}, std: {df['production'].std():.4f}")
        
        # Check for extreme values
        baseload_quantiles = df['baseload'].quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]).to_dict()
        logging.info("\nBaseload Quantiles:")
        for q, val in baseload_quantiles.items():
            logging.info(f"  {q:.2f}: {val:.4f}")
        
        # Examine zero and near-zero values which can cause MAPE issues
        zero_baseload = (df['baseload'] == 0).sum()
        near_zero_baseload = ((df['baseload'] > 0) & (df['baseload'] < 0.1)).sum()
        logging.info(f"\nZero baseload values: {zero_baseload} ({zero_baseload/len(df)*100:.2f}%)")
        logging.info(f"Near-zero baseload values (<0.1): {near_zero_baseload} ({near_zero_baseload/len(df)*100:.2f}%)")
        
        # Plot baseload distribution
        plt.figure(figsize=(14, 6))
        sns.histplot(df['baseload'], kde=True, bins=50)
        plt.title('Baseload Distribution')
        plt.xlabel('Baseload (kW)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(AGENT_PLOTS_SAVE_DIR, 'baseload_distribution.png'))
        plt.close()
        
        # Compare baseload with consumption and production
        plt.figure(figsize=(14, 6))
        plt.scatter(df['consumption'], df['baseload'], alpha=0.5, label='Baseload vs Consumption')
        plt.plot([0, df['consumption'].max()], [0, df['consumption'].max()], 'r--', label='y=x')
        plt.title('Baseload vs Consumption')
        plt.xlabel('Consumption (kW)')
        plt.ylabel('Baseload (kW)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(AGENT_PLOTS_SAVE_DIR, 'baseload_vs_consumption.png'))
        plt.close()

        # Analyze baseload composition
        df['baseload_minus_consumption'] = df['baseload'] - df['consumption']
        plt.figure(figsize=(14, 6))
        plt.plot(df.index, df['baseload_minus_consumption'])
        plt.title('Difference Between Baseload and Consumption')
        plt.ylabel('Baseload - Consumption (kW)')
        plt.tight_layout()
        plt.savefig(os.path.join(AGENT_PLOTS_SAVE_DIR, 'baseload_minus_consumption.png'))
        plt.close()
    
    # 3. Model Evaluation - Compare with Simpler Models
    # ------------------------------------------------
    logging.info("\nComparing model with simpler alternatives:")
    
    # Make predictions using the trained model (XGBoost)
    if PREDICTED_TARGET_COL in df.columns:
        # Get all shared feature names between model and dataframe
        shared_features = [f for f in feature_names if f in df.columns]
        if len(shared_features) < len(feature_names):
            missing_features = set(feature_names) - set(shared_features)
            logging.warning(f"Missing {len(missing_features)} features for prediction:")
            for f in sorted(list(missing_features))[:10]:  # Show first 10 missing features
                logging.warning(f"  {f}")
                
        # Generate predictions
        x_eval = df[shared_features].values
        preds = model.predict(x_eval)
        df['model_pred'] = preds
        
        # Create multiple persistence baselines
        df['persistence_1h'] = df[BASELOAD_TARGET_COL].shift(1)
        df['persistence_24h'] = df[BASELOAD_TARGET_COL].shift(24)
        df['persistence_168h'] = df[BASELOAD_TARGET_COL].shift(168)  # 1 week ago
        
        # Calculate 24h moving average as another baseline
        df['moving_avg_24h'] = df[BASELOAD_TARGET_COL].rolling(24).mean().shift(1)
        
        # Simple time-of-day average as baseline
        df['hour'] = df.index.hour
        time_of_day_avg = df.groupby('hour')[BASELOAD_TARGET_COL].mean()
        df['time_of_day_avg'] = df['hour'].map(time_of_day_avg)
        
        # Calculate errors
        results = {}
        baselines = {
            'XGBoost Model': 'model_pred', 
            'Persistence (t-1h)': 'persistence_1h',
            'Persistence (t-24h)': 'persistence_24h', 
            'Persistence (t-168h)': 'persistence_168h',
            'Moving Avg (24h)': 'moving_avg_24h',
            'Time of Day Avg': 'time_of_day_avg'
        }
        
        # Calculate metrics for each baseline
        valid_mask = ~df[PREDICTED_TARGET_COL].isna()
        for name, col in baselines.items():
            valid = valid_mask & ~df[col].isna()
            if valid.sum() > 0:
                y_true = df.loc[valid, PREDICTED_TARGET_COL]
                y_pred = df.loc[valid, col]
                
                # Calculate metrics
                results[name] = calculate_metrics(y_true, y_pred, name)
        
        # Compare performance improvement/degradation
        if 'XGBoost Model' in results and 'Persistence (t-24h)' in results:
            xgb_metrics = results['XGBoost Model']
            persistence_metrics = results['Persistence (t-24h)']
            
            for metric in ['MAE', 'RMSE', 'MAPE']:
                improvement = (persistence_metrics[metric] - xgb_metrics[metric]) / persistence_metrics[metric] * 100
                logging.info(f"Model {metric} improvement over persistence: {improvement:.2f}%")
            
        # 4. Error Analysis by Time Period
        # --------------------------------
        logging.info("\nError Analysis by Time Period:")
        
        # Create abs error column
        df['abs_error'] = np.abs(df['model_pred'] - df[PREDICTED_TARGET_COL])
        
        # Analysis by hour of day
        hour_error = df.groupby(df.index.hour)['abs_error'].mean().to_dict()
        logging.info("\nMean Absolute Error by Hour:")
        for hour, error in sorted(hour_error.items()):
            logging.info(f"  Hour {hour}: {error:.4f}")
            
        # Plot error by hour
        plt.figure(figsize=(14, 6))
        sns.barplot(x=list(hour_error.keys()), y=list(hour_error.values()))
        plt.title('Mean Absolute Error by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Mean Absolute Error')
        plt.tight_layout()
        plt.savefig(os.path.join(AGENT_PLOTS_SAVE_DIR, 'error_by_hour.png'))
        plt.close()
        
        # Analysis by day of week
        weekday_error = df.groupby(df.index.dayofweek)['abs_error'].mean().to_dict()
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        logging.info("\nMean Absolute Error by Day of Week:")
        for day_num, error in sorted(weekday_error.items()):
            logging.info(f"  {days[day_num]}: {error:.4f}")
            
        # Plot error by day of week
        plt.figure(figsize=(14, 6))
        sns.barplot(x=[days[i] for i in sorted(weekday_error.keys())], 
                    y=[weekday_error[i] for i in sorted(weekday_error.keys())])
        plt.title('Mean Absolute Error by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Mean Absolute Error')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(AGENT_PLOTS_SAVE_DIR, 'error_by_weekday.png'))
        plt.close()
        
        # 5. Additional error analysis
        # ----------------------------
        if 'temperature_2m' in df.columns:
            # Analyze error by temperature
            df['temp_bin'] = pd.cut(df['temperature_2m'], bins=10)
            temp_error = df.groupby('temp_bin')['abs_error'].mean()
            
            plt.figure(figsize=(14, 6))
            temp_error.plot(kind='bar')
            plt.title('Mean Absolute Error by Temperature Range')
            plt.xlabel('Temperature Range (°C)')
            plt.ylabel('Mean Absolute Error')
            plt.tight_layout()
            plt.savefig(os.path.join(AGENT_PLOTS_SAVE_DIR, 'error_by_temperature.png'))
            plt.close()
        
        # 6. Plot prediction residuals vs true values to detect patterns
        # -------------------------------------------------------------
        plt.figure(figsize=(14, 6))
        valid = ~df[PREDICTED_TARGET_COL].isna() & ~df['model_pred'].isna()
        plt.scatter(df.loc[valid, PREDICTED_TARGET_COL], 
                   df.loc[valid, 'model_pred'] - df.loc[valid, PREDICTED_TARGET_COL],
                   alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Residuals vs True Values')
        plt.xlabel('True Values')
        plt.ylabel('Residuals (Predicted - True)')
        plt.tight_layout()
        plt.savefig(os.path.join(AGENT_PLOTS_SAVE_DIR, 'residuals_vs_true.png'))
        plt.close()
        
        # 7. Recommendations
        # -----------------
        logging.info("\nPossible Issues and Recommendations:")
        
        # Check for near-zero baseload values (MAPE issue)
        if zero_baseload > 0 or near_zero_baseload > 0:
            logging.info("1. MAPE is unreliable due to zero/near-zero baseload values.")
            logging.info("   Recommendation: Use MAE and RMSE as primary metrics instead of MAPE.")
        
        # Check for large outliers
        if df['baseload'].max() > df['baseload'].quantile(0.99) * 2:
            logging.info("2. Baseload contains extreme outliers that may affect model training.")
            logging.info("   Recommendation: Consider capping extreme values during preprocessing.")
        
        # Compare relative model performance
        model_vs_persistence = {}
        for name, metrics in results.items():
            if name != 'XGBoost Model':
                mae_diff = (results['XGBoost Model']['MAE'] / metrics['MAE'] - 1) * 100
                model_vs_persistence[name] = mae_diff
        
        better_models = [name for name, diff in model_vs_persistence.items() if diff < -5]
        if better_models:
            logging.info(f"3. The following simpler models outperformed XGBoost by >5%: {', '.join(better_models)}")
            logging.info("   Recommendation: Consider using a simpler model or ensemble approach.")
        
        # Check for hour-specific issues
        worst_hours = sorted(hour_error.items(), key=lambda x: x[1], reverse=True)[:3]
        if worst_hours and worst_hours[0][1] > df['abs_error'].mean() * 1.5:
            hours_str = ', '.join([f"{h}:00" for h, _ in worst_hours])
            logging.info(f"4. Model performs particularly poorly during these hours: {hours_str}")
            logging.info("   Recommendation: Add hour-specific features or train separate models for these hours.")
        
        # Check for temporal patterns in error
        df['date'] = df.index.date
        date_error = df.groupby('date')['abs_error'].mean()
        
        if date_error.max() > date_error.mean() * 2:
            worst_date = date_error.idxmax()
            logging.info(f"5. Model has unusually high errors on {worst_date}.")
            logging.info("   Recommendation: Examine external factors or data quality issues for this date.")

        logging.info("\nDeep analysis complete. Results saved to the baseload_model_evaluation directory.")

def main():
    # Simplified command line arguments
    parser = argparse.ArgumentParser(description='Evaluate demand prediction model')
    parser.add_argument('--month', type=str, help='Month to evaluate in YYYY-MM format')
    parser.add_argument('--show-hmm', action='store_true', help='Show HMM states in plots')
    parser.add_argument('--show-dashboard', action='store_true', help='Show comprehensive dashboard')
    parser.add_argument('--for-agent', action='store_true', help='Evaluate agent-specific baseload model instead of standard demand model')
    parser.add_argument('--deep-analysis', action='store_true', help='Perform deep analysis of baseload model performance')
    args = parser.parse_args()
    
    # Select model path based on mode
    model_path = AGENT_MODEL_LOAD_PATH if args.for_agent else MODEL_LOAD_PATH
    splits_info_path = AGENT_MODEL_SPLITS_INFO_PATH if args.for_agent else MODEL_SPLITS_INFO_PATH
    plots_dir = AGENT_PLOTS_SAVE_DIR if args.for_agent else PLOTS_SAVE_DIR
    target_col = BASELOAD_TARGET_COL if args.for_agent else TARGET_COL
    
    # Load model splits info if available
    model_splits_info = load_splits_info(splits_info_path)
    
    # Display model training splits info if available
    if model_splits_info:
        logging.info("\n=== MODEL TRAINING DATA SPLITS ===")
        if 'full_dataset' in model_splits_info:
            logging.info(f"Full dataset: {model_splits_info['full_dataset']['start_date']} to {model_splits_info['full_dataset']['end_date']} ({model_splits_info['full_dataset']['n_samples']} samples)")
        
        if 'splits' in model_splits_info:
            for i, split in enumerate(model_splits_info['splits']):
                logging.info(f"Split {i+1}: Train {split['train_start']} to {split['train_end']} ({split['train_samples']} samples), " +
                         f"Val {split['val_start']} to {split['val_end']} ({split['val_samples']} samples)")
    else:
        logging.warning("No model splits information available. Continuing with evaluation.")
    
    # If a specific month is provided, just plot that month's data
    if args.month:
        plot_month_data(
            args.month, 
            show_hmm=args.show_hmm,
            show_dashboard=args.show_dashboard,
            save_dir=plots_dir,
            for_agent=args.for_agent
        )
        return
    
    # Otherwise, load the model and evaluate on all data
    logging.info(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    # Get feature names from model
    feature_names = model.feature_names_in_
    
    # Run data pipeline to get evaluation data
    logging.info("Running data pipeline to prepare evaluation data")
    df = run_data_pipeline(CONSUMPTION_DATA_PATH, HAT_PUMP_DATA_PATH, WEATHER_DATA_PATH, for_agent=args.for_agent)
    
    # Engineer features (same as in training)
    df_featured = engineer_features(df, target_col=target_col, create_target=True)
    
    # Perform deep analysis if requested
    if args.deep_analysis and args.for_agent:
        perform_deep_baseload_analysis(df, model, feature_names)
    
    # Evaluate model and get metrics
    results_df = evaluate_model_on_holdout(df_featured, model, feature_names, PREDICTED_TARGET_COL)
    
    # Plot overall results
    create_analysis_dashboard(results_df, title=f"{'Baseload' if args.for_agent else 'Demand'} Model Evaluation", 
                              save_path=os.path.join(plots_dir, 'full_evaluation_dashboard.png'))
    
    # Plot recent months for detailed analysis
    recent_months = sorted(list(set([f"{ts.year}-{ts.month:02d}" for ts in results_df.index])))[-3:]
    logging.info(f"Plotting detailed results for recent months: {recent_months}")
    
    for month in recent_months:
        plot_month_data(month, show_hmm=True, show_dashboard=True, save_dir=plots_dir, for_agent=args.for_agent)
        
    logging.info(f"Evaluation complete. Results saved to {plots_dir}")

if __name__ == "__main__":
    main()