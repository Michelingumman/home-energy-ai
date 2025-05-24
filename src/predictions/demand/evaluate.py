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
from datetime import datetime, timedelta
from matplotlib.dates import DayLocator, DateFormatter
import matplotlib.dates as mdates

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Set aesthetic style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8,
    'axes.titlesize': 4,
    'axes.labelsize': 6,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

from src.predictions.demand.train import CONSUMPTION_DATA_PATH, HEAT_PUMP_DATA_PATH, WEATHER_DATA_PATH, MODEL_SAVE_PATH

# Constants - Mirroring train.py for consistency
MODEL_LOAD_PATH = MODEL_SAVE_PATH
MODEL_SPLITS_INFO_PATH = 'src/predictions/demand/models/villamichelin_demand_model_splits_info.pkl'
PREDICTED_TARGET_COL = 'y' # Name of the target column model was trained to predict (e.g., consumption.shift(-1))
PLOTS_SAVE_DIR = 'src/predictions/demand/plots/demand_model_evaluation/'
EVAL_RESULTS_DIR = 'src/predictions/demand/evaluation_results/'

# Ensure plots directory exists
os.makedirs(PLOTS_SAVE_DIR, exist_ok=True)
os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

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

# --- Core Evaluation Functions ---
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

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Use a more robust MAPE calculation with a minimum threshold
    # to avoid division by very small numbers that inflate MAPE
    threshold = 0.5  # Increased threshold for energy consumption data
    mask = np.abs(y_true) > threshold
    if not np.any(mask):
        # If no values above threshold, use symmetric MAPE
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask_denom = denominator > 0.1
        if not np.any(mask_denom):
            return np.nan
        return np.mean(np.abs(y_true[mask_denom] - y_pred[mask_denom]) / denominator[mask_denom]) * 100
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # R² calculation
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

# --- Aesthetic Dashboard Function ---
def create_aesthetic_weekly_dashboard(eval_df, model_metrics, title="Energy Demand - Weekly Analysis", save_path=None):
    """
    Create a clean, aesthetic dashboard for one week of evaluation data with HMM and temperature analysis.
    """
    if eval_df.empty:
        logging.warning("Empty evaluation dataframe provided for dashboard")
        return None
    
    # Prepare data
    actual = eval_df[PREDICTED_TARGET_COL]
    predicted = eval_df['predictions']
    residuals = actual - predicted
    
    # Create beautiful color palette
    colors = {
        'actual': '#2E86AB',      # Ocean blue
        'predicted': 'orange',   # Magenta
        'residual': '#F18F01',    # Orange
        'hmm_0': '#440154',       # Purple
        'hmm_1': '#31688e',       # Teal
        'hmm_2': '#35b779',       # Green
        'temp_cold': '#3498db',   # Light blue
        'temp_warm': '#e74c3c'    # Red
    }
    
    # Create figure with clean layout
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('white')
    
    # Define grid layout: 3 rows, 4 columns
    gs = fig.add_gridspec(3, 4, height_ratios=[2.5, 1, 1], hspace=0.3, wspace=0.25)
    
    # === 1. MAIN TIME SERIES (Top, spanning all columns) ===
    ax_main = fig.add_subplot(gs[0, :])
    
    # Plot main time series with smooth lines
    dates = eval_df.index
    ax_main.plot(dates, actual, color=colors['actual'], linewidth=1.0, 
                label='Actual Consumption', alpha=0.9, zorder=3)
    ax_main.plot(dates, predicted, color=colors['predicted'], linewidth=1, 
                label='Predicted', alpha=0.8, zorder=2)
    
    # Add confidence bands
    std_residual = np.std(residuals)
    ax_main.fill_between(dates, predicted - std_residual, predicted + std_residual, 
                        color=colors['predicted'], alpha=0.15, zorder=1, 
                        label='±1 Std Prediction')
    
    # Styling
    ax_main.set_title(title, fontsize=18, fontweight='bold', pad=20, color='#2c3e50')
    ax_main.set_ylabel('Energy Consumption (kWh)', fontsize=13, fontweight='bold')
    ax_main.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax_main.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Format x-axis beautifully
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%a %m/%d'))
    ax_main.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=0, ha='center')
    
    # Add metrics box
    metrics_text = f"RMSE: {model_metrics['RMSE']:.3f} | MAE: {model_metrics['MAE']:.3f} | R²: {model_metrics['R2']:.3f}"
    ax_main.text(0.02, 0.98, metrics_text, transform=ax_main.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, 
                         edgecolor='#bdc3c7'), fontsize=12, verticalalignment='top',
                fontweight='bold')
    
    # === 2. HMM STATES BY HOUR (Middle left) ===
    ax_hmm_hour = fig.add_subplot(gs[1, :2])
    
    if 'hmm_state' in eval_df.columns:
        # Create hourly HMM state distribution
        eval_df_copy = eval_df.copy()
        eval_df_copy['hour'] = eval_df_copy.index.hour
        eval_df_copy['day_name'] = eval_df_copy.index.strftime('%a')
        
        # Pivot for heatmap
        hmm_hourly = eval_df_copy.pivot_table(
            index='day_name', columns='hour', values='hmm_state', aggfunc='mean'
        )
        
        # Create beautiful heatmap
        sns.heatmap(hmm_hourly, cmap='viridis', ax=ax_hmm_hour, 
                   cbar_kws={'label': 'HMM State', 'shrink': 0.8},
                   linewidths=0.5, linecolor='white')
        
        ax_hmm_hour.set_title('HMM States by Hour & Day', fontsize=14, fontweight='bold', pad=15)
        ax_hmm_hour.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
        ax_hmm_hour.set_ylabel('Day of Week', fontsize=12, fontweight='bold')
        ax_hmm_hour.tick_params(rotation=0)
    
    # === 3. TEMPERATURE BY HOUR (Middle right) ===
    ax_temp_hour = fig.add_subplot(gs[1, 2:])
    
    if 'temperature_2m' in eval_df.columns:
        eval_df_copy['temperature'] = eval_df_copy['temperature_2m']
        
        # Create temperature heatmap
        temp_hourly = eval_df_copy.pivot_table(
            index='day_name', columns='hour', values='temperature', aggfunc='mean'
        )
        
        sns.heatmap(temp_hourly, cmap='RdYlBu_r', ax=ax_temp_hour,
                   cbar_kws={'label': 'Temperature (°C)', 'shrink': 0.8},
                   linewidths=0.5, linecolor='white')
        
        ax_temp_hour.set_title('Temperature by Hour & Day', fontsize=14, fontweight='bold', pad=15)
        ax_temp_hour.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
        ax_temp_hour.set_ylabel('Day of Week', fontsize=12, fontweight='bold')
        ax_temp_hour.tick_params(rotation=0)
    
    # === 4. HMM STATE TIMELINE (Bottom left) ===
    ax_hmm_timeline = fig.add_subplot(gs[2, :2])
    
    if 'hmm_state' in eval_df.columns:
        # Create clean timeline of HMM states
        states = eval_df['hmm_state']
        state_colors = [colors['hmm_0'], colors['hmm_1'], colors['hmm_2']]
        
        for i, (date, state) in enumerate(zip(dates, states)):
            color = state_colors[int(state)] if int(state) < len(state_colors) else '#gray'
            ax_hmm_timeline.barh(0, 1, left=i, height=0.6, color=color, alpha=0.8)
        
        ax_hmm_timeline.set_xlim(0, len(dates))
        ax_hmm_timeline.set_ylim(-0.5, 0.5)
        ax_hmm_timeline.set_title('HMM State Timeline', fontsize=14, fontweight='bold')
        ax_hmm_timeline.set_xlabel('Time Progression', fontsize=12)
        ax_hmm_timeline.set_yticks([])
        
        # Add state legend
        unique_states = sorted(states.unique())
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=state_colors[int(state)], 
                                       label=f'State {int(state)}') for state in unique_states]
        ax_hmm_timeline.legend(handles=legend_elements, loc='center left', 
                              bbox_to_anchor=(1, 0.5), frameon=False)
    
    # === 5. RESIDUALS ANALYSIS (Bottom right) ===
    ax_residuals = fig.add_subplot(gs[2, 2:])
    
    # Beautiful residuals plot
    ax_residuals.plot(dates, residuals, color=colors['residual'], linewidth=1.5, alpha=0.8)
    ax_residuals.axhline(y=0, color='#e74c3c', linestyle='--', alpha=0.8, linewidth=1)
    ax_residuals.fill_between(dates, residuals, 0, alpha=0.3, color=colors['residual'])
    
    # Add confidence bands for residuals
    std_res = np.std(residuals)
    ax_residuals.axhline(y=std_res, color='#95a5a6', linestyle=':', alpha=0.6)
    ax_residuals.axhline(y=-std_res, color='#95a5a6', linestyle=':', alpha=0.6)
    
    ax_residuals.set_title('Prediction Residuals', fontsize=14, fontweight='bold')
    ax_residuals.set_ylabel('Residual (kWh)', fontsize=12, fontweight='bold')
    ax_residuals.set_xlabel('Time', fontsize=12, fontweight='bold')
    
    # Format x-axis for residuals
    ax_residuals.xaxis.set_major_formatter(mdates.DateFormatter('%a'))
    ax_residuals.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.setp(ax_residuals.xaxis.get_majorticklabels(), rotation=0)
    
    # Final styling
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        logging.info(f"✅ Aesthetic dashboard saved to {save_path}")
    
    plt.show()
    return fig

# --- Main Evaluation Function ---
def evaluate_model_on_test_period(eval_data_featured: pd.DataFrame, model, model_feature_names: list):
    """
    Evaluate model on test data (2 weeks) with clean metrics calculation.
    """
    logging.info(f"Evaluating model on {len(eval_data_featured)} samples")
    
    # Ensure target column exists
    if PREDICTED_TARGET_COL not in eval_data_featured.columns:
        logging.error(f"Target column '{PREDICTED_TARGET_COL}' not found")
        return None, None
    
    # Prepare features and target
    y_true = eval_data_featured[PREDICTED_TARGET_COL].copy()
    X_eval = eval_data_featured.drop(columns=[PREDICTED_TARGET_COL])
    
    # Ensure feature consistency
    if model_feature_names:
        missing_features = set(model_feature_names) - set(X_eval.columns)
        if missing_features:
            logging.warning(f"Adding {len(missing_features)} missing features with zeros")
            for feature in missing_features:
                X_eval[feature] = 0
        
        # Reorder to match training
        X_eval = X_eval[model_feature_names]
    
    # Make predictions
    try:
        y_pred = model.predict(X_eval)
        logging.info("✅ Predictions completed successfully")
    except Exception as e:
        logging.error(f"Error making predictions: {e}")
        return None, None
    
    # Calculate metrics
    model_metrics = calculate_metrics(y_true, y_pred)
    
    # Create results DataFrame
    results_df = eval_data_featured.copy()
    results_df['predictions'] = y_pred
    results_df['residuals'] = y_true - y_pred
    
    # Log summary
    logging.info("📊 Model Performance:")
    logging.info(f"   RMSE: {model_metrics['RMSE']:.4f}")
    logging.info(f"   MAE:  {model_metrics['MAE']:.4f}")
    logging.info(f"   MAPE: {model_metrics['MAPE']:.2f}%")
    logging.info(f"   R²:   {model_metrics['R2']:.4f}")
    
    return model_metrics, results_df

def load_model_and_info():
    """Load the trained model and its metadata"""
    try:
        # Load model
        with open(MODEL_SAVE_PATH, 'rb') as f:
            model = pickle.load(f)
        logging.info(f"Model loaded from {MODEL_SAVE_PATH}")
        
        # Load splits info
        splits_info_path = MODEL_SAVE_PATH.replace('.pkl', '_splits_info.pkl')
        splits_info = None
        if os.path.exists(splits_info_path):
            with open(splits_info_path, 'rb') as f:
                splits_info = pickle.load(f)
            logging.info(f"Model splits info loaded")
        
        # Load feature columns
        feature_columns_path = MODEL_SAVE_PATH.replace('.pkl', '_feature_columns.pkl')
        feature_columns = None
        if os.path.exists(feature_columns_path):
            with open(feature_columns_path, 'rb') as f:
                feature_columns = pickle.load(f)
            logging.info(f"Feature columns loaded")
            
        return model, splits_info, feature_columns
        
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def get_evaluation_data(full_data, model_splits_info):
    """
    Get two weeks of evaluation data from the test set, with sufficient historical context.
    """
    if model_splits_info is None:
        logging.warning("No splits info. Using recent 2 weeks with context.")
        # Get 5 weeks, use middle 2 weeks for evaluation
        recent_data = full_data.tail(840)  # 5 weeks * 168 hours
        if len(recent_data) >= 672:  # At least 4 weeks
            eval_data = recent_data.iloc[-672:-336]  # Third and fourth weeks from end
        else:
            eval_data = recent_data.tail(336)  # Fallback to last 2 weeks
        return recent_data, eval_data.index[0], eval_data.index[-1]
    
    approach = model_splits_info.get('approach', 'unknown')
    
    if approach == 'proper_temporal_splits':
        # Use test set
        test_info = model_splits_info['test']
        test_start = pd.Timestamp(test_info['start_date']).tz_localize('Europe/Stockholm')
        test_end = pd.Timestamp(test_info['end_date']).tz_localize('Europe/Stockholm')
        
        # Get test data
        test_data = full_data[(full_data.index >= test_start) & (full_data.index <= test_end)]
        
        # Select two weeks from the middle of test set for evaluation
        if len(test_data) > 672:  # More than 4 weeks available
            mid_point = len(test_data) // 2
            eval_start_idx = max(336, mid_point - 168)  # Ensure at least 2 weeks of history
            eval_end_idx = eval_start_idx + 336  # Two weeks evaluation period
            
            # Get evaluation period boundaries
            eval_start_time = test_data.index[eval_start_idx]
            eval_end_time = test_data.index[eval_end_idx - 1]
            
            # Return data with context (from start of test to end of evaluation period)
            context_data = test_data.iloc[:eval_end_idx]
            
        elif len(test_data) >= 672:  # Exactly 4 weeks
            # Use third and fourth weeks for evaluation
            eval_start_time = test_data.index[336]
            eval_end_time = test_data.index[671]
            context_data = test_data
            
        else:
            # Less than 4 weeks available - need to get more context from before test period
            logging.warning(f"Test set has only {len(test_data)} hours. Getting additional context.")
            
            # Get additional context from before test start
            context_start = test_start - pd.Timedelta(hours=672)  # 4 weeks before
            context_data = full_data[(full_data.index >= context_start) & (full_data.index <= test_end)]
            
            # Use the last 2 weeks of test data for evaluation
            eval_start_time = test_data.index[-336] if len(test_data) >= 336 else test_data.index[0]
            eval_end_time = test_data.index[-1]
        
        logging.info(f"✅ Using 2 weeks from test set with context:")
        logging.info(f"   Evaluation period: {eval_start_time} to {eval_end_time}")
        logging.info(f"   Context data: {len(context_data)} samples ({context_data.index[0]} to {context_data.index[-1]})")
        
        return context_data, eval_start_time, eval_end_time
        
    else:
        # Fallback for legacy models
        logging.warning("Using recent 2 weeks for legacy model with context")
        recent_data = full_data.tail(840)  # 5 weeks
        if len(recent_data) >= 672:
            eval_start_time = recent_data.index[-672]  # Start of third-to-last 2 weeks
            eval_end_time = recent_data.index[-337]    # End of third-to-last 2 weeks
        else:
            eval_start_time = recent_data.index[-336]
            eval_end_time = recent_data.index[-1]
        
        return recent_data, eval_start_time, eval_end_time

def main():
    """Main evaluation function focused on aesthetic 2-week dashboard"""
    parser = argparse.ArgumentParser(description='Evaluate demand prediction model with aesthetic 2-week dashboard')
    parser.add_argument('--save-only', action='store_true', help='Save plots without displaying')
    
    args = parser.parse_args()
    
    try:
        # Load model and metadata
        logging.info("🔄 Loading model and metadata...")
        model, model_splits_info, feature_columns = load_model_and_info()
        
        # Load and process data
        logging.info("🔄 Loading and processing data...")
        full_data = run_data_pipeline(CONSUMPTION_DATA_PATH, HEAT_PUMP_DATA_PATH, WEATHER_DATA_PATH)
        
        # Add HMM features
        full_data = add_hmm_features(full_data, TARGET_COL, N_HMM_STATES)
        
        # Get evaluation data with sufficient context for feature engineering
        eval_data_raw, eval_start_time, eval_end_time = get_evaluation_data(full_data, model_splits_info)
        
        if len(eval_data_raw) < 24:
            logging.error("❌ Insufficient evaluation data")
            return
        
        # Engineer features on the full context data
        logging.info("🔄 Engineering features with historical context...")
        eval_data_featured = engineer_features(eval_data_raw, TARGET_COL, create_target=True)
        
        # Now filter to just the evaluation period (2 weeks) after feature engineering
        eval_period_featured = eval_data_featured[
            (eval_data_featured.index >= eval_start_time) & 
            (eval_data_featured.index <= eval_end_time)
        ].copy()
        
        if len(eval_period_featured) == 0:
            logging.error("❌ No data in evaluation period after feature engineering")
            return
        
        logging.info(f"✅ Two-week evaluation data prepared: {len(eval_period_featured)} samples")
        logging.info(f"   Final evaluation period: {eval_period_featured.index[0]} to {eval_period_featured.index[-1]}")
        
        # Evaluate model
        logging.info("🔄 Running model evaluation...")
        model_metrics, eval_results = evaluate_model_on_test_period(
            eval_period_featured, model, feature_columns)
        
        if model_metrics is None:
            logging.error("❌ Model evaluation failed")
            return
        
        # Create aesthetic dashboard
        logging.info("🎨 Creating aesthetic 2-week dashboard...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(PLOTS_SAVE_DIR, f'aesthetic_2week_dashboard_{timestamp}.png')
        
        # Get period dates for title  
        start_date = eval_results.index[0].strftime('%B %d')
        end_date = eval_results.index[-1].strftime('%B %d, %Y')
        title = f"Energy Demand Analysis - 2 Weeks: {start_date} to {end_date}"
        
        create_aesthetic_weekly_dashboard(
            eval_results,
            model_metrics,
            title=title,
            save_path=save_path if not args.save_only else save_path
        )
        
        # Save evaluation results
        results_path = os.path.join(EVAL_RESULTS_DIR, f'2week_evaluation_{timestamp}.csv')
        eval_results.to_csv(results_path)
        logging.info(f"✅ Results saved to {results_path}")
        
        # Final summary
        logging.info("="*60)
        logging.info("🎉 2-WEEK EVALUATION COMPLETED")
        logging.info("="*60)
        logging.info(f"📊 Performance Summary:")
        logging.info(f"   RMSE: {model_metrics['RMSE']:.4f}")
        logging.info(f"   MAE:  {model_metrics['MAE']:.4f}")
        logging.info(f"   MAPE: {model_metrics['MAPE']:.2f}%")
        logging.info(f"   R²:   {model_metrics['R2']:.4f}")
        logging.info(f"📁 Dashboard saved to: {save_path}")
        logging.info("="*60)
        
    except Exception as e:
        logging.error(f"❌ Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()