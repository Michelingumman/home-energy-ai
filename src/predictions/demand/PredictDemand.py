import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import joblib
import logging
import sys
import json
import traceback
from scipy.stats import pearsonr
import pickle

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants - using absolute paths to ensure reliability
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
CONSUMPTION_FILE = os.path.join(BASE_DIR, 'data/processed/VillamichelinConsumption.csv')
WEATHER_FILE = os.path.join(BASE_DIR, 'data/processed/weather_data.csv')
TIME_FEATURES_FILE = os.path.join(BASE_DIR, 'data/processed/time_features.csv')
HOLIDAYS_FILE = os.path.join(BASE_DIR, 'data/processed/holidays.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'src/predictions/demand/models')

# Create model directory if it doesn't exist
os.makedirs(MODEL_PATH, exist_ok=True)

def load_consumption_data():
    """
    Load energy consumption data from CSV
    """
    logger.info(f"Loading consumption data from {CONSUMPTION_FILE}...")
    
    try:
        # Check if file exists
        if not os.path.exists(CONSUMPTION_FILE):
            logger.error(f"Consumption file not found: {CONSUMPTION_FILE}")
            raise FileNotFoundError(f"Consumption file not found: {CONSUMPTION_FILE}")
            
        # Read file with explicit UTC parsing
        df = pd.read_csv(CONSUMPTION_FILE)
        
        # Handle the unit column before converting to datetime index
        unit = 'kWh'  # Default unit
        if 'unit' in df.columns:
            unit = df['unit'].iloc[0]
            df = df.drop(columns=['unit'])
        
        # Convert timestamp to datetime with explicit UTC
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Set timestamp as index BEFORE resampling
        df = df.set_index('timestamp')
        
        # Now we can safely resample - using ffill() instead of fillna(method='ffill')
        df = df.resample('h').mean().ffill()
        
        # Store unit as attribute
        df.attrs['unit'] = unit
        
        logger.info(f"Consumption data loaded: {len(df)} records from {df.index.min()} to {df.index.max()}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading consumption data: {e}")
        raise

def load_weather_data():
    """
    Load historical weather data from CSV file
    """
    logger.info(f"Loading weather data from {WEATHER_FILE}...")
    
    try:
        if not os.path.exists(WEATHER_FILE):
            logger.warning(f"Weather file not found: {WEATHER_FILE}")
            return pd.DataFrame()
            
        # Read file
        weather_df = pd.read_csv(WEATHER_FILE)
        
        if weather_df.empty:
            return pd.DataFrame()
            
        # Handle date column, be flexible with column name
        date_col = 'date' if 'date' in weather_df.columns else 'timestamp'
        if date_col not in weather_df.columns:
            logger.warning("No date/timestamp column found in weather data")
            return pd.DataFrame()
            
        weather_df[date_col] = pd.to_datetime(weather_df[date_col], utc=True)
        weather_df = weather_df.set_index(date_col)
        
        # Rename weather columns for consistency
        if 'temperature_2m' in weather_df.columns and 'temp' not in weather_df.columns:
            weather_df['temp'] = weather_df['temperature_2m']
        
        if 'relative_humidity_2m' in weather_df.columns and 'humidity' not in weather_df.columns:
            weather_df['humidity'] = weather_df['relative_humidity_2m']
            
        if 'cloud_cover' in weather_df.columns and 'clouds' not in weather_df.columns:
            weather_df['clouds'] = weather_df['cloud_cover']
            
        if 'wind_speed_100m' in weather_df.columns and 'wind_speed' not in weather_df.columns:
            weather_df['wind_speed'] = weather_df['wind_speed_100m']
            
        logger.info(f"Weather data loaded: {len(weather_df)} records")
        return weather_df
        
    except Exception as e:
        logger.warning(f"Error loading weather data: {e}")
        return pd.DataFrame()

def load_time_features():
    """
    Load time features from CSV file
    """
    logger.info(f"Loading time features from {TIME_FEATURES_FILE}...")
    
    try:
        if not os.path.exists(TIME_FEATURES_FILE):
            logger.warning(f"Time features file not found")
            return pd.DataFrame()
            
        # Try reading with index_col first
        time_df = pd.read_csv(TIME_FEATURES_FILE, index_col=0)
        time_df.index = pd.to_datetime(time_df.index, utc=True)
        
        logger.info(f"Time features loaded: {len(time_df)} records")
        return time_df
        
    except Exception as e:
        logger.warning(f"Error loading time features: {e}")
        return pd.DataFrame()

def load_holidays():
    """
    Load holidays data from CSV file
    """
    logger.info(f"Loading holidays data from {HOLIDAYS_FILE}...")
    
    try:
        if not os.path.exists(HOLIDAYS_FILE):
            logger.warning(f"Holidays file not found")
            return pd.DataFrame()
            
        holidays_df = pd.read_csv(HOLIDAYS_FILE, index_col=0)
        holidays_df.index = pd.to_datetime(holidays_df.index, utc=True)
        
        logger.info(f"Holidays data loaded: {len(holidays_df)} records")
        return holidays_df
        
    except Exception as e:
        logger.warning(f"Error loading holidays data: {e}")
        return pd.DataFrame()

def create_lag_features(df, target_col='consumption', lag_hours=[1, 24, 48, 168]):
    """
    Create lag features based on previous consumption values
    """
    logger.info(f"Creating lag features")
    result_df = df.copy()
    
    # Create lag features
    for lag in lag_hours:
        result_df[f'{target_col}_lag_{lag}h'] = result_df[target_col].shift(lag)
    
    # Create rolling mean features - simpler for stability
    windows = [24, 168]  # 24h, 1 week
    for window in windows:
        result_df[f'{target_col}_rolling_mean_{window}h'] = result_df[target_col].rolling(window=window, min_periods=1).mean()
    
    return result_df

def analyze_residuals(y_test, y_pred, df, weather_df):
    """
    Create advanced residual analysis plots to diagnose model performance.
    
    Args:
        y_test: Actual consumption values
        y_pred: Predicted consumption values
        df: The feature dataframe
        weather_df: Weather dataframe
    """
    logger.info("Creating residual analysis plots...")
    
    try:
        # Create a directory for residual plots
        residual_plots_dir = os.path.join(MODEL_PATH, 'residual_plots')
        os.makedirs(residual_plots_dir, exist_ok=True)
        
        # Create dataframe with residuals
        eval_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'residual': y_test - y_pred,  # Actual - Predicted
            'abs_residual': abs(y_test - y_pred),
            'hour': y_test.index.hour,
            'is_weekend': y_test.index.dayofweek >= 5
        })
        
        # 1. Residuals vs Hour of Day
        plt.figure(figsize=(12, 6))
        
        # Group by hour and calculate mean residual
        hourly_residuals = eval_df.groupby('hour')['residual'].mean()
        hourly_abs_residuals = eval_df.groupby('hour')['abs_residual'].mean()
        
        # Create plot with both signed and absolute residuals
        ax = hourly_residuals.plot(kind='bar', alpha=0.7, color='blue', label='Mean Residual (+ = Underprediction)')
        hourly_abs_residuals.plot(kind='bar', alpha=0.4, color='orange', label='Mean Absolute Residual', ax=ax)
        
        plt.title('Mean Residual by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Residual (kWh)')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(residual_plots_dir, 'residuals_by_hour.png'))
        plt.close()
        
        # 2. Residuals vs Temperature Bins
        # First join with temperature data
        if not weather_df.empty:
            # Create common index between residuals and weather
            eval_df = eval_df.copy()
            
            # Join weather data
            common_indices = eval_df.index.intersection(weather_df.index)
            temp_df = eval_df.loc[common_indices].copy()
            temp_df['temperature'] = weather_df.loc[common_indices, 'temperature_2m'] if 'temperature_2m' in weather_df.columns else weather_df.loc[common_indices, 'temp']
            
            # Create temperature bins (in 2°C increments)
            temp_df['temp_bin'] = pd.cut(temp_df['temperature'], bins=range(-10, 40, 2), labels=[f"{i}-{i+2}°C" for i in range(-10, 38, 2)])
            
            # Group by temperature bin
            temp_residuals = temp_df.groupby('temp_bin')['residual'].mean()
            temp_abs_residuals = temp_df.groupby('temp_bin')['abs_residual'].mean()
            
            # Only plot if we have enough data
            if len(temp_residuals) > 3:
                plt.figure(figsize=(14, 6))
                ax = temp_residuals.plot(kind='bar', alpha=0.7, color='blue', label='Mean Residual (+ = Underprediction)')
                temp_abs_residuals.plot(kind='bar', alpha=0.4, color='orange', label='Mean Absolute Residual', ax=ax)
                
                plt.title('Mean Residual by Temperature')
                plt.xlabel('Temperature Range')
                plt.ylabel('Residual (kWh)')
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.grid(axis='y', alpha=0.3)
                plt.legend()
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.savefig(os.path.join(residual_plots_dir, 'residuals_by_temperature.png'))
                plt.close()
        
        # 3. Residuals by Weekday/Weekend
        weekday_df = eval_df[~eval_df['is_weekend']]
        weekend_df = eval_df[eval_df['is_weekend']]
        
        weekday_mean = weekday_df['residual'].mean()
        weekend_mean = weekend_df['residual'].mean()
        weekday_abs_mean = weekday_df['abs_residual'].mean()
        weekend_abs_mean = weekend_df['abs_residual'].mean()
        
        plt.figure(figsize=(10, 6))
        plt.bar(['Weekday', 'Weekend'], [weekday_mean, weekend_mean], alpha=0.7, color='blue', label='Mean Residual')
        plt.bar(['Weekday', 'Weekend'], [weekday_abs_mean, weekend_abs_mean], alpha=0.4, color='orange', label='Mean Absolute Residual')
        
        plt.title('Mean Residual by Day Type')
        plt.ylabel('Residual (kWh)')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(residual_plots_dir, 'residuals_by_daytype.png'))
        plt.close()
        
        # 4. Scatter plot of actual vs predicted
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        max_val = max(y_test.max(), y_pred.max())
        min_val = min(y_test.min(), y_pred.min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title('Actual vs Predicted Consumption')
        plt.xlabel('Actual Consumption (kWh)')
        plt.ylabel('Predicted Consumption (kWh)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(residual_plots_dir, 'actual_vs_predicted.png'))
        plt.close()
        
        # 5. Residual distribution
        plt.figure(figsize=(10, 6))
        plt.hist(eval_df['residual'], bins=50, alpha=0.75)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        
        plt.title('Distribution of Residuals')
        plt.xlabel('Residual (kWh)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(residual_plots_dir, 'residual_distribution.png'))
        plt.close()
        
        logger.info(f"Residual analysis plots created in {residual_plots_dir}")
        
    except Exception as e:
        logger.error(f"Error creating residual analysis plots: {e}")

def prepare_features(consumption_df):
    """
    Prepare all features for the model
    
    Args:
        consumption_df: DataFrame with consumption data
        
    Returns:
        df: DataFrame with features
        weather_df: DataFrame with weather data for analysis
    """
    logger.info("Preparing features for model training...")
    
    try:
        # Start with consumption data
        df = consumption_df.copy()
        
        # Check for and remove cost and unit_price features to avoid data leakage
        if 'cost' in df.columns:
            logger.info("Removing 'cost' column to prevent target leakage")
            df = df.drop('cost', axis=1)
            
        if 'unit_price' in df.columns:
            logger.info("Removing 'unit_price' column to prevent target leakage")
            df = df.drop('unit_price', axis=1)
        
        # Load time features data - critical for cyclical patterns
        time_features_df = load_time_features()
        if not time_features_df.empty:
            logger.info("Using time features from CSV file")
            # Merge with time features on index (timestamp)
            df = df.join(time_features_df, how='left')
            # Log which time feature columns were merged
            time_cols = [col for col in time_features_df.columns if col in df.columns]
            logger.info(f"Merged time features: {time_cols}")
        else:
            # If time features couldn't be loaded, create basic time features
            logger.info("Creating basic time features (no cyclical features)")
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['is_weekend'] = df.index.dayofweek >= 5
            # Add season (0=Winter, 1=Spring, 2=Summer, 3=Fall)
            df['season'] = (df.index.month % 12 + 3) // 3 % 4
        
        # Load weather data - important for energy consumption
        weather_df = load_weather_data()
        if not weather_df.empty:
            logger.info("Using weather data from CSV file")
            common_indices = df.index.intersection(weather_df.index)
            if len(common_indices) > 0:
                # Only merge the most important weather features
                weather_cols = []
                
                # Temperature is highly important for energy consumption
                temp_col = 'temperature_2m' if 'temperature_2m' in weather_df.columns else 'temp'
                if temp_col in weather_df.columns:
                    weather_cols.append(temp_col)
                    
                # Humidity can affect thermal comfort and energy usage
                humidity_col = 'relative_humidity_2m' if 'relative_humidity_2m' in weather_df.columns else 'humidity'
                if humidity_col in weather_df.columns:
                    weather_cols.append(humidity_col)
                    
                # Cloud cover affects solar radiation and lighting needs
                cloud_col = 'cloud_cover' if 'cloud_cover' in weather_df.columns else 'clouds'
                if cloud_col in weather_df.columns:
                    weather_cols.append(cloud_col)
                    
                # Wind can affect building heat loss
                wind_col = 'wind_speed_100m' if 'wind_speed_100m' in weather_df.columns else 'wind_speed'
                if wind_col in weather_df.columns:
                    weather_cols.append(wind_col)
                    
                # Precipitation might affect behavior
                precip_col = 'precipitation' if 'precipitation' in weather_df.columns else 'rain'
                if precip_col in weather_df.columns:
                    weather_cols.append(precip_col)
                
                # Only merge if we have weather columns to use
                if weather_cols:
                    # Merge only the selected columns
                    df = pd.merge(df, weather_df[weather_cols], 
                                left_index=True, right_index=True, how='left')
                    logger.info(f"Merged weather features: {weather_cols}")
            else:
                logger.warning("No common timestamps between consumption data and weather data")
                
        # Load holiday data
        holidays_df = load_holidays()
        if not holidays_df.empty:
            logger.info("Using holidays data from CSV file")
            common_indices = df.index.intersection(holidays_df.index)
            if len(common_indices) > 0:
                # Use holiday flag (most important calendar feature beyond basic time)
                if 'is_holiday' in holidays_df.columns:
                    df = pd.merge(df, holidays_df[['is_holiday']], 
                                left_index=True, right_index=True, how='left')
                    logger.info("Merged holiday flag")
            else:
                logger.warning("No common timestamps between consumption data and holidays data")
                df['is_holiday'] = False
        else:
            df['is_holiday'] = False
                
        # Fill missing values - using ffill() and bfill() instead of fillna(method=...)
        df = df.ffill().bfill().fillna(0)
        
        # Note: Lag features will be created within the train/test splits to avoid data leakage
        
        logger.info(f"Feature preparation complete: {len(df)} records with {len(df.columns)} features")
        return df, weather_df  # Return weather_df for residual analysis
    
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        raise

def create_weekly_evaluation_plots(y_test, y_pred):
    """
    Create weekly evaluation plots to show model performance over different weeks
    """
    logger.info("Creating weekly evaluation plots...")
    
    try:
        # Create DataFrame with actual and predicted values
        eval_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred
        })
        
        # Group by week (using the ISO week number)
        eval_df['week'] = eval_df.index.isocalendar().week
        eval_df['year'] = eval_df.index.isocalendar().year
        
        # Create a unique identifier for each week (year + week)
        eval_df['year_week'] = eval_df['year'].astype(str) + "-" + eval_df['week'].astype(str)
        
        # Get the last 8 weeks of data instead of 4
        unique_weeks = eval_df['year_week'].unique()
        weeks_to_plot = unique_weeks[-8:] if len(unique_weeks) >= 8 else unique_weeks
        
        # Create a directory for weekly plots if it doesn't exist
        weekly_plots_dir = os.path.join(MODEL_PATH, 'weekly_plots')
        os.makedirs(weekly_plots_dir, exist_ok=True)
        
        # Create a CSV with weekly metrics
        weekly_metrics = []
        
        # Create individual plots for each week
        for week_id in weeks_to_plot:
            week_data = eval_df[eval_df['year_week'] == week_id]
            
            if len(week_data) < 24:  # Skip if less than a day of data
                logger.warning(f"Skipping week {week_id} with only {len(week_data)} data points")
                continue
                
            # Create plot
            fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
            
            # Plot actual and predicted values
            ax.plot(week_data.index, week_data['actual'], label='Actual')
            ax.plot(week_data.index, week_data['predicted'], label='Predicted', alpha=0.7)
            
            # Add residuals plot (prediction error)
            ax.fill_between(week_data.index, 
                           week_data['actual'], 
                           week_data['predicted'], 
                           color='orange', 
                           alpha=0.3, 
                           label='Error')
            
            # Format x-axis to show days and hours
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %H:%M'))
            plt.gcf().autofmt_xdate(rotation=45)
            
            # Add labels and title
            year, week_num = week_id.split("-")
            plt.title(f'Week {week_num}, {year}: Actual vs Predicted Energy Consumption')
            plt.xlabel('Date')
            plt.ylabel('Consumption (kWh)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Compute metrics for this week
            week_mae = mean_absolute_error(week_data['actual'], week_data['predicted'])
            week_rmse = np.sqrt(mean_squared_error(week_data['actual'], week_data['predicted']))
            week_r2 = r2_score(week_data['actual'], week_data['predicted'])
            week_mape = np.mean(np.abs((week_data['actual'] - week_data['predicted']) / week_data['actual'])) * 100
            
            # Store weekly metrics
            weekly_metrics.append({
                'week_id': week_id,
                'year': year,
                'week_num': week_num,
                'mae': week_mae,
                'rmse': week_rmse,
                'r2': week_r2,
                'mape': week_mape,
                'data_points': len(week_data)
            })
            
            # Add metrics as text
            plt.figtext(0.02, 0.02, 
                     f'MAE: {week_mae:.4f}, RMSE: {week_rmse:.4f}, R²: {week_r2:.4f}, MAPE: {week_mape:.2f}%', 
                     bbox=dict(facecolor='white', alpha=0.8))
            
            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(weekly_plots_dir, f'week_{week_id}.png'))
            plt.close()
            
            # Create daily pattern plot for this week
            daily_pattern_fig, daily_pattern_ax = plt.subplots(figsize=(12, 6))
            
            # Group by hour of day and calculate mean
            hourly_pattern = week_data.groupby(week_data.index.hour)
            hourly_actual = hourly_pattern['actual'].mean()
            hourly_predicted = hourly_pattern['predicted'].mean()
            
            # Plot hourly patterns
            daily_pattern_ax.plot(hourly_actual.index, hourly_actual, marker='o', label='Actual')
            daily_pattern_ax.plot(hourly_predicted.index, hourly_predicted, marker='x', label='Predicted')
            
            # Add labels and grid
            daily_pattern_ax.set_title(f'Week {week_num}, {year}: Daily Consumption Pattern')
            daily_pattern_ax.set_xlabel('Hour of Day')
            daily_pattern_ax.set_ylabel('Average Consumption (kWh)')
            daily_pattern_ax.set_xticks(range(0, 24))
            daily_pattern_ax.grid(True, alpha=0.3)
            daily_pattern_ax.legend()
            
            # Save daily pattern plot
            plt.tight_layout()
            plt.savefig(os.path.join(weekly_plots_dir, f'week_{week_id}_daily_pattern.png'))
            plt.close()
            
        # Save weekly metrics to CSV
        if weekly_metrics:
            weekly_metrics_df = pd.DataFrame(weekly_metrics)
            weekly_metrics_df.to_csv(os.path.join(weekly_plots_dir, 'weekly_metrics.csv'), index=False)
            
            # Create a metrics trend plot
            plt.figure(figsize=(14, 8))
            
            # Plot metrics over time
            plt.subplot(2, 1, 1)
            plt.plot(weekly_metrics_df['week_id'], weekly_metrics_df['mae'], marker='o', label='MAE')
            plt.plot(weekly_metrics_df['week_id'], weekly_metrics_df['rmse'], marker='s', label='RMSE')
            plt.title('Weekly Error Metrics Trend')
            plt.ylabel('Error (kWh)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xticks(rotation=45)
            
            plt.subplot(2, 1, 2)
            plt.plot(weekly_metrics_df['week_id'], weekly_metrics_df['r2'], marker='o', label='R²')
            plt.plot(weekly_metrics_df['week_id'], weekly_metrics_df['mape'], marker='s', label='MAPE (%)')
            plt.title('Weekly R² and MAPE Trend')
            plt.ylabel('Value')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(weekly_plots_dir, 'weekly_metrics_trend.png'))
            plt.close()
                
        # Create an overview plot with all weeks in subplots
        if len(weeks_to_plot) > 0:
            # Calculate subplot layout
            n_weeks = len(weeks_to_plot)
            n_cols = min(2, n_weeks)  # Maximum 2 columns
            n_rows = (n_weeks + n_cols - 1) // n_cols  # Ceiling division
            
            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5*n_rows))
            
            # Handle different dimensions of axes array
            if n_weeks == 1:
                axes = np.array([axes])
            axes = axes.flatten()
                
            for i, week_id in enumerate(weeks_to_plot):
                week_data = eval_df[eval_df['year_week'] == week_id]
                
                if len(week_data) < 12:  # Skip if very little data
                    continue
                    
                year, week_num = week_id.split("-")
                
                # Plot on the corresponding subplot
                axes[i].plot(week_data.index, week_data['actual'], label='Actual')
                axes[i].plot(week_data.index, week_data['predicted'], label='Predicted', alpha=0.7)
                
                # Add residuals plot
                axes[i].fill_between(week_data.index, 
                                    week_data['actual'], 
                                    week_data['predicted'], 
                                    color='orange', 
                                    alpha=0.2)
                
                # Format x-axis
                axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%a %H:%M'))
                
                # Add title and legend for each subplot
                axes[i].set_title(f'Week {week_num}, {year}')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
                
                # Compute and display metrics
                week_mae = mean_absolute_error(week_data['actual'], week_data['predicted'])
                week_rmse = np.sqrt(mean_squared_error(week_data['actual'], week_data['predicted']))
                week_r2 = r2_score(week_data['actual'], week_data['predicted'])
                axes[i].text(0.02, 0.85, f'MAE: {week_mae:.4f}, RMSE: {week_rmse:.4f}, R²: {week_r2:.4f}', 
                         transform=axes[i].transAxes,
                         bbox=dict(facecolor='white', alpha=0.8))
            
            # Hide unused subplots
            for j in range(i+1, len(axes)):
                axes[j].axis('off')
                
            # Overall title
            fig.suptitle('Weekly Energy Consumption Evaluation', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for suptitle
            
            # Save the overview plot
            plt.savefig(os.path.join(MODEL_PATH, 'weekly_evaluation_overview.png'))
            plt.close()
            
            logger.info(f"Weekly evaluation plots created in {weekly_plots_dir}")
            
    except Exception as e:
        logger.error(f"Error creating weekly evaluation plots: {e}")
        logger.error(traceback.format_exc())

def train_xgboost_model(X_train, y_train, X_val, y_val, config=None):
    """
    Train an XGBoost model with configurable parameters
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        config: Configuration dictionary (if None, will load from file)
        
    Returns:
        Trained XGBoost model
    """
    logger.info("Training XGBoost model...")
    
    if config is None:
        config = load_config()
    
    try:
        # Apply feature selection based on configuration
        X_train_filtered = filter_features(X_train, config)
        X_val_filtered = filter_features(X_val, config)
        
        # Apply feature weights
        X_train_weighted = apply_feature_weights(X_train_filtered, config)
        X_val_weighted = apply_feature_weights(X_val_filtered, config)
        
        # Get model parameters from config
        params = config.get('model_params', {})
        
        # Ensure minimal required parameters are set
        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 500,
            'max_depth': 7,
            'learning_rate': 0.05,
            'eval_metric': 'rmse'
        }
        
        # Update default params with config params
        for key, value in default_params.items():
            if key not in params:
                params[key] = value
        
        logger.info(f"Training XGBoost with parameters: {params}")
        logger.info(f"Using {X_train_weighted.shape[1]} features after filtering and weighting")
        
        # Train model
        model = xgb.XGBRegressor(**params)
        
        # Fixed: using eval_set only
        eval_set = [(X_val_weighted, y_val)]
        try:
            model.fit(
                X_train_weighted, y_train,
                eval_set=eval_set,
                early_stopping_rounds=30,  # More patience for early stopping
                verbose=True
            )
            logger.info(f"Best iteration: {model.best_iteration}")
        except TypeError as e:
            # Handle case where early_stopping_rounds is not supported
            logger.warning(f"Error with early stopping: {e}. Trying without early stopping.")
            model.fit(X_train_weighted, y_train, eval_set=eval_set, verbose=True)
        
        # Get feature importances
        try:
            importances = model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': X_train_weighted.columns,
                'Importance': importances,
                'Weight': [config['features'].get(f, {}).get('weight', 1.0) 
                          if f in config['features'] else 1.0 
                          for f in X_train_weighted.columns]
            }).sort_values('Importance', ascending=False)
            
            # Save feature importances
            feature_importance.to_csv(os.path.join(MODEL_PATH, 'feature_importance.csv'), index=False)
            
            # Create feature importance plot
            plt.figure(figsize=(12, 8))
            plt.barh(feature_importance['Feature'][:15], feature_importance['Importance'][:15])
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('Top 15 Feature Importances')
            plt.gca().invert_yaxis()  # Display from top to bottom
            plt.tight_layout()
            plt.savefig(os.path.join(MODEL_PATH, 'feature_importance.png'))
            plt.close()
            
            logger.info("Feature importance plot created")
        except Exception as imp_err:
            logger.warning(f"Could not generate feature importance: {imp_err}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error training XGBoost model: {e}")
        
        # Try with simpler parameters if the first attempt fails
        try:
            logger.info("Trying with simpler XGBoost configuration...")
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1
            )
            model.fit(X_train, y_train)
            return model
        except Exception as e2:
            logger.error(f"Second attempt also failed: {e2}")
            raise

def evaluate_model(model, X_test, y_test, weather_df=None):
    """
    Evaluate the model on test data with enhanced metrics
    """
    logger.info("Evaluating XGBoost model...")
    
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate basic metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Calculate additional metrics
        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Root Mean Squared Percentage Error (RMSPE)
        rmspe = np.sqrt(np.mean(np.square((y_test - y_pred) / y_test))) * 100
        
        # Correlation coefficient
        corr, _ = pearsonr(y_test, y_pred)
        
        # Print comprehensive metrics
        logger.info(f"Model performance: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
        logger.info(f"Percentage metrics: MAPE={mape:.2f}%, RMSPE={rmspe:.2f}%")
        logger.info(f"Correlation coefficient: {corr:.4f}")
        
        # Save metrics
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            'rmspe': float(rmspe),
            'correlation': float(corr),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metrics_path = os.path.join(MODEL_PATH, 'xgboost_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Create evaluation plot - limit to last 100 points for clarity
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(y_test.index[-100:], y_test[-100:], label='Actual')
            plt.plot(y_test.index[-100:], y_pred[-100:], label='Predicted', alpha=0.7)
            plt.title("Model Evaluation: Actual vs Predicted Consumption")
            plt.xlabel('Date')
            plt.ylabel('Consumption (kWh)')
            plt.legend()
            plt.savefig(os.path.join(MODEL_PATH, 'evaluation_plot.png'))
            plt.close()
        except Exception as plot_err:
            logger.warning(f"Could not create evaluation plot: {plot_err}")
        
        # Create weekly evaluation plots
        create_weekly_evaluation_plots(y_test, y_pred)
        
        # Create residual analysis plots
        analyze_residuals(y_test, y_pred, X_test, weather_df)
        
        # Create plot showing distribution of errors
        plt.figure(figsize=(10, 6))
        errors = y_test - y_pred
        plt.hist(errors, bins=50, alpha=0.75)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Distribution of Prediction Errors')
        plt.xlabel('Error (Actual - Predicted)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(MODEL_PATH, 'error_distribution.png'))
        plt.close()
        
        # Create hourly error analysis
        hourly_errors = pd.DataFrame({
            'hour': y_test.index.hour,
            'error': y_test - y_pred,
            'abs_error': np.abs(y_test - y_pred)
        })
        
        hourly_mean_error = hourly_errors.groupby('hour')['error'].mean()
        hourly_mean_abs_error = hourly_errors.groupby('hour')['abs_error'].mean()
        
        plt.figure(figsize=(12, 6))
        plt.bar(hourly_mean_error.index, hourly_mean_error, alpha=0.6, label='Mean Error')
        plt.plot(hourly_mean_abs_error.index, hourly_mean_abs_error, 'r-', marker='o', label='Mean Absolute Error')
        plt.title('Hourly Error Analysis')
        plt.xlabel('Hour of Day')
        plt.ylabel('Error')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(range(0, 24))
        plt.savefig(os.path.join(MODEL_PATH, 'hourly_error_analysis.png'))
        plt.close()
        
        return mae, rmse, r2, y_pred
    
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return None, None, None, None

def create_hourly_pattern_plot(df):
    """
    Create a plot showing average consumption patterns by hour of day
    """
    logger.info("Creating hourly pattern plot...")
    
    try:
        # Group by hour and calculate average consumption
        hourly_avg = df.groupby(df.index.hour)['consumption'].mean()
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.bar(hourly_avg.index, hourly_avg.values, alpha=0.7)
        
        # Add formatting
        plt.title('Average Energy Consumption by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Consumption (kWh)')
        plt.xticks(range(0, 24))
        plt.grid(axis='y', alpha=0.3)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_PATH, 'hourly_pattern.png'))
        plt.close()
        
        # Create separate plots for weekdays and weekends
        weekday_df = df[df.index.dayofweek < 5]
        weekend_df = df[df.index.dayofweek >= 5]
        
        weekday_hourly = weekday_df.groupby(weekday_df.index.hour)['consumption'].mean()
        weekend_hourly = weekend_df.groupby(weekend_df.index.hour)['consumption'].mean()
        
        plt.figure(figsize=(12, 6))
        plt.bar(weekday_hourly.index, weekday_hourly.values, alpha=0.7, label='Weekdays')
        plt.bar(weekend_hourly.index, weekend_hourly.values, alpha=0.5, label='Weekends')
        
        plt.title('Average Energy Consumption by Hour: Weekdays vs Weekends')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Consumption (kWh)')
        plt.xticks(range(0, 24))
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_PATH, 'weekday_weekend_pattern.png'))
        plt.close()
        
        logger.info("Hourly pattern plots created")
        
    except Exception as e:
        logger.error(f"Error creating hourly pattern plot: {e}")

def predict_future(model, df, hours=24):
    """
    Make predictions for the next X hours
    
    Args:
        model: Trained XGBoost model
        df: DataFrame with features
        hours: Number of hours to predict
        
    Returns:
        DataFrame with future predictions
    """
    logger.info(f"Predicting for next {hours} hours...")
    
    try:
        # Get the latest data point
        last_timestamp = df.index.max()
        
        # Create dataframe for next X hours
        future_idx = pd.date_range(
            start=last_timestamp + timedelta(hours=1), 
            periods=hours, 
            freq='h'
        )
        
        # Create simple forecast dataframe
        forecast_df = pd.DataFrame(index=future_idx)
        
        # Add time features - both basic and cyclical
        
        # Basic time features
        forecast_df['hour'] = forecast_df.index.hour
        forecast_df['day_of_week'] = forecast_df.index.dayofweek
        forecast_df['month'] = forecast_df.index.month
        forecast_df['is_weekend'] = forecast_df.index.dayofweek >= 5
        forecast_df['season'] = (forecast_df.index.month % 12 + 3) // 3 % 4
        
        # Add cyclical time features (crucial for capturing time patterns)
        # Hour features (24-hour cycle)
        forecast_df['hour_sin'] = np.sin(2 * np.pi * forecast_df['hour'] / 24)
        forecast_df['hour_cos'] = np.cos(2 * np.pi * forecast_df['hour'] / 24)
        
        # Day of week features (7-day cycle)
        forecast_df['day_of_week_sin'] = np.sin(2 * np.pi * forecast_df['day_of_week'] / 7)
        forecast_df['day_of_week_cos'] = np.cos(2 * np.pi * forecast_df['day_of_week'] / 7)
        
        # Month features (12-month cycle)
        forecast_df['month_sin'] = np.sin(2 * np.pi * forecast_df['month'] / 12)
        forecast_df['month_cos'] = np.cos(2 * np.pi * forecast_df['month'] / 12)
        
        # Add peak hour indicators
        morning_peak_hours = [7, 8, 9]
        evening_peak_hours = [17, 18, 19, 20]
        forecast_df['is_morning_peak'] = forecast_df.index.hour.isin(morning_peak_hours).astype(int)
        forecast_df['is_evening_peak'] = forecast_df.index.hour.isin(evening_peak_hours).astype(int)
        
        # Load holidays if available and add holiday indicator
        try:
            holidays_df = load_holidays()
            if not holidays_df.empty:
                holidays = holidays_df[holidays_df['is_holiday'] == True].index
                forecast_df['is_holiday'] = forecast_df.index.date.isin([h.date() for h in holidays]).astype(int)
            else:
                forecast_df['is_holiday'] = 0
        except Exception as e:
            logger.warning(f"Error loading holidays: {e}")
            forecast_df['is_holiday'] = 0
            
        # Add lag features from historical data
        window_data = df.tail(168)  # Use last week of data
        
        predictions = []
        
        # Define feature weights - must match the weights used in training
        feature_weights = {
            # Lag features - most predictive
            'consumption_lag_24h': 2.0,        # Previous day same hour (strongest predictor)
            'consumption_lag_1h': 1.8,         # Previous hour (very strong predictor)
            'consumption_lag_168h': 1.8,       # Previous week same hour (very strong weekly pattern)
            'consumption_rolling_mean_24h': 1.5, # Daily average (strong daily pattern)
            
            # Cyclical time features - crucial for pattern detection
            'hour_sin': 1.5,                   # Hour sine component
            'hour_cos': 1.5,                   # Hour cosine component
            'day_of_week_sin': 1.3,            # Day of week sine
            'day_of_week_cos': 1.3,            # Day of week cosine
            'month_sin': 1.2,                  # Month sine
            'month_cos': 1.2,                  # Month cosine
            
            # Weather and other features
            'temperature': 1.3,                # Temperature effect on consumption
            'is_weekend': 1.2,                 # Weekend pattern distinction
            'is_holiday': 1.2,                 # Holiday pattern distinction
            'is_morning_peak': 1.2,            # Morning peak importance
            'is_evening_peak': 1.2,            # Evening peak importance
            'season': 1.1                      # Seasonal pattern
        }
        
        # Get required feature names from the model
        try:
            # For newer XGBoost versions
            if hasattr(model, 'feature_names_in_'):
                required_cols = list(model.feature_names_in_)
            # For older XGBoost versions
            elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'feature_names'):
                required_cols = model.get_booster().feature_names
            else:
                # Hard-coded fallback features
                logger.warning("Could not determine feature names from model, using hard-coded features")
                required_cols = [
                    'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
                    'is_morning_peak', 'is_evening_peak', 'is_weekend', 'is_holiday', 'temperature',
                    'consumption_lag_1h', 'consumption_lag_24h', 'consumption_lag_168h', 
                    'consumption_rolling_mean_24h', 'consumption_rolling_mean_168h', 'season'
                ]
        except Exception as e:
            logger.warning(f"Error getting feature names: {e}. Using common features.")
            # Hard-coded fallback features
            required_cols = [
                'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
                'is_morning_peak', 'is_evening_peak', 'is_weekend', 'is_holiday', 
                'consumption_lag_1h', 'consumption_lag_24h', 'consumption_lag_168h', 
                'consumption_rolling_mean_24h', 'season'
            ]
        
        # Predict one hour at a time, updating lag features as we go
        temp_df = window_data.copy()
        
        for i, timestamp in enumerate(future_idx):
            # Create temporary row for this hour
            this_hour = pd.DataFrame(index=[timestamp])
            
            # Add time features - both basic and cyclical
            hour = timestamp.hour
            day_of_week = timestamp.dayofweek
            month = timestamp.month
            
            # Basic time features
            this_hour['hour'] = hour
            this_hour['day_of_week'] = day_of_week
            this_hour['month'] = month
            this_hour['is_weekend'] = 1 if day_of_week >= 5 else 0
            this_hour['season'] = (month % 12 + 3) // 3 % 4
            
            # Cyclical time features
            this_hour['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            this_hour['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            this_hour['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
            this_hour['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
            this_hour['month_sin'] = np.sin(2 * np.pi * month / 12)
            this_hour['month_cos'] = np.cos(2 * np.pi * month / 12)
            
            # Peak hours
            this_hour['is_morning_peak'] = 1 if hour in [7, 8, 9] else 0
            this_hour['is_evening_peak'] = 1 if hour in [17, 18, 19, 20] else 0
            
            # Holiday info
            if 'is_holiday' in forecast_df.columns:
                this_hour['is_holiday'] = forecast_df.loc[timestamp, 'is_holiday']
            else:
                this_hour['is_holiday'] = 0
            
            # Add lag features from most recent data
            for lag in [1, 24, 48, 168]:
                lag_time = timestamp - timedelta(hours=lag)
                if lag_time in temp_df.index:
                    this_hour[f'consumption_lag_{lag}h'] = temp_df.loc[lag_time, 'consumption']
                else:
                    this_hour[f'consumption_lag_{lag}h'] = temp_df['consumption'].mean()
            
            # Add rolling means
            for window in [24, 168]:
                this_hour[f'consumption_rolling_mean_{window}h'] = temp_df['consumption'].tail(window).mean()
            
            # Get weather data if available
            try:
                weather_df = load_weather_data()
                if not weather_df.empty and timestamp in weather_df.index:
                    # Try different temperature column names
                    for temp_col in ['temperature_2m', 'temp', 'temperature']:
                        if temp_col in weather_df.columns:
                            this_hour['temperature'] = weather_df.loc[timestamp, temp_col]
                            break
                else:
                    # If no weather data for this timestamp, use the most recent value
                    recent_temps = [col for col in temp_df.columns if col in ['temperature_2m', 'temp', 'temperature']]
                    if recent_temps:
                        this_hour['temperature'] = temp_df[recent_temps[0]].iloc[-1]
                    else:
                        this_hour['temperature'] = 0  # Default value
            except Exception as e:
                logger.warning(f"Error getting weather data: {e}")
                this_hour['temperature'] = 0  # Default value
            
            # Add any other features the model might need
            for col in required_cols:
                if col not in this_hour.columns:
                    # Try to find a suitable value from the data
                    if col in temp_df.columns:
                        this_hour[col] = temp_df[col].iloc[-1]  # Use last value
                    else:
                        this_hour[col] = 0  # Default to 0
            
            # Filter to only include features that the model requires
            available_required_cols = [col for col in required_cols if col in this_hour.columns]
            
            # Apply feature weights
            this_hour_weighted = this_hour.copy()
            for feature, weight in feature_weights.items():
                if feature in this_hour_weighted.columns and feature in available_required_cols:
                    this_hour_weighted[feature] = this_hour_weighted[feature] * weight
            
            # Make prediction
            try:
                X_pred = this_hour_weighted[available_required_cols]
                pred = model.predict(X_pred)[0]
            except Exception as e:
                logger.warning(f"Error during prediction: {e}. Using simple average.")
                pred = temp_df['consumption'].mean()  # Fallback to simple average
                
            predictions.append(pred)
            
            # Add prediction to temp_df for next iteration (updating lag features)
            new_row = pd.DataFrame({'consumption': pred}, index=[timestamp])
            # Copy over other columns
            for col in temp_df.columns:
                if col not in new_row and col != 'consumption':
                    if col in this_hour:
                        new_row[col] = this_hour[col].iloc[0]
                    else:
                        new_row[col] = 0
            
            temp_df = pd.concat([temp_df, new_row])
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'timestamp': future_idx,
            'predicted_consumption': predictions
        })
        results_df.set_index('timestamp', inplace=True)
        
        # Save predictions
        results_df.to_csv(os.path.join(MODEL_PATH, f'forecast_{hours}h.csv'))
        
        # Plot forecast
        try:
            plt.figure(figsize=(12, 6))
            recent_data = df['consumption'].tail(48)
            plt.plot(recent_data.index, recent_data, label='Historical')
            plt.plot(results_df.index, results_df['predicted_consumption'], 
                   label='Forecast', linestyle='--')
            plt.axvline(x=last_timestamp, color='r', linestyle='-', alpha=0.3)
            plt.title(f'Energy Consumption Forecast (Next {hours} Hours)')
            plt.xlabel('Time')
            plt.ylabel('Consumption (kWh)')
            plt.legend()
            plt.savefig(os.path.join(MODEL_PATH, f'forecast_{hours}h_plot.png'))
            plt.close()
            
            # Print prediction summary
            print(f"\nForecast Summary:")
            print(f"Average predicted consumption: {results_df['predicted_consumption'].mean():.2f} kWh")
            print(f"Min predicted consumption: {results_df['predicted_consumption'].min():.2f} kWh")
            print(f"Max predicted consumption: {results_df['predicted_consumption'].max():.2f} kWh")
            
        except Exception as plot_err:
            logger.warning(f"Could not create forecast plot: {plot_err}")
        
        logger.info(f"{hours}-hour forecast saved")
        return results_df
    
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        logger.error(traceback.format_exc())
        return None

def cli():
    """
    Command-line interface for the energy demand prediction
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Energy Demand Prediction Tool')
    
    # Set up the main action options as mutually exclusive
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--train', action='store_true', help='Train a new model')
    action_group.add_argument('--evaluate', action='store_true', help='Evaluate the existing model on test data')
    action_group.add_argument('--predict', type=int, metavar='DAYS', help='Predict future consumption for specified number of days')
    action_group.add_argument('--run', action='store_true', help='Run the full pipeline (train, evaluate, and predict)')
    
    args = parser.parse_args()
    
    if args.train:
        # Train a new model
        try:
            logger.info("Training new model...")
            model = main(mode="train")
            if model:
                logger.info("Model training completed successfully")
                print("Model trained successfully and saved")
        except Exception as e:
            logger.error(f"Error training model: {e}")
            print(f"Error: {e}")
        return
    
    if args.evaluate:
        # Evaluate the model on test data
        try:
            logger.info("Evaluating existing model...")
            model = main(mode="evaluate")
            if model:
                logger.info("Model evaluation completed successfully")
                print("Model evaluation completed. Check the results in the models directory.")
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            print(f"Error: {e}")
        return
    
    if args.predict is not None:
        # Load the saved model and make predictions for X days
        days = args.predict
        hours = days * 24  # Convert days to hours
        
        try:
            logger.info(f"Predicting energy consumption for the next {days} days ({hours} hours)...")
            
            # Check if model exists
            model_path = os.path.join(MODEL_PATH, 'xgboost_model.pkl')
            if not os.path.exists(model_path):
                print("No saved model found. Please train the model first.")
                return
            
            # Load the model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                logger.info(f"Loaded existing model from {model_path}")
                
            # Load the latest data
            consumption_df = load_consumption_data()
            if consumption_df is None or consumption_df.empty:
                print("No consumption data available.")
                return
                
            # Prepare features
            df, _ = prepare_features(consumption_df)
            if df is None or df.empty:
                print("Failed to prepare features.")
                return
                
            # Make predictions
            predictions = predict_future(model, df, hours=hours)
            if predictions is not None:
                # Save predictions with days in filename for clarity
                output_csv = os.path.join(MODEL_PATH, f'forecast_{days}d.csv')
                output_plot = os.path.join(MODEL_PATH, f'forecast_{days}d_plot.png')
                
                # Rename the files for consistency
                forecast_csv = os.path.join(MODEL_PATH, f'forecast_{hours}h.csv')
                forecast_plot = os.path.join(MODEL_PATH, f'forecast_{hours}h_plot.png')
                
                if os.path.exists(forecast_csv):
                    os.rename(forecast_csv, output_csv)
                
                if os.path.exists(forecast_plot):
                    os.rename(forecast_plot, output_plot)
                
                print(f"\nPredictions for the next {days} days saved to {output_csv}")
                print(f"Forecast plot saved to {output_plot}")
            else:
                print("Failed to generate predictions.")
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            print(f"Error: {e}")
        return
    
    if args.run:
        main()
    else:
        parser.print_help()

def main(mode="both"):
    """
    Main function to run the energy demand prediction pipeline
    
    Args:
        mode: Operation mode - "train", "evaluate", or "both"
        
    Returns:
        The trained model object
    """
    logger.info(f"Starting energy demand prediction in {mode} mode...")
    
    try:
        # Create model directory if it doesn't exist
        os.makedirs(MODEL_PATH, exist_ok=True)
        
        # Load and prepare data
        consumption_df = load_consumption_data()
        if consumption_df is None or consumption_df.empty:
            logger.error("No consumption data available. Exiting.")
            return
            
        # Prepare feature dataframe
        df, weather_df = prepare_features(consumption_df)
            
        if df is None or df.empty:
            logger.error("Failed to prepare features. Exiting.")
            return
            
        # Get target and features
        y = df['consumption']
        X = df.drop('consumption', axis=1)
        
        # Use TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Get the indices for the last fold to use as validation/test
        train_indices = []
        test_indices = []
        
        for train_idx, test_idx in tscv.split(X):
            train_indices.append(train_idx)
            test_indices.append(test_idx)
            
        # Use the last fold for testing
        train_idx = train_indices[-2]  # Second to last fold for training
        val_idx = test_indices[-2]     # Second to last fold for validation
        test_idx = test_indices[-1]    # Last fold for testing
        
        # Create train/validation/test sets
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        
        logger.info(f"Data split complete using TimeSeriesSplit - Training: {X_train.shape[0]}, " +
                   f"Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        # Calculate date ranges for data splits
        min_train_date = df.index[train_idx[0]]
        max_train_date = df.index[train_idx[-1]]
        min_val_date = df.index[val_idx[0]]
        max_val_date = df.index[val_idx[-1]]
        min_test_date = df.index[test_idx[0]]
        max_test_date = df.index[test_idx[-1]]
        
        logger.info(f"Training period: {min_train_date} - {max_train_date}")
        logger.info(f"Validation period: {min_val_date} - {max_val_date}")
        logger.info(f"Test period: {min_test_date} - {max_test_date}")
        
        # Add lag features without data leakage
        # Create a subset of the original df for our lag feature creation
        df_subset = df.copy()
        
        # Define lag periods to use (in hours)
        lag_periods = [1, 24, 48, 168]  # 1 hour, 1 day, 2 days, 1 week
        
        # Add lag features to the subset while respecting the time series split
        logger.info("Creating lag features without data leakage...")
        for lag in lag_periods:
            col_name = f'consumption_lag_{lag}h'
            df_subset[col_name] = np.nan
            
            # Shift across the whole dataset
            lag_series = df['consumption'].shift(lag)
            
            # For training: use all available data
            df_subset.loc[min_train_date:max_train_date, col_name] = lag_series.loc[min_train_date:max_train_date]
            
            # For validation: ensure we don't use future data from test period
            df_subset.loc[min_val_date:max_val_date, col_name] = lag_series.loc[min_val_date:max_val_date]
            
            # For test: ensure we don't use future data beyond the available dates
            df_subset.loc[min_test_date:max_test_date, col_name] = lag_series.loc[min_test_date:max_test_date]
        
        # Similarly for rolling means (24 hour and weekly)
        windows = [24, 168]  # 24h, 1 week
            
        for window in windows:
            col_name = f'consumption_rolling_mean_{window}h'
            df_subset[col_name] = np.nan
            
            # Calculate rolling means only from training data
            rolling_means = df.loc[min_train_date:max_train_date, 'consumption'].rolling(window=window, min_periods=1).mean()
            
            # Map those rolling means back to the appropriate dates in df_subset
            common_dates = df_subset.index.intersection(rolling_means.index)
            df_subset.loc[common_dates, col_name] = rolling_means.loc[common_dates]
                
        # Re-extract X and y from our lag-feature-added dataframe
        X_train = df_subset.loc[X_train.index].drop('consumption', axis=1)
        X_val = df_subset.loc[X_val.index].drop('consumption', axis=1)
        X_test = df_subset.loc[X_test.index].drop('consumption', axis=1)
        
        # Ensure we don't have any NaN values in the lag features
        # For validation and test sets, any NaN lag values can be imputed with the mean from training
        train_means = X_train.mean()
        
        X_train = X_train.fillna(train_means)
        X_val = X_val.fillna(train_means)
        X_test = X_test.fillna(train_means)
        
        # =========================================================================
        # HARDCODED FEATURE SELECTION - Only use the features we want
        # =========================================================================
        selected_features = [
            # Cyclical time features (most important for capturing patterns)
            'hour_sin', 'hour_cos',           # Hour of day (cyclical)
            'day_of_week_sin', 'day_of_week_cos', # Day of week (cyclical)
            'month_sin', 'month_cos',         # Month of year (cyclical)
            
            # Binary time features
            'is_morning_peak', 'is_evening_peak', # Peak hour indicators 
            'is_weekend',                      # Weekend indicator
            'is_holiday',                      # Holiday indicator
            
            # Weather features
            'temperature',                     # Temperature (ambient)
            
            # Lag features (consumption history)
            'consumption_lag_1h',              # Previous hour
            'consumption_lag_24h',             # Same hour yesterday
            'consumption_lag_168h',            # Same hour last week
            'consumption_rolling_mean_24h',    # Average over last 24 hours
            'consumption_rolling_mean_168h',   # Average over last week
            
            # Season
            'season'                           # Season indicator
        ]
        
        # Filter to only include features that actually exist in our dataset
        available_features = [f for f in selected_features if f in X_train.columns]
        logger.info(f"Selected features that exist in dataset: {available_features}")
        
        # Filter our datasets to only include selected features
        X_train_filtered = X_train[available_features].copy()
        X_val_filtered = X_val[available_features].copy()
        X_test_filtered = X_test[available_features].copy()
        
        # =========================================================================
        # HARDCODED FEATURE WEIGHTS - Weight the most important features
        # =========================================================================
        X_train_weighted = X_train_filtered.copy()
        X_val_weighted = X_val_filtered.copy()
        X_test_weighted = X_test_filtered.copy()
        
        # Define feature weights - increase weights for most predictive features
        feature_weights = {
            # Lag features - most predictive
            'consumption_lag_24h': 2.0,        # Previous day same hour (strongest predictor)
            'consumption_lag_1h': 1.8,         # Previous hour (very strong predictor)
            'consumption_lag_168h': 1.8,       # Previous week same hour (very strong weekly pattern)
            'consumption_rolling_mean_24h': 1.5, # Daily average (strong daily pattern)
            
            # Cyclical time features - crucial for pattern detection
            'hour_sin': 1.5,                   # Hour sine component
            'hour_cos': 1.5,                   # Hour cosine component
            'day_of_week_sin': 1.3,            # Day of week sine
            'day_of_week_cos': 1.3,            # Day of week cosine
            'month_sin': 1.2,                  # Month sine
            'month_cos': 1.2,                  # Month cosine
            
            # Weather and other features
            'temperature': 1.3,                # Temperature effect on consumption
            'is_weekend': 1.2,                 # Weekend pattern distinction
            'is_holiday': 1.2,                 # Holiday pattern distinction
            'is_morning_peak': 1.2,            # Morning peak importance
            'is_evening_peak': 1.2,            # Evening peak importance
            'season': 1.1                      # Seasonal pattern
        }
        
        # Apply weights to features
        for feature, weight in feature_weights.items():
            if feature in X_train_weighted.columns:
                logger.info(f"Applying weight {weight} to feature {feature}")
                X_train_weighted[feature] = X_train_weighted[feature] * weight
                X_val_weighted[feature] = X_val_weighted[feature] * weight
                X_test_weighted[feature] = X_test_weighted[feature] * weight
        
        # Get or load the model based on mode
        model = None
        model_file = os.path.join(MODEL_PATH, 'xgboost_model.pkl')
        
        if mode in ["train", "both"]:
            # =========================================================================
            # HARDCODED MODEL PARAMETERS - Optimized for energy prediction
            # =========================================================================
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': 500,
                'max_depth': 6,                  # Reduced from 7 to prevent overfitting
                'learning_rate': 0.03,           # Reduced from 0.05 for better generalization
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'reg_alpha': 0.5,                # Increased L1 regularization
                'reg_lambda': 1.0,
                'eval_metric': 'rmse',
                'early_stopping_rounds': 50      # Increased to prevent early stopping on noise
            }
            
            # Train the model
            logger.info(f"Training XGBoost with parameters: {params}")
            
            model = xgb.XGBRegressor(**params)
            eval_set = [(X_val_weighted, y_val)]
            
            # Train with early stopping using validation data
            model.fit(
                X_train_weighted, y_train,
                eval_set=eval_set,
                verbose=True  # Print progress
            )
            
            # Save the model to disk
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model saved to {model_file}")
            
            # Evaluate on training data
            train_pred = model.predict(X_train_weighted)
            train_mae = mean_absolute_error(y_train, train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            train_r2 = r2_score(y_train, train_pred)
            
            logger.info(f"Training set metrics:")
            logger.info(f"  MAE: {train_mae:.4f}")
            logger.info(f"  RMSE: {train_rmse:.4f}")
            logger.info(f"  R²: {train_r2:.4f}")
            
            # Evaluate on validation data
            val_pred = model.predict(X_val_weighted)
            val_mae = mean_absolute_error(y_val, val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_r2 = r2_score(y_val, val_pred)
            
            logger.info(f"Validation set metrics:")
            logger.info(f"  MAE: {val_mae:.4f}")
            logger.info(f"  RMSE: {val_rmse:.4f}")
            logger.info(f"  R²: {val_r2:.4f}")
        
        elif mode in ["evaluate"]:
            # Load existing model
            try:
                if os.path.exists(model_file):
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    logger.info(f"Loaded existing model from {model_file}")
                else:
                    logger.error("No existing model found. Please train the model first.")
                    return None
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return None
                
        if model is None and mode != "train":
            logger.error("No model available for evaluation or prediction")
            return None
            
        # Evaluate on test data if mode is evaluate or both
        if mode in ["evaluate", "both"] and model is not None:
            # Create a timestamped directory for this evaluation run
            run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_dir = os.path.join(MODEL_PATH, f'run_{run_timestamp}')
            os.makedirs(run_dir, exist_ok=True)
            
            # Evaluate on test data
            test_pred = model.predict(X_test_weighted)
            test_mae = mean_absolute_error(y_test, test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            test_r2 = r2_score(y_test, test_pred)
            
            logger.info(f"Test set metrics:")
            logger.info(f"  MAE: {test_mae:.4f}")
            logger.info(f"  RMSE: {test_rmse:.4f}")
            logger.info(f"  R²: {test_r2:.4f}")
            
            # Calculate Pearson correlation coefficient
            test_corr, _ = pearsonr(y_test, test_pred)
            logger.info(f"Test set Pearson correlation: {test_corr:.4f}")
            
            # Save evaluation metrics to file
            metrics = {
                'timestamp': run_timestamp,
                'test_mae': float(test_mae),
                'test_rmse': float(test_rmse),
                'test_r2': float(test_r2),
                'test_corr': float(test_corr),
                'num_features': len(available_features),
                'features': available_features
            }
            
            with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=4)
            
            # Plot actual vs predicted values
            plot_actual_vs_predicted(y_test, test_pred, 
                                  title="Test Set: Actual vs Predicted Energy Consumption",
                                  filename=os.path.join(run_dir, 'actual_vs_predicted.png'))
            
            # Plot feature importance
            if hasattr(model, 'feature_importances_'):
                plot_feature_importance(model, X_train_weighted.columns, 
                                      filename=os.path.join(run_dir, 'feature_importance.png'))
            
            # Plot time series of predictions
            test_dates = df.iloc[test_idx].index
            plot_time_series(test_dates, y_test, test_pred, 
                          title="Test Set: Energy Consumption Over Time",
                          filename=os.path.join(run_dir, 'time_series.png'))
            
            # Create weekly plots for better visualization
            create_weekly_plots(test_dates, y_test, test_pred, output_dir=os.path.join(run_dir, 'weekly_plots'))
            
            # Create residual plots
            if weather_df is not None and not weather_df.empty:
                common_idx = df.iloc[test_idx].index.intersection(weather_df.index)
                if len(common_idx) > 0:
                    temp_col = next((col for col in ['temperature_2m', 'temp', 'temperature'] 
                                  if col in weather_df.columns), None)
                    if temp_col:
                        plot_residuals_vs_feature(test_dates, y_test, test_pred, 
                                             weather_df.loc[common_idx, temp_col],
                                             feature_name="Temperature", 
                                             filename=os.path.join(run_dir, 'residuals_vs_temperature.png'))
            
            # Save predictions to CSV
            test_results = pd.DataFrame({
                'timestamp': test_dates,
                'actual': y_test,
                'predicted': test_pred,
                'residual': y_test - test_pred
            })
            test_results.to_csv(os.path.join(run_dir, 'test_predictions.csv'), index=False)
            
            logger.info(f"Evaluation results and plots saved to {run_dir}")
        
        # Make future predictions if mode is both
        if mode == "both" and model is not None:
            future_predictions = predict_future(model, df)
            if future_predictions is not None:
                logger.info("Future predictions generated successfully")
                
        return model
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.error(traceback.format_exc())
        return None

def plot_feature_importance(model, feature_names, filename=None):
    """
    Plot feature importance from the XGBoost model
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        filename: Where to save the plot (optional)
    """
    logger.info("Creating feature importance plot...")
    
    try:
        # Get feature importances
        importances = model.feature_importances_
        
        # Create a DataFrame for sorting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()  # Display from top to bottom
        plt.tight_layout()
        
        # Save if filename is provided
        if filename:
            plt.savefig(filename)
            logger.info(f"Feature importance plot saved to {filename}")
        
        plt.close()
        
        # Return the importance dataframe for further analysis
        return importance_df
        
    except Exception as e:
        logger.error(f"Error creating feature importance plot: {e}")
        logger.error(traceback.format_exc())
        return None


def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted", filename=None):
    """
    Create a scatter plot of actual vs predicted values
    
    Args:
        y_true: Array-like of true values
        y_pred: Array-like of predicted values
        title: Plot title
        filename: Where to save the plot (optional)
    """
    logger.info("Creating actual vs predicted scatter plot...")
    
    try:
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Labels and title
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Add metrics to the plot
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        plt.figtext(0.15, 0.85, 
                 f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}', 
                 bbox=dict(facecolor='white', alpha=0.8))
        
        # Save if filename is provided
        if filename:
            plt.savefig(filename)
            logger.info(f"Actual vs predicted plot saved to {filename}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating actual vs predicted plot: {e}")
        logger.error(traceback.format_exc())


def plot_time_series(dates, y_true, y_pred, title="Energy Consumption Over Time", filename=None):
    """
    Create a time series plot showing actual and predicted values over time
    
    Args:
        dates: Array-like of datetime indices
        y_true: Array-like of true values
        y_pred: Array-like of predicted values
        title: Plot title
        filename: Where to save the plot (optional)
    """
    logger.info("Creating time series plot...")
    
    try:
        plt.figure(figsize=(14, 8))
        
        # Plot actual and predicted values
        plt.plot(dates, y_true, 'b-', label='Actual', linewidth=1.5)
        plt.plot(dates, y_pred, 'r-', label='Predicted', linewidth=1.5)
        
        # Fill the area between curves to highlight errors
        plt.fill_between(dates, y_true, y_pred, color='gray', alpha=0.3, label='Error')
        
        # Set labels and title
        plt.xlabel('Date')
        plt.ylabel('Energy Consumption (kWh)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis to show dates properly
        plt.gcf().autofmt_xdate(rotation=45)
        plt.tight_layout()
        
        # Save if filename is provided
        if filename:
            plt.savefig(filename)
            logger.info(f"Time series plot saved to {filename}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating time series plot: {e}")
        logger.error(traceback.format_exc())


def create_weekly_plots(dates, y_true, y_pred, output_dir=None):
    """
    Create weekly plots to better visualize the model's performance
    
    Args:
        dates: Array-like of datetime indices
        y_true: Array-like of true values
        y_pred: Array-like of predicted values
        output_dir: Directory to save the plots
    """
    logger.info("Creating weekly plots...")
    
    try:
        # Create output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Create dataframe with dates, actual, and predicted values
        df = pd.DataFrame({
            'actual': y_true,
            'predicted': y_pred
        }, index=dates)
        
        # Add year and week information
        df['year'] = df.index.isocalendar().year
        df['week'] = df.index.isocalendar().week
        df['year_week'] = df['year'].astype(str) + '-' + df['week'].astype(str)
        
        # Get unique weeks for plotting
        unique_weeks = df['year_week'].unique()
        
        # Create weekly metrics storage
        weekly_metrics = []
        
        # Plot each week separately
        for week_id in unique_weeks:
            week_data = df[df['year_week'] == week_id]
            
            # Skip if less than 24 hours of data
            if len(week_data) < 24:
                logger.info(f"Skipping week {week_id} with only {len(week_data)} hours of data")
                continue
            
            # Extract year and week number
            year, week_num = week_id.split('-')
            
            # Calculate metrics for this week
            mae = mean_absolute_error(week_data['actual'], week_data['predicted'])
            rmse = np.sqrt(mean_squared_error(week_data['actual'], week_data['predicted']))
            r2 = r2_score(week_data['actual'], week_data['predicted'])
            
            # Create time series plot for the week
            plt.figure(figsize=(12, 6))
            plt.plot(week_data.index, week_data['actual'], 'b-', label='Actual')
            plt.plot(week_data.index, week_data['predicted'], 'r-', label='Predicted')
            plt.fill_between(week_data.index, week_data['actual'], week_data['predicted'], 
                           color='gray', alpha=0.3)
            
            plt.title(f'Week {week_num}, {year}: Energy Consumption')
            plt.xlabel('Date')
            plt.ylabel('Energy Consumption (kWh)')
            plt.figtext(0.15, 0.85, 
                      f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}', 
                      bbox=dict(facecolor='white', alpha=0.8))
            
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.gcf().autofmt_xdate(rotation=45)
            plt.tight_layout()
            
            # Save the plot
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'week_{week_id}.png'))
            
            plt.close()
            
            # Create daily pattern plot for this week
            plt.figure(figsize=(10, 6))
            
            # Group by hour and calculate average
            hourly_actual = week_data.groupby(week_data.index.hour)['actual'].mean()
            hourly_predicted = week_data.groupby(week_data.index.hour)['predicted'].mean()
            
            # Plot hourly patterns
            plt.plot(hourly_actual.index, hourly_actual, 'bo-', label='Actual')
            plt.plot(hourly_predicted.index, hourly_predicted, 'ro-', label='Predicted')
            
            plt.title(f'Week {week_num}, {year}: Average Daily Pattern')
            plt.xlabel('Hour of Day')
            plt.ylabel('Average Energy Consumption (kWh)')
            plt.xticks(range(0, 24, 2))
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            # Save the daily pattern plot
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'week_{week_id}_daily_pattern.png'))
            
            plt.close()
            
            # Store weekly metrics
            weekly_metrics.append({
                'year_week': week_id,
                'year': year,
                'week': week_num,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'data_points': len(week_data)
            })
        
        # Create and save weekly metrics summary
        if weekly_metrics and output_dir:
            metrics_df = pd.DataFrame(weekly_metrics)
            metrics_df.to_csv(os.path.join(output_dir, 'weekly_metrics.csv'), index=False)
            
            # Create a metrics trend plot
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.plot(metrics_df['year_week'], metrics_df['mae'], 'bo-', label='MAE')
            plt.plot(metrics_df['year_week'], metrics_df['rmse'], 'ro-', label='RMSE')
            plt.title('Weekly Error Metrics')
            plt.ylabel('Error (kWh)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xticks(rotation=45)
            
            plt.subplot(2, 1, 2)
            plt.plot(metrics_df['year_week'], metrics_df['r2'], 'go-', label='R²')
            plt.title('Weekly R²')
            plt.ylabel('R²')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, 'weekly_metrics_trend.png'))
            plt.close()
        
        logger.info(f"Weekly plots created in {output_dir if output_dir else 'current directory'}")
        
    except Exception as e:
        logger.error(f"Error creating weekly plots: {e}")
        logger.error(traceback.format_exc())


def plot_residuals_vs_feature(dates, y_true, y_pred, feature_values, 
                            feature_name="Feature", filename=None):
    """
    Create plots showing how residuals vary with a feature value
    
    Args:
        dates: Array-like of datetime indices
        y_true: Array-like of true values
        y_pred: Array-like of predicted values
        feature_values: Array-like of feature values
        feature_name: Name of the feature
        filename: Where to save the plot (optional)
    """
    logger.info(f"Creating residuals vs {feature_name} plot...")
    
    try:
        # Create a dataframe with all the data
        df = pd.DataFrame({
            'actual': y_true,
            'predicted': y_pred,
            'residual': y_true - y_pred,
            'feature': feature_values
        }, index=dates)
        
        # Create the residual vs feature scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(df['feature'], df['residual'], alpha=0.5)
        
        # Add a trend line
        z = np.polyfit(df['feature'], df['residual'], 1)
        p = np.poly1d(z)
        plt.plot(sorted(df['feature']), p(sorted(df['feature'])), 'r--', 
               label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
        
        # Add a zero line
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Labels and title
        plt.xlabel(feature_name)
        plt.ylabel('Residual (Actual - Predicted)')
        plt.title(f'Residuals vs {feature_name}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add correlation between feature and residuals to the plot
        corr, p_value = pearsonr(df['feature'], df['residual'])
        plt.figtext(0.15, 0.85, 
                 f'Correlation: {corr:.4f}\nP-value: {p_value:.4f}', 
                 bbox=dict(facecolor='white', alpha=0.8))
        
        # Save if filename is provided
        if filename:
            plt.savefig(filename)
            logger.info(f"Residuals vs {feature_name} plot saved to {filename}")
        
        plt.close()
        
        # Create a boxplot of residuals by feature bins
        # Bin the feature values
        if len(df['feature'].unique()) > 10:
            bins = 10
            df['feature_bin'] = pd.cut(df['feature'], bins=bins)
            
            plt.figure(figsize=(12, 6))
            df.boxplot(column='residual', by='feature_bin', grid=True, figsize=(12, 6))
            
            plt.title(f'Distribution of Residuals by {feature_name} Bins')
            plt.suptitle('')  # Remove the auto-generated title
            plt.xlabel(feature_name)
            plt.ylabel('Residual (Actual - Predicted)')
            
            # Save if filename is provided
            if filename:
                bin_filename = filename.replace('.png', '_binned.png')
                plt.savefig(bin_filename)
                logger.info(f"Binned residuals plot saved to {bin_filename}")
            
            plt.close()
        
    except Exception as e:
        logger.error(f"Error creating residuals vs feature plot: {e}")
        logger.error(traceback.format_exc())

# Run main() function when the script is executed directly
if __name__ == "__main__":
    cli()  # Call the CLI function instead of main