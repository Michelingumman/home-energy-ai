import numpy as np
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import joblib
from datetime import datetime, timedelta
from pathlib import Path
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.stats as stats
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import the feature config
from src.predictions.prices.gather_data import FeatureConfig

class PriceModelEvaluator:
    def __init__(self):
        """Initialize the evaluator with model paths and data"""
        # Set up paths
        self.project_root = Path(__file__).resolve().parents[3]
        self.models_dir = Path(__file__).resolve().parent / "models"
        self.eval_dir = self.models_dir / "evaluation"
        self.saved_dir = self.eval_dir / "saved"
        self.test_data_dir = self.eval_dir / "test_data"
        self.results_dir = self.eval_dir / "results"
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive evaluation directory
        self.comprehensive_dir = self.results_dir / "comprehensive_evaluation"
        self.comprehensive_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize feature config
        self.feature_config = FeatureConfig()
        
        # Load model, scalers, and test data
        self.load_model_and_data()
        
        # Initialize dataframes to store full predictions
        self.predictions_df = None
        self.actual_df = None
        
        # Colors for consistent plotting
        self.colors = {
            'actual': '#1f77b4',  # blue
            'predicted': '#ff7f0e',  # orange
            'error': '#d62728',  # red
            'grid': '#7f7f7f',   # gray
            'highlight': '#2ca02c'  # green
        }
    
    def load_model_and_data(self):
        """Load the evaluation model, scalers, and test data"""
        try:
            # Load model and scalers
            self.model = tf.keras.models.load_model(self.saved_dir / "price_model_evaluation.keras")
            self.price_scaler = joblib.load(self.saved_dir / "price_scaler.save")
            self.grid_scaler = joblib.load(self.saved_dir / "grid_scaler.save")
            
            # Load test data
            self.X_test = np.load(self.test_data_dir / "X_test.npy")
            self.y_test = np.load(self.test_data_dir / "y_test.npy")
            self.test_timestamps = pd.to_datetime(
                np.load(self.test_data_dir / "test_timestamps.npy", allow_pickle=True)
            )
            
            # Load validation data 
            self.X_val = np.load(self.test_data_dir / "X_val.npy")
            self.y_val = np.load(self.test_data_dir / "y_val.npy")
            self.val_timestamps = pd.to_datetime(
                np.load(self.test_data_dir / "val_timestamps.npy", allow_pickle=True)
            )
            
            logging.info("Successfully loaded evaluation model and data")
            logging.info(f"Test data shape: {self.X_test.shape}")
            logging.info(f"Validation data shape: {self.X_val.shape}")
            
            # Also load price data for additional context
            self.price_data = pd.read_csv(
                self.project_root / "data/processed/SE3prices.csv",
                parse_dates=['HourSE'],
                index_col='HourSE'
            )
        except Exception as e:
            logging.error(f"Error loading model or data: {str(e)}")
            raise
    
    def process_all_predictions(self):
        """Process predictions and create combined dataframe"""
        logging.info("Generating predictions for all test and validation data...")
        
        # Predict on test data
        test_pred = self.model.predict(self.X_test)
        val_pred = self.model.predict(self.X_val)
        
        # Create results dataframe for test data
        test_actual = self.inverse_transform_prices(self.y_test)
        test_predicted = self.inverse_transform_prices(test_pred)
        
        # Create results dataframe for validation data
        val_actual = self.inverse_transform_prices(self.y_val)
        val_predicted = self.inverse_transform_prices(val_pred)
        
        # Combine test and validation results into one dataframe
        test_timestamps = pd.to_datetime(np.load(self.test_data_dir / 'test_timestamps.npy', allow_pickle=True))
        val_timestamps = pd.to_datetime(np.load(self.test_data_dir / 'val_timestamps.npy', allow_pickle=True))
        
        # Create individual datasets
        test_dfs = []
        val_dfs = []
        
        # For each prediction window
        for i in range(len(test_timestamps)):
            # Create dates for next 24 hours
            dates = [test_timestamps[i] + pd.Timedelta(hours=h) for h in range(24)]
            df = pd.DataFrame({
                'actual': test_actual[i],
                'predicted': test_predicted[i],
                'error': test_actual[i] - test_predicted[i],
                'abs_error': abs(test_actual[i] - test_predicted[i]),
                'dataset': 'test'  # Mark as test data
            }, index=dates)
            test_dfs.append(df)
        
        for i in range(len(val_timestamps)):
            dates = [val_timestamps[i] + pd.Timedelta(hours=h) for h in range(24)]
            df = pd.DataFrame({
                'actual': val_actual[i],
                'predicted': val_predicted[i],
                'error': val_actual[i] - val_predicted[i],
                'abs_error': abs(val_actual[i] - val_predicted[i]),
                'dataset': 'validation'  # Mark as validation data
            }, index=dates)
            val_dfs.append(df)
        
        # Combine all dataframes
        predictions_df = pd.concat(test_dfs + val_dfs)
        
        # Make sure all numeric columns have numeric dtypes (not object)
        for col in ['actual', 'predicted', 'error', 'abs_error']:
            predictions_df[col] = pd.to_numeric(predictions_df[col])
            
        # Sort by timestamp
        predictions_df = predictions_df.sort_index()
        
        # Add hour of day for later analysis
        predictions_df['hour'] = predictions_df.index.hour
        predictions_df['day_of_week'] = predictions_df.index.dayofweek
        predictions_df['month'] = predictions_df.index.month
        
        # Remove duplicate indices if any
        predictions_df = predictions_df[~predictions_df.index.duplicated(keep='first')]
        
        # Store the processed dataframe
        self.predictions_df = predictions_df
        
        # Calculate overall metrics
        self.metrics = self.calculate_metrics(
            predictions_df['actual'].values,
            predictions_df['predicted'].values
        )
        
        total_records = len(predictions_df)
        logging.info(f"Generated predictions for {total_records} hourly data points")
        
        return predictions_df
    
    def inverse_transform_prices(self, y_scaled):
        """Helper to inverse transform scaled prices
        
        Works with either:
        - 1D array of price values
        - 2D array where each row contains 24 hourly predictions
        """
        # Check the shape of the input
        y_scaled = np.asarray(y_scaled)  # Ensure numpy array
        
        # Handle empty array or None
        if y_scaled is None or y_scaled.size == 0:
            return np.array([])
        
        if len(y_scaled.shape) == 1:
            # For 1D array (single sequence of prices)
            dummy = np.zeros((len(y_scaled), len(self.price_scaler.scale_)))
            dummy[:, 0] = y_scaled
            return self.price_scaler.inverse_transform(dummy)[:, 0]
        
        elif len(y_scaled.shape) == 2:
            # For 2D array (multiple sequences of hourly predictions)
            num_sequences, seq_length = y_scaled.shape
            
            # Flatten the array to transform all values at once
            flattened = y_scaled.reshape(-1)
            
            # Create dummy array for inverse transform
            dummy = np.zeros((len(flattened), len(self.price_scaler.scale_)))
            dummy[:, 0] = flattened
            
            # Inverse transform
            inverse_flat = self.price_scaler.inverse_transform(dummy)[:, 0]
            
            # Reshape back to original structure
            return inverse_flat.reshape(num_sequences, seq_length)
        
        else:
            raise ValueError(f"Unexpected shape for y_scaled: {y_scaled.shape}")
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive error metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Calculate MAPE safely (avoiding division by zero)
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            metrics['mape'] = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / 
                                           y_true[non_zero_mask])) * 100
        else:
            metrics['mape'] = np.nan
        
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Additional metrics
        metrics['bias'] = np.mean(y_pred - y_true)
        metrics['std_error'] = np.std(y_pred - y_true)
        metrics['max_error'] = np.max(np.abs(y_pred - y_true))
        
        # Correlation
        metrics['correlation'] = np.corrcoef(y_true, y_pred)[0, 1]
        
        return metrics
    
    def visualize_predictions_by_timeframe(self):
        """Create visualizations showing predictions at different time scales
        
        Creates three separate figures:
        1. Time Period Analysis - Shows overview, yearly, monthly, and weekly patterns
        2. Day Analysis - Shows detailed day comparisons for different price scenarios
        3. Error Analysis - Shows error distributions, patterns, and correlations
        """
        if self.predictions_df is None:
            self.process_all_predictions()
            
        plt.rcParams.update({'font.size': 10})
        
        # Create three separate figures instead of one large cramped figure
        self.create_time_period_figure()
        self.create_day_analysis_figure()
        self.create_error_analysis_figure()
        
        # Also create a separate metrics summary
        self._create_metrics_summary()
    
    def create_time_period_figure(self):
        """Create figure showing time period analysis at different scales"""
        # Figure 1: Time Period Analysis (overview, yearly, monthly, weekly)
        fig, axes = plt.subplots(3, 2, figsize=(16, 18), 
                              gridspec_kw={'height_ratios': [1, 1, 1],
                                         'width_ratios': [1, 1]})
        
        # 1. Full test period overview (top row, spans both columns)
        ax_full = plt.subplot2grid((3, 2), (0, 0), colspan=2, fig=fig)
        self._plot_full_period(ax_full)
        
        # 2. Year comparison (middle row, left)
        ax_year = axes[1, 0]
        self._plot_yearly_comparison(ax_year)
        
        # 3. Monthly comparison (middle row, right)
        ax_month = axes[1, 1]
        self._plot_monthly_comparison(ax_month)
        
        # 4. Week detail (bottom row, spans both columns)
        ax_week = plt.subplot2grid((3, 2), (2, 0), colspan=2, fig=fig)
        self._plot_week_detail(ax_week)
        
        plt.tight_layout()
        plt.savefig(self.comprehensive_dir / "time_period_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_day_analysis_figure(self):
        """Create figure showing detailed day analysis"""
        # Figure 2: Day Analysis (detailed comparison of different day types)
        # Create a slightly larger figure for better readability
        fig, ax_day = plt.subplots(figsize=(14, 12))
        self._plot_day_comparison(ax_day)
        
        plt.tight_layout()
        plt.savefig(self.comprehensive_dir / "day_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_error_analysis_figure(self):
        """Create figure showing error analysis from different perspectives"""
        # Figure 3: Error Analysis (distribution, hourly patterns, residuals, scatter)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Error distribution (top left)
        ax_error_dist = axes[0, 0]
        self._plot_error_distribution(ax_error_dist)
        
        # 2. Error by hour (top right)
        ax_error_hour = axes[0, 1]
        self._plot_error_by_hour(ax_error_hour)
        
        # 3. Residual plot (bottom left)
        ax_residual = axes[1, 0]
        self._plot_residuals(ax_residual)
        
        # 4. Actual vs Predicted scatter (bottom right)
        ax_scatter = axes[1, 1]
        self._plot_scatter(ax_scatter)
        
        plt.tight_layout()
        plt.savefig(self.comprehensive_dir / "error_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_full_period(self, ax):
        """Plot the full test period overview"""
        df = self.predictions_df
        
        # Group by day to reduce visual noise
        # Only include numeric columns when resampling to avoid 'validationvalidation' error
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Check if we have data first
        if len(df) == 0 or len(numeric_cols) == 0:
            ax.text(0.5, 0.5, "Insufficient data for overview", 
                   horizontalalignment='center', verticalalignment='center', 
                   transform=ax.transAxes, fontsize=12)
            return ax
        
        daily = df[numeric_cols].resample('D').mean()
        
        # Check if we have data after resampling
        if len(daily) == 0:
            ax.text(0.5, 0.5, "Insufficient data for overview", 
                   horizontalalignment='center', verticalalignment='center', 
                   transform=ax.transAxes, fontsize=12)
            return ax
        
        ax.plot(daily.index, daily['actual'], color=self.colors['actual'], label='Actual')
        ax.plot(daily.index, daily['predicted'], color=self.colors['predicted'], label='Predicted')
        
        # Format the plot
        ax.set_title('Full Period Overview (Daily Average)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (öre/kWh)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add metrics annotation
        metrics = self.calculate_metrics(df['actual'], df['predicted'])
        metrics_text = f"RMSE: {metrics['rmse']:.2f} öre/kWh\nMAE: {metrics['mae']:.2f} öre/kWh\nMAPE: {metrics['mape']:.2f}%\nR²: {metrics['r2']:.3f}"
        ax.annotate(metrics_text, xy=(0.02, 0.85), xycoords='axes fraction', 
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        return ax
    
    def _plot_yearly_comparison(self, ax):
        """Plot yearly metrics comparison using bar charts"""
        df = self.predictions_df
        
        # Add year column for grouping
        df['year'] = df.index.year
        years = sorted(df['year'].unique())
        
        # Calculate metrics for each year
        yearly_metrics = []
        for year in years:
            year_data = df[df['year'] == year]
            metrics = self.calculate_metrics(year_data['actual'], year_data['predicted'])
            yearly_metrics.append(metrics)
        
        # Extract specific metrics for plotting
        maes = [m['mae'] for m in yearly_metrics]
        rmses = [m['rmse'] for m in yearly_metrics]
        mapes = [m['mape'] for m in yearly_metrics]
        r2s = [m['r2'] for m in yearly_metrics]
        
        # Set up bar positions
        x = np.arange(len(years))
        width = 0.35
        
        # Plot MAE and RMSE
        ax.bar(x - width/2, maes, width, label='MAE', color=self.colors['actual'], alpha=0.7)
        ax.bar(x + width/2, rmses, width, label='RMSE', color=self.colors['predicted'], alpha=0.7)
        
        # Add a second y-axis for R² values
        ax2 = ax.twinx()
        ax2.plot(x, r2s, 'o-', color='forestgreen', label='R²')
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('R² Score', color='forestgreen')
        ax2.tick_params(axis='y', labelcolor='forestgreen')
        
        # Format the plot
        ax.set_title('Yearly Performance Metrics')
        ax.set_xlabel('Year')
        ax.set_ylabel('Error (öre/kWh)')
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add a text box with MAPE values
        mape_text = "MAPE by Year:\n" + "\n".join([f"{y}: {m:.2f}%" for y, m in zip(years, mapes)])
        ax.annotate(mape_text, xy=(0.02, 0.65), xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                   
        return ax
    
    def _plot_monthly_comparison(self, ax):
        """Plot monthly average prices by calendar month"""
        df = self.predictions_df
        
        # Group by month number for consistent comparison across years
        df['month'] = df.index.month
        
        # Calculate mean prices by month
        monthly_actual = df.groupby('month')['actual'].mean()
        monthly_predicted = df.groupby('month')['predicted'].mean()
        monthly_error = df.groupby('month')['abs_error'].mean()
        
        # Set up x-axis positions
        months = range(1, 13)
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Create bar chart with month numbers as x positions
        width = 0.35
        ax.bar([m-width/2 for m in months], monthly_actual, width, color=self.colors['actual'], label='Actual')
        ax.bar([m+width/2 for m in months], monthly_predicted, width, color=self.colors['predicted'], label='Predicted')
        
        # Add error line on secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(months, monthly_error, 'o-', color='red', label='MAE')
        ax2.set_ylabel('Mean Absolute Error', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Format the plot
        ax.set_title('Monthly Average Price Comparison')
        ax.set_xlabel('Month')
        ax.set_ylabel('Average Price (öre/kWh)')
        ax.set_xticks(months)
        ax.set_xticklabels(month_names)
        
        # Add dual legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax.grid(True, alpha=0.3, axis='y')
        
        return ax
    
    def _plot_week_detail(self, ax):
        """Plot a detailed view of a representative week"""
        df = self.predictions_df
        
        # Find a representative week with good data coverage
        # Prefer a recent week that includes both weekends and weekdays
        df['weekday'] = df.index.dayofweek
        df['is_weekend'] = df['weekday'] >= 5
        
        # Get a complete recent week that has weekend data
        end_date = df.index.max() - pd.Timedelta(days=7)  # Avoid partial weeks at the end
        start_date = end_date - pd.Timedelta(days=30)  # Look in the last month for a good week
        
        # Find weeks in this range that have weekend days
        subset = df[(df.index >= start_date) & (df.index <= end_date)]
        if len(subset) == 0:
            # If no data in preferred range, use the last complete week
            end_date = df.index.max() - pd.Timedelta(days=1)
            start_date = end_date - pd.Timedelta(days=6)
            week_data = df[(df.index >= start_date) & (df.index <= end_date)]
        else:
            # Properly set week_start using pandas methods
            # Safer approach using to_series() to avoid numpy/pandas conversion issues
            subset = subset.copy()  # Create a copy to avoid SettingWithCopyWarning
            subset['week_start'] = subset.index.to_series().dt.to_period('W').dt.start_time
            
            # Find week with weekend days
            found_week = False
            for week_start, week_group in subset.groupby('week_start'):
                if week_group['is_weekend'].any() and len(week_group) >= 24*5:  # At least 5 days of data
                    week_data = week_group
                    found_week = True
                    break
            
            if not found_week:
                # If no ideal week found, use the last 7 days in the subset
                week_start = subset.index.max() - pd.Timedelta(days=6)
                week_data = df[(df.index >= week_start) & (df.index <= subset.index.max())]
        
        # Plot the week
        ax.plot(week_data.index, week_data['actual'], color=self.colors['actual'], 
                label='Actual', linewidth=2)
        ax.plot(week_data.index, week_data['predicted'], color=self.colors['predicted'], 
                label='Predicted', linestyle='--', linewidth=2)
        
        # Highlight weekends with shading
        for i, row in week_data.iterrows():
            if row.get('is_weekend', i.dayofweek >= 5):  # Use computed column or calculate
                ax.axvspan(i, i + pd.Timedelta(hours=1), alpha=0.2, color='gray')
        
        # Format the plot
        ax.set_title(f'Detailed Week View ({week_data.index.min().strftime("%Y-%m-%d")} to {week_data.index.max().strftime("%Y-%m-%d")})')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (öre/kWh)')
        
        # Format x-axis to show days
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        
        # Add metrics for this week
        metrics = self.calculate_metrics(week_data['actual'], week_data['predicted'])
        metrics_text = f"Week Metrics:\nRMSE: {metrics['rmse']:.2f}\nMAE: {metrics['mae']:.2f}\nMAPE: {metrics['mape']:.2f}%"
        ax.annotate(metrics_text, xy=(0.02, 0.85), xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def _plot_day_comparison(self, ax):
        """Plot detailed analysis of different day types"""
        df = self.predictions_df
        
        # Identify complete days (having 24 hours)
        day_counts = df.groupby(df.index.date).size()
        complete_days = day_counts[day_counts == 24].index.tolist()
        
        # Filter to only include complete days
        date_mask = pd.Series([d.date() in complete_days for d in df.index], index=df.index)
        df_filtered = df[date_mask]
        
        # Try to focus on recent data (last 2 years) if enough data is available
        # This helps find representative days from more recent market conditions
        current_date = df.index.max().date()
        two_years_ago = (pd.Timestamp(current_date) - pd.DateOffset(years=2)).date()
        recent_mask = pd.Series([d.date() >= two_years_ago for d in df_filtered.index], index=df_filtered.index)
        recent_data = df_filtered[recent_mask]
        
        # Fix: Use pd.Series.nunique() instead of trying to use .unique() on a numpy array
        # Count unique dates in recent_data
        unique_recent_dates = recent_data.index.to_series().dt.date.nunique()
        
        # Use recent data if we have at least 100 days, otherwise use all data
        analysis_df = recent_data if unique_recent_dates >= 100 else df_filtered
        
        # Find representative days
        # 1. Low price day (among complete days)
        daily_avg = analysis_df.groupby(analysis_df.index.date)['actual'].mean()
        low_price_date = daily_avg.idxmin()
        
        # 2. High price day (among complete days)
        high_price_date = daily_avg.idxmax()
        
        # 3. Volatile price day (highest std dev)
        daily_std = analysis_df.groupby(analysis_df.index.date)['actual'].std()
        volatile_price_date = daily_std.idxmax()
        
        # Get day data - ensuring we have clean filtering based on dates
        # Fix: Ensure we're comparing dates properly
        low_day = analysis_df[analysis_df.index.map(lambda x: x.date() == low_price_date)].sort_index()
        high_day = analysis_df[analysis_df.index.map(lambda x: x.date() == high_price_date)].sort_index()
        volatile_day = analysis_df[analysis_df.index.map(lambda x: x.date() == volatile_price_date)].sort_index()
        
        # Check if we have data for each day type (at least 12 hours)
        # If any day has insufficient data, log warning and use fallback approach
        if len(low_day) < 12 or len(high_day) < 12 or len(volatile_day) < 12:
            logging.warning("Insufficient data for some representative days, using fallback approach")
            
            # Fallback: just pick recent days with complete data if possible
            if len(complete_days) >= 3:
                # Use the 3 most recent complete days if possible
                recent_complete = sorted(complete_days, reverse=True)[:3]
                days_data = []
                
                # Create labels based on average price
                for date in recent_complete:
                    day_data = analysis_df[analysis_df.index.map(lambda x: x.date() == date)].sort_index()
                    if len(day_data) >= 24:
                        avg_price = day_data['actual'].mean()
                        days_data.append((day_data, avg_price, date))
                
                if len(days_data) >= 3:
                    # Sort by average price
                    days_data.sort(key=lambda x: x[1])
                    
                    # Assign low, high, volatile based on price
                    low_day = days_data[0][0]
                    high_day = days_data[-1][0]
                    # Pick middle one for "moderate"
                    volatile_day = days_data[1][0]
                    
                    # Update dates for labels
                    low_price_date = days_data[0][2]
                    high_price_date = days_data[-1][2]
                    volatile_price_date = days_data[1][2]
                    
                    # Rename volatile to "Moderate" for fallback case
                    volatile_label = f"Moderate Price Day ({volatile_price_date:%Y-%m-%d})"
                else:
                    ax.text(0.5, 0.5, "Insufficient daily data for comparison", 
                          horizontalalignment='center', verticalalignment='center',
                          transform=ax.transAxes, fontsize=14)
                    return
            else:
                ax.text(0.5, 0.5, "Insufficient daily data for comparison", 
                      horizontalalignment='center', verticalalignment='center',
                      transform=ax.transAxes, fontsize=14)
                return
        else:
            volatile_label = f"Volatile Price Day ({volatile_price_date:%Y-%m-%d})"
        
        # Create the main axis for the full figure
        # This will contain subplots for each representative day
        ax.set_position([0.05, 0.05, 0.9, 0.9])  # [left, bottom, width, height]
        ax.axis('off')  # Hide the main axis
        
        # Grid positions for subplots
        positions = [
            [0.05, 0.55, 0.45, 0.35],  # Low price day  [left, bottom, width, height]
            [0.55, 0.55, 0.45, 0.35],  # High price day
            [0.05, 0.10, 0.45, 0.35],  # Volatile day
            [0.55, 0.10, 0.45, 0.35],  # Metrics summary
        ]
        
        # Create axes for each panel
        ax_low = ax.figure.add_axes(positions[0])
        ax_high = ax.figure.add_axes(positions[1])
        ax_volatile = ax.figure.add_axes(positions[2])
        ax_metrics = ax.figure.add_axes(positions[3])
        
        # Plot each day type
        days = [(low_day, ax_low, f"Low Price Day ({low_price_date:%Y-%m-%d})"),
                (high_day, ax_high, f"High Price Day ({high_price_date:%Y-%m-%d})"),
                (volatile_day, ax_volatile, volatile_label)]
        
        # Find common y limits for better comparison
        all_prices = []
        for day_data, _, _ in days:
            all_prices.extend(day_data['actual'].tolist())
            all_prices.extend(day_data['predicted'].tolist())
        
        # Handle edge case with empty data or all zeros
        if not all_prices or all(p == 0 for p in all_prices):
            ax.text(0.5, 0.5, "No valid price data for comparison", 
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes, fontsize=14)
            return
            
        y_min = max(0, min(all_prices) * 0.9)  # Prevent negative y-axis
        y_max = max(all_prices) * 1.1  # Add 10% padding
        
        # Plot each day
        metrics_data = []
        for day_data, day_ax, title in days:
            # Ensure we have 24 hours of data
            if len(day_data) < 24:
                # Handle case with less than 24 hours by resampling to hourly to fill gaps
                day_data = day_data.resample('H').mean().interpolate()
                # If still not 24 hours, create a message in the plot
                if len(day_data) < 24:
                    day_ax.text(0.5, 0.5, "Incomplete day data", 
                              horizontalalignment='center', verticalalignment='center',
                              transform=day_ax.transAxes, fontsize=12)
                    continue
            
            # Plot actual vs predicted
            hours = range(24)
            day_ax.plot(hours, day_data['actual'].values[:24], 'b-', label='Actual', linewidth=2)
            day_ax.plot(hours, day_data['predicted'].values[:24], 'r--', label='Predicted', linewidth=2)
            
            # Highlight peak hours (7-9 and 17-20)
            morning_peak = [(7, 9), (0.1, 0.3, 0.9, 0.1)]  # position and color
            evening_peak = [(17, 20), (0.1, 0.5, 0.9, 0.1)]  # position and color
            
            for (start, end), (r, g, b, a) in [morning_peak, evening_peak]:
                day_ax.axvspan(start, end, alpha=0.2, color=(r, g, b), label=f"Peak {start}-{end}h" if start == 7 else "")
            
            # Set title and labels
            day_ax.set_title(title, fontsize=12, fontweight='bold')
            day_ax.set_xlabel('Hour of Day', fontsize=10)
            day_ax.set_ylabel('Price (öre/kWh)', fontsize=10)
            
            # Set common y limits for better comparison
            day_ax.set_ylim(y_min, y_max)
            
            # Add grid for better readability
            day_ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend
            day_ax.legend(fontsize=9)
            
            # Calculate metrics for this day
            metrics = self.calculate_metrics(day_data['actual'].values[:24], day_data['predicted'].values[:24])
            day_metrics = {
                'Day Type': title.split('(')[0].strip(),
                'Date': title.split('(')[1].split(')')[0],
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'MAPE': metrics['mape'],
                'R²': metrics['r2']
            }
            metrics_data.append(day_metrics)
            
            # Add metrics annotation in the plot
            stats_text = (f"MAE: {metrics['mae']:.2f} öre/kWh\n"
                        f"RMSE: {metrics['rmse']:.2f} öre/kWh\n"
                        f"MAPE: {metrics['mape']:.1f}%")
            day_ax.annotate(stats_text, xy=(0.05, 0.05), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                        fontsize=9)
            
            # Add day description
            if "Low Price" in title:
                description = "Low price days have minimal variation\nand often occur during periods of\nexcess renewable generation and\nlow demand (e.g., holidays, warm weekends)."
            elif "High Price" in title:
                description = "High price days show elevated levels\nthroughout with morning and evening peaks.\nThese occur during high demand periods\n(cold weather, industrial activity)."
            else:  # Volatile or Moderate
                if "Volatile" in title:
                    description = "Volatile price days show significant\nintra-day price swings, often due to\nunpredictable renewable generation\nor sudden demand changes."
                else:
                    description = "Moderate price days show typical\ndaily patterns with modest peaks\nduring morning and evening hours."
                
            day_ax.annotate(description, xy=(0.95, 0.95), xycoords='axes fraction',
                        xytext=(-5, -5), textcoords='offset points',
                        ha='right', va='top',
                        bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.8),
                        fontsize=9)
        
        # Create a metrics summary table
        ax_metrics.axis('off')
        ax_metrics.set_title("Day Comparison Metrics Summary", fontsize=12, fontweight='bold')
        
        # Create table data
        metrics_df = pd.DataFrame(metrics_data)
        
        # Handle empty metrics
        if metrics_df.empty:
            ax_metrics.text(0.5, 0.5, "Insufficient metrics data for comparison", 
                          horizontalalignment='center', verticalalignment='center',
                          transform=ax_metrics.transAxes, fontsize=12)
        else:
            # Render table
            cell_text = []
            for _, row in metrics_df.iterrows():
                cell_text.append([
                    row['Day Type'],
                    row['Date'],
                    f"{row['MAE']:.2f}",
                    f"{row['RMSE']:.2f}",
                    f"{row['MAPE']:.1f}%",
                    f"{row['R²']:.3f}"
                ])
            
            columns = ['Day Type', 'Date', 'MAE\n(öre/kWh)', 'RMSE\n(öre/kWh)', 'MAPE\n(%)', 'R²']
            
            table = ax_metrics.table(
                cellText=cell_text,
                colLabels=columns,
                loc='center',
                cellLoc='center',
                colColours=['#f2f2f2']*len(columns)
            )
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)  # Make rows taller
            
            # Add explanatory text
            explanation = (
                "This visualization compares model performance across three representative day types:\n\n"
                "• Low Price Days: Typically characterized by low demand and high renewable generation\n"
                "• High Price Days: Associated with high demand, often during cold weather periods\n"
                "• Volatile Days: Days with significant price swings, challenging to predict accurately\n\n"
                "The model shows different performance characteristics for each day type, with generally\n"
                "better performance on low price days and more challenges with volatile days."
            )
            
            ax_metrics.text(0.5, 0.2, explanation, ha='center', va='center',
                          fontsize=10, linespacing=1.5,
                          bbox=dict(boxstyle="round,pad=1", fc="white", ec="gray", alpha=0.8))
        
        # Add overall figure title
        ax.figure.suptitle('Day Type Comparison: Model Performance Across Different Price Scenarios',
                       fontsize=16, fontweight='bold', y=0.98)
    
    def _plot_error_distribution(self, ax):
        """Plot distribution of prediction errors"""
        df = self.predictions_df
        
        # Create a histogram of absolute errors
        sns.histplot(df['abs_error'], bins=50, kde=True, ax=ax, color=self.colors['predicted'])
        
        # Calculate error statistics
        mean_error = df['error'].mean()
        std_error = df['error'].std()
        median_error = df['error'].median()
        
        # Format the plot
        ax.set_title('Error Distribution')
        ax.set_xlabel('Absolute Error (öre/kWh)')
        ax.set_ylabel('Frequency')
        
        # Add vertical lines for statistics
        ax.axvline(df['abs_error'].mean(), color='red', linestyle='--', 
                  label=f'Mean Abs Error: {df["abs_error"].mean():.2f}')
        
        # Add annotation with statistics
        stats_text = (f"Error Stats:\n"
                     f"Mean: {mean_error:.2f}\n"
                     f"Median: {median_error:.2f}\n"
                     f"Std Dev: {std_error:.2f}")
        ax.annotate(stats_text, xy=(0.7, 0.85), xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def _plot_error_by_hour(self, ax):
        """Plot error pattern by hour of day"""
        df = self.predictions_df
        
        # Ensure hour column exists
        if 'hour' not in df.columns:
            df['hour'] = df.index.hour
        
        # Calculate mean metrics by hour
        hourly_metrics = {}
        for hour in range(24):
            hour_data = df[df['hour'] == hour]
            if len(hour_data) > 0:  # Only calculate metrics if we have data for this hour
                metrics = self.calculate_metrics(hour_data['actual'], hour_data['predicted'])
                hourly_metrics[hour] = metrics
        
        # Check if we have metrics for all hours
        if len(hourly_metrics) != 24:
            missing_hours = set(range(24)) - set(hourly_metrics.keys())
            logging.warning(f"Missing metrics for hours: {missing_hours}")
        
        # Extract metrics for plotting
        hours = sorted(hourly_metrics.keys())
        maes = [hourly_metrics[h]['mae'] for h in hours]
        rmses = [hourly_metrics[h]['rmse'] for h in hours]
        mapes = [hourly_metrics[h]['mape'] for h in hours]
        
        # Ensure we have at least some data to plot
        if not hours:
            ax.text(0.5, 0.5, "No hourly data available for analysis", 
                   horizontalalignment='center', verticalalignment='center', 
                   transform=ax.transAxes, fontsize=12)
            return ax
            
        # Create a dual-axis plot
        ax.plot(hours, maes, 'o-', color=self.colors['actual'], label='MAE')
        ax.plot(hours, rmses, 's-', color=self.colors['predicted'], label='RMSE')
        
        # Format the plot
        ax.set_title('Error by Hour of Day')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Error (öre/kWh)')
        ax.set_xticks(range(0, 24, 2))
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add MAPE on secondary y-axis if values are reasonable
        # Replace NaN and unreasonably high values with a reasonable maximum
        filtered_mapes = []
        for m in mapes:
            if np.isnan(m) or m > 100:
                filtered_mapes.append(100)
            else:
                filtered_mapes.append(m)
                
        if filtered_mapes:  # Ensure we have values to plot
            ax2 = ax.twinx()
            ax2.plot(hours, filtered_mapes, '^-', color='green', label='MAPE')
            ax2.set_ylabel('MAPE (%)', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            
            # Add secondary legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Highlight peak hours (typically 7-9 AM and 5-8 PM)
        peak_hours_1 = [7, 8, 9]
        peak_hours_2 = [17, 18, 19, 20]
        
        for hour in peak_hours_1:
            if hour in hours:  # Only highlight if we have this hour
                ax.axvspan(hour-0.5, hour+0.5, alpha=0.2, color='orange')
        for hour in peak_hours_2:
            if hour in hours:  # Only highlight if we have this hour
                ax.axvspan(hour-0.5, hour+0.5, alpha=0.2, color='orange')
        
        return ax
    
    def _plot_residuals(self, ax):
        """Plot residuals against predicted values"""
        df = self.predictions_df
        
        ax.scatter(df['predicted'], df['error'], alpha=0.3, color=self.colors['error'])
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_title('Residual Plot')
        ax.set_xlabel('Predicted Price (öre/kWh)')
        ax.set_ylabel('Residuals (öre/kWh)')
        ax.grid(True, alpha=0.3)
    
    def _plot_scatter(self, ax):
        """Plot scatter of actual vs predicted values with perfect prediction line"""
        df = self.predictions_df
        
        ax.scatter(df['actual'], df['predicted'], alpha=0.3, color=self.colors['predicted'])
        
        # Add perfect prediction line
        min_val = min(df['actual'].min(), df['predicted'].min())
        max_val = max(df['actual'].max(), df['predicted'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_title('Predicted vs. Actual Prices')
        ax.set_xlabel('Actual Price (öre/kWh)')
        ax.set_ylabel('Predicted Price (öre/kWh)')
        ax.grid(True, alpha=0.3)
        
        # Add metrics in corner
        metrics = self.calculate_metrics(df['actual'], df['predicted'])
        metrics_text = f"R²: {metrics['r2']:.3f}\nCorrelation: {metrics['correlation']:.3f}"
        ax.annotate(metrics_text, xy=(0.05, 0.85), xycoords='axes fraction', 
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    def _create_metrics_summary(self):
        """Create a comprehensive metrics summary"""
        df = self.predictions_df
        
        # Overall metrics
        overall_metrics = self.calculate_metrics(df['actual'], df['predicted'])
        
        # Metrics by dataset (test vs validation)
        test_metrics = self.calculate_metrics(
            df[df['dataset'] == 'test']['actual'],
            df[df['dataset'] == 'test']['predicted']
        )
        
        val_metrics = self.calculate_metrics(
            df[df['dataset'] == 'validation']['actual'],
            df[df['dataset'] == 'validation']['predicted']
        )
        
        # Metrics by year
        yearly_metrics = {}
        for year in sorted(df.index.year.unique()):
            year_df = df[df.index.year == year]
            yearly_metrics[year] = self.calculate_metrics(year_df['actual'], year_df['predicted'])
        
        # Metrics by month
        monthly_metrics = {}
        for month in range(1, 13):
            month_df = df[df.index.month == month]
            if not month_df.empty:
                monthly_metrics[month] = self.calculate_metrics(month_df['actual'], month_df['predicted'])
        
        # Metrics by hour of day
        hourly_metrics = {}
        for hour in range(24):
            hour_df = df[df.index.hour == hour]
            hourly_metrics[hour] = self.calculate_metrics(hour_df['actual'], hour_df['predicted'])
        
        # Create a figure for text-based metrics summary
        fig, ax = plt.subplots(figsize=(10, 14))
        ax.axis('off')
        
        # Build the summary text
        summary_text = "# Price Prediction Model Evaluation Metrics\n\n"
        
        # Overall metrics
        summary_text += "## Overall Metrics\n"
        summary_text += f"- RMSE: {overall_metrics['rmse']:.2f} öre/kWh\n"
        summary_text += f"- MAE: {overall_metrics['mae']:.2f} öre/kWh\n"
        summary_text += f"- MAPE: {overall_metrics['mape']:.2f}%\n"
        summary_text += f"- R²: {overall_metrics['r2']:.3f}\n"
        summary_text += f"- Correlation: {overall_metrics['correlation']:.3f}\n"
        summary_text += f"- Bias: {overall_metrics['bias']:.2f} öre/kWh\n"
        summary_text += f"- Standard Error: {overall_metrics['std_error']:.2f} öre/kWh\n"
        summary_text += f"- Maximum Error: {overall_metrics['max_error']:.2f} öre/kWh\n\n"
        
        # Dataset comparison
        summary_text += "## Test vs. Validation Set Performance\n"
        summary_text += f"### Test Set\n"
        summary_text += f"- RMSE: {test_metrics['rmse']:.2f} öre/kWh\n"
        summary_text += f"- MAE: {test_metrics['mae']:.2f} öre/kWh\n"
        summary_text += f"- MAPE: {test_metrics['mape']:.2f}%\n"
        summary_text += f"- R²: {test_metrics['r2']:.3f}\n\n"
        
        summary_text += f"### Validation Set\n"
        summary_text += f"- RMSE: {val_metrics['rmse']:.2f} öre/kWh\n"
        summary_text += f"- MAE: {val_metrics['mae']:.2f} öre/kWh\n"
        summary_text += f"- MAPE: {val_metrics['mape']:.2f}%\n"
        summary_text += f"- R²: {val_metrics['r2']:.3f}\n\n"
        
        # Yearly metrics
        summary_text += "## Yearly Performance\n"
        for year, metrics in yearly_metrics.items():
            summary_text += f"### {year}\n"
            summary_text += f"- RMSE: {metrics['rmse']:.2f} öre/kWh\n"
            summary_text += f"- MAE: {metrics['mae']:.2f} öre/kWh\n"
            summary_text += f"- MAPE: {metrics['mape']:.2f}%\n"
            summary_text += f"- R²: {metrics['r2']:.3f}\n\n"
        
        # Monthly metrics (abbreviated)
        summary_text += "## Monthly Performance (MAE in öre/kWh)\n"
        summary_text += "| Month | MAE | MAPE | R² |\n"
        summary_text += "|-------|-----|------|----|\n"
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month, metrics in monthly_metrics.items():
            summary_text += f"| {month_names[month-1]} | {metrics['mae']:.2f} | {metrics['mape']:.2f}% | {metrics['r2']:.3f} |\n"
        summary_text += "\n"
        
        # Hourly metrics table (abbreviated)
        summary_text += "## Hourly Performance\n"
        summary_text += "| Hour | MAE | MAPE |\n"
        summary_text += "|------|-----|------|\n"
        for hour in range(0, 24, 2):  # Every other hour to save space
            if hour in hourly_metrics:
                metrics = hourly_metrics[hour]
                summary_text += f"| {hour:02d}:00 | {metrics['mae']:.2f} | {metrics['mape']:.2f}% |\n"
        
        # Save the metrics to a markdown file
        with open(self.comprehensive_dir / "metrics_summary.md", "w") as f:
            f.write(summary_text)
        
        # Display the summary in the figure
        ax.text(0.05, 0.95, summary_text, va='top', fontfamily='monospace')
        fig.savefig(self.comprehensive_dir / "metrics_summary.png", dpi=300, bbox_inches='tight')
        
        # Also create CSV files with all metrics
        # Overall metrics
        pd.DataFrame([overall_metrics]).to_csv(self.comprehensive_dir / "overall_metrics.csv")
        
        # Yearly metrics
        yearly_df = pd.DataFrame.from_dict(yearly_metrics, orient='index')
        yearly_df.index.name = 'year'
        yearly_df.to_csv(self.comprehensive_dir / "yearly_metrics.csv")
        
        # Monthly metrics
        monthly_df = pd.DataFrame.from_dict(monthly_metrics, orient='index')
        monthly_df.index.name = 'month'
        monthly_df.to_csv(self.comprehensive_dir / "monthly_metrics.csv")
        
        # Hourly metrics
        hourly_df = pd.DataFrame.from_dict(hourly_metrics, orient='index')
        hourly_df.index.name = 'hour'
        hourly_df.to_csv(self.comprehensive_dir / "hourly_metrics.csv")
        
        return fig
    
    def run_comprehensive_evaluation(self):
        """Run a comprehensive evaluation with all visualizations and metrics"""
        logging.info("Starting comprehensive evaluation...")
        
        # Generate all predictions
        self.process_all_predictions()
        
        # Create visualizations
        self.visualize_predictions_by_timeframe()
        
        logging.info(f"Comprehensive evaluation complete. Results saved to {self.comprehensive_dir}")
        logging.info(f"- Time period analysis: {self.comprehensive_dir / 'time_period_analysis.png'}")
        logging.info(f"- Day type comparison: {self.comprehensive_dir / 'day_comparison.png'}")
        logging.info(f"- Error analysis: {self.comprehensive_dir / 'error_analysis.png'}")
        logging.info(f"- Metrics summary: {self.comprehensive_dir / 'metrics_summary.md'}")

def main():
    """Run the evaluation without requiring command arguments"""
    logging.info("Starting price model evaluation...")
    
    try:
        # Create evaluator
        evaluator = PriceModelEvaluator()
        
        # Run comprehensive evaluation
        evaluator.run_comprehensive_evaluation()
        
        logging.info("Evaluation complete!")
        
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 