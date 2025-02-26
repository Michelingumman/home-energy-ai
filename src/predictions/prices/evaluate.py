import numpy as np
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

# Add the project root to the Python path
import sys
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.predictions.prices.gather_data import FeatureConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PriceModelEvaluator:
    def __init__(self, model_suffix="_production"):
        """Initialize the evaluator with model paths and data"""
        self.project_root = Path("C:/_Projects/home-energy-ai")
        self.model_suffix = model_suffix
        self.feature_config = FeatureConfig()
        self.load_model_and_data()
        
    def load_model_and_data(self):
        """Load the model, scalers, and test data"""
        try:
            self.model = tf.keras.models.load_model(
                self.project_root / f"models/saved/price_model{self.model_suffix}.keras"
            )
            self.price_scaler = joblib.load(
                self.project_root / f"models/saved/price_scaler{self.model_suffix}.save"
            )
            self.grid_scaler = joblib.load(
                self.project_root / f"models/saved/grid_scaler{self.model_suffix}.save"
            )
            self.X_test = np.load(self.project_root / "models/test_data/X_test.npy")
            self.y_test = np.load(self.project_root / "models/test_data/y_test.npy")
            self.timestamps = pd.to_datetime(
                np.load(self.project_root / "models/test_data/test_timestamps.npy", allow_pickle=True)
            )
            logging.info("Successfully loaded model and data")
        except Exception as e:
            logging.error(f"Error loading model or data: {str(e)}")
            raise
    
    def load_feature_data(self):
        """Load all required feature data from CSV files"""
        try:
            # Load price data
            price_df = pd.read_csv(
                self.project_root / "data/processed/SE3prices.csv",
                index_col='HourSE'
            )
            price_df.index = pd.to_datetime(price_df.index)
            
            # Load time features
            time_df = pd.read_csv(
                self.project_root / "data/processed/time_features.csv",
                index_col=0
            )
            time_df.index = pd.to_datetime(time_df.index)
            
            # Load holiday data
            holiday_df = pd.read_csv(
                self.project_root / "data/processed/holidays.csv",
                index_col=0
            )
            holiday_df.index = pd.to_datetime(holiday_df.index)
            
            # Load grid data
            grid_df = pd.read_csv(
                self.project_root / "data/processed/SwedenGrid.csv",
                index_col=0
            )
            grid_df.index = pd.to_datetime(grid_df.index)
            
            # Merge all features
            df = price_df.join(time_df, how='left')
            df = df.join(holiday_df, how='left')
            df = df.join(grid_df, how='left')
            
            # Handle missing values
            df = df.ffill().bfill()
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading feature data: {str(e)}")
            raise

    def predict_next_week(self):
        """Predict prices for the next week starting from tomorrow"""
        logging.info("Preparing next week predictions...")
        
        # Load feature data
        df = self.load_feature_data()
        
        # Get start and end dates
        tomorrow = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
        end_date = tomorrow + pd.Timedelta(days=7)
        
        logging.info(f"Predicting prices from {tomorrow.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Get the window size from config
        window_size = self.feature_config.training["window_size"]
        
        # Get historical window ending today
        window_start = tomorrow - pd.Timedelta(hours=window_size)
        historical_data = df[df.index <= tomorrow].copy()
        
        if len(historical_data) < window_size:
            raise ValueError(f"Insufficient historical data. Need at least {window_size} hours.")
        
        # Get required features in correct order
        feature_cols = self.feature_config.get_ordered_features()
        
        # Verify all required features exist
        missing_features = self.feature_config.verify_features(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Scale price features
        price_cols = self.feature_config.price_cols
        price_data = historical_data[price_cols].values
        scaled_prices = self.price_scaler.transform(price_data)
        for i, col in enumerate(price_cols):
            historical_data[col] = scaled_prices[:, i]
        
        # Scale grid features
        grid_cols = self.feature_config.grid_cols
        grid_data = historical_data[grid_cols].values
        scaled_grid = self.grid_scaler.transform(grid_data)
        for i, col in enumerate(grid_cols):
            historical_data[col] = scaled_grid[:, i]
        
        # Prepare initial window
        window_data = historical_data[feature_cols].values[-window_size:]
        current_window = window_data.reshape(1, window_size, len(feature_cols))
        
        # Generate predictions for each hour
        predictions = []
        prediction_times = pd.date_range(tomorrow, end_date, freq='H', inclusive='left')
        
        for current_time in prediction_times:
            # Get prediction for next 24 hours
            pred = self.model.predict(current_window, verbose=0)[0]
            next_hour_pred = pred[0]  # Take only the next hour
            predictions.append(next_hour_pred)
            
            # Update the window for next prediction
            new_row = current_window[0, -1].copy()
            new_row[0] = next_hour_pred  # Update price
            
            # Update time-based features for next hour
            cyclical_start = len(price_cols)
            new_row[cyclical_start:cyclical_start+6] = [
                np.sin(2 * np.pi * current_time.hour / 24),
                np.cos(2 * np.pi * current_time.hour / 24),
                np.sin(2 * np.pi * current_time.dayofweek / 7),
                np.cos(2 * np.pi * current_time.dayofweek / 7),
                np.sin(2 * np.pi * current_time.month / 12),
                np.cos(2 * np.pi * current_time.month / 12)
            ]
            
            # Shift window and add new row
            current_window = np.roll(current_window, -1, axis=1)
            current_window[0, -1] = new_row
        
        # Inverse transform predictions
        dummy_pred = np.zeros((len(predictions), len(self.price_scaler.scale_)))
        dummy_pred[:, 0] = predictions
        predictions_inv = self.price_scaler.inverse_transform(dummy_pred)[:, 0]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'timestamp': prediction_times,
            'price': predictions_inv
        }).set_index('timestamp')
        
        # Plot predictions
        self._plot_week_prediction(results, tomorrow)
        
        # Print summary statistics
        print("\n=== Next Week Price Predictions ===")
        print(f"Period: {tomorrow.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Mean Price: {results['price'].mean():.2f} öre/kWh")
        print(f"Min Price: {results['price'].min():.2f} öre/kWh")
        print(f"Max Price: {results['price'].max():.2f} öre/kWh")
        
        # Print daily averages
        print("\nDaily Averages:")
        daily_avg = results.resample('D')['price'].mean()
        for date, price in daily_avg.items():
            print(f"{date.strftime('%Y-%m-%d')}: {price:.2f} öre/kWh")
        
        return results
    
    def _plot_week_prediction(self, predictions, start_date):
        """Plot detailed week prediction"""
        plt.figure(figsize=(15, 10))
        
        # Price predictions
        plt.subplot(2, 1, 1)
        plt.plot(predictions.index, predictions['price'], 'b-', label='Predicted Price')
        plt.title(f'Week Price Predictions Starting {start_date.strftime("%Y-%m-%d")}')
    plt.xlabel('Time')
    plt.ylabel('Price (öre/kWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

        # Daily averages
        plt.subplot(2, 1, 2)
        daily_avg = predictions.resample('D')['price'].mean()
        plt.bar(daily_avg.index, daily_avg.values, alpha=0.7)
        plt.title('Daily Average Prices')
        plt.xlabel('Date')
        plt.ylabel('Average Price (öre/kWh)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive error metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Additional metrics
        metrics['bias'] = np.mean(y_pred - y_true)
        metrics['std_error'] = np.std(y_pred - y_true)
        metrics['max_error'] = np.max(np.abs(y_pred - y_true))
        
        # Correlation
        metrics['correlation'] = np.corrcoef(y_true, y_pred)[0, 1]
        
        return metrics
    
    def inverse_transform_prices(self, y_scaled):
        """Helper to inverse transform scaled prices"""
        dummy = np.zeros((len(y_scaled), len(self.price_scaler.scale_)))
        dummy[:, 0] = y_scaled
        return self.price_scaler.inverse_transform(dummy)[:, 0]
    
    def plot_error_distribution(self, errors, title="Error Distribution"):
        """Plot error distribution with normal curve"""
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.title(title)
        plt.xlabel("Prediction Error (öre/kWh)")
        plt.ylabel("Frequency")
        
        # Add normal curve
        mu, std = stats.norm.fit(errors)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        plt.plot(x, p * len(errors) * (xmax - xmin) / 30, 'r-', lw=2, 
                label=f'Normal: μ={mu:.2f}, σ={std:.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_residuals(self, y_true, y_pred, title="Residual Plot"):
        """Plot residuals against predicted values"""
        plt.figure(figsize=(10, 6))
        residuals = y_pred - y_true
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(title)
        plt.xlabel("Predicted Price (öre/kWh)")
        plt.ylabel("Residuals (öre/kWh)")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def evaluate_single_day(self, target_date):
        """Detailed evaluation for a single day"""
    target_date = pd.to_datetime(target_date)
        logging.info(f"Evaluating predictions for: {target_date.strftime('%Y-%m-%d')}")
        
        # Find matching sample
    target_day_start = pd.Timestamp(target_date.date())
    target_day_end = target_day_start + pd.Timedelta(days=1)
        matching_mask = (self.timestamps >= target_day_start) & (self.timestamps < target_day_end)
    matching_indices = np.where(matching_mask)[0]
        
    if len(matching_indices) == 0:
            logging.error(f"No predictions found for {target_date.strftime('%Y-%m-%d')}")
        return
    
        # Get midnight sample if available
        day_timestamps = self.timestamps[matching_indices]
    midnight_mask = day_timestamps.hour == 0
        sample_index = matching_indices[midnight_mask.argmax() if midnight_mask.any() else 0]
        
        # Make prediction
        X_sample = self.X_test[sample_index:sample_index+1]
        y_true = self.y_test[sample_index]
        y_pred = self.model.predict(X_sample, verbose=0)[0]
        
        # Inverse transform
        y_true_inv = self.inverse_transform_prices(y_true)
        y_pred_inv = self.inverse_transform_prices(y_pred)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true_inv, y_pred_inv)
        
        # Create visualizations
        self._plot_day_prediction(y_true_inv, y_pred_inv, self.timestamps[sample_index])
        self.plot_error_distribution(y_pred_inv - y_true_inv, 
                                   f"Error Distribution - {target_date.strftime('%Y-%m-%d')}")
        self.plot_residuals(y_true_inv, y_pred_inv, 
                          f"Residual Plot - {target_date.strftime('%Y-%m-%d')}")
        
        # Print detailed statistics
        self._print_metrics(metrics, f"Daily Metrics - {target_date.strftime('%Y-%m-%d')}")
        
        return metrics
    
    def evaluate_month(self, target_month):
        """Comprehensive evaluation for an entire month"""
        try:
            target_month_start = pd.to_datetime(f"{target_month}-01")
        except Exception as e:
            logging.error("Invalid month format. Please use YYYY-MM")
            return
            
        target_month_end = target_month_start + pd.DateOffset(months=1)
        
        # Filter data for the month
        month_mask = (self.timestamps >= target_month_start) & (self.timestamps < target_month_end)
        month_indices = np.where(month_mask)[0]
        
        if len(month_indices) == 0:
            logging.error(f"No predictions found for {target_month}")
            return
        
        # Collect predictions and actuals
        all_true = []
        all_pred = []
        daily_metrics = []
        
        # Group by day
        df_indices = pd.DataFrame({
            'index': month_indices, 
            'timestamp': self.timestamps[month_indices]
        })
        df_indices['date'] = df_indices['timestamp'].dt.date
        
        # Evaluate each day
        for date in df_indices['date'].unique():
            day_indices = df_indices[df_indices['date'] == date]['index'].values
            day_timestamps = self.timestamps[day_indices]
            
            # Get midnight sample if available
            midnight_mask = day_timestamps.hour == 0
            sample_index = day_indices[midnight_mask.argmax() if midnight_mask.any() else 0]
            
            # Make prediction
            X_sample = self.X_test[sample_index:sample_index+1]
            y_true = self.y_test[sample_index]
            y_pred = self.model.predict(X_sample, verbose=0)[0]
            
            # Inverse transform
            y_true_inv = self.inverse_transform_prices(y_true)
            y_pred_inv = self.inverse_transform_prices(y_pred)
            
            all_true.extend(y_true_inv)
            all_pred.extend(y_pred_inv)
            
            # Calculate daily metrics
            daily_metrics.append({
                'date': date,
                **self.calculate_metrics(y_true_inv, y_pred_inv)
            })
        
        # Calculate monthly metrics
        monthly_metrics = self.calculate_metrics(np.array(all_true), np.array(all_pred))
        
        # Create visualizations
        self._plot_monthly_predictions(all_true, all_pred, target_month)
        self.plot_error_distribution(np.array(all_pred) - np.array(all_true),
                                   f"Monthly Error Distribution - {target_month}")
        self.plot_residuals(np.array(all_true), np.array(all_pred),
                          f"Monthly Residual Plot - {target_month}")
        
        # Plot daily metrics trends
        self._plot_daily_metrics_trends(daily_metrics)
        
        # Print detailed statistics
        self._print_metrics(monthly_metrics, f"Monthly Metrics - {target_month}")
        
        return monthly_metrics, daily_metrics
    
    def _plot_day_prediction(self, y_true, y_pred, start_time):
        """Plot detailed day prediction"""
        hours = pd.date_range(start_time, periods=24, freq='h')
        
    plt.figure(figsize=(15, 6))
        plt.step(hours, y_true, 'b-', label='Actual', where='post')
        plt.step(hours, y_pred, 'r--', label='Predicted', where='post')
        
        plt.fill_between(hours, y_true, y_pred, 
                        where=y_pred >= y_true,
                        color='red', alpha=0.1, 
                        label='Over-prediction')
        plt.fill_between(hours, y_true, y_pred,
                        where=y_pred < y_true,
                        color='blue', alpha=0.1,
                        label='Under-prediction')
        
        plt.title(f'24-Hour Price Prediction - {start_time.strftime("%Y-%m-%d")}')
    plt.xlabel('Time')
    plt.ylabel('Price (öre/kWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    def _plot_monthly_predictions(self, all_true, all_pred, target_month):
        """Plot monthly predictions with additional analysis"""
        plt.figure(figsize=(15, 10))
        
        # Price predictions
        plt.subplot(2, 1, 1)
        x = range(len(all_true))
        plt.plot(x, all_true, 'b-', label='Actual', alpha=0.7)
        plt.plot(x, all_pred, 'r--', label='Predicted', alpha=0.7)
        plt.title(f'Price Predictions - {target_month}')
        plt.xlabel('Hours')
        plt.ylabel('Price (öre/kWh)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Error analysis
        plt.subplot(2, 1, 2)
        errors = np.array(all_pred) - np.array(all_true)
        plt.plot(x, errors, 'g-', label='Prediction Error', alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.fill_between(x, 0, errors, alpha=0.2)
        plt.title('Prediction Errors')
        plt.xlabel('Hours')
        plt.ylabel('Error (öre/kWh)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_daily_metrics_trends(self, daily_metrics):
        """Plot trends in daily metrics"""
        df_metrics = pd.DataFrame(daily_metrics)
        df_metrics['date'] = pd.to_datetime(df_metrics['date'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Daily Metrics Trends')
        
        # MAPE trend
        axes[0, 0].plot(df_metrics['date'], df_metrics['mape'], 'b-')
        axes[0, 0].set_title('MAPE')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('MAPE (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE trend
        axes[0, 1].plot(df_metrics['date'], df_metrics['rmse'], 'r-')
        axes[0, 1].set_title('RMSE')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('RMSE (öre/kWh)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Bias trend
        axes[1, 0].plot(df_metrics['date'], df_metrics['bias'], 'g-')
        axes[1, 0].set_title('Bias')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Bias (öre/kWh)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # R² trend
        axes[1, 1].plot(df_metrics['date'], df_metrics['r2'], 'm-')
        axes[1, 1].set_title('R²')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _print_metrics(self, metrics, title="Model Metrics"):
        """Print formatted metrics"""
        print(f"\n=== {title} ===")
        print(f"MAE: {metrics['mae']:.2f} öre/kWh")
        print(f"RMSE: {metrics['rmse']:.2f} öre/kWh")
        print(f"MAPE: {metrics['mape']:.2f}%")
        print(f"R²: {metrics['r2']:.4f}")
        print(f"Bias: {metrics['bias']:.2f} öre/kWh")
        print(f"Std Error: {metrics['std_error']:.2f} öre/kWh")
        print(f"Max Error: {metrics['max_error']:.2f} öre/kWh")
        print(f"Correlation: {metrics['correlation']:.4f}")

def main():
    # Create evaluator instance
    evaluator = PriceModelEvaluator()
    
    # Predict next week's prices
    print("\nPredicting next week's prices...")
    next_week_predictions = evaluator.predict_next_week()

if __name__ == "__main__":
    main()