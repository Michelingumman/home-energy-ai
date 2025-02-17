import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
import joblib
from pathlib import Path
import os
import sys
import time

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.predictions.prices.feature_config import feature_config

class ModelEvaluator:
    def __init__(self, project_root=None):
        """Initialize evaluator with project paths"""
        # Setup paths
        if project_root is None:
            try:
                # Try script-style path resolution
                self.project_root = Path(__file__).parents[3]
            except NameError:
                # Fallback for notebook environment
                current_dir = Path(os.getcwd())
                self.project_root = current_dir.parents[3]
        else:
            self.project_root = Path(project_root)
            
        self.models_dir = self.project_root / "models/saved"
        self.test_data_dir = self.project_root / "models/test_data"
        
        # Load feature configuration
        self.feature_config = feature_config
        
        # Load model artifacts
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load model, scalers and test data"""
        try:
            self.model = load_model(self.models_dir / "robust_price_model.keras")
            self.price_scaler = joblib.load(self.models_dir / "robust_price_scaler.save")
            self.grid_scaler = joblib.load(self.models_dir / "robust_grid_scaler.save")
            
            # Detect scaler type
            if isinstance(self.price_scaler, MinMaxScaler):
                print("Using MinMaxScaler: Better preserves relative magnitudes of price spikes")
            elif isinstance(self.price_scaler, StandardScaler):
                print("Using StandardScaler: Normalizes features to mean=0, std=1")
            elif isinstance(self.price_scaler, RobustScaler):
                print("Using RobustScaler: Scales using statistics robust to outliers")
            
            self.X_test = np.load(self.test_data_dir / "X_test.npy")
            self.y_test = np.load(self.test_data_dir / "y_test.npy")
            self.test_timestamps = np.load(
                self.test_data_dir / "test_timestamps.npy", 
                allow_pickle=True
            )
            
            print("\nModel and data artifacts loaded successfully")
            print(f"Test set shape: {self.X_test.shape}")
            print(f"Prediction horizon: {self.y_test.shape[1]} hours")
            
        except Exception as e:
            raise Exception(f"Error loading artifacts: {str(e)}")

    def validate_temporal_split(self):
        """Validate that predictions are truly out-of-sample"""
        print("\nTemporal Split Validation:")
        print(f"Test period: {self.test_timestamps[0]} to {self.test_timestamps[-1]}")
        
        # Check for sorted timestamps
        is_sorted = np.all(np.diff(self.test_timestamps) >= np.timedelta64(0))
        print(f"Timestamps are properly sorted: {is_sorted}")
        
        # Calculate the size of the test set
        time_diff = self.test_timestamps[-1] - self.test_timestamps[0]
        test_days = time_diff.astype('timedelta64[D]').astype(int)
        print(f"Test set covers {test_days} days ({test_days/365.25:.1f} years)")
        
        # Print some additional temporal information
        n_samples = len(self.test_timestamps)
        print(f"Number of hourly samples: {n_samples}")
        print(f"Sampling frequency: {n_samples/test_days:.1f} samples per day")

    def evaluate(self):
        """Evaluate model performance"""
        self.validate_temporal_split()
        
        print("\nMaking predictions...")
        start_time = time.time()
        
        # Make predictions in batches to show progress
        batch_size = 1000
        n_batches = len(self.X_test) // batch_size + (1 if len(self.X_test) % batch_size != 0 else 0)
        
        y_pred_list = []
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(self.X_test))
            
            print(f"\rProcessing batch {i+1}/{n_batches} ({(i+1)/n_batches*100:.1f}%)...", end="")
            batch_pred = self.model.predict(self.X_test[start_idx:end_idx], verbose=0)
            y_pred_list.append(batch_pred)
        
        y_pred = np.vstack(y_pred_list)
        pred_time = time.time() - start_time
        print(f"\nPredictions completed in {pred_time:.1f} seconds")
        
        # Get the number of test samples and prediction horizon
        n_samples = y_pred.shape[0]
        horizon = y_pred.shape[1]  # Should be 24
        
        print(f"\nTest set size: {n_samples} samples")
        print(f"Prediction horizon: {horizon} hours")
        
        print("\nInverse transforming predictions...")
        start_time = time.time()
        
        # Initialize arrays for inverse-transformed predictions and actual values
        y_pred_inv = np.zeros_like(y_pred)
        y_test_inv = np.zeros_like(self.y_test)
        
        # Process each time step separately to maintain temporal consistency
        for i in range(horizon):
            if i % 6 == 0:  # Show progress every 6 hours
                print(f"\rProcessing hour {i+1}/{horizon} ({(i+1)/horizon*100:.1f}%)...", end="")
                
            # Create dummy arrays for the full set of price features
            y_pred_step = np.zeros((n_samples, len(self.feature_config.price_cols)))
            y_test_step = np.zeros((n_samples, len(self.feature_config.price_cols)))
            
            # Set the target variable (first column)
            y_pred_step[:, 0] = y_pred[:, i]
            y_test_step[:, 0] = self.y_test[:, i]
            
            # Inverse transform
            y_pred_inv[:, i] = self.price_scaler.inverse_transform(y_pred_step)[:, 0]
            y_test_inv[:, i] = self.price_scaler.inverse_transform(y_test_step)[:, 0]
        
        transform_time = time.time() - start_time
        print(f"\nInverse transformation completed in {transform_time:.1f} seconds")
        
        # Create a DataFrame with timestamps for easier analysis
        print("\nOrganizing results...")
        df_results = self._create_results_dataframe(y_test_inv, y_pred_inv)
        
        # Calculate and display overall metrics
        self._calculate_overall_metrics(df_results)
        
        # Plot yearly overviews
        self._plot_yearly_overview(df_results)
        
        # Plot monthly details
        self._plot_monthly_details(df_results)
        
        # Plot weekly samples
        self._plot_weekly_samples(df_results)
        
        # Plot error distributions
        self._plot_error_distributions(df_results)
        
        # Analyze errors
        self._analyze_detailed_errors(df_results)
        
        return df_results  # Return the results DataFrame for further analysis if needed
    
    def _create_results_dataframe(self, y_test, y_pred):
        """Create a DataFrame with all results and timestamps"""
        # Create timestamps for each prediction
        timestamps = pd.date_range(
            start=self.test_timestamps[0],
            periods=len(self.test_timestamps),
            freq='H'
        )
        
        # Initialize the DataFrame with actual values
        df = pd.DataFrame(index=timestamps)
        df['actual'] = y_test[:, 0]  # First column is the actual value at prediction time
        
        # Add predictions for each hour in the 24-hour window
        for i in range(24):
            df[f'pred_{i+1}h'] = y_pred[:, i]
        
        return df
    
    def _calculate_overall_metrics(self, df):
        """Calculate and display overall performance metrics"""
        print("\nOverall Model Performance Metrics:")
        
        # Calculate metrics for different prediction horizons
        horizons = [1, 6, 12, 24]
        for h in horizons:
            errors = df['actual'] - df[f'pred_{h}h']
            mape = mean_absolute_percentage_error(df['actual'], df[f'pred_{h}h'])
            rmse = np.sqrt(mean_squared_error(df['actual'], df[f'pred_{h}h']))
            
            print(f"\n{h}-hour ahead predictions:")
            print(f"MAPE: {mape:.2%}")
            print(f"RMSE: {rmse:.2f} öre/kWh")
            print(f"Mean Error: {np.mean(errors):.2f} öre/kWh")
            print(f"Std Error: {np.std(errors):.2f} öre/kWh")
    
    def _plot_yearly_overview(self, df):
        """Plot yearly overview of predictions vs actuals"""
        years = df.index.year.unique()
        
        for year in years:
            year_data = df[df.index.year == year]
            
            plt.figure(figsize=(15, 6))
            
            # Plot actual values
            plt.plot(year_data.index, year_data['actual'], 
                    label='Actual', linewidth=1)
            
            # Plot 24h predictions
            plt.plot(year_data.index, year_data['pred_24h'],
                    label='24h Prediction', linewidth=1, alpha=0.7)
            
            plt.title(f'Price Predictions vs Actual Values - {year}')
            plt.xlabel('Time')
            plt.ylabel('Price (öre/kWh)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            # Print yearly statistics
            print(f"\nYear {year} Statistics:")
            mape = mean_absolute_percentage_error(
                year_data['actual'], year_data['pred_24h']
            )
            rmse = np.sqrt(mean_squared_error(
                year_data['actual'], year_data['pred_24h']
            ))
            print(f"MAPE: {mape:.2%}")
            print(f"RMSE: {rmse:.2f} öre/kWh")
    
    def _plot_monthly_details(self, df):
        """Plot monthly details with error metrics"""
        # Calculate monthly metrics
        monthly_metrics = []
        
        for year in df.index.year.unique():
            for month in range(1, 13):
                mask = (df.index.year == year) & (df.index.month == month)
                if not any(mask):
                    continue
                
                month_data = df[mask]
                metrics = {
                    'year': year,
                    'month': month,
                    'mean_price': month_data['actual'].mean(),
                    'mape_24h': mean_absolute_percentage_error(
                        month_data['actual'], month_data['pred_24h']
                    ),
                    'rmse_24h': np.sqrt(mean_squared_error(
                        month_data['actual'], month_data['pred_24h']
                    ))
                }
                monthly_metrics.append(metrics)
        
        # Plot monthly performance
        metrics_df = pd.DataFrame(monthly_metrics)
        
        plt.figure(figsize=(15, 8))
        plt.subplot(2, 1, 1)
        for year in metrics_df['year'].unique():
            year_data = metrics_df[metrics_df['year'] == year]
            plt.plot(year_data['month'], year_data['mape_24h'],
                    label=str(year), marker='o')
        
        plt.title('Monthly MAPE by Year')
        plt.xlabel('Month')
        plt.ylabel('MAPE')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        for year in metrics_df['year'].unique():
            year_data = metrics_df[metrics_df['year'] == year]
            plt.plot(year_data['month'], year_data['mean_price'],
                    label=str(year), marker='o')
        
        plt.title('Average Monthly Prices')
        plt.xlabel('Month')
        plt.ylabel('Price (öre/kWh)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def _plot_weekly_samples(self, df):
        """Plot detailed weekly samples from different periods"""
        # Select sample weeks from different seasons
        for year in df.index.year.unique():
            seasons = [
                ('Winter', f'{year}-01-15'),
                ('Spring', f'{year}-04-15'),
                ('Summer', f'{year}-07-15'),
                ('Fall', f'{year}-10-15')
            ]
            
            for season, date in seasons:
                try:
                    # Get the week centered on the specified date
                    center_date = pd.Timestamp(date)
                    start_date = center_date - pd.Timedelta(days=3)
                    end_date = center_date + pd.Timedelta(days=3)
                    
                    week_data = df[start_date:end_date]
                    if len(week_data) < 7*24:  # Skip if we don't have a full week
                        continue
                    
                    plt.figure(figsize=(15, 6))
                    
                    # Plot actual values
                    plt.step(week_data.index, week_data['actual'],
                            label='Actual', linewidth=2)
                    
                    # Plot predictions made at different horizons
                    for h in [1, 12, 24]:
                        plt.step(week_data.index, week_data[f'pred_{h}h'],
                                label=f'{h}h Prediction',
                                linestyle='--',
                                alpha=0.7)
                    
                    plt.title(f'Weekly Detail - {season} {year}')
                    plt.xlabel('Time')
                    plt.ylabel('Price (öre/kWh)')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.show()
                    
                except Exception as e:
                    print(f"Skipping {season} {year} due to: {str(e)}")
    
    def _plot_error_distributions(self, df):
        """Plot error distributions and patterns"""
        plt.figure(figsize=(15, 10))
        
        # Error distribution by prediction horizon
        plt.subplot(2, 1, 1)
        horizons = [1, 6, 12, 24]
        for h in horizons:
            errors = df['actual'] - df[f'pred_{h}h']
            plt.hist(errors, bins=50, alpha=0.3,
                    label=f'{h}h Prediction', density=True)
        
        plt.title('Error Distribution by Prediction Horizon')
        plt.xlabel('Prediction Error (öre/kWh)')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Error by hour of day
        plt.subplot(2, 1, 2)
        for h in horizons:
            hourly_errors = df.groupby(df.index.hour)['actual'].mean() - \
                          df.groupby(df.index.hour)[f'pred_{h}h'].mean()
            plt.plot(range(24), hourly_errors, marker='o',
                    label=f'{h}h Prediction')
        
        plt.title('Average Error by Hour of Day')
        plt.xlabel('Hour')
        plt.ylabel('Mean Error (öre/kWh)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def _analyze_detailed_errors(self, df):
        """Print detailed error analysis"""
        print("\nDetailed Error Analysis:")
        
        # Analyze errors by price range
        price_ranges = [(0, 50), (50, 100), (100, 200), (200, np.inf)]
        
        for h in [1, 6, 12, 24]:
            print(f"\n{h}-hour Prediction Errors by Price Range:")
            for low, high in price_ranges:
                mask = (df['actual'] >= low) & (df['actual'] < high)
                if any(mask):
                    errors = np.abs(df[mask]['actual'] - df[mask][f'pred_{h}h'])
                    mean_error = errors.mean()
                    print(f"Price {low}-{high} öre/kWh: {mean_error:.2f} öre/kWh")

def main():
    evaluator = ModelEvaluator()
    evaluator.evaluate()

if __name__ == "__main__":
    main()
