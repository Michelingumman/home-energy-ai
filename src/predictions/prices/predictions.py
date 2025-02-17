import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys
import os

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.predictions.prices.feature_config import feature_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PricePredictor:
    def __init__(self, window_size=168):
        """Initialize predictor with model and data"""
        self.window_size = window_size
        
        # Setup paths
        self.project_root = Path(__file__).resolve().parents[3]
        self.models_dir = self.project_root / "models/saved"
        self.data_dir = self.project_root / "data/processed"
        
        # Load feature configuration
        self.feature_config = feature_config
        
        # Load model and scalers
        self.model = load_model(self.models_dir / "price_model.keras")
        self.price_scaler = joblib.load(self.models_dir / "price_scaler.save")
        self.grid_scaler = joblib.load(self.models_dir / "grid_scaler.save")
        
        # Load processed data
        self.df = pd.read_csv(self.data_dir / "SE3prices.csv", index_col=0)
        self.df.index = pd.to_datetime(self.df.index)
        
        # Try to load holiday and grid data
        try:
            holidays_df = pd.read_csv(self.data_dir / "holidays.csv", index_col=0)
            holidays_df.index = pd.to_datetime(holidays_df.index)
            self.df = self.df.join(holidays_df, how='left')
        except FileNotFoundError:
            logging.warning("No holidays.csv found. Holiday features will be set to 0")
        
        try:
            grid_df = pd.read_csv(self.data_dir / "SwedenGrid.csv", index_col=0)
            grid_df.index = pd.to_datetime(grid_df.index)
            self.df = self.df.join(grid_df, how='left')
        except FileNotFoundError:
            logging.warning("No SwedenGrid.csv found. Grid features will use defaults")
        
        # Constants
        self.MAX_REASONABLE_PRICE = 1000  # Maximum reasonable price in öre/kWh
    
    def _add_features(self, df):
        """Add all required features to match training data"""
        # Cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour/24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour/24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek/7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek/7)
        df['month_sin'] = np.sin(2 * np.pi * df.index.month/12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month/12)
        
        # Binary and categorical features
        df['is_peak_hour'] = ((df.index.hour >= 6) & (df.index.hour <= 22)).astype(int)
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['season'] = 0  # Default to spring/fall
        df.loc[(df.index.month >= 6) & (df.index.month <= 8), 'season'] = 1  # Summer
        df.loc[(df.index.month == 12) | (df.index.month <= 2), 'season'] = -1  # Winter
        
        # Holiday features (if not present)
        if 'is_holiday' not in df.columns:
            df['is_holiday'] = 0
        if 'is_holiday_eve' not in df.columns:
            df['is_holiday_eve'] = 0
        
        # Grid features (if not present)
        for feature in self.feature_config.grid_cols:
            if feature not in df.columns:
                df[feature] = df['SE3_price_ore'].mean()  # Use mean as placeholder
        
        # Price-based features
        df['price_24h_avg'] = df['SE3_price_ore'].rolling(window=24, min_periods=1).mean()
        df['price_168h_avg'] = df['SE3_price_ore'].rolling(window=168, min_periods=1).mean()
        df['price_24h_std'] = df['SE3_price_ore'].rolling(window=24, min_periods=1).std()
        df['hour_avg_price'] = df.groupby(df.index.hour)['SE3_price_ore'].transform('mean')
        df['price_vs_hour_avg'] = df['SE3_price_ore'] / df['hour_avg_price']
        
        # Handle any missing values
        df = df.ffill().bfill()
        
        return df
    
    def prepare_window(self, start_date):
        """Prepare the input window for prediction"""
        # Get data up to start_date
        mask = self.df.index <= start_date
        if not mask.any():
            raise ValueError(f"No data available before {start_date}")
        
        historical_data = self.df[mask].copy()
        
        # Add all features
        df = self._add_features(historical_data)
        
        # Scale features
        # Price features
        price_data = df[self.feature_config.price_cols].values
        scaled_prices = self.price_scaler.transform(price_data)
        for i, col in enumerate(self.feature_config.price_cols):
            df[col] = scaled_prices[:, i]
        
        # Grid features
        grid_data = df[self.feature_config.grid_cols].values
        scaled_grid = self.grid_scaler.transform(grid_data)
        for i, col in enumerate(self.feature_config.grid_cols):
            df[col] = scaled_grid[:, i]
        
        # Get the last window of data with all required features in the correct order
        feature_cols = self.feature_config.get_ordered_features()
        
        return df[feature_cols].values[-self.window_size:]
    
    def predict_range(self, start_date, end_date, plot=True):
        """Make predictions for a specific date range"""
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        horizon = int((end_date - start_date).total_seconds() / 3600) + 1
        
        # Prepare initial window
        current_window = self.prepare_window(start_date - timedelta(hours=1))
        
        predictions = []
        dates = []
        
        for step in range(horizon):
            current_date = start_date + timedelta(hours=step)
            dates.append(current_date)
            
            # Make prediction for next 24 hours
            pred = self.model.predict(current_window.reshape(1, *current_window.shape), verbose=0)
            
            # Add dampening for extreme values
            if step > 0:
                last_pred = predictions[-1]
                max_change = abs(last_pred) * 0.5  # Limit maximum change to 50%
                pred = np.clip(pred, last_pred - max_change, last_pred + max_change)
            
            predictions.append(pred[0, 0])
            
            # Update window with new prediction
            new_row = np.zeros(current_window.shape[1])
            
            # Set predicted price (first feature)
            new_row[0] = pred[0, 0]
            
            # Update time features
            hour = current_date.hour
            day = current_date.dayofweek
            month = current_date.month
            
            # Calculate price features and scale them
            if step > 0:
                window_prices = np.array(predictions[-min(24, step):])
                price_features = np.array([
                    pred[0, 0],  # SE3_price_ore
                    np.mean(window_prices),  # price_24h_avg
                    np.mean(predictions),    # price_168h_avg
                    np.std(window_prices),   # price_24h_std
                    current_window[-1, 4],  # hour_avg_price (keep previous)
                    pred[0, 0] / current_window[-1, 4] if current_window[-1, 4] != 0 else 1  # price_vs_hour_avg
                ]).reshape(-1, 1)
                
                # Scale price features
                price_features = self.price_scaler.transform(price_features).flatten()
                new_row[0:len(self.feature_config.price_cols)] = price_features
            else:
                new_row[0:len(self.feature_config.price_cols)] = current_window[-1, 0:len(self.feature_config.price_cols)]
            
            # Set cyclical time features
            cyc_start = len(self.feature_config.price_cols)
            new_row[cyc_start + 0] = np.sin(2 * np.pi * hour/24)     # hour_sin
            new_row[cyc_start + 1] = np.cos(2 * np.pi * hour/24)     # hour_cos
            new_row[cyc_start + 2] = np.sin(2 * np.pi * day/7)      # day_sin
            new_row[cyc_start + 3] = np.cos(2 * np.pi * day/7)      # day_cos
            new_row[cyc_start + 4] = np.sin(2 * np.pi * month/12)   # month_sin
            new_row[cyc_start + 5] = np.cos(2 * np.pi * month/12)   # month_cos
            
            # Set binary features
            bin_start = cyc_start + len(self.feature_config.cyclical_cols)
            new_row[bin_start + 0] = 1 if (hour >= 6 and hour <= 22) else 0  # is_peak_hour
            new_row[bin_start + 1] = 1 if day >= 5 else 0  # is_weekend
            new_row[bin_start + 2] = -1 if (month == 12 or month <= 2) else (1 if month >= 6 and month <= 8 else 0)  # season
            new_row[bin_start + 3] = current_window[-1, bin_start + 3]  # is_holiday (keep previous)
            new_row[bin_start + 4] = current_window[-1, bin_start + 4]  # is_holiday_eve (keep previous)
            
            # Grid features (keep previous values)
            grid_start = bin_start + len(self.feature_config.binary_cols)
            grid_end = grid_start + len(self.feature_config.grid_cols)
            new_row[grid_start:grid_end] = current_window[-1, grid_start:grid_end]
            
            # Update window
            current_window = np.vstack([current_window[1:], new_row])
        
        # Convert predictions back to original scale
        predictions_full = np.zeros((len(predictions), len(self.feature_config.price_cols)))
        predictions_full[:, 0] = predictions  # Set only the target variable
        predictions_inv = self.price_scaler.inverse_transform(predictions_full)[:, 0]
        
        # Clip to reasonable range
        predictions_inv = np.clip(predictions_inv, -100, self.MAX_REASONABLE_PRICE)
        
        predictions_series = pd.Series(predictions_inv, index=pd.to_datetime(dates), name="Prediction")
        
        if plot:
            self.plot_predictions(predictions_series)
        
        return predictions_series
    
    def predict_day(self, date, plot=True):
        """Predict prices for a single day"""
        date = pd.to_datetime(date)
        end_date = date + pd.Timedelta(hours=23)
        return self.predict_range(date, end_date, plot=plot)
    
    def predict_week(self, start_date, plot=True):
        """Predict prices for a week"""
        start_date = pd.to_datetime(start_date)
        end_date = start_date + pd.Timedelta(days=7) - pd.Timedelta(hours=1)
        return self.predict_range(start_date, end_date, plot=plot)
    
    def plot_predictions(self, pred_series):
        """Plot the predictions"""
        plt.figure(figsize=(15, 6))
        plt.step(pred_series.index, pred_series.values, where='post', label='Predicted', linewidth=2)
        plt.title("Predicted Electricity Prices")
        plt.xlabel("Time")
        plt.ylabel("Price (öre/kWh)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        print("\nPrediction Summary:")
        print(f"Period: {pred_series.index[0]} to {pred_series.index[-1]}")
        print(f"Mean: {pred_series.mean():.2f} öre/kWh")
        print(f"Min: {pred_series.min():.2f} öre/kWh")
        print(f"Max: {pred_series.max():.2f} öre/kWh")

def main():
    predictor = PricePredictor(window_size=168)
    
    # Example: Predict a single day
    date = pd.Timestamp('2024-02-08')
    print(f"\nPredicting prices for {date.date()}")
    daily_predictions = predictor.predict_day(date)
    
    # Example: Predict a week
    print(f"\nPredicting prices for week starting {date.date()}")
    weekly_predictions = predictor.predict_week(date)

if __name__ == "__main__":
    main()
