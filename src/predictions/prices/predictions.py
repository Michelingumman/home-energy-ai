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

from src.predictions.prices.gather_data import FeatureConfig
feature_config = FeatureConfig()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PricePredictor:
    def __init__(self, window_size=168, apply_dampening=False):
        """
        Initialize predictor with model, scalers, and processed data.
        :param window_size: The historical window length (in hours) to use.
        :param apply_dampening: Option to limit abrupt changes (set False to capture true spikes).
        """
        self.window_size = window_size
        self.apply_dampening = apply_dampening
        
        # Setup paths
        self.project_root = Path(__file__).resolve().parents[3]
        self.models_dir = self.project_root / "models/saved"
        self.data_dir = self.project_root / "data/processed"
        
        # Load feature configuration
        self.feature_config = feature_config
        
        # Load production model and scalers for future predictions
        self.model = load_model(self.models_dir / "price_model_production.keras")
        self.price_scaler = joblib.load(self.models_dir / "price_scaler_production.save")
        self.grid_scaler = joblib.load(self.models_dir / "grid_scaler_production.save")
        
        # Load processed data
        self._load_preprocessed_data()
        
        # Constant for output clipping (in öre/kWh)
        self.MAX_REASONABLE_PRICE = 1000

    def _load_preprocessed_data(self):
        """Load all preprocessed data files"""
        logging.info("Loading preprocessed data...")
        
        # Load price data
        self.df = pd.read_csv(self.data_dir / "SE3prices.csv", index_col=0)
        self.df.index = pd.to_datetime(self.df.index)
        
        # Load time features
        time_df = pd.read_csv(self.data_dir / "time_features.csv", index_col=0)
        time_df.index = pd.to_datetime(time_df.index)
        
        # Load holiday data
        holiday_df = pd.read_csv(self.data_dir / "holidays.csv", index_col=0)
        holiday_df.index = pd.to_datetime(holiday_df.index)
        
        # Load grid data
        grid_df = pd.read_csv(self.data_dir / "SwedenGrid.csv", index_col=0)
        grid_df.index = pd.to_datetime(grid_df.index)
        
        # Merge all features
        self.df = self.df.join(time_df, how='left')
        self.df = self.df.join(holiday_df, how='left')
        self.df = self.df.join(grid_df, how='left')
        
        # Handle any missing values
        self.df = self.df.ffill().bfill()
        
        # Verify all required features exist
        missing_features = self.feature_config.verify_features(self.df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        logging.info(f"Loaded data from {self.df.index.min()} to {self.df.index.max()}")

    def _prepare_window(self, start_hour, end_hour=None):
        """Prepare data window for prediction"""
        if end_hour is None:
            end_hour = start_hour
            
        # Get window data
        window_start = start_hour - pd.Timedelta(hours=self.window_size)
        window_data = self.df[window_start:start_hour].copy()
        
        # Scale price features
        price_cols = self.feature_config.price_cols
        window_data[price_cols] = self.price_scaler.transform(window_data[price_cols])
        
        # Scale grid features
        grid_cols = self.feature_config.grid_cols
        window_data[grid_cols] = self.grid_scaler.transform(window_data[grid_cols])
        
        # Get features in correct order
        ordered_features = self.feature_config.get_ordered_features()
        window_array = window_data[ordered_features].values
        
        return window_array

    def predict_day(self, target_date):
        """Predict prices for a full day"""
        start_hour = pd.Timestamp(target_date).normalize()
        end_hour = start_hour + pd.Timedelta(hours=23)
        
        # Generate dates array for all 24 hours
        dates = pd.date_range(start=start_hour, end=end_hour, freq='H')
        
        # Check if we have actual data for this period
        has_actual_data = all(date in self.df.index for date in dates[:-24])
        
        if has_actual_data:
            # Use actual data up to start of prediction day
            window_array = self._prepare_window(start_hour)
            predictions = self.model.predict(window_array.reshape(1, *window_array.shape), verbose=0)
            predictions_inv = self.price_scaler.inverse_transform(predictions)[:, 0]
        else:
            # Use recursive prediction
            predictions_inv = self.predict_range(start_hour, end_hour)
        
        # Create Series with predictions
        predictions_series = pd.Series(predictions_inv, index=dates)
        
        # Apply dampening if enabled
        if self.apply_dampening:
            predictions_series = self._dampen_predictions(predictions_series)
        
        return predictions_series

    def _add_features(self, df):
        """Add all required features to match training data.
           (For historical data, we add cyclical, binary, and price-based features.)
        """
        # Cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        
        # Binary and categorical features
        df['is_peak_hour'] = ((df.index.hour >= 6) & (df.index.hour <= 22)).astype(int)
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['season'] = 0  # default for spring/fall
        df.loc[(df.index.month >= 6) & (df.index.month <= 8), 'season'] = 1  # summer
        df.loc[(df.index.month == 12) | (df.index.month <= 2), 'season'] = -1  # winter
        
        # Grid features (if missing, fill with placeholder using mean price)
        for feature in self.feature_config.grid_cols:
            if feature not in df.columns:
                df[feature] = df['SE3_price_ore'].mean()
        
        # Price-based features computed with rolling statistics
        df['price_24h_avg'] = df['SE3_price_ore'].rolling(window=24, min_periods=1).mean()
        df['price_168h_avg'] = df['SE3_price_ore'].rolling(window=168, min_periods=1).mean()
        df['price_24h_std'] = df['SE3_price_ore'].rolling(window=24, min_periods=1).std()
        df['hour_avg_price'] = df.groupby(df.index.hour)['SE3_price_ore'].transform('mean')
        df['price_vs_hour_avg'] = df['SE3_price_ore'] / df['hour_avg_price']
        
        # Fill any missing values
        df = df.ffill().bfill()
        return df

    def compute_time_features(self, current_date):
        """
        Compute cyclical and binary time features for a given timestamp.
        Returns a dictionary with keys for cyclical and binary features.
        """
        features = {}
        
        # Cyclical features (always computed)
        features['hour_sin'] = np.sin(2 * np.pi * current_date.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * current_date.hour / 24)
        features['day_of_week_sin'] = np.sin(2 * np.pi * current_date.weekday() / 7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * current_date.weekday() / 7)
        features['month_sin'] = np.sin(2 * np.pi * current_date.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * current_date.month / 12)
        
        # Binary features (from config)
        binary_cols = self.feature_config.binary_cols
        if 'is_peak_hour' in binary_cols:
            features['is_peak_hour'] = 1 if (6 <= current_date.hour <= 22) else 0
        if 'is_weekend' in binary_cols:
            features['is_weekend'] = 1 if current_date.weekday() >= 5 else 0
        if 'season' in binary_cols:
            if current_date.month in [12, 1, 2]:
                features['season'] = -1  # winter
            elif 6 <= current_date.month <= 8:
                features['season'] = 1   # summer
            else:
                features['season'] = 0   # spring/fall
                
        # Holiday features (from loaded data if available)
        if 'is_holiday' in binary_cols:
            features['is_holiday'] = self.df.loc[current_date, 'is_holiday'] if current_date in self.df.index else 0
        if 'is_holiday_eve' in binary_cols:
            features['is_holiday_eve'] = self.df.loc[current_date, 'is_holiday_eve'] if current_date in self.df.index else 0
        
        return features

    def prepare_window(self, start_date):
        """
        Prepare the input window (historical data) up to a given start_date.
        Adds all required features and applies scaling.
        Returns the numpy array of the ordered features.
        """
        mask = self.df.index <= start_date
        if not mask.any():
            raise ValueError(f"No data available before {start_date}")
        historical_data = self.df[mask].copy()
        df_with_features = self._add_features(historical_data)
        
        # Scale price features
        price_data = df_with_features[self.feature_config.price_cols].values
        scaled_prices = self.price_scaler.transform(price_data)
        for i, col in enumerate(self.feature_config.price_cols):
            df_with_features[col] = scaled_prices[:, i]
        
        # Scale grid features
        grid_data = df_with_features[self.feature_config.grid_cols].values
        scaled_grid = self.grid_scaler.transform(grid_data)
        for i, col in enumerate(self.feature_config.grid_cols):
            df_with_features[col] = scaled_grid[:, i]
        
        feature_cols = self.feature_config.get_ordered_features()
        return df_with_features[feature_cols].values[-self.window_size:]

    def predict_range(self, start_date, end_date, plot=True):
        """
        Make predictions for a specific range recursively.
        Features are dynamically loaded from config.json.
        """
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        horizon = int((end_date - start_date).total_seconds() / 3600) + 1
        
        # Prepare initial historical window ending at start_date - 1h
        current_window = self.prepare_window(start_date - timedelta(hours=1))
        predictions = []
        forecast_dates = []
        
        # Get feature group lengths from config
        price_len = len(self.feature_config.price_cols)
        cyclical_len = len(self.feature_config.cyclical_cols)
        binary_len = len(self.feature_config.binary_cols)
        grid_len = len(self.feature_config.grid_cols)
        
        for step in range(horizon):
            current_date = start_date + timedelta(hours=step)
            forecast_dates.append(current_date)
            
            # Predict next value
            pred = self.model.predict(current_window.reshape(1, *current_window.shape), verbose=0)
            new_price = pred[0, 0]
            
            # Apply dampening if enabled
            if self.apply_dampening and step > 0:
                last_pred = predictions[-1]
                max_change = abs(last_pred) * 0.5
                new_price = np.clip(new_price, last_pred - max_change, last_pred + max_change)
            
            predictions.append(new_price)
            
            # Build new row using feature groups from config
            new_row = np.zeros(current_window.shape[1])
            
            # Update price features
            if step == 0:
                new_row[0:price_len] = current_window[-1, 0:price_len]
            else:
                recent_prices = np.array(predictions[-min(24, step):])
                price_features = np.zeros(price_len)
                price_features[0] = new_price
                price_features[1] = np.mean(recent_prices)  # 24h average
                price_features[2] = np.mean(predictions)    # 168h average
                price_features[3] = np.std(recent_prices) if len(recent_prices) > 1 else 0
                price_features[4] = current_window[-1, price_len - 2]  # hour_avg_price
                price_features[5] = new_price / current_window[-1, price_len - 2] if current_window[-1, price_len - 2] != 0 else 1
                price_features_scaled = self.price_scaler.transform(price_features.reshape(1, -1))
                new_row[0:price_len] = price_features_scaled[0]
            
            # Update time-based features using helper
            time_feats = self.compute_time_features(current_date)
            
            # Fill cyclical features
            cyclical_start = price_len
            for i, col in enumerate(self.feature_config.cyclical_cols):
                new_row[cyclical_start + i] = time_feats[col]
            
            # Fill binary features
            binary_start = cyclical_start + cyclical_len
            for i, col in enumerate(self.feature_config.binary_cols):
                new_row[binary_start + i] = time_feats[col]
            
            # Update grid features from last window
            grid_start = binary_start + binary_len
            new_row[grid_start:grid_start + grid_len] = current_window[-1, grid_start:grid_start + grid_len]
            
            # Update window
            current_window = np.vstack([current_window[1:], new_row])
        
        # Inverse-transform predictions
        predictions_full = np.zeros((len(predictions), price_len))
        predictions_full[:, 0] = predictions
        predictions_inv = self.price_scaler.inverse_transform(predictions_full)[:, 0]
        predictions_inv = np.clip(predictions_inv, -100, self.MAX_REASONABLE_PRICE)
        
        predictions_series = pd.Series(predictions_inv, index=pd.to_datetime(forecast_dates), name="Prediction")
        if plot:
            self.plot_predictions(predictions_series)
        
        return predictions_series

    def _dampen_predictions(self, pred_series):
        """Apply dampening to a series of predictions"""
        dampened_series = pred_series.copy()
        for i in range(1, len(pred_series)):
            last_pred = pred_series[i-1]
            current_pred = pred_series[i]
            max_change = abs(last_pred) * 0.5  # Limit maximum change to 50%
            dampened_series[i] = np.clip(current_pred, last_pred - max_change, last_pred + max_change)
        return dampened_series

    def plot_predictions(self, pred_series):
        """Plot overall predictions."""
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

    def predict_week(self, start_date=None):
        """
        Predict prices for the next week (168 hours) starting from the given date.
        If no date is provided, starts from the latest available data point.
        """
        if start_date is None:
            start_date = self.df.index.max() + pd.Timedelta(hours=1)
        else:
            start_date = pd.to_datetime(start_date)

        end_date = start_date + pd.Timedelta(days=7)
        
        logging.info(f"Predicting prices from {start_date} to {end_date}")
        
        # Get predictions using the predict_range method
        predictions = self.predict_range(start_date, end_date)
        
        # Create a more detailed visualization for the week
        plt.figure(figsize=(15, 10))
        
        # Plot predictions
        plt.subplot(2, 1, 1)
        plt.step(predictions.index, predictions.values, where='post', label='Predicted', linewidth=2)
        plt.title("Weekly Price Prediction")
        plt.xlabel("Time")
        plt.ylabel("Price (öre/kWh)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)
        
        # Plot daily averages
        plt.subplot(2, 1, 2)
        daily_avg = predictions.resample('D').mean()
        plt.bar(daily_avg.index, daily_avg.values, alpha=0.7)
        plt.title("Daily Average Prices")
        plt.xlabel("Date")
        plt.ylabel("Average Price (öre/kWh)")
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\nWeekly Prediction Summary:")
        print(f"Period: {predictions.index[0]} to {predictions.index[-1]}")
        print(f"Overall Mean: {predictions.mean():.2f} öre/kWh")
        print(f"Min Price: {predictions.min():.2f} öre/kWh")
        print(f"Max Price: {predictions.max():.2f} öre/kWh")
        print("\nDaily Averages:")
        for date, avg_price in daily_avg.items():
            print(f"{date.strftime('%Y-%m-%d')}: {avg_price:.2f} öre/kWh")

    def predict_next_week(self):
        """Predict prices for the next week starting from tomorrow"""
        logging.info("Preparing next week predictions...")
        
        # Get start and end dates
        tomorrow = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
        end_date = tomorrow + pd.Timedelta(days=7)
        
        logging.info(f"Predicting prices from {tomorrow.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Get required features in correct order
        feature_cols = self.feature_config.get_ordered_features()
        
        # Prepare initial window
        window_data = self._prepare_prediction_window(tomorrow)
        current_window = window_data.reshape(1, self.window_size, len(feature_cols))
        
        # Generate predictions for each hour
        predictions = []
        prediction_times = pd.date_range(tomorrow, end_date, freq='H', inclusive='left')
        
        for current_time in prediction_times:
            # Get prediction for next 24 hours
            pred = self.model.predict(current_window, verbose=0)[0]
            next_hour_pred = pred[0]  # Take only the next hour
            predictions.append(next_hour_pred)
            
            # Update the window for next prediction
            new_row = self._create_feature_row(current_time, next_hour_pred)
            
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
        self._plot_week_prediction(results)
        
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
    
    def _prepare_prediction_window(self, start_time):
        """Prepare the initial window for prediction"""
        window_start = start_time - pd.Timedelta(hours=self.window_size)
        window_data = self.df[window_start:start_time].copy()
        
        if len(window_data) < self.window_size:
            raise ValueError(f"Insufficient historical data. Need at least {self.window_size} hours.")
        
        # Scale price features
        price_cols = self.feature_config.price_cols
        window_data[price_cols] = self.price_scaler.transform(window_data[price_cols])
        
        # Scale grid features
        grid_cols = self.feature_config.grid_cols
        window_data[grid_cols] = self.grid_scaler.transform(window_data[grid_cols])
        
        # Get features in correct order
        ordered_features = self.feature_config.get_ordered_features()
        return window_data[ordered_features].values[-self.window_size:]
    
    def _create_feature_row(self, timestamp, predicted_price):
        """Create a feature row for the next prediction"""
        feature_cols = self.feature_config.get_ordered_features()
        new_row = np.zeros(len(feature_cols))
        
        # Update price features
        price_features = np.zeros(len(self.feature_config.price_cols))
        price_features[0] = predicted_price  # Current price
        
        # Calculate rolling statistics if we have predictions
        if hasattr(self, '_recent_predictions'):
            self._recent_predictions.append(predicted_price)
            if len(self._recent_predictions) > 168:  # Keep only last week
                self._recent_predictions.pop(0)
            
            price_features[1] = np.mean(self._recent_predictions[-24:])  # 24h avg
            price_features[2] = np.mean(self._recent_predictions)  # 168h avg
            price_features[3] = np.std(self._recent_predictions[-24:])  # 24h std
            price_features[4] = np.mean(self._recent_predictions)  # hour avg (simplified)
            price_features[5] = predicted_price / price_features[4] if price_features[4] != 0 else 1
        else:
            self._recent_predictions = [predicted_price]
            # Use the last known values from historical data for initial predictions
            last_known = self.df.iloc[-1]
            price_features[1:] = [
                last_known['price_24h_avg'],
                last_known['price_168h_avg'],
                last_known['price_24h_std'],
                last_known['hour_avg_price'],
                predicted_price / last_known['hour_avg_price'] if last_known['hour_avg_price'] != 0 else 1
            ]
        
        # Scale price features
        price_features = self.price_scaler.transform(price_features.reshape(1, -1))[0]
        new_row[:len(price_features)] = price_features
        
        # Update time features
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        month = timestamp.month
        
        # Cyclical features
        cyclical_start = len(self.feature_config.price_cols)
        new_row[cyclical_start:cyclical_start+6] = [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day_of_week / 7),
            np.cos(2 * np.pi * day_of_week / 7),
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12)
        ]
        
        # Binary features
        binary_start = cyclical_start + len(self.feature_config.cyclical_cols)
        new_row[binary_start] = 1 if 6 <= hour <= 22 else 0  # is_peak_hour
        new_row[binary_start+1] = 1 if day_of_week >= 5 else 0  # is_weekend
        new_row[binary_start+2] = 0  # is_holiday (simplified)
        new_row[binary_start+3] = 0  # is_holiday_eve (simplified)
        new_row[binary_start+4] = (  # season
            -1 if month in [12, 1, 2] else  # winter
            1 if 6 <= month <= 8 else  # summer
            0  # spring/fall
        )
        
        # Grid features (use last known values)
        grid_start = binary_start + len(self.feature_config.binary_cols)
        last_grid = self.df[self.feature_config.grid_cols].iloc[-1].values
        new_row[grid_start:] = last_grid
        
        return new_row
    
    def _plot_week_prediction(self, predictions):
        """Plot detailed week prediction"""
        plt.figure(figsize=(15, 10))
        
        # Price predictions
        plt.subplot(2, 1, 1)
        plt.plot(predictions.index, predictions['price'], 'b-', label='Predicted Price')
        plt.title('Week Price Predictions')
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

def main():
    # Create predictor instance
    predictor = PricePredictor(window_size=168, apply_dampening=False)
    
    # Predict next week's prices
    print("\nPredicting next week's prices...")
    next_week_predictions = predictor.predict_next_week()

if __name__ == "__main__":
    main()
