import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def add_features(df):
    """Add time-based features to the dataframe"""
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    return df

def prepare_prediction_data(df, window_size=48):
    """Prepare data for prediction including features"""
    # Add features
    df = add_features(df)
    
    # Add cyclical time features
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour/24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour/24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek/7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek/7)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month/12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month/12)
    
    # Add peak hour and weekend features
    df['is_peak_hour'] = ((df.index.hour >= 6) & (df.index.hour <= 22)).astype(int)
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    # Get the last window of data
    feature_cols = [
        'SE3_price_ore',
        'hour_sin', 'hour_cos',
        'day_of_week_sin', 'day_of_week_cos',
        'month_sin', 'month_cos',
        'is_peak_hour', 'is_weekend'
    ]
    
    last_window = df[feature_cols].values[-window_size:]
    return last_window

def make_prediction(model, last_window, steps=24, dampening_factor=0.9):
    """Make predictions with error dampening"""
    predictions = []
    confidence_intervals = []
    current_window = last_window.copy()
    
    for step in range(steps):
        # Make prediction
        pred = model.predict(current_window.reshape(1, *current_window.shape), verbose=0)
        
        # Apply dampening factor for longer-term predictions
        dampening = dampening_factor ** (step / 24)  # Gradually increase dampening
        pred = pred * dampening
        
        # Store prediction
        predictions.append(pred[0, 0])
        
        # Calculate confidence interval
        base_uncertainty = 0.05  # 5% base uncertainty
        time_uncertainty = 0.02 * step  # Additional 2% per step
        confidence = pred[0, 0] * (base_uncertainty + time_uncertainty)
        confidence_intervals.append([pred[0, 0] - confidence, pred[0, 0] + confidence])
        
        # Update window
        hour = (step + current_window[-1, 1]) % 24
        day = (current_window[-1, 3] + (1 if hour == 0 else 0)) % 7
        month = current_window[-1, 5]  # Keep month same for short-term predictions
        
        # Create new row with predicted values and updated features
        new_row = np.array([[
            pred[0, 0],  # Predicted price
            np.sin(2 * np.pi * hour/24),  # hour_sin
            np.cos(2 * np.pi * hour/24),  # hour_cos
            np.sin(2 * np.pi * day/7),    # day_sin
            np.cos(2 * np.pi * day/7),    # day_cos
            np.sin(2 * np.pi * month/12), # month_sin
            np.cos(2 * np.pi * month/12), # month_cos
            1 if (hour >= 6 and hour <= 22) else 0,  # is_peak_hour
            1 if (day >= 5) else 0        # is_weekend
        ]])
        
        # Update the window by dropping oldest row and adding new row
        current_window = np.vstack([current_window[1:], new_row])
    
    return np.array(predictions), np.array(confidence_intervals)

class PricePredictor:
    def __init__(self):
        """Initialize predictor with model and data"""
        # Setup paths relative to this file
        self.project_root = Path(__file__).parents[3]
        self.models_dir = self.project_root / "models/saved"
        self.data_dir = self.project_root / "data/processed"
        
        # Load model and scaler
        self.model = load_model(self.models_dir / "best_model.keras")
        self.scaler = joblib.load(self.models_dir / "scaler.save")
        
        # Load processed data
        self.df = pd.read_csv(self.data_dir / "SE3prices.csv", index_col=0)
        self.df.index = pd.to_datetime(self.df.index)
        
        # Constants
        self.window_size = 48  # Must match training window size
        self.MAX_REASONABLE_PRICE = 1000  # Maximum reasonable price in öre/kWh
    
    def prepare_window(self, start_date):
        """Prepare the input window for prediction"""
        # Get data up to start_date
        mask = self.df.index <= start_date
        if not mask.any():
            raise ValueError(f"No data available before {start_date}")
        
        historical_data = self.df[mask].copy()
        
        # Add features
        historical_data['hour_sin'] = np.sin(2 * np.pi * historical_data.index.hour/24)
        historical_data['hour_cos'] = np.cos(2 * np.pi * historical_data.index.hour/24)
        historical_data['day_of_week_sin'] = np.sin(2 * np.pi * historical_data.index.dayofweek/7)
        historical_data['day_of_week_cos'] = np.cos(2 * np.pi * historical_data.index.dayofweek/7)
        historical_data['month_sin'] = np.sin(2 * np.pi * historical_data.index.month/12)
        historical_data['month_cos'] = np.cos(2 * np.pi * historical_data.index.month/12)
        historical_data['is_peak_hour'] = ((historical_data.index.hour >= 6) & 
                                            (historical_data.index.hour <= 22)).astype(int)
        historical_data['is_weekend'] = (historical_data.index.dayofweek >= 5).astype(int)
        
        # Select features in correct order
        feature_cols = [
            'SE3_price_ore',
            'hour_sin', 'hour_cos',
            'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos',
            'is_peak_hour', 'is_weekend'
        ]
        
        # Get the last window
        last_window = historical_data[feature_cols].values[-self.window_size:]
        if len(last_window) < self.window_size:
            raise ValueError(f"Not enough historical data. Need at least {self.window_size} hours.")
        
        return last_window
    
    def predict_range(self, start_date, end_date, plot=True):
        """Make predictions for a specific date range"""
        # Convert to datetime if string
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Calculate number of hours to predict
        hours_to_predict = int((end_date - start_date).total_seconds() / 3600) + 1
        
        # Prepare initial window
        current_window = self.prepare_window(start_date - timedelta(hours=1))
        
        # Add price limiting and smoothing
        predictions = []
        dates = []
        
        for step in range(hours_to_predict):
            current_date = start_date + timedelta(hours=step)
            dates.append(current_date)
            
            # Make prediction
            pred = self.model.predict(current_window.reshape(1, *current_window.shape), 
                                    verbose=0)
            
            # Add dampening for extreme values
            if step > 0:
                last_pred = predictions[-1]
                max_change = last_pred * 0.5  # Limit maximum change to 50%
                pred = np.clip(pred, last_pred - max_change, last_pred + max_change)
            
            predictions.append(pred[0, 0])
            
            # Update window with new features
            hour = current_date.hour
            day = current_date.dayofweek
            month = current_date.month
            
            new_row = np.array([[
                pred[0, 0],  # Predicted price
                np.sin(2 * np.pi * hour/24),
                np.cos(2 * np.pi * hour/24),
                np.sin(2 * np.pi * day/7),
                np.cos(2 * np.pi * day/7),
                np.sin(2 * np.pi * month/12),
                np.cos(2 * np.pi * month/12),
                1 if (hour >= 6 and hour <= 22) else 0,
                1 if (day >= 5) else 0
            ]])
            
            # Update window
            current_window = np.vstack([current_window[1:], new_row])
        
        # Convert predictions to original scale
        predictions_inv = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        )
        
        # Clip unreasonable values
        predictions_inv = np.clip(predictions_inv, 0, self.MAX_REASONABLE_PRICE)
        
        if plot:
            self.plot_predictions(dates, predictions_inv)
        
        return pd.Series(predictions_inv.flatten(), index=dates, name='predicted_price')
    
    def predict_week_iteratively(self, start_date, plot=True):
        """Predict a week by making seven 24-hour predictions"""
        start_date = pd.to_datetime(start_date)
        all_predictions = []
        
        # Make 7 separate 24-hour predictions with overlap
        for day in range(7):
            day_start = start_date + timedelta(days=day)
            
            # Predict 36 hours but only keep middle 24 hours to avoid edge effects
            # (except for first and last day)
            day_end = day_start + timedelta(hours=35)
            
            # Get predictions for this period
            day_predictions = self.predict_range(
                day_start,
                day_end,
                plot=False
            )
            
            # For middle days, only keep the middle 24 hours
            if day > 0 and day < 6:
                day_predictions = day_predictions[6:30]  # Keep middle 24 hours
            elif day == 0:
                day_predictions = day_predictions[:24]  # Keep first 24 hours for first day
            else:
                day_predictions = day_predictions[-24:]  # Keep last 24 hours for last day
            
            all_predictions.append(day_predictions)
        
        # Combine all predictions
        combined_predictions = pd.concat(all_predictions)
        
        if plot:
            plt.figure(figsize=(15, 6))
            plt.step(combined_predictions.index, combined_predictions.values, 
                    label='Predicted Price', linewidth=2)
            plt.title("Weekly Price Prediction (24h Rolling Forecast)")
            plt.xlabel('Time')
            plt.ylabel('Price (öre/kWh)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            # Print statistics
            print("\nPrediction Summary:")
            print(f"Period: {combined_predictions.index[0]} to {combined_predictions.index[-1]}")
            print(f"Average predicted price: {combined_predictions.mean():.2f} öre/kWh")
            print(f"Min predicted price: {combined_predictions.min():.2f} öre/kWh")
            print(f"Max predicted price: {combined_predictions.max():.2f} öre/kWh")
        
        return combined_predictions
    
    def predict_day(self, start_date, plot=True):
        """Predict prices for a single day
        
        Args:
            start_date: datetime or str, start of prediction period
            plot: bool, whether to plot the predictions
        """
        start_date = pd.to_datetime(start_date)
        day_end = start_date + timedelta(hours=23)
        
        # Get predictions for this period
        day_predictions = self.predict_range(
            start_date,
            day_end,
            plot=False
        )
        
        if plot:
            plt.figure(figsize=(15, 6))
            plt.step(day_predictions.index, day_predictions.values, 
                    label='Predicted Price', linewidth=2)
            plt.title(f"Daily Price Prediction for {start_date.date()}")
            plt.xlabel('Time')
            plt.ylabel('Price (öre/kWh)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            # Print statistics
            print("\nPrediction Summary:")
            print(f"Date: {start_date.date()}")
            print(f"Average predicted price: {day_predictions.mean():.2f} öre/kWh")
            print(f"Min predicted price: {day_predictions.min():.2f} öre/kWh")
            print(f"Max predicted price: {day_predictions.max():.2f} öre/kWh")
            print(f"Peak hours (6-22) average: {day_predictions[6:23].mean():.2f} öre/kWh")
            print(f"Off-peak hours average: {day_predictions[[*range(0,6), *range(23,24)]].mean():.2f} öre/kWh")
        
        return day_predictions
    
    def plot_predictions(self, dates, predictions, title="Price Predictions"):
        """Plot the predictions"""
        plt.figure(figsize=(15, 6))
        plt.step(dates, predictions, label='Predicted Price', linewidth=2)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price (öre/kWh)')
        plt.grid(True, alpha=0.3)
plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
plt.show()

        # Print statistics
print("\nPrediction Summary:")
        print(f"Period: {dates[0]} to {dates[-1]}")
        print(f"Average predicted price: {predictions.mean():.2f} öre/kWh")
        print(f"Min predicted price: {predictions.min():.2f} öre/kWh")
        print(f"Max predicted price: {predictions.max():.2f} öre/kWh")

def main():
    predictor = PricePredictor()
    
    # Example date
    date = pd.Timestamp('2025-02-08')
    
    # Predict single day
    print("\nPredicting prices for a single day...")
    daily_predictions = predictor.predict_day(date)
    
    # Predict week
    print("\nPredicting prices for the whole week...")
    weekly_predictions = predictor.predict_week_iteratively(date)

if __name__ == "__main__":
    main()
