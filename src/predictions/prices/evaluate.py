import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import joblib
from pathlib import Path
import os

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
        
        # Load model artifacts
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load model, scaler and test data"""
        try:
            self.model = load_model(self.models_dir / "best_model.keras")
            self.scaler = joblib.load(self.models_dir / "scaler.save")
            self.X_test = np.load(self.test_data_dir / "X_test.npy")
            self.y_test = np.load(self.test_data_dir / "y_test.npy")
            self.test_timestamps = np.load(
                self.test_data_dir / "test_timestamps.npy", 
                allow_pickle=True
            )
        except Exception as e:
            raise Exception(f"Error loading artifacts: {str(e)}")

    def validate_temporal_split(self):
        """Validate that predictions are truly out-of-sample"""
        print("\nTemporal Split Validation:")
        print(f"Test period: {self.test_timestamps[0]} to {self.test_timestamps[-1]}")
        
        # Check for sorted timestamps
        is_sorted = np.all(np.diff(self.test_timestamps) >= np.timedelta64(0))
        print(f"Timestamps are properly sorted: {is_sorted}")

    def evaluate(self):
        """Evaluate model performance"""
        self.validate_temporal_split()
        # Make predictions
        y_pred = self.model.predict(self.X_test, verbose=0)
        
        # Reshape arrays for inverse transform
        y_pred_reshaped = y_pred.reshape(-1, 1)
        y_test_reshaped = self.y_test.reshape(-1, 1)
        
        # Inverse transform to get original scale
        y_pred_inv = self.scaler.inverse_transform(y_pred_reshaped)
        y_test_inv = self.scaler.inverse_transform(y_test_reshaped)
        
        # Calculate metrics
        mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mse)
        
        print("\nModel Performance Metrics:")
        print(f"MAPE: {mape:.2%}")
        print(f"RMSE: {rmse:.2f} öre/kWh")
        
        # Calculate peak error (for top 10% prices)
        peak_threshold = np.percentile(y_test_inv, 90)
        peak_mask = y_test_inv >= peak_threshold
        peak_error = np.mean(np.abs(y_test_inv[peak_mask] - y_pred_inv[peak_mask]))
        print(f"Peak Price Error: {peak_error:.2f} öre/kWh")
        
        # Plot results
        self.plot_predictions(y_test_inv, y_pred_inv, days=365)
        
        # Analyze errors
        self.analyze_errors(y_test_inv, y_pred_inv)
        
    def plot_predictions(self, y_test, y_pred, days=7):
        """Plot the last n days of predictions vs actual values"""
        hours = days * 24
        plt.figure(figsize=(15, 6))
        
        # Plot only the last n days
        plt.step(self.test_timestamps[-hours:], y_test[-hours:], 
                where='post', label='Actual', linewidth=2)
        plt.step(self.test_timestamps[-hours:], y_pred[-hours:], 
                where='post', label='Predicted', linewidth=2, linestyle='--')
        
        plt.title(f'Price Predictions vs Actual Values (Last {days} Days)')
        plt.xlabel('Time')
        plt.ylabel('Price (öre/kWh)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def analyze_errors(self, y_test, y_pred):
        """Analyze prediction errors in detail"""
        errors = y_test - y_pred
        
        print("\nDetailed Error Analysis:")
        print(f"Mean Error: {np.mean(errors):.2f} öre/kWh")
        print(f"Std Error: {np.std(errors):.2f} öre/kWh")
        print(f"Max Underprediction: {np.max(errors):.2f} öre/kWh")
        print(f"Max Overprediction: {np.min(errors):.2f} öre/kWh")
        
        # Error distribution by price range
        price_ranges = [(0, 50), (50, 100), (100, 200), (200, np.inf)]
        for low, high in price_ranges:
            mask = (y_test >= low) & (y_test < high)
            if np.any(mask):
                range_error = np.mean(np.abs(errors[mask]))
                print(f"Mean Abs Error for {low}-{high} öre/kWh: {range_error:.2f}")

def main():
    evaluator = ModelEvaluator()
    evaluator.evaluate()

if __name__ == "__main__":
    main()
