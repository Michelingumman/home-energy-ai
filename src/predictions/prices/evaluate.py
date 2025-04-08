import numpy as np
import os
import sys
from pathlib import Path
import json

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from datetime import datetime, timedelta
from pathlib import Path
import logging
import matplotlib.dates as mdates

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import the feature config
from getPriceRelatedData import FeatureConfig

# Import the custom layer from the training module
try:
    from train import FeatureWeightingLayer
    custom_layer_loaded = True
except ImportError:
    logging.warning("Could not import FeatureWeightingLayer. Full model loading may fail.")
    custom_layer_loaded = False

# Define the asymmetric Huber loss function 
def asymmetric_huber_loss(y_true, y_pred, delta=1.0, asymmetry_factor=2.0):
    """Custom asymmetric Huber loss that penalizes underprediction more heavily"""
    import tensorflow as tf
    error = y_true - y_pred
    is_underpredict = tf.cast(tf.less(y_pred, y_true), tf.float32)
    
    # Calculate absolute error
    abs_error = tf.abs(error)
    
    # Standard Huber loss calculation
    huber_loss = tf.where(
        abs_error <= delta,
        0.5 * tf.square(error),  # MSE for small errors
        delta * (abs_error - 0.5 * delta)  # MAE for large errors
    )
    
    # Apply asymmetry factor to underpredictions
    asymmetric_factor = 1.0 + (asymmetry_factor - 1.0) * is_underpredict
    weighted_loss = huber_loss * asymmetric_factor
    
    return tf.reduce_mean(weighted_loss)

# Create a specific version with asymmetry factor 2.0 to match the trained model
def asymmetric_huber_loss_2_0(y_true, y_pred):
    """Version of asymmetric Huber loss with asymmetry factor fixed at 2.0"""
    return asymmetric_huber_loss(y_true, y_pred, delta=1.0, asymmetry_factor=2.0)

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
        self.viz_dir = self.results_dir / "visualizations"
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize feature config
        self.feature_config = FeatureConfig()
        
        # Load test data
        self.load_test_data()
        
        # Load model - will raise an error if it fails
        self.try_load_model()
        
        # Colors for consistent plotting
        self.colors = {
            'actual': '#1f77b4',  # blue
            'predicted': '#ff7f0e',  # orange
        }
    
    def load_test_data(self):
        """Load test data (X_test, y_test, timestamps)"""
        try:
            logging.info("Loading test data...")
            self.X_test = np.load(self.test_data_dir / "X_test.npy")
            self.y_test = np.load(self.test_data_dir / "y_test.npy") 
            self.timestamps = pd.to_datetime(
                np.load(self.test_data_dir / "test_timestamps.npy", allow_pickle=True)
            )
            
            # Load the scalers
            self.price_scaler = joblib.load(self.saved_dir / "price_scaler.save")
            self.grid_scaler = joblib.load(self.saved_dir / "grid_scaler.save")
            
            logging.info(f"Loaded test data with {len(self.X_test)} samples")
            return True
        except Exception as e:
            logging.error(f"Error loading test data: {str(e)}")
            return False
    
    def try_load_model(self):
        """Load the trained model, raising detailed errors if it fails"""
        logging.info("Loading model...")
        
        # Create custom objects dict for the loss function
        custom_objects = {
            'asymmetric_huber_loss': asymmetric_huber_loss,
            'asymmetric_huber_loss_2_0': asymmetric_huber_loss_2_0
        }
        
        # Add FeatureWeightingLayer if available
        if custom_layer_loaded:
            custom_objects['FeatureWeightingLayer'] = FeatureWeightingLayer
        else:
            raise ImportError("Could not import FeatureWeightingLayer. This is required for loading the model.")
        
        # First approach: Try to load with compile=False to bypass the partial function issue
        try:
            logging.info("Attempting to load model with compile=False...")
            model_path = self.saved_dir / "price_model_evaluation.keras"
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
                
            self.model = tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects=custom_objects
            )
            
            logging.info("Model loaded successfully, recompiling with custom loss function...")
            
            # Recompile with our custom loss function
            self.model.compile(
                optimizer='adam',
                loss=asymmetric_huber_loss_2_0,  # Use the fixed version with factor 2.0
                metrics=['mae', 'mse']
            )
            
            # Try to predict one sample to verify
            _ = self.model.predict(self.X_test[0:1])
            logging.info("Model prediction test successful")
            
            return True
            
        except Exception as e:
            # If the first approach failed, try the standard loading
            try:
                logging.info(f"First loading approach failed: {e}")
                logging.info("Attempting standard model loading...")
                
                self.model = tf.keras.models.load_model(
                    model_path,
                    custom_objects=custom_objects
                )
                
                # Try to predict one sample to verify
                _ = self.model.predict(self.X_test[0:1])
                logging.info("Model prediction test successful")
                
                return True
                
            except Exception as e2:
                error_msg = f"Failed to load model: {str(e2)}"
                logging.error(error_msg)
                
                # Get more details from the traceback
                import traceback
                tb = traceback.format_exc()
                logging.error(f"Detailed error:\n{tb}")
                
                raise RuntimeError(f"Could not load model: {error_msg}")
    
    def generate_predictions(self):
        """Generate predictions for test data using the loaded model"""
        if not hasattr(self, 'model'):
            raise RuntimeError("Model not loaded. Cannot generate predictions.")
            
        logging.info("Generating predictions using model...")
        try:
            self.y_pred = self.model.predict(self.X_test)
            
            # Save predictions
            np.save(self.test_data_dir / "y_pred.npy", self.y_pred)
            logging.info(f"Saved predictions to {self.test_data_dir / 'y_pred.npy'}")
            return self.y_pred
        except Exception as e:
            error_msg = f"Failed to generate predictions: {str(e)}"
            logging.error(error_msg)
            
            # Get more details from the traceback
            import traceback
            tb = traceback.format_exc()
            logging.error(f"Detailed error:\n{tb}")
            
            raise RuntimeError(f"Failed to generate predictions: {error_msg}")
    
    def inverse_transform_prices(self, y_scaled):
        """Helper to inverse transform scaled prices"""
        # Set target index (default is 0)
        target_idx = 0
        
        if len(y_scaled.shape) == 1:
            # For 1D array (single sequence of prices)
            dummy = np.zeros((len(y_scaled), len(self.price_scaler.scale_)))
            dummy[:, target_idx] = y_scaled
            result = self.price_scaler.inverse_transform(dummy)[:, target_idx]
            return result
        
        elif len(y_scaled.shape) == 2:
            # For 2D array (multiple sequences of hourly predictions)
            num_sequences, seq_length = y_scaled.shape
            
            # Flatten the array to transform all values at once
            flattened = y_scaled.reshape(-1)
            
            # Create dummy array for inverse transform
            dummy = np.zeros((len(flattened), len(self.price_scaler.scale_)))
            dummy[:, target_idx] = flattened
            
            # Inverse transform
            inverse_flat = self.price_scaler.inverse_transform(dummy)[:, target_idx]
            
            # Reshape back to original structure
            result = inverse_flat.reshape(num_sequences, seq_length)

            return result / 100 # Convert to öre/kWh
        
        else:
            raise ValueError(f"Unexpected shape for y_scaled: {y_scaled.shape}")
    
    def prepare_dataframe(self):
        """Process predictions and create dataframe for analysis"""
        # Generate predictions if needed
        if not hasattr(self, 'y_pred'):
            self.generate_predictions()
        
        # Convert scaled values back to original range
        actual = self.inverse_transform_prices(self.y_test)
        predicted = self.inverse_transform_prices(self.y_pred)
        
        # Create a list of dataframes for each prediction window
        dfs = []
        
        # For each prediction window
        for i in range(len(self.timestamps)):
            # Create dates for next 24 hours
            dates = [self.timestamps[i] + pd.Timedelta(hours=h) for h in range(24)]
            
            # Create dataframe
            df = pd.DataFrame({
                'actual': actual[i],
                'predicted': predicted[i]
            }, index=dates)
            dfs.append(df)
        
        # Combine all dataframes
        combined_df = pd.concat(dfs)
        
        # Sort by timestamp
        combined_df = combined_df.sort_index()
        
        # Add hour, day of week, etc.
        combined_df['hour'] = combined_df.index.hour
        combined_df['day_of_week'] = combined_df.index.dayofweek
        combined_df['month'] = combined_df.index.month
        
        # Remove duplicate indices if any
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        
        # Store the processed dataframe
        self.predictions_df = combined_df
        
        return combined_df
    
    def calculate_metrics(self):
        """Calculate evaluation metrics"""
        if not hasattr(self, 'predictions_df'):
            self.prepare_dataframe()
            
        df = self.predictions_df
        
        # Overall metrics
        mae = np.mean(np.abs(df['actual'] - df['predicted']))
        rmse = np.sqrt(np.mean((df['actual'] - df['predicted'])**2))
        
        # Avoid division by zero for MAPE
        mape = np.mean(np.abs((df['actual'] - df['predicted']) / (df['actual'] + 1e-8))) * 100
        
        # Print overall metrics
        logging.info(f"Overall Metrics:")
        logging.info(f"Mean Absolute Error (MAE): {mae:.2f} öre/kWh")
        logging.info(f"Root Mean Squared Error (RMSE): {rmse:.2f} öre/kWh")
        logging.info(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        # Calculate metrics by hour of day
        hourly_metrics = {}
        for hour in range(24):
            hour_data = df[df['hour'] == hour]
            hour_mae = np.mean(np.abs(hour_data['actual'] - hour_data['predicted']))
            hourly_metrics[hour] = hour_mae
        
        # Find worst and best predicted hours
        worst_hour = max(hourly_metrics, key=hourly_metrics.get)
        best_hour = min(hourly_metrics, key=hourly_metrics.get)
        
        logging.info(f"\nHourly Performance:")
        logging.info(f"Best predicted hour: {best_hour}:00 (MAE: {hourly_metrics[best_hour]:.2f})")
        logging.info(f"Worst predicted hour: {worst_hour}:00 (MAE: {hourly_metrics[worst_hour]:.2f})")
        
        # Calculate metrics for morning and evening peaks
        morning_peak = df[(df['hour'] >= 6) & (df['hour'] <= 9)]
        evening_peak = df[(df['hour'] >= 17) & (df['hour'] <= 20)]
        
        morning_mae = np.mean(np.abs(morning_peak['actual'] - morning_peak['predicted']))
        evening_mae = np.mean(np.abs(evening_peak['actual'] - evening_peak['predicted']))
        
        logging.info(f"\nPeak Performance:")
        logging.info(f"Morning peak (6-9): MAE = {morning_mae:.2f}")
        logging.info(f"Evening peak (17-20): MAE = {evening_mae:.2f}")
        
        return {
            'overall': {'mae': mae, 'rmse': rmse, 'mape': mape},
            'hourly': hourly_metrics,
            'peaks': {'morning': morning_mae, 'evening': evening_mae}
        }
    
    def plot_daily_comparison(self, n_days=4):
        """Plot comparison of actual vs predicted prices for representative days"""
        if not hasattr(self, 'predictions_df'):
            self.prepare_dataframe()
            
        df = self.predictions_df
        
        # Find days with complete data (24 hours)
        day_counts = df.groupby(df.index.date).size()
        complete_days = day_counts[day_counts == 24].index
        
        if len(complete_days) == 0:
            logging.warning("No complete days found for daily comparison")
            return False
        
        # Select representative days (most recent ones)
        days_to_plot = min(n_days, len(complete_days))
        selected_days = sorted(complete_days, reverse=True)[:days_to_plot]
        
        # Create figure with subplots
        fig, axes = plt.subplots(days_to_plot, 1, figsize=(12, 4*days_to_plot))
        if days_to_plot == 1:
            axes = [axes]  # Make sure axes is iterable
        
        for i, day in enumerate(selected_days):
            # Get data for this day
            day_data = df[df.index.date == day].sort_index()
            
            # Plot actual and predicted prices
            ax = axes[i]
            ax.plot(day_data.index.hour, day_data['actual'], 'o-', 
                   color=self.colors['actual'], label='Actual', linewidth=2)
            ax.plot(day_data.index.hour, day_data['predicted'], 's--', 
                   color=self.colors['predicted'], label='Predicted', linewidth=2)
            
            # Highlight morning and evening peak hours
            ax.axvspan(6, 9, alpha=0.2, color='lightgray')
            ax.axvspan(17, 20, alpha=0.2, color='lightgray')
            
            # Set title and labels
            ax.set_title(f'Day: {day}')
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Price (öre/kWh)')
            ax.set_xticks(range(0, 24, 2))
            ax.grid(True, alpha=0.3)
            
            # Add legend to first subplot only
            if i == 0:
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "daily_comparison.png", dpi=300)
        plt.close()
        
        logging.info(f"Saved daily comparison plot to {self.viz_dir / 'daily_comparison.png'}")
        return True
    
    def plot_weekly_comparison(self):
        """Plot comparison of actual vs predicted prices for a representative week"""
        if not hasattr(self, 'predictions_df'):
            self.prepare_dataframe()
            
        df = self.predictions_df
        
        # Find a complete week
        max_date = df.index.max().date()
        
        # Look for a week ending on max_date or earlier
        for end_date in [max_date - timedelta(days=i) for i in range(14)]:
            start_date = end_date - timedelta(days=6)
            mask = (df.index.date >= start_date) & (df.index.date <= end_date)
            week_data = df[mask]
            
            # Check if we have at least 5 days with data
            day_coverage = len(np.unique([d.date() for d in week_data.index]))
            if day_coverage >= 5 and len(week_data) >= 120:  # At least 5 days with 120 hours total
                break
        else:
            # If no good week found
            logging.warning("Could not find a representative week with sufficient data")
            return False
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot actual and predicted prices
        ax.plot(week_data.index, week_data['actual'], '-', 
               color=self.colors['actual'], label='Actual', linewidth=2)
        ax.plot(week_data.index, week_data['predicted'], '--', 
               color=self.colors['predicted'], label='Predicted', linewidth=2)
        
        # Shade weekends
        for date in pd.date_range(start=start_date, end=end_date):
            if date.dayofweek >= 5:  # Weekend
                ax.axvspan(date, date + pd.Timedelta(days=1), alpha=0.2, color='lightgray')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%a\n%d %b'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        
        # Set title and labels
        ax.set_title(f'Week: {start_date} to {end_date}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (öre/kWh)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "weekly_comparison.png", dpi=300)
        plt.close()
        
        logging.info(f"Saved weekly comparison plot to {self.viz_dir / 'weekly_comparison.png'}")
        return True
    
    def plot_error_distribution(self):
        """Plot the distribution of prediction errors"""
        if not hasattr(self, 'predictions_df'):
            self.prepare_dataframe()
            
        df = self.predictions_df
        
        # Calculate errors
        errors = df['predicted'] - df['actual']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot error histogram
        bins = np.linspace(errors.min(), errors.max(), 50)
        ax.hist(errors, bins=bins, alpha=0.7, color='steelblue')
        
        # Add vertical line at zero
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
        
        # Add text with statistics
        stats_text = f"Mean: {errors.mean():.2f}\n"
        stats_text += f"Std Dev: {errors.std():.2f}\n"
        stats_text += f"Min: {errors.min():.2f}\n"
        stats_text += f"Max: {errors.max():.2f}"
        
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set title and labels
        ax.set_title('Distribution of Prediction Errors')
        ax.set_xlabel('Error (Predicted - Actual) [öre/kWh]')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "error_distribution.png", dpi=300)
        plt.close()
        
        logging.info(f"Saved error distribution plot to {self.viz_dir / 'error_distribution.png'}")
        return True
    
    def plot_hourly_performance(self):
        """Plot model performance by hour of day"""
        if not hasattr(self, 'predictions_df'):
            self.prepare_dataframe()
            
        df = self.predictions_df
        
        # Calculate metrics by hour
        hourly_metrics = {}
        for hour in range(24):
            hour_data = df[df['hour'] == hour]
            hour_mae = np.mean(np.abs(hour_data['actual'] - hour_data['predicted']))
            hourly_metrics[hour] = hour_mae
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot hourly MAE
        hours = list(hourly_metrics.keys())
        mae_values = list(hourly_metrics.values())
        
        ax.bar(hours, mae_values, alpha=0.7, color='steelblue')
        
        # Highlight morning and evening peak hours
        ax.axvspan(6, 9, alpha=0.2, color='lightgray')
        ax.axvspan(17, 20, alpha=0.2, color='lightgray')
        
        # Set title and labels
        ax.set_title('Prediction Error by Hour of Day')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Mean Absolute Error (öre/kWh)')
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "hourly_performance.png", dpi=300)
        plt.close()
        
        logging.info(f"Saved hourly performance plot to {self.viz_dir / 'hourly_performance.png'}")
        return True
    
    def run_evaluation(self):
        """Run a comprehensive evaluation with metrics and visualizations"""
        logging.info("Starting comprehensive evaluation...")
        
        try:
            # Check if we have test data
            if not hasattr(self, 'X_test') or not hasattr(self, 'y_test'):
                logging.error("Test data not available. Check paths and run training first.")
                return False
            
            # Generate predictions
            self.generate_predictions()
            
            # Prepare dataframe for analysis
            self.prepare_dataframe()
            
            # Calculate metrics
            metrics = self.calculate_metrics()
            
            # Generate visualizations
            self.plot_daily_comparison()
            self.plot_weekly_comparison()
            self.plot_error_distribution()
            self.plot_hourly_performance()
            
            logging.info("\nEvaluation complete! Check visualizations directory for plots.")
            return True
            
        except Exception as e:
            logging.error(f"Error running evaluation: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return False

def main():
    """Run the comprehensive evaluation"""
    logging.info("Starting price model evaluation...")
    
    try:
        # Create evaluator
        evaluator = PriceModelEvaluator()
        
        # Run evaluation
        evaluator.run_evaluation()
        
        logging.info("Evaluation complete!")
        
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 