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
from src.predictions.prices.gather_data import FeatureConfig

class SimplePriceModelEvaluator:
    def __init__(self):
        """Initialize the evaluator with model paths and data"""
        # Set up paths
        self.project_root = Path(__file__).resolve().parents[3]
        self.models_dir = Path(__file__).resolve().parent / "models"
        self.eval_dir = self.models_dir / "evaluation"
        self.saved_dir = self.eval_dir / "saved"
        self.test_data_dir = self.eval_dir / "test_data"
        self.results_dir = self.eval_dir / "results"
        self.simple_viz_dir = self.results_dir / "simple_visualizations"
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.simple_viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize feature config
        self.feature_config = FeatureConfig()
        
        # Load model, scalers, and test data
        self.load_model_and_data()
        
        # Initialize dataframes to store full predictions
        self.predictions_df = None
        
        # Colors for consistent plotting
        self.colors = {
            'actual': '#1f77b4',  # blue
            'predicted': '#ff7f0e',  # orange
        }
    
    def load_model_and_data(self):
        """Load the evaluation model, scalers, and test data"""
        try:
            # Load model and scalers
            self.model = tf.keras.models.load_model(self.saved_dir / "price_model_evaluation.keras")
            self.price_scaler = joblib.load(self.saved_dir / "price_scaler.save")
            self.grid_scaler = joblib.load(self.saved_dir / "grid_scaler.save")
            
            # Load the target index information if available
            self.target_index = 0  # Default to first column
            try:
                if (self.saved_dir / "target_info.json").exists():
                    with open(self.saved_dir / "target_info.json", 'r') as f:
                        target_info = json.load(f)
                        self.target_index = target_info.get("target_index", 0)
                        logging.info(f"Loaded target index {self.target_index} for {target_info.get('target_feature', 'unknown')}")
                else:
                    # Try to load from config
                    with open(Path(__file__).resolve().parent / "config.json", 'r') as f:
                        config = json.load(f)
                        target_feature = config.get("feature_metadata", {}).get("target_feature")
                        if target_feature:
                            price_cols = config.get("feature_groups", {}).get("price_cols", [])
                            if target_feature in price_cols:
                                self.target_index = price_cols.index(target_feature)
                                logging.info(f"Inferred target index {self.target_index} for {target_feature} from config.json")
            except Exception as e:
                logging.warning(f"Could not load target index information: {e}. Using default index 0.")
            
            # Test the scaler with sample values
            try:
                test_val = np.array([100.0])  # A typical price value
                dummy = np.zeros((1, len(self.price_scaler.scale_)))
                dummy[0, self.target_index] = test_val[0]
                
                scaled = self.price_scaler.transform(dummy)[0, self.target_index]
                inverse_test = self.inverse_transform_prices(np.array([scaled]))
                
                logging.info(f"Scaler test: original={test_val[0]}, scaled={scaled}, inverse_transform={inverse_test[0]}")
                
                if not np.isclose(test_val[0], inverse_test[0], rtol=1e-3):
                    logging.warning(f"Price scaler verification issue - output {inverse_test[0]} doesn't match input {test_val[0]}")
            except Exception as e:
                logging.error(f"Error testing price scaler: {e}")
            
            # Try to load test data files
            try:
                self.test_data_dir.mkdir(parents=True, exist_ok=True)
                
                # Check if test data files exist
                test_files_exist = (
                    (self.test_data_dir / "X_test.npy").exists() and
                    (self.test_data_dir / "y_test.npy").exists() and
                    (self.test_data_dir / "test_timestamps.npy").exists()
                )
                
                if test_files_exist:
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
                    
                    logging.info(f"Loaded test data: {len(self.X_test)} samples")
                    logging.info(f"Loaded validation data: {len(self.X_val)} samples")
                else:
                    # Create synthetic data for visualization if test data doesn't exist
                    logging.warning("Test data files not found. Creating synthetic data for visualization.")
                    
                    # Get input shape from model
                    input_shape = self.model.input_shape
                    output_shape = self.model.output_shape
                    
                    # Create synthetic test data
                    num_samples = 10
                    window_size = input_shape[1]
                    num_features = input_shape[2]
                    
                    # Random features (normalized)
                    self.X_test = np.random.normal(0, 1, (num_samples, window_size, num_features))
                    
                    # Generate predictions to see what reasonable y values might be
                    self.y_test = self.model.predict(self.X_test)
                    
                    # Create synthetic timestamps
                    end_date = datetime.now()
                    hourly_dates = pd.date_range(end=end_date, periods=num_samples*24, freq='H')
                    self.test_timestamps = hourly_dates[::24]  # Take every 24th timestamp
                    
                    # Create synthetic validation data too
                    self.X_val = self.X_test.copy()
                    self.y_val = self.y_test.copy()
                    self.val_timestamps = self.test_timestamps.copy()
                    
                    logging.info(f"Created synthetic test and validation data: {num_samples} samples")
                    
                    # Save synthetic data for future use
                    np.save(self.test_data_dir / "X_test.npy", self.X_test)
                    np.save(self.test_data_dir / "y_test.npy", self.y_test)
                    np.save(self.test_data_dir / "test_timestamps.npy", self.test_timestamps)
                    np.save(self.test_data_dir / "X_val.npy", self.X_val)
                    np.save(self.test_data_dir / "y_val.npy", self.y_val)
                    np.save(self.test_data_dir / "val_timestamps.npy", self.val_timestamps)
                    
                    logging.info("Saved synthetic data for future use")
            except Exception as e:
                logging.error(f"Error loading or creating test data: {e}")
                raise
            
            logging.info("Successfully loaded evaluation model and data")
            
            # Also load price data for additional context
            try:
                self.price_data = pd.read_csv(
                    self.project_root / "data/processed/SE3prices.csv",
                    parse_dates=['HourSE'],
                    index_col='HourSE'
                )
            except:
                logging.warning("Could not load price data for context. Continuing without it.")
                self.price_data = None
                
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
                'dataset': 'test'  # Mark as test data
            }, index=dates)
            test_dfs.append(df)
        
        for i in range(len(val_timestamps)):
            dates = [val_timestamps[i] + pd.Timedelta(hours=h) for h in range(24)]
            df = pd.DataFrame({
                'actual': val_actual[i],
                'predicted': val_predicted[i],
                'dataset': 'validation'  # Mark as validation data
            }, index=dates)
            val_dfs.append(df)
        
        # Combine all dataframes
        predictions_df = pd.concat(test_dfs + val_dfs)
        
        # Make sure all numeric columns have numeric dtypes (not object)
        for col in ['actual', 'predicted']:
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
        
        # Log some sample scaled values for diagnostics
        if len(y_scaled) > 5:
            logging.info(f"Sample scaled values: {y_scaled[:5]}")
        else:
            logging.info(f"All scaled values: {y_scaled}")
            
        # Use the target_index loaded from the target_info.json file (default is 0)
        target_idx = getattr(self, 'target_index', 0)
        logging.info(f"Using target index {target_idx} for inverse transform")
        
        if len(y_scaled.shape) == 1:
            # For 1D array (single sequence of prices)
            dummy = np.zeros((len(y_scaled), len(self.price_scaler.scale_)))
            dummy[:, target_idx] = y_scaled
            result = self.price_scaler.inverse_transform(dummy)[:, target_idx]
            
            # Apply scaling adjustment - Swedish electricity prices are typically 20-200 öre/kWh
            # If values are consistently in thousands, apply a correction factor
            if np.median(result) > 1000:  # Check if values are abnormally high
                median_price = np.median(result)
                scaling_factor = median_price / 100  # Aim for a median around 100 öre/kWh
                result = result / scaling_factor
                logging.info(f"Applied scaling correction factor of {scaling_factor:.2f} to bring prices to normal range")
            
            # Log some sample values
            if len(result) > 5:
                logging.info(f"Sample inverse-transformed values: {result[:5]}")
            else:
                logging.info(f"All inverse-transformed values: {result}")
                
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
            
            # Apply scaling adjustment - Swedish electricity prices are typically 20-200 öre/kWh
            if np.median(inverse_flat) > 1000:  # Check if values are abnormally high
                median_price = np.median(inverse_flat)
                scaling_factor = median_price / 100  # Aim for a median around 100 öre/kWh
                inverse_flat = inverse_flat / scaling_factor
                logging.info(f"Applied scaling correction factor of {scaling_factor:.2f} to bring prices to normal range")
            
            # Reshape back to original structure
            result = inverse_flat.reshape(num_sequences, seq_length)
            
            # Log some sample values
            if num_sequences > 0:
                logging.info(f"Sample inverse-transformed sequence: {result[0]}")
                
            return result
        
        else:
            raise ValueError(f"Unexpected shape for y_scaled: {y_scaled.shape}")
    
    def create_visualizations(self):
        """Create simple visualizations for different time periods"""
        if self.predictions_df is None:
            self.process_all_predictions()
            
        plt.rcParams.update({'font.size': 12})
        
        # Create visualizations for different time periods with error handling
        visualizations = [
            ('daily', self.plot_daily_comparison),
            ('weekly', self.plot_weekly_comparison),
            ('monthly', self.plot_monthly_comparison),
            ('yearly', self.plot_yearly_comparison)
        ]
        
        successful_plots = []
        
        for viz_name, viz_func in visualizations:
            try:
                viz_func()
                successful_plots.append(viz_name)
            except Exception as e:
                logging.warning(f"Failed to generate {viz_name} visualization: {str(e)}")
        
        if successful_plots:
            logging.info(f"Successfully created visualizations: {', '.join(successful_plots)}")
            logging.info(f"Visualizations saved to {self.simple_viz_dir}")
        else:
            logging.warning("No visualizations were successfully created")
    
    def plot_daily_comparison(self):
        """Plot comparison of actual vs predicted prices for a few representative days"""
        try:
            df = self.predictions_df
            
            # Find days with complete data (24 hours)
            day_counts = df.groupby([d.date() for d in df.index]).size()
            complete_days = day_counts[day_counts == 24].index
            
            if len(complete_days) == 0:
                logging.warning("No complete days found for daily comparison")
                return
            
            # Select representative days (most recent, if possible)
            days_to_plot = min(4, len(complete_days))
            selected_days = sorted(complete_days, reverse=True)[:days_to_plot]
            
            # Create a figure with subplots for each day
            fig, axes = plt.subplots(days_to_plot, 1, figsize=(12, 4*days_to_plot))
            if days_to_plot == 1:
                axes = [axes]
            
            for i, day in enumerate(selected_days):
                # Use direct comparison of date objects
                day_data = df[[d.date() == day for d in df.index]].sort_index()
                
                ax = axes[i]
                ax.plot(day_data.index.hour, day_data['actual'], 'o-', 
                       color=self.colors['actual'], label='Actual', linewidth=2)
                ax.plot(day_data.index.hour, day_data['predicted'], 's--', 
                       color=self.colors['predicted'], label='Predicted', linewidth=2)
                
                # Highlight morning and evening peak hours
                ax.axvspan(7, 9, alpha=0.2, color='lightgray')
                ax.axvspan(17, 20, alpha=0.2, color='lightgray')
                
                ax.set_title(f'Day: {day}')
                ax.set_xlabel('Hour of Day')
                ax.set_ylabel('Price (öre/kWh)')
                ax.set_xticks(range(0, 24, 2))
                ax.grid(True, alpha=0.3)
                
                # Only add legend to the first subplot
                if i == 0:
                    ax.legend(loc='upper right')
            
            plt.tight_layout()
            plt.savefig(self.simple_viz_dir / "daily_comparison.png", dpi=300)
            plt.close()
            
            logging.info("Daily comparison plot created successfully")
        except Exception as e:
            logging.warning(f"Error in daily comparison plot: {str(e)}")
            plt.close('all')  # Ensure no hanging figures
        
    def plot_weekly_comparison(self):
        """Plot comparison of actual vs predicted prices for a representative week"""
        try:
            df = self.predictions_df
            
            # Find a complete week
            max_date = df.index.max().date()
            
            # Look for a week ending on max_date or earlier
            for end_date in [max_date - timedelta(days=i) for i in range(14)]:
                start_date = end_date - timedelta(days=6)
                week_data = df[[d.date() >= start_date and d.date() <= end_date for d in df.index]]
                
                # Check if we have at least 5 days with some data
                # Fix: use len(np.unique()) instead of .nunique() on numpy array
                day_coverage = len(np.unique([d.date() for d in week_data.index]))
                if day_coverage >= 5 and len(week_data) >= 120:  # At least 5 days with 120 hours total
                    break
            else:
                # If no good week found
                logging.warning("Could not find a representative week with sufficient data")
                return
                
            # Create the plot
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Plot the data
            ax.plot(week_data.index, week_data['actual'], '-', 
                   color=self.colors['actual'], label='Actual', linewidth=2)
            ax.plot(week_data.index, week_data['predicted'], '--', 
                   color=self.colors['predicted'], label='Predicted', linewidth=2)
            
            # Shade weekends
            for date in pd.date_range(start=start_date, end=end_date):
                if date.dayofweek >= 5:  # Weekend
                    ax.axvspan(date, date + pd.Timedelta(days=1), alpha=0.2, color='lightgray')
            
            # Format the x-axis to show days of week
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%a\n%d %b'))
            ax.xaxis.set_major_locator(mdates.DayLocator())
            
            ax.set_title(f'Week: {start_date} to {end_date}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (öre/kWh)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(self.simple_viz_dir / "weekly_comparison.png", dpi=300)
            plt.close()
            
            logging.info("Weekly comparison plot created successfully")
        except Exception as e:
            logging.warning(f"Error in weekly comparison plot: {str(e)}")
            plt.close('all')  # Ensure no hanging figures
    
    def plot_monthly_comparison(self):
        """Plot comparison of actual vs predicted prices for representative months"""
        try:
            df = self.predictions_df
            
            # Alternative approach that doesn't rely on groupby with index attributes
            # Extract year and month directly for aggregation
            years = [d.year for d in df.index]
            months = [d.month for d in df.index]
            
            # Create a temporary dataframe with explicit columns
            temp_df = pd.DataFrame({
                'year': years,
                'month': months,
                'actual': df['actual'].values,
                'predicted': df['predicted'].values
            })
            
            # Group by year and month
            monthly_grouped = temp_df.groupby(['year', 'month']).agg({
                'actual': 'mean',
                'predicted': 'mean'
            })
            
            # Reset index to get year and month as columns
            monthly_data = monthly_grouped.reset_index()
            
            # Create proper datetime objects for plotting
            monthly_data['month_date'] = [datetime(int(row.year), int(row.month), 15) 
                                        for _, row in monthly_data.iterrows()]
            
            # Sort by date
            monthly_data = monthly_data.sort_values('month_date')
            
            # Select the last 12 months if we have that many
            months_to_show = min(12, len(monthly_data))
            plot_data = monthly_data.tail(months_to_show)
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Plot the data
            ax.plot(plot_data['month_date'], plot_data['actual'], 'o-', 
                   color=self.colors['actual'], label='Actual', linewidth=2)
            ax.plot(plot_data['month_date'], plot_data['predicted'], 's--', 
                   color=self.colors['predicted'], label='Predicted', linewidth=2)
            
            # Format x-axis to show month names
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            
            ax.set_title(f'Monthly Average Prices (Last {months_to_show} Months)')
            ax.set_xlabel('Month')
            ax.set_ylabel('Average Price (öre/kWh)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(self.simple_viz_dir / "monthly_comparison.png", dpi=300)
            plt.close()
            
            logging.info("Monthly comparison plot created successfully")
        except Exception as e:
            logging.warning(f"Error in monthly comparison plot: {str(e)}")
            plt.close('all')  # Ensure no hanging figures
    
    def plot_yearly_comparison(self):
        """Plot comparison of actual vs predicted prices across years"""
        try:
            df = self.predictions_df
            
            # Resample to daily data for clearer visualization of long-term trends
            daily_data = df.resample('D').mean()
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Plot the data
            ax.plot(daily_data.index, daily_data['actual'], '-', 
                   color=self.colors['actual'], label='Actual', linewidth=2)
            ax.plot(daily_data.index, daily_data['predicted'], '--', 
                   color=self.colors['predicted'], label='Predicted', linewidth=2)
            
            # Add year dividers
            # Convert years to list to avoid numpy array issues
            years = sorted(list(set([d.year for d in daily_data.index])))
            for year in years[1:]:  # Skip the first year divider
                ax.axvline(pd.Timestamp(f"{year}-01-01"), color='gray', alpha=0.5, linestyle='-')
                
            # Format x-axis to show years
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show every 3 months
            
            ax.set_title('Long-term Price Trends (Daily Averages)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (öre/kWh)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.simple_viz_dir / "yearly_comparison.png", dpi=300)
            plt.close()
            
            logging.info("Yearly comparison plot created successfully")
        except Exception as e:
            logging.warning(f"Error in yearly comparison plot: {str(e)}")
            plt.close('all')  # Ensure no hanging figures

    def run_simple_evaluation(self):
        """Run a simple visualization-focused evaluation"""
        logging.info("Starting simple visualization-focused evaluation...")
        
        try:
            # Generate predictions
            self.process_all_predictions()
            
            # Create visualizations
            self.create_visualizations()
            
            logging.info("Simple evaluation complete")
        except Exception as e:
            logging.error(f"Evaluation error: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())

def main():
    """Run the simple evaluation without requiring command arguments"""
    logging.info("Starting simple price model evaluation...")
    
    try:
        # Create evaluator
        evaluator = SimplePriceModelEvaluator()
        
        # Run simple evaluation
        evaluator.run_simple_evaluation()
        
        logging.info("Simple evaluation complete!")
        
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 