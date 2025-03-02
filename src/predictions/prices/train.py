import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
import joblib
from pathlib import Path
import logging
import os
import sys
import argparse
import datetime

# Configure TensorFlow to use available CPU instructions
try:
    tf.config.optimizer.set_jit(True)  # Enable XLA optimization
except:
    logging.warning("Could not enable XLA optimization")

# Add the project root to the Python path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.predictions.prices.gather_data import FeatureConfig
feature_config = FeatureConfig()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PriceModelTrainer:
    def __init__(self, window_size=None, scaler_type="minmax", train_from_all_data=False):
        """Initialize trainer with project paths and scaler type
        
        Args:
            window_size (int): Size of the input window
            scaler_type (str): Type of scaler to use ('minmax', 'standard', 'robust')
                - minmax: MinMaxScaler(feature_range=(-1, 1)), preserves relative magnitudes
                - standard: StandardScaler(), normalizes to mean=0, std=1
                - robust: RobustScaler(), scales using statistics robust to outliers
            train_from_all_data (bool): If True, uses all data for training without test split
        """
        # Load feature configuration
        self.feature_config = feature_config
        
        # Get window size from config if not provided
        self.window_size = window_size or self.feature_config.training["window_size"]
        self.model = None
        self.train_from_all_data = train_from_all_data
        
        # Setup project paths
        self.project_root = Path(__file__).resolve().parents[3]
        
        # Model directory paths
        # Models directory structure:
        # models/
        #   ├── production/  - Production model (trained on all data)
        #   │   ├── saved/   - Model files and scalers
        #   │   └── logs/    - TensorBoard logs
        #   └── evaluation/  - Evaluation model (with test split)
        #       ├── saved/   - Model files and scalers
        #       ├── logs/    - TensorBoard logs
        #       └── test_data/ - Test data for evaluation
        self.models_dir = Path(__file__).resolve().parent / "models"
        
        # Create specific paths based on training mode
        if self.train_from_all_data:
            # Production model paths
            self.model_dir = self.models_dir / "production"
            self.saved_dir = self.model_dir / "saved"
            self.logs_dir = self.model_dir / "logs"
            self.model_name = "price_model_production.keras"
        else:
            # Evaluation model paths
            self.model_dir = self.models_dir / "evaluation"
            self.saved_dir = self.model_dir / "saved"
            self.logs_dir = self.model_dir / "logs"
            self.test_data_dir = self.model_dir / "test_data"
            self.model_name = "price_model_evaluation.keras"
        
        # Create directories
        self.saved_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if not self.train_from_all_data:
            self.test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize scalers based on type
        self.scaler_type = scaler_type
        self._initialize_scalers()
    
    def _initialize_scalers(self):
        """Initialize the appropriate scalers based on scaler_type"""
        if self.scaler_type == "minmax":
            self.price_scaler = MinMaxScaler(feature_range=(-1, 1))  # Symmetric range for price variations
            self.grid_scaler = MinMaxScaler(feature_range=(-1, 1))
            logging.info("Using MinMaxScaler: Better preserves relative magnitudes of price spikes")
        elif self.scaler_type == "standard":
            self.price_scaler = StandardScaler()
            self.grid_scaler = StandardScaler()
            logging.info("Using StandardScaler: Normalizes features to mean=0, std=1")
        elif self.scaler_type == "robust":
            self.price_scaler = RobustScaler(quantile_range=(1, 99))
            self.grid_scaler = RobustScaler()
            logging.info("Using RobustScaler: Scales using statistics that are robust to outliers")
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}. Must be one of: minmax, standard, robust")
    
    def get_required_features(self):
        """List all required features for the model"""
        return self.feature_config.get_ordered_features()
    
    def load_data(self):
        """Load and prepare the processed data"""
        logging.info("Loading datasets...")
        
        # Load price data first
        price_df = pd.read_csv(
            self.project_root / "data/processed/SE3prices.csv", 
            index_col=0
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
        
        # Load grid features
        grid_df = pd.read_csv(
            self.project_root / "data/processed/SwedenGrid.csv",
            index_col=0
        )
        grid_df.index = pd.to_datetime(grid_df.index)
        
        # Get common date range where all data is available
        start_date = max(
            price_df.index.min(),
            time_df.index.min(),
            holiday_df.index.min(),
            grid_df.index.min()
        )
        
        # For production model, use all available data up to the latest date
        end_date = min(
            price_df.index.max(),
            time_df.index.max(),
            holiday_df.index.max(),
            grid_df.index.max()
        )
        
        logging.info(f"Using data from {start_date} to {end_date}")
        
        # Trim all dataframes to common date range
        price_df = price_df[
            (price_df.index >= start_date)
        ]
        
        time_df = time_df[
            (time_df.index >= start_date)
        ]
        
        holiday_df = holiday_df[
            (holiday_df.index >= start_date)
        ]
        
        grid_df = grid_df[
            (grid_df.index >= start_date)
        ]
        
        # Merge all features
        df = price_df.join(time_df, how='left')
        df = df.join(holiday_df, how='left')
        df = df.join(grid_df, how='left')
        
        # Handle any missing values
        df = df.ffill().bfill()
        
        logging.info(f"Final dataset shape: {df.shape}")
        
        return df

    def create_sequences(self, data):
        """Convert dataframe of features to sequences for LSTM training"""
        X, y = [], []
        values = data.values
        for i in range(len(values) - self.window_size - 24):  # Need 24 more steps for targets
            X.append(values[i:(i + self.window_size)])
            # Get next 24 hours of prices as target
            y.append(values[i + self.window_size:i + self.window_size + 24, 0])  # First column is price
        return np.array(X), np.array(y)
    
    def prepare_data(self):
        """Prepare data for training"""
        df = self.load_data()
        
        # Get required features from config
        feature_cols = self.get_required_features()
        
        # Verify all required features exist
        missing_cols = self.feature_config.verify_features(df.columns)
        if missing_cols:
            logging.error("Missing features:")
            for col in missing_cols:
                logging.error(f"  - {col}")
                logging.error(f"Available columns: {sorted(df.columns.tolist())}")
            raise ValueError(f"Missing required features: {missing_cols}")
        
        # Scale price features together (including target)
        price_cols = self.feature_config.price_cols
        price_data = df[price_cols].values
        scaled_prices = self.price_scaler.fit_transform(price_data)
        
        # Log price scaling statistics
        if self.scaler_type != "robust":
            logging.info("\nPrice Scaling Statistics:")
            for i, col in enumerate(price_cols):
                orig_std = np.std(price_data[:, i])
                scaled_std = np.std(scaled_prices[:, i])
                logging.info(f"{col}:")
                logging.info(f"  Original std: {orig_std:.2f}")
                logging.info(f"  Scaled std: {scaled_std:.2f}")
                logging.info(f"  Scale factor: {scaled_std/orig_std:.2f}")
        
        for i, col in enumerate(price_cols):
            df[col] = scaled_prices[:, i]
        
        # Scale grid features together
        grid_cols = self.feature_config.grid_cols
        grid_data = df[grid_cols].values
        scaled_grid = self.grid_scaler.fit_transform(grid_data)
        for i, col in enumerate(grid_cols):
            df[col] = scaled_grid[:, i]
        
        # Binary and cyclical features are left as is - no scaling needed
        # Log feature statistics after scaling
        logging.info("\nFeature Statistics after scaling:")
        for col in feature_cols:
            stats = df[col].describe()
            logging.info(f"{col}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
                        f"min={stats['min']:.3f}, max={stats['max']:.3f}")
        
        # Create sequences with ordered features
        X, y = self.create_sequences(df[feature_cols])
        
        if self.train_from_all_data:
            logging.info("\nUsing all data for production model training")
            # Use all data for training, keeping a small validation set for monitoring
            val_size = min(int(len(X) * 0.05), 5000)  # Use 5% or max 5000 samples for validation
            
            # Take the validation set from the middle of the dataset to avoid recent data
            mid_point = len(X) // 2
            val_start = mid_point - val_size // 2
            val_end = val_start + val_size
            
            # Create validation set from middle
            X_val = X[val_start:val_end]
            y_val = y[val_start:val_end]
            
            # Create training set from remaining data
            X_train = np.concatenate([X[:val_start], X[val_end:]], axis=0)
            y_train = np.concatenate([y[:val_start], y[val_end:]], axis=0)
            
            logging.info(f"Total samples: {len(X)}")
            logging.info(f"Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
            logging.info(f"Validation samples: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
            
            return (X_train, y_train, X_val, y_val, None, None), df.index[self.window_size + 24:]
        else:
            # Get data split ratios from config for evaluation model
            split_ratios = self.feature_config.get_data_split_ratios()
            train_ratio = split_ratios["train_ratio"]
            val_ratio = split_ratios["val_ratio"]
            
            # Calculate split indices
            total_samples = len(X)
            train_split = int(total_samples * train_ratio)
            val_split = int(total_samples * (train_ratio + val_ratio))
            
            # Log split information
            logging.info("\nData Split Information for Evaluation Model:")
            logging.info(f"Total samples: {total_samples}")
            logging.info(f"Training samples: {train_split} ({train_ratio*100:.1f}%)")
            logging.info(f"Validation samples: {val_split - train_split} ({val_ratio*100:.1f}%)")
            logging.info(f"Test samples: {total_samples - val_split} ({split_ratios['test_ratio']*100:.1f}%)")
            
            # Split the data
            X_train = X[:train_split]
            y_train = y[:train_split]
            X_val = X[train_split:val_split]
            y_val = y[train_split:val_split]
            X_test = X[val_split:]
            y_test = y[val_split:]
            
            return (X_train, y_train, X_val, y_val, X_test, y_test), df.index[self.window_size + 24:]
    
    def build_model(self, input_shape):
        """Build LSTM model using Keras Functional API for better architecture"""
        # Input layer with explicit shape
        inputs = Input(shape=(input_shape[0], input_shape[1]), name='sequence_input')
        
        # Get architecture config
        lstm_layers = self.feature_config.architecture["lstm_layers"]
        dense_layers = self.feature_config.architecture["dense_layers"]
        
        # Build LSTM layers
        x = inputs
        for i, layer in enumerate(lstm_layers):
            return_sequences = layer["return_sequences"]
            dropout_rate = layer.get("dropout", 0)
            
            # Add LSTM layer
            lstm_layer = LSTM(
                units=layer["units"],
                return_sequences=return_sequences,
                activation='tanh',
                name=f'lstm_{i+1}'
            )(x)
            
            # Add dropout if specified
            if dropout_rate > 0:
                lstm_layer = Dropout(dropout_rate, name=f'lstm_dropout_{i+1}')(lstm_layer)
            
            x = lstm_layer
        
        # Add Dense layers
        for i, layer in enumerate(dense_layers[:-1]):
            x = Dense(
                units=layer["units"],
                activation='relu',
                name=f'dense_{i+1}'
            )(x)
        
        # Final output layer
        outputs = Dense(
            units=dense_layers[-1]["units"],
            activation=None,  # Linear activation for price prediction
            name='price_output'
        )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='price_prediction_model')
        
        # Get training parameters
        training_params = self.feature_config.get_training_params()
        
        # Configure optimizer with gradient clipping
        if training_params["optimizer"].lower() == "adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=training_params["learning_rate"],
                clipnorm=1.0  # Add gradient clipping to prevent exploding gradients
            )
        
        # Use Huber loss for better handling of outliers
        huber_loss = tf.keras.losses.Huber(delta=0.5)
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=huber_loss,
            metrics=training_params["metrics"]
        )
        
        # Print model summary
        model.summary()
        
        return model
    
    def train(self, epochs=None, batch_size=None):
        """Train the model using config parameters"""
        logging.info("Preparing data...")
        (X_train, y_train, X_val, y_val, X_test, y_test), timestamps = self.prepare_data()
        
        # Get training parameters from config
        training_params = self.feature_config.get_training_params()
        epochs = epochs or training_params["max_epochs"]
        batch_size = batch_size or training_params["batch_size"]
        
        # Get callback parameters
        callback_params = self.feature_config.get_callback_params()
        
        # Save scalers
        joblib.dump(self.price_scaler, self.saved_dir / 'price_scaler.save')
        joblib.dump(self.grid_scaler, self.saved_dir / 'grid_scaler.save')
        
        if not self.train_from_all_data:
            # Save test data for evaluation
            train_split = int(len(timestamps) * self.feature_config.data_split["train_ratio"])
            val_split = int(len(timestamps) * (
                self.feature_config.data_split["train_ratio"] + 
                self.feature_config.data_split["val_ratio"]
            ))
            
            np.save(self.test_data_dir / 'X_test.npy', X_test)
            np.save(self.test_data_dir / 'y_test.npy', y_test)
            np.save(self.test_data_dir / 'test_timestamps.npy', timestamps[val_split:])
            
            # Save validation data for quick evaluation later
            np.save(self.test_data_dir / 'X_val.npy', X_val)
            np.save(self.test_data_dir / 'y_val.npy', y_val)
            np.save(self.test_data_dir / 'val_timestamps.npy', timestamps[train_split:val_split])
        
        logging.info("Building model...")
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Configure callbacks
        model_checkpoint_path = self.saved_dir / self.model_name
        log_dir = self.logs_dir / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        callbacks = [
            EarlyStopping(
                monitor=callback_params["early_stopping"]["monitor"],
                patience=callback_params["early_stopping"]["patience"],
                restore_best_weights=callback_params["early_stopping"]["restore_best_weights"],
                mode='min'
            ),
            ModelCheckpoint(
                model_checkpoint_path,
                monitor=callback_params["early_stopping"]["monitor"],
                save_best_only=True,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor=callback_params["reduce_lr"]["monitor"],
                factor=callback_params["reduce_lr"]["factor"],
                patience=callback_params["reduce_lr"]["patience"],
                min_lr=callback_params["reduce_lr"]["min_lr"],
                mode='min'
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True
            )
        ]
        
        model_type = 'production' if self.train_from_all_data else 'evaluation'
        logging.info(f"Training {model_type} model...")
        logging.info(f"Model will be saved to: {model_checkpoint_path}")
        logging.info(f"TensorBoard logs: {log_dir}")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot training history
        self.plot_training_history(history, model_type)
        
        # Save the history for later reference
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(self.saved_dir / f'training_history.csv')
        
        # If it's an evaluation model, run a quick evaluation
        if not self.train_from_all_data and X_test is not None:
            self.quick_evaluation(X_test, y_test)
        
        return history
    
    def quick_evaluation(self, X_test, y_test):
        """Run a quick evaluation on test data and save results"""
        logging.info("\nRunning quick evaluation on test data...")
        
        # Predict on test data
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        test_loss = self.model.evaluate(X_test, y_test, verbose=0)
        test_mae = test_loss[1] if len(test_loss) > 1 else None
        
        # Set consistent style for plots
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create a 2x2 dashboard with key evaluation insights
        fig = plt.figure(figsize=(16, 14), dpi=100)
        
        # Define a consistent color scheme
        colors = {
            'actual': '#1F77B4',    # Blue
            'predicted': '#FF7F0E', # Orange
            'error': '#D62728',     # Red
            'grid': '#BDBDBD'       # Light gray
        }
        
        # 1. Top left: Representative sample with prediction
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        
        # Get a representative sample from middle of test set
        sample_idx = len(y_test) // 2  # Middle sample is often more representative
        true_vals = y_test[sample_idx]
        pred_vals = y_pred[sample_idx]
        
        # Inverse transform the values for realistic prices
        true_dummy = np.zeros((len(true_vals), len(self.price_scaler.scale_)))
        true_dummy[:, 0] = true_vals
        pred_dummy = np.zeros((len(pred_vals), len(self.price_scaler.scale_)))
        pred_dummy[:, 0] = pred_vals
        
        true_inv = self.price_scaler.inverse_transform(true_dummy)[:, 0]
        pred_inv = self.price_scaler.inverse_transform(pred_dummy)[:, 0]
        
        # Calculate error for this sample
        error = np.abs(true_inv - pred_inv)
        
        # Plot sample with error band
        hours = range(len(true_vals))
        ax1.plot(hours, true_inv, '-', color=colors['actual'], label='Actual', linewidth=2.5)
        ax1.plot(hours, pred_inv, '--', color=colors['predicted'], label='Predicted', linewidth=2.5)
        
        # Add error shading
        ax1.fill_between(hours, pred_inv - error/2, pred_inv + error/2, color=colors['predicted'], alpha=0.2)
        
        # Format plot
        ax1.set_title('Representative 24-Hour Prediction Sample', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Hour of Day', fontsize=12)
        ax1.set_ylabel('Price (öre/kWh)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
        ax1.legend(fontsize=10)
        
        # Add metrics annotation for this sample
        sample_mae = np.mean(error)
        sample_mape = np.mean(np.abs(error / (true_inv + 1e-8))) * 100
        metrics_text = f"Sample MAE: {sample_mae:.2f} öre/kWh\nSample MAPE: {sample_mape:.2f}%"
        ax1.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        # 2. Top right: Error distribution histogram
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        
        # Calculate all errors across test set
        all_errors = []
        for i in range(min(len(y_test), 100)):  # Use up to 100 samples for efficiency
            true_sample = y_test[i]
            pred_sample = y_pred[i]
            
            # Inverse transform
            true_dummy = np.zeros((len(true_sample), len(self.price_scaler.scale_)))
            true_dummy[:, 0] = true_sample
            pred_dummy = np.zeros((len(pred_sample), len(self.price_scaler.scale_)))
            pred_dummy[:, 0] = pred_sample
            
            true_inv = self.price_scaler.inverse_transform(true_dummy)[:, 0]
            pred_inv = self.price_scaler.inverse_transform(pred_dummy)[:, 0]
            
            # Add errors to list
            all_errors.extend(np.abs(true_inv - pred_inv))
        
        # Plot error distribution
        ax2.hist(all_errors, bins=30, color=colors['error'], alpha=0.7)
        ax2.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Absolute Error (öre/kWh)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
        
        # Add distribution statistics
        mean_error = np.mean(all_errors)
        median_error = np.median(all_errors)
        p90_error = np.percentile(all_errors, 90)
        stats_text = f"Mean Error: {mean_error:.2f}\nMedian Error: {median_error:.2f}\n90th Percentile: {p90_error:.2f}"
        ax2.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction', 
                    ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        # 3. Bottom left: Hourly pattern analysis
        ax3 = plt.subplot2grid((2, 2), (1, 0))
        
        # Analyze error patterns by hour of day
        hourly_errors = [[] for _ in range(24)]
        for i in range(min(len(y_test), 100)):  # Use up to 100 samples
            true_sample = y_test[i]
            pred_sample = y_pred[i]
            
            # Inverse transform
            true_dummy = np.zeros((len(true_sample), len(self.price_scaler.scale_)))
            true_dummy[:, 0] = true_sample
            pred_dummy = np.zeros((len(pred_sample), len(self.price_scaler.scale_)))
            pred_dummy[:, 0] = pred_sample
            
            true_inv = self.price_scaler.inverse_transform(true_dummy)[:, 0]
            pred_inv = self.price_scaler.inverse_transform(pred_dummy)[:, 0]
            
            # Group errors by hour
            for h in range(min(24, len(true_inv))):
                hourly_errors[h].append(abs(true_inv[h] - pred_inv[h]))
        
        # Calculate mean and std for each hour
        hourly_means = [np.mean(errors) for errors in hourly_errors if errors]
        hourly_stds = [np.std(errors) for errors in hourly_errors if errors]
        hours = range(len(hourly_means))
        
        # Plot hourly patterns
        ax3.bar(hours, hourly_means, yerr=hourly_stds, color=colors['error'], alpha=0.7, 
               error_kw=dict(ecolor='black', lw=1, capsize=5, capthick=1))
        
        # Highlight peak hours (typically morning and evening)
        peak_hours = [7, 8, 9, 17, 18, 19, 20]
        for hour in peak_hours:
            if hour < len(hourly_means):
                ax3.axvspan(hour-0.4, hour+0.4, color='yellow', alpha=0.2)
        
        ax3.set_title('Error by Hour of Day', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Hour', fontsize=12)
        ax3.set_ylabel('Mean Absolute Error (öre/kWh)', fontsize=12)
        ax3.set_xticks(hours)
        ax3.grid(True, linestyle='--', alpha=0.7, color=colors['grid'])
        
        # Annotate peak hours
        ax3.annotate('Morning Peak', xy=(8, ax3.get_ylim()[1]*0.95), 
                    ha='center', fontsize=9, 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.2))
        ax3.annotate('Evening Peak', xy=(18.5, ax3.get_ylim()[1]*0.95), 
                    ha='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.2))
        
        # 4. Bottom right: Overall metrics summary
        ax4 = plt.subplot2grid((2, 2), (1, 1))
        ax4.axis('off')
        
        # Calculate additional metrics
        mean_abs_error = np.mean(all_errors)
        median_abs_error = np.median(all_errors)
        p90_error = np.percentile(all_errors, 90)
        max_error = np.max(all_errors)
        
        # Create metrics summary
        metrics_summary = [
            f"Test Dataset Metrics:",
            f"",
            f"Mean Absolute Error: {mean_abs_error:.2f} öre/kWh",
            f"Median Absolute Error: {median_abs_error:.2f} öre/kWh",
            f"90th Percentile Error: {p90_error:.2f} öre/kWh",
            f"Maximum Error: {max_error:.2f} öre/kWh",
            f"",
            f"Model Loss: {test_loss[0]:.6f}",
        ]
        
        if test_mae is not None:
            metrics_summary.append(f"Model MAE (scaled): {test_mae:.6f}")
        
        # Add error pattern insights
        highest_error_hour = np.argmax(hourly_means)
        lowest_error_hour = np.argmin(hourly_means)
        
        insights = [
            f"",
            f"Key Insights:",
            f"",
            f"• Hardest to predict: Hour {highest_error_hour}:00 (MAE: {hourly_means[highest_error_hour]:.2f})",
            f"• Most accurate: Hour {lowest_error_hour}:00 (MAE: {hourly_means[lowest_error_hour]:.2f})",
        ]
        
        # Determine if morning or evening peaks have higher errors
        morning_peak_error = np.mean([hourly_means[h] for h in [7, 8, 9] if h < len(hourly_means)])
        evening_peak_error = np.mean([hourly_means[h] for h in [17, 18, 19, 20] if h < len(hourly_means)])
        
        if morning_peak_error > evening_peak_error:
            insights.append(f"• Morning peak hours are more challenging to predict")
        else:
            insights.append(f"• Evening peak hours are more challenging to predict")
        
        # Create a text box with all metrics
        ax4.text(0.5, 0.5, '\n'.join(metrics_summary + insights), 
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round,pad=1', facecolor='#f0f0f0', edgecolor='gray', alpha=0.9),
                transform=ax4.transAxes)
        
        # Add overall title
        plt.suptitle('Model Quick Evaluation Dashboard', fontsize=18, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save figure
        plt.savefig(self.saved_dir / 'quick_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics to file
        metrics = {
            'test_loss': test_loss[0],
            'test_mae': test_mae,
            'mean_abs_error': mean_abs_error,
            'median_abs_error': median_abs_error,
            'p90_error': p90_error,
            'max_error': max_error
        }
        
        # Save metrics to file
        with open(self.saved_dir / 'test_metrics.txt', 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        
        logging.info(f"Quick evaluation complete. Saved to quick_evaluation.png")
        logging.info(f"Test Loss: {test_loss[0]:.4f}")
        if test_mae is not None:
            logging.info(f"Test MAE: {test_mae:.4f}")
        logging.info(f"Mean Absolute Error: {mean_abs_error:.2f} öre/kWh")

    def plot_training_history(self, history, model_type=''):
        """Plot and save training history with improved styling"""
        # Create a larger figure with better resolution
        plt.figure(figsize=(16, 6), dpi=100)
        
        # Set a consistent style for better readability
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Plot loss with improved styling
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], linewidth=2.5, label='Training Loss', color='#2471A3')
        plt.plot(history.history['val_loss'], linewidth=2.5, label='Validation Loss', color='#E67E22', linestyle='--')
        plt.title(f'{model_type.capitalize()} Model - Loss', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Loss', fontsize=12, fontweight='bold')
        plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='gray')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Add minor gridlines for better readability
        plt.minorticks_on()
        plt.grid(True, which='minor', linestyle=':', alpha=0.4)
        
        # Plot MAE if available with consistent styling
        if 'mae' in history.history:
            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'], linewidth=2.5, label='Training MAE', color='#2471A3')
            plt.plot(history.history['val_mae'], linewidth=2.5, label='Validation MAE', color='#E67E22', linestyle='--')
            plt.title(f'{model_type.capitalize()} Model - MAE', fontsize=16, fontweight='bold', pad=15)
            plt.xlabel('Epoch', fontsize=12, fontweight='bold')
            plt.ylabel('MAE', fontsize=12, fontweight='bold')
            plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='gray')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add minor gridlines for better readability
            plt.minorticks_on()
            plt.grid(True, which='minor', linestyle=':', alpha=0.4)
        
        # Add a dataset text to show data splits
        if not self.train_from_all_data:
            train_ratio = self.feature_config.data_split["train_ratio"]
            val_ratio = self.feature_config.data_split["val_ratio"]
            test_ratio = self.feature_config.data_split["test_ratio"]
            plt.figtext(0.5, 0.01, 
                       f"Data split: Training {train_ratio*100:.1f}% | Validation {val_ratio*100:.1f}% | Test {test_ratio*100:.1f}%",
                       ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5, "edgecolor":"gray"})
        else:
            plt.figtext(0.5, 0.01, 
                       "Production model: Trained on all available data with small validation sample",
                       ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5, "edgecolor":"gray"})
        
        # Adjust layout and save with higher quality
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make room for the text at the bottom
        plt.savefig(self.saved_dir / f'{model_type}_training_history.png', dpi=300, bbox_inches='tight')
        
        # Additional informative plot: Learning rate if available
        if 'lr' in history.history:
            plt.figure(figsize=(10, 4))
            plt.plot(history.history['lr'], 'o-', color='#16A085', linewidth=2)
            plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Learning Rate', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.yscale('log')  # Log scale makes it easier to see learning rate decay
            plt.savefig(self.saved_dir / f'{model_type}_learning_rate.png', dpi=300, bbox_inches='tight')
            
        # Don't show plots during automated training
        plt.close('all')

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train price prediction model')
    parser.add_argument('mode', choices=['production', 'evaluation'], 
                        help='Training mode: production (uses all data) or evaluation (uses train/val/test split)')
    parser.add_argument('--scaler', default='robust', choices=['robust', 'minmax', 'standard'],
                        help='Type of scaler to use (default: robust)')
    args = parser.parse_args()

    # Set training mode based on argument
    train_from_all_data = args.mode == 'production'
    
    # Log the training mode
    logging.info(f"\n=== Training {args.mode.capitalize()} Model ===")
    logging.info(f"Using {args.scaler} scaler")
    logging.info(f"Training with {'all available' if train_from_all_data else 'train/val/test split'} data")
    
    # Initialize and train the model
    trainer = PriceModelTrainer(scaler_type=args.scaler, train_from_all_data=train_from_all_data)
    history = trainer.train()
    
    logging.info(f"{args.mode.capitalize()} model training complete!")

if __name__ == "__main__":
    main()


