import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        self.models_dir = self.project_root / "models/saved"
        self.test_data_dir = self.project_root / "models/test_data"
        
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
        
        # Determine model and data paths based on training mode
        model_suffix = "_production" if self.train_from_all_data else ""
        scaler_suffix = "_production" if self.train_from_all_data else ""
        
        # Save scalers
        self.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.price_scaler, self.models_dir / f'price_scaler{scaler_suffix}.save')
        joblib.dump(self.grid_scaler, self.models_dir / f'grid_scaler{scaler_suffix}.save')
        
        if not self.train_from_all_data:
            # Save test data for evaluation
            self.test_data_dir.mkdir(parents=True, exist_ok=True)
            train_split = int(len(timestamps) * self.feature_config.data_split["train_ratio"])
            val_split = int(len(timestamps) * (
                self.feature_config.data_split["train_ratio"] + 
                self.feature_config.data_split["val_ratio"]
            ))
            
            np.save(self.test_data_dir / 'X_test.npy', X_test)
            np.save(self.test_data_dir / 'y_test.npy', y_test)
            np.save(self.test_data_dir / 'test_timestamps.npy', timestamps[val_split:])
        
        logging.info("Building model...")
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Configure callbacks
        model_checkpoint_path = self.models_dir / f'price_model{model_suffix}.keras'
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
            )
        ]
        
        logging.info(f"Training {'production' if self.train_from_all_data else 'evaluation'} model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot training history
        self.plot_training_history(history)
        
        return history

    
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(15, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # # Plot MAE
        # plt.subplot(1, 2, 2)
        # plt.plot(history.history['mae'], label='Training MAE')
        # plt.plot(history.history['val_mae'], label='Validation MAE')
        # plt.title('Model MAE')
        # plt.xlabel('Epoch')
        # plt.ylabel('MAE')
        # plt.legend()
        # plt.grid(True)
        
        # plt.tight_layout()
        plt.show()

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


