import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PriceModelTrainer:
    def __init__(self, window_size=48):
        """Initialize trainer with project paths"""
        self.window_size = window_size
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Setup project paths
        self.project_root = Path(__file__).resolve().parents[3]  # Go up 3 levels to project root
        self.models_dir = self.project_root / "models/saved"
        self.test_data_dir = self.project_root / "models/test_data"
        
        
    def load_data(self):
        """Load and prepare the processed data"""
        logging.info("Loading datasets...")
        
        # Load price data first to establish the date range
        price_df = pd.read_csv(
            self.project_root / "data/processed/SE3prices.csv", 
            index_col=0
        )
        price_df.index = pd.to_datetime(price_df.index)
        
        # Load holiday features
        holidays_df = pd.read_csv(
            self.project_root / "data/processed/holidays.csv",
            index_col=0
        )
        holidays_df.index = pd.to_datetime(holidays_df.index)
        
        # Load grid features
        grid_df = pd.read_csv(
            self.project_root / "data/processed/SwedenGrid.csv",
            index_col=0
        )
        grid_df.index = pd.to_datetime(grid_df.index)
        
        # Get common date range
        start_date = price_df.index.min()
        end_date = price_df.index.max()
        
        logging.info(f"Using data from {start_date} to {end_date}")
        
        # Trim all dataframes to common date range
        holidays_df = holidays_df[
            (holidays_df.index >= start_date) & 
            (holidays_df.index <= end_date)
        ]
        
        grid_df = grid_df[
            (grid_df.index >= start_date) & 
            (grid_df.index <= end_date)
        ]
        
        # Merge all features
        df = price_df.join(holidays_df, how='left')
        df = df.join(grid_df, how='left')
        
        # Handle any missing values
        df = df.ffill().bfill()
        
        # Verify we have all required features
        missing_cols = set(self.get_required_features()) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required features: {missing_cols}")
        
        logging.info(f"Final dataset shape: {df.shape}")
        logging.info(f"Dataset columns: {df.columns}")
        logging.info(f"Dataset sample: {df.head()}")
        return df
    
    def get_required_features(self):
        """List all required features for the model"""
        # Base time features
        time_features = [
            'hour_sin', 'hour_cos',
            'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos',
            'is_peak_hour', 'is_weekend'
        ]
        
        # Holiday features
        holiday_features = [
            'is_holiday', 'is_holiday_eve',
            'days_to_next_holiday', 'days_from_last_holiday'
        ]
        
        # Grid supply features (match actual columns from SwedenGrid.csv)
        grid_features = [
            'renewable_percentage',
            'nuclear_percentage', 
            'thermal_percentage',
            'import_percentage',
            'total_supply',
            'hydro',
            'wind',
            'nuclear',
            'total_supply_3m_ma',
            'renewable_percentage_3m_ma',
            'nuclear_percentage_3m_ma'
        ]
        
        # Price features
        price_features = ['SE3_price_ore']
        
        return price_features + time_features + holiday_features + grid_features
    
    def create_sequences(self, data):
        """Create sequences for LSTM training"""
        feature_cols = self.get_required_features()
        
        X, y = [], []
        values = data[feature_cols].values
        
        for i in range(len(values) - self.window_size):
            X.append(values[i:(i + self.window_size)])
            y.append(values[i + self.window_size, 0])  # First column is price
            
        return np.array(X), np.array(y)
    
    def prepare_data(self):
        """Prepare data for training"""
        df = self.load_data()
        
        # Add time-based features if not present
        if 'hour_sin' not in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df.index.hour/24)
            df['hour_cos'] = np.cos(2 * np.pi * df.index.hour/24)
            df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek/7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek/7)
            df['month_sin'] = np.sin(2 * np.pi * df.index.month/12)
            df['month_cos'] = np.cos(2 * np.pi * df.index.month/12)
            df['is_peak_hour'] = ((df.index.hour >= 6) & (df.index.hour <= 22)).astype(int)
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Get required features
        feature_cols = self.get_required_features()
        
        # Verify all required features exist
        missing_cols = set(feature_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required features: {missing_cols}")
        
        # Scale the price data
        prices = df['SE3_price_ore'].values.reshape(-1, 1)
        scaled_prices = self.scaler.fit_transform(prices)
        df['SE3_price_ore'] = scaled_prices.flatten()
        
        # Create sequences
        X, y = self.create_sequences(df[feature_cols])
        
        # Split into train, validation, and test sets
        train_split = int(len(X) * 0.7)
        val_split = int(len(X) * 0.85)
        
        X_train = X[:train_split]
        y_train = y[:train_split]
        X_val = X[train_split:val_split]
        y_val = y[train_split:val_split]
        X_test = X[val_split:]
        y_test = y[val_split:]
        
        return (X_train, y_train, X_val, y_val, X_test, y_test), df.index[self.window_size:]
    
    def build_model(self, input_shape):
        """Build the LSTM model"""
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train(self, epochs=50, batch_size=32):
        """Train the model"""
        logging.info("Preparing data...")
        (X_train, y_train, X_val, y_val, X_test, y_test), timestamps = self.prepare_data()
        
        # Get the validation split index for saving test data later
        train_split = int(len(timestamps) * 0.7)
        val_split = int(len(timestamps) * 0.85)
        
        logging.info("Building model...")
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ModelCheckpoint(
                self.models_dir / 'best_model.keras',
                monitor='val_loss',
                save_best_only=True
            ),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        logging.info("Training model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model and scaler
        self.model.save(self.models_dir / 'final_model.keras')
        joblib.dump(self.scaler, self.models_dir / 'scaler.save')
        
        # Save test data
        np.save(self.test_data_dir / 'X_test.npy', X_test)
        np.save(self.test_data_dir / 'y_test.npy', y_test)
        np.save(self.test_data_dir / 'test_timestamps.npy', timestamps[val_split:])
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
    
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    trainer = PriceModelTrainer()
    history = trainer.train()
    logging.info("Training complete!")

if __name__ == "__main__":
    main()

