#!/usr/bin/env python3
"""
CNN Energy Demand Prediction Model

This script implements a CNN model for forecasting energy consumption.
It can be run independently from the main XGBoost implementation.
"""

import os
import sys
import json
import logging
import traceback
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File paths (adjust if needed)
CONSUMPTION_FILE = 'data/processed/VillamichelinConsumption.csv'
WEATHER_FILE = 'data/processed/weather_data.csv'
TIME_FEATURES_FILE = 'data/processed/time_features.csv'
HOLIDAYS_FILE = 'data/processed/holidays.csv'
MODEL_PATH = 'src/predictions/demand/models/CNN'
OUTPUT_DIR = 'src/predictions/demand/models/CNN/output'

def load_consumption_data():
    """
    Load consumption data from CSV file
    """
    logger.info(f"Loading consumption data from {os.path.abspath(CONSUMPTION_FILE)}...")
    try:
        df = pd.read_csv(CONSUMPTION_FILE)
        
        # Convert the timestamp to datetime with UTC=True to handle timezone consistently
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df.set_index('timestamp', inplace=True)
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        logger.info(f"Consumption data loaded: {df.shape[0]} records from {df.index.min()} to {df.index.max()}")
        return df
    except Exception as e:
        logger.error(f"Error loading consumption data: {e}")
        return None

def load_weather_data():
    """
    Load weather data from CSV file
    """
    logger.info(f"Loading weather data from {os.path.abspath(WEATHER_FILE)}...")
    try:
        weather_df = pd.read_csv(WEATHER_FILE)
        
        # Convert the timestamp to datetime with UTC=True
        weather_df['date'] = pd.to_datetime(weather_df['date'], utc=True)
        weather_df.set_index('date', inplace=True)
        
        # Sort by timestamp
        weather_df.sort_index(inplace=True)
        
        # Rename columns if needed to match expected names
        column_mapping = {
            'temperature': 'temp',
            'relativehumidity_2m': 'humidity',
            'cloudcover': 'clouds',
            'windspeed_100m': 'wind_speed'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in weather_df.columns and new_name not in weather_df.columns:
                weather_df[new_name] = weather_df[old_name]
        
        # Calculate feels_like if not present
        if 'feels_like' not in weather_df.columns and 'temp' in weather_df.columns and 'humidity' in weather_df.columns and 'wind_speed' in weather_df.columns:
            # Simple approximation of feels_like based on temp, humidity and wind
            weather_df['feels_like'] = weather_df['temp'] - 0.25 * (1 - weather_df['humidity']) * (weather_df['temp'] - 10) + 0.1 * weather_df['wind_speed']
        
        logger.info(f"Weather data loaded: {weather_df.shape[0]} records")
        return weather_df
    except Exception as e:
        logger.error(f"Error loading weather data: {e}")
        return None

def load_time_features():
    """
    Load time features from CSV file
    """
    logger.info(f"Loading time features from {os.path.abspath(TIME_FEATURES_FILE)}...")
    try:
        time_df = pd.read_csv(TIME_FEATURES_FILE, index_col=0)
        
        # Convert the index to datetime with UTC=True
        time_df.index = pd.to_datetime(time_df.index, utc=True)
        
        # Sort by timestamp
        time_df = time_df.sort_index()
        
        logger.info(f"Time features loaded: {time_df.shape[0]} records")
        return time_df
    except Exception as e:
        logger.error(f"Error loading time features: {e}")
        logger.error(traceback.format_exc())
        return None

def load_holidays():
    """
    Load holidays data from CSV file
    """
    logger.info(f"Loading holidays data from {os.path.abspath(HOLIDAYS_FILE)}...")
    try:
        holidays_df = pd.read_csv(HOLIDAYS_FILE, index_col=0)
        
        # Convert the index to datetime with UTC=True
        holidays_df.index = pd.to_datetime(holidays_df.index, utc=True)
        
        # Sort by timestamp
        holidays_df = holidays_df.sort_index()
        
        logger.info(f"Holidays data loaded: {holidays_df.shape[0]} records")
        return holidays_df
    except Exception as e:
        logger.error(f"Error loading holidays data: {e}")
        logger.error(traceback.format_exc())
        return None

def create_lag_features(df, target_col='consumption', lag_hours=[1, 24, 48, 168]):
    """
    Create lag features for time series forecasting
    
    Args:
        df: DataFrame with time series data
        target_col: Target column to create lags for
        lag_hours: List of hours to lag
        
    Returns:
        DataFrame with lag features
    """
    for lag in lag_hours:
        df[f'{target_col}_lag_{lag}h'] = df[target_col].shift(lag)
    
    return df

def split_data(df):
    """
    Split data into training, validation and test sets.
    For time series data, we use time-based splits rather than random sampling.
    
    Args:
        df: DataFrame with features and target
        
    Returns:
        train_df, val_df, test_df: Split dataframes
    """
    logging.info("Splitting data into train, validation, and test sets...")
    
    # Calculate split indices (70% train, 15% validation, 15% test)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    # Split the data
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # Log the size of each dataset
    logging.info(f"Data split complete - Training: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df

def prepare_features(consumption_df):
    """
    Prepare features for the energy demand prediction model
    
    Args:
        consumption_df: DataFrame with consumption data
        
    Returns:
        DataFrame with features ready for modeling
    """
    logger.info("Preparing features for model training...")
    
    try:
        # Create a copy to avoid modifying the original dataframe
        df = consumption_df.copy()
        
        # Load weather data
        weather_df = load_weather_data()
        time_df = load_time_features()
        holidays_df = load_holidays()
        
        # Merge time features
        if time_df is not None:
            logger.info("Using time features from CSV file")
            
            # Merge with time features on index (timestamp)
            df = df.join(time_df, how='left')
            
            # Check which columns were merged
            time_cols = [col for col in time_df.columns if col in df.columns]
            logger.info(f"Merged time features: {time_cols}")
        
        # Merge weather data
        if weather_df is not None:
            logger.info("Using weather data from CSV file")
            
            # Merge with weather data on index (timestamp)
            df = df.join(weather_df, how='left')
            
            # Check which columns were merged
            weather_cols = [col for col in weather_df.columns if col in df.columns]
            logger.info(f"Merged weather features: {weather_cols}")
        
        # Merge holidays data
        if holidays_df is not None:
            logger.info("Using holidays data from CSV file")
            
            # Merge with holidays data on index (timestamp)
            df = df.join(holidays_df, how='left')
            
            # Check which columns were merged
            holiday_cols = [col for col in holidays_df.columns if col in df.columns]
            logger.info(f"Merged holiday features: {holiday_cols}")
        
        # Create lag features for consumption
        df = create_lag_features(df)
        
        # Fill missing values using modern pandas syntax
        df = df.ffill().bfill()  # Forward fill then backward fill for any remaining NaNs
        
        # Handle non-numeric columns
        for col in df.columns:
            if df[col].dtype == 'object':
                logger.warning(f"Converting {col} from {df[col].dtype} to numeric")
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop any columns that are entirely NaN
        df = df.dropna(axis=1, how='all')
        
        # Log a warning if any columns still have NaN values
        cols_with_nan = df.columns[df.isna().any()].tolist()
        if cols_with_nan:
            logger.warning(f"Columns with NaN values after cleaning: {cols_with_nan}")
            
            # For columns with NaN values, fill with median or 0
            for col in cols_with_nan:
                if df[col].dtype in ['float64', 'int64']:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.info(f"Filled NaN values in {col} with median: {median_val}")
                else:
                    df[col] = df[col].fillna(0)
                    logger.info(f"Filled NaN values in {col} with 0")
        
        # Now drop rows with any remaining NaN values
        original_count = len(df)
        df = df.dropna()
        dropped_count = original_count - len(df)
        
        if dropped_count > 0:
            logger.warning(f"Dropped {dropped_count} rows with NaN values")
        
        # Check if the dataframe is empty after all processing
        if df.empty:
            logger.error("Feature preparation resulted in empty dataframe. Check data compatibility.")
            # Return original consumption dataframe as fallback
            logger.info("Using only consumption data with lag features as fallback")
            df_fallback = consumption_df.copy()
            df_fallback = create_lag_features(df_fallback)
            df_fallback = df_fallback.ffill().bfill().dropna()
            
            if not df_fallback.empty:
                logger.info(f"Fallback feature preparation: {df_fallback.shape[0]} records with {df_fallback.shape[1]} features")
                return df_fallback
            else:
                raise ValueError("Unable to prepare features even with fallback approach")
        
        logger.info(f"Feature preparation complete: {df.shape[0]} records with {df.shape[1]} features")
        return df
    
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        logger.error(traceback.format_exc())
        raise

def sequence_data_preparation(df, n_steps=24):
    """
    Prepare sequence data for CNN model by creating sliding windows
    
    Args:
        df: DataFrame with features and target
        n_steps: Number of time steps in each sequence (lookback window)
        
    Returns:
        X_seq: Sequence data for features
        y_seq: Target values
        indices: Indices for the sequences in the original DataFrame
    """
    logging.info(f"Preparing sequence data with {n_steps} time steps lookback window")
    
    try:
        # Separate target and features
        y = df['consumption'].values
        X = df.drop('consumption', axis=1).values
        
        # Create sequences
        X_seq = []
        y_seq = []
        indices = []
        
        # For each possible sequence in the data
        for i in range(len(df) - n_steps):
            # Get sequence of features
            X_seq.append(X[i:(i + n_steps)])
            # Get target value (the next value after the sequence)
            y_seq.append(y[i + n_steps])
            # Store the index of the target (used for plotting later)
            indices.append(i + n_steps)
        
        # Convert to numpy arrays
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        logging.info(f"Sequence data prepared: {len(X_seq)} samples with shape {n_steps}x{X.shape[1]}")
        logging.info(f"Dropped {n_steps} samples at the beginning of the series")
        
        return X_seq, y_seq, indices
        
    except Exception as e:
        logging.error(f"Error preparing sequence data: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def save_weights(model, name='cnn_weights', path=None):
    """
    Save model weights to a specified path
    
    Args:
        model: Model to save weights from
        name: Name for the weights file
        path: Directory to save weights to
    """
    if path is None:
        path = MODEL_PATH
        
    os.makedirs(path, exist_ok=True)
    
    # Save in TensorFlow weights format
    weights_file = os.path.join(path, f"{name}.weights.h5")
    model.save_weights(weights_file)
    logging.info(f"Model weights saved to {os.path.abspath(weights_file)}")
    
    # Save weights for easy manipulation using pickle
    weights_dict = {}
    for i, layer in enumerate(model.layers):
        if len(layer.weights) > 0:  # Only layers with weights
            layer_name = layer.name
            # Store as list of numpy arrays
            weights_dict[layer_name] = [w.numpy() for w in layer.weights]
    
    # Save using pickle instead of numpy.savez
    pickle_file = os.path.join(path, f"{name}_data.pkl")
    with open(pickle_file, 'wb') as f:
        pickle.dump(weights_dict, f)
    
    logging.info(f"Model weights also saved in pickle format for manual modification: {os.path.abspath(pickle_file)}")
    
    return weights_file, pickle_file

def load_weights(model, weights_file):
    """
    Load model weights from a file
    
    Args:
        model: Model to load weights into
        weights_file: Path to weights file
    """
    try:
        model.load_weights(weights_file)
        logging.info(f"Model weights loaded from {os.path.abspath(weights_file)}")
        return True
    except Exception as e:
        logging.error(f"Error loading model weights: {e}")
        return False

def modify_weights(weights_file, layer_name, weight_type='kernel', scale_factor=1.0):
    """
    Modify weights in a pickle weights file by scaling them
    
    Args:
        weights_file: Path to pickle weights file (.pkl)
        layer_name: Name of the layer to modify
        weight_type: Type of weights to modify ('kernel' or 'bias')
        scale_factor: Factor to scale weights by
    
    Returns:
        Path to new weights file
    """
    try:
        # Load weights from pickle file
        with open(weights_file, 'rb') as f:
            weights_dict = pickle.load(f)
        
        # Find the right weights to modify
        if layer_name in weights_dict:
            if weight_type == 'kernel':
                weight_index = 0  # Usually kernel is the first weight
            else:  # bias
                weight_index = 1  # Usually bias is the second weight
            
            # Modify weights
            if weight_index < len(weights_dict[layer_name]):
                # Get the weights
                weights = weights_dict[layer_name]
                
                # Scale the weights
                weights[weight_index] = weights[weight_index] * scale_factor
                
                # Update the dictionary
                weights_dict[layer_name] = weights
                
                logging.info(f"Modified {weight_type} weights in layer {layer_name} by scale factor {scale_factor}")
            else:
                logging.error(f"Weight index {weight_index} out of range for layer {layer_name}")
                return None
        else:
            logging.error(f"Layer {layer_name} not found in weights file")
            return None
        
        # Save modified weights
        new_weights_file = os.path.splitext(weights_file)[0] + "_modified.pkl"
        with open(new_weights_file, 'wb') as f:
            pickle.dump(weights_dict, f)
            
        logging.info(f"Modified weights saved to {os.path.abspath(new_weights_file)}")
        
        return new_weights_file
    
    except Exception as e:
        logging.error(f"Error modifying weights: {e}")
        logging.error(traceback.format_exc())
        return None

def load_model_from_file(model_path=None):
    """
    Load a trained model from file
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded model or None if loading fails
    """
    if model_path is None:
        model_path = os.path.join(MODEL_PATH, 'cnn_model.keras')
    
    try:
        model = load_model(model_path)
        logging.info(f"Model loaded from {os.path.abspath(model_path)}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        logging.error(traceback.format_exc())
        return None

def load_model_with_weights(weights_file, time_steps, n_features):
    """
    Create a model with the given architecture and load weights
    
    Args:
        weights_file: Path to weights file
        time_steps: Number of time steps for input
        n_features: Number of features for input
        
    Returns:
        Model with loaded weights or None if loading fails
    """
    try:
        # Create model with same architecture
        model = train_cnn_model(None, None, None, None, time_steps, n_features, train=False)
        
        # Load weights
        if weights_file.endswith('.pkl'):
            success = load_model_from_pickle_weights(model, weights_file)
        elif weights_file.endswith('.npz'):
            success = load_model_from_numpy_weights(model, weights_file)
        else:
            success = load_weights(model, weights_file)
            
        if success:
            return model
        else:
            return None
    except Exception as e:
        logging.error(f"Error loading model with weights: {e}")
        logging.error(traceback.format_exc())
        return None

def load_model_from_numpy_weights(model, weights_npz_file):
    """
    Load model weights from a numpy weights file (.npz)
    
    Args:
        model: Model to load weights into
        weights_npz_file: Path to numpy weights file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load weights
        weights_dict = dict(np.load(weights_npz_file))
        
        # Apply weights to model
        for layer in model.layers:
            if layer.name in weights_dict:
                layer_weights = weights_dict[layer.name]
                
                # Convert from object array if needed
                if isinstance(layer_weights, np.ndarray) and layer_weights.dtype == np.dtype('object'):
                    layer_weights = [w for w in layer_weights]
                
                # Make sure we have the right number of weights
                if len(layer_weights) == len(layer.weights):
                    layer.set_weights(layer_weights)
                    logging.info(f"Loaded weights for layer {layer.name}")
                else:
                    logging.warning(f"Weight count mismatch for layer {layer.name}: " + 
                                   f"expected {len(layer.weights)}, got {len(layer_weights)}")
        
        logging.info(f"Model weights loaded from numpy file {os.path.abspath(weights_npz_file)}")
        return True
    
    except Exception as e:
        logging.error(f"Error loading model from numpy weights: {e}")
        logging.error(traceback.format_exc())
        return False

def load_model_from_pickle_weights(model, weights_pkl_file):
    """
    Load model weights from a pickle weights file (.pkl)
    
    Args:
        model: Model to load weights into
        weights_pkl_file: Path to pickle weights file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load weights from pickle file
        with open(weights_pkl_file, 'rb') as f:
            weights_dict = pickle.load(f)
        
        # Apply weights to model
        for layer in model.layers:
            if layer.name in weights_dict:
                layer_weights = weights_dict[layer.name]
                
                # Make sure we have the right number of weights
                if len(layer_weights) == len(layer.weights):
                    layer.set_weights(layer_weights)
                    logging.info(f"Loaded weights for layer {layer.name}")
                else:
                    logging.warning(f"Weight count mismatch for layer {layer.name}: " + 
                                   f"expected {len(layer.weights)}, got {len(layer_weights)}")
        
        logging.info(f"Model weights loaded from pickle file {os.path.abspath(weights_pkl_file)}")
        return True
    
    except Exception as e:
        logging.error(f"Error loading model from pickle weights: {e}")
        logging.error(traceback.format_exc())
        return False

def print_model_weights_info(model):
    """
    Print information about model layers and weights to help with weight modification
    
    Args:
        model: Model to inspect
    """
    logging.info("Model layers and weights information:")
    logging.info("=" * 50)
    
    print("\nMODEL LAYERS AND WEIGHTS INFORMATION:")
    print("=" * 50)
    
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if len(weights) > 0:
            logging.info(f"Layer {i}: {layer.name} - Type: {layer.__class__.__name__}")
            print(f"Layer {i}: {layer.name} - Type: {layer.__class__.__name__}")
            
            # Print weight shapes
            for j, w in enumerate(weights):
                weight_type = "kernel" if j == 0 else "bias" if j == 1 else f"weight_{j}"
                logging.info(f"  - {weight_type} shape: {w.shape}, min: {w.min():.6f}, max: {w.max():.6f}, mean: {w.mean():.6f}")
                print(f"  - {weight_type} shape: {w.shape}, min: {w.min():.6f}, max: {w.max():.6f}, mean: {w.mean():.6f}")
    
    logging.info("=" * 50)
    print("=" * 50)

def train_cnn_model(X_train, y_train, X_val, y_val, time_steps, n_features, train=True):
    """
    Train a CNN model for energy demand prediction
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        time_steps: Number of time steps in sequence
        n_features: Number of features
        train: If True, train the model; otherwise, just create the model architecture
        
    Returns:
        Trained CNN model
    """
    logging.info("Training CNN model..." if train else "Creating CNN model architecture...")
    
    # Ensure model directory exists
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Define model architecture with increased regularization to reduce overfitting
    model = Sequential([
        # First convolutional block
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', 
               input_shape=(time_steps, n_features),
               kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # Second convolutional block
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),  # Increased dropout
        
        # Third convolutional block
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Flatten(),
        
        # Dense layers
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.4),  # Increased dropout
        Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dense(1)
    ])
    
    # Compile model with Adam optimizer using a fixed learning rate
    initial_learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Print model summary
    model.summary(print_fn=logging.info)
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,  # Increased patience
        restore_best_weights=True,
        verbose=1
    )
    
    # Model checkpoint to save best model
    model_checkpoint = ModelCheckpoint(
        os.path.join(MODEL_PATH, 'cnn_model.keras'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Reduce LR by half
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # Train model
    if train:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
            verbose=1
        )
        
        # Save model
        model.save(os.path.join(MODEL_PATH, 'cnn_model.keras'))
        logging.info("CNN model trained and saved successfully")
        
        # Save weights separately for easier manipulation
        save_weights(model, 'cnn_weights')
        
        # Plot training history
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 1, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('CNN Model Training - Loss', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('CNN Model Training - Mean Absolute Error', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MAE', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plots_dir = os.path.join(OUTPUT_DIR, 'cnn_plots')
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, 'training_history.png'), dpi=300)
        plt.close()
    
    return model

def evaluate_cnn_model(model, X_test, y_test, test_idx, test_df):
    """
    Evaluate CNN model and create evaluation plots
    
    Args:
        model: Trained CNN model
        X_test: Test features
        y_test: Test targets
        test_idx: Test data indices
        test_df: Original test dataframe
    """
    logger.info("Evaluating CNN model...")
    
    try:
        # Make predictions
        y_pred = model.predict(X_test).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Log metrics
        logger.info(f"CNN Model Metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")
        
        # Save metrics to JSON
        metrics = {
            'CNN': {
                'MAE': float(mae),
                'RMSE': float(rmse),
                'R2': float(r2),
                'MAPE': float(mape)
            }
        }
        
        with open(os.path.join(MODEL_PATH, 'cnn_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Create evaluation plots
        create_evaluation_plots(model, X_test, y_test, test_df, test_idx)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating CNN model: {e}")
        logger.error(traceback.format_exc())
        return None

def create_evaluation_plots(model, X_test, y_test, test_df, test_idx):
    """
    Create evaluation plots for the CNN model
    
    Args:
        model: Trained CNN model
        X_test: Test data features
        y_test: Test data targets
        test_df: Original test dataframe
        test_idx: Indices for test data
    """
    logging.info("Creating CNN evaluation plots...")
    
    # Create directory for plots if it doesn't exist
    plots_dir = os.path.join(OUTPUT_DIR, 'cnn_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Make predictions
    y_pred = model.predict(X_test).flatten()
    
    # Create a dataframe with actual and predicted values
    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred
    }, index=test_df.iloc[test_idx].index)
    
    # Save results to CSV
    results_df.to_csv(os.path.join(plots_dir, 'test_results.csv'))
    
    # Plot actual vs predicted - Higher resolution
    plt.figure(figsize=(20, 10))
    plt.plot(results_df.index, results_df['actual'], 'b-', label='Actual', linewidth=2)
    plt.plot(results_df.index, results_df['predicted'], 'r-', label='Predicted', linewidth=2)
    plt.title('CNN Model: Actual vs Predicted Energy Consumption', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Energy Consumption (kWh)', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted.png'), dpi=300)
    plt.close()
    
    # Plot weekly comparisons with higher resolution
    # Get min and max dates to determine weeks
    min_date = results_df.index.min()
    max_date = results_df.index.max()
    
    # Create weekly plots
    current_date = min_date
    week_num = 1
    
    while current_date <= max_date:
        week_end = current_date + pd.Timedelta(days=7)
        week_data = results_df.loc[current_date:week_end]
        
        if not week_data.empty and len(week_data) > 12:  # Make sure we have enough data points
            plt.figure(figsize=(20, 10))
            plt.plot(week_data.index, week_data['actual'], 'b-', marker='o', markersize=4, label='Actual', linewidth=2)
            plt.plot(week_data.index, week_data['predicted'], 'r-', marker='x', markersize=4, label='Predicted', linewidth=2)
            plt.title(f'CNN Model: Week {week_num} - Actual vs Predicted Energy Consumption', fontsize=16)
            plt.xlabel('Time', fontsize=14)
            plt.ylabel('Energy Consumption (kWh)', fontsize=14)
            plt.legend(fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45, fontsize=12)
            plt.yticks(fontsize=12)
            
            # Add shaded regions for day/night
            for day in pd.date_range(current_date, week_end):
                night_start = pd.Timestamp(day.year, day.month, day.day, 22, 0)
                next_day = day + pd.Timedelta(days=1)
                morning_end = pd.Timestamp(next_day.year, next_day.month, next_day.day, 6, 0)
                plt.axvspan(night_start, morning_end, alpha=0.1, color='gray')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'week_{week_num}_comparison.png'), dpi=300)
            plt.close()
            
            week_num += 1
        
        current_date = week_end
    
    # Scatter plot of actual vs predicted values
    plt.figure(figsize=(12, 10))
    plt.scatter(results_df['actual'], results_df['predicted'], alpha=0.5)
    plt.plot([0, results_df['actual'].max()], [0, results_df['actual'].max()], 'r--')
    plt.title('CNN Model: Actual vs Predicted Scatter Plot', fontsize=16)
    plt.xlabel('Actual Energy Consumption (kWh)', fontsize=14)
    plt.ylabel('Predicted Energy Consumption (kWh)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'scatter_plot.png'), dpi=300)
    plt.close()
    
    # Error distribution histogram
    errors = results_df['actual'] - results_df['predicted']
    plt.figure(figsize=(12, 8))
    plt.hist(errors, bins=50, alpha=0.75)
    plt.title('CNN Model: Error Distribution', fontsize=16)
    plt.xlabel('Prediction Error (kWh)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'error_distribution.png'), dpi=300)
    plt.close()
    
    logging.info(f"CNN evaluation plots created in {plots_dir}")

def predict_future_cnn(model, df, time_steps, n_features):
    """Predict the next 24 hours of energy consumption using the trained CNN model."""
    try:
        # Get the last time_steps records
        last_window = df.iloc[-time_steps:].copy()
        
        # Make sure we have the right feature count for prediction
        feature_count = last_window.select_dtypes(include=['float64', 'int64']).shape[1]
        
        # Create timestamps for next 24 hours
        last_timestamp = df.index[-1]
        future_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1),
            periods=24,
            freq="H"
        )
        
        # Create dataframe for future predictions
        future_df = pd.DataFrame(index=future_timestamps)
        predictions = []
        
        # Prepare the initial sequence
        last_sequence = last_window.select_dtypes(include=['float64', 'int64']).values
        
        # Ensure we have exactly n_features columns
        if last_sequence.shape[1] != n_features:
            logging.warning(f"Feature count mismatch: Model expects {n_features} features but got {last_sequence.shape[1]}. Adjusting...")
            # If we have more features than needed, trim
            if last_sequence.shape[1] > n_features:
                last_sequence = last_sequence[:, :n_features]
            # If we have fewer features, pad with zeros
            else:
                padding = np.zeros((last_sequence.shape[0], n_features - last_sequence.shape[1]))
                last_sequence = np.hstack((last_sequence, padding))
        
        # Reshape to match model input shape
        last_sequence = last_sequence.reshape(1, time_steps, n_features)
        
        # Make predictions for next 24 hours
        for i in range(24):
            # Predict next value
            next_pred = model.predict(last_sequence)[0][0]
            predictions.append(next_pred)
            
            # Update the sequence by removing first record and adding new prediction
            # Create a row with the same structure as our training data
            new_row = np.zeros(n_features)
            new_row[0] = next_pred  # Set consumption value
            
            # Roll the window forward: remove first, add new prediction at the end
            new_sequence = np.concatenate([last_sequence[0, 1:, :], [new_row]])
            last_sequence = new_sequence.reshape(1, time_steps, n_features)
        
        # Add predictions to future dataframe
        future_df['consumption'] = predictions
        
        return future_df
    
    except Exception as e:
        logging.error(f"Error predicting with CNN model: {str(e)}")
        traceback.print_exc()
        return None

def compare_with_xgboost():
    """
    Compare CNN model with XGBoost model if available
    """
    logger.info("Comparing CNN model with XGBoost model...")
    
    try:
        # Check if CNN metrics exist
        cnn_metrics_path = os.path.join(MODEL_PATH, 'cnn_metrics.json')
        if not os.path.exists(cnn_metrics_path):
            logger.error("CNN metrics not found. Run CNN evaluation first.")
            return
            
        # Check if XGBoost metrics exist
        xgb_metrics_path = os.path.join(MODEL_PATH, 'model_metrics.json')
        if not os.path.exists(xgb_metrics_path):
            logger.warning("XGBoost metrics not found. Skipping comparison.")
            return
            
        # Load metrics
        with open(cnn_metrics_path, 'r') as f:
            cnn_metrics = json.load(f)
            
        with open(xgb_metrics_path, 'r') as f:
            xgb_metrics = json.load(f)
            
        # Create comparison
        comparison = {
            'XGBoost': xgb_metrics,
            'CNN': cnn_metrics['CNN']
        }
        
        # Save comparison
        with open(os.path.join(MODEL_PATH, 'model_comparison.json'), 'w') as f:
            json.dump(comparison, f, indent=4)
            
        # Print comparison
        logger.info("Model Comparison:")
        logger.info(f"XGBoost - MAE: {xgb_metrics['MAE']:.4f}, RMSE: {xgb_metrics['RMSE']:.4f}, R²: {xgb_metrics['R2']:.4f}")
        logger.info(f"CNN - MAE: {cnn_metrics['CNN']['MAE']:.4f}, RMSE: {cnn_metrics['CNN']['RMSE']:.4f}, R²: {cnn_metrics['CNN']['R2']:.4f}")
        
        # Determine better model
        if xgb_metrics['MAE'] <= cnn_metrics['CNN']['MAE']:
            logger.info("XGBoost model performs better based on MAE")
            return 'XGBoost'
        else:
            logger.info("CNN model performs better based on MAE")
            return 'CNN'
            
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        logger.error(traceback.format_exc())
        return None

def test_weight_modifications(model, X_test, y_test, test_idx, test_df, time_steps, n_features):
    """
    Test different weight modifications and evaluate their impact
    
    Args:
        model: Trained model to modify
        X_test, y_test, test_idx, test_df: Test data for evaluation
        time_steps, n_features: Model parameters
    """
    # Original model metrics as baseline
    logging.info("Baseline model performance:")
    baseline_metrics = evaluate_cnn_model(model, X_test, y_test, test_idx, test_df)
    
    # Save weights for modification
    _, weights_pkl = save_weights(model, 'cnn_weights_baseline')
    
    # Define layers and scale factors to test
    modifications = [
        ('conv1d', 'kernel', 1.1),     # Increase first conv layer weights slightly
        ('conv1d', 'kernel', 0.9),     # Decrease first conv layer weights slightly
        ('conv1d_2', 'kernel', 1.2),   # Increase last conv layer weights more
        ('dense', 'kernel', 0.8),      # Decrease dense layer weights
        ('dense_2', 'kernel', 1.5)     # Increase output layer weights significantly
    ]
    
    results = {}
    
    for layer_name, weight_type, scale_factor in modifications:
        try:
            logging.info(f"\nTesting modification: {layer_name} {weight_type} * {scale_factor}")
            print(f"\nTesting modification: {layer_name} {weight_type} * {scale_factor}")
            
            # Modify weights
            modified_weights = modify_weights(weights_pkl, layer_name, weight_type, scale_factor)
            
            if modified_weights:
                # Create fresh model
                new_model = train_cnn_model(None, None, None, None, time_steps, n_features, train=False)
                
                # Load modified weights
                if load_model_from_pickle_weights(new_model, modified_weights):
                    # Evaluate modified model
                    metrics = evaluate_cnn_model(new_model, X_test, y_test, test_idx, test_df)
                    
                    # Store results
                    mod_name = f"{layer_name}_{weight_type}_{scale_factor}"
                    results[mod_name] = {
                        'mae': metrics['CNN']['MAE'],
                        'rmse': metrics['CNN']['RMSE'],
                        'r2': metrics['CNN']['R2'],
                        'pct_change_mae': (metrics['CNN']['MAE'] - baseline_metrics['CNN']['MAE']) / baseline_metrics['CNN']['MAE'] * 100
                    }
        except Exception as e:
            logging.error(f"Error testing modification {layer_name} {weight_type} * {scale_factor}: {e}")
    
    # Summarize results
    logging.info("\nWeight Modification Results:")
    logging.info("=" * 70)
    print("\nWeight Modification Results:")
    print("=" * 70)
    
    print(f"{'Modification':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MAE %Δ':>8}")
    print("-" * 70)
    logging.info(f"{'Modification':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MAE %Δ':>8}")
    logging.info("-" * 70)
    
    # Baseline
    print(f"{'Baseline':<25} {baseline_metrics['CNN']['MAE']:>8.4f} {baseline_metrics['CNN']['RMSE']:>8.4f} {baseline_metrics['CNN']['R2']:>8.4f} {0:>8.2f}")
    logging.info(f"{'Baseline':<25} {baseline_metrics['CNN']['MAE']:>8.4f} {baseline_metrics['CNN']['RMSE']:>8.4f} {baseline_metrics['CNN']['R2']:>8.4f} {0:>8.2f}")
    
    # Modified versions
    for mod_name, metrics in results.items():
        print(f"{mod_name:<25} {metrics['mae']:>8.4f} {metrics['rmse']:>8.4f} {metrics['r2']:>8.4f} {metrics['pct_change_mae']:>8.2f}")
        logging.info(f"{mod_name:<25} {metrics['mae']:>8.4f} {metrics['rmse']:>8.4f} {metrics['r2']:>8.4f} {metrics['pct_change_mae']:>8.2f}")
    
    print("=" * 70)
    logging.info("=" * 70)
    
    return results

def demonstrate_weight_modification(model, X_test, y_test, test_idx, test_df, time_steps, n_features):
    """
    Demonstrate how to load a model, modify weights, and compare results
    
    Args:
        model: Original trained model
        X_test, y_test, test_idx, test_df: Test data
        time_steps, n_features: Model parameters
    """
    try:
        logging.info("\n" + "="*50)
        logging.info("DEMONSTRATING MODEL WEIGHT MODIFICATION WORKFLOW")
        logging.info("="*50)
        
        print("\n" + "="*50)
        print("DEMONSTRATING MODEL WEIGHT MODIFICATION WORKFLOW")
        print("="*50)
        
        # Step 1: Save original model weights
        print("\nStep 1: Saving original model weights...")
        weights_file, pickle_file = save_weights(model, 'cnn_weights_demo')
        
        # Step 2: Display model layers and weights info
        print("\nStep 2: Displaying model layers and weights info...")
        print_model_weights_info(model)
        
        # Step 3: Create modified weights by scaling a layer
        print("\nStep 3: Creating modified weights (scaling conv1d layer by 1.2)...")
        modified_weights = modify_weights(pickle_file, 'conv1d', 'kernel', 1.2)
        
        if modified_weights:
            # Step 4: Create a new model instance
            print("\nStep 4: Creating a new model with the same architecture...")
            new_model = train_cnn_model(None, None, None, None, time_steps, n_features, train=False)
            
            # Step 5: Load modified weights
            print("\nStep 5: Loading modified weights into the new model...")
            load_model_from_pickle_weights(new_model, modified_weights)
            
            # Step 6: Compare performance
            print("\nStep 6: Comparing performance of original vs modified models...")
            print("\nOriginal model metrics:")
            orig_metrics = evaluate_cnn_model(model, X_test, y_test, test_idx, test_df)
            
            print("\nModified model metrics:")
            mod_metrics = evaluate_cnn_model(new_model, X_test, y_test, test_idx, test_df)
            
            # Step 7: Save the modified model if it performs better
            if mod_metrics['CNN']['MAE'] < orig_metrics['CNN']['MAE']:
                print("\nStep 7: Modified model performs better! Saving it...")
                new_model.save(os.path.join(MODEL_PATH, 'cnn_model_improved.keras'))
                save_weights(new_model, 'cnn_weights_improved')
                logging.info("Improved model saved!")
            else:
                print("\nStep 7: Original model performs better. No need to save modified version.")
        
        logging.info("Weight modification demonstration completed.")
        print("\nWeight modification demonstration completed.")
        print("="*50)
        
    except Exception as e:
        logging.error(f"Error in weight modification demonstration: {e}")
        logging.error(traceback.format_exc())

def main():
    """Main function to run the model."""
    try:
        logging.info("Starting CNN energy demand prediction...")
        
        # Create necessary directories
        os.makedirs(MODEL_PATH, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Load consumption data
        consumption_df = load_consumption_data()
        
        # Prepare features
        df = prepare_features(consumption_df)
        
        # Split data
        train_df, val_df, test_df = split_data(df)
        
        # Define model parameters
        time_steps = 24
        
        # Prepare sequence data first to get the actual feature count
        X_train, y_train, _ = sequence_data_preparation(train_df, time_steps)
        X_val, y_val, _ = sequence_data_preparation(val_df, time_steps)
        X_test, y_test, test_idx = sequence_data_preparation(test_df, time_steps)
        
        # Use the actual feature count from the sequence data
        n_features = X_train.shape[2]
        logging.info(f"Number of features for model: {n_features}")
        
        # Train model
        model = train_cnn_model(X_train, y_train, X_val, y_val, time_steps, n_features)
        
        # Evaluate model
        evaluate_cnn_model(model, X_test, y_test, test_idx, test_df)
        
        # Predict future
        future_df = predict_future_cnn(model, df, time_steps, n_features)
        
        if future_df is not None:
            # Save predictions
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            future_df.to_csv(os.path.join(OUTPUT_DIR, 'cnn_predictions.csv'))
            logging.info(f"Future predictions saved to {os.path.abspath(os.path.join(OUTPUT_DIR, 'cnn_predictions.csv'))}")
            
            # Plot future predictions with higher resolution
            plt.figure(figsize=(20, 10))
            plt.plot(future_df.index, future_df['consumption'], 'r-', linewidth=2, marker='o', label='Predicted Consumption')
            
            # Add historical data for context (last 48 hours)
            last_48h = df.iloc[-48:]['consumption']
            plt.plot(last_48h.index, last_48h.values, 'b-', linewidth=2, label='Historical Consumption')
            
            plt.title('CNN Model: Energy Consumption Forecast (Next 24 Hours)', fontsize=16)
            plt.xlabel('Time', fontsize=14)
            plt.ylabel('Energy Consumption (kWh)', fontsize=14)
            plt.legend(fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45, fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'cnn_future_prediction.png'), dpi=300)
            plt.close()
        
        # Compare with XGBoost if available
        compare_with_xgboost()
        
        # Demonstrate weight modification capabilities
        try:
            # Run demonstration of weight modification workflow showing how to modify and test model weights
            demonstrate_weight_modification(model, X_test, y_test, test_idx, test_df, time_steps, n_features)
            
            # Run comprehensive test of various weight modifications to find optimal settings
            test_weight_modifications(model, X_test, y_test, test_idx, test_df, time_steps, n_features)
        except Exception as e:
            logging.error(f"Error in weight modification: {e}")
            logging.error(traceback.format_exc())
        
        logging.info("CNN energy demand prediction completed successfully")
        
    except Exception as e:
        logging.error(f"Error in CNN model pipeline: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 