import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, RepeatVector, TimeDistributed, Dense

def build_model(input_window, forecast_horizon, lstm_units):
    """
    Build a sequence-to-sequence LSTM model that forecasts multiple future time steps.
    
    Args:
        input_window (int): Number of historical time steps provided as input.
        forecast_horizon (int): Number of future time steps to forecast.
        lstm_units (int): Number of units in the LSTM layers.
    
    Returns:
        tf.keras.Model: The compiled model.
    """
    model = Sequential([
        InputLayer(input_shape=(input_window, 1)),
        # Encoder LSTM
        LSTM(lstm_units, activation='tanh'),
        # Repeat the context vector forecast_horizon times
        RepeatVector(forecast_horizon),
        # Decoder LSTM returns a sequence
        LSTM(lstm_units, activation='tanh', return_sequences=True),
        # For each time step, produce a single output value
        TimeDistributed(Dense(1, activation='linear'))
    ])
    return model
