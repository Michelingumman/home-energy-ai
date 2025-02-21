import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import json
import pvlib
from datetime import datetime, timedelta
import requests
import os

def load_config():
    with open('src/predictions/solar/config.json', 'r') as f:
        return json.load(f)

def create_solar_position_data(config, start_date, end_date):
    location = config['system_specs']['location']
    latitude, longitude = location['latitude'], location['longitude']
    
    times = pd.date_range(start=start_date, end=end_date, freq='15min')
    solpos = pvlib.solarposition.get_solarposition(
        times, latitude, longitude
    )
    return pd.DataFrame({
        'solar_elevation': solpos['elevation'],
        'solar_azimuth': solpos['azimuth']
    }, index=times)

def fetch_weather_data(start_date, end_date, latitude, longitude, api_key):
    # You'll need to implement this using your preferred weather API
    # Example using OpenWeatherMap historical data
    base_url = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
    weather_data = []
    
    current_date = start_date
    while current_date <= end_date:
        response = requests.get(
            f"{base_url}?lat={latitude}&lon={longitude}&dt={int(current_date.timestamp())}&appid={api_key}"
        )
        if response.status_code == 200:
            data = response.json()
            weather_data.append({
                'timestamp': current_date,
                'temperature': data['current']['temp'] - 273.15,  # Convert to Celsius
                'cloud_cover': data['current']['clouds'] / 100.0
            })
        current_date += timedelta(hours=1)
    
    return pd.DataFrame(weather_data).set_index('timestamp')

def prepare_features(config, weather_data, solar_position_data):
    df = pd.concat([weather_data, solar_position_data], axis=1)
    
    # Add time-based features
    df['time_of_day'] = df.index.hour + df.index.minute / 60.0
    df['day_of_year'] = df.index.dayofyear
    
    # Calculate clear sky radiation using pvlib
    location = config['system_specs']['location']
    times = df.index
    latitude, longitude = location['latitude'], location['longitude']
    
    # Get clear sky radiation
    solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
    atmosphere = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'])
    dni_extra = pvlib.irradiance.get_extra_radiation(times.dayofyear)
    clarity_index = pvlib.atmosphere.simplified_solis(
        solpos['apparent_elevation'],
        airmass=atmosphere,
        dni_extra=dni_extra
    )
    df['clear_sky_radiation'] = clarity_index['dni']
    
    return df

def create_sequences(data, seq_length, prediction_horizon):
    X, y = [], []
    for i in range(len(data) - seq_length - prediction_horizon + 1):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length:i + seq_length + prediction_horizon, -1])
    return np.array(X), np.array(y)

def build_model(config, input_shape):
    model = Sequential()
    
    for i, units in enumerate(config['model_params']['lstm_layers']):
        return_sequences = i < len(config['model_params']['lstm_layers']) - 1
        if i == 0:
            model.add(LSTM(units, input_shape=input_shape, return_sequences=return_sequences))
        else:
            model.add(LSTM(units, return_sequences=return_sequences))
        
        model.add(Dropout(config['model_params']['dropout_rate']))
    
    model.add(Dense(config['model_params']['prediction_horizon']))
    
    model.compile(
        optimizer=Adam(learning_rate=config['model_params']['learning_rate']),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def main():
    config = load_config()
    
    # Load your historical solar production data
    # This should be replaced with your actual data loading logic
    historical_data = pd.read_csv('path_to_your_historical_data.csv')
    
    # Get weather and solar position data
    start_date = historical_data.index.min()
    end_date = historical_data.index.max()
    
    solar_position = create_solar_position_data(config, start_date, end_date)
    weather_data = fetch_weather_data(
        start_date, end_date,
        config['system_specs']['location']['latitude'],
        config['system_specs']['location']['longitude'],
        os.getenv('WEATHER_API_KEY')
    )
    
    # Prepare features
    features_df = prepare_features(config, weather_data, solar_position)
    features_df['historical_production'] = historical_data['production']
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features_df)
    
    # Create sequences
    X, y = create_sequences(
        scaled_data,
        config['model_params']['sequence_length'],
        config['model_params']['prediction_horizon']
    )
    
    # Split the data
    train_size = int(len(X) * config['data_params']['train_test_split'])
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build and train the model
    model = build_model(config, (X_train.shape[1], X_train.shape[2]))
    
    model.fit(
        X_train, y_train,
        validation_split=config['data_params']['validation_split'],
        batch_size=config['model_params']['batch_size'],
        epochs=config['model_params']['epochs'],
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/solar_prediction_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
    )
    
    # Evaluate the model
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    # print(f"Test MAE: {test_mae:.4f}")
    
    # Save the model and scaler
    model.save('models/solar_prediction_model.h5')
    np.save('models/feature_scaler.npy', scaler.scale_)

if __name__ == "__main__":
    main()
