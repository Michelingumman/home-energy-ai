# Home Energy Prediction System

This folder contains scripts for predicting home energy consumption.

## FetchConsumptionData.py

This script fetches energy consumption data from the Tibber API and saves it to a CSV file for further analysis.

### Features

- Fetches historical consumption data from Tibber
- Saves data to a CSV file (`VillamichelinConsumption.csv`)
- Handles pagination for large datasets
- Only fetches new data when run again (avoids redundant API calls)
- Error handling and retry logic
- Includes detailed logging

### Usage

```bash
python FetchConsumptionData.py
```

### Requirements

- Tibber API Token (set as `TIBBER_API_TOKEN` environment variable)
- Python 3.x
- Required packages: pandas, requests, python-dotenv

## PredictDemand.py

This script uses machine learning (XGBoost) to predict future energy consumption based on historical data, weather data, and time features.

### Features

- Loads consumption data from CSV
- Integrates weather data, time features, and holiday information
- Creates lag features for time series forecasting
- Trains an XGBoost model for energy demand prediction
- Provides comprehensive model evaluation
- Generates visualizations of predictions and model performance
- Makes 24-hour forecasts for future consumption

### Usage

```bash
python PredictDemand.py
```

### Model Features

The model uses multiple feature sources:

1. **Weather Features**:
   - Temperature
   - Cloud cover
   - Humidity
   - Wind speed and direction
   - Solar radiation

2. **Time Features**:
   - Hour of day (encoded as sine/cosine)
   - Day of week (encoded as sine/cosine)
   - Month (encoded as sine/cosine)
   - Season
   - Is weekend flag
   - Peak time indicators (morning/evening)

3. **Holiday Features**:
   - Is holiday flag
   - Is holiday eve flag
   - Days to next holiday
   - Days since last holiday

4. **Lag Features**:
   - Previous hour consumption
   - Previous day same hour consumption
   - Previous week same hour consumption

### Outputs

- Trained XGBoost model (saved to `models/xgboost_model.pkl`)
- Performance metrics (MAE, RMSE, R²)
- Weekly evaluation plots
- Residual analysis plots
- 24-hour forecast

## CnnEnergyModel.py

This script implements a Convolutional Neural Network (CNN) approach to energy demand forecasting, separate from the XGBoost implementation.

### Features

- Uses a deep learning approach with CNN architecture
- Processes time series data as sequences for better pattern recognition
- Creates 3D tensor inputs (samples, time steps, features)
- Includes regularization techniques to prevent overfitting
- Provides comprehensive evaluation metrics and visualizations
- Makes 24-hour forecasts using the CNN model
- Can be compared with XGBoost results

### Usage

```bash
python CnnEnergyModel.py
```

### CNN Model Architecture

The model uses a multi-layer CNN architecture:
- Multiple convolutional layers with batch normalization
- MaxPooling layers for dimensionality reduction
- Dropout layers for regularization
- Dense layers for final regression output

### Outputs

- Trained CNN model (saved to `models/cnn_model.h5`)
- Training history visualization
- Performance metrics (MAE, RMSE, R², MAPE)
- Prediction vs. actual plots
- Error distribution analysis
- Hourly error pattern analysis
- 24-hour forecast

## Comparison of Models

You can run both models independently and compare their performance:
- XGBoost typically excels at feature importance and handling diverse feature types
- CNN may capture more complex temporal patterns and dependencies

The scripts will generate metrics and visualizations that allow direct comparison of model performance.

# Energy Demand Prediction with XGBoost

This module provides an XGBoost-based energy demand prediction system with configurable features and weights.

## Features

- **Configurable Feature Selection**: Easily enable or disable features through the configuration file.
- **Feature Weighting**: Apply custom weights to features to influence the model's decision-making.
- **Time Series Cross-Validation**: Uses proper time series split to prevent data leakage.
- **Lag Feature Generation**: Creates lag features from historical data for better prediction.
- **Automated Visualization**: Creates visualizations for model evaluation and feature importance.
- **Command-Line Interface**: Provides an easy-to-use CLI for model configuration and training.

## Configuration

The model's behavior is controlled by `config.yaml`, which contains:

- Feature settings (enabled/disabled and weights)
- Model parameters
- Data split settings

You can edit this file directly or use the CLI tools to modify settings.

## Usage

### Command-Line Interface

```bash
# List all available features and their current settings
python PredictDemand.py --list-features

# Enable a feature
python PredictDemand.py --enable-feature temperature

# Disable a feature
python PredictDemand.py --disable-feature humidity

# Set a custom weight for a feature
python PredictDemand.py --set-weight temperature 1.5

# Update a model parameter
python PredictDemand.py --update-model-param max_depth 6

# Open the configuration file in your default editor
python PredictDemand.py --update-config

# Run the full model training and evaluation pipeline
python PredictDemand.py --run

# Generate predictions for the next 24 hours
python PredictDemand.py --predict
```

### Feature Weights

Feature weights allow you to control the influence of each feature on the model's predictions:

- A weight of 1.0 is neutral (normal influence)
- Weights > 1.0 increase a feature's influence
- Weights < 1.0 decrease a feature's influence

For example, if you believe temperature is very important for energy consumption patterns, you might set its weight to 1.5 or higher.

### Available Features

The following features can be configured:

#### Time Features
- `hour`: Hour of the day (0-23)
- `day_of_week`: Day of the week (0-6, where 0 is Monday)
- `month`: Month of the year (1-12)
- `is_weekend`: Whether the day is a weekend (True/False)
- `season`: Season of the year (0=Winter, 1=Spring, 2=Summer, 3=Fall)

#### Weather Features
- `temperature`: Ambient temperature
- `humidity`: Relative humidity
- `clouds`: Cloud cover
- `wind_speed`: Wind speed
- `precipitation`: Precipitation amount

#### Holiday Features
- `is_holiday`: Whether the day is a holiday
- `is_holiday_eve`: Whether the day is the eve of a holiday
- `days_to_next_holiday`: Number of days until the next holiday
- `days_from_last_holiday`: Number of days since the last holiday

#### Lag Features
- `consumption_lag_1h`: Consumption from 1 hour ago
- `consumption_lag_24h`: Consumption from 24 hours ago
- `consumption_lag_48h`: Consumption from 48 hours ago
- `consumption_lag_168h`: Consumption from 168 hours ago (1 week)

#### Rolling Statistics
- `consumption_rolling_mean_24h`: Average consumption over the past 24 hours
- `consumption_rolling_mean_168h`: Average consumption over the past week

## Output

The model generates several outputs in the `models` directory:

- Trained XGBoost model (`xgboost_model.pkl`)
- Model configuration (`model_config.yaml`)
- Feature importance plot and CSV
- Residual analysis plots
- Weekly evaluation plots
- 24-hour forecast and plot

## Best Practices

1. Start with all features enabled and default weights.
2. Run the model and evaluate feature importance.
3. Gradually adjust weights based on domain knowledge and observed importance.
4. Disable features with very low importance to simplify the model.
5. Fine-tune model parameters after feature selection.
