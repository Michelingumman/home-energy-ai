# Price Prediction Module

This module provides functionality for predicting electricity prices in the SE3 region of Sweden using machine learning.

## Overview

The price prediction system uses an LSTM-based neural network to forecast electricity prices up to 24 hours ahead. The model takes into account historical price data, grid conditions, time-based features, and holiday information.

## Key Components

- `predictions.py`: Main prediction class for making forecasts
- `train.py`: Model training pipeline
- `evaluate.py`: Tools for evaluating model performance
- `feature_config.py`: Configuration for features and model architecture
- `gather_price_data.py`: Script for fetching and updating historical price data

## Features Used

- Historical price data (SE3 region)
- Time-based features (hour, day, month, seasonality)
- Grid condition data
- Holiday information
- Rolling statistics (24h and 168h averages, standard deviations)

## Model Details

- Architecture: LSTM-based neural network
- Input: 168-hour (1 week) historical window
- Output: 24-hour price predictions
- Training data: Hourly prices from 1999-06-30 onwards
- Features are scaled using RobustScaler to handle price spikes

## Usage

### Making Predictions

```python
from predictions import PricePredictor

# Initialize predictor
predictor = PricePredictor()

# Predict next 24 hours
predictions = predictor.predict_day("2024-02-20")

# Get predictions for a specific month
monthly_predictions = predictor.predict_month("2024-02")
```

### Evaluation

```python
from evaluate import predict_single_day, predict_month, predict_rolling_forecast

# Evaluate a single day
predict_single_day("2024-02-20")

# Evaluate an entire month
predict_month("2024-02")

# Evaluate with rolling forecasts
predict_rolling_forecast("2024-02-20", horizon=24)
```

## Data Updates

The system automatically updates price data using the mgrey.se API. New data is fetched and processed when running `gather_price_data.py`.

## Performance Metrics

The model's performance is evaluated using:
- MAPE (Mean Absolute Percentage Error)
- RMSE (Root Mean Square Error)

## Dependencies

- TensorFlow
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib 