# Price Prediction Module

This module provides functionality for predicting electricity prices in the SE3 region of Sweden using machine learning.

## Overview

The price prediction system uses an LSTM-based neural network to forecast electricity prices up to 24 hours ahead. The model takes into account historical price data, grid conditions, time-based features, and holiday information.

## Key Components

- `predictions.py`: Main prediction class for making forecasts
- `train.py`: Model training pipeline with support for both evaluation and production models
- `evaluate.py`: Comprehensive model evaluation with automated visualizations
- `config.json`: Configuration for features and model architecture
- `gather_data.py`: Scripts for fetching and updating historical price and grid data

## Features Used

- Historical price data (SE3 region)
- Time-based features (hour, day, month, seasonality)
- Grid condition data (production, consumption, imports/exports)
- Holiday information
- Rolling statistics (24h and 168h averages, standard deviations)

## Model Details

- Architecture: LSTM-based neural network
- Input: 168-hour (1 week) historical window
- Output: 24-hour price predictions
- Training data: Hourly prices and grid data
- Features are scaled using RobustScaler to handle price spikes

## Usage

### Making Predictions

```python
from predictions import PricePredictor

# Initialize predictor
predictor = PricePredictor()

# Predict next 24 hours
predictions = predictor.predict_day()

# Predict the coming week
weekly_predictions = predictor.predict_week()

# Get predictions for a specific date range
custom_predictions = predictor.predict_range("2024-02-20", "2024-02-27")
```

### Evaluation

The evaluation system now provides a comprehensive, automatic analysis without requiring command-line arguments:

```bash
# Run the comprehensive evaluation
python evaluate.py
```

This generates three focused visualization figures:

1. **Time Period Analysis** (`time_period_analysis.png`):
   - Full test period overview
   - Yearly performance comparisons
   - Monthly price patterns
   - Weekly detail with weekend highlights

2. **Day Analysis** (`day_comparison.png`):
   - Representative day comparisons (low price, high price, volatile)
   - Detailed hourly patterns for each day type

3. **Error Analysis** (`error_analysis.png`):
   - Error distribution and statistics
   - Error patterns by hour of day
   - Residual analysis
   - Actual vs. predicted scatter plots

All results are automatically saved to `models/evaluation/results/comprehensive_evaluation/` along with detailed metrics in both visual and text formats.

## Data Updates

The system automatically updates price and grid data using external APIs:
- Electricity price data from mgrey.se API
- Grid data from Electricity Maps API

Data is fetched and processed when running `gather_data.py`.

## Performance Metrics

The model's performance is evaluated using:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)
- Error bias and distribution analysis
- Time-based performance (yearly, monthly, hourly)

## Dependencies

- TensorFlow (2.x)
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Seaborn 