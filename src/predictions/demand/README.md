# Home Energy Demand Prediction

This module provides a machine learning solution for predicting home energy demand using XGBoost. The codebase consists of three main components: training, evaluation, and prediction.

## Overview

The system predicts energy demand (consumption) for the next hour based on:
- Historical consumption data
- Heat pump power data
- Weather data
- Time and calendar features
- Hidden Markov Model (HMM) occupancy states

## Files and Functionality

### 1 - `train.py`

Trains an XGBoost regression model for energy demand prediction.

- Loads and processes data from multiple sources
- Creates an HMM model to detect occupancy states
- Engineers features including time, weather, and lag features
- Uses TimeSeriesSplit for proper time series cross-validation
- Performs hyperparameter optimization using Optuna
- Saves the trained model and split information for later use

```bash
python train.py
```

### 2 - `evaluate.py`

Evaluates the trained model on test data and generates visualizations.

- Loads the trained model and testing data
- Computes key metrics (MAE, RMSE, MAPE)
- Compares model performance against a persistence baseline
- Creates various plots: residuals analysis, time series comparisons
- Daily date ticks with angled labels for improved readability
- Uses modern matplotlib colormap API for better visualizations
- Optimized DataFrame handling for improved performance
- Supports evaluating on specific time periods

```bash
# Basic evaluation on held-out data
python evaluate.py

# Plot predictions for a specific month
python evaluate.py --plot 2025-05

# Show the comprehensive dashboard with additional analysis panels
python evaluate.py --plot 2025-05 --dashboard

# Save plots to disk instead of displaying
python evaluate.py --plot 2025-05 --save

# Set custom evaluation ratio
python evaluate.py --eval-ratio 0.3
```

### 3 - `predict.py`

Makes predictions using the trained model for future or arbitrary time periods.

## Data Pipeline

1. **Data Loading**: Consumption data, heat pump data, and weather data are loaded and merged
2. **HMM Occupancy Detection**: A Hidden Markov Model identifies occupancy patterns
3. **Feature Engineering**:
   - Time features: hour of day, day of week, month (using sine/cosine encoding)
   - Calendar features: weekends, holidays
   - Weather features: temperature, humidity, cloud cover, etc.
   - Lagged features: consumption from previous hours/days
   - Rolling statistics: 24-hour and 7-day moving averages
   - Interaction features: combinations of HMM states with other features

## Model Training Process

1. **Data Preparation**: Raw data is processed and split into train/validation sets
2. **Hyperparameter Optimization**: Optuna is used to find optimal model parameters
3. **Final Model Training**: The model is trained on training data with early stopping
4. **Model Saving**: Model and its metadata are saved for later use

## Feature Importance and Insights

The model automatically captures important patterns affecting energy demand:
- Temporal patterns (daily, weekly cycles)
- Weather dependency (heating/cooling degree hours)
- Occupancy patterns (via HMM states)

## Evaluation and Visualization

The evaluation module provides:
- Overall metrics on test data
- Comparison with persistence baseline
- Visualizations of actual vs. predicted demand
- Daily date ticks with angled labels for better readability
- Calendar-based plots (month, week, day views)
- Residual analysis plots

## Logging and Metadata

The system logs detailed information during training and evaluation:
- Data splits and date ranges
- Hyperparameter optimization progress
- Model performance metrics
- Warnings about data overlaps between training and testing

This makes it easy to track model performance and ensure proper validation.