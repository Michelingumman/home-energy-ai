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

#### Agent Mode
To train a model specifically for use with the RL agent, use the `--for-agent` flag:

```bash
python train.py --for-agent --trials 30
```

This will train a model to predict baseload consumption with the following enhancements:
- **Improved baseload calculation**: Accounts for grid import + solar production - battery discharge
- **Enhanced feature engineering**: Includes day-of-week and hour-specific patterns
- **Advanced weather interactions**: Non-linear temperature effects and weather condition features
- **Heat pump usage patterns**: Time and temperature-dependent heat pump consumption patterns
- **Solar/temperature interactions**: Features capturing baseload behavior during different radiation levels

The `--trials` parameter controls the number of hyperparameter optimization runs (default: 5).

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

#### Agent Mode
To evaluate a model trained for agent use, use the `--for-agent` flag:

```bash
python evaluate.py --for-agent
```

This will evaluate the baseload prediction model instead of the standard demand model. You can also evaluate specific months:

```bash
python evaluate.py --for-agent --month 2025-05 --show-hmm --show-dashboard
```

For deeper analysis of baseload model performance, use the `--deep-analysis` flag:

```bash
python evaluate.py --for-agent --deep-analysis
```

### 3 - `predict.py`

Makes predictions using the trained model for future or arbitrary time periods.

#### Agent Mode
To make predictions with a model trained for agent use, use the `--for-agent` flag:

```bash
python predict.py --for-agent
```

This will use the baseload model for predictions instead of the standard demand model.

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

## Agent vs. Standard Mode

The demand prediction system offers two distinct training and prediction modes:

### Standard Mode (Default)
- Predicts **net consumption** from the grid
- Includes effects from solar production and battery usage
- Suitable for general energy monitoring and reporting

### Agent Mode (`--for-agent`)
- Predicts **baseload** (pure household demand)
- Calculated as grid consumption + solar production - battery discharge
- Excludes effects of battery charge/discharge to avoid circular dependencies
- Designed specifically for RL agent decision-making
- Enhanced with specialized features for baseload modeling:
  - Day-of-week and hour-specific consumption patterns
  - Heat pump energy usage during different temperature ranges (cold/normal/warm)
  - Temperature variations during different parts of the day (morning/evening/night)
  - Enhanced weather condition complexes (heat index, discomfort index)
  - Advanced time lag features with multiple time horizons

When training for agent use, specify the `--for-agent` flag to train on baseload instead of net consumption:
```bash
python train.py --for-agent
```

Similarly, when making predictions or evaluating for agent use:
```bash
python predict.py --for-agent
python evaluate.py --for-agent
```

## Model Training Process

1. **Data Preparation**: Raw data is processed and split into train/validation sets
2. **Hyperparameter Optimization**: Optuna is used to find optimal model parameters
3. **Final Model Training**: The model is trained on training data with early stopping
4. **Model Saving**: Model and its metadata are saved for later use

## Advanced Feature Engineering for Baseload

The enhanced baseload model uses several specialized feature categories:

### Weather Transformations
- Heating and cooling degree days (HDD/CDD)
- Non-linear temperature effects (squared terms)
- Daily temperature ranges 
- Temperature change rates
- Heat index and discomfort index for comfort modeling
- Weather severity index combining multiple parameters
- Dew point approximation

### Time-Based Patterns
- Hour-of-day patterns for each day of the week
- Morning/evening peak detectors
- Weekend vs. weekday patterns
- Holiday effects

### Heat Pump Interactions
- Heat pump usage during different temperature ranges
- Heat pump usage during different times of day
- Heat pump interaction with occupancy states

### Lag Features
- Multiple time horizons (hourly, daily, weekly)
- Statistical aggregations (mean, median, min, max, std)
- Exponentially weighted moving averages
- Week-over-week ratios for each hour
- Day-of-week and hour-specific averages

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