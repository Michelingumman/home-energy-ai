# Electricity Price Prediction

This module provides a machine learning-based electricity price forecasting system for Sweden's SE3 price area. It uses an LSTM-based neural network to predict hourly electricity prices for the next 24 hours.

## Key Features

- **24-hour Price Forecasting**: Predicts prices for the next 24 hours with high accuracy
- **Feature Engineering**: Combines price history, time features, grid data, and holidays
- **Grid Data Integration**: Uses data from the Swedish power grid including import/export flows
- **Enhanced Scaling**: Advanced scaling techniques for handling high-magnitude grid features
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

## Advanced Scaling Enhancements

The system now includes specialized handling for high-magnitude grid features:

- **Signed Log Transformation**: Handles both positive and negative values using sign(x) * log(|x| + 1)
- **Log Transformation**: Optional log(x+1) transformation for import/export features
- **Individual Column Scaling**: Option to scale problematic columns individually
- **Winsorization**: Capping of extreme values based on configurable Z-score thresholds
- **Robust Scaling**: RobustScaler with custom quantile parameters for handling outliers
- **Unit Variance Normalization**: Normalizes data to have unit variance

These scaling improvements can be configured in the `config.json` file under the `scaling` section.

## Usage

### Training a Model

```bash
# Train a production model (uses all available data)
python src/predictions/prices/train.py production --scaler robust

# Train an evaluation model (with train/val/test split)
python src/predictions/prices/train.py evaluation --scaler robust
```

### Making Predictions

```python
from src.predictions.prices.predictions import PricePredictor

# Initialize predictor
predictor = PricePredictor()

# Get predictions for next 24 hours
predictions = predictor.predict_next_day()
print(predictions)

# Predict for a specific date
predictions = predictor.predict_for_date('2023-06-01')
```

## Configuration

The `config.json` file contains all model parameters and feature definitions:

- Feature groups and ordering
- Model architecture (LSTM and dense layers)
- Training parameters (batch size, learning rate, etc.)
- Scaling configuration
- Evaluation settings

Example scaling configuration:

```json
"scaling": {
    "price_scaler": {
        "type": "robust", 
        "quantile_range": [1, 99]
    },
    "grid_scaler": {
        "type": "robust",
        "quantile_range": [1, 99],
        "unit_variance": true,
        "outlier_threshold": 10,
        "handle_extreme_values": true,
        "log_transform_large_values": true,
        "signed_log": true,
        "individual_scaling": true
    }
}
```

### Handling Negative Values

The system is designed to properly handle negative values that can occur in price and grid data:

1. **Negative Energy Prices**: During periods of excess production, electricity prices can go negative. Our model correctly handles these scenarios through:
   - Signed log transformation for features with negative values
   - RobustScaler which handles both positive and negative inputs appropriately

2. **Bidirectional Power Flows**: For import/export features, the model can handle negative values that might represent flow reversals using the signed log transform approach.

## Evaluation Metrics

The model evaluation produces several metrics:

- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- RMSE (Root Mean Squared Error)
- Hourly error breakdowns
- Visual analysis (scatter plots, error distributions, sample predictions)

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