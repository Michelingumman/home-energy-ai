# Electricity Price Prediction Module

This module provides tools for training, evaluating, and deploying price prediction models for electricity markets.

## Overview

The price prediction module uses LSTM neural networks to forecast electricity prices for the next 24 hours based on:
- Historical price data
- Time-based features (hour, day, month, etc.)
- Holiday information
- Grid data (import/export, production)

## Quick Start

### Training a Model

```bash
# Train a production model (using all data)
python train.py production

# Train an evaluation model (with test split)
python train.py evaluation
```

### Evaluating a Model

```bash
# Run evaluation on a trained model
python evaluate.py
```

## Training Options

The training script (`train.py`) supports two modes:

### 1. Production Mode

Trains using all available data with a small validation set for monitoring. This creates a model ready for deployment.

```bash
python train.py production [--scaler SCALER_TYPE] [--window WINDOW_SIZE]
```

### 2. Evaluation Mode

Creates a train/validation/test split to assess model performance. Test data is saved for later evaluation.

```bash
python train.py evaluation [--scaler SCALER_TYPE] [--window WINDOW_SIZE]
```

### Options

- `--scaler`: Type of scaler for feature normalization
  - `robust`: RobustScaler (handles outliers, default)
  - `minmax`: MinMaxScaler (preserves feature magnitude)
  - `standard`: StandardScaler (normalizes to mean=0, std=1)
- `--window`: Override window size (hours of history for prediction)

## Evaluation

The evaluation script (`evaluate.py`) creates visualizations and metrics to assess model performance:

- Daily, weekly, monthly, and yearly comparisons of actual vs. predicted prices
- Error metrics (MAE, RMSE, etc.)
- Visual charts of prediction accuracy

```bash
python evaluate.py
```

## Directory Structure

```
src/predictions/prices/
├── train.py           # Training script
├── evaluate.py        # Evaluation script
├── feature_config.py  # Feature configuration
├── models/            # Saved models and data
│   ├── production/    # Production model (all data)
│   │   ├── saved/     # Model files and scalers
│   │   └── logs/      # TensorBoard logs
│   └── evaluation/    # Evaluation model (with test split)
│       ├── saved/     # Model files and scalers
│       ├── logs/      # TensorBoard logs
│       └── test_data/ # Test data for evaluation
└── README.md          # This file
```

## Advanced Usage

### Model and Scaler Files

Both training modes save:
- Trained model file (`.keras`)
- Price scaler (`.save`)
- Grid scaler (`.save`)
- Target information (`.json`)
- Feature configuration (`.json`)

### TensorBoard Monitoring

Training progress can be monitored with TensorBoard:

```bash
tensorboard --logdir src/predictions/prices/models/evaluation/logs
```

### Customizing Features

Features used for training are configured in `feature_config.py`. This includes:
- Price features
- Grid features (import/export data)
- Time-based features (hour of day, day of week, etc.)
- Holiday features

## Troubleshooting

### Scaling Issues

If predictions show unexpected scale, check:
1. The saved scalers match the input data range
2. Target column index is correctly specified in `target_info.json`
3. Feature order is consistent between training and prediction

### Memory Errors

For large datasets:
1. Reduce batch size
2. Limit the window size
3. Train on a subset of data first

## References

- Features based on Swedish electricity market (SE3 price area)
- Uses hourly resolution data
- LSTM architecture adapted for time series forecasting 