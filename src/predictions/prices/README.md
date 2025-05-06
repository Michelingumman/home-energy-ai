# Electricity Price Forecasting System

This streamlined codebase provides a simplified structure for electricity price forecasting with three core scripts and a configuration file.

## Core Files

1. **config.py** - Contains all configuration parameters for the models
2. **train.py** - Unified script for training different model types
3. **evaluate.py** - Unified script for evaluating model performance
4. **predict.py** - Unified script for making predictions with trained models
5. **utils.py**

## Model Types

The system supports three types of models:

1. **Trend Model** - Predicts the general trend of electricity prices
2. **Peak Model** - Detects price peaks (periods of unusually high prices)
3. **Valley Model** - Detects price valleys (periods of unusually low prices)

## Usage Instructions

### Training Models

```bash
# Train the trend model
python train.py --model trend

# Train the peak detection model
python train.py --model peak

# Train the valley detection model
python train.py --model valley
```

### Evaluating Models

```bash
# Evaluate the trend model
python evaluate.py --model trend

# Evaluate the peak detection model
python evaluate.py --model peak

# Evaluate the valley detection model
python evaluate.py --model valley

# Evaluate all models and generate combined predictions
python evaluate.py --model all
```

### Making Predictions

```bash
# Make predictions with the trend model
python predict.py --model trend --horizon 24 --show_plot

# Make predictions with the peak detection model
python predict.py --model peak --horizon 24 --show_plot

# Make predictions with the valley detection model
python predict.py --model valley --horizon 24 --show_plot

# Make predictions with all models and combine them
python predict.py --model all --horizon 24 --show_plot

# Run in production mode and save predictions to a file
python predict.py --model all --horizon 24 --production_mode --output_file predictions.csv
```

## Model Architecture

All models are based on Temporal Convolutional Networks (TCN) with different configurations:

- **Trend Model**: Regression model that predicts actual price values
- **Peak/Valley Models**: Binary classification models that predict the probability of price spikes or valleys

## Data Requirements

The system expects the following data files in the `data/processed` directory:

- `SE3prices.csv` - Historical electricity prices
- `SwedenGrid.csv` - Power grid data
- `time_features.csv` - Time-based features
- `holidays.csv` - Holiday information
- `weather_data.csv` - Weather data

## Output

- Trained models are saved in the `models` directory
- Evaluation results and plots are saved in the `plots/evaluation` directory
- Predictions are saved in the `plots/predictions` directory 