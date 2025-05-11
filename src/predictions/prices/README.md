# Electricity Price Prediction System

This project provides a robust pipeline for forecasting electricity prices and detecting price peaks and valleys using machine learning models. The system is designed for both research and production use, supporting model training, evaluation, and visualization.

## Project Structure

```
src/predictions/prices/
├── train.py         # Training script for all models
├── run_model.py      # Evaluation and visualization script
├── config.py        # Configuration for features, paths, and model parameters
├── utils.py         # Feature engineering and helper functions
├── models/
│   ├── trend_model/                # Trend model artifacts
│   |   |   |── production_model/       # Production-ready model
│   |   |   |── test_model/             # Test/development model
│   ├── peak_model/                 # Peak detection model artifacts
│   |   |   |── production_model/       # Production-ready model
│   |   |   |── test_model/             # Test/development model
│   ├── valley_model/               # Valley detection model artifacts
│   |   |   |── production_model/       # Production-ready model
│   |   |   |── test_model/             # Test/development model
├── plots/                   # Output plots and visualizations
│   ├── evaluation/          # Evaluation plots
│   |   ├── merged/          # Validation plots for merged model performance
│   |   ├── peak/            # Validation plots for peak model performance
│   |   ├── trend/           # Validation plots for trend model performance
│   |   ├── valley/          # Validation plots for valley model performance
│   ├── predictions/         # Future prediction plots
```

## Models Overview

### Trend Model
The trend model predicts the general price movement using XGBoost regression with configurable smoothing levels.

![Trend Model Evaluation](https://github.com/user-attachments/assets/461003ed-d6c6-42c3-ab47-eefdbd23e407)
*Example of trend model prediction showing the underlying price movement pattern*

### Peak Model
The peak model detects price spikes using a Temporal Convolutional Network (TCN) classifier with specialized loss functions.

![Peak Model Evaluation](https://github.com/user-attachments/assets/ffeacde8-33de-401a-9fa7-91595fbf97f7)
*Visualization of peak detection results with prediction probabilities*

### Valley Model
The valley model identifies price drops using a TCN classifier with recall-oriented loss and class balancing techniques.

![Valley Model Evaluation](https://github.com/user-attachments/assets/647af62e-7ba2-41fc-a455-68322d4a89d7)
*Valley detection results showing correctly identified low-price periods*

### Merged Evaluation
The merged approach combines the trend forecast with detected peaks and valleys to simulate realistic price volatility.

![Merged Model Evaluation](https://github.com/user-attachments/assets/3d5c9993-a60f-4b22-a6c1-50fd49c12fa1)
*Complete merged prediction incorporating trend, peaks, and valleys for comprehensive price forecasting*

## Data Preparation

All models use a unified data pipeline that merges price, grid, weather, time, and holiday features. Feature engineering includes lagged values, rolling statistics, and domain-specific indicators.

Peak and valley labels are generated using robust, Scipy-based detection algorithms, ensuring high-quality targets for classification.

## Training Process

Run the training script to train any of the three base models:

```bash
python train.py --model [trend|peak|valley]
```

### Training Options:
- `--model`: Selects which model to train.
- `--production`: (Optional) Trains on all available data for production deployment.
- `--test-mode`: (Optional) Uses a reduced dataset for quick testing.

Artifacts (model weights, scalers, feature lists, and thresholds) are saved in the corresponding `models/` subdirectory.

## Evaluation

Evaluate any trained model or the merged prediction strategy:

```bash
python run_model.py --model [trend|peak|valley|merged] [options]
```

### Evaluation Options:
- `--model`: Specifies the model to evaluate (`trend`, `peak`, `valley`, or `merged`).
- `--weeks N`: Number of weeks to visualize (plots are spaced across the dataset).
- `--test-data`: Evaluate on the test set instead of validation.
- `--valley-threshold X`: Set the probability threshold for valley detection (default: 0.4 for merged model).
- `--peak-threshold X`: Set the probability threshold for peak detection (default: 0.5 for merged model).
- `--optimize-threshold`: Automatically find the threshold that maximizes F1 score for peak or valley models.

### Example Commands:

**Valley Model Evaluation:**
```bash
python run_model.py --model valley --weeks 8 --valley-threshold 0.65
```

**Peak Model Evaluation:**
```bash
python run_model.py --model peak --weeks 8 --peak-threshold 0.90
```

**Trend Model Evaluation:**
```bash
python run_model.py --model trend --weeks 8
```

**Merged Model Evaluation:**
```bash
python run_model.py --model merged --peak-threshold 0.90 --valley-threshold 0.65 --test-data
```

## Trend Model Smoothing

The trend model predictions can be smoothed using different techniques:

| Smoothing Level | Description |
|-----------------|-------------|
| `light`         | Exponential smoothing (α=0.6) + small median filter |
| `medium`        | Exponential smoothing (α=0.3) + median filter + Savitzky-Golay filter |
| `heavy`         | Stronger exponential smoothing (α=0.1) + larger median filter + Savitzky-Golay filter |
| `daily`         | Daily averaging of predictions |
| `weekly`        | Weekly averaging of predictions |

The smoothing level can be configured in the `run_model.py` script using the `SIMPLE_TREND_MODEL_SMOOTHING_LEVEL` variable.

## Peak and Valley Amplitude

In the merged model evaluation, the system applies:

- **Peak Amplitude**: 80% of the average price is added to trend predictions for detected peaks
- **Valley Amplitude**: 80% of the average price is subtracted from trend predictions for detected valleys

The effect is scaled by the detection probability (minimum 75% effect). When both peak and valley are predicted for the same time point, the conflict is resolved based on probability.

## Evaluation Metrics and Outputs

- **Weekly plots**: Shows actual prices, trend predictions, and merged predictions incorporating peaks/valleys
- **Classification metrics**: For peak/valley models: accuracy, precision, recall, F1 score, and ROC AUC
- **Regression metrics**: For trend and merged models: MAE, RMSE
- All outputs are saved in the corresponding `plots/evaluation/[model_type]/` directory

## Customization

- **Feature Engineering:** Modify or extend features in `utils.py` and `config.py`.
- **Model Parameters:** Adjust hyperparameters in `config.py` for each model.
- **Labeling:** The labeling logic for peaks and valleys is in `train.py`.
- **Merging Logic:** The strategy for combining predictions is in `run_model.py`.

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
- Predictions (if a prediction script is added) would typically be saved in `plots/predictions` or a dedicated `output/predictions` directory.

---
