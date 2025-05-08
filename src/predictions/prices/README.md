# Electricity Price Prediction System

This project provides a robust pipeline for forecasting electricity prices and detecting price peaks and valleys using machine learning models. The system is designed for both research and production use, supporting model training, evaluation, and visualization.

## Project Structure

```
src/predictions/prices/
├── train.py         # Training script for all models
├── evaluate.py      # Evaluation and visualization script
├── config.py        # Configuration for features, paths, and model parameters
├── utils.py         # Feature engineering and helper functions
├── models/
│   ├── trend_model/            # Saved trend model artifacts
│   ├── peak_model/             # Saved peak model artifacts
│   |   |── peak_validation/    # plot of the generated peak labels for visualization
│   └── valley_model/           # Saved valley model artifacts
│   |   |── valley_validation/  # plot of the generated peak labels for visualization
├── data/                    # Input data files (prices, grid, weather, etc.)
└── plots/                   # Output plots and visualizations
│   ├── evaluation/   # evaluation plots
│   |   ├── merged/    # validation plots for visual performance of the trend+peak+valley model
│   |   ├── peak/      # validation plots for visual performance
│   |   ├── trend/     # validation plots for visual performance
│   |   ├── valley/    # validation plots for visual performance
│   ├── predictions/   # future prediction plots
```

## Models

- **Trend Model:** Predicts the general price trend using XGBoost regression with changeable smoothing mode.
- **Peak Model:** Detects price spikes (peaks) using a Temporal Convolutional Network (TCN) classifier.
- **Valley Model:** Detects price valleys using a TCN classifier with recall-oriented loss and class balancing.
- **Merged Evaluation:** The `evaluate.py` script can also assess a combined prediction that merges the trend forecast with detected peaks and valleys to simulate realistic price volatility.

## Data Preparation

All models use a unified data pipeline that merges price, grid, weather, time, and holiday features. Feature engineering includes lagged values, rolling statistics, and domain-specific indicators.

Peak and valley labels are generated using robust, Scipy-based detection algorithms, ensuring high-quality targets for classification.

## Training

Run the training script to train any of the three base models:

```bash
python train.py --model [trend|peak|valley]
```

- `--model`: Selects which model to train.
- `--production`: (Optional) Trains on all available data for production deployment.
- `--test-mode`: (Optional) Uses a reduced dataset for quick testing.

Artifacts (model weights, scalers, feature lists, and thresholds) are saved in the corresponding `models/` subdirectory.

During training, the script also generates validation plots for peak and valley detection, as well as method comparison plots for label quality assurance.

## Evaluation

Evaluate any trained model or the merged prediction strategy and generate detailed weekly plots and metrics:

```bash
python evaluate.py --model [trend|peak|valley|merged] [options]
```

**Key options:**
- `--model`: Specifies the model to evaluate (`trend`, `peak`, `valley`, or `merged`).
- `--weeks N` : Number of weeks to visualize (plots are spaced across the dataset).
- `--test-data` : Evaluate on the test set instead of validation.
- `--valley-threshold X` : Set the probability threshold for valley detection. For the `merged` model, this also serves as the default threshold.
- `--peak-threshold X` : Set a separate probability threshold specifically for peak detection
- `--optimize-threshold` : Automatically find the threshold that maximizes F1 score for peak or valley models (not applicable to `merged` directly, but influences the underlying peak/valley models if they are re-evaluated with this option).

**Examples:**
Evaluate the valley model for 8 weeks with a specific threshold:
```bash
python evaluate.py --model valley --weeks 8 --valley-threshold 0.65
```
Evaluate the peak model for 8 weeks with a specific threshold:
```bash
python evaluate.py --model peak --weeks 8 --valley-threshold 0.90
```
Evaluate the trend model for 8 weeks:
```bash
python evaluate.py --model trend --weeks 8
```
Evaluate the merged model using specific thresholds for peaks and valleys for new data (test data is closer to current date):
```bash
python evaluate.py --model merged --weeks 4 --peak-threshold 0.90 --valley-threshold 0.65 --test-data
```

Evaluation produces:
- Weekly plots showing price, predictions, and relevant scores.
- For `peak` and `valley` models: actual vs. predicted events, and probability scores.
- For the `merged` model: plots comparing actual price, base trend, and the merged prediction line that incorporates peak/valley volatility. Also includes probability plots for peak and valley detection.
- Comprehensive metrics including accuracy, precision, recall, F1, ROC AUC, and class rates for classification models (peak/valley). For the trend and merged models, MAE, RMSE, and directional accuracy are provided.
- All outputs are saved in the corresponding `plots/evaluation/[model_type]/` directory.

## Customization

- **Feature Engineering:** Modify or extend features in `utils.py` and `config.py`.
- **Model Parameters:** Adjust hyperparameters in `config.py` for each model.
- **Labeling:** The labeling logic for peaks and valleys is in `train.py` and uses robust Scipy-based methods for consistency.
- **Merging Logic:** The strategy for combining trend, peak, and valley predictions in the `merged` evaluation (including amplitude and smoothing) is within `evaluate.py`.

## File Overview

- **train.py**: Main entry point for training. Handles data loading, feature engineering, model fitting, and artifact saving.
- **evaluate.py**: Main entry point for evaluation. Loads models and artifacts, applies them to validation/test data, computes metrics, and generates plots. Includes logic for evaluating individual models and the merged prediction strategy.
- **config.py**: Centralized configuration for paths, features, and model parameters.
- **utils.py**: Helper functions for feature engineering, sequence creation, and label generation.

## Data Requirements

The system expects the following data files in the `data/processed` directory (Note: `data/processed` is a common convention, adapt if your structure is `data/` directly):

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