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
│   ├── trend_model/                # Saved peak model artifacts
│   |   |   |── production_model/       # Plot of the generated peak labels for visualization
│   |   |   |── test_model/             # Plot of the generated peak labels for visualization
│   ├── peak_model/                 # Saved peak model artifacts
│   |   |   |── production_model/       # Plot of the generated peak labels for visualization
│   |   |   |── test_model/             # Plot of the generated peak labels for visualization
│   ├── valley_model/               # Saved peak model artifacts
│   |   |   |── production_model/       # Plot of the generated peak labels for visualization
│   |   |   |── test_model/             # Plot of the generated peak labels for visualization
├── plots/                   # Output plots and visualizations
│   ├── evaluation/          # Evaluation plots
│   |   ├── merged/          # Validation plots for merged model performance (trend+peak+valley)
│   |   ├── peak/            # Validation plots for peak model performance
│   |   ├── trend/           # Validation plots for trend model performance
│   |   ├── valley/          # Validation plots for valley model performance
│   ├── predictions/         # Future prediction plots
```

## Models

- **Trend Model:** Predicts the general price trend using XGBoost regression with configurable smoothing levels.
![image](https://github.com/user-attachments/assets/461003ed-d6c6-42c3-ab47-eefdbd23e407)

- **Peak Model:** Detects price spikes (peaks) using a Temporal Convolutional Network (TCN) classifier.
![image](https://github.com/user-attachments/assets/ffeacde8-33de-401a-9fa7-91595fbf97f7)

- **Valley Model:** Detects price valleys using a TCN classifier with recall-oriented loss and class balancing.
![image](https://github.com/user-attachments/assets/74e67aa1-5969-40cd-b2c8-5452acf53265)

- **Merged Evaluation:** Combines the trend forecast with detected peaks and valleys to simulate realistic price volatility.
![image](https://github.com/user-attachments/assets/3d5c9993-a60f-4b22-a6c1-50fd49c12fa1)


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
- `--weeks N`: Number of weeks to visualize (plots are spaced across the dataset).
- `--test-data`: Evaluate on the test set instead of validation.
- `--valley-threshold X`: Set the probability threshold for valley detection (default: 0.4 for merged model).
- `--peak-threshold X`: Set the probability threshold for peak detection (default: 0.5 for merged model).
- `--optimize-threshold`: Automatically find the threshold that maximizes F1 score for peak or valley models.

**Examples:**
Evaluate the valley model with a specific threshold:
```bash
python evaluate.py --model valley --weeks 8 --valley-threshold 0.65
```
Evaluate the peak model with a specific threshold:
```bash
python evaluate.py --model peak --weeks 8 --peak-threshold 0.90
```
Evaluate the trend model:
```bash
python evaluate.py --model trend --weeks 8
```
Evaluate the merged model using specific thresholds for peaks and valleys:
```bash
python evaluate.py --model merged --peak-threshold 0.90 --valley-threshold 0.65 --test-data
```

## Trend Model Smoothing

The trend model predictions can be smoothed using different techniques:

- `light`: Exponential smoothing (α=0.6) + small median filter
- `medium`: Exponential smoothing (α=0.3) + median filter + Savitzky-Golay filter
- `heavy`: Stronger exponential smoothing (α=0.1) + larger median filter + Savitzky-Golay filter
- `daily`: Daily averaging of predictions
- `weekly`: Weekly averaging of predictions

The smoothing level can be configured in the `evaluate.py` script using the `SIMPLE_TREND_MODEL_SMOOTHING_LEVEL` variable.

## Peak and Valley Amplitude

In the merged model evaluation, the script applies:
- **Peak Amplitude**: 80% of the average price is added to trend predictions for detected peaks
- **Valley Amplitude**: 80% of the average price is subtracted from trend predictions for detected valleys

The effect is scaled by the detection probability (minimum 75% effect). For cases where both peak and valley are predicted, the model resolves conflicts based on probability.

## Evaluation Metrics and Outputs

- **Weekly plots**: Shows actual prices, trend predictions, and merged predictions incorporating peaks/valleys
- **Classification metrics**: For peak/valley models: accuracy, precision, recall, F1 score, and ROC AUC
- **Regression metrics**: For trend and merged models: MAE, RMSE
- All outputs are saved in the corresponding `plots/evaluation/[model_type]/` directory

## Customization

- **Feature Engineering:** Modify or extend features in `utils.py` and `config.py`.
- **Model Parameters:** Adjust hyperparameters in `config.py` for each model.
- **Labeling:** The labeling logic for peaks and valleys is in `train.py`.
- **Merging Logic:** The strategy for combining predictions is in `evaluate.py`.

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
