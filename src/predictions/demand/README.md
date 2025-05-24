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

## Recent Issues Fixed

### 1. Length Mismatch Error (✅ RESOLVED)
- **Issue**: `ValueError: Found input variables with inconsistent numbers of samples: [4675, 4677]`
- **Root Cause**: Inconsistent data handling between evaluation and plotting functions
- **Solution**: Implemented consistent data cleaning and NaN handling throughout the pipeline

### 2. Data Leakage Warning (✅ RESOLVED)
- **Issue**: Evaluation data overlapped with model training data
- **Root Cause**: Insufficient post-training data for meaningful evaluation
- **Solution**: Added intelligent evaluation period selection with fallback to ratio-based split

### 3. Feature Importance Extraction (✅ RESOLVED)
- **Issue**: XGBoost model couldn't retrieve feature importance
- **Root Cause**: Incorrect method calls for feature importance extraction
- **Solution**: Implemented multiple fallback methods for feature importance extraction

## Current Model Performance Analysis

### Metrics (as of latest evaluation):
- **XGBoost Model**: MAE=0.9742, RMSE=1.3878, MAPE=888.72%
- **Persistence Baseline**: MAE=1.4937, RMSE=2.3560, MAPE=1018.68%

### Key Observations:
1. **High MAPE values** indicate significant relative prediction errors
2. **Model outperforms persistence** in absolute terms (MAE, RMSE)
3. **Very low baseline consumption** makes MAPE artificially high
4. **Data leakage concerns** due to limited post-training evaluation data

## Improvement Strategy

### Phase 1: Data Quality and Preprocessing Improvements

#### 1.1 Target Variable Refinement
```python
# Current approach creates very low values leading to high MAPE
# Consider alternative target formulations:

# Option A: Log-transform for low values
y_log = np.log1p(consumption_data)

# Option B: Scaled target to avoid extreme MAPE
y_scaled = (consumption_data - consumption_data.min()) / consumption_data.std()

# Option C: Focus on relative changes rather than absolute values
y_relative = consumption_data.pct_change()
```

#### 1.2 Enhanced Data Preprocessing
- **Reduced aggressive outlier capping** (now using 6σ instead of 5σ)
- **IQR-based outlier detection** for skewed distributions
- **Median imputation** instead of mean for robustness
- **Conservative extreme value thresholds** (only cap if >1% of data)

#### 1.3 Improved Feature Engineering
Currently generates 220+ features including:
- 53 lagged and time series features
- 25 calendar features
- 25 weather transformation features
- 58 interaction features
- 50 heat pump baseload features

**Recommendations**:
- Feature selection to reduce dimensionality
- Principal Component Analysis (PCA) for correlated features
- Recursive Feature Elimination (RFE)

### Phase 2: Model Architecture Improvements

#### 2.1 Alternative Model Architectures
```python
# Current: XGBoost with standard hyperparameters
# Consider:

# Option A: Ensemble approach
from sklearn.ensemble import VotingRegressor
ensemble = VotingRegressor([
    ('xgb', XGBRegressor()),
    ('rf', RandomForestRegressor()),
    ('gbm', GradientBoostingRegressor())
])

# Option B: Time series specific models
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import ExtraTreesRegressor

# Option C: Neural network for complex patterns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
```

#### 2.2 Enhanced Hyperparameter Optimization
```python
# Current: Basic Optuna optimization
# Improvements:
def objective_heat_pump_baseload_v2(trial, X, y, cv):
    """
    Enhanced objective function with:
    1. Multi-objective optimization (MAE + MAPE)
    2. Seasonal performance weighting
    3. Time-of-day specific optimization
    4. Heating/cooling season emphasis
    """
    # Implementation with custom loss function
    pass
```

### Phase 3: Evaluation and Monitoring Improvements

#### 3.1 Enhanced Evaluation Metrics
```python
def calculate_comprehensive_metrics(y_true, y_pred, timestamps):
    """
    Calculate domain-specific metrics:
    - Seasonal MAE (winter vs summer)
    - Time-of-day accuracy
    - Peak demand prediction accuracy
    - Heat pump cycling correlation
    """
    return metrics_dict
```

#### 3.2 Cross-Validation Strategy
```python
# Current: Standard K-fold
# Better: Time series aware CV
from sklearn.model_selection import TimeSeriesSplit

# Or seasonal split
def seasonal_split(data, test_seasons=['winter_2024']):
    # Split by heating/cooling seasons
    pass
```

### Phase 4: Feature Engineering Enhancements

#### 4.1 Advanced Time Series Features
```python
def add_advanced_time_features(df):
    """
    Add sophisticated time-based features:
    - Fourier transforms for cyclical patterns
    - Wavelet decomposition for multi-scale patterns
    - Autoregressive features
    - Change point detection features
    """
    pass
```

#### 4.2 Weather Integration Improvements
```python
def add_weather_comfort_features(df):
    """
    Add comfort-based weather features:
    - Thermal comfort indices
    - Perceived temperature
    - Weather change indicators
    - Seasonal transition markers
    """
    pass
```

#### 4.3 Heat Pump Specific Features
```python
def add_smart_heat_pump_features(df):
    """
    Add intelligent heat pump features:
    - Defrost cycle detection
    - Efficiency curves
    - Load factor calculations
    - Operating mode classification
    """
    pass
```

## Implementation Priority

### High Priority (Immediate - Week 1)
1. ✅ Fix length mismatch and data leakage issues
2. ✅ Improve feature importance extraction
3. 🔄 Implement alternative target variable formulations to reduce MAPE
4. 🔄 Add feature selection to reduce overfitting

### Medium Priority (Week 2-3)
1. Implement ensemble methods
2. Enhanced cross-validation strategy
3. Advanced time series features
4. Comprehensive evaluation metrics

### Low Priority (Week 4+)
1. Neural network architectures
2. Real-time model updating
3. Advanced visualization dashboards
4. Production deployment optimization

## Model Files Structure

```
models/
├── villamichelin_demand_model.pkl           # Main trained model
├── villamichelin_demand_model_splits_info.pkl # Training data splits
├── villamichelin_demand_model_feature_columns.pkl # Feature names
└── feature_importance_plot.png              # Feature importance visualization
```

## Usage Examples

### Basic Evaluation
```bash
python src/predictions/demand/evaluate.py
```

### Plot Specific Month
```bash
python src/predictions/demand/evaluate.py --plot 2025-01 --dashboard
```

### Save Plots
```bash
python src/predictions/demand/evaluate.py --save
```

### Custom Evaluation Period
```bash
python src/predictions/demand/evaluate.py --eval-ratio 0.15
```

## Key Takeaways from Current Analysis

1. **Model Structure Works**: XGBoost outperforms persistence baseline
2. **MAPE Issue**: Very low baseline consumption makes MAPE misleading
3. **Feature Richness**: 220+ features may be causing overfitting
4. **Data Quality**: Excessive preprocessing suggests data quality issues
5. **Evaluation**: Need more sophisticated metrics for low-consumption scenarios

## Next Steps

1. **Immediate**: Implement target variable transformation to address MAPE
2. **Short-term**: Feature selection and ensemble methods
3. **Medium-term**: Advanced time series modeling
4. **Long-term**: Real-time adaptation and production deployment

---

*Last updated: 2025-05-24*
*Model version: v2.1*
*Performance: MAE=0.97, RMSE=1.39, outperforms persistence by 35%*