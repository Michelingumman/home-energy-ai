# Home Energy AI - Prefect Orchestration System

## Overview

The `src/organizer.py` script is a comprehensive Prefect-based orchestration system that manages all data fetching, model training, and prediction tasks for the home energy AI system. It replaces the original TODO-based approach with a production-ready, scheduled workflow management solution.

## Features

### 🔄 **Automated Data Fetching**
- **Exogenic Data** (hourly): Price data, CO2/gas/coal commodities, weather data
- **Home Data** (15 minutes): Energy consumption, Thermia heat pump data
- **Solar Data** (daily): Actual production data and predictions

### 🤖 **Model Training Workflows**
- **Price Models** (Sunday 02:00): trend, peak, valley prediction models
- **Demand Model** (Monday 02:00): Consumption prediction with XGBoost + HMM
- **RL Agent** (Tuesday 02:00): Battery control optimization with RecurrentPPO

### 📊 **Production Flows**
- **Daily Energy Pipeline**: Comprehensive data updates and forecasting
- **Weekly Model Retraining**: Automated model updates with fresh data
- **Continuous Monitoring**: Real-time data freshness checks and alerts

### 🛡️ **Robust Error Handling**
- Task retries with exponential backoff
- Comprehensive logging and monitoring
- Execution reports with success/failure tracking
- Timeout protection for long-running tasks

## Quick Start

### 1. Testing Individual Flows

```bash
# Test the daily energy pipeline
python src/organizer.py --test-flow daily-pipeline

# Test hourly exogenic data fetching
python src/organizer.py --test-flow hourly-exogenic

# Test home data fetching (15-minute cycle)
python src/organizer.py --test-flow home-data

# Test weekly model training
python src/organizer.py --test-flow weekly-training

# Test RL agent training
python src/organizer.py --test-flow rl-training
```

### 2. Development Mode (Serve Flows Locally)

```bash
# Start Prefect server locally and serve all flows
python src/organizer.py --serve
```

This will:
- Create all deployment schedules
- Start serving flows for testing
- Print the Prefect UI URL for monitoring

### 3. Production Deployment

```bash
# Deploy to Prefect Cloud or self-hosted Prefect server
prefect deploy --all
```

## Flow Architecture

### Data Fetching Flows

#### Hourly Exogenic Data (`0 * * * *`)
- **Price Data**: `src/predictions/prices/getPriceRelatedData.py`
- **CO2/Gas/Coal**: `data/FetchCO2GasCoal.py`
- **Weather**: `src/predictions/demand/FetchWeatherData.py`

#### 15-Minute Home Data (every 15 minutes)
- **Consumption**: `src/predictions/demand/FetchEnergyData.py`
- **Thermia Heat Pump**: `src/predictions/demand/Thermia/UpdateHeatPumpData.py`

#### Daily Solar Data (`0 6 * * *`)
- **Actual Production**: `src/predictions/solar/actual_data/FetchSolarProductionData.py`
- **Predictions**: `src/predictions/solar/makeSolarPrediction.py`

### Training Flows

#### Weekly Price Model Training (`0 2 * * 0` - Sunday 2 AM)
```python
# Trains all three price models with production settings
train_price_models(production_mode=True)
# Models: trend (XGBoost), peak (TCN), valley (TCN)
```

#### Weekly Demand Model Training (`0 2 * * 1` - Monday 2 AM)
```python
# Trains demand prediction model with enhanced features
train_demand_model(production_mode=True, trials=50)
# Features: HMM occupancy, weather, calendar, lagged variables
```

#### Weekly RL Agent Training (`0 2 * * 2` - Tuesday 2 AM)
```python
# Trains battery control agent with RecurrentPPO
train_rl_agent(
    sanity_check_steps=10,
    start_date="2027-01-01",
    end_date=today
)
```

## Configuration

### Script Paths
All script paths are centrally managed in `SCRIPT_PATHS` dictionary:

```python
SCRIPT_PATHS = {
    # Data fetching scripts
    'solar_actual': 'src/predictions/solar/actual_data/FetchSolarProductionData.py',
    'solar_predictions': 'src/predictions/solar/makeSolarPrediction.py',
    'prices_data': 'src/predictions/prices/getPriceRelatedData.py',
    'co2_gas_coal': 'data/FetchCO2GasCoal.py',
    'weather_data': 'src/predictions/demand/FetchWeatherData.py',
    'consumption_data': 'src/predictions/demand/FetchEnergyData.py',
    'thermia_data': 'src/predictions/demand/Thermia/UpdateHeatPumpData.py',
    
    # Training scripts
    'prices_train': 'src/predictions/prices/train.py',
    'demand_train': 'src/predictions/demand/train.py',
    'rl_train': 'src/rl/train.py',
    
    # Prediction/inference scripts
    'price_predictions': 'src/predictions/prices/run_model.py',
    'demand_predictions': 'src/predictions/demand/predict.py',
    'rl_control': 'src/rl/run_production_agent.py'
}
```

### Task Configuration
- **Retries**: 1-3 attempts with exponential backoff
- **Timeouts**: 1-6 hours depending on task complexity
- **Retry Delays**: 60-300 seconds between attempts

## Monitoring and Alerts

### Execution Reports
Each flow generates detailed markdown reports including:
- Task execution times
- Success/failure rates
- Error messages and debugging info
- Performance metrics

### Prefect UI Integration
- Real-time flow run monitoring
- Historical execution data
- Log aggregation and searching
- Alert configuration for failures

### Artifacts
- Execution reports stored as Prefect artifacts
- Model training metrics and plots
- Data quality assessments
- Performance dashboards

## Error Handling

### Graceful Failures
- Individual task failures don't stop entire flows
- Comprehensive error logging with context
- Automatic retry logic with backoff
- Fallback strategies for critical dependencies

### Timeout Protection
- Script-level timeouts prevent hanging processes
- Long-running training tasks get extended timeouts
- Resource cleanup on timeout/failure

### Monitoring Integration
- Failed task notifications
- Performance degradation alerts
- Data freshness monitoring
- Model accuracy tracking

## Scheduling Details

| Flow | Schedule | Description |
|------|----------|-------------|
| Hourly Exogenic Data | `0 * * * *` | Every hour at minute 0 |
| 15-Minute Home Data | Every 15 minutes | Continuous home data updates |
| Daily Energy Pipeline | `0 6 * * *` | Daily at 6 AM |
| Price Model Training | `0 2 * * 0` | Sunday at 2 AM |
| Demand Model Training | `0 2 * * 1` | Monday at 2 AM |
| RL Agent Training | `0 2 * * 2` | Tuesday at 2 AM |

## Original TODO Implementation

The new organizer successfully implements all items from the original TODO:

### ✅ **Exogenic Data fetching (every hour)**
- ✅ `price_updateData_script` → `fetch_exogenic_data()`
- ✅ `Co2GasCoal_update_script` → `fetch_exogenic_data()`
- ✅ `weather_updateData_script` → `fetch_exogenic_data()`

### ✅ **Home related Data (every 15 minutes)**
- ✅ `HomeConsumption_updateData_script` → `fetch_home_data()`
- ✅ `Thermia_updateData_script` → `fetch_home_data()`

### ✅ **Price Model Training (02:00 AM every Sunday)**
- ✅ `train.py --model trend --production` → `train_price_models()`
- ✅ `train.py --model peak --production` → `train_price_models()`
- ✅ `train.py --model valley --production` → `train_price_models()`

### ✅ **Demand Model Training (02:00 AM every Monday)**
- ✅ `demand/train.py --production` → `train_demand_model()`

### ✅ **Battery Agent Training (every Tuesday at 02:00 AM)**
- ✅ `rl/train.py --sanity-check-steps 10 --start-date 2027-01-01 --end-date {current date}` → `train_rl_agent()`

## Advanced Usage

### Custom Deployment Parameters

```python
# Deploy with custom parameters
from src.organizer import weekly_model_training

# Deploy demand training with more trials
deployment = Deployment.build_from_flow(
    flow=weekly_model_training,
    name="custom-demand-training",
    parameters={
        "train_price_models_flag": False,
        "train_demand_model_flag": True,
        "demand_trials": 100  # More thorough optimization
    }
)
```

### Manual Flow Execution

```python
# Run flows programmatically
from src.organizer import daily_energy_pipeline, train_rl_agent

# Run daily pipeline with custom settings
result = daily_energy_pipeline(
    update_data=True,
    run_price_predictions=True,
    price_model="trend",
    horizon_days=2.0
)

# Train RL agent with specific date range
result = train_rl_agent(
    sanity_check_steps=50,
    start_date="2024-01-01",
    end_date="2024-12-31",
    total_timesteps=1000000
)
```

### Data Freshness Monitoring

```python
from src.organizer import check_data_freshness

# Check if solar data is fresh (within 24 hours)
freshness = check_data_freshness(
    "src/predictions/solar/actual_data/ActualSolarProductionData.csv",
    max_age_hours=24
)

if not freshness['is_fresh']:
    print(f"Solar data is stale: {freshness['reason']}")
```

## Dependencies

### Required Packages
- `prefect >= 3.4.1`
- `prefect-shell`
- `pandas`
- `python-dateutil`

### Environment Setup
Ensure all training scripts and data fetching scripts are properly configured with:
- API keys in `api.env`
- Python environment with all dependencies
- Proper file permissions for data directories

## Troubleshooting

### Common Issues

1. **Script Not Found Errors**
   - Verify all paths in `SCRIPT_PATHS` are correct
   - Ensure scripts exist and are executable

2. **Timeout Errors**
   - Increase timeout values for long-running tasks
   - Check system resources during training

3. **Data Freshness Warnings**
   - Verify data source APIs are accessible
   - Check network connectivity and API limits

4. **Model Training Failures**
   - Ensure sufficient disk space for models
   - Verify training data quality and completeness

### Debugging

```bash
# Run with detailed logging
PREFECT_LOGGING_LEVEL=DEBUG python src/organizer.py --test-flow daily-pipeline

# Check Prefect logs
prefect server logs

# View deployment status
prefect deployment ls
```

## Contributing

When adding new tasks or modifying existing ones:

1. Update `SCRIPT_PATHS` with new script locations
2. Add appropriate retry logic and timeouts
3. Include comprehensive error handling
4. Update documentation and tests
5. Test flows individually before deployment

The organizer system is designed to be easily extensible while maintaining robust error handling and monitoring capabilities. 