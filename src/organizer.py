# prefect version 3.4.1
"""
Home Energy AI - Prefect Orchestration System

This module organizes and schedules all data fetching, training, and prediction tasks
for the home energy AI system. It includes:

1. Data Fetching Tasks (by frequency):
   - Exogenic Data (hourly): prices, CO2/gas/coal, weather
   - Home Data (15 minutes): consumption, Thermia heat pump
   - Solar Data (daily): actual production and predictions

2. Model Training Tasks (scheduled):
   - Price Models (Sunday 02:00): trend, peak, valley
   - Demand Model (Monday 02:00): consumption prediction
   - RL Agent (Tuesday 02:00): battery control optimization

3. Production Flows:
   - Daily energy pipeline with forecasts
   - Weekly model retraining
   - Continuous data updates

Features:
- Proper error handling and logging
- Task dependencies and flow orchestration
- Configurable parameters and scheduling
- Production and development modes
- Comprehensive monitoring and alerts

examples:
organizer.py --test-flow daily-pipeline
organizer.py --test-flow hourly-exogenic
organizer.py --test-flow weekly-training

# Serve flows for development and testing
organizer.py --serve

# Deploy to production prefect server
prefect deploy --all

"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
import pandas as pd

from prefect import flow, task, serve
from prefect.artifacts import create_markdown_artifact

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# File paths for all scripts
SCRIPT_PATHS = {
    # Data fetching scripts
    'solar_actual': 'src/predictions/solar/actual_data/FetchSolarProductionData.py',
    'solar_predictions': 'src/predictions/solar/makeSolarPrediction.py',
    'prices_data': 'src/predictions/prices/getPriceRelatedData.py',
    'co2_gas_coal': 'data/FetchCO2GasCoal.py',
    'weather_data': 'src/predictions/demand/FetchWeatherData.py',
    'consumption_data': 'src/predictions/demand/FetchEnergyData.py',
    'consumption_actual_load': 'src/predictions/demand/FetchActualLoad.py',
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

# =============================================================================
# UTILITY TASKS
# =============================================================================

@task(retries=1, retry_delay_seconds=60)
def run_python_script(script_path: str, args: List[str] = None, 
                     working_dir: str = None, timeout: int = 3600) -> Dict[str, Any]:
    """
    Run a Python script with proper error handling and logging.
    
    Args:
        script_path: Path to the Python script
        args: Command line arguments for the script
        working_dir: Working directory for script execution
        timeout: Timeout in seconds (default 1 hour)
    
    Returns:
        Dict with execution results and metadata
    """
    if args is None:
        args = []
    
    if working_dir is None:
        working_dir = str(PROJECT_ROOT)
    
    full_script_path = PROJECT_ROOT / script_path
    if not full_script_path.exists():
        raise FileNotFoundError(f"Script not found: {full_script_path}")
    
    cmd = [sys.executable, str(full_script_path)] + args
    cmd_str = ' '.join(cmd)
    
    logger.info(f"Executing: {cmd_str}")
    logger.info(f"Working directory: {working_dir}")
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        execution_result = {
            'script_path': script_path,
            'args': args,
            'duration_seconds': duration,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'success': True,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'return_code': result.returncode
        }
        
        logger.info(f"Script completed successfully in {duration:.2f}s")
        if result.stdout:
            logger.info(f"Output: {result.stdout[:500]}...")
        
        return execution_result
        
    except subprocess.TimeoutExpired as e:
        logger.error(f"Script timed out after {timeout}s: {script_path}")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"Script failed with return code {e.returncode}: {script_path}")
        logger.error(f"Error output: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error running script {script_path}: {str(e)}")
        raise

@task
def check_data_freshness(file_path: str, max_age_hours: int = 24) -> Dict[str, Any]:
    """
    Check if a data file exists and is fresh enough.
    
    Args:
        file_path: Path to the data file
        max_age_hours: Maximum age in hours before data is considered stale
    
    Returns:
        Dict with freshness status and metadata
    """
    full_path = PROJECT_ROOT / file_path
    
    if not full_path.exists():
        return {
            'exists': False,
            'is_fresh': False,
            'file_path': file_path,
            'reason': 'File does not exist'
        }
    
    file_stat = full_path.stat()
    file_mtime = datetime.fromtimestamp(file_stat.st_mtime)
    age_hours = (datetime.now() - file_mtime).total_seconds() / 3600
    
    is_fresh = age_hours <= max_age_hours
    
    return {
        'exists': True,
        'is_fresh': is_fresh,
        'file_path': file_path,
        'last_modified': file_mtime.isoformat(),
        'age_hours': age_hours,
        'max_age_hours': max_age_hours,
        'reason': f"File is {age_hours:.1f} hours old" if not is_fresh else "File is fresh"
    }

@task
def create_execution_report(task_results: List[Dict[str, Any]], 
                          flow_name: str) -> str:
    """
    Create a markdown report of task execution results.
    
    Args:
        task_results: List of task execution results
        flow_name: Name of the flow being reported on
    
    Returns:
        Markdown report string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    successful_tasks = [r for r in task_results if r.get('success', False)]
    failed_tasks = [r for r in task_results if not r.get('success', False)]
    
    report = f"""# {flow_name} Execution Report
    
**Execution Time:** {timestamp}

## Summary
- **Total Tasks:** {len(task_results)}
- **Successful:** {len(successful_tasks)}
- **Failed:** {len(failed_tasks)}

## Task Details

### Successful Tasks ✅
"""
    
    for task in successful_tasks:
        duration = task.get('duration_seconds', 0)
        report += f"- **{task.get('script_path', 'Unknown')}** ({duration:.1f}s)\n"
    
    if failed_tasks:
        report += "\n### Failed Tasks ❌\n"
        for task in failed_tasks:
            report += f"- **{task.get('script_path', 'Unknown')}**: {task.get('error', 'Unknown error')}\n"
    
    # Create artifact for better visibility in Prefect UI
    create_markdown_artifact(
        key=f"{flow_name.lower().replace(' ', '-')}-report",
        markdown=report,
        description=f"Execution report for {flow_name}"
    )
    
    return report 

# =============================================================================
# DATA FETCHING TASKS
# =============================================================================

@task(retries=1, retry_delay_seconds=120)
def fetch_exogenic_data() -> Dict[str, Any]:
    """Fetch external/exogenic data: prices, CO2/gas/coal, weather."""
    logger.info("Starting exogenic data fetching")
    
    results = []
    
    # Fetch price-related data
    try:
        result = run_python_script(SCRIPT_PATHS['prices_data'])
        results.append(result)
        logger.info("✅ Price data fetched successfully")
    except Exception as e:
        logger.error(f"❌ Failed to fetch price data: {e}")
        results.append({'script_path': SCRIPT_PATHS['prices_data'], 'success': False, 'error': str(e)})
    
    # Fetch CO2, gas, coal data
    try:
        result = run_python_script(SCRIPT_PATHS['co2_gas_coal'])
        results.append(result)
        logger.info("✅ CO2/Gas/Coal data fetched successfully")
    except Exception as e:
        logger.error(f"❌ Failed to fetch CO2/Gas/Coal data: {e}")
        results.append({'script_path': SCRIPT_PATHS['co2_gas_coal'], 'success': False, 'error': str(e)})
    
    # Fetch weather data
    try:
        result = run_python_script(SCRIPT_PATHS['weather_data'])
        results.append(result)
        logger.info("✅ Weather data fetched successfully")
    except Exception as e:
        logger.error(f"❌ Failed to fetch weather data: {e}")
        results.append({'script_path': SCRIPT_PATHS['weather_data'], 'success': False, 'error': str(e)})
    
    successful = sum(1 for r in results if r.get('success', False))
    logger.info(f"Exogenic data fetching completed: {successful}/{len(results)} successful")
    
    return {
        'task_name': 'fetch_exogenic_data',
        'results': results,
        'successful_count': successful,
        'total_count': len(results)
    }

@flow(name="15-Minute Home Data Update")
def home_data_flow() -> str:
    """15-minute flow to update home data (consumption, Thermia)."""
    logger.info("Starting 15-minute home data update flow")
    
    # Fetch home data
    home_result = fetch_home_data()
    
    # Create execution report
    all_results = home_result['results']
    report = create_execution_report(all_results, "15-Minute Home Data Update")
    
    # Log summary
    successful = home_result['successful_count']
    total = home_result['total_count']
    logger.info(f"Home data flow completed: {successful}/{total} tasks successful")
    
    return report

@task(retries=1, retry_delay_seconds=120)
def fetch_home_data() -> Dict[str, Any]:
    """Fetch home-related data: consumption and Thermia heat pump."""
    logger.info("Starting home data fetching")
    
    results = []
    
    # Fetch consumption data
    try:
        result = run_python_script(SCRIPT_PATHS['consumption_data'])
        results.append(result)
        logger.info("✅ Consumption data fetched successfully")
    except Exception as e:
        logger.error(f"❌ Failed to fetch consumption data: {e}")
        results.append({'script_path': SCRIPT_PATHS['consumption_data'], 'success': False, 'error': str(e)})
    
    # Fetch consumption data
    try:
        result = run_python_script(SCRIPT_PATHS['consumption_actual_load'])
        results.append(result)
        logger.info("✅ Consumption actual load data fetched successfully")
    except Exception as e:
        logger.error(f"❌ Failed to fetch consumption actual load data: {e}")
        results.append({'script_path': SCRIPT_PATHS['consumption_actual_load'], 'success': False, 'error': str(e)})
    
    # Fetch Thermia heat pump data
    try:
        result = run_python_script(SCRIPT_PATHS['thermia_data'])
        results.append(result)
        logger.info("✅ Thermia data fetched successfully")
    except Exception as e:
        logger.error(f"❌ Failed to fetch Thermia data: {e}")
        results.append({'script_path': SCRIPT_PATHS['thermia_data'], 'success': False, 'error': str(e)})
    
    successful = sum(1 for r in results if r.get('success', False))
    logger.info(f"Home data fetching completed: {successful}/{len(results)} successful")
    
    return {
        'task_name': 'fetch_home_data',
        'results': results,
        'successful_count': successful,
        'total_count': len(results)
    }

@task(retries=1, retry_delay_seconds=120)
def fetch_solar_data() -> Dict[str, Any]:
    """Fetch solar production data and generate predictions."""
    logger.info("Starting solar data fetching and predictions")
    
    results = []
    
    # Fetch actual solar production data
    try:
        result = run_python_script(SCRIPT_PATHS['solar_actual'])
        results.append(result)
        logger.info("✅ Solar actual data fetched successfully")
    except Exception as e:
        logger.error(f"❌ Failed to fetch solar actual data: {e}")
        results.append({'script_path': SCRIPT_PATHS['solar_actual'], 'success': False, 'error': str(e)})
    
    # Generate solar predictions
    try:
        result = run_python_script(SCRIPT_PATHS['solar_predictions'])
        results.append(result)
        logger.info("✅ Solar predictions generated successfully")
    except Exception as e:
        logger.error(f"❌ Failed to generate solar predictions: {e}")
        results.append({'script_path': SCRIPT_PATHS['solar_predictions'], 'success': False, 'error': str(e)})
    
    successful = sum(1 for r in results if r.get('success', False))
    logger.info(f"Solar data tasks completed: {successful}/{len(results)} successful")
    
    return {
        'task_name': 'fetch_solar_data',
        'results': results,
        'successful_count': successful,
        'total_count': len(results)
    } 

# =============================================================================
# MODEL TRAINING TASKS
# =============================================================================

@task(retries=1, retry_delay_seconds=300, timeout_seconds=7200)  # 2 hour timeout
def train_price_models(production_mode: bool = True) -> Dict[str, Any]:
    """Train all price prediction models: trend, peak, valley."""
    logger.info(f"Starting price model training (production={production_mode})")
    
    models = ['trend', 'peak', 'valley']
    results = []
    
    for model_type in models:
        try:
            args = ['--model', model_type]
            if production_mode:
                args.append('--production')
            
            result = run_python_script(
                SCRIPT_PATHS['prices_train'], 
                args=args,
                timeout=7200  # 2 hours per model
            )
            results.append(result)
            logger.info(f"✅ {model_type} price model trained successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to train {model_type} price model: {e}")
            results.append({
                'script_path': SCRIPT_PATHS['prices_train'],
                'success': False,
                'error': str(e),
                'model_type': model_type
            })
    
    successful = sum(1 for r in results if r.get('success', False))
    logger.info(f"Price model training completed: {successful}/{len(results)} successful")
    
    return {
        'task_name': 'train_price_models',
        'results': results,
        'successful_count': successful,
        'total_count': len(results),
        'production_mode': production_mode
    }

@task(retries=1, retry_delay_seconds=300, timeout_seconds=10800)  # 3 hour timeout
def train_demand_model(production_mode: bool = True, trials: int = 30) -> Dict[str, Any]:
    """Train demand prediction model."""
    logger.info(f"Starting demand model training (production={production_mode}, trials={trials})")
    
    try:
        args = ['--trials', str(trials)]
        if production_mode:
            # Add any production-specific flags if needed
            pass
        
        result = run_python_script(
            SCRIPT_PATHS['demand_train'],
            args=args,
            timeout=10800  # 3 hours
        )
        
        logger.info("✅ Demand model trained successfully")
        return {
            'task_name': 'train_demand_model',
            'results': [result],
            'successful_count': 1,
            'total_count': 1,
            'production_mode': production_mode,
            'trials': trials
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to train demand model: {e}")
        return {
            'task_name': 'train_demand_model',
            'results': [{
                'script_path': SCRIPT_PATHS['demand_train'],
                'success': False,
                'error': str(e)
            }],
            'successful_count': 0,
            'total_count': 1,
            'production_mode': production_mode,
            'trials': trials
        }

@task(retries=1, retry_delay_seconds=300, timeout_seconds=21600)  # 6 hour timeout
def train_rl_agent(sanity_check_steps: int = 10, start_date: str = None, 
                  end_date: str = None, total_timesteps: int = None) -> Dict[str, Any]:
    """Train RL battery control agent."""
    if start_date is None:
        start_date = "2027-01-01"  # Default as in original TODO
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Starting RL agent training (sanity_check_steps={sanity_check_steps}, "
                f"date_range={start_date} to {end_date})")
    
    try:
        args = [
            '--sanity-check-steps', str(sanity_check_steps),
            '--start-date', start_date,
            '--end-date', end_date
        ]
        
        if total_timesteps:
            args.extend(['--total-timesteps', str(total_timesteps)])
        
        result = run_python_script(
            SCRIPT_PATHS['rl_train'],
            args=args,
            timeout=21600  # 6 hours
        )
        
        logger.info("✅ RL agent trained successfully")
        return {
            'task_name': 'train_rl_agent',
            'results': [result],
            'successful_count': 1,
            'total_count': 1,
            'sanity_check_steps': sanity_check_steps,
            'date_range': f"{start_date} to {end_date}"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to train RL agent: {e}")
        return {
            'task_name': 'train_rl_agent',
            'results': [{
                'script_path': SCRIPT_PATHS['rl_train'],
                'success': False,
                'error': str(e)
            }],
            'successful_count': 0,
            'total_count': 1,
            'sanity_check_steps': sanity_check_steps,
            'date_range': f"{start_date} to {end_date}"
        } 

# =============================================================================
# PREDICTION/INFERENCE TASKS
# =============================================================================

@task(retries=1, retry_delay_seconds=60)
def run_price_predictions(model: str = "merged", start_date: str = None, 
                         horizon_days: float = 1.0) -> Dict[str, Any]:
    """Run price predictions using trained models."""
    if start_date is None:
        start_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Running price predictions (model={model}, start={start_date}, horizon={horizon_days})")
    
    try:
        args = [
            '--model', model,
            '--start', start_date,
            '--horizon', str(horizon_days),
            '--valley-threshold', '0.5'
        ]
        
        result = run_python_script(SCRIPT_PATHS['price_predictions'], args=args)
        logger.info("✅ Price predictions completed successfully")
        
        return {
            'task_name': 'run_price_predictions',
            'results': [result],
            'successful_count': 1,
            'total_count': 1,
            'model': model,
            'start_date': start_date,
            'horizon_days': horizon_days
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to run price predictions: {e}")
        return {
            'task_name': 'run_price_predictions',
            'results': [{
                'script_path': SCRIPT_PATHS['price_predictions'],
                'success': False,
                'error': str(e)
            }],
            'successful_count': 0,
            'total_count': 1,
            'model': model,
            'start_date': start_date,
            'horizon_days': horizon_days
        }

# =============================================================================
# MAIN FLOWS
# =============================================================================

@flow(name="Hourly Exogenic Data Update")
def hourly_exogenic_flow() -> str:
    """Hourly flow to update exogenic data (prices, CO2/gas/coal, weather)."""
    logger.info("Starting hourly exogenic data update flow")
    
    # Fetch exogenic data
    exogenic_result = fetch_exogenic_data()
    
    # Create execution report
    all_results = exogenic_result['results']
    report = create_execution_report(all_results, "Hourly Exogenic Data Update")
    
    # Log summary
    successful = exogenic_result['successful_count']
    total = exogenic_result['total_count']
    logger.info(f"Hourly exogenic flow completed: {successful}/{total} tasks successful")
    
    return report

@flow(name="15-Minute Home Data Update")
def home_data_flow() -> str:
    """15-minute flow to update home data (consumption, Thermia)."""
    logger.info("Starting 15-minute home data update flow")
    
    # Fetch home data
    home_result = fetch_home_data()
    
    # Create execution report
    all_results = home_result['results']
    report = create_execution_report(all_results, "15-Minute Home Data Update")
    
    # Log summary
    successful = home_result['successful_count']
    total = home_result['total_count']
    logger.info(f"Home data flow completed: {successful}/{total} tasks successful")
    
    return report

@flow(name="Daily Energy Pipeline")
def daily_energy_pipeline(
    update_data: bool = True,
    run_price_predictions_flag: bool = True,
    run_solar_predictions: bool = True,
    price_model: str = "merged",
    start_date: Optional[str] = None,
    horizon_days: float = 1.0,
) -> str:
    """
    Daily comprehensive energy pipeline with data updates and predictions.
    
    Args:
        update_data: Whether to update all data sources
        run_price_predictions_flag: Whether to run price predictions
        run_solar_predictions: Whether to run solar predictions
        price_model: Price model to use for predictions
        start_date: Start date for predictions (default: today)
        horizon_days: Prediction horizon in days
    """
    if start_date is None:
        start_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Starting daily energy pipeline (start_date={start_date}, horizon={horizon_days})")
    
    all_results = []
    
    # Step 1: Update all data sources if requested
    if update_data:
        logger.info("📥 Updating all data sources...")
        
        # Fetch exogenic data
        exogenic_result = fetch_exogenic_data()
        all_results.extend(exogenic_result['results'])
        
        # Fetch home data
        home_result = fetch_home_data()
        all_results.extend(home_result['results'])
        
        # Fetch solar data
        solar_result = fetch_solar_data()
        all_results.extend(solar_result['results'])
    
    # Step 2: Run price predictions if requested
    if run_price_predictions_flag:
        logger.info("💰 Running price predictions...")
        price_pred_result = run_price_predictions(
            model=price_model,
            start_date=start_date,
            horizon_days=horizon_days
        )
        all_results.extend(price_pred_result['results'])
    
    # Step 3: Solar predictions are handled in fetch_solar_data step above
    if run_solar_predictions and not update_data:
        # Only run solar predictions separately if we didn't already update solar data
        logger.info("☀️ Running solar predictions...")
        solar_result = fetch_solar_data()
        all_results.extend(solar_result['results'])
    
    # Create comprehensive execution report
    report = create_execution_report(all_results, "Daily Energy Pipeline")
    
    # Log summary
    successful = sum(1 for r in all_results if r.get('success', False))
    total = len(all_results)
    logger.info(f"Daily energy pipeline completed: {successful}/{total} tasks successful")
    
    return report

@flow(name="Weekly Model Training")
def weekly_model_training(
    train_price_models_flag: bool = True,
    train_demand_model_flag: bool = True,
    price_production_mode: bool = True,
    demand_production_mode: bool = True,
    demand_trials: int = 50
) -> str:
    """
    Weekly comprehensive model training flow.
    
    Args:
        train_price_models_flag: Whether to train price models
        train_demand_model_flag: Whether to train demand model
        price_production_mode: Use production mode for price models
        demand_production_mode: Use production mode for demand model
        demand_trials: Number of trials for demand model optimization
    """
    logger.info("Starting weekly model training flow")
    
    all_results = []
    
    # Ensure we have fresh data first
    logger.info("📥 Updating all data sources before training...")
    exogenic_result = fetch_exogenic_data()
    home_result = fetch_home_data()
    solar_result = fetch_solar_data()
    
    all_results.extend(exogenic_result['results'])
    all_results.extend(home_result['results'])
    all_results.extend(solar_result['results'])
    
    # Train price models if requested
    if train_price_models_flag:
        logger.info("💰 Training price models...")
        price_training_result = train_price_models(production_mode=price_production_mode)
        all_results.extend(price_training_result['results'])
    
    # Train demand model if requested
    if train_demand_model_flag:
        logger.info("🏠 Training demand model...")
        demand_training_result = train_demand_model(
            production_mode=demand_production_mode,
            trials=demand_trials
        )
        all_results.extend(demand_training_result['results'])
    
    # Create comprehensive execution report
    report = create_execution_report(all_results, "Weekly Model Training")
    
    # Log summary
    successful = sum(1 for r in all_results if r.get('success', False))
    total = len(all_results)
    logger.info(f"Weekly model training completed: {successful}/{total} tasks successful")
    
    return report

@flow(name="RL Agent Training")
def rl_training_flow(
    sanity_check_steps: int = 10,
    start_date: str = None,
    end_date: str = None,
    total_timesteps: int = None,
    update_data_first: bool = True
) -> str:
    """
    RL agent training flow with optional data updates.
    
    Args:
        sanity_check_steps: Number of sanity check steps
        start_date: Training start date (default: "2027-01-01")
        end_date: Training end date (default: today)
        total_timesteps: Total training timesteps (optional override)
        update_data_first: Whether to update data before training
    """
    logger.info("Starting RL agent training flow")
    
    all_results = []
    
    # Update data if requested
    if update_data_first:
        logger.info("📥 Updating all data sources before RL training...")
        exogenic_result = fetch_exogenic_data()
        home_result = fetch_home_data()
        solar_result = fetch_solar_data()
        
        all_results.extend(exogenic_result['results'])
        all_results.extend(home_result['results'])
        all_results.extend(solar_result['results'])
    
    # Train RL agent
    logger.info("🤖 Training RL agent...")
    rl_result = train_rl_agent(
        sanity_check_steps=sanity_check_steps,
        start_date=start_date,
        end_date=end_date,
        total_timesteps=total_timesteps
    )
    all_results.extend(rl_result['results'])
    
    # Create execution report
    report = create_execution_report(all_results, "RL Agent Training")
    
    # Log summary
    successful = sum(1 for r in all_results if r.get('success', False))
    total = len(all_results)
    logger.info(f"RL agent training flow completed: {successful}/{total} tasks successful")
    
    return report

# =============================================================================
# DEPLOYMENT CONFIGURATION
# =============================================================================

def serve_flows():
    """Serve flows for development and testing with schedules."""
    logger.info("Setting up Prefect flow serving with schedules...")
    
    # Serve flows with their schedules using the new Prefect API
    serve(
        # Hourly exogenic data update
        hourly_exogenic_flow.to_deployment(
            name="hourly-exogenic-data",
            cron="0 * * * *",  # Every hour at minute 0
            description="Hourly update of exogenic data (prices, CO2/gas/coal, weather)",
            tags=["data-fetching", "hourly", "exogenic"]
        ),
        
        # 15-minute home data update  
        home_data_flow.to_deployment(
            name="15min-home-data",
            interval=900,  # 15 minutes in seconds
            description="15-minute update of home data (consumption, Thermia)",
            tags=["data-fetching", "15min", "home"]
        ),
        
        # Daily energy pipeline
        daily_energy_pipeline.to_deployment(
            name="daily-energy-pipeline", 
            cron="0 6 * * *",  # Daily at 6 AM
            description="Daily comprehensive energy pipeline with data updates and predictions",
            tags=["daily", "pipeline", "predictions"]
        ),
        
        # Weekly price model training (Sunday 02:00)
        weekly_model_training.to_deployment(
            name="weekly-price-training",
            cron="0 2 * * 0",  # Sunday at 2 AM
            description="Weekly training of price prediction models (trend, peak, valley)",
            tags=["training", "weekly", "prices"],
            parameters={
                "train_price_models_flag": True,
                "train_demand_model_flag": False,
                "price_production_mode": True
            }
        ),
        
        # Weekly demand model training (Monday 02:00) 
        weekly_model_training.to_deployment(
            name="weekly-demand-training",
            cron="0 2 * * 1",  # Monday at 2 AM
            description="Weekly training of demand prediction model",
            tags=["training", "weekly", "demand"],
            parameters={
                "train_price_models_flag": False,
                "train_demand_model_flag": True,
                "demand_production_mode": True,
                "demand_trials": 50
            }
        ),
        
        # Weekly RL agent training (Tuesday 02:00)
        rl_training_flow.to_deployment(
            name="weekly-rl-training", 
            cron="0 2 * * 2",  # Tuesday at 2 AM
            description="Weekly training of RL battery control agent",
            tags=["training", "weekly", "rl"],
            parameters={
                "sanity_check_steps": 10,
                "start_date": "2027-01-01",
                "update_data_first": True
            }
        )
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Home Energy AI Prefect Orchestration")
    parser.add_argument("--serve", action="store_true", 
                       help="Serve flows for development/testing")
    parser.add_argument("--test-flow", choices=[
        "hourly-exogenic", "home-data", "daily-pipeline", 
        "weekly-training", "rl-training"
    ], help="Test a specific flow")
    
    args = parser.parse_args()
    
    if args.serve:
        serve_flows()
    elif args.test_flow:
        # Test individual flows
        if args.test_flow == "hourly-exogenic":
            result = hourly_exogenic_flow()
            print("Hourly Exogenic Flow Result:", result)
        elif args.test_flow == "home-data":
            result = home_data_flow()
            print("Home Data Flow Result:", result)
        elif args.test_flow == "daily-pipeline":
            result = daily_energy_pipeline()
            print("Daily Pipeline Result:", result)
        elif args.test_flow == "weekly-training":
            result = weekly_model_training()
            print("Weekly Training Result:", result)
        elif args.test_flow == "rl-training":
            result = rl_training_flow()
            print("RL Training Result:", result)
    else:
        print("Use --serve to start serving flows or --test-flow to test a specific flow")
        print("Available test flows: hourly-exogenic, home-data, daily-pipeline, weekly-training, rl-training") 