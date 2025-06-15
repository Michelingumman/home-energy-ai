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
from prefect.schedules import Cron

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
    'rl_agent': 'src/rl/run_production_agent.py'
}

# =============================================================================
# UTILITY TASKS
# =============================================================================

def get_descriptive_name_from_script_path(script_path: str, args: List[str] = None) -> str:
    """Generate a descriptive task name based on the script path and arguments."""
    if args is None:
        args = []
    
    # Map script paths to descriptive names
    script_name_map = {
        'src/predictions/solar/actual_data/FetchSolarProductionData.py': 'fetch-solar-actual-data',
        'src/predictions/solar/makeSolarPrediction.py': 'fetch-solar-predictions',
        'src/predictions/prices/getPriceRelatedData.py': 'fetch-price-data',
        'data/FetchCO2GasCoal.py': 'fetch-co2-gas-coal-data',
        'src/predictions/demand/FetchWeatherData.py': 'fetch-weather-data',
        'src/predictions/demand/FetchEnergyData.py': 'fetch-energy-consumption',
        'src/predictions/demand/FetchActualLoad.py': 'fetch-actual-load',
        'src/predictions/demand/Thermia/UpdateHeatPumpData.py': 'fetch-thermia-data',
        'src/predictions/prices/train.py': 'train-price-models',
        'src/predictions/demand/train.py': 'train-demand-model',
        'src/rl/train.py': 'train-rl-agent',
        'src/predictions/prices/run_model.py': 'generate-price-predictions',
        'src/predictions/demand/predict.py': 'generate-demand-predictions',
        'src/rl/run_production_agent.py': 'run-rl-control',
    }
    
    base_name = script_name_map.get(script_path)
    if base_name:
        # Add model type for training scripts if specified in args
        if 'train.py' in script_path and args:
            for i, arg in enumerate(args):
                if arg == '--model' and i + 1 < len(args):
                    return f"{base_name}-{args[i+1]}"
                elif arg == '--trials' and i + 1 < len(args):
                    return f"{base_name}-{args[i+1]}-trials"
        return base_name
    
    # Fallback: extract filename
    filename = script_path.split('/')[-1].replace('.py', '').lower().replace('_', '-')
    return f"run-{filename}"

@task(retries=1, retry_delay_seconds=60, 
      task_run_name="{task_name}")
def run_python_script(script_path: str, args: List[str] = None, 
                     working_dir: str = None, timeout: int = 24*60*60,
                     task_name: str = None) -> Dict[str, Any]:
    """
    Run a Python script with proper error handling and logging.
    
    Args:
        script_path: Path to the Python script
        args: Command line arguments for the script
        working_dir: Working directory for script execution
        timeout: Timeout in seconds (default 1 hour)
        task_name: Custom name for the task run (auto-generated if None)
    
    Returns:
        Dict with execution results and metadata
    """
    if args is None:
        args = []
    
    if working_dir is None:
        working_dir = str(PROJECT_ROOT)
    
    # Generate descriptive name if not provided
    if task_name is None:
        task_name = get_descriptive_name_from_script_path(script_path, args)
    
    full_script_path = PROJECT_ROOT / script_path
    if not full_script_path.exists():
        raise FileNotFoundError(f"Script not found: {full_script_path}")
    
    cmd = [sys.executable, str(full_script_path)] + args
    cmd_str = ' '.join(cmd)
    
    logger.info(f"Executing ({task_name}): {cmd_str}")
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
            'return_code': result.returncode,
            'task_name': task_name
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

# Specific named tasks for better UI readability
@task(retries=1, retry_delay_seconds=120, task_run_name="fetch-price-data")
def fetch_price_data() -> Dict[str, Any]:
    """Fetch SE3 electricity price data."""
    return run_python_script(
        SCRIPT_PATHS['prices_data'], 
        task_name="fetch-price-data"
    )

@task(retries=1, retry_delay_seconds=120, task_run_name="fetch-co2-gas-coal")  
def fetch_co2_gas_coal() -> Dict[str, Any]:
    """Fetch CO2, gas, and coal commodity data."""
    return run_python_script(
        SCRIPT_PATHS['co2_gas_coal'],
        task_name="fetch-co2-gas-coal"
    )

@task(retries=1, retry_delay_seconds=120, task_run_name="fetch-weather-data")
def fetch_weather_data() -> Dict[str, Any]:
    """Fetch weather data from Open-Meteo."""
    return run_python_script(
        SCRIPT_PATHS['weather_data'],
        task_name="fetch-weather-data"
    )

@task(retries=1, retry_delay_seconds=120, task_run_name="fetch-energy-consumption")
def fetch_energy_consumption() -> Dict[str, Any]:
    """Fetch home energy consumption data."""
    return run_python_script(
        SCRIPT_PATHS['consumption_data'],
        task_name="fetch-energy-consumption"
    )

@task(retries=1, retry_delay_seconds=120, task_run_name="fetch-actual-load")
def fetch_actual_load() -> Dict[str, Any]:
    """Fetch actual load consumption data."""
    return run_python_script(
        SCRIPT_PATHS['consumption_actual_load'],
        task_name="fetch-actual-load"
    )

@task(retries=1, retry_delay_seconds=120, task_run_name="fetch-thermia-data")
def fetch_thermia_data() -> Dict[str, Any]:
    """Fetch Thermia heat pump data."""
    return run_python_script(
        SCRIPT_PATHS['thermia_data'],
        task_name="fetch-thermia-data"
    )

@task(retries=1, retry_delay_seconds=120, task_run_name="fetch-solar-actual")
def fetch_solar_actual() -> Dict[str, Any]:
    """Fetch actual solar production data."""
    return run_python_script(
        SCRIPT_PATHS['solar_actual'],
        task_name="fetch-solar-actual"
    )

@task(retries=1, retry_delay_seconds=120, task_run_name="fetch-solar-predictions")
def fetch_solar_predictions() -> Dict[str, Any]:
    """Generate solar production predictions.""" 
    return run_python_script(
        SCRIPT_PATHS['solar_predictions'],
        task_name="fetch-solar-predictions"
    )
    
    
@task(retries=1, retry_delay_seconds=120, task_run_name="run-agent")
def run_agent() -> Dict[str, Any]:
    """Execute the RL agent for battery control and return formatted output."""
    logger.info("Running RL agent for battery control")
    
    import subprocess
    import io
    import contextlib
    from datetime import datetime
    
    try:
        # Use the same Python interpreter that's running Prefect
        cmd = [sys.executable, str(SCRIPT_PATHS['rl_agent']), "--dry-run"]
        
        # Set up environment to ensure proper module access
        env = os.environ.copy()
        env['PYTHONPATH'] = str(PROJECT_ROOT)
        
        logger.info(f"Executing RL agent: {' '.join(cmd)}")
        logger.info(f"Working directory: {PROJECT_ROOT}")
        logger.info(f"Python executable: {sys.executable}")
        
        # Run the script and capture both stdout and stderr
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300,  # 5 minute timeout
            cwd=PROJECT_ROOT,
            env=env
        )
        
        if result.returncode == 0:
            # Parse the output to extract key information
            output_lines = result.stdout.split('\n')
            agent_info = _parse_agent_output(output_lines)
            
            logger.info("✅ RL Agent executed successfully")
            logger.info(f"Agent action: {agent_info.get('action_value', 'N/A')}")
            logger.info(f"Battery command: {agent_info.get('battery_command', 'N/A')}")
            
            return {
                'script_path': str(SCRIPT_PATHS['rl_agent']),
                'success': True,
                'agent_info': agent_info,
                'full_output': result.stdout,
                'timestamp': datetime.now().isoformat(),
                'return_code': result.returncode
            }
        else:
            logger.error(f"RL Agent script failed with return code {result.returncode}")
            logger.error(f"Command: {' '.join(cmd)}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return {
                'script_path': str(SCRIPT_PATHS['rl_agent']),
                'success': False,
                'error': f"Return code {result.returncode}: {result.stderr}",
                'stdout': result.stdout,
                'stderr': result.stderr,
                'timestamp': datetime.now().isoformat(),
                'return_code': result.returncode
            }
            
    except subprocess.TimeoutExpired:
        logger.error("RL Agent script timed out after 5 minutes")
        return {
            'script_path': str(SCRIPT_PATHS['rl_agent']),
            'success': False,
            'error': "Script execution timed out",
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error running RL agent: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            'script_path': str(SCRIPT_PATHS['rl_agent']),
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def _parse_agent_output(output_lines: List[str]) -> Dict[str, Any]:
    """Parse the RL agent output to extract key information."""
    info = {
        'soc': None,
        'soc_percentage': None,
        'current_price': None,
        'price_trend': None,
        'action_value': None,
        'battery_command': None,
        'command_type': None,
        'power_kw': None,
        'decision_context': [],
        'forecast_summary': {},
        'raw_sections': {}
    }
    
    try:
        # Find the main report section
        report_start = -1
        for i, line in enumerate(output_lines):
            if "ENERGY MANAGEMENT SYSTEM DECISION" in line:
                report_start = i
                break
        
        if report_start == -1:
            return info
            
        # Extract specific information
        for i in range(report_start, len(output_lines)):
            line = output_lines[i].strip()
            
            if "State of Charge:" in line:
                # Extract SoC percentage and kWh values
                parts = line.split()
                for j, part in enumerate(parts):
                    if part.endswith('%'):
                        info['soc_percentage'] = part.rstrip('%')
                        # Try to extract the numerical SoC (0-1 range)
                        try:
                            info['soc'] = float(info['soc_percentage']) / 100.0
                        except:
                            pass
                        break
                        
            elif "Current Price:" in line:
                # Extract current price and trend
                parts = line.split()
                for j, part in enumerate(parts):
                    try:
                        if 'ore/kWh' in parts[j+1:j+2]:
                            info['current_price'] = float(part)
                            break
                    except (ValueError, IndexError):
                        continue
                
                if "[" in line and "]" in line:
                    start = line.find("[") + 1
                    end = line.find("]")
                    info['price_trend'] = line[start:end]
                    
            elif "Raw Action Value:" in line:
                # Extract raw action value
                parts = line.split()
                for part in parts:
                    try:
                        if part.startswith(('+', '-')) or part.replace('.', '').replace('-', '').isdigit():
                            info['action_value'] = float(part)
                            break
                    except ValueError:
                        continue
                        
            elif "Battery Command:" in line:
                # Extract battery command
                logger.debug(f"Parsing battery command line: '{line}'")
                
                # Check for DISCHARGE first (before CHARGE) to avoid substring matching
                if "DISCHARGE" in line:
                    info['command_type'] = "DISCHARGE" 
                    logger.debug("Detected DISCHARGE command")
                    # Extract power value
                    import re
                    match = re.search(r'DISCHARGE at ([\d.]+) kW', line)
                    if match:
                        info['power_kw'] = float(match.group(1))
                        info['battery_command'] = f"Discharge at {info['power_kw']} kW"
                        logger.debug(f"Extracted discharge power: {info['power_kw']} kW")
                    else:
                        logger.warning(f"Could not extract power from DISCHARGE line: {line}")
                        
                elif "CHARGE" in line:
                    info['command_type'] = "CHARGE"
                    logger.debug("Detected CHARGE command")
                    # Extract power value
                    import re
                    match = re.search(r'CHARGE at ([\d.]+) kW', line)
                    if match:
                        info['power_kw'] = float(match.group(1))
                        info['battery_command'] = f"Charge at {info['power_kw']} kW"
                        logger.debug(f"Extracted charge power: {info['power_kw']} kW")
                    else:
                        logger.warning(f"Could not extract power from CHARGE line: {line}")
                        
                elif "IDLE" in line:
                    info['command_type'] = "IDLE"
                    info['power_kw'] = 0.0
                    info['battery_command'] = "Battery idle / no change"
                    logger.debug("Detected IDLE command")
                else:
                    logger.warning(f"Could not parse battery command from line: {line}")
                    
            elif "Solar (24h):" in line:
                # Extract solar forecast
                parts = line.split()
                for j, part in enumerate(parts):
                    try:
                        if 'kWh' in parts[j+1:j+2]:
                            info['forecast_summary']['solar_24h'] = float(part)
                            break
                    except (ValueError, IndexError):
                        continue
                        
            elif "Load Avg (24h):" in line:
                # Extract load forecast
                parts = line.split()
                for j, part in enumerate(parts):
                    try:
                        if 'kW' in parts[j+1:j+2]:
                            info['forecast_summary']['load_avg_24h'] = float(part)
                            break
                    except (ValueError, IndexError):
                        continue
                        
            elif any(strategy in line for strategy in ["CHARGING STRATEGY:", "DISCHARGING STRATEGY:", "CONSERVATION STRATEGY:"]):
                # Start capturing decision context
                strategy_lines = []
                for k in range(i+1, min(i+6, len(output_lines))):
                    context_line = output_lines[k].strip()
                    if context_line.startswith("•") or context_line.startswith("    •"):
                        strategy_lines.append(context_line.lstrip("• ").strip())
                    elif "NEXT EVALUATION:" in context_line:
                        break
                info['decision_context'] = strategy_lines
                
    except Exception as e:
        logger.warning(f"Error parsing agent output: {e}")
        
    return info

@task(task_run_name="create-agent-report")
def create_agent_report(agent_result: Dict[str, Any]) -> str:
    """Create a beautiful HTML/Markdown report from the RL agent results."""
    
    if not agent_result.get('success', False):
        return f"""
            # ⚠️ RL Agent Execution Failed

            **Timestamp:** {agent_result.get('timestamp', 'Unknown')}

            **Error:** {agent_result.get('error', 'Unknown error')}

            **Script:** `{agent_result.get('script_path', 'Unknown')}`
                """
    
    agent_info = agent_result.get('agent_info', {})
    timestamp = agent_result.get('timestamp', 'Unknown')
    
    # Debug logging
    logger.debug(f"Creating report with agent_info: {agent_info}")
    
    # Create status badges
    command_type = agent_info.get('command_type', 'UNKNOWN')
    logger.debug(f"Command type: {command_type}")
    
    if command_type == 'CHARGE':
        command_badge = "🔋 **CHARGING**"
        command_color = "🟢"
    elif command_type == 'DISCHARGE':
        command_badge = "⚡ **DISCHARGING**" 
        command_color = "🔴"
    else:
        command_badge = "⏸️ **IDLE**"
        command_color = "🟡"
    
    logger.debug(f"Command badge: {command_badge}")
    
    price_trend = agent_info.get('price_trend', 'UNKNOWN')
    price_emoji = "📈" if price_trend == "HIGH" else "📉" if price_trend == "LOW" else "📊"
    
    soc_percentage = agent_info.get('soc_percentage')
    soc_emoji = "🔋" if float(soc_percentage or 0) > 70 else "🔋" if float(soc_percentage or 0) > 30 else "🪫"
    
    report = f"""
# Energy Management System Report

**Report Generated:** {timestamp}

---

## {command_color} Current System Status

| **Metric** | **Value** | **Status** |
|------------|-----------|------------|
| {soc_emoji} **Battery SoC** | {soc_percentage}% | {'✅ Good' if float(soc_percentage or 0) > 20 else '⚠️ Low'} |
| {price_emoji} **Current Price** | {agent_info.get('current_price', 'N/A')} öre/kWh | {price_trend} |
| ⚡ **Command** | {agent_info.get('battery_command', 'Unknown')} | {command_badge} |

---

## AI Agent Decision

**Neural Network Output:** `{agent_info.get('action_value', 'N/A')}`

**Recommended Action:** {command_badge}

"""

    # Add power details if available
    power_kw = agent_info.get('power_kw')
    if power_kw is not None:
        if command_type == 'CHARGE':
            report += f"- 🔌 **Charging at {power_kw} kW**\n"
        elif command_type == 'DISCHARGE':
            report += f"- ⚡ **Discharging at {power_kw} kW**\n"
        else:
            report += f"- ⏸️ **Battery idle (0 kW)**\n"
    
    # Add forecast summary
    forecast = agent_info.get('forecast_summary', {})
    if forecast:
        report += f"""
---

## 24-Hour Forecast Summary

| **Forecast** | **Value** |
|--------------|-----------|
| ☀️ **Solar Production** | {forecast.get('solar_24h', 'N/A')} kWh |
| ⚡ **Average Load** | {forecast.get('load_avg_24h', 'N/A')} kW |

"""

    # Add decision context
    decision_context = agent_info.get('decision_context', [])
    if decision_context:
        report += f"""
---

## 🧠 Decision Context

"""
        for context in decision_context:
            report += f"- {context}\n"
    
    # Add next evaluation
    from datetime import datetime, timedelta
    try:
        current_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        next_eval = current_time + timedelta(minutes=15)
        report += f"""
---

## ⏰ Next Evaluation

**Next Check:** {next_eval.strftime('%H:%M')} (15 minutes)

---

*Report generated by Home Energy AI System*
"""
    except:
        report += """
---

*Report generated by Home Energy AI System*
"""
    
    return report

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

@task(task_run_name="create-execution-report")
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

@task(retries=1, retry_delay_seconds=120, task_run_name="fetch-exogenic-data")
def fetch_exogenic_data() -> Dict[str, Any]:
    """Fetch external/exogenic data: prices, CO2/gas/coal, weather."""
    logger.info("Starting exogenic data fetching")
    
    results = []
    
    # Fetch price-related data
    try:
        result = fetch_price_data()
        results.append(result)
        logger.info("✅ Price data fetched successfully")
    except Exception as e:
        logger.error(f"❌ Failed to fetch price data: {e}")
        results.append({'script_path': SCRIPT_PATHS['prices_data'], 'success': False, 'error': str(e)})
    
    # Fetch CO2, gas, coal data
    try:
        result = fetch_co2_gas_coal()
        results.append(result)
        logger.info("✅ CO2/Gas/Coal data fetched successfully")
    except Exception as e:
        logger.error(f"❌ Failed to fetch CO2/Gas/Coal data: {e}")
        results.append({'script_path': SCRIPT_PATHS['co2_gas_coal'], 'success': False, 'error': str(e)})
    
    # Fetch weather data
    try:
        result = fetch_weather_data()
        results.append(result)
        logger.info("✅ Weather data fetched successfully")
    except Exception as e:
        logger.error(f"❌ Failed to fetch weather data: {e}")
        results.append({'script_path': SCRIPT_PATHS['weather_data'], 'success': False, 'error': str(e)})
    
    # Fetch solar predictions
    try:
        result = fetch_solar_predictions()
        results.append(result)
        logger.info("✅ Solar predictions fetched successfully")
    except Exception as e:
        logger.error(f"❌ Failed to fetch solar predictions: {e}")
        results.append({'script_path': SCRIPT_PATHS['solar_predictions'], 'success': False, 'error': str(e)})
    
    # Fetch solar actual data
    try:
        result = fetch_solar_actual()
        results.append(result)
        logger.info("✅ Solar actual data fetched successfully")
    except Exception as e:
        logger.error(f"❌ Failed to fetch solar actual data: {e}")
        results.append({'script_path': SCRIPT_PATHS['solar_actual'], 'success': False, 'error': str(e)})
    
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
    """15-minute flow to update home data and run RL agent control."""
    logger.info("Starting 15-minute home data update flow with RL agent control")
    
    # Fetch home data and run RL agent
    home_result = fetch_home_data()
    
    # Extract agent report if available
    agent_report = home_result.get('agent_report')
    agent_success = False
    
    # Check if the RL agent specifically succeeded
    for result in home_result.get('results', []):
        if 'rl_agent' in result.get('script_path', '') or 'run_production_agent' in result.get('script_path', ''):
            agent_success = result.get('success', False)
            break
    
    if agent_report and agent_success:
        # Create a Prefect artifact for the agent report for better UI visibility
        create_markdown_artifact(
            markdown=agent_report,
            key="rl-agent-control-report",
            description="✅ RL Agent Control Decision Report - SUCCESS"
        )
        
        # Log the agent report prominently for Prefect UI
        logger.info("🤖 RL AGENT CONTROL REPORT - SUCCESS")
        logger.info("=" * 60)
        
        # Split the markdown report into lines and log each one
        for line in agent_report.split('\n'):
            if line.strip():  # Only log non-empty lines
                logger.info(line)
        
        logger.info("=" * 60)
        logger.info("✅ RL Agent control completed successfully")
    else:
        # Find the specific agent error
        agent_error = "Unknown error"
        for result in home_result.get('results', []):
            if 'rl_agent' in result.get('script_path', '') or 'run_production_agent' in result.get('script_path', ''):
                agent_error = result.get('error', 'Unknown error')
                # Log additional details if available
                if 'stderr' in result:
                    logger.error(f"RL Agent STDERR: {result['stderr']}")
                if 'stdout' in result:
                    logger.info(f"RL Agent STDOUT: {result['stdout']}")
                break
        
        logger.error(f"❌ RL Agent control failed: {agent_error}")
        
        # Create a failure artifact with more details
        failure_report = f"""# ❌ RL Agent Control Failed

**Error:** {agent_error}

**Timestamp:** {datetime.now().isoformat()}

## Troubleshooting

1. **Check Dependencies**: Ensure `sb3-contrib` and other RL dependencies are installed
2. **Verify Model**: Check if the trained model exists at the expected path
3. **Environment**: Ensure the correct Python environment is active
4. **Logs**: Check the detailed logs above for more specific error information

## Next Steps

- Check the agent script can run independently: `python src/rl/run_production_agent.py --dry-run`
- Verify all dependencies are installed in the current environment
- Check if the model file exists and is accessible

"""
        
        create_markdown_artifact(
            markdown=failure_report,
            key="rl-agent-control-report",
            description="❌ RL Agent Control Failure Report"
        )
    
    # Create standard execution report for other tasks
    all_results = home_result['results']
    standard_report = create_execution_report(all_results, "15-Minute Home Data Update")
    
    # Determine overall flow status
    total_tasks = home_result['total_count']
    successful_tasks = home_result['successful_count']
    
    # Consider the flow a warning if RL agent failed but other tasks succeeded
    if agent_success:
        flow_status = "✅ SUCCESS"
        status_emoji = "🟢"
    elif successful_tasks > 0:
        flow_status = "⚠️ PARTIAL SUCCESS - RL AGENT FAILED"
        status_emoji = "🟡"
        logger.warning("Flow completed with warnings: RL Agent failed but other tasks succeeded")
    else:
        flow_status = "❌ FAILED"
        status_emoji = "🔴"
        logger.error("Flow failed: All tasks failed including RL Agent")
    
    # Combine standard report with agent report if available
    if agent_report and agent_success:
        combined_report = f"""# {status_emoji} 15-Minute Home Data Update Flow Report

**Status:** {flow_status}

## 🤖 RL Agent Control Report

{agent_report}

---

## 📋 System Tasks Report

{standard_report}

---

*Flow completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    else:
        combined_report = f"""# {status_emoji} 15-Minute Home Data Update Flow Report

**Status:** {flow_status}

## ⚠️ RL Agent Status
RL Agent control failed. See the failure artifact and logs for details.

**Tasks Summary:** {successful_tasks}/{total_tasks} successful

---

## 📋 System Tasks Report

{standard_report}

---

*Flow completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # Log summary with clear status
    logger.info(f"🏠 Home data flow completed: {flow_status}")
    logger.info(f"📊 Tasks: {successful_tasks}/{total_tasks} successful")
    logger.info(f"🤖 RL Agent: {'✅ Success' if agent_success else '❌ Failed'}")
    
    return combined_report

@task(retries=1, retry_delay_seconds=120, task_run_name="fetch-home-data")
def fetch_home_data() -> Dict[str, Any]:
    """Fetch current home data and run RL agent control."""
    logger.info("Starting home data fetching and agent control")
    
    results = []
    
    # Fetch Thermia data first (heat pump data)
    try:
        result = fetch_thermia_data()
        results.append(result)
        logger.info("✅ Thermia data fetched successfully")
    except Exception as e:
        logger.error(f"❌ Failed to fetch Thermia data: {e}")
        results.append({'script_path': SCRIPT_PATHS['thermia_data'], 'success': False, 'error': str(e)})
    
    # Fetch energy consumption data
    try:
        result = fetch_energy_consumption()
        results.append(result)
        logger.info("✅ Energy consumption data fetched successfully")
    except Exception as e:
        logger.error(f"❌ Failed to fetch energy consumption data: {e}")
        results.append({'script_path': SCRIPT_PATHS['consumption_data'], 'success': False, 'error': str(e)})
    
    # Fetch actual load data
    try:
        result = fetch_actual_load()
        results.append(result)
        logger.info("✅ Actual load data fetched successfully")
    except Exception as e:
        logger.error(f"❌ Failed to fetch actual load data: {e}")
        results.append({'script_path': SCRIPT_PATHS['consumption_actual_load'], 'success': False, 'error': str(e)})
    
    successful = sum(1 for r in results if r.get('success', False))
    logger.info(f"Home data fetching completed: {successful}/{len(results)} successful")
    
    # Run the RL agent control task and create a report
    try:
        agent_result = run_agent()
        results.append(agent_result)
        
        if agent_result.get('success', False):
            logger.info("✅ Agent control task completed successfully")
            
            # Create the beautiful report
            agent_report = create_agent_report(agent_result)
            
            # Log the report for Prefect UI
            logger.info("📊 RL Agent Report Generated:")
            logger.info("\n" + agent_report)
            
        else:
            logger.error("❌ Failed to run agent control task")
            
    except Exception as e:
        logger.error(f"❌ Failed to run agent control task: {e}")
        results.append({'script_path': SCRIPT_PATHS['rl_agent'], 'success': False, 'error': str(e)})
    
    successful = sum(1 for r in results if r.get('success', False))
    logger.info(f"Home data fetching and agent control completed: {successful}/{len(results)} successful")
    
    return {
        'task_name': 'fetch_home_data',
        'results': results,
        'successful_count': successful,
        'total_count': len(results),
        'agent_report': agent_report if 'agent_report' in locals() else None
    }

@task(retries=1, retry_delay_seconds=120, task_run_name="fetch-solar-data")
def fetch_solar_data() -> Dict[str, Any]:
    """Fetch solar production data and generate predictions."""
    logger.info("Starting solar data fetching and predictions")
    
    results = []
    
    # Fetch actual solar production data
    try:
        result = fetch_solar_actual()
        results.append(result)
        logger.info("✅ Solar actual data fetched successfully")
    except Exception as e:
        logger.error(f"❌ Failed to fetch solar actual data: {e}")
        results.append({'script_path': SCRIPT_PATHS['solar_actual'], 'success': False, 'error': str(e)})
    
    # Generate solar predictions
    try:
        result = fetch_solar_predictions()
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

@task(retries=1, retry_delay_seconds=300, timeout_seconds=24*60*60, task_run_name="train-price-models")  # 1 day timeout
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
                timeout=24*60*60,  # 1 day per model
                task_name=f"train-price-model-{model_type}"
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

@task(retries=1, retry_delay_seconds=300, timeout_seconds=24*60*60, task_run_name="train-demand-model")  # 1 day timeout
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
            timeout=24*60*60,  # 1 day
            task_name=f"train-demand-model-{trials}-trials"
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

@task(retries=1, retry_delay_seconds=300, timeout_seconds=48*60*60, task_run_name="train-rl-agent")  # 2 day timeout
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
            timeout=48*60*60,  # 2 days
            task_name=f"train-rl-agent-{sanity_check_steps}-steps"
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

@task(retries=1, retry_delay_seconds=60, task_run_name="generate-price-predictions")
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
        
        result = run_python_script(
            SCRIPT_PATHS['price_predictions'], 
            args=args,
            task_name=f"generate-price-predictions-{model}"
        )
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
            description="Hourly update of exogenic data (prices, CO2/gas/coal, weather)",
            tags=["data-fetching", "hourly", "exogenic"],
            schedules=[
                Cron(
                    "0 * * * *",  # Every hour at minute 0
                    timezone="UTC"
                )
            ]
        ),
        
        # 15-minute home data update  
        home_data_flow.to_deployment(
            name="15min-home-data",
            description="15-minute update of home data (consumption, Thermia) - runs at specific clock times",
            tags=["data-fetching", "15min", "home"],
            schedules=[
                Cron(
                    "0,15,30,45 * * * *",  # At 00, 15, 30, 45 minutes past every hour
                    timezone="UTC"
                )
            ]
        ),
        
        # Daily energy pipeline
        daily_energy_pipeline.to_deployment(
            name="daily-energy-pipeline",
            description="Daily comprehensive energy pipeline with data updates and predictions",
            tags=["daily", "pipeline", "predictions"],
            schedules=[
                Cron(
                    "0 6 * * *",  # Daily at 6 AM
                    timezone="UTC"
                )
            ]
        ),
        
        # Weekly price model training (Sunday 02:00)
        weekly_model_training.to_deployment(
            name="weekly-price-training",
            description="Weekly training of price prediction models (trend, peak, valley)",
            tags=["training", "weekly", "prices"],
            parameters={
                "train_price_models_flag": True,
                "train_demand_model_flag": False,
                "price_production_mode": True
            },
            schedules=[
                Cron(
                    "0 2 * * 0",  # Sunday at 2 AM
                    timezone="UTC"
                )
            ]
        ),
        
        # Weekly demand model training (Monday 02:00) 
        weekly_model_training.to_deployment(
            name="weekly-demand-training",
            description="Weekly training of demand prediction model",
            tags=["training", "weekly", "demand"],
            parameters={
                "train_price_models_flag": False,
                "train_demand_model_flag": True,
                "demand_production_mode": True,
                "demand_trials": 50
            },
            schedules=[
                Cron(
                    "0 2 * * 1",  # Monday at 2 AM
                    timezone="UTC"
                )
            ]
        ),
        
        # Weekly RL agent training (Tuesday 02:00)
        rl_training_flow.to_deployment(
            name="weekly-rl-training",
            description="Weekly training of RL battery control agent",
            tags=["training", "weekly", "rl"],
            parameters={
                "sanity_check_steps": 10,
                "start_date": "2027-01-01",
                "update_data_first": True
            },
            schedules=[
                Cron(
                    "0 2 * * 2",  # Tuesday at 2 AM
                    timezone="UTC"
                )
            ]
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
    parser.add_argument("--test-parsing", action="store_true",
                       help="Test the agent output parsing logic")
    
    args = parser.parse_args()
    
    if args.test_parsing:
        # Test the parsing logic with sample output
        sample_output = [
            "================================================================================",
            "                        ENERGY MANAGEMENT SYSTEM DECISION",
            "================================================================================",
            "",
            "  TIMESTAMP: 2025-06-02 09:53 (Monday)",
            "",
            "--------------------------------------------------------------------------------",
            "                               CURRENT SYSTEM STATE",
            "--------------------------------------------------------------------------------",
            "",
            "  BATTERY STATUS:",
            "     State of Charge:      64.0%  ( 14.1 kWh / 22.0 kWh)",
            "",
            "  MARKET CONDITIONS:",
            "     Current Price:         42.3 ore/kWh  [HIGH]",
            "     24h Average:           30.9 ore/kWh",
            "     Price Range (24h):      5.7 - 65.2 ore/kWh",
            "",
            "  FORECAST SUMMARY:",
            "     Solar (24h):           54.7 kWh     [GOOD]",
            "     Load Avg (24h):         0.5 kW      [LOW]",
            "",
            "--------------------------------------------------------------------------------",
            "                             AI AGENT RECOMMENDATION",
            "--------------------------------------------------------------------------------",
            "",
            "  NEURAL NETWORK OUTPUT:",
            "     Raw Action Value:    +0.2682  (range: -1.0 to +1.0)",
            "",
            "  RECOMMENDED ACTION:",
            "     Battery Command:     <<<      DISCHARGE at 2.7 kW       <<<",
            "",
            "--------------------------------------------------------------------------------",
            "                                 DECISION CONTEXT",
            "--------------------------------------------------------------------------------",
            "",
            "  DISCHARGING STRATEGY:",
            "     • Battery has sufficient charge (64.0%)",
            "     • Current price is high (42.3 ore/kWh)",
            "     • Selling energy to grid during favorable conditions",
            "",
            "  NEXT EVALUATION: 10:08 (15 minutes)",
            "",
            "================================================================================",
        ]
        
        # Enable debug logging
        logging.getLogger().setLevel(logging.DEBUG)
        
        print("Testing agent output parsing...")
        parsed_info = _parse_agent_output(sample_output)
        
        print("\nParsed information:")
        for key, value in parsed_info.items():
            print(f"  {key}: {value}")
        
        print(f"\nCommand type: {parsed_info.get('command_type')}")
        print(f"Power kW: {parsed_info.get('power_kw')}")
        print(f"Battery command: {parsed_info.get('battery_command')}")
        print(f"Action value: {parsed_info.get('action_value')}")
        
        # Test report creation
        mock_result = {
            'success': True,
            'agent_info': parsed_info,
            'timestamp': '2025-06-02T09:53:34'
        }
        
        print("\nGenerating report...")
        report = create_agent_report(mock_result)
        print("\nGenerated report:")
        print(report)
        
    elif args.serve:
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
        print("Use --test-parsing to test the agent output parsing logic") 