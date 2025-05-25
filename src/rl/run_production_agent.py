"""
run_production_agent.py

This script loads a trained RL agent and uses it to predict battery actions
based on current and forecasted conditions.

UPDATED: Fixed critical bugs and added robust error handling for production use.
"""

import argparse
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import logging
import subprocess
from dotenv import load_dotenv
import os, sys
import requests
import json
import traceback
from typing import Dict, Tuple, Optional, Union

# This allows for absolute imports from 'src'
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

dotenv_path = project_root / 'api.env'
load_dotenv(dotenv_path=dotenv_path)

SONNEN_TOKEN = os.getenv("SONNEN_TOKEN")

from sb3_contrib.ppo_recurrent import RecurrentPPO
from src.rl.config import get_config_dict


def setup_logging(log_to_file: bool = False, log_level: str = "INFO") -> None:
    """Setup logging configuration for production use."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    log_dir = project_root / "src" / "rl" / "logs"
    log_dir.mkdir(exist_ok=True)
    
    if log_to_file:
        log_file = log_dir / f"production_agent_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        print(f"Logging to file: {log_file}")
    else:
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )

# Configure logging
logger = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_HIGH_PRICE_ORE = 200.0  # öre/kWh, used as fallback

# --- Data Fetching Helper for Prices ---
def fetch_prices_for_date(target_date: datetime.date) -> Optional[list[float]]:
    """
    Fetches 24 hourly prices for a specific date and area from mgrey.se.
    Returns a list of 24 prices in öre/kWh, or None if request fails.
    """
    date_str = target_date.strftime("%Y-%m-%d")
    api_url = f'https://mgrey.se/espot?format=json&date={date_str}'
    logger.info(f"Fetching prices from {api_url} for area SE3")
    
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()  # Raises HTTPError for bad responses

        data = response.json()
        if 'SE3' in data:
            se3_data = data['SE3']
            hour_data = []
            for hour in range(24):
                if hour < len(se3_data) and 'price_sek' in se3_data[hour]:
                    hour_data.append(se3_data[hour]['price_sek'])
                else:
                    logger.warning(f"Missing price data for hour {hour} on {target_date}")
                    hour_data.append(None)
            return hour_data
        else:
            logger.error(f"SE3 data not found in response for {target_date}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch price data for {target_date}: {e}")
        return None
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        logger.error(f"Failed to parse price data for {target_date}: {e}")
        return None

def get_current_battery_soc(config: dict) -> float:
    """
    Fetches the current battery State of Charge (SoC) by calling the
    downloadEntityData.py script.
    """
    # Check for test scenario override
    if "_test_soc_override" in config:
        test_soc = config["_test_soc_override"]
        logger.info(f"Using test scenario SoC: {test_soc:.1%}")
        return test_soc
    
    logger.info("Getting current battery SoC...")

    script_path = project_root / "src" / "downloadEntityData.py"
    entity_id_to_fetch = "sensor.battery_soc" # Alias handled by downloadEntityData.py

    try:
        logger.debug(f"Calling {script_path} to get latest SoC for {entity_id_to_fetch}...")
        # Always refresh for production to get the very latest value
        process = subprocess.run(
            [sys.executable, str(script_path), "--entity", entity_id_to_fetch, "--latest_value", "--refresh"],
            capture_output=True,
            text=True,
            check=True, # Will raise CalledProcessError if script exits with non-zero
            timeout=30 # Add a timeout for robustness
        )
        
        soc_str = process.stdout.strip()
        logger.debug(f"Successfully fetched SoC string: '{soc_str}'")
        # The value from Home Assistant is likely a percentage (0-100).
        # The RL agent expects SoC as a fraction (0.0-1.0).
        soc_percentage = float(soc_str)
        current_soc_fraction = soc_percentage / 100.0
        
        # Clamp the value to be safe, though HA should give valid percentages
        current_soc_fraction = max(0.0, min(1.0, current_soc_fraction))
        logger.info(f"Current SoC: {current_soc_fraction:.2f} (from {soc_percentage:.1f}%)")
        return current_soc_fraction

    except subprocess.CalledProcessError as e:
        logger.error(f"Error calling downloadEntityData.py for SoC: {e}")
        logger.error(f"Stderr from script: {e.stderr}")
        fallback_soc = config.get("battery_initial_soc", 0.5)
        logger.warning(f"Falling back to default SoC: {fallback_soc}")
        return fallback_soc
    except ValueError as e:
        logger.error(f"Error converting SoC string to float: {e}")
        fallback_soc = config.get("battery_initial_soc", 0.5)
        logger.warning(f"Falling back to default SoC: {fallback_soc}")
        return fallback_soc
    except Exception as e:
        logger.error(f"Unexpected error in get_current_battery_soc: {e}")
        fallback_soc = config.get("battery_initial_soc", 0.5)
        logger.warning(f"Falling back to default SoC: {fallback_soc}")
        return fallback_soc

def get_price_forecast_production(current_dt: datetime.datetime, config: dict) -> np.ndarray:
    """
    Fetches electricity price forecast for the next 24 hours.
    Prices are in öre/kWh.
    """
    logger.info("Generating 24-hour price forecast...")
    
    today_date = current_dt.date()
    tomorrow_date = today_date + datetime.timedelta(days=1)

    prices_today_sek = fetch_prices_for_date(today_date) # List of 24 floats (öre/kWh) or None
    
    # If before 14:00, tomorrow's prices might not be available yet
    if datetime.datetime.now().hour < 14:
        # Use today's average as estimate for tomorrow
        if prices_today_sek and all(p is not None for p in prices_today_sek):
            prices_tomorrow_sek = [np.mean(prices_today_sek)] * 24
            logger.info("Using today's average price as tomorrow's estimate (early in day)")
        else:
            prices_tomorrow_sek = [DEFAULT_HIGH_PRICE_ORE] * 24
            logger.warning(f"No price data available, using fallback price {DEFAULT_HIGH_PRICE_ORE}")
    else:
        prices_tomorrow_sek = fetch_prices_for_date(tomorrow_date) # List of 24 floats (öre/kWh) or None
        if not prices_tomorrow_sek:
            # Fallback to today's prices shifted or default
            if prices_today_sek and all(p is not None for p in prices_today_sek):
                prices_tomorrow_sek = prices_today_sek.copy()
                logger.warning("Tomorrow's prices not available, using today's prices")
            else:
                prices_tomorrow_sek = [DEFAULT_HIGH_PRICE_ORE] * 24
                logger.warning(f"No price data available, using fallback price {DEFAULT_HIGH_PRICE_ORE}")

    # Handle missing data with fallbacks
    if not prices_today_sek:
        prices_today_sek = [DEFAULT_HIGH_PRICE_ORE] * 24
        logger.warning(f"Today's prices not available, using fallback price {DEFAULT_HIGH_PRICE_ORE}")
    
    # Fill any None values with nearby values or default
    for i, price in enumerate(prices_today_sek):
        if price is None:
            prices_today_sek[i] = DEFAULT_HIGH_PRICE_ORE
    
    for i, price in enumerate(prices_tomorrow_sek):
        if price is None:
            prices_tomorrow_sek[i] = DEFAULT_HIGH_PRICE_ORE

    logger.debug(f"Today's prices: {prices_today_sek}")
    logger.debug(f"Tomorrow's prices: {prices_tomorrow_sek}")
    current_hour_of_day = current_dt.hour

    merged_prices = np.concatenate((prices_today_sek, prices_tomorrow_sek))

    np.set_printoptions(suppress=True, precision=2)
    output_forecast_ore = merged_prices[current_hour_of_day:current_hour_of_day+24]
    
    # Apply test scenario price multiplier if present
    if "_test_price_multiplier" in config:
        multiplier = config["_test_price_multiplier"]
        output_forecast_ore = output_forecast_ore * multiplier
        logger.info(f"Applied test price multiplier: {multiplier}x")
    
    logger.info(f"Price forecast range: {output_forecast_ore.min():.2f} to {output_forecast_ore.max():.2f} öre/kWh")
    return output_forecast_ore.astype(np.float32)

def get_solar_forecast_production(current_dt: datetime.datetime, config: dict) -> np.ndarray:
    """
    Fetches/constructs solar production forecast (kW) for the next 72 hours (3 days).
    The forecast should be hourly, starting from the current hour of current_dt.
    """
    logger.info("Generating 72-hour solar forecast...")

    solar_prediction_path = project_root / "src" / "predictions" / "solar" / "forecasted_data" / "merged_predictions.csv"
    output_72h_forecast = np.zeros(3 * 24, dtype=np.float32) # Initialize with zeros

    try:
        if not solar_prediction_path.exists():
            logger.error(f"Solar prediction file not found at {solar_prediction_path}")
            return output_72h_forecast
            
        df = pd.read_csv(solar_prediction_path, index_col='timestamp', parse_dates=True)
        
        if df.empty:
            logger.error("Solar prediction file is empty")
            return output_72h_forecast
        
        # Ensure DataFrame index is timezone-aware (e.g., Europe/Stockholm)
        if df.index.tz is None:
            df.index = df.index.tz_localize('Europe/Stockholm', ambiguous='infer', nonexistent='shift_forward')
        else:
            df.index = df.index.tz_convert('Europe/Stockholm')

        # Ensure current_dt is also timezone-aware for proper comparison
        target_tz = df.index.tz 
        if current_dt.tzinfo is None:
            aware_current_dt = target_tz.localize(current_dt, is_dst=None)
        else:
            aware_current_dt = current_dt.astimezone(target_tz)

        # Define the start and end of the 72-hour window
        forecast_start_dt = aware_current_dt.replace(minute=0, second=0, microsecond=0)
        df_filtered = df[df.index >= forecast_start_dt]

        # Create a target hourly index for the 72-hour window
        target_hourly_index = pd.date_range(start=forecast_start_dt, periods=72, freq='h', tz=target_tz)
        logger.debug(f"Solar forecast window: {target_hourly_index[0]} to {target_hourly_index[-1]}")
        
        # Check available columns
        solar_column_name = 'kilowatt_hours'
        if solar_column_name not in df_filtered.columns:
            available_cols = df_filtered.columns.tolist()
            logger.debug(f"Column '{solar_column_name}' not found. Available: {available_cols}")
            # Try alternative column names
            alt_names = ['solar_production_kw', 'power', 'kw', 'production']
            for alt_name in alt_names:
                if alt_name in df_filtered.columns:
                    solar_column_name = alt_name
                    logger.info(f"Using alternative column '{solar_column_name}'")
                    break
            else:
                logger.error(f"No suitable solar column found in {available_cols}")
                return output_72h_forecast # Return zeros

        df_reindexed = df_filtered[solar_column_name].reindex(target_hourly_index, fill_value=0.0)
        
        if len(df_reindexed) == 72:
            output_72h_forecast = df_reindexed.values.astype(np.float32)
            logger.info(f"Solar forecast: {output_72h_forecast.sum():.2f} kWh total over 72h")
        else:
            logger.warning(f"Reindexed solar data length is {len(df_reindexed)}, expected 72. Padding with zeros.")
            temp_output = np.zeros(72, dtype=np.float32)
            actual_len = min(len(df_reindexed.values), 72)
            temp_output[:actual_len] = df_reindexed.values[:actual_len].astype(np.float32)
            output_72h_forecast = temp_output

    except Exception as e:
        logger.error(f"Error processing solar forecast: {e}")
        logger.debug(traceback.format_exc())

    return output_72h_forecast

def get_consumption_forecast_production(current_dt: datetime.datetime, config: dict) -> np.ndarray:
    """
    Fetches household consumption forecast (kW) for the next 72 hours (3 days).
    """
    logger.info("Generating 72-hour consumption forecast...")

    baseload_file = project_root / "src" / "predictions" / "demand" / "predictions" / "demand_predictions_4days.csv"
    output_72h_forecast = np.zeros(72, dtype=np.float32)  # Initialize with zeros
    
    try:
        if not baseload_file.exists():
            logger.error(f"Consumption prediction file not found at {baseload_file}")
            # Return a reasonable default consumption pattern
            default_consumption = config.get("fixed_baseload_kw", 0.5)
            return np.full(72, default_consumption, dtype=np.float32)
            
        df = pd.read_csv(baseload_file, index_col='timestamp', parse_dates=True)
        
        if df.empty:
            logger.error("Consumption prediction file is empty")
            default_consumption = config.get("fixed_baseload_kw", 0.5)
            return np.full(72, default_consumption, dtype=np.float32)
        
        logger.debug(f"Available consumption columns: {df.columns.tolist()}")
        
        # Find the consumption column - updated to include 'predictions'
        consumption_col = None
        potential_cols = ['predictions', 'predicted_demand_kw', 'consumption_kw', 'demand_kw', 'load_kw', 'prediction']
        for col in potential_cols:
            if col in df.columns:
                consumption_col = col
                break
        
        if consumption_col is None:
            logger.error(f"No consumption column found. Available columns: {df.columns.tolist()}")
            default_consumption = config.get("fixed_baseload_kw", 0.5)
            return np.full(72, default_consumption, dtype=np.float32)
        
        # Ensure timezone consistency
        if df.index.tz is None:
            df.index = df.index.tz_localize('Europe/Stockholm', ambiguous='infer', nonexistent='shift_forward')
        else:
            df.index = df.index.tz_convert('Europe/Stockholm')
        
        # Align current_dt timezone
        if current_dt.tzinfo is None:
            current_dt = df.index.tz.localize(current_dt)
        else:
            current_dt = current_dt.astimezone(df.index.tz)
            
        # Create target index for 72 hours
        forecast_start_dt = current_dt.replace(minute=0, second=0, microsecond=0)
        target_hourly_index = pd.date_range(start=forecast_start_dt, periods=72, freq='h', tz=df.index.tz)

        # Reindex to get exact 72-hour forecast
        df_reindexed = df[consumption_col].reindex(target_hourly_index, fill_value=config.get("fixed_baseload_kw", 0.5))
        
        output_72h_forecast = df_reindexed.values.astype(np.float32)
        logger.info(f"Consumption forecast: {output_72h_forecast.mean():.2f} kW average over 72h (from '{consumption_col}')")
        
    except Exception as e:
        logger.error(f"Error processing consumption forecast: {e}")
        logger.debug(traceback.format_exc())
        default_consumption = config.get("fixed_baseload_kw", 0.5)
        output_72h_forecast = np.full(72, default_consumption, dtype=np.float32)
        
    return output_72h_forecast


def get_capacity_metrics_production(config: dict) -> tuple[list[float], float, float]:
    """
    Fetches or calculates current capacity metrics.
    TODO: Replace with actual data tracking from persistent storage.
    """
    logger.debug("Getting capacity metrics (using placeholder data)")
    # These values would typically be loaded from a persistent store updated regularly.
    top3_peaks = [1.0, 0.8, 0.6]  # Example dummy values in kW
    peak_rolling_average = np.mean(top3_peaks)
    
    # Calculate actual month progress
    now = datetime.datetime.now()
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if now.month == 12:
        start_of_next_month = now.replace(year=now.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        start_of_next_month = now.replace(month=now.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
    
    month_duration = (start_of_next_month - start_of_month).total_seconds()
    elapsed = (now - start_of_month).total_seconds()
    month_progress = elapsed / month_duration
    
    return top3_peaks, peak_rolling_average, month_progress

def get_price_averages_production(price_forecast_hourly: np.ndarray, config: dict) -> tuple[float, float]:
    """
    Calculates 24-hour and 168-hour (7-day) average prices from the hourly forecast.
    """
    vat_mult = config.get("vat_mult", 1.25)
    avg_24h = np.mean(price_forecast_hourly[:24]) * vat_mult if len(price_forecast_hourly) >=24 else 0.0

    # For 168h average, in a real scenario, you'd query historical data.
    # Here, we'll use the provided forecast, acknowledging it's not a true 168h historical average.
    avg_168h = np.mean(price_forecast_hourly) * vat_mult if len(price_forecast_hourly) > 0 else 0.0
    logger.debug(f"Price averages (incl. VAT): 24h={avg_24h:.2f}, forecast-based={avg_168h:.2f} öre/kWh")
    return float(avg_24h), float(avg_168h)

# --- Helper Functions ---

def map_action_to_battery_power(action: float, config: dict) -> float:
    """
    Maps the normalized action value (-1 to 1) to battery power in kW.
    Negative for charging, positive for discharging.
    """
    max_charge_kw = config.get("battery_max_charge_power_kw", 5.0)
    max_discharge_kw = config.get("battery_max_discharge_power_kw", 10.0)

    if action >= 0:  # Discharging (0 to 1 maps to 0 to max_discharge)
        return action * max_discharge_kw
    else:  # Charging (-1 to 0 maps to -max_charge to 0)
        return action * max_charge_kw

def _log_beautiful_state_summary(
    current_soc: float,
    current_dt: datetime.datetime,
    price_forecast: np.ndarray,
    solar_forecast: np.ndarray,
    load_forecast: np.ndarray,
    action: float,
    battery_power: float,
    config: dict
) -> None:
    """
    Creates a beautiful, aesthetic summary of the current state and recommended action.
    """
    # Calculate some derived metrics
    battery_capacity = config.get("battery_capacity", 22.0)
    current_energy = current_soc * battery_capacity
    
    # Determine action description
    if battery_power < 0:
        action_desc = f"CHARGE at {abs(battery_power):.1f} kW"
        action_symbol = ">>>"
    elif battery_power > 0:
        action_desc = f"DISCHARGE at {battery_power:.1f} kW"
        action_symbol = "<<<"
    else:
        action_desc = "IDLE / NO CHANGE"
        action_symbol = "==="
    
    # Price context
    current_price = price_forecast[0]
    avg_price_24h = np.mean(price_forecast[:24])
    price_trend = "HIGH" if current_price > avg_price_24h * 1.1 else "LOW" if current_price < avg_price_24h * 0.9 else "AVERAGE"
    
    # Solar context
    solar_sum_24h = np.sum(solar_forecast[:24])
    solar_context = "GOOD" if solar_sum_24h > 2.0 else "LIMITED" if solar_sum_24h > 0.5 else "MINIMAL"
    
    # Load context
    avg_load_24h = np.mean(load_forecast[:24])
    load_context = "HIGH" if avg_load_24h > 2.0 else "MODERATE" if avg_load_24h > 1.0 else "LOW"
    
    # Time context
    time_str = current_dt.strftime("%H:%M")
    date_str = current_dt.strftime("%Y-%m-%d")
    weekday = current_dt.strftime("%A")
    
    # Create the beautiful display
    border = "=" * 80
    section_border = "-" * 80
    
    lines = [
        "",
        border,
        f"{'ENERGY MANAGEMENT SYSTEM DECISION':^80}",
        border,
        "",
        f"  TIMESTAMP: {date_str} {time_str} ({weekday})",
        "",
        section_border,
        f"{'CURRENT SYSTEM STATE':^80}",
        section_border,
        "",
        f"  BATTERY STATUS:",
        f"    State of Charge:     {current_soc:>6.1%}  ({current_energy:>5.1f} kWh / {battery_capacity:.1f} kWh)",
        "",
        f"  MARKET CONDITIONS:",
        f"    Current Price:       {current_price:>6.1f} ore/kWh  [{price_trend}]",
        f"    24h Average:         {avg_price_24h:>6.1f} ore/kWh",
        f"    Price Range (24h):   {price_forecast[:24].min():>6.1f} - {price_forecast[:24].max():.1f} ore/kWh",
        "",
        f"  FORECAST SUMMARY:",
        f"    Solar (24h):         {solar_sum_24h:>6.1f} kWh     [{solar_context}]",
        f"    Load Avg (24h):      {avg_load_24h:>6.1f} kW      [{load_context}]",
        "",
        section_border,
        f"{'AI AGENT RECOMMENDATION':^80}",
        section_border,
        "",
        f"  NEURAL NETWORK OUTPUT:",
        f"    Raw Action Value:    {action:>+7.4f}  (range: -1.0 to +1.0)",
        "",
        f"  RECOMMENDED ACTION:",
        f"    Battery Command:     {action_symbol} {action_desc:^30} {action_symbol}",
        "",
    ]
    
    # Add reasoning context
    reasoning_lines = [
        section_border,
        f"{'DECISION CONTEXT':^80}",
        section_border,
        "",
    ]
    
    if battery_power > 0:  # Discharging
        reasoning_lines.extend([
            f"  DISCHARGING STRATEGY:",
            f"    • Battery has sufficient charge ({current_soc:.1%})",
            f"    • Current price is {price_trend.lower()} ({current_price:.1f} ore/kWh)",
            f"    • Selling energy to grid during favorable conditions",
        ])
    elif battery_power < 0:  # Charging
        reasoning_lines.extend([
            f"  CHARGING STRATEGY:",
            f"    • Battery can accept charge (SoC: {current_soc:.1%})",
            f"    • Current price is {price_trend.lower()} ({current_price:.1f} ore/kWh)",
            f"    • Storing energy for later use during expensive periods",
        ])
    else:  # Idle
        reasoning_lines.extend([
            f"  CONSERVATION STRATEGY:",
            f"    • Maintaining current battery state ({current_soc:.1%})",
            f"    • Market conditions do not favor immediate action",
            f"    • Waiting for more favorable price signals",
        ])
    
    reasoning_lines.extend([
        "",
        f"  NEXT EVALUATION: {(current_dt + datetime.timedelta(minutes=15)).strftime('%H:%M')} (15 minutes)",
        "",
        border,
        ""
    ])
    
    # Log all lines
    for line in lines + reasoning_lines:
        logger.info(line)

def build_flattened_observation_for_production(
    current_soc: float,
    current_dt: datetime.datetime,
    price_forecast_hourly: np.ndarray, # Expected 24 values
    solar_forecast_hourly: np.ndarray, # Expected 72 values (3*24 hours)
    load_forecast_hourly: np.ndarray,  # Expected 72 values (3*24 hours)
    capacity_metrics: tuple[list[float], float, float],
    price_averages: tuple[float, float],
    config: dict,
    expected_size: Optional[int] = None
) -> np.ndarray:
    """
    Constructs the flattened observation array for the agent.
    Matches the observation space of HomeEnergyEnv after FlattenObservation wrapper.
    
    Args:
        expected_size: If provided, will adjust observation to match this size
    """
    time_idx = np.array([
        current_dt.hour,
        current_dt.minute / 60.0, # Normalized minute
        current_dt.weekday()
    ], dtype=np.float32)

    hour = current_dt.hour
    is_night = float(hour >= 22 or hour < 6)

    top3_peaks, peak_rolling_avg, month_prog = capacity_metrics
    price_24h_avg, price_168h_avg = price_averages

    # Build observation components in alphabetical order (FlattenObservation default)
    soc_arr = np.array([current_soc], dtype=np.float32)  # 1
    time_idx_arr = time_idx  # 3
    price_fc_arr = price_forecast_hourly[:24].astype(np.float32)  # 24 
    cap_metrics_arr = np.array([
        top3_peaks[0],
        top3_peaks[1], 
        top3_peaks[2],
        peak_rolling_avg,
        month_prog
    ], dtype=np.float32)  # 5
    price_avgs_arr = np.array([price_24h_avg, price_168h_avg], dtype=np.float32)  # 2
    is_night_arr = np.array([is_night], dtype=np.float32)  # 1
    load_fc_arr = load_forecast_hourly[:72].astype(np.float32)  # 72
    
    # Calculate the base size without solar forecast
    base_size = 1 + 3 + 24 + 5 + 2 + 1 + 72  # 108
    
    if expected_size is not None:
        # Calculate needed solar forecast size
        solar_size_needed = expected_size - base_size
        if solar_size_needed > 0:
            solar_fc_arr = np.zeros(solar_size_needed, dtype=np.float32)
            solar_fc_arr[:min(len(solar_forecast_hourly), solar_size_needed)] = solar_forecast_hourly[:solar_size_needed]
            logger.debug(f"Adjusted solar forecast to {solar_size_needed} features for expected size {expected_size}")
        else:
            logger.warning(f"Expected size {expected_size} is too small for base components ({base_size})")
            solar_fc_arr = solar_forecast_hourly[:72].astype(np.float32)  # fallback
    else:
        # Default behavior - use 72 hours of solar forecast
        solar_fc_arr = solar_forecast_hourly[:72].astype(np.float32)  # 72
    
    # The order must match the FlattenObservation wrapper's flattening order
    # which typically follows alphabetical order of dict keys:
    # capacity_metrics, is_night_discount, load_forecast, price_averages, price_forecast, soc, solar_forecast, time_idx
    observation_components = [
        cap_metrics_arr,      # 5
        is_night_arr,         # 1  
        load_fc_arr,          # 72
        price_avgs_arr,       # 2
        price_fc_arr,         # 24
        soc_arr,              # 1
        solar_fc_arr,         # Variable size
        time_idx_arr          # 3
    ]
    
    flat_observation = np.concatenate(observation_components).astype(np.float32)
    
    # Final safety check if expected_size is provided
    if expected_size is not None:
        if flat_observation.shape[0] < expected_size:
            padding = np.zeros(expected_size - flat_observation.shape[0], dtype=np.float32)
            flat_observation = np.concatenate([flat_observation, padding])
            logger.debug(f"Padded observation to reach {expected_size} features")
        elif flat_observation.shape[0] > expected_size:
            flat_observation = flat_observation[:expected_size]
            logger.debug(f"Truncated observation to {expected_size} features")
    
    logger.debug(f"Final observation shape: {flat_observation.shape}")
    return flat_observation

# --- Main Production Logic ---

def run_agent(model_path_str: Union[str, Path], config: dict, dry_run: bool = False):
    """
    Loads the model, gets current data, constructs observation,
    predicts action, and outputs it.
    
    Args:
        model_path_str: Path to the trained model
        config: Configuration dictionary
        dry_run: If True, only simulate without actual battery control
    """
    logger.info(f"Loading agent from: {model_path_str}")
    try:
        model = RecurrentPPO.load(str(model_path_str))
        logger.info("Agent loaded successfully.")
        
        # Get expected observation size from the model
        expected_obs_size = None
        if hasattr(model.policy, 'observation_space'):
            if hasattr(model.policy.observation_space, 'shape'):
                expected_obs_size = model.policy.observation_space.shape[0]
                logger.info(f"Model expects observation size: {expected_obs_size}")
        
    except Exception as e:
        logger.error(f"Failed to load the model from {model_path_str}: {e}")
        logger.debug(traceback.format_exc())
        return None

    # 1. Get current time
    current_datetime = datetime.datetime.now()
    logger.info(f"Current datetime: {current_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # 2. Fetch current data and forecasts
        logger.info("Fetching current data and forecasts...")
        current_soc = get_current_battery_soc(config)
        price_fc = get_price_forecast_production(current_datetime, config)
        solar_fc = get_solar_forecast_production(current_datetime, config)
        load_fc = get_consumption_forecast_production(current_datetime, config)
        cap_metrics = get_capacity_metrics_production(config)
        price_avgs = get_price_averages_production(price_fc, config)

        # Log summary
        logger.info(f"Data summary - SoC: {current_soc:.2f}, Prices: {price_fc[:3]}, Solar: {solar_fc[:3]}, Load: {load_fc[:3]}")

        # 3. Build flattened observation (matches FlattenObservation wrapper output)
        logger.info("Building flattened observation for the agent...")
        observation = build_flattened_observation_for_production(
            current_soc, current_datetime, price_fc, solar_fc, load_fc, cap_metrics, price_avgs, config,
            expected_size=expected_obs_size
        )

        # For RecurrentPPO, we need to manage the LSTM states.
        lstm_states = None # For the first prediction

        # 4. Predict action
        logger.info("Agent predicting action...")
        action, lstm_states = model.predict(observation, state=lstm_states, deterministic=True)
        action_scalar = float(action[0]) if isinstance(action, np.ndarray) else float(action)

        # 5. Map action to battery power
        target_battery_power_kw = map_action_to_battery_power(action_scalar, config)

        # 6. Log results
        logger.info(f"Raw agent action: {action_scalar:.4f}")
        logger.info(f"Recommended target battery power: {target_battery_power_kw:.2f} kW")
        if target_battery_power_kw < 0:
            logger.info(f"  -> Charge battery at {abs(target_battery_power_kw):.2f} kW")
        elif target_battery_power_kw > 0:
            logger.info(f"  -> Discharge battery at {target_battery_power_kw:.2f} kW")
        else:
            logger.info("  -> Battery idle / no change")

        # 7. Beautiful state summary
        _log_beautiful_state_summary(
            current_soc, current_datetime, price_fc, solar_fc, load_fc, 
            action_scalar, target_battery_power_kw, config
        )

        if dry_run:
            logger.info("DRY RUN: No actual battery control commands sent")
        else:
            # Here, you would send target_battery_power_kw to your battery control system.
            # e.g., set_battery_power(target_battery_power_kw)
            logger.warning("TODO: Implement actual battery control API call")

        return target_battery_power_kw
        
    except Exception as e:
        logger.error(f"Error during agent execution: {e}")
        logger.debug(traceback.format_exc())
        return None

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Run production RL agent for battery control")
    parser.add_argument("--model", type=str, help="Path to trained model file")
    parser.add_argument("--config", type=str, help="Path to config file (not yet implemented)")
    parser.add_argument("--log-file", action="store_true", help="Log to file in addition to console")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--dry-run", action="store_true", help="Simulate without actual battery control")
    parser.add_argument("--test-scenario", type=str, choices=["low-soc", "high-price", "low-price"], 
                       help="Test scenario with simulated conditions")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_to_file=args.log_file, log_level=args.log_level)
    
    # Load configuration
    try:
        config = get_config_dict()
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Apply test scenario modifications
    if args.test_scenario:
        config = _apply_test_scenario(config, args.test_scenario)
    
    # Determine model path
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = project_root / config["model_dir"] / "best_recurrent_model" / "best_model.zip"
        if not model_path.exists():
            # Try alternative path
            model_path = Path("src/rl/saved_models/final_recurrent_model.zip")
    
    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}")
        return 1
    
    # Run the agent
    try:
        result = run_agent(model_path_str=model_path, config=config, dry_run=args.dry_run or args.test_scenario)
        if result is not None:
            logger.info(f"[SUCCESS] Production agent completed successfully. Battery power: {result:.2f} kW")
            return 0
        else:
            logger.error("[FAILED] Production agent failed to complete")
            return 1
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Critical error in production agent: {e}")
        logger.debug(traceback.format_exc())
        return 1

def _apply_test_scenario(config: dict, scenario: str) -> dict:
    """Apply test scenario modifications to config."""
    logger.info(f"Applying test scenario: {scenario}")
    
    if scenario == "low-soc":
        config["_test_soc_override"] = 0.15  # 15% SoC
        config["_test_price_multiplier"] = 0.5  # Lower prices to encourage charging
        logger.info("Test scenario: Low SoC with favorable charging prices")
    elif scenario == "high-price":
        config["_test_soc_override"] = 0.85  # 85% SoC
        config["_test_price_multiplier"] = 3.0  # Very high prices
        logger.info("Test scenario: High SoC with very high electricity prices")
    elif scenario == "low-price":
        config["_test_soc_override"] = 0.30  # 30% SoC
        config["_test_price_multiplier"] = 0.1  # Very low prices
        logger.info("Test scenario: Medium SoC with very low electricity prices")
    
    return config

if __name__ == "__main__":
    sys.exit(main()) 