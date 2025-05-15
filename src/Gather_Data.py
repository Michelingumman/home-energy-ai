import subprocess
import datetime
import json
from pathlib import Path
# THIS RUNS AUTOMATICALLY VIA TASK SCHEDULER IN WINDOWS


# Define the paths to the scripts

# This script fetches the actual solar production data from Home Assistant and merged with the previous data in "ActualSolarProductionData.csv"
solar_updateData_script = "C:/_Projects/home-energy-ai/src/predictions/solar/actual_data/FetchSolarProductionData.py" 

# This script fetches the price related data from ElectricityMaps API (grid features) and mgrey.se/espot API (current e-spot price)
price_updateData_script = "C:/_Projects/home-energy-ai/src/predictions/prices/getPriceRelatedData.py"

# This script fetches the CO2, Gas and Coal data from the ElectricityMaps API
Co2GasCoal_update_script = "C:/_Projects/home-energy-ai/data/FetchCO2GasCoal.py"

# Weather script to fetch the weather data from OpenMeteo API
weather_updateData_script = "C:/_Projects/home-energy-ai/data/FetchWeatherData.py"

HomeConsumption_updateData_script = "C:/_Projects/home-energy-ai/src/predictions/demand/FetchConsumptionData.py"


# Log file paths
log_file = "C:/_Projects/home-energy-ai/src/logs/Gather_Data_log.log"
summary_log_file = "C:/_Projects/home-energy-ai/src/logs/Gather_Data_summary.log"
json_log_file = "C:/_Projects/home-energy-ai/src/logs/Gather_Data_machine_log.json"


def log_message(message):
    """Logs messages with timestamps to a log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

def log_summary(message):
    """Logs a summary message to the summary log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(summary_log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

def log_json(entry):
    """Appends a structured log entry to the JSON log file as a list of events."""
    # Read existing log
    try:
        with open(json_log_file, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    data.append(entry)
    with open(json_log_file, "w") as f:
        json.dump(data, f, indent=2)

def run_script(script_path, script_name):
    """Runs a script and logs the output and errors in a structured way."""
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message(f"\n==== STARTING: {script_name} ====")
    log_summary(f"STARTING: {script_name}")
    json_entry = {
        "script": script_name,
        "path": script_path,
        "start_time": start_time,
        "end_time": None,
        "status": None,
        "return_code": None,
        "stdout": None,
        "stderr": None
    }
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            check=True
        )
        log_message(f"STATUS: SUCCESS")
        if result.stdout.strip():
            log_message(f"--- OUTPUT ---\n{result.stdout.strip()}")
        if result.stderr.strip():
            log_message(f"--- STDERR (non-fatal) ---\n{result.stderr.strip()}")
        print(f"SUCCESS in {script_name}")
        summary = f"{script_name}: SUCCESS"
        status = "SUCCESS"
        return_code = result.returncode
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
    except subprocess.CalledProcessError as e:
        log_message(f"STATUS: FAILED")
        log_message(f"--- OUTPUT ---\n{e.stdout.strip() if e.stdout else ''}")
        log_message(f"--- ERROR ---\n{e.stderr.strip() if e.stderr else ''}")
        log_message(f"Return code: {e.returncode}")
        print(f"ERROR in {script_name}:\n{e.stderr}")
        summary = f"{script_name}: FAILED"
        status = "FAILED"
        return_code = e.returncode
        stdout = e.stdout.strip() if e.stdout else ''
        stderr = e.stderr.strip() if e.stderr else ''
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message(f"==== END: {script_name} ====\n")
    log_summary(f"{summary}")
    # Write to JSON log
    json_entry.update({
        "end_time": end_time,
        "status": status,
        "return_code": return_code,
        "stdout": stdout,
        "stderr": stderr
    })
    log_json(json_entry)

# Run both scripts
log_message(f"\n\n\n\n")
log_message(f"################################## START OF SCRIPT {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ##################################")
log_summary(f"################################## START OF SCRIPT {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ##################################")

run_script(solar_updateData_script, "Solar Data Update")
run_script(price_updateData_script, "Price Data Update")
run_script(Co2GasCoal_update_script, "CO2, Gas and Coal Data Update")
run_script(weather_updateData_script, "Weather Data Update")
run_script(HomeConsumption_updateData_script, "Consumption Data Update")

log_message(f"################################## END OF SCRIPT  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ##################################")
log_summary(f"################################## END OF SCRIPT  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ##################################")

print("Done!")