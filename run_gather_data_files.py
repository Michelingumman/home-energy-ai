import subprocess
import datetime
from pathlib import Path
# THIS RUNS AUTOMATICALLY VIA TASK SCHEDULER IN WINDOWS


# Define the paths to the scripts

# This script fetches the actual solar production data from Home Assistant and merged with the previous data in "ActualSolarProductionData.csv"
solar_updateData_script = "src/predictions/solar/actual_data/FetchSolarProductionData.py" 

# This script makes the solar prediction based on the Forecast.Solar API
solar_prediction_script = "src/predictions/solar/makeSolarPrediction.py"


# This script fetches the price related data from ElectricityMaps API (grid features) and mgrey.se/espot API (current e-spot price)
price_updateData_script = "src/predictions/prices/getPriceRelatedData.py"






# Log file path
log_file = "gather_data_log.txt"

def log_message(message):
    """Logs messages with timestamps to a log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

def run_script(script_path, script_name):
    """Runs a script and logs the output and errors."""

    log_message(f"Starting {script_name}...")
    try:
        result = subprocess.run(["python", script_path], capture_output=True, text=True, check=True)
        log_message(f"{script_name} completed successfully.")
        log_message(f"Output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        log_message(f"ERROR in {script_name}:\n{e.stderr}")
    


# Run both scripts
log_message(f"\n\n\n\n")
log_message(f"################################## START OF SCRIPT {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ##################################")

run_script(solar_updateData_script, "Solar Data Update")
log_message(f"\n\n")
run_script(solar_prediction_script, "Solar Prediction")
log_message(f"\n\n")
run_script(price_updateData_script, "Price Data Update")
log_message(f"\n\n")

log_message(f"################################## END OF SCRIPT  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ##################################")

print("Done!")