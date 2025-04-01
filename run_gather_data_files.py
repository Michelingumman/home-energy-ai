import subprocess
import datetime

# THIS RUNS AUTOMATICALLY VIA TASK SCHEDULER IN WINDOWS

# Define the paths to the scripts
solar_script = r"C:\_Projects\home-energy-ai\src\predictions\solar\prediction.py"
price_script = r"C:\_Projects\home-energy-ai\src\predictions\prices\gather_data.py"


# Log file path
log_file = r"C:\_Projects\home-energy-ai\gather_data_log.txt"

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
log_message(f"################################## START OF SCRIPT {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ##################################")
run_script(solar_script, "Solar Prediction")
run_script(price_script, "Price Gathering")

log_message(f"################################## END OF SCRIPT ##################################")

print("Done!")