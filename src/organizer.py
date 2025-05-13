# prefect version is 3.4.1
from prefect import flow, task
import subprocess
import logging
from typing import Optional, Literal
from datetime import datetime
# Gathers data for Solar, Prices (SE3 prices and Grid Features), Weather and CO2, Gas and Coal
import Gather_Data

# Price Model Pipeline
# Readn predictions/prices/README.md for what args and parameters are needed to utilize the scripts
import predictions.prices.train
import predictions.prices.run_model

# Solar Model Pipeline
import predictions.solar.makeSolarPrediction

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@task(name="Gather Energy Data")
def gather_data(solar: bool = True, prices: bool = True, weather: bool = True) -> None:
    """
    Task to gather all necessary data for predictions.
    
    Args:
        solar: Whether to gather solar production data
        prices: Whether to gather electricity price data
        weather: Whether to gather weather data
    """
    logger.info("Starting data gathering process")
    
    # Call the Gather_Data module - using subprocess for isolation
    cmd = ["python", "-m", "Gather_Data"]
    
    # Add any required arguments based on parameters
    if not solar:
        cmd.append("--no-solar")
    if not prices:
        cmd.append("--no-prices")
    if not weather:
        cmd.append("--no-weather")
        
    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info("Data gathering completed successfully")

@task(name="Run Price Model")
def run_price_model(
    model_type: Literal["trend", "peak", "valley", "merged"] = "merged",
    production_mode: bool = True,
    start_date: Optional[str] = None,
    horizon_days: float = 1.0,
    peak_threshold: Optional[float] = None,
    valley_threshold: float = 0.5,
) -> None:
    """
    Task to run the price prediction model.
    
    Args:
        model_type: The type of price model to run ("trend", "peak", "valley", or "merged")
        production_mode: Whether to run in production mode (true) or evaluation mode (false)
        start_date: Start date for predictions in YYYY-MM-DD format (default: current date)
        horizon_days: Prediction horizon in days (default: 1 day)
        peak_threshold: Probability threshold for peak detection (default: model default)
        valley_threshold: Probability threshold for valley detection (default: 0.5)
    """
    logger.info(f"Running price model: {model_type} (production_mode={production_mode})")
    
    # Build the command
    cmd = ["python", "-m", "predictions.prices.run_model",
           "--model", model_type]
    
    if production_mode:
        cmd.append("--production-mode")
    
    if start_date:
        cmd.extend(["--start", start_date])
    
    cmd.extend(["--horizon", str(horizon_days)])
    
    if peak_threshold is not None:
        cmd.extend(["--peak-threshold", str(peak_threshold)])
    
    cmd.extend(["--valley-threshold", str(valley_threshold)])
    
    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info(f"Price prediction completed successfully")

@task(name="Train Price Model")
def train_price_model(
    model_type: Literal["trend", "peak", "valley"] = "merged",
    test_mode: bool = False,
    production: bool = False,
) -> None:
    """
    Task to train a price prediction model.
    
    Args:
        model_type: The type of price model to train ("trend", "peak", or "valley")
        test_mode: Whether to run in test mode (faster training with less data)
        production: Whether to train for production (uses all data)
    """
    logger.info(f"Training price model: {model_type}")
    
    # Build the command
    cmd = ["python", "-m", "predictions.prices.train",
           "--model", model_type]
    
    if test_mode:
        cmd.append("--test-mode")
    
    if production:
        cmd.append("--production")
    
    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info(f"Price model training completed successfully")

@task(name="Make Solar Prediction")
def make_solar_prediction() -> None:
    """
    Task to make solar production predictions.
    """
    logger.info("Starting solar prediction")
    subprocess.run(["python", "-m", "predictions.solar.makeSolarPrediction"], check=True)
    logger.info("Solar prediction completed successfully")

@flow(name="Daily Home Energy Pipeline")
def daily_energy_pipeline(
    run_data_gathering: bool = True,
    run_price_prediction: bool = True,
    run_solar_prediction: bool = True,
    price_model_type: Literal["trend", "peak", "valley", "merged"] = "merged",
    price_start_date: Optional[str] = None,
    price_horizon_days: float = 1.0,
) -> None:
    """
    Daily pipeline to gather data and run predictions for home energy system.
    
    This flow orchestrates the entire process of collecting data, running price 
    predictions, and running solar predictions. Each step can be enabled/disabled
    with parameters.
    
    Args:
        run_data_gathering: Whether to run the data gathering step
        run_price_prediction: Whether to run the price prediction step
        run_solar_prediction: Whether to run the solar prediction step
        price_model_type: Type of price model to use in prediction
        price_start_date: Start date for price predictions (YYYY-MM-DD)
        price_horizon_days: Number of days to predict prices for
    """
    logger.info(f"Starting daily energy pipeline at {datetime.now().isoformat()}")
    
    # Only run steps that are enabled
    if run_data_gathering:
        gather_data()
    
    if run_price_prediction:
        # Use current date if no start date provided
        if not price_start_date:
            price_start_date = datetime.now().strftime("%Y-%m-%d")
            
        run_price_model(
            model_type=price_model_type,
            production_mode=True,
            start_date=price_start_date,
            horizon_days=price_horizon_days
        )
    
    if run_solar_prediction:
        make_solar_prediction()
    
    logger.info(f"Daily energy pipeline completed at {datetime.now().isoformat()}")

@flow(name="Weekly Model Training")
def weekly_model_training(
    train_trend: bool = True,
    train_peak: bool = True,
    train_valley: bool = True,
    production: bool = True
) -> None:
    """
    Weekly flow to retrain all price prediction models.
    
    This flow handles the retraining of price models. It's designed to run
    less frequently than the daily pipeline (e.g., weekly) as training is
    more resource intensive and benefits from having more accumulated data.
    
    Args:
        train_trend: Whether to train the trend model
        train_peak: Whether to train the peak model
        train_valley: Whether to train the valley model
        production: Whether to train for production (uses all data)
    """
    logger.info(f"Starting weekly model training at {datetime.now().isoformat()}")
    
    # First gather the latest data
    gather_data()
    
    # Train each requested model
    if train_trend:
        train_price_model(model_type="trend", production=production)
    
    if train_peak:
        train_price_model(model_type="peak", production=production)
    
    if train_valley:
        train_price_model(model_type="valley", production=production)
    
    logger.info(f"Weekly model training completed at {datetime.now().isoformat()}")

if __name__ == "__main__":
    # When run directly, execute the daily pipeline with default settings
    daily_energy_pipeline()
    
    # To run the weekly training, uncomment the line below:
    # weekly_model_training()
    
    # Note: For actual deployment, use prefect CLI to create deployments:
    # Example: prefect deployment build src/organizer.py:daily_energy_pipeline -n "Daily Run" --cron "0 6 * * *" --apply
    # Example: prefect deployment build src/organizer.py:weekly_model_training -n "Weekly Training" --cron "0 2 * * 0" --apply




