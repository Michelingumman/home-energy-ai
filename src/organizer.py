# prefect version 3.4.1
from prefect import flow, task, prefect_shell
from prefect_shell import shell_run
from datetime import datetime
from typing import List, Optional

"""
solar_updateData_script = "C:/_Projects/home-energy-ai/src/predictions/solar/actual_data/FetchSolarProductionData.py" 

solar_predictions_script = "C:/_Projects/home-energy-ai/src/predictions/solar/makeSolarPrediction.py"

price_updateData_script = "C:/_Projects/home-energy-ai/src/predictions/prices/getPriceRelatedData.py"

Co2GasCoal_update_script = "C:/_Projects/home-energy-ai/data/FetchCO2GasCoal.py"

weather_updateData_script = "C:/_Projects/home-energy-ai/src/predictions/demand/FetchWeatherData.py"

HomeConsumption_updateData_script = "C:/_Projects/home-energy-ai/src/predictions/demand/FetchEnergyData.py"

Thermia_updateData_script = "C:/_Projects/home-energy-ai/src/predictions/demand/Thermia/UpdateHeatPumpData.py"

"""

@task
def run_module(module: str, args: List[str] = []) -> None:
    """
    Generic task to invoke any `python -m <module> [args...]`
    """
    cmd = f"python -m {module} {' '.join(args)}"
    shell_run(command=cmd, stream_output=True)

@flow
def daily_energy_pipeline(
    gather: bool = True,
    price: bool = True,
    solar: bool = True,
    price_model: str = "merged",
    start_date: Optional[str] = None,
    horizon_days: float = 1.0,
):
    # default to today's date if none provided
    start_date = start_date or datetime.now().strftime("%Y-%m-%d")

    if gather:
        run_module("Gather_Data", args=[
            # you can expose these flags as params, e.g. no_solar: bool
            # "--no-solar", "--no-prices", "--no-weather"
        ])

    if price:
        run_module("predictions.prices.run_model", args=[
            "--model", price_model,
            "--start", start_date,
            "--horizon", str(horizon_days),
            "--valley-threshold", "0.5",
            # add "--production-mode" if needed
        ])

    if solar:
        run_module("predictions.solar.makeSolarPrediction")

@flow
def weekly_model_training(
    models: List[str] = ["trend", "peak", "valley"],
    production: bool = True,
):
    # always gather fresh data first
    run_module("Gather_Data")

    # spin up one task per model type
    for m in models:
        args = ["--model", m]
        if production:
            args.append("--production")
        run_module.submit("predictions.prices.train", args=args)
