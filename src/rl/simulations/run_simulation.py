"""
Run a simulation of the hierarchical RL system.

Loads trained agents and runs a week-long simulation with the hierarchical controller.
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import pandas as pd

# Import our custom components
from src.rl.custom_env import HomeEnergyEnv
from src.rl.wrappers import LongTermEnv
from src.rl.agent import HierarchicalController


def load_config(config_path: str = "src/rl/rl_config.json") -> dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to the config file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            print(f"Loaded configuration from {config_path}")
            return config
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Using default configuration")
        return {}


def run_simulation(config: dict, short_term_model_path: str, long_term_model_path: str) -> dict:
    """
    Run a simulation with trained models.
    
    Args:
        config: Configuration dictionary
        short_term_model_path: Path to trained short-term model
        long_term_model_path: Path to trained long-term model
        
    Returns:
        dict: Simulation results
    """
    # Create the hierarchical controller
    controller = HierarchicalController(
        config=config,
        short_term_model_path=short_term_model_path,
        long_term_model_path=long_term_model_path
    )
    
    # Run a single episode
    print("Running simulation...")
    
    # Reset environments
    short_term_obs, _ = controller.base_env.reset(seed=config.get("random_seed", 42))
    long_term_obs, _ = controller.long_term_env.reset(seed=config.get("random_seed", 42))
    
    # Collect simulation data
    soc_values = []
    price_values = []
    demand_values = []
    solar_values = []
    grid_costs = []
    battery_costs = []
    battery_actions = []
    appliance_states = []
    net_consumption = []
    
    # Step through the environment
    total_reward = 0
    done = False
    info = None
    
    # Track time
    start_datetime = controller.base_env.start_datetime
    
    while not done:
        # Get long-term plan
        long_term_action, _ = controller.long_term_agent.predict(long_term_obs, deterministic=True)
        
        # Execute long-term step (which handles 4 hours internally)
        long_term_obs, reward, done, _, info = controller.long_term_env.step(long_term_action)
        
        # Record data from the info dictionary
        for i in range(min(4, len(info.get("step_grid_costs", [])))):
            current_hour = controller.base_env.current_hour - 4 + i
            current_datetime = start_datetime + datetime.timedelta(hours=current_hour)
            
            # Record state of charge
            soc_values.append({
                "timestamp": current_datetime,
                "value": controller.base_env.battery.soc
            })
            
            # Record price, if available
            if current_hour < len(controller.base_env.price_history):
                price_values.append({
                    "timestamp": current_datetime,
                    "value": controller.base_env.price_history[current_hour]
                })
            
            # Record grid costs
            if i < len(info.get("step_grid_costs", [])):
                grid_costs.append({
                    "timestamp": current_datetime,
                    "value": info["step_grid_costs"][i]
                })
            
            # Record battery costs
            if i < len(info.get("step_battery_costs", [])):
                battery_costs.append({
                    "timestamp": current_datetime,
                    "value": info["step_battery_costs"][i]
                })
            
            # Record solar production
            if i < len(controller.base_env.solar_history) and current_hour < len(controller.base_env.solar_history):
                solar_values.append({
                    "timestamp": current_datetime,
                    "value": controller.base_env.solar_history[current_hour]
                })
            
            # Record net consumption
            if "net_consumption" in info:
                net_consumption.append({
                    "timestamp": current_datetime,
                    "value": info.get("net_consumption", 0)
                })
        
        total_reward += reward
        
        # Print progress
        print(f"Hour: {controller.base_env.current_hour}/{controller.base_env.simulation_hours}, " +
              f"SoC: {controller.base_env.battery.soc:.2f}, " +
              f"Cost: {controller.base_env.total_cost:.2f}")
    
    # Compile results
    results = {
        "total_reward": total_reward,
        "total_cost": controller.base_env.total_cost,
        "peak_power": controller.base_env.peak_power,
        "soc_values": soc_values,
        "price_values": price_values,
        "grid_costs": grid_costs,
        "battery_costs": battery_costs,
        "solar_values": solar_values,
        "net_consumption": net_consumption
    }
    
    return results


def plot_results(results: dict, output_dir: str) -> None:
    """
    Plot and save simulation results.
    
    Args:
        results: Simulation results dictionary
        output_dir: Directory to save plots
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert lists of dictionaries to DataFrames for easier plotting
    soc_df = pd.DataFrame(results["soc_values"])
    price_df = pd.DataFrame(results["price_values"])
    grid_costs_df = pd.DataFrame(results["grid_costs"])
    battery_costs_df = pd.DataFrame(results["battery_costs"])
    solar_df = pd.DataFrame(results["solar_values"])
    net_consumption_df = pd.DataFrame(results["net_consumption"])
    
    # Plot 1: State of Charge
    plt.figure(figsize=(12, 6))
    plt.plot(soc_df["timestamp"], soc_df["value"], label="Battery SoC")
    plt.fill_between(soc_df["timestamp"], 0.2, 0.9, alpha=0.2, color="green", label="Allowed SoC Range")
    plt.title("Battery State of Charge")
    plt.xlabel("Time")
    plt.ylabel("SoC")
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path / "battery_soc.png")
    
    # Plot 2: Electricity Price and Costs
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Price on left axis
    ax1.plot(price_df["timestamp"], price_df["value"], 'b-', label="Electricity Price")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Price (SEK/kWh)", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Costs on right axis
    ax2 = ax1.twinx()
    ax2.plot(grid_costs_df["timestamp"], grid_costs_df["value"], 'r-', label="Grid Cost")
    ax2.plot(battery_costs_df["timestamp"], battery_costs_df["value"], 'g-', label="Battery Degradation Cost")
    ax2.set_ylabel("Cost (SEK)", color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    
    plt.title("Electricity Price and Costs")
    plt.grid(True)
    plt.savefig(output_path / "costs.png")
    
    # Plot 3: Power Balance
    plt.figure(figsize=(12, 6))
    plt.plot(solar_df["timestamp"], solar_df["value"], 'y-', label="Solar Production")
    plt.plot(net_consumption_df["timestamp"], net_consumption_df["value"], 'r-', label="Net Grid Consumption")
    plt.axhline(y=0, color='k', linestyle='--')
    plt.title("Power Balance")
    plt.xlabel("Time")
    plt.ylabel("Power (kW)")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path / "power_balance.png")
    
    # Save summary data to text file
    with open(output_path / "summary.txt", "w") as f:
        f.write(f"Total Reward: {results['total_reward']:.2f}\n")
        f.write(f"Total Cost: {results['total_cost']:.2f} SEK\n")
        f.write(f"Peak Power: {results['peak_power']:.2f} kW\n")
        
        # Calculate average battery state of charge
        avg_soc = np.mean(soc_df["value"])
        f.write(f"Average Battery SoC: {avg_soc:.2f}\n")
        
        # Calculate average electricity price and costs
        avg_price = np.mean(price_df["value"])
        total_grid_cost = np.sum(grid_costs_df["value"])
        total_battery_cost = np.sum(battery_costs_df["value"])
        
        f.write(f"Average Electricity Price: {avg_price:.2f} SEK/kWh\n")
        f.write(f"Total Grid Cost: {total_grid_cost:.2f} SEK\n")
        f.write(f"Total Battery Degradation Cost: {total_battery_cost:.2f} SEK\n")
    
    print(f"Results saved to {output_path}")


def main(args):
    """
    Main function to run simulation.
    
    Args:
        args: Command-line arguments
    """
    # Load configuration
    config = load_config(args.config)
    
    # Override with command-line arguments if provided
    short_term_model_path = args.short_term_model or config.get("short_term_model_path")
    long_term_model_path = args.long_term_model or config.get("long_term_model_path")
    
    # Ensure model paths are valid
    if not short_term_model_path or not os.path.exists(short_term_model_path):
        print(f"Error: Short-term model not found at {short_term_model_path}")
        return
    
    if not long_term_model_path or not os.path.exists(long_term_model_path):
        print(f"Error: Long-term model not found at {long_term_model_path}")
        return
    
    print(f"Using short-term model: {short_term_model_path}")
    print(f"Using long-term model: {long_term_model_path}")
    
    # Run simulation
    results = run_simulation(config, short_term_model_path, long_term_model_path)
    
    # Determine output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"src/rl/simulations/results/sim_{timestamp}"
    
    # Plot and save results
    plot_results(results, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation with trained models")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="src/rl/rl_config.json",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--short_term_model", 
        type=str, 
        help="Path to trained short-term model"
    )
    
    parser.add_argument(
        "--long_term_model", 
        type=str, 
        help="Path to trained long-term model"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    main(args) 