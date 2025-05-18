# Home Energy RL System

This is a hierarchical reinforcement learning system for optimizing home energy usage. The system controls a battery, manages appliances, and interfaces with solar production to minimize electricity costs while maintaining user comfort.

## Project Structure

```
src/rl/
├ components.py          # Battery, ApplianceManager, SolarSystem  
├ custom_env.py          # HomeEnergyEnv (1 h)  
├ wrappers.py            # LongTermEnv (aggregates 4 h → 1 step)  
├ agent.py               # ShortTermAgent, LongTermAgent, HierarchicalController  
├ train.py               # loads demand/price models, trains agents  
├ config.py            # Python-based configuration for RL training and environment
├ evaluate_agent.py    # Script to evaluate a trained agent and visualize performance
├ saved_models/        # Directory for storing trained model files (e.g., .zip)
├ logs/                # Directory for TensorBoard logs and other training logs
└ simulations/
    ├ run_simulation.py  # roll out a week with saved models
    └ results/           # saved simulation results
```

## Installation

1. The system requires Python 3.12.10 and the following packages:

```
gymnasium>=0.28.1
stable-baselines3>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
tensorflow>=2.12.0  # For loading prediction models
```

2. Install required packages:

```bash
pip install gymnasium stable-baselines3 numpy pandas matplotlib tensorflow
```

## Usage

### Configuration

Key parameters for the RL environment, training, and agent behavior are managed in `src/rl/config.py`.
This Python file allows for detailed configuration with comments and type hints.

Previously, configuration was in `rl_config.json`. This has been replaced by `config.py`.

### Training

To train an agent (currently focused on a short-term PPO agent):

```bash
python src/rl/train.py --model_name my_agent_v1 
# Optionally, specify a config (though the script imports src.rl.config by default):
# python src/rl/train.py --config src/rl/config.py --model_name my_agent_v1
```

- The script will use parameters defined in `src/rl/config.py`.
- Trained models will be saved in `src/rl/saved_models/`.
- TensorBoard logs will be in `src/rl/logs/`.

### Evaluating a Trained Agent

To evaluate a trained agent and generate performance plots:

```bash
python src/rl/evaluate_agent.py --model_path src/rl/saved_models/short_term_agent_final.zip 
# Optionally, specify a config (though the script imports src.rl.config by default):
# python src/rl/evaluate_agent.py --model_path src/rl/saved_models/short_term_agent_final.zip --config src/rl/config.py
```

- This will run the agent in the evaluation environment (configured via `src/rl/config.py` for evaluation settings).
- Detailed data will be saved as a CSV and a performance plot will be generated in `src/rl/simulations/results/`.

### Running a Simulation (Example)

To run a simulation with a trained agent (similar to evaluation but can be more flexible for specific scenarios):

```bash
python src/rl/simulations/run_simulation.py --model_path src/rl/saved_models/short_term_agent_final.zip
# To render the simulation (if supported by the environment's render mode):
# python src/rl/simulations/run_simulation.py --model_path src/rl/saved_models/short_term_agent_final.zip --render
# The script uses settings from src/rl/config.py by default.
```

## Features

- **Hierarchical Control**: A two-level control system with:
  - Long-term agent (4h time steps) planning SoC corridors and appliance windows
  - Short-term agent (1h time steps) keeping the plan on track and handling overrides

- **Battery Management**: Protects a 22 kWh battery between 20%-90% SoC with degradation modeling
  - Asymmetric action mapping that properly handles different charge/discharge power limits
  - Charge limit: 5 kW, Discharge limit: 10 kW
  - Action space (-1 to +1) maps proportionally to these different limits

- **Appliance Control**: Manages high-power appliances to avoid simultaneous peaks:
  - Floor heating
  - Heat pump
  - EV charger
  - Sauna
  - And other household appliances

- **Electricity Cost Optimization**: Minimizes grid costs by:
  - Using solar energy when available
  - Charging battery during low-price periods
  - Discharging during high-price periods
  - Avoiding peak loads that might be penalized
  - Applying export reward bonus when selling electricity back to the grid

## Customization

- **Environment Parameters**: Adjust settings in `src/rl/config.py` (e.g., `battery_capacity`, `simulation_days`, data paths).
- **Reward Function**: Modify reward components and their scaling factors in `src/rl/config.py` (e.g., `peak_penalty_factor`, `charge_bonus_multiplier`). These are used by `custom_env.py`.
- **Agent Hyperparameters**: Tune PPO agent parameters (e.g., `short_term_learning_rate`, `short_term_n_steps`) in `src/rl/config.py` for the `train.py` script.
- **Battery capacity and constraints**: Adjust parameters in `rl_config.json` to customize:
- Training parameters (learning rates, episode length, etc.)
- Penalty factors for peak loads
- Reward factors for comfort 