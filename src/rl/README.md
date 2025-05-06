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
├ rl_config.json         # default hyper-params  
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

### Training

To train the hierarchical RL system:

```bash
python src/rl/train.py
```

Options:
- `--config PATH`: Specify a custom config file (default: src/models/rl/rl_config.json)
- `--mode MODE`: Training mode: 'hierarchical', 'short_term_only', or 'long_term_only'
- `--short_term_model PATH`: Path to pre-trained short-term model
- `--long_term_model PATH`: Path to pre-trained long-term model
- `--evaluate`: Evaluate the trained model after training

Example:
```bash
python src/rl/train.py --mode short_term_only --evaluate
```

### Running Simulations

To run a simulation with trained models:

```bash
python src/rl/simulations/run_simulation.py --short_term_model PATH --long_term_model PATH
```

Options:
- `--config PATH`: Specify a custom config file
- `--short_term_model PATH`: Path to trained short-term model (required)
- `--long_term_model PATH`: Path to trained long-term model (required)
- `--output_dir PATH`: Directory to save simulation results

Example:
```bash
python src/rl/simulations/run_simulation.py \
    --short_term_model src/models/rl/saved_models/short_term_agent_latest \
    --long_term_model src/rl/saved_models/long_term_agent_latest
```

## Features

- **Hierarchical Control**: A two-level control system with:
  - Long-term agent (4h time steps) planning SoC corridors and appliance windows
  - Short-term agent (1h time steps) keeping the plan on track and handling overrides

- **Battery Management**: Protects a 22 kWh battery between 20%-90% SoC with degradation modeling

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

## Customization

Adjust parameters in `rl_config.json` to customize:
- Battery capacity and constraints
- Training parameters (learning rates, episode length, etc.)
- Penalty factors for peak loads
- Reward factors for comfort 