# Home Energy Reinforcement Learning (RL) Module

This directory contains the core components for a reinforcement learning (RL) system designed to optimize home energy usage. The system learns to control a home battery, considering grid prices, solar production (optional), and household consumption (optional) to minimize electricity costs and manage grid load effectively.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Core Scripts and Functionality](#core-scripts-and-functionality)
  - [`run_optimization.py`](#run_optimizationpy)
  - [`hpo.py`](#hpopy)
  - [`train.py`](#trainpy)
  - [`evaluate_agent.py`](#evaluate_agentpy)
- [Key Components](#key-components)
  - [`custom_env.py` - HomeEnergyEnv](#custom_envpy---homeenergyenv)
    - [Observation Space](#observation-space)
    - [Action Space](#action-space)
    - [Reward Components](#reward-components)
  - [`agent.py` - Agent Classes](#agentpy---agent-classes)
    - [RecurrentEnergyAgent](#recurrentenergyagent)
  - [`config.py`](#configpy)
  - [`components.py`](#componentspy)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
  - [1. Configuration](#1-configuration)
  - [2. Running Hyperparameter Optimization (HPO) and Training](#2-running-hyperparameter-optimization-hpo-and-training)
  - [3. Training Only (from parameters file or default config)](#3-training-only-from-parameters-file-or-default-config)
  - [4. Evaluating a Trained Agent](#4-evaluating-a-trained-agent)
- [Evaluation Metrics and Plotting](#evaluation-metrics-and-plotting)
- [Hyperparameter Optimization (HPO) Details](#hyperparameter-optimization-hpo-details)
- [Key Simulated Features](#key-simulated-features)
- [Agent Types](#agent-types)
  - [RecurrentEnergyAgent](#recurrentenergyagent)
- [Reward Function](#reward-function)
- [Command Line Options](#command-line-options)
- [Usage Examples](#usage-examples)

## Overview

The primary goal of this RL module is to train an agent that can intelligently manage a home battery system. It aims to:
- Minimize overall electricity costs by buying, selling, or storing energy optimally.
- Reduce peak power consumption from the grid to avoid capacity-based fees.
- Maximize the self-consumption of locally generated solar power.
- Maintain battery health by operating within desired State of Charge (SoC) limits and penalizing excessive degradation.
- Optimize charging/discharging based on daily patterns such as morning consumption and price forecasts.

The system uses a **RecurrentEnergyAgent** that uses RecurrentPPO with LSTM layers to maintain memory of past states, enabling complex strategies that can learn patterns over time.

## Directory Structure

```
src/rl/
├── agent.py                # Defines the RecurrentEnergyAgent
├── components.py           # Defines simulation components like the Battery
├── config.py               # Central configuration file for all parameters
├── custom_env.py           # Defines the `HomeEnergyEnv` RL environment
├── evaluate_agent.py       # Script for evaluating trained agents and plotting results
├── hpo.py                  # Handles Hyperparameter Optimization using Optuna
├── README.md               # This file
├── run_optimization.py     # Main script to run HPO and then train an agent with best params
├── safety_buffer.py        # (for future use)
├── test_load_agent.py      # Helper script to test agent loading
├── train.py                # Script for training agents
├── logs/                   # Directory for TensorBoard logs, HPO results, etc.
│   ├── hpo_results/        # Stores results from HPO runs
│   └── recurrent_<timestamp>/  # TensorBoard logs for recurrent agent training runs
├── saved_models/           # Directory for saved model checkpoints
│   ├── best_recurrent_model/   # Best recurrent model saved during training
│   └── recurrent_checkpoints/  # Periodic checkpoints for recurrent agent
└── simulations/
    └── results/            # Directory for evaluation plots and metrics from `run_optimization.py`
```

## Core Scripts and Functionality

### `run_optimization.py`
This is the **primary script** for a full workflow: running hyperparameter optimization (HPO) and then training a final agent with the best-found parameters.

- **Purpose**: Orchestrates HPO using Optuna, applies the best hyperparameters to the agent's configuration, trains the agent, evaluates it, and saves the model, metrics, and performance plots.
- **Key Arguments**:
    - `--trials VAL`: (int, default: 25) Number of Optuna HPO trials.
    - `--timeout VAL`: (int, default: None) Timeout in *seconds* for the HPO process.
    - `--skip-optimization`: (flag) Skip HPO and load parameters from a file. Requires `--params-file`.
    - `--params-file FILE_PATH`: (str) JSON file with parameters if skipping HPO.
    - `--timesteps VAL`: (int, default: None) Override `short_term_timesteps` from `config.py` for the final training.
    - `--simulation-days VAL`: (int, default: None) Override `simulation_days` from `config.py`.
- **How to run**: See the [How to Run](#how-to-run) section.

### `hpo.py`
- **Purpose**: Implements the hyperparameter optimization logic using Optuna and the TPESampler. It defines the `objective` function that Optuna tries to maximize. This script is primarily called by `run_optimization.py`.
- **Functionality**:
    - Samples hyperparameters for reward component weights and PPO algorithm parameters.
    - Trains a temporary RecurrentPPO model for a limited number of timesteps.
    - Evaluates the temporary model based on a custom score (combining mean reward, price correlation, peak average, SoC violations, costs, arbitrage, and export).
    - Saves the best parameters found to a JSON file (e.g., `best_hpo_params.json`) in a timestamped HPO results directory.
    - Generates HPO diagnostic plots (optimization history, parameter importances, parallel coordinate).

### `train.py`
- **Purpose**: Script for training the `RecurrentEnergyAgent`. It offers granular control over Stable Baselines3 callbacks (like `CheckpointCallback` and `EvalCallback`) for a single training run.
- **Key Arguments (consult script for full list)**:
    - `--config FILE_PATH`: Path to a custom configuration JSON file.
    - `--total_timesteps VAL`: Set the total number of training timesteps.
    - `--start-date YYYY-MM-DD --end-date YYYY-MM-DD`: Restrict training data to a specific period.
    - `--skip-sanity-check`, `--sanity-check-only`, `--sanity-check-steps N`: Options related to pre-training environment checks.
- **Relationship to `run_optimization.py`**: `run_optimization.py` is generally preferred for finding good hyperparameters and then training. `train.py` can be used for more manual training sessions.

### `evaluate_agent.py`
- **Purpose**: Contains functions to evaluate a trained agent's performance and generate detailed visualizations. It's used by `run_optimization.py` for final model evaluation and can also be run standalone.
- **Functionality**:
    - `evaluate_episode()`: Runs the agent for one episode and collects detailed step-by-step data.
    - `calculate_performance_metrics()`: Computes various metrics from episode data.
    - `plot_agent_performance()`: Generates a multi-panel plot showing SoC, prices, actions, power flows, grid peaks, and cumulative reward.
    - `plot_reward_components()`: Generates a plot showing the time series of each individual reward component.
    - `analyze_reward_component_distributions()`: Prints statistics and plots histograms for reward components.
- **Standalone Usage Arguments**:
    - `--model_path FILE_PATH`: Path to the saved agent model to evaluate.
    - `--start-month YYYY-MM`: Evaluate for a specific month.
    - `--random-start`: Use a random start date for evaluation instead of aligning to a month.
    - `--num_episodes N`: Number of evaluation episodes to run.

## Key Components

### `custom_env.py` - HomeEnergyEnv
This file defines the `HomeEnergyEnv` class, which is the core simulation environment for the RL agent. It adheres to the `gymnasium.Env` interface.

#### Observation Space
A `spaces.Dict` containing:
- `soc`: (Box, 1x1) Current battery State of Charge (0.0 to 1.0).
- `time_idx`: (Box, 3x1) `[hour_of_day, minute_of_hour (normalized 0-1), day_of_week (0-6)]`.
- `price_forecast`: (Box, 24x1) Grid electricity prices for the next 24 hours (öre/kWh).
- `solar_forecast`: (Box, 96x1) Solar production forecast for the next 4 days (96 hours) in kW.
- `capacity_metrics`: (Box, 5x1) `[top1_peak_kW, top2_peak_kW, top3_peak_kW, rolling_avg_peak_kW, month_progress (0-1)]`.
- `price_averages`: (Box, 2x1) `[24h_avg_price, 168h_avg_price]` (öre/kWh, VAT inclusive).
- `is_night_discount`: (Box, 1x1) Flag (0 or 1) if current time is in night discount period (22:00-06:00).

#### Action Space
A `spaces.Box` with a single continuous value in `[-1.0, 1.0]`:
- `-1.0`: Maximum battery charging rate.
- `0.0`: Battery idle.
- `1.0`: Maximum battery discharging rate.
The actual power is scaled by `battery_max_charge_power_kw` or `battery_max_discharge_power_kw` from the config. A `safe_action_mask` ensures actions do not immediately violate SoC limits.

#### Reward Components
The total reward is a weighted sum of several components, designed to guide the agent towards desired behaviors. Penalties are typically positive values that are subtracted from the total reward. Weights (`w_*`) are configurable and can be tuned by HPO.

1.  **Grid Cost (`grid_cost`)**:
    - **Purpose**: Penalizes the financial cost of importing electricity from the grid.
    - **Calculation**: Based on `grid_power_kw`, `spot_price` (with VAT), `grid_fee`, `energy_tax`, and `time_step_hours`. Scaled by `grid_cost_scaling_factor`.
    - **Contribution**: `weights['w_grid'] * (-components['grid_cost'])`

2.  **Capacity Penalty (`capacity_penalty`)**:
    - **Purpose**: Penalizes high peak power consumption from the grid to reduce capacity-based fees.
    - **Calculation**: If `grid_power_kw > peak_power_threshold_kw`, penalty is `(grid_power_kw - peak_power_threshold_kw) * peak_penalty_factor`.
    - **Contribution**: `weights['w_cap'] * (-components['capacity_penalty'])`

3.  **Battery Degradation Cost (`degradation_cost`)**:
    - **Purpose**: Accounts for the wear and tear on the battery due to charging/discharging.
    - **Calculation**: `abs(energy_change_in_storage_kwh) * battery_degradation_cost_per_kwh * battery_degradation_reward_scaling_factor`.
    - **Contribution**: `weights['w_deg'] * (-components['degradation_cost'])`

4.  **SoC Reward (`soc_reward`)**:
    - **Purpose**: Encourages maintaining SoC within a preferred range (`preferred_soc_min_base`, `preferred_soc_max_base`) and heavily penalizes violating hard limits (`soc_min_limit`, `soc_max_limit`). Includes extra penalties for very high SoC.
    - **Factors**: `soc_limit_penalty_factor`, `preferred_soc_reward_factor`, `high_soc_penalty_multiplier`, `very_high_soc_threshold`.
    - **Contribution**: `weights['w_soc'] * components['soc_reward']`

5.  **Potential-Based Shaping Reward (`shaping_reward`)**:
    - **Purpose**: Uses potential-based reward shaping (`gamma * soc_potential(next_soc) - soc_potential(current_soc)`) to guide the agent towards preferred SoC states smoothly without altering the optimal policy.
    - **Contribution**: `weights['w_shape'] * components['shaping_reward']`

6.  **Night-Time Charging Incentive**:
    - **Purpose**: Rewards charging the battery during night hours (22:00-06:00), typically when prices are lower or grid demand is less.
    - **Calculation**: `components['night_discount'] * components['battery_charging_power']` (only if battery is charging).
    - **Contribution**: `weights['w_night'] * components['night_discount'] * components['battery_charging_power']`

7.  **Arbitrage Bonus (`arbitrage_bonus`)**:
    - **Purpose**: Rewards charging at low prices and discharging at high prices (if `enable_explicit_arbitrage_reward` is true).
    - **Factors**: `charge_at_low_price_reward_factor`, `discharge_at_high_price_reward_factor`, price thresholds (fixed or percentile-based: `low_price_threshold_ore_kwh`, `high_price_threshold_ore_kwh`, `low_price_percentile`, `high_price_percentile`).
    - **Contribution**: `weights['w_arbitrage'] * components['arbitrage_bonus']`

8.  **Export Bonus (`export_bonus`)**:
    - **Purpose**: Rewards exporting surplus energy to the grid.
    - **Calculation**: If exporting: `abs(grid_energy_kwh) * (current_price_ore_per_kwh + export_reward_bonus_ore_kwh)`.
    - **Contribution**: `weights['w_export'] * components['export_bonus']`

9.  **Action Modification Penalty (`action_mod_penalty`)**:
    - **Purpose**: Penalizes the agent if its raw action was modified by the `safe_action_mask`. Encourages learning valid actions.
    - **Calculation**: If action modified: `abs(original_action - safe_action) * action_modification_penalty * escalation_factor`. Escalation factor increases with consecutive invalid actions up to `max_consecutive_penalty_multiplier`.
    - **Contribution**: `total_reward -= weights['w_action_mod'] * components['action_mod_penalty']`

### `agent.py` - Agent Classes

#### RecurrentEnergyAgent
- **Purpose**: Defines the `RecurrentEnergyAgent` class using the `RecurrentPPO` ("MlpLstmPolicy") model from sb3-contrib.
- **Key features**:
  - Maintains memory across timesteps using LSTM layers
  - Capable of learning patterns over time and developing multi-step strategies
  - Can adapt to diurnal patterns and remember charging/discharging history
- **Configuration**: Uses `n_lstm_layers` and `lstm_hidden_size` parameters from config to set up LSTM structure
- **Best suited for**: Complex scenarios requiring memory of past states, such as coordinating night charging with day peaks or adapting to recurring patterns.

### `config.py`
- **Purpose**: The central hub for all configurable parameters. This includes paths, simulation settings, battery characteristics, electricity pricing details, reward function parameters (including the `w_*` weights for HPO), PPO hyperparameters, and data augmentation toggles.
- **`get_config_dict()`**: Utility function to make all global variables in this file accessible as a dictionary.

### `components.py`
- **Purpose**: Defines classes for individual physical components of the home energy system, primarily the `Battery` class.
- **`Battery` Class**: Simulates battery operations including charging, discharging, efficiency losses, SoC tracking, and degradation cost calculation.

## Setup and Installation

1.  Ensure Python 3.10+ is installed.
2.  It's highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate    # On Windows
    ```
3.  Install the required packages from the main project's `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
    Key RL-specific dependencies include `gymnasium`, `stable-baselines3`, `sb3-contrib` (for RecurrentPPO), `optuna`.

## How to Run

All commands should be run from the root directory of the `home-energy-ai` project.

### 1. Configuration
Before any run, carefully review and adjust parameters in `src/rl/config.py`. This file controls nearly every aspect of the simulation, training, and HPO. Pay special attention to:
- Data paths: `price_predictions_path`, `consumption_data_path`, `solar_data_path`.
- Simulation settings: `simulation_days`, `time_step_minutes`.
- Battery parameters: `battery_capacity`, SoC limits, efficiencies.
- Reward weights: `w_grid`, `w_cap`, etc., if not using HPO to find them.
- Recurrent agent parameters: `n_lstm_layers`, `lstm_hidden_size`.

### 2. Running Hyperparameter Optimization (HPO) and Training
This is the recommended primary workflow using `run_optimization.py`.

**Example: Run HPO for 50 trials, then train the best model.**
```bash
python src/rl/run_optimization.py --trials 50
```

**Example: Run HPO with a 2-hour timeout.**
```bash
python src/rl/run_optimization.py --trials 100 --timeout 7200 # 7200 seconds = 2 hours
```
Output:
- HPO progress will be logged to the console.
- Best parameters and HPO plots will be saved in `src/rl/logs/hpo_results/<study_name>_<timestamp>/`.
- The script will then train a new agent using these best parameters.
- The trained model will be saved (e.g., `src/rl/saved_models/optimized_model_<timestamp>.zip`).
- Evaluation results (metrics JSON, config JSON, plots) for this model will be in `src/rl/simulations/results/optimized_performance_<timestamp>/`.

### 3. Training Only (from parameters file or default config)

**Option A: Skip HPO and use a specific parameters file (e.g., from a previous HPO run).**
```bash
python src/rl/run_optimization.py --skip-optimization --params-file src/rl/logs/hpo_results/<study_name>_<timestamp>/best_hpo_params.json
```
(Adjust the path to your actual `best_hpo_params.json` file, specifically the `params` part of it).

**Option B: Using `train.py` for a more manual/dev training run (uses defaults from `config.py` unless model loaded).**
```bash
python src/rl/train.py --total_timesteps 200000
```
To continue training a specific model using `train.py`:
```bash
python src/rl/train.py --short_term_model src/rl/saved_models/short_term_checkpoints/your_model.zip --total_timesteps 100000
```

### 4. Evaluating a Trained Agent
Use `evaluate_agent.py` to assess a previously trained model.

**Example: Evaluate a specific model for a specific month.**
```bash
python src/rl/evaluate_agent.py --model_path src/rl/saved_models/optimized_model_20231026_153000.zip --start-month 2023-07
```

**Example: Evaluate with a random start date for 5 episodes.**
```bash
python src/rl/evaluate_agent.py --model_path src/rl/saved_models/best_short_term_model/best_model.zip --random-start --num_episodes 5
```

**Example: Evaluate a recurrent model.**
```bash
python src/rl/evaluate_agent.py --model_path src/rl/saved_models/best_recurrent_model/best_model.zip --start-month 2023-07
```

Output:
- Metrics printed to the console.
- Plots saved in `src/rl/simulations/results/<model_name_or_timestamp>/`.

## Evaluation Metrics and Plotting

The `evaluate_agent.py` script (and `run_optimization.py` when it evaluates) calculates and can plot:

- **Metrics (from `calculate_performance_metrics`)**:
    - `total_reward`: Sum of all rewards over the episode.
    - `price_power_correlation`: Correlation between grid power and electricity price.
    - `cost_metrics`: `grid_cost` (SEK), `battery_cost` (SEK).
    - `peak_metrics`: `peak_rolling_average` (kW over top 3 hourly peaks), `max_peak` (kW).
    - `soc_metrics`: `violation_percentage` (SoC outside hard limits), `mean_soc`, `time_in_preferred_range`.
    - `reward_metrics`: Sum of each individual raw reward component.
    - `arbitrage_metrics`: `arbitrage_score`, `low_price_charging_pct`, `high_price_discharging_pct`.
    - `export_metrics`: `export_revenue` (SEK), `export_percentage`.
    - Financial Outcome: Net financial result of the episode.
- **Plots**:
    - **`plot_agent_performance`**: A multi-panel plot showing:
        1.  Battery SoC and Electricity Price (hourly).
        2.  Agent Actions (original and safe/masked) and Battery Power (15-min).
        3.  Household Consumption, Net Grid Power, Solar Production (hourly).
        4.  Grid Power Peaks (hourly, discounted for night) and calculated Capacity Fee.
        5.  Cumulative Reward (15-min).
        *Night periods (22:00-06:00) are shaded on all subplots.*
    - **`plot_reward_components`**: Time series of each individual reward component and the total reward.
    - **`analyze_reward_component_distributions`**: Histograms showing the distribution of each reward component.

## Hyperparameter Optimization (HPO) Details

- **Tool**: Optuna with TPESampler.
- **Process**: Managed by `hpo.py`, typically invoked via `run_optimization.py`.
- **Objective**: The `objective` function in `hpo.py` trains a PPO agent for a limited number of steps with sampled hyperparameters and returns a custom score. This score is maximized by Optuna.
- **Sampled Parameters**:
    - Reward component weights (`w_grid`, `w_cap`, `w_deg`, `w_soc`, `w_shape`, `w_night`, `w_arbitrage`, `w_export`, `w_action_mod`, `w_morning`, `w_chain`).
    - Base reward factors (e.g., `soc_limit_penalty_factor`, `preferred_soc_reward_factor`, `peak_penalty_factor`).
    - Arbitrage parameters (e.g., `charge_at_low_price_reward_factor`, price percentiles).
    - SoC target parameters (`preferred_soc_min_base`, `preferred_soc_max_base`, `morning_target_soc`).
    - RL Algorithm (PPO) parameters (`learning_rate`, `n_steps`, `batch_size`, `n_epochs`, `gamma`, `ent_coef`, `gae_lambda`).
- **Output**:
    - A study database (e.g., `home_energy_rl_hpo_v3_reward_rework.db`).
    - In the `src/rl/logs/hpo_results/<study_name>_<timestamp>/` directory:
        - `best_hpo_params.json`: JSON file with the best hyperparameters found and their corresponding score and metrics.
        - `hpo_history.png`: Plot of optimization history.
        - `hpo_param_importance.png`: Plot of hyperparameter importances.
        - `hpo_parallel_coordinate.png`: Parallel coordinate plot.

## Key Simulated Features

-   **Battery Management**: Controls battery charging/discharging within SoC limits, considering degradation.
-   **Cost Optimization**: Aims to minimize costs by leveraging solar, price arbitrage, and peak shaving.
-   **Swedish Electricity Tariffs**: Models energy tax, VAT, grid fees, and capacity charges (with night discounts).
-   **Data-Driven**: Uses historical/predicted data for prices, consumption, and solar.
-   **Data Augmentation**: Optionally augments solar and consumption data during training to improve robustness.
-   **Advanced Reward Shaping**: Includes morning SoC targeting and night-to-peak chain bonus to optimize battery usage.
-   **Memory-Based Strategies**: RecurrentEnergyAgent can remember past decisions and patterns to implement multi-step strategies.

## Agent Types

### RecurrentEnergyAgent
The RecurrentEnergyAgent uses RecurrentPPO with LSTM layers to maintain memory across timesteps. This allows the agent to learn patterns over time and make decisions based on historical data.

Key features:
- Uses LSTM layers to maintain state across timesteps
- Better suited for capturing temporal patterns in price and consumption data
- Can learn to anticipate daily and weekly patterns
- Requires flattened observation space (handled automatically during evaluation)

When evaluating a RecurrentEnergyAgent, the script automatically:
1. Detects the agent type
2. Applies a FlattenObservation wrapper to the environment
3. Properly handles the LSTM state during prediction
4. Ensures metrics and plots work with both agent types

## Reward Function

The reward function is composed of multiple components that balance different objectives:

1. **Grid Cost Minimization**: Reduces electricity costs by considering spot prices, VAT, grid fees, and energy taxes
2. **Capacity Fee Management**: Avoids creating new peak loads that would increase monthly capacity fees
3. **Battery Degradation**: Accounts for battery wear based on charge/discharge cycles
4. **State of Charge (SoC) Optimization**: Maintains optimal battery levels for flexibility
5. **Morning SoC Targeting**: Ensures the battery has sufficient charge in the morning to handle peak loads
6. **Night-to-Peak Chain Bonus**: Rewards charging at night and using that energy during peak price periods
7. **Export Bonus**: Encourages feeding excess solar to the grid when favorable
8. **Night Charging Incentive**: Promotes charging during off-peak night hours
9. **Action Modification Penalty**: Discourages frequent and drastic changes in charging/discharging behavior

Each component can be weighted through configuration parameters to emphasize different aspects of the optimization.

## Command Line Options

### Training Script (`train.py`)
```
python src/rl/train.py [options]
  --recurrent            Use RecurrentPPO agent instead of standard PPO
  --load-model PATH      Path to load a pre-trained model
  --test-only            Only test the model, no training
  --timesteps N          Override number of timesteps to train
  --seed N               Random seed for reproducibility
  --start-date DATE      Start date for training data (format: YYYY-MM-DD)
  --end-date DATE        End date for training data (format: YYYY-MM-DD)
  --sanity-check-only    Only run sanity checks, then exit
  --skip-sanity-check    Skip sanity checks before training
  --sanity-check-steps N Number of steps for sanity checks (default: 100)
  --augment-data         Enable data augmentation for solar and consumption data
```

### Evaluation Script (`evaluate_agent.py`) 
```
python src/rl/evaluate_agent.py [options]
  --render               Render the environment during evaluation
  --model-path PATH      Path to the trained model to evaluate
  --recurrent            Evaluate a recurrent model (uses best_recurrent_model if no path provided)
  --num-episodes N       Number of evaluation episodes to run (default: 1)
  --start-month YYYY-MM  Specific month to evaluate
  --random-start         Use random episode start dates
```

## Usage Examples

### Training a recurrent agent
```bash
python src/rl/train.py --recurrent --start-date 2025-01-01 --end-date 2025-03-31
```

### Running sanity checks only
```bash
python src/rl/train.py --sanity-check-only --sanity-check-steps 20
```

### Evaluating a trained model
```bash
python src/rl/evaluate_agent.py --model-path src/rl/saved_models/best_recurrent_model/best_model.zip --start-month 2025-02
```

# Training and Evaluation

## Training Agents

To train a new agent, use the `train.py` script with the following options:

```bash
# Train a recurrent agent
python src/rl/train.py --recurrent --timesteps 500000

# Train with specific date range
python src/rl/train.py --start-date 2025-01-01 --end-date 2025-01-31

# Run sanity checks only
python src/rl/train.py --sanity-check-only
```

## Evaluating Agents

To evaluate a trained agent, use the `evaluate_agent.py` script:

```bash
# Evaluate a recurrent agent by specifying the model path
python src/rl/evaluate_agent.py --model-path src/rl/saved_models/final_recurrent_model.zip --start-month 2025-02

# Evaluate a recurrent agent using the default best model (simpler approach)
python src/rl/evaluate_agent.py --recurrent --start-month 2025-02

# Run multiple evaluation episodes
python src/rl/evaluate_agent.py --model-path src/rl/saved_models/best_recurrent_model/best_model.zip --start-month 2025-02 --num-episodes 3
```