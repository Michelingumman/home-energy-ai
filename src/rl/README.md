# Home Energy Reinforcement Learning System

This directory contains the core components for a reinforcement learning (RL) system designed to optimize home energy usage. It learns to control a home battery system, considering grid prices, solar production, and household consumption to minimize electricity costs.

## Core Components

-   **`custom_env.py`**: Defines the `HomeEnergyEnv`, which is the simulation environment where the RL agent learns. It simulates the home energy system, including the battery, grid interaction, solar panels, and energy consumption. It's responsible for calculating rewards based on the agent's actions and the current state of the system.
-   **`config.py`**: This is the central configuration file. It holds all key parameters for the RL environment (e.g., battery capacity, simulation length), agent training (e.g., learning rates, number of timesteps), and reward function (e.g., penalties for peak loads, bonuses for arbitrage).
-   **`train.py`**: The main script used to train the RL agent. It initializes the `HomeEnergyEnv`, sets up the PPO (Proximal Policy Optimization) agent from Stable Baselines3, and runs the training loop. It saves trained models and logs training progress (viewable with TensorBoard).
-   **`evaluate_agent.py`**: This script is used to evaluate the performance of a trained RL agent. It loads a saved model, runs it in the `HomeEnergyEnv` for a specified period (e.g., a specific month), and generates plots and metrics to analyze its behavior and effectiveness.
-   **`agent.py`**: Contains the definition for the `ShortTermAgent`, which is a wrapper around the Stable Baselines3 PPO model. (Note: The system was previously envisioned as hierarchical, but current focus is on the short-term agent).
-   **`components.py`**: Defines classes for individual parts of the home energy system, such as the `Battery`.

## Setup

1.  Ensure you have Python 3.10+ installed.
2.  Install the required packages:
    ```bash
    pip install gymnasium stable-baselines3 numpy pandas matplotlib tensorflow
    ```
    (Refer to the main project `requirements.txt` for a more comprehensive list of dependencies if issues arise).

## How to Run

All scripts are run from the root directory of the `home-energy-ai` project.

### 1. Configuration

Before training or evaluation, review and adjust parameters in `src/rl/config.py` to match your desired setup. This includes:
    - Data paths (prices, consumption, solar).
    - Battery specifications.
    - Electricity pricing details (tariffs, taxes).
    - Reward function components and weights.
    - Agent hyperparameters.

### 2. Training the Agent (`train.py`)

The `train.py` script trains the PPO agent.

**Basic Training:**
```bash
python src/rl/train.py
```
This will train the agent using the default settings in `config.py` and save the model periodically and at the end of training in `src/rl/saved_models/`. Logs for TensorBoard will be in `src/rl/logs/`.

**Training with Specific Date Ranges:**
You can restrict the training data to a specific period using `--start-date` and `--end-date`. This is useful if you want the agent to focus on particular seasons or data segments.
```bash
python src/rl/train.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```
Example:
```bash
python src/rl/train.py --start-date 2023-03-01 --end-date 2023-05-31
```

**Training with Data Augmentation:**
To improve generalization, especially with limited data (like solar production), you can enable data augmentation. This will apply slight random variations to solar and consumption data during each training episode.
```bash
python src/rl/train.py --augment-data
```
Example:
```bash
python src/rl/train.py --augment-data --start-date 2023-01-01 --end-date 2023-12-31
```
Data augmentation settings (like the strength of augmentation) can be configured in `config.py`.

**Continuing Training from a Checkpoint:**
If you have a previously saved model checkpoint, you can continue training from it:
```bash
python src/rl/train.py --short_term_model src/rl/saved_models/short_term_checkpoints/your_model_checkpoint.zip
```

**Other `train.py` Options:**
-   `--skip-sanity-check`: Skips pre-training checks of the environment and reward function.
-   `--sanity-check-only`: Runs only the sanity checks and then exits. Useful for debugging the environment.
-   `--sanity-check-steps <N>`: Specifies the number of steps for the sanity check episode (default: 100).

### 3. Evaluating a Trained Agent (`evaluate_agent.py`)

The `evaluate_agent.py` script loads a trained model and runs it in the environment to assess its performance. Data augmentation is **always disabled** during evaluation to test the agent on real, unaltered data.

**Basic Evaluation (Random Month):**
By default, the script selects a random full month from your available data range for evaluation.
```bash
python src/rl/evaluate_agent.py
```
This command assumes your final trained model is saved as `src/rl/saved_models/short_term_agent_final.zip`. If your model has a different name or path, use the `--model_path` argument (though the default model name is hardcoded in `evaluate_agent.py` currently, this might change).

**Evaluating a Specific Month:**
Use the `--start-month` argument to evaluate the agent's performance for a specific month.
```bash
python src/rl/evaluate_agent.py --start-month YYYY-MM
```
Example:
```bash
python src/rl/evaluate_agent.py --start-month 2023-07
```

**Evaluating with Random Start Dates (Not a Specific Month):**
If you prefer the evaluation episode to start at a completely random date and time within the available data (rather than being aligned to a specific month), use the `--random-start` flag. The episode length will be determined by `simulation_days_eval` in `config.py`.
```bash
python src/rl/evaluate_agent.py --random-start
```

**Evaluation Output:**
-   Performance plots (e.g., SoC, prices, actions, grid power, reward components) will be saved in a timestamped subdirectory within `src/rl/simulations/results/`.
-   Metrics will be printed to the console.
-   Raw episode data may also be saved for further analysis (check script details).

## Key Features Simulated

-   **Battery Management**: Controls a battery (e.g., 22 kWh capacity) within specified SoC limits (e.g., 20%-80%), considering degradation costs.
-   **Cost Optimization**: Aims to minimize electricity costs by:
    -   Leveraging solar energy.
    -   Charging the battery during low grid prices.
    -   Discharging the battery during high grid prices or to reduce peak loads.
    -   Considering complex Swedish electricity tariffs (energy tax, VAT, grid fees, capacity charges with night discounts).
-   **Peak Shaving**: Penalizes high grid import peaks to avoid capacity fees.
-   **Data-Driven**: Uses historical or predicted data for prices, household consumption, and solar production.

## Customization

The system is highly customizable through `src/rl/config.py`. You can adjust:
-   Environment parameters (simulation length, data paths, fixed baseloads).
-   Battery characteristics (capacity, charge/discharge rates, efficiency, degradation cost).
-   Electricity pricing model details.
-   The entire reward function structure and the weights of its components.
-   PPO agent hyperparameters (learning rate, batch size, etc.).
-   Data augmentation settings.

By modifying `config.py`, you can tailor the simulation and agent training to specific scenarios, hardware, or optimization goals.

## Project Structure

```
src/rl/
├── __pycache__/
├── agent.py
├── components.py
├── config.py
├── custom_env.py
├── evaluate_agent.py
├── hyperparameter_optimization.py
├── logs/
├── README.md
├── run_optimization.py
├── safety_buffer.py
├── saved_models/
├── simulations/
│   └── results/
└── train.py
``` 