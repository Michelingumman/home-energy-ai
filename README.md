# Home Energy AI Optimizer

This project is an AI-driven system for optimizing home energy usage, with a focus on reducing electricity costs and minimizing energy peaks. It integrates with Home Assistant and Node-RED to control batteries, appliances, and renewable energy sources dynamically.

## Features
- Predict hourly energy demand using machine learning models.
- Optimize battery charging and discharging based on energy prices, solar generation, and demand forecasts.
- Control home appliances (e.g., floor heating, water radiators, EV chargers) to reduce energy peaks.
- Minimize electricity costs under the Swedish energy pricing model.

## Repository Structure
- `config/`: Configuration files for integrations (Home Assistant, Node-RED).
- `data/`: Raw and processed datasets for model training.
- `deployment/`: Deployment scripts and resources (e.g., Docker, cloud instructions).
- `models/`: Trained models for prediction and optimization.
- `notebooks/`: Jupyter notebooks for analysis and development.
- `results/`: Simulation and experiment outputs.
- `scripts/`: Core scripts for preprocessing, training, and control logic.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/Michelingumman/home-energy-ai.git
   cd home-energy-ai
  ```

AI-Driven-Home-Energy-Management-System/
├── README.md                   # Project overview, quickstart, etc.
├── LICENSE                     # License file
├── .gitignore                  # Files/folders to ignore in git
├── requirements.txt            # Python dependencies
├── setup.py                    # Optional packaging script
│
├── docs/                       # Documentation
│   ├── architecture.md
│   ├── installation.md
│   ├── user_guide.md
│   └── design_notes.md
│
├── data/
│   ├── raw/                    # Raw datasets
│   │   └── jena_climate_2009_2016.csv
│   └── processed/              # Processed data for experiments
│
├── src/
│   ├── main.py                 # Application entry point
│   │
│   ├── data/                   # Data-related scripts
│   │   └── preprocess.py       # Data cleaning/preprocessing code
│   │
│   ├── models/                 # Machine learning models
│   │   ├── __init__.py         # (Empty file to mark as a package)
│   │   ├── lstm/               # LSTM forecasting model
│   │   │   ├── __init__.py     # (Empty file to mark as a package)
│   │   │   ├── lstm_config.json# Configuration parameters for the LSTM model
│   │   │   ├── utils.py        # Utility functions for data loading & windowing
│   │   │   ├── model.py        # Model definition
│   │   │   ├── train.py        # Script to train the LSTM model
│   │   │   └── inference.py    # Script to run inference with the trained model
│   │   │
│   │   └── rl/                 # RL agent code (to be implemented)
│   │       ├── train.py
│   │       ├── agent.py
│   │       ├── env.py
│   │       └── rl_config.json
│   │
│   ├── controllers/            # Control logic for battery scheduling
│   │   ├── long_term.py
│   │   └── short_term.py
│   │
│   ├── home_assistant/         # Integration with Home Assistant
│   │   ├── mqtt_client.py
│   │   └── integration.py
│   │
│   └── utils/                  # Shared utilities (logging, config, etc.)
│       ├── logger.py
│       ├── config.py
│       └── helpers.py
│
├── tests/                      # Unit and integration tests
│   ├── test_preprocess.py
│   ├── test_lstm.py
│   ├── test_rl.py
│   └── test_controllers.py
│
└── simulations/                # Simulation experiments & environments
    ├── custom_env.py
    ├── run_simulation.py
    └── results/


