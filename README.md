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

    home-battery-ai/
    ├── README.md                # Project overview and setup instructions
    ├── LICENSE                  # Licensing information
    ├── requirements.txt         # Python dependencies
    ├── .gitignore               # Ignored files (e.g., .csv, .pkl, .h5, etc.)
    ├── data/                    # Placeholder for sample or synthetic datasets
    │   ├── raw/                 # Raw data exported from Home Assistant
    │   ├── processed/           # Preprocessed data ready for modeling
    │   ├── example_data.csv     # Example dataset (anonymized/synthetic if needed)
    ├── models/                  # Trained models for deployment
    │   ├── demand_predictor.pkl # Trained energy demand prediction model
    │   ├── optimizer_agent.onnx # RL or optimization model for battery control
    ├── notebooks/               # Jupyter notebooks for analysis and experimentation
    │   ├── 01_data_exploration.ipynb  # Data exploration and visualization
    │   ├── 02_model_training.ipynb   # Model training and evaluation
    │   ├── 03_control_simulation.ipynb # Battery/appliance control simulations
    ├── scripts/                 # Core scripts for pipeline and automation
    │   ├── preprocess.py        # Data preprocessing pipeline
    │   ├── train_model.py       # Script to train AI models
    │   ├── predict.py           # Script for real-time predictions
    │   ├── control_agent.py     # Agent for controlling battery and appliances
    ├── config/                  # Configuration files for deployment
    │   ├── home_assistant.yaml  # Example Home Assistant configuration
    │   ├── node_red.json        # Example Node-RED flows
    ├── deployment/              # Deployment-specific files and tools
    │   ├── docker/              # Docker configurations (if applicable)
    │   ├── edge_inference.py    # Inference logic for local deployment
    │   ├── cloud_pipeline.md    # Instructions for cloud-based model training
    ├── results/                 # Store results from simulations and experiments
    │   ├── figures/             # Graphs, charts, and visualizations
    │   ├── logs/                # Logs for debugging or tracking experiments
    │   ├── simulations.csv      # Summary of simulation results

