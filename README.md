# Home Energy AI Optimizer

This project is an AI-driven system for optimizing home energy usage, with a focus on reducing electricity costs and minimizing energy peaks. It integrates with Home Assistant and Node-RED to control batteries, appliances, and renewable energy sources dynamically.

## Features
- Predict hourly energy demand using machine learning models.
- Optimize battery charging and discharging based on energy prices, solar generation, and demand forecasts.
- Control home appliances (e.g., floor heating, water radiators, EV chargers) to reduce energy peaks.
- Minimize electricity costs under the Swedish energy pricing model (power tariff).
- Forecast solar energy production for 4 days starting from today using the forecast.solar API with hourly resolution, customized for multi-orientation panel arrays.
- Visualize and compare predicted vs actual solar production with hourly detail, daily summaries, and heatmap views.


## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/Michelingumman/home-energy-ai.git
   cd home-energy-ai
  ```
