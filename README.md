# Home Energy AI Optimizer

This project is an AI-driven system for optimizing home energy usage, with a focus on reducing electricity costs and minimizing energy peaks. It integrates with Home Assistant and Node-RED to control batteries, appliances, and renewable energy sources dynamically.

## Features
- Predict hourly energy demand using machine learning models.
- Optimize battery charging and discharging based on energy prices, solar generation, and demand forecasts.
- Control home appliances (e.g., floor heating, water radiators, EV chargers) to reduce energy peaks.
- Minimize electricity costs under the Swedish energy pricing model (power tariff).
- Forecast solar energy production for 4 days starting from today using the forecast.solar API with hourly resolution, customized for multi-orientation panel arrays.
- Visualize and compare predicted vs actual solar production with hourly detail, daily summaries, and heatmap views.

## RecurrentPPO Agent with Enhanced Reward Structure

The project now includes a recurrent policy agent using PPO with LSTM that maintains memory across timesteps. This allows the agent to learn patterns over time, which is ideal for:

1. **Morning SoC Targeting**: The agent can learn to maintain a low SoC in the morning before solar production starts, to maximize self-consumption.

2. **Night-to-Peak Chain Bonus**: The agent receives a bonus for charging at night and then discharging during peak hours, creating a clear signal to charge at night to shave the next day's peak.

3. **Memory of Past Actions**: The recurrent network allows the agent to remember its past actions and their consequences, enabling more consistent behavior over time.

### Usage

To train the recurrent agent:

```bash
python src/rl/train.py --recurrent --timesteps 50000
```

### New Reward Components

- `morning_soc_reward`: Encourages the agent to have a low SoC in the morning before solar production starts.
- `night_peak_chain_bonus`: Rewards the agent for using energy charged during night hours to discharge during peak hours.

These components help the agent learn the behaviors needed for optimal home energy management.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/Michelingumman/home-energy-ai.git
   cd home-energy-ai
  ```
