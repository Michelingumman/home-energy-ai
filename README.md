<div align="center">
  <h1>🏠 Home Energy AI Optimizer 💡</h1>
  <p><em>An AI-driven system for optimizing home energy usage, reducing costs, and minimizing energy peaks</em></p>

  <p>
    <img src="https://img.shields.io/badge/Energy-Optimization-FDB813?style=flat-square&logo=energy" alt="Energy"/>
    <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python" alt="Python"/>
    <img src="https://img.shields.io/badge/Machine%20Learning-Forecasting-01D277?style=flat-square&logo=pytorch" alt="Machine Learning"/>
    <img src="https://img.shields.io/badge/Smart%20Home-Automation-4FADA7?style=flat-square&logo=homeassistant" alt="Smart Home"/>
    <img src="https://img.shields.io/badge/Battery-Management-00B388?style=flat-square&logo=power" alt="Battery"/>
  </p>
</div>

<!-- <div align="center">
  <table>
    <tr>
      <td width="70%">
        <img src="docs/images/dashboard.png" width="100%" alt="Energy Dashboard"/>
        <p align="center"><strong>Energy Dashboard</strong> - Real-time monitoring and optimization</p>
      </td>
      <td width="50%">
        <img src="docs/images/prediction.png" width="100%" alt="Energy Prediction"/>
        <p align="center"><strong>AI Predictions</strong> - Smart energy forecasting</p>
      </td>
    </tr>
  </table>
</div>

<p align="center">
  <img src="docs/images/system-overview.png" width="90%" alt="System Overview"/>
  <br>
  <em>Home Energy AI System Architecture and Data Flow</em>
</p> -->

---

## ✨ Features

- **🔮 Predictive Analytics**: Uses machine learning models to forecast hourly energy demand
- **🔋 Smart Battery Management**: Optimizes battery charging/discharging based on energy prices, solar generation, and demand forecasts
- **🏡 Appliance Control**: Intelligently manages floor heating, water radiators, EV chargers, and other appliances to reduce energy peaks
- **💰 Cost Optimization**: Minimizes electricity costs under the Swedish energy pricing model (power tariff)
- **☀️ Renewable Integration**: Maximizes usage of solar and other renewable energy sources
- **📊 Real-time Monitoring**: Visualizes energy usage, production, and optimization in real-time dashboards
- **🤖 Automated Decision Making**: Uses reinforcement learning to make optimal energy decisions automatically

## 🛠️ System Requirements

<div align="center">
  <table>
    <tr>
      <td align="center"><b>Component</b></td>
      <td align="center"><b>Purpose</b></td>
    </tr>
    <tr>
      <td>Home Assistant Server</td>
      <td>Core automation platform and data hub</td>
    </tr>
    <tr>
      <td>Node-RED Instance</td>
      <td>Flow-based programming for device control</td>
    </tr>
    <tr>
      <td>Python Environment</td>
      <td>For running machine learning models and optimization algorithms</td>
    </tr>
    <tr>
      <td>Smart Meter Integration</td>
      <td>For real-time energy usage data</td>
    </tr>
    <tr>
      <td>Smart Plugs/Switches</td>
      <td>For controlling appliances</td>
    </tr>
    <tr>
      <td>Battery System (Optional)</td>
      <td>For energy storage optimization</td>
    </tr>
    <tr>
      <td>Solar PV System (Optional)</td>
      <td>For renewable energy generation</td>
    </tr>
  </table>
</div>

## 📚 Software Dependencies

- **Python 3.8+** with the following libraries:
  - `TensorFlow` or `PyTorch` - For machine learning models
  - `pandas`, `numpy` - For data processing
  - `scikit-learn` - For feature engineering and model evaluation
  - `requests` - For API interactions
  - `paho-mqtt` - For MQTT communication
  - `aiohttp` - For async HTTP requests
  - `plotly` - For visualization (if enabled)

- **Home Assistant** with integrations:
  - Energy monitoring
  - MQTT
  - RESTful sensors
  - History stats

- **Node-RED** with nodes:
  - node-red-contrib-home-assistant-websocket
  - node-red-contrib-mqtt
  - node-red-contrib-schedex

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Michelingumman/home-energy-ai.git
   cd home-energy-ai
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the system**:
   - Copy `config.example.yaml` to `config.yaml`
   - Update with your specific settings:
     ```yaml
     home_assistant:
       url: "http://your-home-assistant:8123"
       token: "your_long_lived_access_token"
     
     mqtt:
       broker: "mqtt-broker-address"
       port: 1883
       username: "your_mqtt_username"
       password: "your_mqtt_password"
     
     # Additional configuration options...
     ```

4. **Train the prediction model** (optional):
   ```bash
   python src/models/lstm/train.py
   ```

5. **Start the main application**:
   ```bash
   python src/main.py
   ```

## 📖 Usage

### 🧠 Energy Prediction

The system uses LSTM neural networks to predict energy consumption patterns:

- **Historical Analysis**: Uses past energy usage data to identify patterns
- **Weather Integration**: Considers weather forecasts for more accurate predictions
- **Continuous Learning**: Model improves over time as more data is collected

### ⚡ Battery Optimization

For homes with battery systems:

- **Peak Shaving**: Discharge during demand peaks to reduce grid power consumption
- **Price Arbitrage**: Charge when electricity is cheap, discharge when expensive
- **Solar Integration**: Prioritize storing excess solar energy for later use

### 🔌 Appliance Control

Smart management of energy-intensive devices:

1. **Prioritization**: Define which appliances are critical vs. flexible
2. **Time Shifting**: Run non-critical appliances during low-demand or low-price periods
3. **Dynamic Control**: Real-time adjustments based on current energy situation

## ⚙️ Project Structure

```
home-energy-ai/
├── README.md                   # Project overview
├── requirements.txt            # Python dependencies
├── config.yaml                 # Configuration file
│
├── src/
│   ├── main.py                 # Application entry point
│   ├── data/                   # Data processing
│   ├── models/                 # ML models (LSTM, RL)
│   ├── controllers/            # Control logic
│   ├── home_assistant/         # HA integration
│   └── utils/                  # Shared utilities
│
├── docs/                       # Documentation
├── data/                       # Data storage
├── tests/                      # Unit/integration tests
└── simulations/                # Simulation environments
```

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome as branches! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- Thanks to the Home Assistant and Node-RED communities
- Inspired by various smart energy management projects

---

<p align="center">
  <em>Made with ❤️ for a more energy-efficient future</em>
</p>


