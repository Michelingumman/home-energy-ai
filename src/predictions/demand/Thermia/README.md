# Thermia Heat Pump Utilities

This directory contains Python scripts for interacting with a Thermia heat pump via the Thermia Online API. These utilities allow for data extraction, power estimation, data updating, and direct control of the heat pump.

---

## 1. `ExtractHistoricData.py`: Data Extraction & Power Estimation

This script connects to the Thermia Online API to fetch historical operational data for a Thermia Atlas 12 heat pump. It processes this data to estimate thermal output and electrical power input, then saves the results to a CSV file. An optional plot of the estimated power input can be generated.

### Key Features

-   **Data Retrieval**: Fetches multiple data registers (e.g., temperatures, compressor runtime) from the Thermia API.
-   **Data Cleaning**: Cleans raw temperature readings by removing outliers.
-   **Resampling**: Performs independent 15-minute resampling of each data series:
    -   `power_status` (derived from compressor runtime): Uses the **maximum** value in each 15-minute interval.
    -   Temperature readings: Uses the **mean** value in each 15-minute interval.
-   **Thermal Output Calculation**: Computes thermal output ($\dot Q_{\rm heat}$) using:
    \[ \dot Q_{\rm heat} = \dot m \times c_p \times \Delta T_{\rm clamped} \]
    Where $\Delta T_{\rm clamped}$ is the supply/return temperature difference, clamped between 2K and 10K:
    \[ \Delta T_{\rm clamped} = \min\bigl(\max(\Delta T,\,2\text{ K}),\,10\text{ K}\bigr) \]
-   **Electrical Input Estimation**: Estimates electrical power input ($P_{\rm in}$) using a fixed Coefficient of Performance (COP):
    \[ P_{\rm in} = \frac{\dot Q_{\rm heat}}{\mathrm{COP}} \]
    (Utilizes a fixed $\mathrm{COP}=4.75$ and clamps $P_{\rm in} \le 4.7\text{ kW}$ for the Thermia Atlas 12 model).
-   **CSV Output**: Saves the processed 15-minute data (including raw and calculated columns) to a CSV file.
-   **Plotting**: Optionally generates a Matplotlib plot of `power_input_kw` over time.

### Usage Examples

-   **Fetch data for the last 7 days (default) without plotting:**
    ```bash
    python ExtractHistoricData.py
    ```
    *(Alternatively: `python ExtractHistoricData.py --days 7`)*

-   **Fetch data for the last 30 days and display a plot:**
    ```bash
    python ExtractHistoricData.py --days 30 --plot
    ```

### Output File

The generated CSV file will be located at:
`data/processed/villamichelin/Thermia/heat_pump_power_15min_{date-range}.csv`

---

## 2. `UpdateHeatPumpData.py`: Automated Data Updates

This script automates the process of keeping the historical heat pump power consumption data up-to-date. It checks the latest timestamp in the main data file (`HeatPumpPower.csv`) and then utilizes `ExtractHistoricData.py` to fetch any new data from that point forward.

### Key Features

-   **Incremental Updates**: Identifies the last recorded data point in `data/processed/villamichelin/Thermia/HeatPumpPower.csv`.
-   **Efficient Fetching**: Calculates the precise period required for fetching only new data.
-   **Data Integration**: Calls `ExtractHistoricData.py` to retrieve new heat pump operational data.
-   **Data Management**: Appends new data to `HeatPumpPower.csv`, removes duplicate entries, and sorts the data by timestamp.
-   **Scheduled Use**: Designed for periodic execution (e.g., daily cron job) to maintain a current dataset.

### Usage

To update the heat pump data, navigate to the project's root directory and run:

```bash
python src/predictions/demand/Thermia/UpdateHeatPumpData.py
```

### Dependencies

-   Relies on `ExtractHistoricData.py` (located in the same directory) for the core data fetching and processing.
-   Requires the `pandas` library for data manipulation.
-   Environment variables for `ExtractHistoricData.py` (e.g., `THERMIA_USERNAME`, `THERMIA_PASSWORD`) must be configured in the `api.env` file at the project root.

---

## 3. `ThermiaControl.py`: Direct Heat Pump Control

This script provides command-line tools to query and control certain parameters of the Thermia heat pump.

### Fetching Information

To retrieve all available information and current settings from the heat pump:

```bash
python ThermiaControl.py --info
```

### Modifying Settings

To change a specific setting, use the `--set` and `--value` arguments.

**Example:**

```bash
python ThermiaControl.py --set <setting_name> --value <new_value>
```

**Controllable Settings:**

-   `temperature`: Sets the desired room temperature (integer value).
-   `operation_mode`: Toggles the heat pump operation mode.
    -   `--value ON`
    -   `--value OFF`
-   `hot_water_switch_state`: Controls the hot water production state (integer value, device-specific).
-   `set_hot_water_boost_switch_state`: Activates or deactivates hot water boost (integer value, device-specific).

*Note: Some parameters may also be controllable via Home Assistant, though direct temperature control might be exclusive to this script or the Thermia API.*
