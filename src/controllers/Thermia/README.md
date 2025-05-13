# Thermia Heat Pump Data Extraction & Power Estimation  
*(ExtractHistoricData.py)*

This script connects to the Thermia Online API, fetches historical data for a specified Thermia Atlas 12 heat pump, processes it, estimates thermal output and electrical power input, and writes the results to a CSV. You can optionally plot the estimated power input over time.

---

## Features

- Fetch multiple data registers from the Thermia Atlas 12 via the API  
- Clean raw temperature readings by removing outliers  
- **Independent 15 min resampling** of each series:  
  - `power_status` (from compressor runtime) → **max** in each 15 min bin  
  - All temperatures → **mean** in each 15 min bin  
- Compute **thermal output**  
  $$
    \dot Q_{\rm heat}
    = \dot m \,\times\, c_p \,\times\, \Delta T_{\rm clamped}
  $$  
- Clamp  
  $$
    \Delta T_{\rm clamped} = \min\bigl(\max(\Delta T,\,2\text{ K}),\,10\text{ K}\bigr)
  $$  
- Estimate **electrical input**  
  $$
    P_{\rm in}
    = \frac{\dot Q_{\rm heat}}{\mathrm{COP}}
  $$  
  – with fixed $\mathrm{COP}=4.75$ and clamp $P_{\rm in}\le4.7\text{ kW}$  
- Save the final 15 min DataFrame (raw + calculated columns) to CSV  
- Optional matplotlib plot of `power_input_kw` over time  

## Examples:

-   Fetch data for the last 7 days and don\'t plot:
    ```bash
    python ExtractHistoricData.py --days 7
    ```
    (or simply `python ExtractHistoricData.py` as 7 is the default).

    The .csv file will be generated placed in:
    
    `data/processed/villamichelin/heat_pump_power_15min_{date-range}.csv`
    

-   Fetch data for the last 30 days and display a plot:
    ```bash
    python ExtractHistoricData.py --days 30 --plot
    ```

---