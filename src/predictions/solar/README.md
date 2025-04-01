# Solar Production Prediction System

This system predicts and visualizes solar energy production for a solar panel installation in Herrängen, Stockholm. It uses the [forecast.solar](https://forecast.solar/) API for predictions and compares these predictions with actual production data from Home Assistant / SolarEdge.

## System Overview

The solar prediction system consists of several components:
- Solar panel configuration management
- API-based production predictions (4-day forecasts) with authenticated access
- Actual production data processing
- Visualization of predicted vs actual production
- Continuous data storage in merged files

### Solar Panel Installation Specifications

<div style="display: flex; align-items: top;">
<div style="flex: 1;">

The system is configured for a specific installation with the following parameters:
- Total Capacity: 20.3 kW
- Panel Count: 50 panels
  - 24 panels facing Southeast
  - 26 panels facing Northwest
- Individual Panel Power: 405W
- Panel Tilt: 27 degrees
- Location: Herrängen, Stockholm

</div>
<div style="flex: 1;">

![image](https://github.com/user-attachments/assets/f11e901e-36b5-4208-8c99-0420377e566f)

</div>
</div>

### API Integration and Panel Grouping

The system makes intelligent use of the forecast.solar API by:
1. **Panel Grouping**: Panels are grouped by orientation
   - Southeast group: 24 panels (9.72 kW total)
   - Northwest group: 26 panels (10.53 kW total)

2. **Multiple API Calls**: The system makes separate API calls for each panel group using authenticated access:
   ```
   /:apikey/estimate/watthours/{lat}/{lon}/27/135/10.53?resolution=60   # Northwest panels
   /:apikey/estimate/watthours/{lat}/{lon}/27/-45/9.72?resolution=60    # Southeast panels
   ```

3. **4-Day Forecasts**: Each API call retrieves predictions for the current day plus 3 days ahead

4. **Data Resolution**: All forecasts use hourly resolution (60-minute intervals)

5. **Data Combination**: Results from both API calls are:
   - Automatically aggregated by timestamp
   - Combined to give total system production
   - Handled for any duplicate timestamps
   - Stored in individual CSV files per day
   - Accumulated in a master merged file

This approach ensures accurate predictions by accounting for different sun exposure patterns for each panel orientation.

## Directory Structure

```
solar/
├── prediction.py        # Main prediction script for API calls
├── plot_solar.py        # Visualization script
├── merge_predictions.py # Script to merge prediction CSVs
├── forecasted_data/     # Contains all prediction data
│   ├── per_day/         # Individual daily prediction files
│   │   └── YYYYMMDD.csv # One file per forecasted day
│   └── merged_predictions.csv # Combined predictions file (all days)
├── actual/              # Actual production data
│   ├── per_day/         # Daily actual data files
│   │   └── YYYYMMDD.csv
│   └── merged_cleaned_actual_data.csv # Combined actual data

```

## Usage

### Generating Predictions

Run the prediction script to generate 4-day forecasts starting from the current day:

```bash
# Generate 4-day forecast
python src/predictions/solar/prediction.py
```

The prediction script will:
1. Create individual CSV files in the `forecasted_data/per_day/` directory for each day in the forecast
2. Automatically append all predictions to `forecasted_data/merged_predictions.csv`
3. Handle any duplicates when adding to the merged file (newer predictions replace older ones)

### API Rate Limiting

The forecast.solar API has rate limits that affect usage:

- **Authenticated Access**: The system uses a personal API key stored in `api.env` for enhanced features
- API key requests are limited based on your subscription tier (typically higher than public API)
- If you encounter HTTP 429 "Too Many Requests" errors, wait before making additional requests (typically 1 hour)
- The system displays your remaining request limit with each API call

### Authentication

The system reads the API key from the `api.env` file in the project root:

```
FORECASTSOLAR=your_api_key_here
```

If the API key is not found, the system will fall back to the public API with more restrictive limits.

### Visualizing Solar Production

The visualization script allows you to plot solar production data with a simplified interface focused on date selection:

```bash
# Visualize a single day
python src/predictions/solar/plot_solar.py 2025-03-15

# Visualize a date range
python src/predictions/solar/plot_solar.py 2025-03-01:2025-03-15
```

By default, plots are displayed interactively in matplotlib windows.

## Visualization Types

The system provides three types of visualizations, all available with a single command:

### 1. Hourly Comparison
- Detailed hourly data showing predicted vs actual production
- Color-coded bars for easy comparison
- Time-of-day labels on the x-axis
- Daily production totals in the title

</div>
<div style="flex: 1;">

![image](https://github.com/user-attachments/assets/c4b842dc-4251-4544-81d6-25d26e9d7e23)



</div>
</div>

### 2. Daily Summary
- Bar chart showing daily total production
- Side-by-side comparison of predicted and actual values
- Daily totals with numerical labels
- Clear visual indication of forecast accuracy

</div>
<div style="flex: 1;">

![image](https://github.com/user-attachments/assets/70c8b1d2-c809-4f12-84d3-ffe75cd2fce1)

</div>
</div>

### 3. Heatmap View
- Shows production patterns by hour and day
- Color intensity indicates production level
- Helps identify peak production hours
- Useful for spotting patterns across multiple days

</div>
<div style="flex: 1;">

![image](https://github.com/user-attachments/assets/bb2d06c1-5b53-404c-9d8f-22461dce74c9)

</div>
</div>

## Data Processing

### Prediction Data
- Retrieved from forecast.solar API
- Covers 4 days (current day plus 3 days ahead)
- Hourly resolution (60-minute intervals) in kilowatt-hours (kWh)
- Stored in CSV files with `timestamp`, `watt_hours`, and `kilowatt_hours` columns
- Continuously accumulated in the merged predictions file

### Actual Data
- Exported from Home Assistant
- Contains `last_changed` timestamp and `state` value in Watts
- Converted to kilowatt-hours for accurate comparison with predictions

The system automatically handles unit conversion between Watts and kilowatt-hours to ensure consistent comparison between predicted and actual data.

## Advanced Options

The visualization script supports additional options for advanced users:

```bash
# Specify custom data directories
python src/predictions/solar/plot_solar.py --date 2025-03-15 --forecast_dir /path/to/forecast --actual_dir /path/to/actual

# Generate only specific plot types
python src/predictions/solar/plot_solar.py --date 2025-03-15 --plot_type hourly
```

Available plot types: `hourly`, `summary`, `heatmap`, or `all` (default).
