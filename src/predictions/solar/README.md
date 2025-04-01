# Solar Production Prediction System

This system predicts and visualizes solar energy production for a solar panel installation in Herrängen, Stockholm. It uses the [forecast.solar](https://forecast.solar/) API for predictions and compares these predictions with actual production data from Home Assistant / SolarEdge.

## System Overview

The solar prediction system consists of several components:
- Solar panel configuration management
- API-based production predictions (5-day forecasts)
- Actual production data processing
- Visualization of predicted vs actual production
- Continuous data storage in merged files

### Solar Panel Installation Specifications
![image](https://github.com/user-attachments/assets/f11e901e-36b5-4208-8c99-0420377e566f)


The system is configured for a specific installation with the following parameters:
- Total Capacity: 20.3 kW
- Panel Count: 50 panels
  - 24 panels facing Southeast
  - 26 panels facing Northwest
- Individual Panel Power: 405W
- Panel Tilt: 27 degrees
- Location: Herrängen, Stockholm

### API Integration and Panel Grouping

The system makes intelligent use of the forecast.solar API by:
1. **Panel Grouping**: Panels are grouped by orientation
   - Southeast group: 30 panels (12.15 kW total)
   - Northwest group: 20 panels (8.1 kW total)

2. **Multiple API Calls**: The system makes separate API calls for each panel group:
   ```
   /estimate/watthours/{lat}/{lon}/27/135/8.1    # Northwest panels
   /estimate/watthours/{lat}/{lon}/27/-45/12.15  # Southeast panels
   ```

3. **5-Day Forecasts**: Each API call retrieves predictions for the current day plus 4 days ahead

4. **Data Combination**: Results from both API calls are:
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
├── data/                # Predicted data storage (individual days)
│   ├── YYYYMMDD.csv     # Daily prediction files
│   └── ...
├── forecasted_data/     # Contains the merged prediction data
│   └── merged_predictions.csv # Combined predictions file (all days)
├── actual/              # Actual production data
│   ├── per_day/         # Daily actual data files
│   │   └── YYYYMMDD.csv
│   └── merged_cleaned_actual_data.csv # Combined actual data
└── plots/               # Generated plots when saving is enabled
```

## Usage

### Generating Predictions

Run the prediction script to generate 5-day forecasts starting from today or a specific date:

```bash
# Generate 5-day forecast starting from today
python src/predictions/solar/prediction.py

# Generate 5-day forecast starting from a specific date
python src/predictions/solar/prediction.py 2025-03-15
```

The prediction script will:
1. Create individual CSV files in the `data` directory for each day in the forecast
2. Automatically append all predictions to `forecasted_data/merged_predictions.csv`
3. Handle any duplicates when adding to the merged file (newer predictions replace older ones)

### API Rate Limiting

The forecast.solar API has rate limits that may affect frequent usage:

- The free public API tier is limited to approximately 12 requests per hour
- If you encounter HTTP 429 "Too Many Requests" errors, wait before making additional requests
- For more intensive usage, consider signing up for a [paid plan](https://doc.forecast.solar/api:estimate) with higher rate limits

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

### 2. Daily Summary
- Bar chart showing daily total production
- Side-by-side comparison of predicted and actual values
- Daily totals with numerical labels
- Clear visual indication of forecast accuracy

### 3. Heatmap View
- Shows production patterns by hour and day
- Color intensity indicates production level
- Helps identify peak production hours
- Useful for spotting patterns across multiple days

## Data Processing

### Prediction Data
- Retrieved from forecast.solar API
- Covers 5 days (current day plus 4 days ahead)
- Hourly resolution in kilowatt-hours (kWh)
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
