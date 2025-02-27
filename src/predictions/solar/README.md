# Solar Production Prediction System

This system predicts and visualizes solar energy production for a solar panel installation in Herrängen, Stockholm. It uses the [forecast.solar](https://forecast.solar/) API for predictions and can compare these predictions with actual production data from Home Assistant.

## System Overview

The solar prediction system consists of several components:
- Solar panel configuration management
- API-based production predictions
- Actual production data processing
- Visualization of predicted vs actual production

### Solar Panel Installation Specifications

The system is configured for a specific installation with the following parameters:
- Total Capacity: 20.3 kW
- Panel Count: 50 panels
  - 30 panels facing Southeast
  - 20 panels facing Northwest
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

3. **Data Combination**: Results from both API calls are:
   - Automatically aggregated by timestamp
   - Combined to give total system production
   - Handled for any duplicate timestamps
   - Stored in a single CSV file per day

This approach ensures accurate predictions by accounting for different sun exposure patterns for each panel orientation.

## Directory Structure

```
solar/
├── config.json          # Solar system configuration
├── prediction.py        # Main prediction script
├── plot_solar.py        # Visualization script
├── data/                # Predicted data storage
│   └── YYYYMMDD.csv     # Daily prediction files
├── actual/              # Actual production data
│   └── YYYYMMDD.csv     # Daily actual data files
└── images/              # Generated plots
    └── YYYYMMDD/        # Date-based folders
        ├── YYYYMMDD.png # Daily visualization
        ├── last_week_hourly.png
        ├── last_week_summary.png
        └── last_week_heatmap.png
```

## Usage

### Generating Predictions

Run the prediction script to generate predictions for today or a specific date:

```bash
# Generate predictions for today
python prediction.py

# Generate predictions for a specific date
python prediction.py 20240227

# Generate predictions for multiple days
python prediction.py 20240227 3      # Generate for 3 days starting Feb 27, 2024
```

### Visualizing Solar Production

The system provides multiple visualization options:

```bash
# Visualize a single day's production
python plot_solar.py 20240227

# Visualize the last 7 days
python plot_solar.py last_week

# Visualize the last 30 days
python plot_solar.py last_month
```

All visualizations are automatically saved in date-based folders under the `images/` directory, making it easy to locate and review historical data.

## Visualization Types

### Single Day View
- Hourly production data with color-coded bars
- Actual vs predicted comparison when available
- Clear hour and value labels
- Daily total production summary

### Weekly View
When using `last_week`, the system generates:
1. **Detailed Hourly Visualization** (`last_week_hourly.png`)
   - Shows every hour across the week
   - Day separators with alternating background colors
   - Predicted vs actual data comparison
   - Hourly and daily production values

2. **Daily Summary** (`last_week_summary.png`)
   - One bar per day showing total production
   - Daily values and overall weekly total
   - Percentage difference between prediction and actual

3. **Heatmap View** (`last_week_heatmap.png`)
   - Shows production patterns by hour and day
   - Color intensity indicates production level

### Monthly View
When using `last_month`, the system generates:
1. **Monthly Summary** (`last_month_summary.png`)
   - Daily totals across the month
   - Overall monthly production

2. **Monthly Heatmap** (`last_month_heatmap.png`)
   - Shows production patterns across all days of the month
   - Hour-by-day grid with color intensity

## Data Processing

### Prediction Data
- Retrieved from forecast.solar API
- Hourly resolution in kilowatt-hours (kWh)

### Actual Data
- Exported from Home Assistant
- Processed to hourly means
- Converted to kilowatt-hours for comparison

## Error Handling

The system includes robust error handling for:
- Duplicate timestamps in data sources
- Missing or incomplete data
- Date formatting and validation
- Non-numeric data in aggregation operations
