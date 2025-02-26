# Solar Production Prediction System

This system predicts and visualizes solar energy production for a solar panel installation in Herrängen, Stockholm. It uses the [forecast.solar](https://forecast.solar/) API for predictions and can compare these predictions with actual production data fetchedfrom Home Assistant.

## System Overview

The solar prediction system consists of several components:
- Solar panel configuration management
- API-based production predictions
- Actual production data processing
- Visualization of predicted vs actual production

### Solar Panel Installation Specifications

The system is configured for a specific installation with the following parameters gathered from the user:
- Total Capacity: 20.3 kW
- Panel Count: 50 panels
  - 30 panels facing Southeast
  - 20 panels facing Northwest
- Individual Panel Power: 405W
- Panel Tilt: 27 degrees
- Azimuth: 50 degrees
- Location: Herrängen, Stockholm (59.273..., 17.958...)
- Shading Loss: 1%

## Directory Structure

```
solar/
├── config.json          # Solar system
├── prediction.py        # Main prediction script
├── plot_solar.py        # Visualization script
├── data/                # Predicted data storage
│   └── YYYYMMDD.csv     
├── actual/              # Actual production data
│   └── YYYYMMDD.csv     
└── images/              # Generated plots
    └── YYYYMMDD.png     # Daily comparison plots
```

## File Formats

### Prediction Data (data/YYYYMMDD.csv)
```csv
timestamp,watt_hours,kilowatt_hours
2024-02-24 00:00:00,0.0,0.0
2024-02-24 01:00:00,0.0,0.0
...
```

### Actual Data (actual/YYYYMMDD.csv)
```csv
entity_id,state,last_changed
sensor.solar_production,1234.5,2024-02-24 10:15:00
...
```

## Usage

### Generating Predictions

Run the prediction script to generate predictions for today or a specific date (cannot be historic dates). You can specify the date in either YYYYMMDD or YYYY-MM-DD format, and optionally specify the forecast horizon (1-8 days):

```bash
# Generate predictions for today (only today's data will be stored)
python prediction.py

# Generate predictions for a specific date (YYYYMMDD format)
python prediction.py 20250227        # Only Feb 27, 2025 data will be stored

# Generate predictions for a specific date with custom horizon
python prediction.py 20250227 3      # Only Feb 27, 2025 data will be stored,
                                    # but API will fetch 3 days of data

# Generate predictions for a specific date (YYYY-MM-DD format)
python prediction.py 2025-02-27      # Only Feb 27, 2025 data will be stored
```

The script will:
1. Accept the date in either format (YYYYMMDD or YYYY-MM-DD)
2. Accept an optional horizon window (1-8 days, default is 7)
3. Query the forecast.solar API for predictions over the specified horizon
4. **Filter and save only the data for the requested date**
5. Save the results in the data/ directory with the format YYYYMMDD.csv

> Note: The horizon parameter only affects how far ahead the API looks for predictions. Regardless of the horizon value, only data for the specified date will be saved to the CSV file.

### Visualizing Predictions and Actual Production

Run the plotting script to create visualizations. You can specify the date in either YYYYMMDD or YYYY-MM-DD format:

```bash
# Plot today's predictions
python plot_solar.py

# Plot predictions for a specific date (YYYYMMDD format)
python plot_solar.py 20240224

# Plot predictions for a specific date (YYYY-MM-DD format)
python plot_solar.py 2024-02-24
```

The script will:
1. Load prediction data from the corresponding CSV file (must be generated first using prediction.py)
2. Look for actual production data in the actual/ directory
3. Generate a plot comparing predicted and actual production
4. Save the plot as a PNG file in the images/ directory

Note: Make sure to run prediction.py first to generate the prediction data for the date you want to plot.

### Visualization Features

The generated plots include:
- Hourly predicted production as orange gradient bars
- Hourly actual production as blue bars (when available)
- Value labels for significant production values (>0.1 kWh)
- Daily totals for both predicted and actual production
- Clear date and time labels
- Legend indicating predicted vs actual values

## Data Processing

### Prediction Data
- Retrieved from forecast.solar API
- Hourly resolution
- Values in kilowatt-hours (kWh)
- Stored in CSV format

### Actual Data
- Exported from Home Assistant
- Raw data in watts at variable intervals
- Processed to hourly means
- Converted to kilowatt-hours for comparison
- Original timestamps preserved with timezone handling
