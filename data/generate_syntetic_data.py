import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Number of hourly observations (for example, about 5000 hours ~ 208 days)
n_hours = 5000

# Create a date range starting at a specific date
start_date = datetime(2024, 1, 1)
dates = [start_date + timedelta(hours=i) for i in range(n_hours)]

# Generate a synthetic power consumption signal:
# A sinusoidal pattern (to simulate daily cycles) plus some noise and a slight upward trend
time = np.arange(n_hours)
sinusoid = np.sin(2 * np.pi * time / 24)  # daily cycle
trend = time * 0.01  # slight upward trend
noise = np.random.normal(0, 50, n_hours)  # Gaussian noise

values = 1500 + 200 * sinusoid + trend + noise  # base level 1500, scaled sinusoid, trend, noise

# Create the DataFrame with the expected columns
df = pd.DataFrame({
    "entity_id": ["sensor.synthetic_power"] * n_hours,
    "state": values,
    "last_changed": dates
})

# Save the CSV file to data/raw
output_path = "data/raw/synthetic_power.csv"
df.to_csv(output_path, index=False)
print(f"Synthetic dataset saved to {output_path}")
