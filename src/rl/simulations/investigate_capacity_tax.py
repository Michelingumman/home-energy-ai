import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import ace_tools

# Load data
price_path = r"C:\_Projects\home-energy-ai\data\processed\SE3prices.csv"
cons_path = r"C:\_Projects\home-energy-ai\data\processed\villamichelin\VillamichelinEnergyData.csv"
solar_path = r"C:\_Projects\home-energy-ai\src\predictions\solar\actual_data\ActualSolarProductionData.csv"

price_df = pd.read_csv(price_path, parse_dates=['HourSE'], index_col='HourSE')
cons_df = pd.read_csv(cons_path, parse_dates=['timestamp'], index_col='timestamp')
solar_df = pd.read_csv(solar_path, parse_dates=['Timestamp'], index_col='Timestamp').tz_convert('Europe/Stockholm')

for df in (price_df, cons_df):
    df.index = df.index.tz_localize(
        'Europe/Stockholm',
        ambiguous='NaT',           # drop the duplicated 02:00→02:59 run-back hour :contentReference[oaicite:0]{index=0}
        nonexistent='shift_forward' # shift missing spring-forward timestamps into existence :contentReference[oaicite:1]{index=1}
    )


# Resample all to hourly on a common timeline
start = max(price_df.index.min(),
            cons_df.index.min(),
            solar_df.index.min())

end   = min(price_df.index.max(),
            cons_df.index.max(),
            solar_df.index.max())

idx = pd.date_range(start, end, freq='h', tz='Europe/Stockholm')

price_hourly = price_df['SE3_price_ore'].reindex(idx).interpolate()
cons_hourly = cons_df['consumption'].reindex(idx).interpolate()
solar_hourly = solar_df['solar_production_kwh'].reindex(idx).interpolate()  # adjust column name if needed

# Compute net load
net_load = cons_hourly - solar_hourly

# Summary metrics
metrics = {
    'Metric': [
        'Price Mean (öre)', 'Price Std (öre)',
        'Consumption Mean (kW)', 'Consumption Max (kW)',
        'Solar Mean (kW)', 'Solar Max (kW)',
        'Net Load Mean (kW)', 'Net Load Max (kW)',
        'Price-Consumption Corr', 'Solar-Price Corr'
    ],
    'Value': [
        price_hourly.mean(), price_hourly.std(),
        cons_hourly.mean(), cons_hourly.max(),
        solar_hourly.mean(), solar_hourly.max(),
        net_load.mean(), net_load.max(),
        price_hourly.corr(cons_hourly), price_hourly.corr(solar_hourly)
    ]
}
metrics_df = pd.DataFrame(metrics)

# Capacity fee simulation
capacity_fee_per_kw = 81.25  # SEK per kW
monthly_avg_peaks = cons_hourly.resample('M').apply(lambda x: x.nlargest(3).mean())
monthly_capacity_fee = monthly_avg_peaks * capacity_fee_per_kw
monthly_capacity_df = monthly_capacity_fee.rename('Capacity Fee (SEK)').to_frame()

# Display summary metrics
display_dataframe_to_user("Summary Metrics", metrics_df)

# Plotting
plt.figure(figsize=(14, 10))

# (1) Time-series over last week
week_idx = idx[-7*24:]
plt.subplot(3, 1, 1)
plt.plot(week_idx, price_hourly.loc[week_idx], label='Price (öre/kWh)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(week_idx, cons_hourly.loc[week_idx], label='Consumption (kW)')
plt.plot(week_idx, solar_hourly.loc[week_idx], label='Solar Prod (kW)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(week_idx, net_load.loc[week_idx], label='Net Load (kW)')
plt.legend()

plt.tight_layout()
plt.show()

# (2) Distribution histograms
plt.figure(figsize=(12, 5))
plt.hist(price_hourly, bins=50, alpha=0.5, label='Price')
plt.hist(cons_hourly, bins=50, alpha=0.5, label='Consumption')
plt.hist(solar_hourly, bins=50, alpha=0.5, label='Solar')
plt.legend()
plt.title("Distributions of Price, Consumption, Solar")
plt.show()

# (3) Monthly capacity fee
plt.figure(figsize=(8, 4))
monthly_capacity_df.plot(kind='bar')
plt.title("Monthly Capacity Fees from Consumption Peaks")
plt.ylabel("Fee (SEK)")
plt.tight_layout()
plt.show()

# Display monthly capacity fee table
display_dataframe_to_user("Monthly Capacity Fees", monthly_capacity_df)
