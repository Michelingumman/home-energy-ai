import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

class DataVisualizer:
    def __init__(self, data_paths: Dict[str, str]):
        """
        Initialize the visualizer with paths to data files.
        """
        self.data_paths = data_paths
        self.data = {}
        self.load_data()
        
        # Set visualization style
        plt.style.use('fivethirtyeight')
        self.colors = sns.color_palette()
    
    def load_data(self):
        """Loads all datasets specified in data_paths."""
        for key, path in self.data_paths.items():
            try:
                self.data[key] = pd.read_csv(path, index_col=0, parse_dates=True)
                print(f"Successfully loaded {key} data from {path}")
            except Exception as e:
                print(f"Error loading {key} data: {e}")
    
    def plot_price_vs_renewable(self, start_date: str, end_date: str):
        """Plots SE3 electricity prices vs. renewable energy percentage."""
        print(f"Running plot_price_vs_renewable from {start_date} to {end_date}")
        if 'prices' not in self.data or 'grid' not in self.data:
            print("Missing required data sources.")
            return
        
        fig, ax1 = plt.subplots(figsize=(15, 6))
        
        prices = self.data['prices'].loc[start_date:end_date, 'SE3_price_ore']
        ax1.plot(prices.index, prices.values, color=self.colors[0], label='Price')
        ax1.set_ylabel('Price (öre/kWh)', color=self.colors[0])
        
        ax2 = ax1.twinx()
        renewable = self.data['grid'].loc[start_date:end_date, 'renewable_percentage']
        ax2.plot(renewable.index, renewable.values, color=self.colors[1], label='Renewable %')
        ax2.set_ylabel('Renewable %', color=self.colors[1])
        
        plt.title(f'Price vs Renewable Percentage ({start_date} to {end_date})')
        plt.show()
    
    def plot_holiday_price_impact(self, year: int):
        """Plots electricity price distribution on holidays vs non-holidays for one year."""
        print(f"Running plot_holiday_price_impact for year {year}")
        if 'prices' not in self.data or 'holidays' not in self.data:
            print("Missing required data sources.")
            return
        
        prices = self.data['prices']
        holidays = self.data['holidays']
        yearly_prices = prices[prices.index.year == year]
        yearly_holidays = holidays[holidays.index.year == year]
        
        holiday_prices = yearly_prices[yearly_prices.index.isin(yearly_holidays.index)]
        non_holiday_prices = yearly_prices[~yearly_prices.index.isin(yearly_holidays.index)]
        
        plt.figure(figsize=(12, 6))
        plt.boxplot([holiday_prices['SE3_price_ore'], non_holiday_prices['SE3_price_ore']],
                    labels=['Holidays', 'Non-Holidays'])
        plt.title(f'Price Distribution: Holidays vs Non-Holidays ({year})')
        plt.ylabel('Price (öre/kWh)')
        plt.show()
    
    def compare_holiday_price_impacts(self, *years: int):
        """
        Compares holiday price impacts across multiple years by plotting subplots.
        Each subplot shows a boxplot for one year.
        """
        print(f"Running compare_holiday_price_impacts for years: {', '.join(map(str, years))}")
        if 'prices' not in self.data or 'holidays' not in self.data:
            print("Missing required data sources.")
            return
        
        prices = self.data['prices']
        holidays = self.data['holidays']
        num_years = len(years)
        fig, axes = plt.subplots(1, num_years, figsize=(6*num_years, 6), squeeze=False)
        for i, year in enumerate(years):
            ax = axes[0][i]
            yearly_prices = prices[prices.index.year == year]
            yearly_holidays = holidays[holidays.index.year == year]
            holiday_prices = yearly_prices[yearly_prices.index.isin(yearly_holidays.index)]
            non_holiday_prices = yearly_prices[~yearly_prices.index.isin(yearly_holidays.index)]
            ax.boxplot([holiday_prices['SE3_price_ore'], non_holiday_prices['SE3_price_ore']],
                       labels=['Holidays', 'Non-Holidays'])
            ax.set_title(f'Year {year}')
            ax.set_ylabel('Price (öre/kWh)')
        plt.suptitle('Holiday Price Impacts Comparison')
        plt.show()
    
    def plot_supply_mix(self, date: str):
        """
        Plots the energy supply mix for a specific month.
        Since grid data is monthly aggregated, a pie chart is used.
        """
        print(f"Running plot_supply_mix for date {date}")
        if 'grid' not in self.data:
            print("Missing grid data.")
            return
        
        supply_columns = ['hydro', 'wind', 'nuclear', 'solar', 'thermal_total']
        try:
            monthly_data = self.data['grid'].loc[date]
        except KeyError:
            print(f"No grid data available for {date}.")
            return

        values = monthly_data[supply_columns]
        plt.figure(figsize=(8, 8))
        plt.pie(values, labels=supply_columns, autopct='%1.1f%%', 
                colors=self.colors[:len(supply_columns)])
        plt.title(f'Energy Supply Mix for {date}')
        plt.show()
    
    def plot_monthly_price_comparison(self, year: int, month: int):
        """
        Compares daily average prices for the specified month in one year against the previous year.
        """
        print(f"Running plot_monthly_price_comparison for {year}-{month:02d}")
        if 'prices' not in self.data:
            print("Missing price data.")
            return
        
        prices = self.data['prices']
        current_month = prices[(prices.index.year == year) & (prices.index.month == month)]
        prev_month = prices[(prices.index.year == year - 1) & (prices.index.month == month)]
        
        current_daily = current_month.groupby(current_month.index.day)['SE3_price_ore'].mean()
        prev_daily = prev_month.groupby(prev_month.index.day)['SE3_price_ore'].mean()
        
        plt.figure(figsize=(15, 6))
        plt.plot(current_daily.index, current_daily.values, label=f'{year}', marker='o')
        plt.plot(prev_daily.index, prev_daily.values, label=f'{year - 1}', marker='o', linestyle='--', alpha=0.7)
        
        plt.title(f'Daily Average Prices - {pd.Timestamp(year, month, 1).strftime("%B %Y")}')
        plt.xlabel('Day of Month')
        plt.ylabel('Average Price (öre/kWh)')
        plt.legend()
        plt.show()
    
    def compare_monthly_price_across_years(self, month: int, *years: int):
        """
        Compares daily average prices for a given month across multiple years.
        Each year's daily averages are plotted on the same chart.
        """
        print(f"Running compare_monthly_price_across_years for month {month} and years: {', '.join(map(str, years))}")
        if 'prices' not in self.data:
            print("Missing price data.")
            return
        
        prices = self.data['prices']
        plt.figure(figsize=(15, 6))
        for year in years:
            monthly_prices = prices[(prices.index.year == year) & (prices.index.month == month)]
            if monthly_prices.empty:
                print(f"No price data for year {year} and month {month}")
                continue
            daily_avg = monthly_prices.groupby(monthly_prices.index.day)['SE3_price_ore'].mean()
            plt.plot(daily_avg.index, daily_avg.values, marker='o', label=str(year))
        plt.title(f"Daily Average Prices for Month {month} Across Years")
        plt.xlabel("Day of Month")
        plt.ylabel("Average Price (öre/kWh)")
        plt.legend()
        plt.show()
    
    def plot_yearly_grid_usage(self, years: list = None):
        """
        Aggregates grid usage by year and plots key energy sources.
        If a list of years is provided, only those years are plotted.
        """
        print("Running plot_yearly_grid_usage")
        if 'grid' not in self.data:
            print("Missing grid data.")
            return
        
        grid = self.data['grid'].copy()
        grid['year'] = grid.index.year
        yearly = grid.groupby('year').sum()
        if years is not None:
            yearly = yearly.loc[yearly.index.isin(years)]
        
        energy_sources = ['total_supply', 'hydro', 'wind', 'nuclear', 'solar', 'thermal_total']
        plt.figure(figsize=(15, 8))
        for source in energy_sources:
            if source in yearly.columns:
                plt.plot(yearly.index, yearly[source], marker='o', label=source)
            else:
                print(f"Column '{source}' not found in grid data.")
        plt.xlabel("Year")
        plt.ylabel("Aggregated Energy Supply")
        plt.title("Yearly Aggregated Grid Usage")
        plt.legend()
        plt.show()
    
    def plot_yearly_renewable_trend(self, years: list = None):
        """
        Plots the trend of average renewable percentage for each year.
        Optionally, only the specified years are included.
        """
        print("Running plot_yearly_renewable_trend")
        if 'grid' not in self.data:
            print("Missing grid data.")
            return
        
        grid = self.data['grid'].copy()
        grid['year'] = grid.index.year
        yearly_renewable = grid.groupby('year')['renewable_percentage'].mean()
        if years is not None:
            yearly_renewable = yearly_renewable[yearly_renewable.index.isin(years)]
        plt.figure(figsize=(12, 6))
        plt.plot(yearly_renewable.index, yearly_renewable.values, marker='o', color=self.colors[2])
        plt.xlabel("Year")
        plt.ylabel("Average Renewable Percentage")
        plt.title("Yearly Trend of Renewable Percentage")
        plt.show()
    
    def compare_supply_mix_between_years(self, *years: int):
        """
        Compares the average supply mix across multiple years using grid data.
        A grouped bar chart shows the average values of selected energy sources.
        """
        print(f"Running compare_supply_mix_between_years for years: {', '.join(map(str, years))}")
        if 'grid' not in self.data:
            print("Missing grid data.")
            return
        
        grid = self.data['grid'].copy()
        grid['year'] = grid.index.year
        supply_columns = ['hydro', 'wind', 'nuclear', 'solar', 'thermal_total']
        averages = {}
        for year in years:
            if year in grid['year'].unique():
                averages[year] = grid[grid['year'] == year][supply_columns].mean()
            else:
                print(f"No grid data for year {year}")
        if not averages:
            print("No valid years provided for comparison.")
            return
        
        df_averages = pd.DataFrame(averages).T  # rows: years, columns: supply sources
        x = np.arange(len(supply_columns))
        total_years = len(df_averages)
        width = 0.8 / total_years
        
        plt.figure(figsize=(10, 6))
        for idx, year in enumerate(df_averages.index):
            plt.bar(x + idx*width - (total_years - 1)*width/2,
                    df_averages.loc[year].values,
                    width, label=str(year))
        plt.xticks(x, supply_columns)
        plt.xlabel("Energy Source")
        plt.ylabel("Average Supply")
        plt.title("Comparison of Average Supply Mix Across Years")
        plt.legend()
        plt.show()

def main():
    # Define file paths
    data_paths = {
        'prices': 'C:/_Projects/home-energy-ai/data/processed/SE3prices.csv',
        'grid': 'C:/_Projects/home-energy-ai/data/processed/SwedenGrid.csv',
        'holidays': 'C:/_Projects/home-energy-ai/data/processed/holidays.csv'
    }
    
    # Initialize and use the visualizer
    visualizer = DataVisualizer(data_paths)
    
    print("Plotting Monthly Price Comparison (single-year vs previous year)...")
    visualizer.plot_monthly_price_comparison(2023, 12)
    
    print("Plotting Price vs Renewable...")
    visualizer.plot_price_vs_renewable('2023-01-01', '2023-01-07')
    
    print("Plotting Supply Mix...")
    visualizer.plot_supply_mix('2023-01-01')
    
    print("Plotting Holiday Price Impact for 2023...")
    visualizer.plot_holiday_price_impact(2023)
    
    print("Comparing Holiday Price Impacts for 2022, 2023...")
    visualizer.compare_holiday_price_impacts(2022, 2023)
    
    print("Plotting Yearly Grid Usage (all years)...")
    visualizer.plot_yearly_grid_usage()
    
    print("Plotting Yearly Renewable Trend (all years)...")
    visualizer.plot_yearly_renewable_trend()
    
    print("Comparing Supply Mix Between Years (2021, 2022, 2023)...")
    visualizer.compare_supply_mix_between_years(2021, 2022, 2023)
    
    print("Comparing Monthly Price Across Years for December (2021, 2022, 2023)...")
    visualizer.compare_monthly_price_across_years(12, 2021, 2022, 2023)

if __name__ == "__main__":
    main()
