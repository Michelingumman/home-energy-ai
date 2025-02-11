import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

class PriceVisualizer:
    def __init__(self, data_path=None):
        """Initialize visualizer with data path"""
        if data_path is None:
            # Get the path relative to this file
            self.data_path = Path(__file__).parent / "SE3prices.csv"
        else:
            self.data_path = Path(data_path)
        self.df = self.load_data()
        
    def load_data(self):
        """Load and prepare the data"""
        df = pd.read_csv(self.data_path, index_col=0)
        df.index = pd.to_datetime(df.index)
        return df
    
    def plot_date_range(self, start_date, end_date, title=None):
        """Plot prices for a specific date range"""
        mask = (self.df.index >= start_date) & (self.df.index <= end_date)
        data = self.df[mask]
        
        plt.figure(figsize=(15, 6))
        plt.step(data.index, data['price_ore'], where='post', label='Price')
        plt.grid(True, alpha=0.3)
        plt.title(title or f'Electricity Prices ({start_date} to {end_date})')
        plt.xlabel('Date')
        plt.ylabel('Price (öre/kWh)')
        
        # Add day markers
        days = pd.date_range(start_date, end_date, freq='D')
        plt.xticks(days, [d.strftime('%Y-%m-%d') for d in days], rotation=45)
        
        # Add statistics
        avg_price = data['price_ore'].mean()
        max_price = data['price_ore'].max()
        min_price = data['price_ore'].min()
        plt.axhline(y=avg_price, color='r', linestyle='--', alpha=0.5, label=f'Average: {avg_price:.2f}')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\nStatistics for period {start_date} to {end_date}:")
        print(f"Average price: {avg_price:.2f} öre/kWh")
        print(f"Maximum price: {max_price:.2f} öre/kWh")
        print(f"Minimum price: {min_price:.2f} öre/kWh")
    
    def plot_daily_profile(self, date):
        """Plot 24-hour profile for a specific date"""
        day_start = pd.Timestamp(date).normalize()
        day_end = day_start + pd.Timedelta(days=1)
        
        mask = (self.df.index >= day_start) & (self.df.index < day_end)
        data = self.df[mask]
        
        plt.figure(figsize=(12, 6))
        plt.step(range(24), data['price_ore'].values, where='post', label='Price')
        plt.grid(True, alpha=0.3)
        plt.title(f'24-Hour Price Profile for {date}')
        plt.xlabel('Hour of Day')
        plt.ylabel('Price (öre/kWh)')
        plt.xticks(range(24))
        
        # Add average line
        avg_price = data['price_ore'].mean()
        plt.axhline(y=avg_price, color='r', linestyle='--', alpha=0.5, 
                   label=f'Daily Average: {avg_price:.2f}')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_monthly_comparison(self, year, month):
        """Plot daily averages for a specific month with previous year comparison"""
        # Current year data
        current_month = self.df[
            (self.df.index.year == year) & 
            (self.df.index.month == month)
        ]
        
        # Previous year data
        prev_month = self.df[
            (self.df.index.year == year-1) & 
            (self.df.index.month == month)
        ]
        
        # Calculate daily averages
        current_daily = current_month.groupby(current_month.index.day)['price_ore'].mean()
        prev_daily = prev_month.groupby(prev_month.index.day)['price_ore'].mean()
        
        plt.figure(figsize=(15, 6))
        plt.plot(current_daily.index, current_daily.values, 
                label=f'{year}', marker='o')
        plt.plot(prev_daily.index, prev_daily.values, 
                label=f'{year-1}', marker='o', linestyle='--', alpha=0.7)
        
        plt.grid(True, alpha=0.3)
        plt.title(f'Daily Average Prices - {pd.Timestamp(year, month, 1).strftime("%B %Y")}')
        plt.xlabel('Day of Month')
        plt.ylabel('Average Price (öre/kWh)')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_price_distribution(self, year=None):
        """Plot price distribution for a specific year or all data"""
        if year:
            data = self.df[self.df.index.year == year]
            title = f'Price Distribution for {year}'
        else:
            data = self.df
            title = 'Price Distribution for All Data'
        
        plt.figure(figsize=(12, 6))
        sns.histplot(data=data, x='price_ore', bins=50)
        plt.title(title)
        plt.xlabel('Price (öre/kWh)')
        plt.ylabel('Count')
        
        # Add statistics
        plt.axvline(data['price_ore'].mean(), color='r', linestyle='--', 
                   label=f'Mean: {data["price_ore"].mean():.2f}')
        plt.axvline(data['price_ore'].median(), color='g', linestyle='--', 
                   label=f'Median: {data["price_ore"].median():.2f}')
        
        plt.legend()
        plt.tight_layout()
        plt.show()

def main():
    visualizer = PriceVisualizer()
    
    # Example usage:
    
    # 1. Plot a specific week
    visualizer.plot_date_range('2024-02-01', '2024-02-07', 
                            'First Week of February 2024')
    
    visualizer.plot_date_range('2023-02-01', '2023-02-07', 
                            'First Week of February 2023')
    
    # 2. Plot a specific day's 24-hour profile
    visualizer.plot_daily_profile('2024-02-05')
    
    # 3. Plot monthly comparison
    visualizer.plot_monthly_comparison(2024, 2)  # February 2024 vs February 2023
    
    # 4. Plot price distribution for 2023
    visualizer.plot_price_distribution(2023)

if __name__ == "__main__":
    main()
