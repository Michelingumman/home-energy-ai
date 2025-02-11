import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ElectricityPricePreprocessor:
    def __init__(self, file_path, price_area='SE3', eur_to_sek=11.25):
        """
        Initialize preprocessor for electricity price data
        
        Args:
            file_path (str): Path to the CSV file
            price_area (str): Price area to filter (default: 'SE3')
            eur_to_sek (float): Exchange rate EUR to SEK (default: 11.8 - Feb 2024 rate)
        
        Note:
            Input prices are in EUR/MWh, output will be in öre/kWh
        """
        self.file_path = Path(file_path)
        self.price_area = price_area
        self.eur_to_sek = eur_to_sek
        self.df = None
        
    def load_data(self):
        """Load and perform initial data cleaning"""
        logging.info(f"Loading data from {self.file_path}")
        
        try:
            self.df = pd.read_csv(self.file_path, sep=';', decimal=',')
            logging.info(f"Successfully loaded {len(self.df)} rows")
        except Exception as e:
            logging.error(f"Error loading file: {e}")
            raise
            
        return self
    
    def filter_area(self):
        """Filter for specific price area"""
        self.df = self.df[self.df['PriceArea'] == self.price_area].copy()
        logging.info(f"Filtered for {self.price_area}: {len(self.df)} rows")
        return self
    
    def process_timestamps(self):
        """Process timestamp column"""
        self.df['HourSE'] = pd.to_datetime(self.df['HourUTC'])
        self.df.set_index('HourSE', inplace=True)
        self.df.sort_index(inplace=True)
        
        logging.info(f"Time range: {self.df.index.min()} to {self.df.index.max()}")
        return self
    
    def convert_prices(self):
        """Convert EUR/MWh to öre/kWh"""
        conversion_factor = self.eur_to_sek * 100 / 1000  # EUR/MWh -> öre/kWh
        self.df['SE3_price_ore'] = self.df['SpotPriceEUR'] * conversion_factor
        
        # Log price statistics
        logging.info("\nPrice Statistics:")
        logging.info(f"Input (EUR/MWh): mean={self.df['SpotPriceEUR'].mean():.2f}, "
                    f"min={self.df['SpotPriceEUR'].min():.2f}, "
                    f"max={self.df['SpotPriceEUR'].max():.2f}")
        logging.info(f"Output (öre/kWh): mean={self.df['SE3_price_ore'].mean():.2f}, "
                    f"min={self.df['SE3_price_ore'].min():.2f}, "
                    f"max={self.df['SE3_price_ore'].max():.2f}")
        return self
    
    def add_features(self):
        """Add time-based features"""
        # Basic time features
        self.df['hour'] = self.df.index.hour
        self.df['day_of_week'] = self.df.index.dayofweek
        self.df['month'] = self.df.index.month
        self.df['week_of_year'] = self.df.index.isocalendar().week
        
        # Cyclical encoding
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour']/24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour']/24)
        self.df['day_of_week_sin'] = np.sin(2 * np.pi * self.df['day_of_week']/7)
        self.df['day_of_week_cos'] = np.cos(2 * np.pi * self.df['day_of_week']/7)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month']/12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month']/12)
        
        # Additional features
        self.df['is_peak_hour'] = ((self.df['hour'] >= 6) & (self.df['hour'] <= 22)).astype(int)
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        
        return self
    
    def handle_missing_values(self):
        """Handle missing values"""
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            logging.info("\nHandling missing values:")
            logging.info(missing[missing > 0])
            self.df = self.df.ffill().bfill()
            
        return self
    
    def select_final_columns(self):
        """Select and order final columns"""
        # Keep PriceArea, SE3_price_ore and all feature columns
        feature_cols = ['hour', 'day_of_week', 'month', 'week_of_year',
                       'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                       'month_sin', 'month_cos', 'is_peak_hour', 'is_weekend']
        
        self.df = self.df[['PriceArea', 'SE3_price_ore'] + feature_cols]
        return self
    
    def process(self):
        """Run all preprocessing steps"""
        return (self.load_data()
                .filter_area()
                .process_timestamps()
                .convert_prices()
                .add_features()
                .handle_missing_values()
                .select_final_columns())
    
    def get_processed_data(self):
        """Return the processed dataframe"""
        if self.df is None:
            raise ValueError("Data not processed. Call process() first.")
        return self.df

def main():
    # Get project root (2 levels up from this file)
    project_root = Path(__file__).parents[1]
    
    # Input and output paths relative to project root
    file_path = project_root / "data/raw/Elspotprices/Elspotprices to - 2024.csv"
    output_path = project_root / "data/processed/SE3prices.csv"
    
    preprocessor = ElectricityPricePreprocessor(file_path)
    df = preprocessor.process().get_processed_data()
    
    # Display sample and info
    print("\nProcessed Data Sample:")
    print(df.head())
    print("\nDataframe Info:")
    print(df.info())
    
    # Save processed data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    logging.info(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    main() 