import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TimeFeatureGenerator:
    def __init__(self, price_data_path=None, decimal_precision=3):
        """
        Initialize time feature generator
        
        Args:
            price_data_path (str, optional): Path to the processed price data CSV.
                                            Only needed for validation of date range.
            decimal_precision (int): Number of decimal places to keep in output
        """
        self.price_data_path = Path(price_data_path) if price_data_path else None
        self.df = None
        self.start_date = None
        self.end_date = None
        self.decimal_precision = decimal_precision
        
    def load_price_data(self):
        """Load the processed price data to determine start date and create full date range up to 2050-01-01."""
        if self.price_data_path:
            logging.info(f"Loading price data from {self.price_data_path}")
            price_df = pd.read_csv(self.price_data_path, index_col=0)
            price_df.index = pd.to_datetime(price_df.index)
            # Use the earliest date from the price data as the start date
            self.start_date = price_df.index.min()
            # Set the end date to 2050-01-01 23:00
            self.end_date = pd.to_datetime("2050-01-01 23:00")
            # Create a new DataFrame with an hourly index from start_date to end_date
            self.df = pd.DataFrame(index=pd.date_range(start=self.start_date, end=self.end_date, freq='H'))
            logging.info(f"Using date range: {self.start_date} to {self.end_date}")
        else:
            logging.warning("No price data path provided. Features will be generated without date validation.")
        return self
    
    def generate_cyclical_features(self):
        """Generate cyclical time features."""
        logging.info("Generating cyclical time features")
        
        # Hour of day
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df.index.hour / 24).round(self.decimal_precision)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df.index.hour / 24).round(self.decimal_precision)
        
        # Day of week
        self.df['day_of_week_sin'] = np.sin(2 * np.pi * self.df.index.dayofweek / 7).round(self.decimal_precision)
        self.df['day_of_week_cos'] = np.cos(2 * np.pi * self.df.index.dayofweek / 7).round(self.decimal_precision)
        
        # Month
        self.df['month_sin'] = np.sin(2 * np.pi * self.df.index.month / 12).round(self.decimal_precision)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df.index.month / 12).round(self.decimal_precision)
        
        return self
    
    def generate_binary_features(self):
        """Generate binary and categorical time features."""
        logging.info("Generating binary and categorical features")
        
        # Morning peak hours (6:00-9:00)
        self.df['is_morning_peak'] = ((self.df.index.hour >= 6) & (self.df.index.hour <= 9)).astype(int)
        
        # Evening peak hours (17:00-20:00)
        self.df['is_evening_peak'] = ((self.df.index.hour >= 17) & (self.df.index.hour <= 20)).astype(int)
        
        # Weekend indicator
        self.df['is_weekend'] = (self.df.index.dayofweek >= 5).astype(int)
        
        # Season (-1: winter, 0: spring/fall, 1: summer)
        self.df['season'] = 0  # Default to spring/fall
        self.df.loc[(self.df.index.month >= 6) & (self.df.index.month <= 8), 'season'] = 1  # Summer
        self.df.loc[(self.df.index.month == 12) | (self.df.index.month <= 2), 'season'] = -1  # Winter
        
        return self
    
    def generate_features(self, start_date=None, end_date=None):
        """Generate all features for a given date range.
        
        If start_date and end_date are provided, a new DataFrame is created.
        Otherwise, if price data was loaded, the DataFrame from load_price_data is used.
        """
        if start_date and end_date:
            dates = pd.date_range(start=start_date, end=end_date, freq='H')
            self.df = pd.DataFrame(index=dates)
        elif self.df is None:
            raise ValueError("Must either provide a date range or a valid price data path.")
        
        return (self.generate_cyclical_features()
                    .generate_binary_features())
    
    def get_feature_data(self):
        """Return the generated feature DataFrame."""
        if self.df is None:
            raise ValueError("Features not generated. Call generate_features() first.")
        return self.df

def main():
    # Get project root
    project_root = Path(__file__).parents[1]
    
    # Input and output paths
    price_data_path = project_root / "data/processed/SE3prices.csv"
    output_path = project_root / "data/processed/time_features.csv"
    
    # Get decimal precision from args if provided
    parser = argparse.ArgumentParser(description='Generate time features with specified precision')
    parser.add_argument('--precision', type=int, default=3, help='Number of decimal places for floating-point values')
    args = parser.parse_args()
    
    # Generate features
    generator = TimeFeatureGenerator(price_data_path, decimal_precision=args.precision)
    feature_df = (generator
                    .load_price_data()
                    .generate_features()
                    .get_feature_data())
    
    # Display sample and info
    print("\nGenerated Features Sample:")
    print(feature_df.head())
    
    # Calculate file size estimate
    original_size_kb = feature_df.memory_usage(deep=True).sum() / 1024
    print(f"\nEstimated memory usage: {original_size_kb:.2f} KB")
    
    # Save features with space-efficient settings
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, float_format=f'%.{args.precision}f')
    
    # Log output info
    file_size_kb = output_path.stat().st_size / 1024
    logging.info(f"Saved feature data to {output_path} ({file_size_kb:.2f} KB)")
    logging.info(f"Using {args.precision} decimal places for floating-point values")

if __name__ == "__main__":
    main()
