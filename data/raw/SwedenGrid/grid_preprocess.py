import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GridDataProcessor:
    def __init__(self, input_file):
        self.input_file = Path(input_file)
        self.df = None
        
    def load_data(self):
        """Load and clean the raw grid supply data"""
        logging.info(f"Loading data from {self.input_file}")
        
        # Read the CSV file, skipping the first two rows (header description)
        self.df = pd.read_csv(self.input_file, sep='\t', skiprows=2)
        
        # Clean column names and select only required columns
        self.df.columns = [
            'month',
            'total_supply',
            'hydro',
            'wind',
            'nuclear',
            'solar',
            'thermal_total',
            'thermal_industrial',
            'thermal_district_heating',
            'thermal_condensation',
            'thermal_gas_turbine',
            'import'
        ]
        
        # Convert month column to datetime
        self.df['month'] = pd.to_datetime(self.df['month'].str.replace('M', '-'), format='%Y-%m')
        
        # Replace '..' with 0 (missing values for early wind/solar)
        self.df = self.df.replace('..', 0)
        
        # Convert all columns except 'month' to float
        numeric_columns = self.df.columns.difference(['month'])
        self.df[numeric_columns] = self.df[numeric_columns].astype(float)
        
        # Select only the required columns
        self.df = self.df[[
            'month',
            'total_supply',
            'hydro',
            'wind',
            'nuclear',
            'solar',
            'thermal_total',
            'import'
        ]]
        
        # Set month as index
        self.df.set_index('month', inplace=True)
        
        return self
    
    def process(self):
        """Run all processing steps"""
        return self.load_data()
    
    def save_data(self, output_file):
        """Save processed data to CSV"""
        self.df.to_csv(output_file)
        logging.info(f"Saved processed data to {output_file}")

def main():
    # Setup paths
    input_file = Path(__file__).parent / "Gridsupply_GWh_monthly.csv"
    output_file = Path(__file__).parents[2] / "processed/SwedenGrid.csv"
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Process data
    processor = GridDataProcessor(input_file)
    processor.process()
    processor.save_data(output_file)
    
    # Print sample of processed data
    print("\nSample of processed data:")
    print(processor.df.head())
    
    # Print feature summary
    print("\nFeatures in processed data:")
    for col in processor.df.columns:
        print(f"- {col}")

if __name__ == "__main__":
    main()