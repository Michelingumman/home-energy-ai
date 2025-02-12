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
        
        # Clean column names
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
        
        return self
    
    def add_features(self):
        """Add calculated features to the dataset"""
        # Calculate renewable percentage
        self.df['renewable_percentage'] = (
            (self.df['hydro'] + self.df['wind'] + self.df['solar']) / 
            self.df['total_supply'] * 100
        )
        
        # Calculate nuclear percentage
        self.df['nuclear_percentage'] = (
            self.df['nuclear'] / self.df['total_supply'] * 100
        )
        
        # Calculate thermal percentage
        self.df['thermal_percentage'] = (
            self.df['thermal_total'] / self.df['total_supply'] * 100
        )
        
        # Calculate import dependency
        self.df['import_percentage'] = (
            self.df['import'] / self.df['total_supply'] * 100
        )
        
        # Add year-over-year growth rates
        for col in ['total_supply', 'hydro', 'wind', 'nuclear', 'import']:
            self.df[f'{col}_yoy_change'] = (
                self.df[col].pct_change(periods=12) * 100
            )
        
        # Calculate energy mix complexity (Shannon diversity index)
        def calculate_diversity(row):
            sources = ['hydro', 'wind', 'solar', 'nuclear', 'thermal_total', 'import']
            proportions = [row[source]/row['total_supply'] for source in sources]
            proportions = [p for p in proportions if p > 0]  # Remove zero values
            return -sum(p * np.log(p) for p in proportions)
        
        self.df['energy_mix_diversity'] = self.df.apply(calculate_diversity, axis=1)
        
        return self
    
    def add_rolling_features(self, windows=[3, 6, 12]):
        """Add rolling average features"""
        for window in windows:
            for col in ['total_supply', 'renewable_percentage', 'nuclear_percentage']:
                self.df[f'{col}_{window}m_ma'] = (
                    self.df[col].rolling(window=window, min_periods=1).mean()
                )
        return self
    
    def process(self):
        """Run all processing steps"""
        return (self.load_data()
                .add_features()
                .add_rolling_features())
    
    def save_data(self, output_file):
        """Save processed data to CSV"""
        self.df.to_csv(output_file, index=False)
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
    print("\nFeatures created:")
    for col in processor.df.columns:
        print(f"- {col}")

if __name__ == "__main__":
    main()