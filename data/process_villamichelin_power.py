import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PowerConsumptionProcessor:
    def __init__(self, input_file):
        """Initialize processor with input file path"""
        self.input_file = Path(input_file)
        self.df = None
        
    def load_data(self):
        """Load and perform initial data cleaning"""
        logging.info(f"Loading data from {self.input_file}")
        
        try:
            self.df = pd.read_csv(self.input_file)
            logging.info(f"Successfully loaded {len(self.df)} rows")
        except Exception as e:
            logging.error(f"Error loading file: {e}")
            raise
            
        return self
    
    def process_timestamps(self):
        """Process and clean timestamp columns"""
        # Convert From column to datetime
        self.df['datetime'] = pd.to_datetime(self.df['From'])
        
        # Extract date and hour
        self.df['date'] = self.df['datetime'].dt.date
        self.df['hour'] = self.df['datetime'].dt.hour
        
        # Sort by datetime
        self.df.sort_values('datetime', inplace=True)
        
        # Check for duplicates
        duplicates = self.df.duplicated(subset=['date', 'hour'], keep=False)
        if duplicates.any():
            logging.warning(f"Found {duplicates.sum()} duplicate entries")
            logging.warning("Keeping the last entry for each duplicate")
            self.df.drop_duplicates(subset=['date', 'hour'], keep='last', inplace=True)
        
        return self
    
    def select_columns(self):
        """Select and rename final columns"""
        self.df = self.df[['date', 'hour', 'Consumption', 'Unit']]
        self.df.columns = ['date', 'hour', 'consumption', 'unit']
        return self
    
    def validate_data(self):
        """Perform data validation"""
        # Check for missing values
        missing = self.df.isnull().sum()
        if missing.any():
            logging.warning(f"Found missing values:\n{missing[missing > 0]}")
        
        # Check unit consistency
        units = self.df['unit'].unique()
        if len(units) > 1:
            logging.warning(f"Multiple units found: {units}")
        
        # Check consumption range
        logging.info("\nConsumption Statistics:")
        logging.info(f"Min: {self.df['consumption'].min():.2f}")
        logging.info(f"Max: {self.df['consumption'].max():.2f}")
        logging.info(f"Mean: {self.df['consumption'].mean():.2f}")
        
        return self
    
    def process(self):
        """Run all processing steps"""
        return (self.load_data()
                .process_timestamps()
                .select_columns()
                .validate_data())
    
    def save_data(self, output_file):
        """Save processed data to CSV"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        logging.info(f"Saved processed data to {output_path}")
        
        # Display sample
        print("\nSample of processed data:")
        print(self.df.head())
        print("\nShape:", self.df.shape)

def main():
    # Setup paths relative to this file
    current_dir = Path(__file__).parent
    input_file = current_dir / "raw/villamichelin/villamichelin_power_consumption.csv"
    output_file = current_dir / "processed/villamichelin_power_consumption.csv"
    
    # Process data
    processor = PowerConsumptionProcessor(input_file)
    processor.process()
    processor.save_data(output_file)

if __name__ == "__main__":
    main()
