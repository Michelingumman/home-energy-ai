import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
from holidays import Sweden

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HolidayProcessor:
    def __init__(self, start_date='1970-01-01', end_date='2025-12-31'):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.holidays = Sweden()
        
    def create_holiday_features(self):
        """Create DataFrame with holiday features"""
        # Generate date range
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='h')
        
        # Create DataFrame with dates as index
        df = pd.DataFrame(index=dates)
        
        # Add holiday features
        df['is_holiday'] = df.index.map(lambda x: x.date() in self.holidays).astype(int)
        df['is_holiday_eve'] = df.index.map(self._is_holiday_eve).astype(int)
        
        # Add distance to next holiday
        df['days_to_next_holiday'] = df.index.map(lambda x: self._days_to_next_holiday(x.date()))
        df['days_from_last_holiday'] = df.index.map(lambda x: self._days_from_last_holiday(x.date()))
        
        return df
    
    def _is_holiday_eve(self, date):
        """Check if date is day before a holiday"""
        next_day = (date + timedelta(days=1)).date()
        return next_day in self.holidays
    
    def _days_to_next_holiday(self, date):
        """Calculate days until next holiday"""
        current = date
        days = 0
        while days < 30:  # Limit search to 30 days
            current += timedelta(days=1)
            if current in self.holidays:
                return days + 1
            days += 1
        return 30  # Cap at 30 days
    
    def _days_from_last_holiday(self, date):
        """Calculate days since last holiday"""
        current = date
        days = 0
        while days < 30:  # Limit search to 30 days
            current -= timedelta(days=1)
            if current in self.holidays:
                return days + 1
            days += 1
        return 30  # Cap at 30 days

def main():
    # Create processor
    processor = HolidayProcessor()
    
    try:
        # Generate features
        logging.info("Generating holiday features...")
        df = processor.create_holiday_features()
        
        # Save to CSV
        output_path = Path(__file__).parent / "processed/holidays.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path)
        logging.info(f"Saved holiday features to {output_path}")
        
        # Print sample
        print("\nSample of generated features:")
        print(df.head())
        print(f"\nTotal rows: {len(df)}")
        
        # Print holiday statistics
        holidays_count = df['is_holiday'].sum()
        holiday_eves_count = df['is_holiday_eve'].sum()
        logging.info(f"Total holidays: {holidays_count/24:.0f} days")
        logging.info(f"Total holiday eves: {holiday_eves_count/24:.0f} days")
        
    except Exception as e:
        logging.error(f"Error processing holidays: {str(e)}")
        raise

if __name__ == "__main__":
    main() 