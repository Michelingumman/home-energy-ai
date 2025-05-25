import os
import pandas as pd
import yfinance as yf
import datetime
from pathlib import Path
import logging
import sys
import numpy as np

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data/logs/co2_gas_coal_fetch.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parents[1]

# Path to the SE3Prices file
se3_prices_path = project_root / 'data/processed/SE3prices.csv'

def validate_commodity_data_quality(df, data_type="commodity"):
    """
    Comprehensive data quality validation for commodity data.
    
    Args:
        df: DataFrame to validate
        data_type: Type of data for logging context
    
    Returns:
        dict: Quality metrics and status
    """
    logger.info(f"Starting data quality validation for {data_type}")
    
    if df.empty:
        logger.error(f"CRITICAL: {data_type} DataFrame is empty!")
        return {"status": "FAILED", "reason": "Empty DataFrame"}
    
    quality_report = {
        "status": "PASSED",
        "total_records": len(df),
        "date_range": None,
        "duplicates": 0,
        "missing_values": {},
        "gaps": [],
        "outliers": {},
        "issues": [],
        "commodity_columns": [],
        "problematic_records": {}
    }
    
    # Basic info
    logger.info(f"{data_type} - Total records: {len(df)}")
    
    # Check for commodity columns
    commodity_cols = ["CO2_Price", "Gas_Price", "Coal_Price"]
    found_commodity_cols = [col for col in commodity_cols if col in df.columns]
    quality_report["commodity_columns"] = found_commodity_cols
    
    if not found_commodity_cols:
        logger.error(f"CRITICAL: {data_type} - No commodity price columns found!")
        quality_report["status"] = "FAILED"
        quality_report["issues"].append("No commodity price columns found")
        return quality_report
    else:
        logger.info(f"COLUMNS FOUND: {data_type} - Found commodity columns: {found_commodity_cols}")
    
    # Date range analysis
    date_col = df.index.name if df.index.name else df.index
    try:
        if hasattr(df.index, 'min'):
            start_date = df.index.min()
            end_date = df.index.max()
            quality_report["date_range"] = f"{start_date} to {end_date}"
            logger.info(f"{data_type} - Date range: {start_date} to {end_date}")
            
            # Gap detection for daily/hourly data
            if len(df) > 1:
                expected_freq = pd.infer_freq(df.index)
                if expected_freq:
                    logger.info(f"FREQUENCY DETECTED: {data_type} - Detected frequency: {expected_freq}")
                    # Create expected date range
                    if 'H' in expected_freq:  # Hourly data
                        expected_range = pd.date_range(start=start_date, end=end_date, freq='H')
                    elif 'D' in expected_freq:  # Daily data
                        expected_range = pd.date_range(start=start_date, end=end_date, freq='D')
                    else:
                        expected_range = None
                    
                    if expected_range is not None:
                        missing_dates = expected_range.difference(df.index)
                        if len(missing_dates) > 0:
                            quality_report["gaps"] = [f"{len(missing_dates)} missing time periods"]
                            logger.warning(f"GAP WARNING: {data_type} - Found {len(missing_dates)} missing time periods")
                            quality_report["issues"].append(f"Missing {len(missing_dates)} time periods")
                            
                            # Show specific missing dates (sample)
                            sample_missing = missing_dates[:10] if len(missing_dates) > 10 else missing_dates
                            logger.warning(f"MISSING TIMESTAMPS: Sample missing dates: {sample_missing.tolist()}")
                            if len(missing_dates) > 10:
                                logger.warning(f"... and {len(missing_dates) - 10} more missing timestamps")
                            quality_report["problematic_records"]["missing_timestamps"] = missing_dates[:20].tolist()
                        else:
                            logger.info(f"NO GAPS: {data_type} - No gaps detected in time series")
                            
    except Exception as e:
        logger.error(f"DATE ERROR: {data_type} - Error processing dates: {e}")
        quality_report["issues"].append(f"Date processing error: {e}")
    
    # Duplicate detection
    duplicates = df.index.duplicated().sum()
    quality_report["duplicates"] = duplicates
    if duplicates > 0:
        duplicate_indices = df.index[df.index.duplicated()]
        logger.warning(f"DUPLICATE WARNING: {data_type} - Found {duplicates} duplicate timestamps")
        quality_report["issues"].append(f"{duplicates} duplicate timestamps")
        
        # Show specific duplicate timestamps (sample)
        sample_duplicates = duplicate_indices[:10] if len(duplicate_indices) > 10 else duplicate_indices
        logger.warning(f"DUPLICATE TIMESTAMPS: Sample duplicates: {sample_duplicates.tolist()}")
        if len(duplicate_indices) > 10:
            logger.warning(f"... and {len(duplicate_indices) - 10} more duplicate timestamps")
        quality_report["problematic_records"]["duplicate_timestamps"] = duplicate_indices[:20].tolist()
    else:
        logger.info(f"NO DUPLICATES: {data_type} - No duplicate timestamps found")
    
    # Missing values analysis for commodity columns
    missing_summary = df[found_commodity_cols].isnull().sum()
    missing_dict = missing_summary[missing_summary > 0].to_dict()
    quality_report["missing_values"] = missing_dict
    
    if missing_dict:
        logger.warning(f"MISSING VALUES WARNING: {data_type} - Missing values detected:")
        for col, count in missing_dict.items():
            pct = (count / len(df)) * 100
            logger.warning(f"   {col}: {count} missing ({pct:.1f}%)")
            
            # Show specific timestamps with missing values
            missing_mask = df[col].isnull()
            missing_timestamps = df.index[missing_mask]
            sample_missing = missing_timestamps[:10] if len(missing_timestamps) > 10 else missing_timestamps
            logger.warning(f"   MISSING VALUE TIMESTAMPS for {col}: {sample_missing.tolist()}")
            if len(missing_timestamps) > 10:
                logger.warning(f"   ... and {len(missing_timestamps) - 10} more missing value timestamps")
            
            if pct > 20:  # More than 20% missing is concerning for commodity data
                quality_report["issues"].append(f"{col} has {pct:.1f}% missing values")
            
            quality_report["problematic_records"][f"missing_{col}"] = missing_timestamps[:20].tolist()
    else:
        logger.info(f"NO MISSING VALUES: {data_type} - No missing values found in commodity columns")
    
    # Value validation for commodity columns
    for col in found_commodity_cols:
        if col in df.columns and not df[col].empty:
            # Check for negative values (prices should be positive)
            negative_mask = df[col] < 0
            negative_count = negative_mask.sum()
            if negative_count > 0:
                logger.warning(f"NEGATIVE VALUES WARNING: {data_type} - {col}: {negative_count} negative values found")
                quality_report["issues"].append(f"{col} has {negative_count} negative values")
                
                # Show specific negative value records
                negative_records = df[negative_mask][[col]]
                sample_negative = negative_records.head(10)
                logger.warning(f"NEGATIVE VALUE RECORDS for {col}:")
                for idx, row in sample_negative.iterrows():
                    logger.warning(f"   {idx}: {row[col]}")
                if len(negative_records) > 10:
                    logger.warning(f"   ... and {len(negative_records) - 10} more negative value records")
                
                quality_report["problematic_records"][f"negative_{col}"] = [
                    {"timestamp": str(idx), "value": float(row[col])} 
                    for idx, row in negative_records.head(20).iterrows()
                ]
            
            # Check for zero values (might indicate data issues)
            zero_mask = df[col] == 0
            zero_count = zero_mask.sum()
            if zero_count > 0:
                zero_pct = (zero_count / len(df)) * 100
                if zero_pct > 5:  # More than 5% zeros is suspicious
                    logger.warning(f"ZERO VALUES WARNING: {data_type} - {col}: {zero_count} zero values ({zero_pct:.1f}%)")
                    quality_report["issues"].append(f"{col} has {zero_pct:.1f}% zero values")
                    
                    # Show specific zero value records
                    zero_records = df[zero_mask][[col]]
                    sample_zero = zero_records.head(10)
                    logger.warning(f"ZERO VALUE TIMESTAMPS for {col}: {sample_zero.index.tolist()}")
                    if len(zero_records) > 10:
                        logger.warning(f"   ... and {len(zero_records) - 10} more zero value timestamps")
                    
                    quality_report["problematic_records"][f"zero_{col}"] = zero_records.head(20).index.tolist()
                else:
                    logger.info(f"ZERO VALUES INFO: {data_type} - {col}: {zero_count} zero values ({zero_pct:.1f}%) - acceptable")
            
            # Outlier detection using IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                outlier_pct = (outlier_count / len(df)) * 100
                quality_report["outliers"][col] = outlier_count
                
                # Show statistics and sample outliers
                logger.info(f"OUTLIER STATS for {col}: Q1={Q1:.3f}, Q3={Q3:.3f}, IQR={IQR:.3f}")
                logger.info(f"OUTLIER BOUNDS for {col}: Lower={lower_bound:.3f}, Upper={upper_bound:.3f}")
                
                if outlier_pct > 25:  # More than 25% outliers is concerning for commodity data (less strict than other data types)
                    logger.warning(f"OUTLIER WARNING: {data_type} - {col}: {outlier_count} outliers ({outlier_pct:.1f}%)")
                    quality_report["issues"].append(f"{col} has {outlier_pct:.1f}% outliers")
                    
                    # Show specific outlier records (most extreme ones)
                    outlier_records = df[outlier_mask][[col]]
                    outlier_records_sorted = outlier_records.sort_values(col)
                    
                    # Show most extreme low and high outliers
                    extreme_low = outlier_records_sorted.head(5)
                    extreme_high = outlier_records_sorted.tail(5)
                    
                    logger.warning(f"EXTREME LOW OUTLIERS for {col}:")
                    for idx, row in extreme_low.iterrows():
                        logger.warning(f"   {idx}: {row[col]:.3f}")
                    
                    logger.warning(f"EXTREME HIGH OUTLIERS for {col}:")
                    for idx, row in extreme_high.iterrows():
                        logger.warning(f"   {idx}: {row[col]:.3f}")
                    
                    if outlier_count > 10:
                        logger.warning(f"   ... and {outlier_count - 10} more outlier records")
                    
                    quality_report["problematic_records"][f"outliers_{col}"] = [
                        {"timestamp": str(idx), "value": float(row[col])} 
                        for idx, row in outlier_records_sorted.head(10).iterrows()
                    ] + [
                        {"timestamp": str(idx), "value": float(row[col])} 
                        for idx, row in outlier_records_sorted.tail(10).iterrows()
                    ]
                else:
                    logger.info(f"OUTLIER INFO: {data_type} - {col}: {outlier_count} outliers ({outlier_pct:.1f}%) - within acceptable range for commodity data")
                    
                    # Still log a few examples for reference
                    outlier_records = df[outlier_mask][[col]]
                    sample_outliers = outlier_records.head(5)
                    logger.info(f"SAMPLE OUTLIERS for {col}: {[(str(idx), float(row[col])) for idx, row in sample_outliers.iterrows()]}")
    
    # Final status determination
    if quality_report["issues"]:
        if any("FAILED" in str(issue) or "No commodity" in str(issue) for issue in quality_report["issues"]):
            quality_report["status"] = "FAILED"
        else:
            quality_report["status"] = "WARNING"
        
        logger.warning(f"QUALITY SUMMARY: {data_type} - Data quality issues found: {len(quality_report['issues'])}")
        for issue in quality_report["issues"]:
            logger.warning(f"   Issue: {issue}")
    else:
        logger.info(f"QUALITY PASSED: {data_type} - All data quality checks passed")
    
    return quality_report

def main():
    """Main function with comprehensive logging and error handling"""
    logger.info("STARTING: CO2/Gas/Coal commodity data update")
    logger.info(f"FILE PATH: Working with file: {se3_prices_path}")
    
    try:
        # Load the existing SE3Prices.csv file
        if not se3_prices_path.exists():
            logger.error(f"FILE NOT FOUND: SE3Prices file not found: {se3_prices_path}")
            return False
            
        se3_prices = pd.read_csv(se3_prices_path, parse_dates=True, index_col=0)
        logger.info(f"FILE LOADED: SE3Prices data loaded - {len(se3_prices)} records")
        logger.info(f"INDEX INFO: SE3Prices index name: {se3_prices.index.name}")
        logger.info(f"COLUMNS INFO: SE3Prices columns: {list(se3_prices.columns)}")

        # Determine the latest date with commodity data
        latest_date = None
        commodity_cols = ["CO2_Price", "Gas_Price", "Coal_Price"]
        
        if all(col in se3_prices.columns for col in commodity_cols):
            logger.info("COLUMNS CHECK: All commodity price columns found in existing data")
            
            # First, make sure the index is actually datetime
            se3_prices.index = pd.to_datetime(se3_prices.index)
            
            # Check if we have hourly data
            is_hourly = 'hour' in str(se3_prices.index.dtype).lower() or 'time' in str(se3_prices.index.dtype).lower()
            logger.info(f"FREQUENCY INFO: Data frequency appears to be: {'hourly' if is_hourly else 'daily/other'}")
            
            # Find the last timestamp that has non-NaN values for all three commodities
            mask = se3_prices[commodity_cols].notna().all(axis=1)
            if mask.any():
                # Get the maximum date with valid data
                valid_timestamps = se3_prices.index[mask]
                latest_date = max(valid_timestamps)
                logger.info(f"LATEST DATA: Latest timestamp with complete commodity data: {latest_date}")
                
                # For display purposes, get the date part
                latest_date_only = latest_date.strftime('%Y-%m-%d')
                latest_time_only = latest_date.strftime('%H:%M')
                logger.info(f"   Date: {latest_date_only}, Time: {latest_time_only}")
            else:
                logger.warning("DATA WARNING: No existing commodity data found with complete values")
        else:
            missing_cols = [col for col in commodity_cols if col not in se3_prices.columns]
            logger.warning(f"COLUMNS WARNING: Missing commodity price columns: {missing_cols}")

        # If no valid date found, start from a default date
        if latest_date is None:
            start_date = "2017-01-01"
            logger.info(f"DEFAULT START: Using default start date: {start_date}")
        else:
            # For daily data sources like Yahoo Finance, we need to get the last calendar date 
            # that needs updating (earliest date missing data)
            
            # Get the last calendar date in the dataset
            last_calendar_date = latest_date.date()
            
            # Get today's date object for comparison
            today_date = datetime.datetime.now().date()
            
            # If the latest data is from today and it's the last hour (or close to it),
            # then we might be fully up to date
            if last_calendar_date == today_date and latest_date.hour >= 22:
                logger.info("UP TO DATE: Data appears to be up to date for today")
                is_up_to_date = True
                start_date = today_date.strftime('%Y-%m-%d')
            else:
                # Otherwise, we need to update from the last calendar date
                is_up_to_date = False
                start_date = last_calendar_date.strftime('%Y-%m-%d')
                logger.info(f"UPDATE NEEDED: Need to update commodity prices from {start_date} onwards")

        # Get today's date
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        logger.info(f"FETCH RANGE: Fetching new data from {start_date} to {today}")

        # Only fetch new data if needed
        if start_date <= today and not (is_up_to_date if 'is_up_to_date' in locals() else False):
            success_count = 0
            total_symbols = 3
            
            try:
                logger.info("API FETCH: Fetching commodity data from Yahoo Finance...")
                
                # Use Yahoo Finance for commodities
                # Natural Gas futures
                logger.info("NATURAL GAS: Fetching Natural Gas data (NG=F)...")
                gas = yf.download("NG=F", start=start_date, end=today, progress=False)
                logger.info(f"   Gas data points: {len(gas)}")
                if not gas.empty:
                    success_count += 1
                
                # Coal - use BTU (Peabody Energy) instead of KOL (ETF which may be delisted)
                logger.info("COAL: Fetching Coal data (BTU)...")
                coal = yf.download("BTU", start=start_date, end=today, progress=False)
                logger.info(f"   Coal data points: {len(coal)}")
                if not coal.empty:
                    success_count += 1
                
                # Carbon allowances ETF (KRBN)
                logger.info("CO2: Fetching CO2 data (KRBN)...")
                co2 = yf.download("KRBN", start=start_date, end=today, progress=False)
                logger.info(f"   CO2 data points: {len(co2)}")
                if not co2.empty:
                    success_count += 1
                
                logger.info(f"FETCH SUMMARY: Successfully fetched {success_count}/{total_symbols} commodity datasets")
                
                # Check if we have at least some data
                has_data = not (co2.empty and gas.empty and coal.empty)
                
                if has_data:
                    logger.info("PROCESSING: Processing downloaded commodity data...")
                    
                    # Important: Create a DataFrame first to avoid keeping the original ticker in the MultiIndex
                    co2_prices = pd.DataFrame()
                    if not co2.empty:
                        co2_prices["CO2_Price"] = co2["Close"]

                    gas_prices = pd.DataFrame()
                    if not gas.empty:
                        gas_prices["Gas_Price"] = gas["Close"]

                    coal_prices = pd.DataFrame()
                    if not coal.empty:
                        coal_prices["Coal_Price"] = coal["Close"]

                    # Combine the data
                    new_energy_prices = pd.concat([co2_prices, gas_prices, coal_prices], axis=1)
                    
                    if not new_energy_prices.empty:
                        # Make sure indices are datetime
                        new_energy_prices.index = pd.to_datetime(new_energy_prices.index)
                        
                        logger.info(f"DATA SUMMARY: New commodity data summary:")
                        logger.info(f"   Date range: {min(new_energy_prices.index)} to {max(new_energy_prices.index)}")
                        logger.info(f"   Columns: {', '.join(new_energy_prices.columns)}")
                        logger.info(f"   Records: {len(new_energy_prices)}")

                        # Validate new commodity data quality
                        new_data_quality = validate_commodity_data_quality(new_energy_prices, "new commodity data")

                        # Check if SE3Prices is hourly data
                        if 'hour' in str(se3_prices.index.dtype).lower() or 'time' in str(se3_prices.index.dtype).lower():
                            logger.info("HOURLY UPDATE: SE3Prices is hourly data, forward-filling daily commodity prices...")
                            
                            # For each date in energy_prices, assign the same price to all hours in that day
                            updated_hours = 0
                            updated_dates = 0
                            for date, row in new_energy_prices.iterrows():
                                date_str = date.strftime('%Y-%m-%d')
                                mask = se3_prices.index.strftime('%Y-%m-%d') == date_str
                                
                                if mask.any():
                                    hours_for_date = sum(mask)
                                    for col in new_energy_prices.columns:
                                        if pd.notna(row[col]):  # Only update if we have a value
                                            se3_prices.loc[mask, col] = row[col]
                                    logger.info(f"   Updated {hours_for_date} hours for {date_str}")
                                    updated_hours += hours_for_date
                                    updated_dates += 1
                            
                            logger.info(f"UPDATE SUCCESS: Total updated: {updated_dates} dates, {updated_hours} hours")
                        else:
                            logger.info("DIRECT MERGE: SE3Prices is not hourly, using direct merge...")
                            
                            # Create new columns if they don't exist
                            for col in new_energy_prices.columns:
                                if col not in se3_prices.columns:
                                    se3_prices[col] = None
                                    logger.info(f"   Created new column: {col}")
                            
                            # Update the new data
                            updated_rows = 0
                            new_rows = 0
                            for date, row in new_energy_prices.iterrows():
                                if date in se3_prices.index:
                                    for col in new_energy_prices.columns:
                                        if pd.notna(row[col]):  # Only update if we have a value
                                            se3_prices.loc[date, col] = row[col]
                                    updated_rows += 1
                                else:
                                    # This handles new dates that aren't in the original dataset
                                    new_row = pd.Series(index=se3_prices.columns)
                                    for col in new_energy_prices.columns:
                                        if pd.notna(row[col]):
                                            new_row[col] = row[col]
                                    se3_prices.loc[date] = new_row
                                    new_rows += 1
                            
                            logger.info(f"MERGE SUCCESS: Updated {updated_rows} existing rows, added {new_rows} new rows")

                        # Fill any missing values with forward fill then backward fill
                        # First check if we had data before
                        had_data_before = se3_prices[commodity_cols].notna().any().any()
                        
                        # Only apply fill methods if we already had some data
                        if had_data_before:
                            logger.info("FILLING: Applying forward fill and backward fill to handle missing values...")
                            before_fill = se3_prices[commodity_cols].isnull().sum().sum()
                            se3_prices = se3_prices.ffill().bfill()
                            after_fill = se3_prices[commodity_cols].isnull().sum().sum()
                            filled_count = before_fill - after_fill
                            if filled_count > 0:
                                logger.info(f"   Filled {filled_count} missing values")

                        # Round all numeric columns to 3 decimal places
                        logger.info("ROUNDING: Rounding numeric values to 3 decimal places...")
                        numeric_columns = se3_prices.select_dtypes(include=['number']).columns
                        se3_prices[numeric_columns] = se3_prices[numeric_columns].round(3)

                        # Clean up old ticker columns if they exist
                        old_cols = ['KRBN', 'NG=F', 'KOL', 'BTU']
                        removed_cols = []
                        for col in old_cols:
                            if col in se3_prices.columns:
                                se3_prices = se3_prices.drop(columns=[col])
                                removed_cols.append(col)
                        if removed_cols:
                            logger.info(f"CLEANUP: Removed old ticker columns: {removed_cols}")

                        # Final data quality validation
                        logger.info("FINAL VALIDATION: Performing final data quality validation...")
                        final_quality = validate_commodity_data_quality(se3_prices, "final SE3Prices data")

                        # Save the updated data
                        logger.info("SAVING: Saving updated data to file...")
                        se3_prices.to_csv(se3_prices_path)
                        logger.info(f"SAVE SUCCESS: Successfully updated {se3_prices_path}")
                        logger.info(f"FILE STATS: Final file contains {len(se3_prices)} total records")
                        
                        # Verify the file was saved
                        try:
                            updated_file = pd.read_csv(se3_prices_path, nrows=5)
                            logger.info(f"VERIFICATION SUCCESS: File verification successful. Columns: {', '.join(updated_file.columns)}")
                        except Exception as e:
                            logger.error(f"VERIFICATION FAILED: File verification failed: {str(e)}")
                            return False
                        
                        # Final status report
                        if final_quality["status"] == "PASSED":
                            logger.info("SYSTEM SUCCESS: COMMODITY DATA UPDATE COMPLETED SUCCESSFULLY!")
                            return True
                        elif final_quality["status"] == "WARNING":
                            logger.warning("SYSTEM WARNING: COMMODITY DATA UPDATE COMPLETED WITH WARNINGS")
                            return True  # Still consider it successful but with issues
                        else:
                            logger.error("SYSTEM FAILED: COMMODITY DATA UPDATE COMPLETED BUT WITH QUALITY ISSUES")
                            return False
                            
                    else:
                        logger.error("API ERROR: No new price data was combined. Check for API issues.")
                        return False
                else:
                    logger.error("API ERROR: No data available for the specified date range from any source")
                    
                    # Check if it's a weekend or markets are closed (this is normal, not an error)
                    today_weekday = datetime.datetime.now().weekday()  # 0=Monday, 6=Sunday
                    
                    if today_weekday >= 5:  # Saturday (5) or Sunday (6)
                        logger.info("MARKET CLOSED: No data available because markets are closed (weekend)")
                        
                        # Still validate existing data
                        if not se3_prices.empty:
                            logger.info("VALIDATION: Validating existing commodity data...")
                            existing_quality = validate_commodity_data_quality(se3_prices, "existing SE3Prices data")
                            if existing_quality["status"] == "PASSED":
                                logger.info("VALIDATION PASSED: EXISTING DATA VALIDATION SUCCESSFUL")
                                return True
                            elif existing_quality["status"] == "WARNING":
                                logger.warning("VALIDATION WARNING: EXISTING DATA VALIDATION FOUND ISSUES")
                                return True  # Still successful, just with warnings
                            else:
                                logger.error("VALIDATION FAILED: EXISTING DATA VALIDATION FAILED")
                                return False
                        else:
                            logger.error("FILE ERROR: SE3Prices file is empty")
                            return False
                    else:
                        logger.warning("MARKET WARNING: No data available during market hours - may be a holiday or API issue")
                        
                        # Still validate existing data and return True with warning
                        if not se3_prices.empty:
                            logger.info("VALIDATION: Validating existing commodity data...")
                            existing_quality = validate_commodity_data_quality(se3_prices, "existing SE3Prices data")
                            if existing_quality["status"] != "FAILED":
                                logger.info("SYSTEM WARNING: NO NEW DATA AVAILABLE BUT EXISTING DATA IS VALID")
                                return True  # Not a failure, just no new data
                            else:
                                logger.error("VALIDATION FAILED: EXISTING DATA VALIDATION FAILED")
                                return False
                        else:
                            logger.error("FILE ERROR: SE3Prices file is empty")
                            return False
            except Exception as e:
                logger.error(f"API EXCEPTION: Error fetching commodity data: {str(e)}")
                return False
        else:
            logger.info("NO UPDATE NEEDED: Data is already up to date, no new data to fetch")
            
            # Still validate existing data
            if not se3_prices.empty:
                logger.info("VALIDATION: Validating existing commodity data...")
                existing_quality = validate_commodity_data_quality(se3_prices, "existing SE3Prices data")
                if existing_quality["status"] == "PASSED":
                    logger.info("VALIDATION PASSED: EXISTING DATA VALIDATION SUCCESSFUL")
                    return True
                else:
                    logger.warning("VALIDATION WARNING: EXISTING DATA VALIDATION FOUND ISSUES")
                    return False
            else:
                logger.error("FILE ERROR: SE3Prices file is empty")
                return False

    except Exception as e:
        logger.error(f"CRITICAL ERROR: Fatal error in main function: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("SCRIPT SUCCESS: Script completed successfully")
        exit(0)
    else:
        logger.error("SCRIPT FAILED: Script completed with errors")
        exit(1)
