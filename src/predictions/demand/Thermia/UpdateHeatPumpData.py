import os
import pandas as pd
from datetime import datetime, timedelta
import subprocess
import sys
import logging
import numpy as np

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('src/predictions/demand/logs/thermia_data_fetch.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Define the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

# Define the path to the main CSV file
MAIN_CSV_FILE = os.path.join(ROOT_DIR, 'data', 'processed', 'villamichelin', 'Thermia', 'HeatPumpPower.csv')

# Define the path to the script that fetches data
EXTRACT_SCRIPT_PATH = os.path.join(ROOT_DIR, 'src', 'predictions', 'demand', 'Thermia', 'ExtractHistoricData.py')

# Temporary file to store newly fetched data
TEMP_CSV_FILE = os.path.join(ROOT_DIR, 'data', 'processed', 'villamichelin', 'Thermia', 'temp_heat_pump_data.csv')

def validate_thermia_data_quality(df, data_type="thermia"):
    """
    Comprehensive data quality validation for Thermia heat pump data.
    
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
        "thermia_columns": [],
        "problematic_records": {}
    }
    
    # Basic info
    logger.info(f"{data_type} - Total records: {len(df)}")
    
    # Check for Thermia heat pump columns
    expected_thermia_cols = ["power_status", "supply_temp", "return_temp", "brine_in_temp", "brine_out_temp", "outdoor_temp", "desired_supply_temp"]
    found_thermia_cols = [col for col in expected_thermia_cols if col in df.columns]
    quality_report["thermia_columns"] = found_thermia_cols
    
    if not found_thermia_cols:
        logger.error(f"CRITICAL: {data_type} - No Thermia heat pump data columns found!")
        quality_report["status"] = "FAILED"
        quality_report["issues"].append("No Thermia heat pump data columns found")
        return quality_report
    else:
        logger.info(f"COLUMNS FOUND: {data_type} - Found Thermia columns: {found_thermia_cols}")
    
    # Date range analysis
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            start_date = df['timestamp'].min()
            end_date = df['timestamp'].max()
            quality_report["date_range"] = f"{start_date} to {end_date}"
            logger.info(f"{data_type} - Date range: {start_date} to {end_date}")
            
            # Gap detection for 15-minute interval data
            expected_intervals = (end_date - start_date).total_seconds() / (15 * 60) + 1
            actual_intervals = len(df)
            gap_count = int(expected_intervals - actual_intervals)
            
            if gap_count > 0:
                quality_report["gaps"] = [f"{gap_count} missing 15-minute intervals"]
                logger.warning(f"GAP WARNING: {data_type} - Found {gap_count} missing 15-minute intervals (expected {int(expected_intervals)}, got {actual_intervals})")
                quality_report["issues"].append(f"Missing {gap_count} 15-minute intervals of data")
                
                # Find specific missing intervals
                expected_range = pd.date_range(start=start_date, end=end_date, freq='15T')
                missing_intervals = expected_range.difference(df['timestamp'])
                sample_missing = missing_intervals[:30] if len(missing_intervals) > 30 else missing_intervals
                logger.warning(f"MISSING INTERVAL TIMESTAMPS: Sample missing intervals: {sample_missing.tolist()}")
                if len(missing_intervals) > 30:
                    logger.warning(f"... and {len(missing_intervals) - 30} more missing interval timestamps")
                quality_report["problematic_records"]["missing_intervals"] = missing_intervals[:100].tolist()
            else:
                logger.info(f"GAP CHECK OK: {data_type} - No gaps detected in 15-minute interval data")
                
        except Exception as e:
            logger.error(f"DATE ERROR: {data_type} - Error processing timestamps: {e}")
            quality_report["issues"].append(f"Timestamp processing error: {e}")
    
    # Duplicate detection
    if 'timestamp' in df.columns:
        duplicates = df.duplicated(subset=['timestamp']).sum()
        quality_report["duplicates"] = duplicates
        if duplicates > 0:
            duplicate_mask = df.duplicated(subset=['timestamp'], keep=False)
            duplicate_records = df[duplicate_mask]
            logger.warning(f"DUPLICATE WARNING: {data_type} - Found {duplicates} duplicate timestamps")
            quality_report["issues"].append(f"{duplicates} duplicate timestamps")
            
            # Show specific duplicate timestamps
            duplicate_timestamps = duplicate_records['timestamp'].tolist()
            sample_duplicates = duplicate_timestamps[:20] if len(duplicate_timestamps) > 20 else duplicate_timestamps
            logger.warning(f"DUPLICATE TIMESTAMPS: Sample duplicates: {sample_duplicates}")
            if len(duplicate_timestamps) > 20:
                logger.warning(f"... and {len(duplicate_timestamps) - 20} more duplicate timestamps")
            quality_report["problematic_records"]["duplicate_timestamps"] = duplicate_timestamps[:50]
        else:
            logger.info(f"DUPLICATE CHECK OK: {data_type} - No duplicate timestamps found")
    
    # Missing values analysis for Thermia columns
    missing_summary = df[found_thermia_cols].isnull().sum()
    missing_dict = missing_summary[missing_summary > 0].to_dict()
    quality_report["missing_values"] = missing_dict
    
    if missing_dict:
        logger.warning(f"MISSING VALUES WARNING: {data_type} - Missing values detected:")
        for col, count in missing_dict.items():
            pct = (count / len(df)) * 100
            logger.warning(f"   {col}: {count} missing ({pct:.1f}%)")
            
            # Show specific timestamps with missing values
            if 'timestamp' in df.columns:
                missing_mask = df[col].isnull()
                missing_timestamps = df[missing_mask]['timestamp'].tolist()
                sample_missing = missing_timestamps[:10] if len(missing_timestamps) > 10 else missing_timestamps
                logger.warning(f"   MISSING VALUE TIMESTAMPS for {col}: {sample_missing}")
                if len(missing_timestamps) > 10:
                    logger.warning(f"   ... and {len(missing_timestamps) - 10} more missing value timestamps")
                quality_report["problematic_records"][f"missing_{col}"] = missing_timestamps[:30]
            
            if pct > 20:  # More than 20% missing is concerning for sensor data
                quality_report["issues"].append(f"{col} has {pct:.1f}% missing values")
    else:
        logger.info(f"MISSING VALUES OK: {data_type} - No missing values found in Thermia columns")
    
    # Value validation for Thermia columns
    for col in found_thermia_cols:
        if col in df.columns and not df[col].empty:
            
            # Temperature range validation
            if 'temp' in col:
                # Check for extreme temperature values
                if 'outdoor' in col:
                    # Outdoor temperature: -40°C to 50°C is reasonable
                    extreme_mask = (df[col] < -40) | (df[col] > 50)
                else:
                    # Indoor/system temperatures: -20°C to 100°C is reasonable
                    extreme_mask = (df[col] < -20) | (df[col] > 100)
                
                extreme_count = extreme_mask.sum()
                if extreme_count > 0:
                    extreme_pct = (extreme_count / len(df)) * 100
                    if extreme_pct > 1:  # More than 1% extreme values is suspicious
                        logger.warning(f"EXTREME TEMPERATURE WARNING: {data_type} - {col}: {extreme_count} extreme temperature values")
                        quality_report["issues"].append(f"{col} has {extreme_count} extreme temperature values")
                        
                        # Show specific extreme temperature records
                        if 'timestamp' in df.columns:
                            extreme_records = df[extreme_mask][['timestamp', col]].sort_values(col)
                            extreme_low = extreme_records.head(5)
                            extreme_high = extreme_records.tail(5)
                            
                            logger.warning(f"EXTREME TEMPERATURE RECORDS for {col}:")
                            for _, row in extreme_low.iterrows():
                                logger.warning(f"   {row['timestamp']}: {row[col]:.2f}°C (LOW)")
                            for _, row in extreme_high.iterrows():
                                logger.warning(f"   {row['timestamp']}: {row[col]:.2f}°C (HIGH)")
                            
                            if extreme_count > 10:
                                logger.warning(f"   ... and {extreme_count - 10} more extreme temperature records")
                            
                            quality_report["problematic_records"][f"extreme_{col}"] = [
                                {"timestamp": str(row['timestamp']), "value": float(row[col])} 
                                for _, row in extreme_records.head(20).iterrows()
                            ]
                    else:
                        logger.info(f"EXTREME TEMPERATURE INFO: {data_type} - {col}: {extreme_count} extreme values - {extreme_pct:.1f}% within acceptable range")
            
            # Power status validation (should be 0 or 1)
            if 'power_status' in col:
                invalid_power_mask = (df[col] < 0) | (df[col] > 1)
                invalid_power_count = invalid_power_mask.sum()
                if invalid_power_count > 0:
                    logger.warning(f"INVALID POWER STATUS WARNING: {data_type} - {col}: {invalid_power_count} invalid power status values (should be 0 or 1)")
                    quality_report["issues"].append(f"{col} has {invalid_power_count} invalid values")
                    
                    # Show specific invalid power status records
                    if 'timestamp' in df.columns:
                        invalid_records = df[invalid_power_mask][['timestamp', col]]
                        sample_invalid = invalid_records.head(10)
                        logger.warning(f"INVALID POWER STATUS RECORDS for {col}:")
                        for _, row in sample_invalid.iterrows():
                            logger.warning(f"   {row['timestamp']}: {row[col]}")
                        if len(invalid_records) > 10:
                            logger.warning(f"   ... and {len(invalid_records) - 10} more invalid power status records")
                        
                        quality_report["problematic_records"][f"invalid_{col}"] = [
                            {"timestamp": str(row['timestamp']), "value": float(row[col])} 
                            for _, row in invalid_records.head(20).iterrows()
                        ]
            
            # General outlier detection using IQR method
            if col != 'power_status':  # Skip outlier detection for binary power status
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
                    
                    # Show statistics
                    logger.info(f"OUTLIER STATS for {col}: Q1={Q1:.3f}, Q3={Q3:.3f}, IQR={IQR:.3f}")
                    logger.info(f"OUTLIER BOUNDS for {col}: Lower={lower_bound:.3f}, Upper={upper_bound:.3f}")
                    
                    if outlier_pct > 10:  # More than 10% outliers is concerning for sensor data
                        logger.warning(f"OUTLIER WARNING: {data_type} - {col}: {outlier_count} outliers ({outlier_pct:.1f}%)")
                        quality_report["issues"].append(f"{col} has {outlier_pct:.1f}% outliers")
                        
                        # Show specific outlier records
                        if 'timestamp' in df.columns:
                            outlier_records = df[outlier_mask][['timestamp', col]].sort_values(col)
                            
                            # Show most extreme outliers
                            extreme_low = outlier_records.head(5)
                            extreme_high = outlier_records.tail(5)
                            
                            logger.warning(f"EXTREME LOW OUTLIERS for {col}:")
                            for _, row in extreme_low.iterrows():
                                logger.warning(f"   {row['timestamp']}: {row[col]:.3f}")
                            
                            logger.warning(f"EXTREME HIGH OUTLIERS for {col}:")
                            for _, row in extreme_high.iterrows():
                                logger.warning(f"   {row['timestamp']}: {row[col]:.3f}")
                            
                            if outlier_count > 10:
                                logger.warning(f"   ... and {outlier_count - 10} more outlier records")
                            
                            quality_report["problematic_records"][f"outliers_{col}"] = [
                                {"timestamp": str(row['timestamp']), "value": float(row[col])} 
                                for _, row in outlier_records.head(10).iterrows()
                            ] + [
                                {"timestamp": str(row['timestamp']), "value": float(row[col])} 
                                for _, row in outlier_records.tail(10).iterrows()
                            ]
                    else:
                        logger.info(f"OUTLIER INFO: {data_type} - {col}: {outlier_count} outliers ({outlier_pct:.1f}%) - within acceptable range")
                        
                        # Still log a few examples for reference
                        if 'timestamp' in df.columns:
                            outlier_records = df[outlier_mask][['timestamp', col]].head(5)
                            sample_outliers = [(str(row['timestamp']), float(row[col])) for _, row in outlier_records.iterrows()]
                            logger.info(f"SAMPLE OUTLIERS for {col}: {sample_outliers}")
    
    # Thermia-specific checks
    if 'supply_temp' in df.columns and 'return_temp' in df.columns:
        # Check for cases where return temperature is higher than supply temperature (unusual)
        temp_issue_mask = df['return_temp'] > df['supply_temp']
        temp_issue_count = temp_issue_mask.sum()
        if temp_issue_count > 0:
            temp_issue_pct = (temp_issue_count / len(df)) * 100
            if temp_issue_pct > 10:  # More than 10% is concerning
                logger.warning(f"TEMPERATURE LOGIC WARNING: {data_type} - {temp_issue_count} records ({temp_issue_pct:.1f}%) where return temp > supply temp")
                quality_report["issues"].append(f"Temperature logic issue: {temp_issue_count} records with return temp > supply temp")
                
                # Show specific examples
                if 'timestamp' in df.columns:
                    temp_issue_records = df[temp_issue_mask][['timestamp', 'supply_temp', 'return_temp']].head(10)
                    logger.warning(f"TEMPERATURE LOGIC ISSUE RECORDS:")
                    for _, row in temp_issue_records.iterrows():
                        logger.warning(f"   {row['timestamp']}: supply={row['supply_temp']:.2f}°C, return={row['return_temp']:.2f}°C")
                    
                    quality_report["problematic_records"]["temp_logic_issues"] = [
                        {"timestamp": str(row['timestamp']), "supply_temp": float(row['supply_temp']), "return_temp": float(row['return_temp'])} 
                        for _, row in temp_issue_records.iterrows()
                    ]
            else:
                logger.info(f"TEMPERATURE LOGIC OK: {data_type} - {temp_issue_count} records ({temp_issue_pct:.1f}%) with return temp > supply temp - within normal range")
    
    # Final status determination
    if quality_report["issues"]:
        if any("FAILED" in str(issue) or "No Thermia" in str(issue) for issue in quality_report["issues"]):
            quality_report["status"] = "FAILED"
        else:
            quality_report["status"] = "WARNING"
        
        logger.warning(f"QUALITY SUMMARY: {data_type} - Data quality issues found: {len(quality_report['issues'])}")
        for issue in quality_report["issues"]:
            logger.warning(f"   Issue: {issue}")
    else:
        logger.info(f"QUALITY PASSED: {data_type} - All data quality checks passed")
    
    return quality_report

def get_latest_timestamp(csv_file_path):
    """Reads the CSV and returns the latest timestamp."""
    logger.info(f"TIMESTAMP CHECK: Checking latest timestamp in {csv_file_path}")
    
    try:
        if not os.path.exists(csv_file_path) or os.path.getsize(csv_file_path) == 0:
            logger.info(f"FILE STATUS: Main CSV file '{csv_file_path}' not found or empty. Will attempt to fetch initial data.")
            return None
        
        df = pd.read_csv(csv_file_path)
        if 'timestamp' not in df.columns:
            logger.error(f"COLUMN ERROR: 'timestamp' column not found in {csv_file_path}")
            return None
        if df.empty:
            logger.warning(f"EMPTY FILE: CSV file {csv_file_path} is empty. No latest timestamp to read.")
            return None
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        latest_timestamp = df['timestamp'].max()
        logger.info(f"LATEST TIMESTAMP: Latest timestamp in {csv_file_path}: {latest_timestamp}")
        logger.info(f"EXISTING RECORDS: File contains {len(df)} records")
        return latest_timestamp
    except Exception as e:
        logger.error(f"READ ERROR: Error reading latest timestamp from {csv_file_path}: {e}")
        return None

def main():
    """Main function with comprehensive logging and error handling"""
    logger.info("STARTING: Thermia heat pump data update process")
    logger.info(f"MAIN FILE: {MAIN_CSV_FILE}")
    logger.info(f"EXTRACT SCRIPT: {EXTRACT_SCRIPT_PATH}")
    
    try:
        latest_timestamp = get_latest_timestamp(MAIN_CSV_FILE)
        
        days_to_fetch = 7 # Default days to fetch if no existing data
        if latest_timestamp:
            # Calculate days from latest_timestamp to now
            time_since_latest = datetime.now() - latest_timestamp
            days_to_fetch = max(1, time_since_latest.days + 1) 
            logger.info(f"TIME CALCULATION: Time since last data point: {time_since_latest}. Will fetch {days_to_fetch} day(s) of data.")
        else:
            logger.info(f"INITIAL FETCH: No existing data or unable to read latest timestamp. Fetching default {days_to_fetch} days of data.")

        # Run the ExtractHistoricData.py script
        logger.info(f"SCRIPT EXECUTION: Running {EXTRACT_SCRIPT_PATH} to fetch {days_to_fetch} day(s) of data...")
        try:
            # Ensure the script is executed with the Python interpreter from the current environment
            python_executable = sys.executable
            process = subprocess.Popen(
                [python_executable, EXTRACT_SCRIPT_PATH, '--days', str(days_to_fetch), '--output-file', TEMP_CSV_FILE],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True, # Ensures stdout and stderr are strings
                encoding='utf-8' # Specify encoding
            )
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                logger.info("SCRIPT SUCCESS: ExtractHistoricData.py executed successfully.")
                if stdout:
                    logger.info(f"SCRIPT OUTPUT:\n{stdout}")
            else:
                logger.error(f"SCRIPT ERROR: Error running ExtractHistoricData.py. Return code: {process.returncode}")
                if stdout:
                    logger.error(f"SCRIPT OUTPUT:\n{stdout}")
                if stderr:
                    logger.error(f"SCRIPT ERRORS:\n{stderr}")
                # Clean up temp file if script failed before creating it or created an empty/invalid one
                if os.path.exists(TEMP_CSV_FILE):
                     os.remove(TEMP_CSV_FILE)
                return False

        except FileNotFoundError:
            logger.error(f"SCRIPT NOT FOUND: The script {EXTRACT_SCRIPT_PATH} was not found.")
            return False
        except Exception as e:
            logger.error(f"SCRIPT EXCEPTION: An error occurred while running ExtractHistoricData.py: {e}")
            if os.path.exists(TEMP_CSV_FILE):
                 os.remove(TEMP_CSV_FILE)
            return False

        # Check if the temp file was created and has data
        if not os.path.exists(TEMP_CSV_FILE) or os.path.getsize(TEMP_CSV_FILE) == 0:
            logger.warning(f"NO TEMP DATA: Temporary data file {TEMP_CSV_FILE} was not created or is empty. No new data to append.")
            # No need to remove if it doesn't exist, but ensure it's gone if empty
            if os.path.exists(TEMP_CSV_FILE):
                os.remove(TEMP_CSV_FILE)
            return True  # Not an error, just no new data

        # Process and validate the new data
        try:
            logger.info("NEW DATA PROCESSING: Reading and processing newly fetched data")
            new_data_df = pd.read_csv(TEMP_CSV_FILE)
            if new_data_df.empty:
                logger.warning(f"EMPTY TEMP DATA: Fetched data in {TEMP_CSV_FILE} is empty. Nothing to append.")
                os.remove(TEMP_CSV_FILE)
                return True
                
            new_data_df['timestamp'] = pd.to_datetime(new_data_df['timestamp'])
            logger.info(f"NEW DATA STATS: Processed {len(new_data_df)} new records")
            logger.info(f"NEW DATA RANGE: Covers {new_data_df['timestamp'].min()} to {new_data_df['timestamp'].max()}")
            
            # Validate new data quality
            new_data_quality = validate_thermia_data_quality(new_data_df, "new Thermia data")

            if os.path.exists(MAIN_CSV_FILE) and os.path.getsize(MAIN_CSV_FILE) > 0:
                logger.info("MERGING: Merging with existing data")
                existing_data_df = pd.read_csv(MAIN_CSV_FILE)
                existing_data_df['timestamp'] = pd.to_datetime(existing_data_df['timestamp'])
                combined_df = pd.concat([existing_data_df, new_data_df], ignore_index=True)
                logger.info(f"MERGE SUCCESS: Appended {len(new_data_df)} new records to existing {len(existing_data_df)} records.")
            else:
                combined_df = new_data_df
                logger.info(f"NEW FILE: Main CSV file did not exist or was empty. Initializing with {len(new_data_df)} new records.")
                # Ensure the directory for the main CSV exists if it's the first time
                os.makedirs(os.path.dirname(MAIN_CSV_FILE), exist_ok=True)

            # Remove duplicates based on timestamp, keeping the last occurrence (new data if overlapping)
            before_dedup = len(combined_df)
            combined_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
            after_dedup = len(combined_df)
            duplicate_count = before_dedup - after_dedup
            if duplicate_count > 0:
                logger.info(f"DUPLICATES REMOVED: Removed {duplicate_count} duplicate records during merge")
            
            # Sort by timestamp
            combined_df.sort_values(by='timestamp', inplace=True)
            
            # Final quality validation
            logger.info("FINAL VALIDATION: Performing final data quality validation...")
            final_quality = validate_thermia_data_quality(combined_df, "final merged Thermia data")
            
            # Save the updated DataFrame
            try:
                combined_df.to_csv(MAIN_CSV_FILE, index=False, date_format='%Y-%m-%d %H:%M:%S')
                logger.info(f"SAVE SUCCESS: Successfully updated {MAIN_CSV_FILE}. Total records: {len(combined_df)}")
                final_latest_timestamp = combined_df['timestamp'].max()
                logger.info(f"NEW LATEST TIMESTAMP: New latest timestamp in {MAIN_CSV_FILE}: {final_latest_timestamp}")
                
                # Verify the file was saved
                try:
                    updated_file = pd.read_csv(MAIN_CSV_FILE, nrows=5)
                    logger.info(f"VERIFICATION SUCCESS: File verification successful. Columns: {', '.join(updated_file.columns)}")
                except Exception as e:
                    logger.error(f"VERIFICATION FAILED: File verification failed: {str(e)}")
                    return False
                
                # Final status report
                if final_quality["status"] == "PASSED":
                    logger.info("SYSTEM SUCCESS: THERMIA DATA UPDATE COMPLETED SUCCESSFULLY!")
                    return True
                elif final_quality["status"] == "WARNING":
                    logger.warning("SYSTEM WARNING: THERMIA DATA UPDATE COMPLETED WITH WARNINGS")
                    return True
                else:
                    logger.error("SYSTEM FAILED: THERMIA DATA UPDATE COMPLETED BUT WITH QUALITY ISSUES")
                    return False
                    
            except Exception as e:
                logger.error(f"SAVE ERROR: Error processing or saving data: {e}")
                return False
                
        except Exception as e:
            logger.error(f"PROCESSING ERROR: Error processing new data: {e}")
            return False
        finally:
            # Clean up the temporary file
            if os.path.exists(TEMP_CSV_FILE):
                os.remove(TEMP_CSV_FILE)
                logger.info(f"CLEANUP: Removed temporary file {TEMP_CSV_FILE}.")
                
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