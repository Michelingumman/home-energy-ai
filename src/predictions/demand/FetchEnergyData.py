import os
import pandas as pd
import requests
import json
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import logging
import sys
import numpy as np
from dateutil.parser import parse

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('src/predictions/demand/logs/energy_data_fetch.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv('api.env')
TIBBER_TOKEN = os.getenv('TIBBER_TOKEN')

if not TIBBER_TOKEN:
    logger.error("TIBBER_TOKEN not found in api.env file!")
    sys.exit(1)

# File paths
CSV_FILE_PATH = 'data/processed/villamichelin/VillamichelinEnergyData.csv'

def fetch_tibber_data_paginated(after=None, first=1000):
    """
    Fetch a single page of consumption and production data from Tibber API
    
    Args:
        after (str, optional): Cursor to fetch data after
        first (int): Number of records to fetch
    
    Returns:
        tuple: (dict of consumption and production nodes, end cursor, has next page)
    """
    url = 'https://api.tibber.com/v1-beta/gql'
    
    headers = {
        'Authorization': f'Bearer {TIBBER_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    # Build the GraphQL query with pagination that includes both consumption and production
    if after:
        query = """
        {
          viewer {
            homes {
              consumption(resolution: HOURLY, first: %d, after: "%s") {
                pageInfo {
                  endCursor
                  hasNextPage
                }
                edges {
                  cursor
                  node {
                    from
                    to
                    cost
                    unitPrice
                    unitPriceVAT
                    consumption
                    consumptionUnit
                  }
                }
              }
              production(resolution: HOURLY, first: %d, after: "%s") {
                pageInfo {
                  endCursor
                  hasNextPage
                }
                edges {
                  cursor
                  node {
                    from
                    to
                    profit
                    unitPrice
                    production
                    productionUnit
                  }
                }
              }
            }
          }
        }
        """ % (first, after, first, after)
    else:
        query = """
        {
          viewer {
            homes {
              consumption(resolution: HOURLY, first: %d) {
                pageInfo {
                  endCursor
                  hasNextPage
                }
                edges {
                  cursor
                  node {
                    from
                    to
                    cost
                    unitPrice
                    unitPriceVAT
                    consumption
                    consumptionUnit
                  }
                }
              }
              production(resolution: HOURLY, first: %d) {
                pageInfo {
                  endCursor
                  hasNextPage
                }
                edges {
                  cursor
                  node {
                    from
                    to
                    profit
                    unitPrice
                    production
                    productionUnit
                  }
                }
              }
            }
          }
        }
        """ % (first, first)
    
    data = {'query': query}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract data from the response
        data_nodes = {'consumption': [], 'production': []}
        end_cursor = {'consumption': None, 'production': None}
        has_next_page = {'consumption': False, 'production': False}
        
        if ('data' in result and 
            'viewer' in result['data'] and 
            'homes' in result['data']['viewer'] and 
            result['data']['viewer']['homes']):
            
            homes = result['data']['viewer']['homes']
            
            for home in homes:
                # Process consumption data
                if 'consumption' in home and 'edges' in home['consumption']:
                    # Get page info
                    if 'pageInfo' in home['consumption']:
                        end_cursor['consumption'] = home['consumption']['pageInfo'].get('endCursor')
                        has_next_page['consumption'] = home['consumption']['pageInfo'].get('hasNextPage', False)
                    
                    # Get consumption nodes
                    for edge in home['consumption']['edges']:
                        if 'node' in edge:
                            data_nodes['consumption'].append(edge['node'])
                
                # Process production data
                if 'production' in home and 'edges' in home['production']:
                    # Get page info
                    if 'pageInfo' in home['production']:
                        end_cursor['production'] = home['production']['pageInfo'].get('endCursor')
                        has_next_page['production'] = home['production']['pageInfo'].get('hasNextPage', False)
                    
                    # Get production nodes
                    for edge in home['production']['edges']:
                        if 'node' in edge:
                            data_nodes['production'].append(edge['node'])
            
        # Determine overall pagination status (continue if either has more pages)
        overall_end_cursor = after
        if end_cursor['consumption']:
            overall_end_cursor = end_cursor['consumption']
        elif end_cursor['production']:
            overall_end_cursor = end_cursor['production']
            
        overall_has_next_page = has_next_page['consumption'] or has_next_page['production']
            
        return data_nodes, overall_end_cursor, overall_has_next_page
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from Tibber API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response content: {e.response.text}")
        return {'consumption': [], 'production': []}, None, False

def fetch_all_tibber_data():
    """
    Fetch all available consumption and production data from Tibber API with pagination
    
    Returns:
        dict: Dictionary with lists of consumption and production data points
    """
    all_data_nodes = {'consumption': [], 'production': []}
    after_cursor = None
    has_more_pages = True
    page_count = 0
    
    logger.info("Fetching all available historical consumption and production data")
    
    # Use cursor-based pagination 
    while has_more_pages:
        page_count += 1
        logger.info(f"Fetching page {page_count} of data...")
        
        data_nodes, end_cursor, has_next_page = fetch_tibber_data_paginated(
            after=after_cursor,
            first=1000
        )
        
        if not data_nodes['consumption'] and not data_nodes['production']:
            logger.warning(f"No data found on page {page_count}")
            break
        
        # Add all nodes
        all_data_nodes['consumption'].extend(data_nodes['consumption'])
        all_data_nodes['production'].extend(data_nodes['production'])
        
        # Update for next page
        after_cursor = end_cursor
        has_more_pages = has_next_page
        
        # Log progress
        logger.info(f"Retrieved {len(data_nodes['consumption'])} consumption records and {len(data_nodes['production'])} production records on page {page_count}")
        logger.info(f"Total records so far: {len(all_data_nodes['consumption'])} consumption, {len(all_data_nodes['production'])} production")
        
        # Break if we've reached the maximum pages to fetch (safety measure)
        if page_count >= 100:  # Limit to 100 pages as a safety measure
            logger.warning("Reached maximum page count (100). Breaking pagination loop.")
            break
    
    logger.info(f"Fetched a total of {len(all_data_nodes['consumption'])} consumption records and {len(all_data_nodes['production'])} production records from {page_count} pages")
    return all_data_nodes

def parse_timestamp_strip_timezone(timestamp_str):
    """
    Parse a timestamp string to a datetime object, converting to local time (UTC+2) and removing timezone info
    
    Args:
        timestamp_str (str): Timestamp string to parse
        
    Returns:
        datetime: Naive datetime (no timezone) or None on error
    """
    if not timestamp_str:
        logger.error("Received empty timestamp string")
        return None
        
    try:
        # Parse with dateutil into a Python datetime
        dt = parse(timestamp_str)
        
        # If no timezone, assume UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
            
        # Convert to local time (UTC+2 for CEST)
        local_tz = timezone(timedelta(hours=2))
        dt_local = dt.astimezone(local_tz)
            
        # Then remove timezone info completely
        dt_naive = dt_local.replace(tzinfo=None)
        
        return dt_naive
        
    except Exception as e:
        logger.error(f"Failed to parse timestamp '{timestamp_str}': {e}")
        return None

def fetch_recent_tibber_data(from_datetime):
    """
    Fetch only recent data from a certain point in time
    Uses an optimized approach to avoid fetching all historical data
    
    Args:
        from_datetime (datetime): Datetime to fetch data from (naive)
    
    Returns:
        dict: Dictionary with lists of consumption and production data points
    """
    # Convert to UTC for API request (required for comparison with API data)
    from_datetime_utc = from_datetime.replace(tzinfo=timezone.utc)
        
    # Convert to string for logging
    from_date_str = from_datetime.strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Fetching new data since: {from_date_str}")
    
    # Target date (used for checking when to start collecting)
    target_date = from_datetime
    
    # Initialize variables
    new_data_nodes = {'consumption': [], 'production': []}
    after_cursor = None
    has_more_pages = True
    page_count = 0
    found_new_data = False
    
    # We use three phases for fetching:
    # 1. Skip phase: Quickly skip through pages until we find data around our target date
    # 2. Collection phase: Collect all data that's newer than our target date
    # 3. Finalize phase: Finish collection once we've collected all new data
    
    # ---- PHASE 1: Fast skipping to find recent data ----
    # First, try to skip quickly through the data until we get close to our target date
    logger.info("PHASE 1: Skipping through older data to find recent records...")
    
    # We'll use a binary search-like approach to skip quickly
    # Start by trying big jumps (1000 records at a time)
    records_per_request = 1000
    
    while has_more_pages and not found_new_data:
        page_count += 1
        
        data_nodes, end_cursor, has_next_page = fetch_tibber_data_paginated(
            after=after_cursor,
            first=records_per_request
        )
        
        if not data_nodes['consumption'] and not data_nodes['production']:
            logger.warning(f"No data found on page {page_count}")
            break
        
        # Check consumption data
        if data_nodes['consumption']:
            # Check first and last record in this batch to see if we're close to target date
            first_node = data_nodes['consumption'][0]
            last_node = data_nodes['consumption'][-1]
            
            # Parse dates, stripping timezone info
            first_date = parse_timestamp_strip_timezone(first_node['from'])
            last_date = parse_timestamp_strip_timezone(last_node['from'])
            
            # If last record is still before our target date, continue skipping
            if last_date < target_date:
                after_cursor = end_cursor
                has_more_pages = has_next_page
                continue
                
            # If first record is after our target date, we've found new data
            if first_date >= target_date:
                logger.info(f"All consumption records on page {page_count} are after target date. Starting collection.")
                new_data_nodes['consumption'].extend(data_nodes['consumption'])
                found_new_data = True
            else:
                # We're in the transition page where some records are before and some after
                # Filter and keep only records that are >= target date
                logger.info(f"Found transition page {page_count} with some consumption records after target date.")
                for node in data_nodes['consumption']:
                    # Parse timestamp, strip timezone
                    node_date = parse_timestamp_strip_timezone(node['from'])
                    if node_date is not None and node_date >= target_date:
                        new_data_nodes['consumption'].append(node)
                found_new_data = True
        
        # Check production data
        if data_nodes['production']:
            # Check first and last record in this batch to see if we're close to target date
            first_node = data_nodes['production'][0]
            last_node = data_nodes['production'][-1]
            
            # Parse dates, stripping timezone info
            first_date = parse_timestamp_strip_timezone(first_node['from'])
            last_date = parse_timestamp_strip_timezone(last_node['from'])
            
            # If last record is still before our target date, continue skipping (unless we already found consumption data)
            if last_date < target_date and not found_new_data:
                after_cursor = end_cursor
                has_more_pages = has_next_page
                continue
                
            # If first record is after our target date, we've found new data
            if first_date >= target_date:
                logger.info(f"All production records on page {page_count} are after target date. Starting collection.")
                new_data_nodes['production'].extend(data_nodes['production'])
                found_new_data = True
            else:
                # We're in the transition page where some records are before and some after
                # Filter and keep only records that are >= target date
                logger.info(f"Found transition page {page_count} with some production records after target date.")
                for node in data_nodes['production']:
                    # Parse timestamp, strip timezone
                    node_date = parse_timestamp_strip_timezone(node['from'])
                    if node_date is not None and node_date >= target_date:
                        new_data_nodes['production'].append(node)
                found_new_data = True
        
        # Update for next page
        after_cursor = end_cursor
        has_more_pages = has_next_page
    
    # ---- PHASE 2: Continue collecting remaining new data ----
    if found_new_data and has_more_pages:
        logger.info("PHASE 2: Collecting remaining new data...")
        
        while has_more_pages:
            page_count += 1
            
            data_nodes, end_cursor, has_next_page = fetch_tibber_data_paginated(
                after=after_cursor,
                first=records_per_request
            )
            
            if not data_nodes['consumption'] and not data_nodes['production']:
                logger.warning(f"No data found on collection page {page_count}")
                break
                
            # Add all nodes (all must be after our target date)
            new_data_nodes['consumption'].extend(data_nodes['consumption'])
            new_data_nodes['production'].extend(data_nodes['production'])
            
            # Update for next page
            after_cursor = end_cursor
            has_more_pages = has_next_page
            
            # Log progress
            logger.info(f"Collected {len(data_nodes['consumption'])} consumption records and {len(data_nodes['production'])} production records on page {page_count}")
            logger.info(f"Total new records so far: {len(new_data_nodes['consumption'])} consumption, {len(new_data_nodes['production'])} production")
            
            # Break if we've reached the maximum pages to fetch (safety measure)
            if page_count >= 100:  # Limit to 100 pages as a safety measure
                logger.warning("Reached maximum page count (100). Breaking pagination loop.")
                break
    
    # Check and log results
    if not found_new_data:
        logger.info("No new data found since the target date.")
    else:
        logger.info(f"Fetched a total of {len(new_data_nodes['consumption'])} new consumption records and {len(new_data_nodes['production'])} new production records")
    
    return new_data_nodes

def process_energy_data(data):
    """
    Process consumption and production data into a pandas DataFrame
    
    Args:
        data (dict): Dictionary with lists of consumption and production data from Tibber API
    
    Returns:
        pd.DataFrame: Processed data
    """
    if not data or (not data['consumption'] and not data['production']):
        return pd.DataFrame()
    
    # Process consumption data
    consumption_list = []
    for item in data['consumption']:
        if item and 'from' in item and 'consumption' in item:
            try:
                # Parse timestamp and strip timezone
                dt = parse_timestamp_strip_timezone(item['from'])
                if dt is None:
                    logger.warning(f"Skipping consumption data point with invalid timestamp: {item['from']}")
                    continue
                
                consumption_list.append({
                    'timestamp': dt,
                    'consumption': float(item.get('consumption', 0)),
                    'consumption_cost': float(item.get('cost', 0)),
                    'consumption_unit_price': float(item.get('unitPrice', 0)),
                    'consumption_unit_price_vat': float(item.get('unitPriceVAT', 0)),
                    'consumption_unit': item.get('consumptionUnit', 'kWh'),
                    'production': 0,  # Default value for merging
                    'production_profit': 0,  # Default value for merging
                    'production_unit_price': 0,  # Default value for merging
                    'production_unit': 'kWh'  # Default value for merging
                })
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing consumption data point: {e}, data: {item}")
    
    # Process production data
    production_list = []
    for item in data['production']:
        if item and 'from' in item and 'production' in item:
            try:
                # Parse timestamp and strip timezone
                dt = parse_timestamp_strip_timezone(item['from'])
                if dt is None:
                    logger.warning(f"Skipping production data point with invalid timestamp: {item['from']}")
                    continue
                
                production_list.append({
                    'timestamp': dt,
                    'consumption': 0,  # Default value for merging
                    'consumption_cost': 0,  # Default value for merging
                    'consumption_unit_price': 0,  # Default value for merging
                    'consumption_unit_price_vat': 0,  # Default value for merging
                    'consumption_unit': 'kWh',  # Default value for merging
                    'production': float(item.get('production', 0)),
                    'production_profit': float(item.get('profit', 0)),
                    'production_unit_price': float(item.get('unitPrice', 0)),
                    'production_unit': item.get('productionUnit', 'kWh')
                })
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing production data point: {e}, data: {item}")
    
    # Combine consumption and production data
    data_list = consumption_list + production_list
    
    if not data_list:
        return pd.DataFrame()
        
    df = pd.DataFrame(data_list)
    
    # Combine records with the same timestamp (merge consumption and production data)
    df = df.groupby('timestamp').agg({
        'consumption': 'sum',
        'consumption_cost': 'sum',
        'consumption_unit_price': 'mean',
        'consumption_unit_price_vat': 'mean',
        'consumption_unit': 'first',
        'production': 'sum',
        'production_profit': 'sum',
        'production_unit_price': 'mean',
        'production_unit': 'first'
    }).reset_index()
    
    # Remove duplicates here before sorting
    before_dedup = len(df)
    df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
    after_dedup = len(df)
    if before_dedup != after_dedup:
        logger.info(f"Removed {before_dedup - after_dedup} duplicate entries during processing")
    
    # Sort by timestamp
    df.sort_values('timestamp', inplace=True)
    return df

def format_datetime_for_tibber(dt):
    """
    Format a datetime object to the format expected by the Tibber API.
    Converts to UTC and formats as ISO-8601 with Z suffix.
    
    Args:
        dt (datetime): Datetime object to format
        
    Returns:
        str: Formatted date string
    """
    # If the datetime doesn't have a timezone, assume UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        # Otherwise, convert to UTC
        dt = dt.astimezone(timezone.utc)
    
    # Format as ISO with Z (Zulu time / UTC)
    return dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')

def validate_energy_data_quality(df, data_type="energy"):
    """
    Comprehensive data quality validation for energy data.
    
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
        "energy_columns": [],
        "problematic_records": {}
    }
    
    # Basic info
    logger.info(f"{data_type} - Total records: {len(df)}")
    
    # Check for energy columns
    expected_energy_cols = ["consumption", "production", "consumption_cost", "production_profit"]
    found_energy_cols = [col for col in expected_energy_cols if col in df.columns]
    quality_report["energy_columns"] = found_energy_cols
    
    if not found_energy_cols:
        logger.error(f"CRITICAL: {data_type} - No energy data columns found!")
        quality_report["status"] = "FAILED"
        quality_report["issues"].append("No energy data columns found")
        return quality_report
    else:
        logger.info(f"COLUMNS FOUND: {data_type} - Found energy columns: {found_energy_cols}")
    
    # Date range analysis
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            start_date = df['timestamp'].min()
            end_date = df['timestamp'].max()
            quality_report["date_range"] = f"{start_date} to {end_date}"
            logger.info(f"{data_type} - Date range: {start_date} to {end_date}")
            
            # Gap detection for hourly energy data
            expected_hours = (end_date - start_date).total_seconds() / 3600 + 1
            actual_hours = len(df)
            gap_count = int(expected_hours - actual_hours)
            
            if gap_count > 0:
                quality_report["gaps"] = [f"{gap_count} missing hours"]
                logger.warning(f"GAP WARNING: {data_type} - Found {gap_count} missing hours (expected {int(expected_hours)}, got {actual_hours})")
                quality_report["issues"].append(f"Missing {gap_count} hours of data")
                
                # Find specific missing hours
                expected_range = pd.date_range(start=start_date, end=end_date, freq='H')
                missing_hours = expected_range.difference(df['timestamp'])
                sample_missing = missing_hours[:30] if len(missing_hours) > 30 else missing_hours
                logger.warning(f"MISSING HOUR TIMESTAMPS: Sample missing hours: {sample_missing.tolist()}")
                if len(missing_hours) > 30:
                    logger.warning(f"... and {len(missing_hours) - 30} more missing hour timestamps")
                quality_report["problematic_records"]["missing_hours"] = missing_hours[:100].tolist()
            else:
                logger.info(f"GAP CHECK OK: {data_type} - No gaps detected in hourly data")
                
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
    
    # Missing values analysis for energy columns
    missing_summary = df[found_energy_cols].isnull().sum()
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
            
            if pct > 10:  # More than 10% missing is concerning for energy data
                quality_report["issues"].append(f"{col} has {pct:.1f}% missing values")
    else:
        logger.info(f"MISSING VALUES OK: {data_type} - No missing values found in energy columns")
    
    # Value validation for energy columns
    for col in found_energy_cols:
        if col in df.columns and not df[col].empty:
            # Check for negative values (energy should be non-negative)
            negative_mask = df[col] < 0
            negative_count = negative_mask.sum()
            if negative_count > 0:
                logger.warning(f"NEGATIVE VALUES WARNING: {data_type} - {col}: {negative_count} negative values found")
                quality_report["issues"].append(f"{col} has {negative_count} negative values")
                
                # Show specific negative value records
                if 'timestamp' in df.columns:
                    negative_records = df[negative_mask][['timestamp', col]]
                    sample_negative = negative_records.head(10)
                    logger.warning(f"NEGATIVE VALUE RECORDS for {col}:")
                    for _, row in sample_negative.iterrows():
                        logger.warning(f"   {row['timestamp']}: {row[col]}")
                    if len(negative_records) > 10:
                        logger.warning(f"   ... and {len(negative_records) - 10} more negative value records")
                    
                    quality_report["problematic_records"][f"negative_{col}"] = [
                        {"timestamp": str(row['timestamp']), "value": float(row[col])} 
                        for _, row in negative_records.head(20).iterrows()
                    ]
            
            # Check for extremely high values (might indicate data errors)
            if 'consumption' in col or 'production' in col:
                # Energy values above 100 kWh per hour are very unusual for residential
                extreme_mask = df[col] > 100
                extreme_count = extreme_mask.sum()
                if extreme_count > 0:
                    extreme_pct = (extreme_count / len(df)) * 100
                    if extreme_pct > 1:  # More than 1% extreme values is suspicious
                        logger.warning(f"EXTREME VALUES WARNING: {data_type} - {col}: {extreme_count} extremely high values (>{100} kWh)")
                        quality_report["issues"].append(f"{col} has {extreme_count} extremely high values")
                        
                        # Show specific extreme value records
                        if 'timestamp' in df.columns:
                            extreme_records = df[extreme_mask][['timestamp', col]].sort_values(col, ascending=False)
                            sample_extreme = extreme_records.head(10)
                            logger.warning(f"EXTREME HIGH VALUE RECORDS for {col}:")
                            for _, row in sample_extreme.iterrows():
                                logger.warning(f"   {row['timestamp']}: {row[col]:.3f} kWh")
                            if len(extreme_records) > 10:
                                logger.warning(f"   ... and {len(extreme_records) - 10} more extreme value records")
                            
                            quality_report["problematic_records"][f"extreme_{col}"] = [
                                {"timestamp": str(row['timestamp']), "value": float(row[col])} 
                                for _, row in extreme_records.head(20).iterrows()
                            ]
                    else:
                        logger.info(f"EXTREME VALUES INFO: {data_type} - {col}: {extreme_count} high values (>{100} kWh) - {extreme_pct:.1f}% within normal range")
            
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
                
                # Show statistics
                logger.info(f"OUTLIER STATS for {col}: Q1={Q1:.3f}, Q3={Q3:.3f}, IQR={IQR:.3f}")
                logger.info(f"OUTLIER BOUNDS for {col}: Lower={lower_bound:.3f}, Upper={upper_bound:.3f}")
                
                if outlier_pct > 15:  # More than 15% outliers is concerning for energy data
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
    
    # Energy-specific checks
    if 'consumption' in df.columns and 'production' in df.columns:
        # Check for simultaneous high consumption and production (unusual pattern)
        high_both_mask = (df['consumption'] > df['consumption'].quantile(0.9)) & (df['production'] > df['production'].quantile(0.9))
        high_both_count = high_both_mask.sum()
        if high_both_count > 0:
            high_both_pct = (high_both_count / len(df)) * 100
            if high_both_pct > 5:  # More than 5% is unusual
                logger.warning(f"ENERGY PATTERN WARNING: {data_type} - {high_both_count} records ({high_both_pct:.1f}%) with simultaneously high consumption and production")
                quality_report["issues"].append(f"Unusual energy pattern: {high_both_count} records with high consumption and production")
                
                # Show specific examples
                if 'timestamp' in df.columns:
                    high_both_records = df[high_both_mask][['timestamp', 'consumption', 'production']].head(10)
                    logger.warning(f"HIGH CONSUMPTION & PRODUCTION RECORDS:")
                    for _, row in high_both_records.iterrows():
                        logger.warning(f"   {row['timestamp']}: consumption={row['consumption']:.3f}, production={row['production']:.3f}")
                    
                    quality_report["problematic_records"]["high_both_energy"] = [
                        {"timestamp": str(row['timestamp']), "consumption": float(row['consumption']), "production": float(row['production'])} 
                        for _, row in high_both_records.iterrows()
                    ]
            else:
                logger.info(f"ENERGY PATTERN OK: {data_type} - {high_both_count} records ({high_both_pct:.1f}%) with high consumption and production - normal range")
    
    # Final status determination
    if quality_report["issues"]:
        if any("FAILED" in str(issue) or "No energy" in str(issue) for issue in quality_report["issues"]):
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
    logger.info("STARTING: Energy consumption and production data update")
    logger.info(f"FILE PATH: Working with file: {CSV_FILE_PATH}")
    
    try:
        # Check if the CSV file exists and has data
        if os.path.exists(CSV_FILE_PATH) and os.path.getsize(CSV_FILE_PATH) > 0:
            # Read existing data
            logger.info("FILE LOADED: Reading existing energy data")
            existing_data = pd.read_csv(CSV_FILE_PATH)
            
            # Convert timestamp to datetime without timezone
            existing_data['timestamp'] = pd.to_datetime(existing_data['timestamp'])
            
            # Get the latest timestamp and add 1 hour to avoid duplicates
            latest_timestamp = existing_data['timestamp'].max()
            logger.info(f"EXISTING DATA: Latest timestamp in existing data: {latest_timestamp}")
            logger.info(f"EXISTING DATA: Current file contains {len(existing_data)} records")
            
            # Add 1 hour to avoid duplicates
            next_timestamp = latest_timestamp + timedelta(hours=1)
            logger.info(f"FETCH RANGE: Fetching new data from {next_timestamp} onwards")
            
            # Use the optimized approach to fetch only recent data
            logger.info("API CALL: Fetching recent energy data from Tibber API")
            energy_data = fetch_recent_tibber_data(next_timestamp)
            
            if not energy_data['consumption'] and not energy_data['production']:
                logger.info("NO UPDATE NEEDED: No new data available from API")
                
                # Still validate existing data
                logger.info("VALIDATION: Validating existing energy data...")
                existing_quality = validate_energy_data_quality(existing_data, "existing energy data")
                if existing_quality["status"] == "PASSED":
                    logger.info("VALIDATION PASSED: EXISTING DATA VALIDATION SUCCESSFUL")
                    return True
                elif existing_quality["status"] == "WARNING":
                    logger.warning("VALIDATION WARNING: EXISTING DATA VALIDATION FOUND ISSUES BUT CONTINUING")
                    return True  # Accept warnings for energy data - they're often legitimate
                else:
                    logger.error("VALIDATION FAILED: EXISTING DATA VALIDATION FAILED")
                    return False
            
            logger.info(f"API SUCCESS: Retrieved {len(energy_data['consumption'])} consumption records and {len(energy_data['production'])} production records")
            
            # Process new data
            logger.info("PROCESSING: Processing newly fetched energy data")
            new_data_df = process_energy_data(energy_data)
            
            if new_data_df.empty:
                logger.info("NO DATA: No new data to add after processing")
                return True
                
            # Log the date range of new data
            min_date = new_data_df['timestamp'].min()
            max_date = new_data_df['timestamp'].max()
            logger.info(f"NEW DATA RANGE: New data covers period: {min_date} to {max_date}")
            logger.info(f"NEW DATA STATS: {len(new_data_df)} new records processed")
            
            # Validate new data quality
            new_data_quality = validate_energy_data_quality(new_data_df, "new energy data")
            
            # Check if existing data has all the required columns
            required_columns = ['consumption', 'consumption_cost', 'consumption_unit_price',
                               'consumption_unit_price_vat', 'consumption_unit',
                               'production', 'production_profit', 'production_unit_price',
                               'production_unit']
            
            # Add missing columns with default values if needed
            missing_cols = []
            for col in required_columns:
                if col not in existing_data.columns:
                    if 'cost' in col or 'price' in col or 'production' in col or 'consumption' in col:
                        existing_data[col] = 0.0
                    else:
                        existing_data[col] = 'kWh'
                    missing_cols.append(col)
            
            if missing_cols:
                logger.info(f"COLUMN UPDATE: Added missing columns with default values: {missing_cols}")
            
            # Merge with existing data
            logger.info(f"MERGING: Adding {len(new_data_df)} new data points to existing {len(existing_data)} records")
            combined_df = pd.concat([existing_data, new_data_df], ignore_index=True)
            
            # Remove any duplicates based on timestamp
            before_dedup = len(combined_df)
            combined_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
            after_dedup = len(combined_df)
            if before_dedup != after_dedup:
                duplicate_count = before_dedup - after_dedup
                logger.info(f"DUPLICATES REMOVED: Removed {duplicate_count} duplicate entries during merge")
            
            # Sort by timestamp
            combined_df.sort_values('timestamp', inplace=True)
            
            # Final quality validation
            logger.info("FINAL VALIDATION: Performing final data quality validation...")
            final_quality = validate_energy_data_quality(combined_df, "final merged energy data")
            
            # Save to CSV
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)
                combined_df.to_csv(CSV_FILE_PATH, index=False)
                logger.info(f"SAVE SUCCESS: Updated energy data saved to {CSV_FILE_PATH}")
                logger.info(f"FILE STATS: Total records in the updated file: {len(combined_df)}")
                logger.info(f"DATE RANGE: Data now covers: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
                
                # Verify the file was saved
                try:
                    updated_file = pd.read_csv(CSV_FILE_PATH, nrows=5)
                    logger.info(f"VERIFICATION SUCCESS: File verification successful. Columns: {', '.join(updated_file.columns)}")
                except Exception as e:
                    logger.error(f"VERIFICATION FAILED: File verification failed: {str(e)}")
                    return False
                
                # Final status report
                if final_quality["status"] == "PASSED":
                    logger.info("SYSTEM SUCCESS: ENERGY DATA UPDATE COMPLETED SUCCESSFULLY!")
                    return True
                elif final_quality["status"] == "WARNING":
                    logger.warning("SYSTEM WARNING: ENERGY DATA UPDATE COMPLETED WITH WARNINGS")
                    return True
                else:
                    logger.error("SYSTEM FAILED: ENERGY DATA UPDATE COMPLETED BUT WITH QUALITY ISSUES")
                    return False
                    
            except Exception as e:
                logger.error(f"SAVE ERROR: Error saving updated energy data: {str(e)}")
                return False
            
        else:
            # Fetch all available data
            logger.info("NEW FILE: No existing data found. Fetching all available consumption and production data")
            
            logger.info("API CALL: Fetching all energy data from Tibber API")
            energy_data = fetch_all_tibber_data()
            
            if not energy_data['consumption'] and not energy_data['production']:
                logger.error("API ERROR: No data available from Tibber API")
                return False
                
            logger.info(f"API SUCCESS: Retrieved {len(energy_data['consumption'])} consumption records and {len(energy_data['production'])} production records")
                
            # Process data
            logger.info("PROCESSING: Processing all fetched energy data")
            data_df = process_energy_data(energy_data)
            
            if data_df.empty:
                logger.error("PROCESSING ERROR: Failed to process energy data")
                return False
                
            logger.info(f"PROCESSING SUCCESS: Processed {len(data_df)} total records")
            logger.info(f"DATE RANGE: Data covers period: {data_df['timestamp'].min()} to {data_df['timestamp'].max()}")
            
            # Validate data quality
            quality_report = validate_energy_data_quality(data_df, "new energy data")
                
            # Make sure the directory exists
            try:
                os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)
                    
                # Save to CSV
                data_df.to_csv(CSV_FILE_PATH, index=False)
                logger.info(f"SAVE SUCCESS: Energy data saved to {CSV_FILE_PATH}")
                logger.info(f"FILE STATS: Total records: {len(data_df)}")
                
                # Verify the file was saved
                try:
                    created_file = pd.read_csv(CSV_FILE_PATH, nrows=5)
                    logger.info(f"VERIFICATION SUCCESS: File verification successful. Columns: {', '.join(created_file.columns)}")
                except Exception as e:
                    logger.error(f"VERIFICATION FAILED: File verification failed: {str(e)}")
                    return False
                
                # Final status report
                if quality_report["status"] == "PASSED":
                    logger.info("SYSTEM SUCCESS: ENERGY DATA CREATION COMPLETED SUCCESSFULLY!")
                    return True
                elif quality_report["status"] == "WARNING":
                    logger.warning("SYSTEM WARNING: ENERGY DATA CREATION COMPLETED WITH WARNINGS")
                    return True
                else:
                    logger.error("SYSTEM FAILED: ENERGY DATA CREATION COMPLETED BUT WITH QUALITY ISSUES")
                    return False
                    
            except Exception as e:
                logger.error(f"SAVE ERROR: Error saving energy data to CSV: {str(e)}")
                return False
            
    except Exception as e:
        logger.error(f"CRITICAL ERROR: Fatal error in main function: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("SCRIPT SUCCESS: Script completed successfully")
        exit(0)
    else:
        logger.error("SCRIPT FAILED: Script completed with errors")
        exit(1)
