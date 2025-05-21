import os
import pandas as pd
import requests
import json
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import logging
import sys
from dateutil.parser import parse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
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

def check_data_quality(df):
    """
    Check data quality for duplicates and NaN values
    
    Args:
        df (pd.DataFrame): DataFrame to check
    """
    logger.info("Checking data quality...")
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['timestamp']).sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate timestamps in the data")
    else:
        logger.info("No duplicate timestamps found")
    
    # Check for NaN values
    nan_counts = df.isna().sum()
    total_rows = len(df)
    
    for column, nan_count in nan_counts.items():
        if nan_count > 0:
            percent = (nan_count / total_rows) * 100
            logger.warning(f"Column '{column}' has {nan_count} NaN values ({percent:.2f}%)")
    
    if nan_counts.sum() == 0:
        logger.info("No NaN values found in the data")
    
    # Check for zero consumption and production values
    zero_consumption = (df['consumption'] == 0).sum()
    if zero_consumption > 0:
        percent = (zero_consumption / total_rows) * 100
        logger.info(f"Found {zero_consumption} records with zero consumption ({percent:.2f}%)")
    
    zero_production = (df['production'] == 0).sum()
    if zero_production > 0:
        percent = (zero_production / total_rows) * 100
        logger.info(f"Found {zero_production} records with zero production ({percent:.2f}%)")
    
    # Check for missing hours (gaps in time series)
    df_sorted = df.sort_values('timestamp')
    time_diffs = df_sorted['timestamp'].diff().dropna()
    
    # Expected difference for hourly data is 1 hour
    expected_diff = pd.Timedelta(hours=1)
    gaps = time_diffs[time_diffs > expected_diff]
    
    if len(gaps) > 0:
        logger.warning(f"Found {len(gaps)} gaps in the time series data")
        for i in range(len(gaps)):
            idx = gaps.index[i]
            prev_time = df_sorted.loc[idx-1, 'timestamp']
            curr_time = df_sorted.loc[idx, 'timestamp']
            gap_size = curr_time - prev_time
            logger.warning(f"Gap of {gap_size} between {prev_time} and {curr_time}")

def main():
    logger.info("Starting to fetch consumption and production data from Tibber API")
    logger.warning("Using simple approach: timestamps stored without timezone information")
    
    try:
        # Check if the CSV file exists and has data
        if os.path.exists(CSV_FILE_PATH) and os.path.getsize(CSV_FILE_PATH) > 0:
            # Read existing data
            logger.info("Reading existing energy data")
            existing_data = pd.read_csv(CSV_FILE_PATH)
            
            # Convert timestamp to datetime without timezone
            existing_data['timestamp'] = pd.to_datetime(existing_data['timestamp'])
            
            # Get the latest timestamp and add 1 hour to avoid duplicates
            latest_timestamp = existing_data['timestamp'].max()
            logger.info(f"Latest timestamp in existing data: {latest_timestamp}")
            
            # Add 1 hour to avoid duplicates
            next_timestamp = latest_timestamp + timedelta(hours=1)
            
            # Use the optimized approach to fetch only recent data
            energy_data = fetch_recent_tibber_data(next_timestamp)
            
            if not energy_data['consumption'] and not energy_data['production']:
                logger.info("No new data available")
                return
            
            # Process new data
            new_data_df = process_energy_data(energy_data)
            
            if new_data_df.empty:
                logger.info("No new data to add after processing")
                return
                
            # Log the date range of new data
            min_date = new_data_df['timestamp'].min()
            max_date = new_data_df['timestamp'].max()
            logger.info(f"New data covers period: {min_date} to {max_date}")
            
            # Check if existing data has all the required columns
            required_columns = ['consumption', 'consumption_cost', 'consumption_unit_price',
                               'consumption_unit_price_vat', 'consumption_unit',
                               'production', 'production_profit', 'production_unit_price',
                               'production_unit']
            
            # Add missing columns with default values if needed
            for col in required_columns:
                if col not in existing_data.columns:
                    if 'cost' in col or 'price' in col or 'production' in col or 'consumption' in col:
                        existing_data[col] = 0.0
                    else:
                        existing_data[col] = 'kWh'
            
            # Merge with existing data
            logger.info(f"Adding {len(new_data_df)} new data points")
            combined_df = pd.concat([existing_data, new_data_df], ignore_index=True)
            
            # Remove any duplicates based on timestamp
            before_dedup = len(combined_df)
            combined_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
            after_dedup = len(combined_df)
            if before_dedup != after_dedup:
                logger.info(f"Removed {before_dedup - after_dedup} duplicate entries")
            
            # Sort by timestamp
            combined_df.sort_values('timestamp', inplace=True)
            
            # Save to CSV
            combined_df.to_csv(CSV_FILE_PATH, index=False)
            logger.info(f"Updated energy data saved to {CSV_FILE_PATH}")
            logger.info(f"Total records in the updated file: {len(combined_df)}")
            logger.info(f"Data now covers: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
            
            # Check for data quality issues
            check_data_quality(combined_df)
            
        else:
            # Fetch all available data
            logger.info("No existing data found. Fetching all available consumption and production data")
            energy_data = fetch_all_tibber_data()
            
            if not energy_data['consumption'] and not energy_data['production']:
                logger.error("No data available from Tibber API")
                return
                
            # Process data
            data_df = process_energy_data(energy_data)
            
            if data_df.empty:
                logger.error("Failed to process energy data")
                return
                
            # Make sure the directory exists
            os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)
                
            # Save to CSV
            data_df.to_csv(CSV_FILE_PATH, index=False)
            logger.info(f"Energy data saved to {CSV_FILE_PATH}")
            logger.info(f"Total records: {len(data_df)}")
            logger.info(f"Data covers period: {data_df['timestamp'].min()} to {data_df['timestamp'].max()}")
            
            # Check for data quality issues
            check_data_quality(data_df)
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
