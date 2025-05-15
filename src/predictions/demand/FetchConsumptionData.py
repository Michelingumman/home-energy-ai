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
CSV_FILE_PATH = 'data/processed/villamichelin/VillamichelinConsumption.csv'

def fetch_tibber_data_paginated(after=None, first=1000):
    """
    Fetch a single page of consumption data from Tibber API
    
    Args:
        after (str, optional): Cursor to fetch data after
        first (int): Number of records to fetch
    
    Returns:
        tuple: (list of consumption nodes, end cursor, has next page)
    """
    url = 'https://api.tibber.com/v1-beta/gql'
    
    headers = {
        'Authorization': f'Bearer {TIBBER_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    # Build the GraphQL query with pagination
    # The Tibber API only supports cursor-based pagination with 'after'
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
            }
          }
        }
        """ % (first, after)
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
            }
          }
        }
        """ % first
    
    data = {'query': query}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract consumption data from the response
        consumption_nodes = []
        end_cursor = None
        has_next_page = False
        
        if ('data' in result and 
            'viewer' in result['data'] and 
            'homes' in result['data']['viewer'] and 
            result['data']['viewer']['homes']):
            
            homes = result['data']['viewer']['homes']
            
            for home in homes:
                if 'consumption' in home and 'edges' in home['consumption']:
                    # Get page info
                    if 'pageInfo' in home['consumption']:
                        end_cursor = home['consumption']['pageInfo'].get('endCursor')
                        has_next_page = home['consumption']['pageInfo'].get('hasNextPage', False)
                    
                    # Get consumption nodes
                    for edge in home['consumption']['edges']:
                        if 'node' in edge:
                            consumption_nodes.append(edge['node'])
            
        return consumption_nodes, end_cursor, has_next_page
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from Tibber API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response content: {e.response.text}")
        return [], None, False

def fetch_all_tibber_data():
    """
    Fetch all available consumption data from Tibber API with pagination
    
    Returns:
        list: List of consumption data points
    """
    all_consumption_nodes = []
    after_cursor = None
    has_more_pages = True
    page_count = 0
    
    logger.info("Fetching all available historical consumption data")
    
    # Use cursor-based pagination 
    while has_more_pages:
        page_count += 1
        logger.info(f"Fetching page {page_count} of consumption data...")
        
        consumption_nodes, end_cursor, has_next_page = fetch_tibber_data_paginated(
            after=after_cursor,
            first=1000
        )
        
        if not consumption_nodes:
            logger.warning(f"No data found on page {page_count}")
            break
        
        # Add all nodes
        all_consumption_nodes.extend(consumption_nodes)
        
        # Update for next page
        after_cursor = end_cursor
        has_more_pages = has_next_page
        
        # Log progress
        logger.info(f"Retrieved {len(consumption_nodes)} records on page {page_count}")
        logger.info(f"Total records so far: {len(all_consumption_nodes)}")
        
        # Break if we've reached the maximum pages to fetch (safety measure)
        if page_count >= 100:  # Limit to 100 pages as a safety measure
            logger.warning("Reached maximum page count (100). Breaking pagination loop.")
            break
    
    logger.info(f"Fetched a total of {len(all_consumption_nodes)} records from {page_count} pages")
    return all_consumption_nodes

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
        list: List of consumption data points
    """
    # Convert to UTC for API request (required for comparison with API data)
    from_datetime_utc = from_datetime.replace(tzinfo=timezone.utc)
        
    # Convert to string for logging
    from_date_str = from_datetime.strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Fetching new data since: {from_date_str}")
    
    # Target date (used for checking when to start collecting)
    target_date = from_datetime
    
    # Initialize variables
    new_consumption_nodes = []
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
        
        consumption_nodes, end_cursor, has_next_page = fetch_tibber_data_paginated(
            after=after_cursor,
            first=records_per_request
        )
        
        if not consumption_nodes:
            logger.warning(f"No data found on page {page_count}")
            break
        
        # Check first and last record in this batch to see if we're close to target date
        first_node = consumption_nodes[0]
        last_node = consumption_nodes[-1]
        
        # Parse dates, stripping timezone info
        first_date = parse_timestamp_strip_timezone(first_node['from'])
        last_date = parse_timestamp_strip_timezone(last_node['from'])
        
        # Skip this page if we couldn't parse the dates
        if first_date is None or last_date is None:
            logger.warning(f"Skipping page {page_count} due to unparseable dates")
            after_cursor = end_cursor
            has_more_pages = has_next_page
            continue
        
        # If last record is still before our target date, continue skipping
        if last_date < target_date:
            after_cursor = end_cursor
            has_more_pages = has_next_page
            continue
            
        # If first record is after our target date, we've found new data
        if first_date >= target_date:
            logger.info(f"All records on page {page_count} are after target date. Starting collection.")
            new_consumption_nodes.extend(consumption_nodes)
            found_new_data = True
        else:
            # We're in the transition page where some records are before and some after
            # Filter and keep only records that are >= target date
            logger.info(f"Found transition page {page_count} with some records after target date.")
            for node in consumption_nodes:
                # Parse timestamp, strip timezone
                node_date = parse_timestamp_strip_timezone(node['from'])
                if node_date is not None and node_date >= target_date:
                    new_consumption_nodes.append(node)
            found_new_data = True
        
        # Update for next page
        after_cursor = end_cursor
        has_more_pages = has_next_page
    
    # ---- PHASE 2: Continue collecting remaining new data ----
    if found_new_data and has_more_pages:
        logger.info("PHASE 2: Collecting remaining new data...")
        
        while has_more_pages:
            page_count += 1
            
            consumption_nodes, end_cursor, has_next_page = fetch_tibber_data_paginated(
                after=after_cursor,
                first=records_per_request
            )
            
            if not consumption_nodes:
                logger.warning(f"No data found on collection page {page_count}")
                break
                
            # Add all nodes (all must be after our target date)
            new_consumption_nodes.extend(consumption_nodes)
            
            # Update for next page
            after_cursor = end_cursor
            has_more_pages = has_next_page
            
            # Log progress
            logger.info(f"Collected {len(consumption_nodes)} records on page {page_count}")
            logger.info(f"Total new records so far: {len(new_consumption_nodes)}")
            
            # Break if we've reached the maximum pages to fetch (safety measure)
            if page_count >= 100:  # Limit to 100 pages as a safety measure
                logger.warning("Reached maximum page count (100). Breaking pagination loop.")
                break
    
    # Check and log results
    if not found_new_data:
        logger.info("No new data found since the target date.")
    else:
        logger.info(f"Fetched a total of {len(new_consumption_nodes)} new records")
    
    return new_consumption_nodes

def process_consumption_data(consumption_data):
    """
    Process consumption data into a pandas DataFrame
    
    Args:
        consumption_data (list): List of consumption data points from Tibber API
    
    Returns:
        pd.DataFrame: Processed data
    """
    if not consumption_data:
        return pd.DataFrame()
    
    data_list = []
    for item in consumption_data:
        if item and 'from' in item and 'consumption' in item:
            try:
                # Parse timestamp and strip timezone
                dt = parse_timestamp_strip_timezone(item['from'])
                if dt is None:
                    logger.warning(f"Skipping data point with invalid timestamp: {item['from']}")
                    continue
                
                data_list.append({
                    'timestamp': dt,
                    'consumption': float(item.get('consumption', 0)),
                    'cost': float(item.get('cost', 0)),
                    'unit_price': float(item.get('unitPrice', 0)),
                    'unit_price_vat': float(item.get('unitPriceVAT', 0)),
                    'unit': item.get('consumptionUnit', 'kWh')
                })
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing data point: {e}, data: {item}")
    
    if not data_list:
        return pd.DataFrame()
        
    df = pd.DataFrame(data_list)
    
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
    
    # Check for zero consumption values (could indicate data issues)
    zero_consumption = (df['consumption'] == 0).sum()
    if zero_consumption > 0:
        percent = (zero_consumption / total_rows) * 100
        logger.info(f"Found {zero_consumption} records with zero consumption ({percent:.2f}%)")
    
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
    logger.info("Starting to fetch consumption data from Tibber API")
    logger.warning("Using simple approach: timestamps stored without timezone information")
    
    try:
        # Check if the CSV file exists and has data
        if os.path.exists(CSV_FILE_PATH) and os.path.getsize(CSV_FILE_PATH) > 0:
            # Read existing data
            logger.info("Reading existing consumption data")
            existing_data = pd.read_csv(CSV_FILE_PATH)
            
            # Convert timestamp to datetime without timezone
            existing_data['timestamp'] = pd.to_datetime(existing_data['timestamp'])
            
            # Get the latest timestamp and add 1 hour to avoid duplicates
            latest_timestamp = existing_data['timestamp'].max()
            logger.info(f"Latest timestamp in existing data: {latest_timestamp}")
            
            # Add 1 hour to avoid duplicates
            next_timestamp = latest_timestamp + timedelta(hours=1)
            
            # Use the optimized approach to fetch only recent data
            consumption_data = fetch_recent_tibber_data(next_timestamp)
            
            if not consumption_data:
                logger.info("No new data available")
                return
            
            # Process new data
            new_data_df = process_consumption_data(consumption_data)
            
            if new_data_df.empty:
                logger.info("No new data to add after processing")
                return
                
            # Log the date range of new data
            min_date = new_data_df['timestamp'].min()
            max_date = new_data_df['timestamp'].max()
            logger.info(f"New data covers period: {min_date} to {max_date}")
            
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
            logger.info(f"Updated consumption data saved to {CSV_FILE_PATH}")
            logger.info(f"Total records in the updated file: {len(combined_df)}")
            logger.info(f"Data now covers: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
            
            # Check for data quality issues
            check_data_quality(combined_df)
            
        else:
            # Fetch all available data
            logger.info("No existing data found. Fetching all available consumption data")
            consumption_data = fetch_all_tibber_data()
            
            if not consumption_data:
                logger.error("No data available from Tibber API")
                return
                
            # Process data
            data_df = process_consumption_data(consumption_data)
            
            if data_df.empty:
                logger.error("Failed to process consumption data")
                return
                
            # Make sure the directory exists
            os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)
                
            # Save to CSV
            data_df.to_csv(CSV_FILE_PATH, index=False)
            logger.info(f"Consumption data saved to {CSV_FILE_PATH}")
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
