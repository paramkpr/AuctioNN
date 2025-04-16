"""
This script contains the functions to clean and merge the impression and conversion data from Claritas.

The functions are:
- clean_and_merge_data: Cleans and merges the impression and conversion data.
- save_dataframe_to_parquet: Saves a pandas DataFrame to a Parquet file.

"""

import pandas as pd
from user_agents import parse
from tqdm import tqdm
import multiprocessing
import numpy as np
import math # Import math for ceiling division

tqdm.pandas()


def parse_ua_chunk(ua_series_chunk):
    """Parses a chunk of user agent strings and returns a DataFrame."""
    features = []
    default_features = {
        'ua_browser': 'Unknown', 'ua_os': 'Unknown', 'ua_device_family': 'Unknown',
        'ua_device_brand': 'Unknown', 'ua_is_mobile': False, 'ua_is_tablet': False,
        'ua_is_pc': False, 'ua_is_bot': False
    }
    for ua_str in ua_series_chunk:
        if ua_str and isinstance(ua_str, str): # Check if ua_str is a non-empty string
            try:
                ua = parse(ua_str)
                features.append({
                    'ua_browser': ua.browser.family if ua.browser.family else 'Unknown',
                    'ua_os': ua.os.family if ua.os.family else 'Unknown',
                    'ua_device_family': ua.device.family if ua.device.family else 'Unknown',
                    'ua_device_brand': ua.device.brand if ua.device.brand else 'Unknown',
                    'ua_is_mobile': ua.is_mobile,
                    'ua_is_tablet': ua.is_tablet,
                    'ua_is_pc': ua.is_pc,
                    'ua_is_bot': ua.is_bot,
                })
            except Exception: # Catch potential parsing errors for specific UAs
                 features.append(default_features.copy())
        else:
            features.append(default_features.copy())
    # Create DataFrame with original index from the chunk to preserve alignment
    return pd.DataFrame(features, index=ua_series_chunk.index)


def clean_and_merge_data(impressions_path: str, conversions_path: str) -> pd.DataFrame:
    """
    Loads impression and conversion data from Claritas, merges them, cleans the data,
    and performs feature engineering. User agent parsing is parallelized.

    Args:
        impressions_path: Path to the impressions parquet dataset directory.
        conversions_path: Path to the conversions parquet dataset directory.

    Returns:
        A pandas DataFrame containing the cleaned and merged data.
    """
    # Load datasets
    df_impressions = pd.read_parquet(impressions_path)
    df_conversions = pd.read_parquet(conversions_path)

    print(f"Loaded datasets: <Impressions: {df_impressions.shape}>, <Conversions: {df_conversions.shape}>")

    # Drop AIP columns if they exist since they are mostly empty
    aip_imp_cols = [col for col in df_impressions.columns if col.startswith('aip')]
    aip_conv_cols = [col for col in df_conversions.columns if col.startswith('aip')]
    df_impressions = df_impressions.drop(columns=aip_imp_cols)
    df_conversions = df_conversions.drop(columns=aip_conv_cols)

    print(f"Dropped AIP columns: <Impressions: {df_impressions.shape}>, <Conversions: {df_conversions.shape}>")

    # --- Linking using Unique IDs ---
    # Select necessary columns from conversions for the merge
    df_conversions_subset = df_conversions[['imp_click_unique_id', 'conv_dttm_utc', 'goal_name']].copy()

    # Perform the left merge
    df_merged = pd.merge(
        df_impressions,
        df_conversions_subset,
        left_on='unique_id',
        right_on='imp_click_unique_id',
        how='left'
    )

    print(f"Merged dataset: <Merged: {df_merged.shape}>")

    # --- Data Cleaning and Feature Engineering ---
    # Create conversion flag
    df_merged['conversion_flag'] = (~df_merged['conv_dttm_utc'].isnull()).astype(int)

    print(f"Created conversion flag: <Merged: {df_merged.shape}>")

    # Drop redundant join key
    columns_to_drop = ['imp_click_unique_id']
    df_merged = df_merged.drop(columns=columns_to_drop)

    # Rename impression timestamp
    df_merged = df_merged.rename(columns={'dttm_utc': 'impression_dttm_utc'})

    # Handle missing values
    df_merged['goal_name'] = df_merged['goal_name'].fillna('No Goal Name')
    cols_to_fill_unknown = ['prizm_premier_code', 'device_type']
    for col in cols_to_fill_unknown:
        if col in df_merged.columns: # Check if column exists
             df_merged[col] = df_merged[col].fillna('Unknown')

    print(f"Filled missing values: <Merged: {df_merged.shape}>")

    # --- Multiprocessing User Agent Parsing ---
    if 'user_agent' in df_merged.columns and not df_merged['user_agent'].isnull().all():
        ua_series = df_merged['user_agent'].fillna('') # Fill NaNs before processing
        num_processes = multiprocessing.cpu_count()
        # Adjust chunk size for potentially better load balancing with many small tasks
        # Aim for more chunks than processes if parsing is very fast per item
        chunk_size = max(1, math.ceil(len(ua_series) / (num_processes * 4))) # Example: 4 chunks per process
        num_chunks = math.ceil(len(ua_series) / chunk_size)

        print(f"Starting User Agent parsing using {num_processes} processes ({num_chunks} chunks)...")

        # Create chunks manually to preserve index within chunks
        chunks = [ua_series[i:i + chunk_size] for i in range(0, len(ua_series), chunk_size)]

        ua_features_list = []
        try:
            # Use multiprocessing Pool with tqdm for progress
            with multiprocessing.Pool(processes=num_processes) as pool:
                 # Use imap_unordered for potential slight speedup if order doesn't matter for progress
                 # Wrap with tqdm to show progress based on completed chunks
                 ua_features_list = list(tqdm(pool.imap_unordered(parse_ua_chunk, chunks), total=num_chunks, desc="Parsing User Agents"))

            # Concatenate results
            if ua_features_list:
                 ua_features = pd.concat(ua_features_list)
                 # Ensure the index of ua_features matches df_merged's index
                 ua_features = ua_features.reindex(df_merged.index)
            else:
                 # Handle cases where no features were generated (e.g., all input was invalid)
                  ua_features = pd.DataFrame(index=df_merged.index, columns=[ # Define expected columns
                     'ua_browser', 'ua_os', 'ua_device_family', 'ua_device_brand',
                     'ua_is_mobile', 'ua_is_tablet', 'ua_is_pc', 'ua_is_bot'
                 ]).fillna('Unknown') # Fill with defaults
                  # Ensure boolean columns are boolean
                  bool_cols = ['ua_is_mobile', 'ua_is_tablet', 'ua_is_pc', 'ua_is_bot']
                  ua_features[bool_cols] = ua_features[bool_cols].fillna(False).astype(bool)


        except Exception as e:
            print(f"Error during parallel user agent parsing: {e}")
            # Fallback or define empty features df? For now, let's create an empty one.
            ua_features = pd.DataFrame(index=df_merged.index, columns=[ # Define expected columns
                 'ua_browser', 'ua_os', 'ua_device_family', 'ua_device_brand',
                 'ua_is_mobile', 'ua_is_tablet', 'ua_is_pc', 'ua_is_bot'
             ]).fillna('Unknown')
            bool_cols = ['ua_is_mobile', 'ua_is_tablet', 'ua_is_pc', 'ua_is_bot']
            ua_features[bool_cols] = ua_features[bool_cols].fillna(False).astype(bool)


        df_merged = pd.concat([df_merged, ua_features], axis=1)
        print(f"Parsed user agent strings: <Merged: {df_merged.shape}>")

        columns_to_drop_after_ua = ['user_agent']
        columns_to_drop_exists = [col for col in columns_to_drop_after_ua if col in df_merged.columns]
        df_merged = df_merged.drop(columns=columns_to_drop_exists)
    else:
         print("Skipping User Agent parsing as 'user_agent' column is missing or empty.")


    # Add temporal features
    df_merged['impression_dttm_utc'] = pd.to_datetime(df_merged['impression_dttm_utc'])
    df_merged['impression_hour'] = df_merged['impression_dttm_utc'].dt.hour
    df_merged['impression_dayofweek'] = df_merged['impression_dttm_utc'].dt.dayofweek # Monday=0, Sunday=6

    print(f"Added temporal features: <Merged: {df_merged.shape}>")

    # Ensure correct data types for potentially problematic columns after merge/fillna
    bool_cols = ['ua_is_mobile', 'ua_is_tablet', 'ua_is_pc', 'ua_is_bot']
    for col in bool_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].astype(bool)

    int_cols = ['impression_hour', 'impression_dayofweek']
    for col in int_cols:
         if col in df_merged.columns:
             df_merged[col] = df_merged[col].astype('int32') # More efficient than int64

    print(f"Ensured correct data types: <Merged: {df_merged.shape}>")

    print(f"Data cleaning and merging complete. Final shape: <Merged: {df_merged.shape}>")
    return df_merged


def save_dataframe_to_parquet(df: pd.DataFrame, output_path: str):
    """
    Saves a pandas DataFrame to a Parquet file.

    Args:
        df: The pandas DataFrame to save.
        output_path: The path where the Parquet file will be saved.
                     Should end with '.parquet'.
    """
    try:
        df.to_parquet(output_path, index=False, engine='pyarrow') # Specify engine for clarity
        print(f"DataFrame successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving DataFrame to {output_path}: {e}")

