"""
This script contains the functions to clean and merge the impression and conversion data from Claritas.

The functions are:
- clean_and_merge_data: Cleans and merges the impression and conversion data.
- save_dataframe_to_parquet: Saves a pandas DataFrame to a Parquet file.

"""

import pandas as pd
from user_agents import parse

def clean_and_merge_data(impressions_path: str, conversions_path: str) -> pd.DataFrame:
    """
    Loads impression and conversion data from Claritas, merges them, cleans the data,
    and performs feature engineering.
    Note: This function takes a while to run since the User Agent parsing is slow for
     large datasets.

    Args:
        impressions_path: Path to the impressions parquet dataset directory.
        conversions_path: Path to the conversions parquet dataset directory.

    Returns:
        A pandas DataFrame containing the cleaned and merged data.
    """
    # Load datasets
    df_impressions = pd.read_parquet(impressions_path)
    df_conversions = pd.read_parquet(conversions_path)

    # Drop AIP columns if they exist since they are mostly empty
    aip_imp_cols = [col for col in df_impressions.columns if col.startswith('aip')]
    aip_conv_cols = [col for col in df_conversions.columns if col.startswith('aip')]
    df_impressions = df_impressions.drop(columns=aip_imp_cols)
    df_conversions = df_conversions.drop(columns=aip_conv_cols)

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

    # --- Data Cleaning and Feature Engineering ---
    # Create conversion flag
    df_merged['conversion_flag'] = (~df_merged['conv_dttm_utc'].isnull()).astype(int)

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

    # Parse user agent strings
    ua_features = df_merged['user_agent'].apply(lambda ua_str: pd.Series({
        'ua_browser': parse(ua_str).browser.family if ua_str else 'Unknown',
        'ua_os': parse(ua_str).os.family if ua_str else 'Unknown',
        'ua_device_family': parse(ua_str).device.family if ua_str else 'Unknown',
        'ua_device_brand': parse(ua_str).device.brand if ua_str else 'Unknown',
        'ua_is_mobile': parse(ua_str).is_mobile if ua_str else False, # Default bool to False
        'ua_is_tablet': parse(ua_str).is_tablet if ua_str else False,
        'ua_is_pc': parse(ua_str).is_pc if ua_str else False,
        'ua_is_bot': parse(ua_str).is_bot if ua_str else False,
    }))
    df_merged = pd.concat([df_merged, ua_features], axis=1)

    columns_to_drop_after_ua = ['user_agent']
    columns_to_drop_exists = [col for col in columns_to_drop_after_ua if col in df_merged.columns]
    df_merged = df_merged.drop(columns=columns_to_drop_exists)



    # Add temporal features
    df_merged['impression_dttm_utc'] = pd.to_datetime(df_merged['impression_dttm_utc'])
    df_merged['impression_hour'] = df_merged['impression_dttm_utc'].dt.hour
    df_merged['impression_dayofweek'] = df_merged['impression_dttm_utc'].dt.dayofweek # Monday=0, Sunday=6

    # Ensure correct data types for potentially problematic columns after merge/fillna
    bool_cols = ['ua_is_mobile', 'ua_is_tablet', 'ua_is_pc', 'ua_is_bot']
    for col in bool_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].astype(bool)

    int_cols = ['impression_hour', 'impression_dayofweek']
    for col in int_cols:
         if col in df_merged.columns:
             df_merged[col] = df_merged[col].astype('int32') # More efficient than int64

    print(f"Data cleaning and merging complete. Final shape: {df_merged.shape}")
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

