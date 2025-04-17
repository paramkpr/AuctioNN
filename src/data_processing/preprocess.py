"""
This script contains the functions to clean and merge the impression and conversion data from Claritas.

The functions are:
- clean_and_merge_data: Cleans and merges the impression and conversion data.
- save_dataframe_to_parquet: Saves a pandas DataFrame to a Parquet file.

"""
import math # Import math for ceiling division
import os
import multiprocessing


import joblib
import pandas as pd
from user_agents import parse
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

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


# --- New Function for Preprocessing Fitting ---


def fit_and_save_preprocessors(
    train_df: pd.DataFrame,
    output_dir: str = './preprocessors',
    categorical_features: list[str] | None = None,
    boolean_features: list[str] | None = None,
    cyclical_features: list[str] | None = None
):
    """
    Fits OrdinalEncoder and StandardScaler using ONLY the provided training DataFrame,
    and saves the fitted objects and feature lists.

    Args:
        train_df: The training DataFrame slice.
        output_dir: Directory to save the fitted preprocessors and feature lists.
        categorical_features: List of categorical column names. If None, defaults are used.
        boolean_features: List of boolean column names. If None, defaults are used.
        cyclical_features: List of cyclical column names. If None, defaults are used.

    Returns:
        tuple: Contains the fitted components (useful if called programmatically).
    """
    print(f"\n--- Starting Preprocessor Fitting on Provided Training Data (shape: {train_df.shape}) ---")
    # Removed: Loading data, splitting data
    os.makedirs(output_dir, exist_ok=True)

    # --- Define Feature Groups (Use defaults if not provided) ---
    if categorical_features is None:
        categorical_features = [
            'placement_id', 'cnxn_type', 'dma', 'country', 'prizm_premier_code',
            'campaign_id', 'ua_browser', 'ua_os', 'ua_device_family', 'ua_device_brand'
        ]
    if boolean_features is None:
        boolean_features = [
            'ua_is_mobile', 'ua_is_tablet', 'ua_is_pc', 'ua_is_bot'
        ]
    if cyclical_features is None:
        cyclical_features = [
            'impression_hour', 'impression_dayofweek'
        ]

    # --- Verify Features Exist in DataFrame ---
    all_input_features = categorical_features + boolean_features + cyclical_features
    missing_cols = [col for col in all_input_features if col not in train_df.columns]
    if missing_cols:
        # Error message is now simpler as the DataFrame is passed directly
        print("ERROR: The provided train_df is missing expected feature columns:")
        for col in missing_cols:
            print(f"  - {col}")
        raise ValueError("Missing expected feature columns in train_df.")

    # --- Fit Categorical Encoder ---
    print("Fitting categorical encoder...")
    categorical_encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1,
        dtype=np.int64
    )
    categorical_encoder.fit(train_df[categorical_features])
    category_sizes = {
        col: len(cats) + 1
        for col, cats in zip(categorical_features, categorical_encoder.categories_)
    }
    print("Categorical encoder fitted.") # Simplified output

    # --- Prepare and Fit Numerical Scaler ---
    print("Preparing numerical features for scaling...")
    temp_numeric_df = pd.DataFrame(index=train_df.index)
    for col in boolean_features:
        temp_numeric_df[col] = train_df[col].astype(float)
    hour = train_df['impression_hour']
    day = train_df['impression_dayofweek']
    temp_numeric_df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    temp_numeric_df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    temp_numeric_df['day_sin'] = np.sin(2 * np.pi * day / 7.0)
    temp_numeric_df['day_cos'] = np.cos(2 * np.pi * day / 7.0)
    numerical_features_to_scale = temp_numeric_df.columns.tolist()
    print(f"Numerical columns created for scaling: {len(numerical_features_to_scale)}")

    print("Fitting numerical scaler...")
    numerical_scaler = StandardScaler()
    numerical_scaler.fit(temp_numeric_df[numerical_features_to_scale])
    print("Numerical scaler fitted.")

    # --- Save Preprocessors and Feature Lists (logic unchanged) ---
    joblib.dump(categorical_encoder, os.path.join(output_dir, 'categorical_encoder.joblib'))
    joblib.dump(numerical_scaler, os.path.join(output_dir, 'numerical_scaler.joblib'))
    joblib.dump(category_sizes, os.path.join(output_dir, 'category_sizes.joblib'))
    joblib.dump(categorical_features, os.path.join(output_dir, 'categorical_features.joblib'))
    joblib.dump(boolean_features, os.path.join(output_dir, 'boolean_features.joblib'))
    joblib.dump(cyclical_features, os.path.join(output_dir, 'cyclical_features.joblib'))
    joblib.dump(numerical_features_to_scale, os.path.join(output_dir, 'numerical_features_to_scale.joblib'))

    print(f"Preprocessors and feature lists saved successfully to '{output_dir}'")
    print("--- Preprocessor Fitting Complete ---")

    return (categorical_encoder, numerical_scaler, category_sizes,
            categorical_features, boolean_features, cyclical_features, numerical_features_to_scale)

def apply_preprocessors_to_split(
    df_split: pd.DataFrame,
    preprocessor_dir: str,
    output_dir: str,
    split_name: str, # e.g., 'train', 'val', 'test' for filenames
    target_column: str = 'conversion_flag'
):
    """
    Applies pre-fitted preprocessors to a data split (DataFrame) and saves
    the resulting processed NumPy arrays.

    Args:
        df_split: The DataFrame slice to process (e.g., train_df, val_df).
        preprocessor_dir: Directory where fitted preprocessors and feature lists are saved.
        output_dir: Directory to save the processed NumPy arrays.
        split_name: Name for this split (e.g., 'train', 'val') used in output filenames.
        target_column: Name of the target variable column.
    """
    if df_split.empty:
        print(f"Skipping preprocessing for empty DataFrame split '{split_name}'.")
        return

    print(f"\n--- Applying Preprocessors to '{split_name}' Split (shape: {df_split.shape}) ---")
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Preprocessors and Feature Lists ---
    print(f"Loading preprocessors from: {preprocessor_dir}")
    try:
        categorical_encoder = joblib.load(os.path.join(preprocessor_dir, 'categorical_encoder.joblib'))
        numerical_scaler = joblib.load(os.path.join(preprocessor_dir, 'numerical_scaler.joblib'))
        # category_sizes = joblib.load(os.path.join(preprocessor_dir, 'category_sizes.joblib')) # Not needed here
        categorical_features = joblib.load(os.path.join(preprocessor_dir, 'categorical_features.joblib'))
        boolean_features = joblib.load(os.path.join(preprocessor_dir, 'boolean_features.joblib'))
        cyclical_features = joblib.load(os.path.join(preprocessor_dir, 'cyclical_features.joblib'))
        numerical_features_to_scale = joblib.load(os.path.join(preprocessor_dir, 'numerical_features_to_scale.joblib'))
    except FileNotFoundError as e:
        print(f"ERROR: Failed to load preprocessor files from {preprocessor_dir}. Details: {e}")
        raise

    # --- Verify Features Exist ---
    required_features = categorical_features + boolean_features + cyclical_features + [target_column]
    missing_cols = [col for col in required_features if col not in df_split.columns]
    if missing_cols:
        raise ValueError(f"DataFrame split '{split_name}' is missing required columns: {missing_cols}")

    # --- 1. Process Categorical Features ---
    print("Processing categorical features...")
    cat_df_slice = df_split[categorical_features]
    # Transform the whole slice
    encoded_cats = categorical_encoder.transform(cat_df_slice) # Shape: (n_samples, n_cat_features)

    # Handle unknowns (-1 -> 0, shift others +1) - Apply element-wise
    encoded_cats[encoded_cats == -1] = 0
    encoded_cats[encoded_cats > -1] += 1
    categorical_data_np = encoded_cats.astype(np.int64) # Ensure correct dtype

    # --- 2. Process Numerical Features (Boolean + Cyclical -> Scale) ---
    print("Processing numerical features...")
    # Create DataFrame for scaling in the correct order
    numerical_df_for_scaling = pd.DataFrame(index=df_split.index)
    # Booleans
    for col in boolean_features:
        numerical_df_for_scaling[col] = df_split[col].astype(float)
    # Cyclical
    hour = df_split['impression_hour']
    day = df_split['impression_dayofweek']
    numerical_df_for_scaling['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    numerical_df_for_scaling['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    numerical_df_for_scaling['day_sin'] = np.sin(2 * np.pi * day / 7.0)
    numerical_df_for_scaling['day_cos'] = np.cos(2 * np.pi * day / 7.0)

    # Ensure columns are in the order expected by the scaler
    numerical_df_ordered = numerical_df_for_scaling[numerical_features_to_scale]

    # Scale the whole slice
    numerical_data_np = numerical_scaler.transform(numerical_df_ordered).astype(np.float32)

    # --- 3. Process Target ---
    print("Processing target variable...")
    target_data_np = df_split[target_column].values.astype(np.float32)

    # --- 4. Save Processed Arrays ---
    cat_out_path = os.path.join(output_dir, f"{split_name}_categorical_data.npy")
    num_out_path = os.path.join(output_dir, f"{split_name}_numerical_data.npy")
    tgt_out_path = os.path.join(output_dir, f"{split_name}_target_data.npy")

    print(f"Saving processed arrays for '{split_name}' split...")
    np.save(cat_out_path, categorical_data_np)
    np.save(num_out_path, numerical_data_np)
    np.save(tgt_out_path, target_data_np)

    print(f"  Categorical data saved to: {cat_out_path} (shape: {categorical_data_np.shape})")
    print(f"  Numerical data saved to:   {num_out_path} (shape: {numerical_data_np.shape})")
    print(f"  Target data saved to:      {tgt_out_path} (shape: {target_data_np.shape})")
    print(f"--- Preprocessing Complete for '{split_name}' Split ---")