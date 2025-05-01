"""
This script contains functions to process impression and conversion data using Dask
for large datasets, perform feature engineering, split the data, fit preprocessors
on a sample, apply preprocessors to the full dataset, and save the results.

Main orchestration should happen outside these functions, potentially in a
main script or notebook, calling these functions in sequence.
"""
import re
import os
import glob # For finding files
import joblib
import pandas as pd
import numpy as np
import dask
from dask import delayed
from dask.distributed import wait
from dask.diagnostics import ProgressBar
from dask.dataframe.utils import make_meta
import dask.dataframe as dd
from user_agents import parse
from tqdm import tqdm # tqdm can be used with Dask diagnostics or for pandas operations
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Configure pandas display for head() checks if needed
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)


dask.config.set({
    # --- worker RAM management ---
    "distributed.worker.memory.target"   : 0.45,   # start spilling early
    "distributed.worker.memory.spill"    : 0.55,
    "distributed.worker.memory.pause"    : 0.80,
    "distributed.worker.memory.terminate": 0.95,
    "shuffle.split_out": 32,
    "distributed.worker.memory.spill-compression": "auto",   # lz4/snappy if available


    # --- keep big shuffles on disk, not in RAM ---
    "dataframe.shuffle.method": "p2p",            # options: "disk", "tasks", "p2p"
})

# --- User Agent Parsing Helper (Used by map_partitions) ---

def parse_ua_chunk(ua_series_chunk: pd.Series) -> pd.DataFrame:
    """
    Parses a chunk (pandas Series) of user agent strings and returns a DataFrame.
    Designed to be used with Dask's map_partitions.
    """
    features = []
    default_features = {
        'ua_browser': 'Unknown', 'ua_os': 'Unknown', 'ua_device_family': 'Unknown',
        'ua_device_brand': 'Unknown', 'ua_is_mobile': False, 'ua_is_tablet': False,
        'ua_is_pc': False, 'ua_is_bot': False
    }
    for ua_str in ua_series_chunk:
        # Ensure it's a string and not None or NaN before parsing
        if ua_str and isinstance(ua_str, str):
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
            except Exception: # Catch rare parsing errors for specific UAs
                 features.append(default_features.copy())
        else:
            # Handle None, NaN, or non-string inputs
            features.append(default_features.copy())

    # Return DataFrame with the same index as the input chunk
    return pd.DataFrame(features, index=ua_series_chunk.index)

# --- Dask-based Data Cleaning and Merging ---

# ──────────────────────────────────────────────────────────────────────────────
#  define_dask_cleaning_graph  –  campaign-aware version
# ──────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────
def define_dask_cleaning_graph(
    impressions_path: str,
    conversions_path: str,

    impression_cols_needed: list[str],
    conversion_cols_needed: list[str] = [
        "imp_click_unique_id", "imp_click_campaign_id", "conv_dttm_utc",
    ],
    split: tuple[float] = (0.8, 0.1, 0.1),
    out_base: str = './data/merged',
    seed: int = 42
) -> dd.DataFrame:

    print("Defining campaign-aware graph (delayed→from_delayed)…")

    # 0️⃣  discover campaign folders on the conversion side
    conv_ids = sorted(
        int(re.search(r"\d+", name).group())
        for name in os.listdir(conversions_path)
        if name.startswith("imp_click_campaign_id=")
    )
    # cut off from 17562
    conv_ids = conv_ids[conv_ids.index(47589):]
    print(f"Found {len(conv_ids)} campaign partitions")

    # 1️ meta template (empty DF with correct dtypes)
    meta_cols = {
        "campaign_id"          : "int64",
        "unique_id"            : "int64",
        "dttm_utc"             : "datetime64[ns]",
        "cxnn_type"            : "object",
        "user_agent"           : "object",
        "dma"                  : "object",
        "country"              : "object",
        "prizm_premier_code"   : "object",
        "device_type"          : "object",
        "conv_dttm_utc"        : "datetime64[ns]",
        "conversion_flag"      : "int8",
        "impression_hour"      : "Int32",
        "impression_dayofweek" : "Int32",
    }
    _meta = make_meta(meta_cols)

    # 2️  delayed loader that **returns pandas**
    def load_merge_one(cid: int) -> pd.DataFrame:
        imp = dd.read_parquet(
            impressions_path,
            columns=impression_cols_needed,
            filters=[("campaign_id", "==", cid)],
            split_row_groups=True,
            chunksize="64MB",
        )
        conv = dd.read_parquet(
            conversions_path,
            columns=conversion_cols_needed,
            filters=[("imp_click_campaign_id", "==", cid)],
            split_row_groups=True,
            chunksize="64MB",
        )

        # normalise dtypes for the join keys
        imp  = imp.astype({"campaign_id": "int64"})
        conv = conv.astype({"imp_click_campaign_id": "int64"})

        merged = (
            dd.merge(
                imp,
                conv,
                left_on = ["campaign_id", "unique_id"],
                right_on= ["imp_click_campaign_id", "imp_click_unique_id"],
                how="left",
                # _meta=_meta
            )
            .drop(columns=["imp_click_campaign_id", "imp_click_unique_id"])
        )

        # feature engineering inside the graph
        merged["conversion_flag"] = (~merged.conv_dttm_utc.isnull()).astype("int8")
        merged = merged.rename(columns={"dttm_utc": "impression_dttm_utc"})
        merged["impression_dttm_utc"] = dd.to_datetime(
            merged.impression_dttm_utc, errors="coerce"
        )
        merged["impression_hour"]      = merged.impression_dttm_utc.dt.hour.astype("Int32")
        merged["impression_dayofweek"] = merged.impression_dttm_utc.dt.dayofweek.astype("Int32")

        # persist as pandas so only *one* object per campaign enters the graph
        train, val, test = merged.random_split(split, random_state=seed + cid)

        train.to_parquet(f"{out_base}/train/campaign_id={cid}", write_index=False, overwrite=True, compute=True)
        val.to_parquet(  f"{out_base}/val/campaign_id={cid}",   write_index=False, overwrite=True, compute=True)
        test.to_parquet( f"{out_base}/test/campaign_id={cid}",  write_index=False, overwrite=True, compute=True)
        return len(merged)

    return[load_merge_one(cid) for cid in conv_ids]


# --- Preprocessor Fitting (on Sample) ---

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
        train_df: The training DataFrame slice (Pandas).
        output_dir: Directory to save the fitted preprocessors and feature lists.
        categorical_features: List of categorical column names. If None, defaults are used.
        boolean_features: List of boolean column names. If None, defaults are used.
        cyclical_features: List of cyclical column names. If None, defaults are used.

    Returns:
        tuple: Contains the fitted components (useful if called programmatically).
    """
    print(f"\n--- Starting Preprocessor Fitting on Provided Training Data (shape: {train_df.shape}) ---")
    os.makedirs(output_dir, exist_ok=True)

    # --- Define Feature Groups (Use defaults if not provided) ---
    if categorical_features is None:
        categorical_features = [
            'cnxn_type', 'dma', 'country', 'prizm_premier_code',
            'campaign_id', 'ua_browser', 'ua_os', 'ua_device_family', 'ua_device_brand'
            # Ensure these defaults match the columns actually present in your data sample
        ]
        # Filter default features to only those present in the sample df
        categorical_features = [f for f in categorical_features if f in train_df.columns]

    if boolean_features is None:
        boolean_features = [
            'ua_is_mobile', 'ua_is_tablet', 'ua_is_pc', 'ua_is_bot'
        ]
        boolean_features = [f for f in boolean_features if f in train_df.columns]


    if cyclical_features is None:
        cyclical_features = [
            'impression_hour', 'impression_dayofweek'
        ]
        cyclical_features = [f for f in cyclical_features if f in train_df.columns]


    # --- Verify Features Exist in DataFrame ---
    all_input_features = categorical_features + boolean_features + cyclical_features
    missing_cols = [col for col in all_input_features if col not in train_df.columns]
    # This check might be redundant now given the filtering above, but good as a safeguard
    if missing_cols:
        print(f"Warning: Columns specified or defaulted are missing from the sample train_df: {missing_cols}")
        raise ValueError("Missing expected feature columns in train_df sample.")

    # Only proceed if there are features to process
    if not categorical_features and not boolean_features and not cyclical_features:
         raise ValueError("No features available or specified for preprocessing in the provided sample.")


    category_sizes = {}
    numerical_features_to_scale = []
    categorical_encoder = None
    numerical_scaler = None

    # --- Fit Categorical Encoder ---
    if categorical_features:
        print(f"Fitting categorical encoder on: {categorical_features}")
        categorical_encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1, # Will be mapped to 0 later
            dtype=np.int64
        )
        try:
            categorical_encoder.fit(train_df[categorical_features])
            # +1 for the unknown category mapped to 0
            category_sizes = {
                col: len(cats) + 1
                for col, cats in zip(categorical_features, categorical_encoder.categories_)
            }
            print("Categorical encoder fitted.")
        except Exception as e:
            print(f"Error fitting categorical encoder: {e}")
            raise
    else:
        print("No categorical features to encode.")

    # --- Prepare and Fit Numerical Scaler ---
    if boolean_features or cyclical_features:
        print("Preparing numerical features for scaling...")
        temp_numeric_df = pd.DataFrame(index=train_df.index)

        if boolean_features:
             print(f"Adding boolean features: {boolean_features}")
             for col in boolean_features:
                 temp_numeric_df[col] = train_df[col].astype(float)

        if cyclical_features:
             print(f"Adding cyclical features: {cyclical_features}")
             if 'impression_hour' in cyclical_features and 'impression_hour' in train_df.columns:
                 hour = train_df['impression_hour']
                 temp_numeric_df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
                 temp_numeric_df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
             if 'impression_dayofweek' in cyclical_features and 'impression_dayofweek' in train_df.columns:
                 day = train_df['impression_dayofweek']
                 temp_numeric_df['day_sin'] = np.sin(2 * np.pi * day / 7.0)
                 temp_numeric_df['day_cos'] = np.cos(2 * np.pi * day / 7.0)

        numerical_features_to_scale = temp_numeric_df.columns.tolist()

        if not numerical_features_to_scale:
             print("Warning: No numerical features generated for scaling.")
        else:
             print(f"Numerical columns created for scaling: {numerical_features_to_scale}")
             print("Fitting numerical scaler...")
             numerical_scaler = StandardScaler()
             try:
                 numerical_scaler.fit(temp_numeric_df[numerical_features_to_scale])
                 print("Numerical scaler fitted.")
             except Exception as e:
                 print(f"Error fitting numerical scaler: {e}")
                 raise
    else:
         print("No boolean or cyclical features to scale.")


    # --- Save Preprocessors and Feature Lists ---
    print(f"Saving preprocessors and feature lists to '{output_dir}'")
    joblib.dump(categorical_encoder, os.path.join(output_dir, 'categorical_encoder.joblib'))
    joblib.dump(numerical_scaler, os.path.join(output_dir, 'numerical_scaler.joblib'))
    joblib.dump(category_sizes, os.path.join(output_dir, 'category_sizes.joblib'))
    joblib.dump(categorical_features, os.path.join(output_dir, 'categorical_features.joblib'))
    joblib.dump(boolean_features, os.path.join(output_dir, 'boolean_features.joblib'))
    joblib.dump(cyclical_features, os.path.join(output_dir, 'cyclical_features.joblib'))
    joblib.dump(numerical_features_to_scale, os.path.join(output_dir, 'numerical_features_to_scale.joblib'))

    print("Preprocessors and feature lists saved successfully.")
    print("--- Preprocessor Fitting Complete ---")

    return (categorical_encoder, numerical_scaler, category_sizes,
            categorical_features, boolean_features, cyclical_features, numerical_features_to_scale)



def fit_preprocessors_on_sample(
    train_data_path: str, # Path to the saved Dask training data directory
    output_dir: str = './preprocessors',
    sample_fraction: float | None = None, # Fraction of data to sample
    sample_n_files: int | None = 1, # Alternative: number of files to read
    categorical_features: list[str] | None = None,
    boolean_features: list[str] | None = None,
    cyclical_features: list[str] | None = None
):
    """
    Loads a sample from the saved training data, fits preprocessors (OrdinalEncoder,
    StandardScaler) using the fit_and_save_preprocessors function, and saves them.

    Args:
        train_data_path: Path to the directory containing the saved training Parquet files.
        output_dir: Directory to save the fitted preprocessors.
        sample_fraction: Fraction of the training data to load for fitting (e.g., 0.01 for 1%).
                         If specified, overrides sample_n_files.
        sample_n_files: Number of Parquet files to load from the training directory for fitting.
                        Used if sample_fraction is None. Defaults to 1.
        categorical_features: List of categorical columns (passed to fit_and_save_preprocessors).
        boolean_features: List of boolean columns (passed to fit_and_save_preprocessors).
        cyclical_features: List of cyclical columns (passed to fit_and_save_preprocessors).
    """
    print(f"\n--- Fitting Preprocessors on Sample from '{train_data_path}' ---")

    # --- Load Sample ---
    print("Loading training data sample...")
    if sample_fraction is not None:
        print(f"Sampling fraction: {sample_fraction}")
        # Use Dask to read and sample, then compute to get pandas DataFrame
        ddf_train_sample = dd.read_parquet(train_data_path).sample(frac=sample_fraction, random_state=42)
        with ProgressBar():
            train_sample_df = ddf_train_sample.compute()
    else:
        n_files = sample_n_files if sample_n_files is not None else 1
        print(f"Sampling {n_files} file(s)...")
        train_files = sorted(glob.glob(os.path.join(train_data_path, 'part*.parquet')))
        if not train_files:
            raise FileNotFoundError(f"No Parquet files found in {train_data_path}")
        files_to_read = train_files[:n_files]
        print(f"Reading: {files_to_read}")
        train_sample_df = pd.concat([pd.read_parquet(f) for f in files_to_read], ignore_index=True)

    print(f"Loaded training sample with shape: {train_sample_df.shape}")
    if train_sample_df.empty:
        raise ValueError("Loaded training sample is empty. Cannot fit preprocessors.")

    # --- Fit and Save ---
    # Assuming fit_and_save_preprocessors is defined in this file or imported
    fit_and_save_preprocessors(
        train_df=train_sample_df,
        output_dir=output_dir,
        categorical_features=categorical_features,
        boolean_features=boolean_features,
        cyclical_features=cyclical_features
    )

    print(f"--- Preprocessor Fitting on Sample Complete. Saved to '{output_dir}' ---")
    del train_sample_df # Free memory


# --- Applying Preprocessors with Dask ---

def apply_preprocessors_partition(
    df_partition: pd.DataFrame,
    preprocessor_dir: str,
    target_column: str = 'conversion_flag'
) -> pd.DataFrame:
    """
    Applies pre-fitted preprocessors to a PANDAS DataFrame partition.
    Designed to be used with Dask's map_partitions. Loads preprocessors from disk.

    Args:
        df_partition: A pandas DataFrame representing one partition of the data.
        preprocessor_dir: Directory where fitted preprocessors and feature lists are saved.
        target_column: Name of the target variable column.

    Returns:
        A pandas DataFrame containing the processed features and target.
        Features are stored as list/array objects within columns.
    """
    if df_partition.empty:
        # Return structure must match the 'meta' definition
        return pd.DataFrame({
            'categorical_data': pd.Series(dtype='object'),
            'numerical_data': pd.Series(dtype='object'),
            target_column: pd.Series(dtype='float32')
        })

    # --- Load Preprocessors ---
    try:
        categorical_encoder = joblib.load(os.path.join(preprocessor_dir, 'categorical_encoder.joblib'))
        numerical_scaler = joblib.load(os.path.join(preprocessor_dir, 'numerical_scaler.joblib'))
        categorical_features = joblib.load(os.path.join(preprocessor_dir, 'categorical_features.joblib'))
        boolean_features = joblib.load(os.path.join(preprocessor_dir, 'boolean_features.joblib'))
        cyclical_features = joblib.load(os.path.join(preprocessor_dir, 'cyclical_features.joblib'))
        numerical_features_to_scale = joblib.load(os.path.join(preprocessor_dir, 'numerical_features_to_scale.joblib'))
    except FileNotFoundError as e:
        print(f"ERROR in partition: Failed to load preprocessor files from {preprocessor_dir}. Details: {e}")
        raise # Re-raise to signal error in Dask task

    # --- Verify Features Exist in Partition ---
    required_features = categorical_features + boolean_features + cyclical_features + [target_column]
    missing_cols = [col for col in required_features if col not in df_partition.columns]
    if missing_cols:
        # Provide info about the partition's columns for debugging
        raise ValueError(
            f"Partition is missing required columns: {missing_cols}. "
            f"Partition columns: {df_partition.columns.tolist()}"
        )

    # --- Apply Transformations ---
    # 1. Categorical
    cat_df_slice = df_partition[categorical_features]
    encoded_cats = categorical_encoder.transform(cat_df_slice)
    encoded_cats[encoded_cats == -1] = 0 # Handle unknowns mapped to -1
    # No need to increment others if 0 is now the unknown category ID
    categorical_data_np = encoded_cats.astype(np.int64)

    # 2. Numerical (Booleans + Cyclical Scaled)
    numerical_df_for_scaling = pd.DataFrame(index=df_partition.index)
    for col in boolean_features:
        numerical_df_for_scaling[col] = df_partition[col].astype(float)
    # Handle potential missing temporal columns if graph definition skipped them
    if 'impression_hour' in df_partition.columns and 'impression_dayofweek' in df_partition.columns:
        hour = df_partition['impression_hour']
        day = df_partition['impression_dayofweek']
        numerical_df_for_scaling['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
        numerical_df_for_scaling['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
        numerical_df_for_scaling['day_sin'] = np.sin(2 * np.pi * day / 7.0)
        numerical_df_for_scaling['day_cos'] = np.cos(2 * np.pi * day / 7.0)
    else:
        # Need to handle missing cyclical features if they are in numerical_features_to_scale
        # Fill with 0 or raise error depending on requirements
        for col in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']:
            if col in numerical_features_to_scale and col not in numerical_df_for_scaling:
                 numerical_df_for_scaling[col] = 0.0 # Example: fill with 0
        print("Warning: Missing temporal columns in partition for scaling. Filled sin/cos with 0.")


    # Ensure columns are in the order expected by the scaler
    # Handle cases where a column might be missing from the scaling list used during fit
    cols_present_for_scaling = [col for col in numerical_features_to_scale if col in numerical_df_for_scaling.columns]
    numerical_df_ordered = numerical_df_for_scaling[cols_present_for_scaling]
    # Check if scaler expects columns that are now missing
    if len(cols_present_for_scaling) != len(numerical_features_to_scale):
        print(f"Warning: Mismatch between expected scaler features ({len(numerical_features_to_scale)}) and available features ({len(cols_present_for_scaling)}) in partition. Scaling may be incorrect or fail.")
        # Option: Pad missing columns with zeros/mean, or re-fit scaler if this happens often. For now, proceed with available.

    numerical_data_np = numerical_scaler.transform(numerical_df_ordered).astype(np.float32)

    # 3. Target
    target_data_np = df_partition[target_column].values.astype(np.float32)

    # --- Combine results into a single DataFrame ---
    # Store NumPy arrays as objects in the DataFrame columns.
    # Note: Parquet might not be the most efficient for this structure, but many
    # NN loaders can handle reading columns of lists/arrays.
    processed_df = pd.DataFrame({
        'categorical_data': [row for row in categorical_data_np], # Store as list
        'numerical_data': [row for row in numerical_data_np],     # Store as list
        target_column: target_data_np
    }, index=df_partition.index)

    return processed_df


def apply_and_save_preprocessed_data(
    split_paths: dict[str, str], # Dict like {'train': train_path, 'val': val_path, ...}
    preprocessor_dir: str,
    output_base_dir: str,
    target_column: str = 'conversion_flag'
):
    """
    Loads data splits (train, val, test) as Dask DataFrames, applies the pre-fitted
    preprocessors using map_partitions, and saves the results to Parquet.

    Args:
        split_paths: Dictionary mapping split names ('train', 'val', 'test') to
                     the paths of their saved Parquet directories.
        preprocessor_dir: Directory containing the fitted preprocessor .joblib files.
        output_base_dir: Base directory to save the processed NN-ready data.
                         Subdirectories for 'train', 'val', 'test' will be created.
        target_column: Name of the target column.
    """
    print(f"\n--- Applying Preprocessors to Full Data Splits using Dask map_partitions ---")
    os.makedirs(output_base_dir, exist_ok=True)

    # --- Define MetaData for map_partitions ---
    # Structure MUST match the DataFrame returned by apply_preprocessors_partition
    meta_processed = pd.DataFrame({
        'categorical_data': pd.Series(dtype='object'), # object for list/array
        'numerical_data': pd.Series(dtype='object'),   # object for list/array
        target_column: pd.Series(dtype='float32')
    })

    # --- Process each split ---
    for split_name, input_path in split_paths.items():
        print(f"\nProcessing '{split_name}' split from: {input_path}")
        if not os.path.exists(input_path):
            print(f"Warning: Input path not found for split '{split_name}'. Skipping: {input_path}")
            continue

        # Load split as Dask DataFrame
        ddf_split = dd.read_parquet(input_path)

        # Define the processing graph for this split
        ddf_processed = ddf_split.map_partitions(
            apply_preprocessors_partition,
            preprocessor_dir=preprocessor_dir,
            target_column=target_column,
            meta=meta_processed
        )

        # Define output path and save (executes computation for this split)
        output_path = os.path.join(output_base_dir, split_name)
        print(f"Defining save operation for '{split_name}' to: {output_path}")
        print(f"Triggering computation and saving for '{split_name}'...")
        with ProgressBar():
            ddf_processed.to_parquet(output_path, write_index=False, overwrite=True, compute=True)
        print(f"Computation and saving complete for '{split_name}'.")

    print(f"--- Preprocessing Application Complete. Results in '{output_base_dir}' ---")



# --- Main Orchestration Functions ---

# STAGE 1: Clean and Save Data
def clean_and_save_data():
    print("This script provides functions for Dask-based preprocessing.")
    from dask.distributed import Client, LocalCluster
    from dask.diagnostics import ProgressBar

    # --- Explicitly create a Dask Cluster/Client ---
    # Limit memory per worker to encourage spilling if needed, adjust based on your RAM
    # n_workers = os.cpu_count() # Start with number of CPU cores
    dask_temp_dir = "/Users/paramkapur/dask-worker-staging"
    os.makedirs(dask_temp_dir, exist_ok=True)
    cluster = LocalCluster(
            n_workers=1,            # Try fewer workers than cores initially
            threads_per_worker=4,   # Often better for CPU-bound tasks than hyperthreading
            memory_limit='14GB',      # Or '8GB', etc. - total RAM / n_workers roughly
            local_directory=dask_temp_dir,
            env={"DASK_PARTD_LOCATION": dask_temp_dir}
    )
    client = Client(cluster)
    print(f"Dask Dashboard Link: {client.dashboard_link}")


    # --- Configuration ---
    IMPRESSIONS_PATH = './data/snapshot_20250429/impressions/' # Input
    CONVERSIONS_PATH = './data/snapshot_20250429/conversions/' # Input
    MERGED_DATA_DIR = './data/merged'                           # Intermediate output
    PREPROCESSOR_DIR = './preprocessors'                        # Intermediate output
    PROCESSED_NN_DATA_DIR = './data/processed'               # Final output

    # Define ALL columns needed from impressions for merge, cleaning, final features
    # Be explicit to minimize data loaded/shuffled during merge
    IMPRESSION_COLS_NEEDED = [
        'unique_id', 'dttm_utc', 'user_agent', 'cnxn_type',
        'dma', 'country', 'prizm_premier_code', 'campaign_id', 'device_type'
        # Add any other columns used in cleaning, feature eng, or needed for the final NN input
    ]

    print("Defining Dask graph for cleaning and merging...")
    tasks = define_dask_cleaning_graph(
        impressions_path=IMPRESSIONS_PATH,
        conversions_path=CONVERSIONS_PATH,
        impression_cols_needed=IMPRESSION_COLS_NEEDED
    )
    print("Executing Dask graph for cleaning and merging...")
    row_counts = client.gather(tasks)            # dashboard shows 92 tasks
    print("Rows written per campaign:", row_counts)
    print("Dask graph execution complete.")

    print("Execution finished. Shutting down Dask client.")
    client.close()
    cluster.close()

# STAGE 2: Parse User Agent Strings

def parse_user_agent_strings():
    print("Parsing user agent strings...")
    # TODO: Implement user agent string parsing
    print("Parsing complete.")




# TODO: Update calls in main.py CLI to use these functions
# if __name__ == '__main__':
    

    # --- 3. Fit Preprocessors on a Sample of Training Data ---
    # Define feature groups (or let the function use defaults)
    # CAT_FEATURES = [...]
    # BOOL_FEATURES = [...]
    # CYC_FEATURES = [...]
    # fit_preprocessors_on_sample(
    #     train_data_path=train_path, # Use the path returned above
    #     output_dir=PREPROCESSOR_DIR,
    #     sample_n_files=2, # Example: use first 2 parquet files
    #     # categorical_features=CAT_FEATURES, # Pass if not using defaults
    #     # boolean_features=BOOL_FEATURES,
    #     # cyclical_features=CYC_FEATURES
    # )

    # --- 4. Apply Preprocessors to Full Splits and Save Final Data ---
    # split_paths_dict = {'train': train_path, 'val': val_path, 'test': test_path}
    # apply_and_save_preprocessed_data(
    #     split_paths=split_paths_dict,
    #     preprocessor_dir=PREPROCESSOR_DIR,
    #     output_base_dir=PROCESSED_NN_DATA_DIR,
    #     target_column='conversion_flag'
    # )
