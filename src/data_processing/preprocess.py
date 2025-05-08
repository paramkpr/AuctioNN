"""
This script contains functions to process impression and conversion data using Dask
for large datasets, perform feature engineering, split the data, fit preprocessors
on a sample, apply preprocessors to the full dataset, and save the results.

Main orchestration should happen outside these functions, potentially in a
main script or notebook, calling these functions in sequence.
"""

import os
import joblib
import pandas as pd
import numpy as np
import pyarrow.dataset as ds
from dask.diagnostics import ProgressBar
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


CATEGORICAL_FEATURES = {
    "campaign_id": "int64",
    "cnxn_type": "string",
    "dma": "int16",
    "country": "string",
    "prizm_premier_code": "string",
    "ua_browser": "object",
    "ua_os": "object",
    "ua_device_family": "object",
    "ua_device_brand": "object",
}

BOOLEAN_FEATURES = [
    "ua_is_mobile",
    "ua_is_tablet",
    "ua_is_pc",
    "ua_is_bot",
]

CYCLICAL_FEATURES = [
    "impression_hour",
    "impression_dayofweek",
]


# ---------------------------------------------------------------------------
#  Core routine: fit encoders on a *given* DataFrame sample
# ---------------------------------------------------------------------------


def fit_and_save_preprocessors(
    train_df: pd.DataFrame,
    output_dir: str = "./preprocessors",
):
    """
    Fits an OrdinalEncoder (categoricals) + StandardScaler (numericals) on the
    provided sample and saves the fitted artefacts.

    Returns
    -------
    (ordinal_encoder, numerical_scaler, category_sizes)  – handy for notebooks
    """
    print(
        f"\n--- Fitting preprocessors on sample "
        f"(shape={train_df.shape}, output_dir='{output_dir}') ---"
    )
    os.makedirs(output_dir, exist_ok=True)

    # ───────────────────────────────────────────────────────────────────
    #  CATEGORICALS  – fill NA *then* cast so each column is uniform
    # ───────────────────────────────────────────────────────────────────
    print("• fitting OrdinalEncoder …")
    categorical_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.int64,
    )

    cat_df = train_df[list(CATEGORICAL_FEATURES)].copy()

    for col, target_dtype in CATEGORICAL_FEATURES.items():
        if col in {"campaign_id", "dma"}:
            # ⇒ numeric feature (e.g. campaign_id or dma)
            cat_df[col] = (
                cat_df[col]
                .fillna(-1)  # sentinel that survives casting
                .astype(target_dtype)  # e.g. "int16", "int64"
            )
        else:
            # ⇒ string / object feature: use pandas StringDtype for uniformity
            cat_df[col] = (
                cat_df[col]
                .astype("string")  # ensures NA is <NA>, not NaN/None/…
                .fillna("-1")  # now column is purely StringDtype
            )

    categorical_encoder.fit(cat_df)

    category_sizes = {
        col: len(cats) + 1
        for col, cats in zip(CATEGORICAL_FEATURES, categorical_encoder.categories_)
    }
    print("  ↳ done.")

    # ───────────────────────────────────────────────────────────────────
    #  NUMERICALS  (boolean flags + cyclical embeddings)
    # ───────────────────────────────────────────────────────────────────
    print("• preparing numerical matrix …")
    temp_num = pd.DataFrame(index=train_df.index)

    for col in BOOLEAN_FEATURES:
        if col in train_df.columns:
            temp_num[col] = train_df[col].astype(float)

    hour = train_df["impression_hour"]
    temp_num["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    temp_num["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    day = train_df["impression_dayofweek"]
    temp_num["day_sin"] = np.sin(2 * np.pi * day / 7.0)
    temp_num["day_cos"] = np.cos(2 * np.pi * day / 7.0)

    numerical_scaler = StandardScaler()
    numerical_scaler.fit(temp_num)
    print("  ↳ scaler fitted.")

    # ───────────────────────────────────────────────────────────────────
    #  PERSIST
    # ───────────────────────────────────────────────────────────────────
    print("• saving artefacts …")
    joblib.dump(
        categorical_encoder, os.path.join(output_dir, "categorical_encoder.joblib")
    )
    joblib.dump(numerical_scaler, os.path.join(output_dir, "numerical_scaler.joblib"))
    joblib.dump(category_sizes, os.path.join(output_dir, "category_sizes.joblib"))
    print("  ↳ all saved.")

    print("--- Preprocessor fitting complete. ---\n")
    return categorical_encoder, numerical_scaler, category_sizes


# ---------------------------------------------------------------------------
#  Helper: build a stratified 8.8 M‑row sample then delegate to fitter
# ---------------------------------------------------------------------------


def fit_preprocessors_on_sample(
    train_data_path: str,
    output_dir: str = "./preprocessors",
    n_rows_per_campaign: int = 100_000,
):
    """
    Samples `n_rows_per_campaign` rows from every `campaign_id=*` partition in
    `train_data_path`, concatenates them, then calls `fit_and_save_preprocessors`.
    """
    print(
        f"\n=== Sampling {n_rows_per_campaign} rows per campaign "
        f"from '{train_data_path}' ==="
    )

    # 1. discover partition folders (campaign_id=12345/ …)
    campaign_ids = sorted(
        int(p.split("=")[-1])
        for p in os.listdir(train_data_path)
        if p.startswith("campaign_id=")
        and os.path.isdir(os.path.join(train_data_path, p))
    )
    print(f"• found {len(campaign_ids)} campaigns.")

    # 2. build a dataset view with Hive partitioning
    dset = ds.dataset(train_data_path, format="parquet", partitioning="hive")

    # 3. only request the needed columns to keep scan light
    colset = list(CATEGORICAL_FEATURES) + BOOLEAN_FEATURES + CYCLICAL_FEATURES

    dfs = []
    for cid in campaign_ids:
        frag_df = (
            dset.scanner(
                filter=ds.field("campaign_id") == cid,
                columns=colset,
                use_threads=True,
            )
            .head(n_rows_per_campaign)
            .to_pandas()
        )
        dfs.append(frag_df)
    sample_df = pd.concat(dfs, ignore_index=True)
    print(f"• concatenated sample shape: {sample_df.shape}")

    # 4. fit + save
    fit_and_save_preprocessors(sample_df, output_dir)
    del sample_df  # free memory


# --- Applying Preprocessors with Dask ---


def apply_preprocessors_partition(
    df: pd.DataFrame,
    encoders,  # tuple (categorical_encoder, numerical_scaler)
    target_column: str = "conversion_flag",
) -> pd.DataFrame:
    cat_enc, num_scal = encoders

    # -------- categorical --------
    cat_df = df[list(CATEGORICAL_FEATURES)].copy()
    for col in CATEGORICAL_FEATURES:
        if col in {"campaign_id", "dma"}:
            cat_df[col] = pd.to_numeric(cat_df[col], errors="raise").astype("int64")
        else:
            cat_df[col] = cat_df[col].astype("string").fillna("-1")
    cat_df = cat_df.astype(CATEGORICAL_FEATURES)

    cat_np = cat_enc.transform(cat_df).astype(np.int64)
    cat_np[cat_np == -1] = 0  # reserve 0 for “unknown”

    # -------- numerical --------
    num_df = pd.DataFrame(index=df.index)
    for col in BOOLEAN_FEATURES:
        num_df[col] = df[col].astype(float)
    # This code performs cyclical encoding of time features:
    # - Converts hours (0-23) into circular coordinates using sin/cos
    # - Converts days of week (0-6) into circular coordinates using sin/cos
    # This preserves the cyclic nature of time - e.g. hour 23 is close to hour 0,
    # and Sunday (6) is close to Monday (0). Regular numeric encoding would lose
    # this cyclical relationship.
    hour = df["impression_hour"]
    day = df["impression_dayofweek"]
    num_df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    num_df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    num_df["day_sin"] = np.sin(2 * np.pi * day / 7.0)
    num_df["day_cos"] = np.cos(2 * np.pi * day / 7.0)

    num_np = num_scal.transform(num_df).astype(np.float32)

    # -------- target --------
    y = df[target_column].to_numpy(dtype=np.float32)

    # -------- flatten into columns --------
    processed = pd.DataFrame(index=df.index)
    for i in range(cat_np.shape[1]):
        processed[f"cat_{i}"] = cat_np[:, i]
    for i in range(num_np.shape[1]):
        processed[f"num_{i}"] = num_np[:, i]
    processed[target_column] = y
    return processed


def apply_and_save_preprocessed_data(
    input_path: str,
    preprocessor_dir: str,
    output_base_dir: str,
    target_column: str = "conversion_flag",
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
    dask_temp_dir = "/Users/paramkapur/dask-worker-staging"
    os.makedirs(dask_temp_dir, exist_ok=True)
    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        memory_limit="8GB",
        local_directory=dask_temp_dir,
        env={"DASK_PARTD_LOCATION": dask_temp_dir},
    )
    client = Client(cluster)
    print(f"Dask Dashboard Link: {client.dashboard_link}")

    print(
        "\n--- Applying Preprocessors to Full Data Splits using Dask map_partitions ---"
    )
    os.makedirs(output_base_dir, exist_ok=True)

    cat_enc = joblib.load(os.path.join(preprocessor_dir, "categorical_encoder.joblib"))
    num_scal = joblib.load(os.path.join(preprocessor_dir, "numerical_scaler.joblib"))

    n_cat = len(cat_enc.categories_)
    n_num = num_scal.mean_.shape[0]
    meta_cols = {f"cat_{i}": pd.Series(dtype="int64") for i in range(n_cat)}
    meta_cols.update({f"num_{i}": pd.Series(dtype="float32") for i in range(n_num)})
    meta_cols[target_column] = pd.Series(dtype="float32")
    meta_processed = pd.DataFrame(meta_cols)

    for split_name in ["train", "val", "test"]:
        print(f"\nProcessing '{split_name}' split from: {input_path}/{split_name}")

        # 1) Read only needed columns
        needed_cols = (
            list(CATEGORICAL_FEATURES)
            + BOOLEAN_FEATURES
            + CYCLICAL_FEATURES
            + [target_column]
        )

        ddf_split = dd.read_parquet(
            f"{input_path}/{split_name}",
            columns=needed_cols,
            blocksize="32MB",
            split_row_groups="adaptive",
        )

        ddf_processed = ddf_split.map_partitions(
            apply_preprocessors_partition,
            encoders=(cat_enc, num_scal),
            target_column=target_column,
            meta=meta_processed,
        )

        output_path = os.path.join(output_base_dir, split_name)
        print(f"Triggering computation and saving for '{split_name}'...")

        with ProgressBar():
            ddf_processed.to_parquet(
                output_path,
                write_index=False,
                overwrite=True,
                compression="snappy",
                ignore_divisions=True,
            )

        print(f"Computation and saving complete for '{split_name}'.")

    print(f"--- Preprocessing Application Complete. Results in '{output_base_dir}' ---")
    client.close()
    cluster.close()


if __name__ == "__main__":
    pass
    fit_preprocessors_on_sample(
        train_data_path="./data/cleaned/train",
        output_dir="./preprocessors",
        n_rows_per_campaign=100_000,  # 100 k × 88  ≈ 8.8 M rows
    )

    apply_and_save_preprocessed_data(
        input_path="./data/cleaned",
        preprocessor_dir="./preprocessors",
        output_base_dir="./data/processed",
        target_column="conversion_flag",
    )
