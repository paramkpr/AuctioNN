import re
import os
import pandas as pd
import dask

from dask.dataframe.utils import make_meta
import dask.dataframe as dd

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

    #  discover campaign folders on the conversion side
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


if __name__ == "__main__":
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

    clean_and_save_data()

