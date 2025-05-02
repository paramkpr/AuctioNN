import pandas as pd
from user_agents import parse as ua_parse
import pyarrow as pa
import dask.dataframe as dd
from functools import lru_cache
from user_agents import parse as ua_parse0


@lru_cache(maxsize=100_000)
def ua_parse(ua_string: str):
    return ua_parse0(ua_string)


def parse_ua_partition(pdf: pd.DataFrame) -> pd.DataFrame:
    # 1) fillna once
    uas = pdf["user_agent"].fillna("").astype(str)

    # 2) find the unique strings
    uniques = uas.unique()

    # 3) parse each unique UA exactly once
    mapping = {}
    for ua in uniques:
        p = ua_parse(ua)
        mapping[ua] = (
            p.browser.family or "Unknown",
            p.os.family or "Unknown",
            p.device.family or "Unknown",
            p.device.brand or "Unknown",
            bool(p.is_mobile),
            bool(p.is_tablet),
            bool(p.is_pc),
            bool(p.is_bot),
        )

    # 4) map back to a list of 8‐tuples, in one pass
    tuples = uas.map(mapping)

    # 5) expand into a DataFrame of shape (len(pdf), 8)
    df8 = pd.DataFrame(
        tuples.tolist(),
        index=pdf.index,
        columns=[
            "ua_browser",
            "ua_os",
            "ua_device_family",
            "ua_device_brand",
            "ua_is_mobile",
            "ua_is_tablet",
            "ua_is_pc",
            "ua_is_bot",
        ],
    )

    # 6) drop the old column, join the new ones
    return pdf.drop(columns=["user_agent"]).join(df8)


if __name__ == "__main__":
    from dask.distributed import Client
    from dask.distributed import LocalCluster
    import os

    # 4 workers, 1 thread each, 4 GB RAM each, spill to /tmp
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

    for split in ["train", "val", "test"]:
        df = dd.read_parquet(
            f"./data/repartitioned/{split}/",
            engine="pyarrow",
            split_row_groups="adaptive",
            blocksize="64MB",
        )

        meta = {
            **{c: df.dtypes[c] for c in df.columns if c != "user_agent"},
            "ua_browser": object,
            "ua_os": object,
            "ua_device_family": object,
            "ua_device_brand": object,
            "ua_is_mobile": bool,
            "ua_is_tablet": bool,
            "ua_is_pc": bool,
            "ua_is_bot": bool,
        }

        # transform
        parsed = df.map_partitions(parse_ua_partition, meta=meta)

        dtypes_to_cast = {
            "campaign_id": "int64",
            "dma": "int16",  # was int8 but needs up to at least 718
            "conversion_flag": "int8",  # should still fit in -128…127
            "impression_hour": "int8",  # should still fit in -128…127
            "impression_dayofweek": "int8",  # should still fit in -128…127
        }

        parsed = parsed.astype(dtypes_to_cast)

        # write to a new folder, partitioned by campaign_id
        parsed.to_parquet(
            f"./data/cleaned/{split}/",
            partition_on=["campaign_id"],
            engine="pyarrow",
            write_index=False,
            overwrite=True,
            compression="snappy",
            ignore_divisions=True,  # sections may not align exactly
        )

        print(f"Wrote {split} to ./data/cleaned/{split}/")

    print("Done!")
    client.close()
    cluster.close()
