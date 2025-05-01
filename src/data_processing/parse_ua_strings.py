import pandas as pd
from user_agents import parse as ua_parse
import pyarrow as pa
import dask.dataframe as dd

def parse_ua_partition(pdf: pd.DataFrame) -> pd.DataFrame:
    browsers, oss, fams, brands = [], [], [], []
    mobiles, tablets, pcs, bots = [], [], [], []

    for ua_str in pdf["user_agent"].fillna(""):
        ua = ua_parse(ua_str)
        browsers.append(ua.browser.family or "Unknown")
        oss.append(ua.os.family or "Unknown")
        fams.append(ua.device.family or "Unknown")
        brands.append(ua.device.brand or "Unknown")
        mobiles.append(bool(ua.is_mobile))
        tablets.append(bool(ua.is_tablet))
        pcs.append(bool(ua.is_pc))
        bots.append(bool(ua.is_bot))

    pdf = pdf.drop(columns=["user_agent"])
    pdf["ua_browser"]       = browsers
    pdf["ua_os"]            = oss
    pdf["ua_device_family"] = fams
    pdf["ua_device_brand"]  = brands
    pdf["ua_is_mobile"]     = mobiles
    pdf["ua_is_tablet"]     = tablets
    pdf["ua_is_pc"]         = pcs
    pdf["ua_is_bot"]        = bots

    return pdf


if __name__ == "__main__":
    from dask.distributed import Client
    from dask.distributed import LocalCluster
    import os

    # 4 workers, 1 thread each, 4 GB RAM each, spill to /tmp
    dask_temp_dir = "/Users/paramkapur/dask-worker-staging"
    os.makedirs(dask_temp_dir, exist_ok=True)
    cluster = LocalCluster(
            n_workers=5,            
            threads_per_worker=2,   
            memory_limit='3GB',      
            local_directory=dask_temp_dir,
            env={"DASK_PARTD_LOCATION": dask_temp_dir}
    )
    client = Client(cluster)
    print(f"Dask Dashboard Link: {client.dashboard_link}")

    # Read in the data
    meta_cols = {
        "campaign_id"          : "int64",
        "unique_id"            : "str",
        "impression_dttm_utc"  : "time64[ns]",
        "cnxn_type"            : "str",
        "user_agent"           : "str",
        "dma"                  : "int8", 
        "country"              : "str",
        "prizm_premier_code"   : "str",
        "device_type"          : "str",
        "conv_dttm_utc"        : "time64[ns]",
        "conversion_flag"      : "int8",
        "impression_hour"      : "Int32",
        "impression_dayofweek" : "Int32",
    }
    schema = pa.schema(meta_cols)

    for split in ["train", "val", "test"]:
        df = dd.read_parquet(f"./data/merged/{split}/", dataset={"schema": schema}, engine="pyarrow", split_row_groups="adaptive")

        meta = {
            "campaign_id"          : "int64",
            "unique_id"            : "str",
            "impression_dttm_utc"  : "datetime64[ns]",
            "cnxn_type"            : "str",
            "dma"                  : "int8", 
            "country"              : "str",
            "prizm_premier_code"   : "str",
            "device_type"          : "str",
            "conv_dttm_utc"        : "datetime64[ns]",
            "conversion_flag"      : "int8",
            "impression_hour"      : "Int32",
            "impression_dayofweek" : "Int32",
            "ua_browser"           : "str",
            "ua_os"                : "str",
            "ua_device_family"     : "str",
            "ua_device_brand"      : "str",
            "ua_is_mobile"         : "bool",
            "ua_is_tablet"         : "bool",
            "ua_is_pc"             : "bool",
            "ua_is_bot"            : "bool",
        }

        # transform
        parsed = df.map_partitions(parse_ua_partition, meta=meta)

        dtypes_to_cast = {
            "campaign_id":        "int64",
            "dma":                "int16",   # was int8 but needs up to at least 718
            "prizm_premier_code": "int16",   # likewise widen to avoid overflow
            "conversion_flag":    "int8",    # should still fit in -128…127
        }

        parsed = parsed.astype(dtypes_to_cast)

        # write to a new folder, partitioned by campaign_id
        parsed.to_parquet(
            f"./data/cleaned/{split}/",
            partition_on=["campaign_id"],
            engine="pyarrow",
            write_index=False,
            append=True,              # << important for restartability
            ignore_divisions=True,    # sections may not align exactly
        )

        print(f"Wrote {split} to ./data/cleaned/{split}/")

    print("Done!")
    client.close()
    cluster.close()
