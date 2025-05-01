#!/usr/bin/env python
# src/data_processing/repartition.py
from pathlib import Path
import shutil
import pyarrow.dataset as ds

# ------------------  write-time Parquet settings  ------------------
parquet_format = ds.ParquetFileFormat()
file_options = parquet_format.make_write_options(
    compression="snappy",
    write_statistics=False,
)

# ------------------  re-chunk each split  -------------------------
for split in ("train", "val", "test"):
    src_path = Path("data/merged") / split
    dst_path = Path("data/repartitioned") / split

    if dst_path.exists():  # start fresh
        shutil.rmtree(dst_path)

    src = ds.dataset(src_path, format="parquet")

    ds.write_dataset(
        data=src,
        base_dir=dst_path,
        format=parquet_format,
        file_options=file_options,
        partitioning=["campaign_id"],  # ← just the column names
        partitioning_flavor="hive",  # ← tell Arrow it’s Hive style
        use_threads=True,
        max_rows_per_file=1_200_000,  # ~32 MB if rows ≈160 B encoded
        max_rows_per_group=300_000,
        existing_data_behavior="overwrite_or_ignore",  # safe to re-run
    )

print("✅  Repartitioning done – go get some sleep!")
