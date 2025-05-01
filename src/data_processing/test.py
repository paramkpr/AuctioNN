import pyarrow.dataset as ds

sample_file = "data/merged/train/campaign_id=17562/part.2.parquet"  # or any one file
tbl = ds.dataset(sample_file).to_table()  # loads it into memory once
bpr = tbl.nbytes / tbl.num_rows  # ≈ bytes per row
print(f"{bpr:.1f} bytes/row")
