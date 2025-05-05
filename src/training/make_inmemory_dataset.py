# prepare_inmemory_tensors.py
import pyarrow.dataset as ds
import torch
import numpy as np
from pathlib import Path

BASE_DIR = Path("processed")           # root that holds train/ val/ test/
SPLITS   = ["train", "val", "test"]

CAT_COLS = [f"cat_{i}" for i in range(9)]
NUM_COLS = [f"num_{i}" for i in range(8)]
LABEL_COL = "conversion_flag"
ALL_COLS  = CAT_COLS + NUM_COLS + [LABEL_COL]

def parquet_dir_to_tensors(split_dir: Path):
    """Read every Parquet in split_dir → three torch tensors (cat, num, label)."""
    print(f"🔄  Loading {split_dir} …")
    # pyarrow treats the directory as one logical dataset
    table = ds.dataset(split_dir, format="parquet").to_table(columns=ALL_COLS)
    df    = table.to_pandas(split_blocks=True, ignore_metadata=True)  # zero‑copy

    cat   = torch.from_numpy(df[CAT_COLS].values.astype(np.int32)).long()
    num   = torch.from_numpy(df[NUM_COLS].values.astype(np.float32)).float()
    label = torch.from_numpy(df[LABEL_COL].values.astype(np.float32)).float()
    del df  # free the pandas frame
    return {"cat": cat, "num": num, "label": label}

for split in SPLITS:
    cache = parquet_dir_to_tensors(BASE_DIR / split)
    out_file = BASE_DIR / f"{split}_tensor_cache.pt"
    torch.save(cache, out_file)
    print(f"✅  Saved {out_file} ({len(cache['label']):,} rows)")
