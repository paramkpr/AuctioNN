"""
Stream Parquet → torch tensors in 2M row chunks,
so only one copy of data lives in RAM at a time.
"""

import random
import pyarrow.dataset as ds
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

BASE_DIR = Path("./data/processed")
SPLITS   = ["test"]

# FILE_NUMBERS = random.sample(range(0, 121), 5)
FILE_NUMBERS = [0, 1, 2, 3, 4]

CAT_COLS = [f"cat_{i}" for i in range(9)]
NUM_COLS = [f"num_{i}" for i in range(8)]
LABEL_COL = "conversion_flag"
ALL_COLS  = CAT_COLS + NUM_COLS + [LABEL_COL]

ROWS_PER_CHUNK = 2_000_000     # tweak if you want smaller/larger RAM chunks

def tensors_from_recordbatch(rb):
    """Arrow RecordBatch → three torch tensors (zero-copy)."""
    cat   = np.stack([np.asarray(rb[c]) for c in CAT_COLS], axis=1).astype("int32")
    num   = np.stack([np.asarray(rb[c]) for c in NUM_COLS], axis=1).astype("float32")
    label = np.asarray(rb[LABEL_COL]).astype("float32")
    return (
        torch.from_numpy(cat),
        torch.from_numpy(num),
        torch.from_numpy(label),
    )

def process_split(split_dir: Path, out_file: Path):
    cats, nums, labels = [], [], []
    total_rows = 0
    for file in FILE_NUMBERS:
        ds_split = ds.dataset(split_dir / f"part.{file}.parquet", format="parquet")
        batches = ds_split.to_batches(batch_size=ROWS_PER_CHUNK, columns=ALL_COLS)

        for rb in tqdm(batches, desc=f"{split_dir.name}/part.{file:05d}"):
            c, n, l = tensors_from_recordbatch(rb)
            cats.append(c)
            nums.append(n)
            labels.append(l)
            total_rows += l.size(0)
    cat   = torch.cat(cats,   dim=0).long()
    num   = torch.cat(nums,   dim=0).float()
    label = torch.cat(labels, dim=0).float()

    torch.save({"cat": cat, "num": num, "label": label}, out_file)
    print(f"✅  {split_dir.name}: {total_rows:,} rows → {out_file}")

for split in SPLITS:
    process_split(BASE_DIR / split, BASE_DIR / f"{split}_tensor_cache_local.pt")
