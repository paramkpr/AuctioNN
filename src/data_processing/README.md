The pipeline looks like this to get the data in the right format for the model once we download the data from S3:

1. Run `merge_and_clean.py` to merge the data and clean it. This creates a folder called `data/merged` with the merged impressions and conversions. This is really expensive. We also perform the split into train/val/test sets here.
2. Run `repartition.py` to repartition the data. This creates a folder called `data/repartitioned` with the repartitioned data. This is pretty fast. It uses direct PyArrow underlying C++ operations. We need to do this
so that the parquet row-group size is around 64MB, so that when reading and doing batch operations, we don't
get OOM errors on Dask workers. 
3. Run `parse_ua_strings.py` to parse the user agent strings. This creates a folder called `data/cleaned` with the parsed user agent strings. This is probably as fast as it gets. It still takes an hour or so with
4 workers and 16GB memory total. At this point, we can delete the `data/merged` folder to save space. One 
imporovement that I will do next time is to also ensure that the data is cast to the right dtypes before writing to parquet and we have done required `fillna` operations on all columns to avoid mixed dtype pandas errors.
4. Finally, we can run `preprocess.py` to preprocess the data. This creates a folder called `data/processed` with the preprocessed data. Here we fit the encoders and standard scaler on a sample of the data and save them to the `preprocessors` folder. Then we use these preprocessors to transform the data and save it to the `data/processed` folder.

Now we can use the data in the `data/processed` folder for training. 


## Dataset Design Notes
| 🛠️                             | Recommendation                                                                                                              | Why                                                                 |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| **File layout**                 | ⚙︎  Keep row‑groups ≈ `50 k – 100 k` rows. You already write 64 MB files; each is one row‑group by default, so you’re good. | Each PyArrow scan can mmap one group and read **contiguously**.     |
| **Column order**                | Place target last (you do) so a *column projection* that excludes the target for inference is trivial.                      |                                                                     |
| **On‑disk dtypes**              | `cat_*` = **int64**, `num_*` = **float32** → zero casting inside `DataLoader`.                                              |                                                                     |
| **DataLoader**                  | Use one worker *per local CPU core* (not per GPU) and pin memory.                                                           | Parquet → NumPy → pinned Tensor avoids a host copy in `to(device)`. |
| **Prefetch**                    | `prefetch_factor=4` or 8.                                                                                                   | Hides parquet‑to‑tensor latency.                                    |
| **Batches as struct‑of‑arrays** | In `collate_fn`, stack cats and nums separately, e.g.  `(cats, nums, y)`.                                                   | Keeps `nn.Embedding` (cats) and MLP branch (nums) cache‑friendly.   |


```python
import pyarrow.dataset as ds
import torch, numpy as np

class ParquetIterable(torch.utils.data.IterableDataset):
    def __init__(self, path, batch_size):
        self.ds = ds.dataset(path, format="parquet")
        self.batch_size = batch_size
        self.cat_cols = [f"cat_{i}" for i in range( len(self.ds.schema.names)
                                                    if n.startswith("cat_"))]
        self.num_cols = [f"num_{i}" for i in range( len(self.ds.schema.names)
                                                    if n.startswith("num_"))]
        self.target = "conversion_flag"

    def __iter__(self):
        scanner = self.ds.scanner(columns=self.cat_cols + self.num_cols + [self.target],
                                  batch_size=self.batch_size)
        for record_batch in scanner.to_batches():
            arr = record_batch.to_pandas(types_mapper=np.asarray)
            cats = torch.from_numpy(arr[self.cat_cols].values).long()
            nums = torch.from_numpy(arr[self.num_cols].values).float()
            y    = torch.from_numpy(arr[[self.target]].values).float()
            yield cats, nums, y
```