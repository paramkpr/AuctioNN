"""
exchange.py
This module simulates an external ad exchange that the core decision loop interacts with.

The external ad exchange is responsible for two things:
- Generating a stream of bid-requests (impression features) 
- Setting the market price for each bid-request
    - Subsequently, deciding if the bid amount from the decision loop wins or not
"""

from dataclasses import dataclass
import os
import joblib
import torch
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.dataset as ds
import pandas as pd


@dataclass(slots=True, frozen=True)
class Impression:
    """
    A single impression I_t. Does not include the campaign_id.
    """
    features: dict[str, int | float | str]



class ImpressionGenerator:
    """
    Yields Impression objects drawn from a directory of Parquet files using PyArrow Datasets.
    The directory may contain arbitrarily large files.
    """

    def __init__(
        self,
        parquet_dir_path: str | Path,
        seed: int | None = 42,
        num_users: int | None = None, # number of users the market provides
    ) -> None:
        self._ds = ds.dataset(parquet_dir_path, format="parquet")
        # Eagerly load fragments and their row counts for efficient indexing
        self._fragments = list(self._ds.get_fragments())
        if not self._fragments:
             raise ValueError(f"No Parquet data found in directory: {parquet_dir_path}")
        self._fragment_row_counts = [frag.count_rows() for frag in self._fragments]
        self._cum_fragment_rows = np.cumsum([0] + self._fragment_row_counts)
        self._num_rows = self._cum_fragment_rows[-1]
        self._rng = np.random.default_rng(seed)
        self._num_users = num_users
        self._schema_names = self._ds.schema.names # Cache schema names

    # –– public –––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def stream(
        self,
        shuffle: bool = True,
        # rowgroup_cache parameter removed as caching is less direct with datasets/fragments
    ) -> Iterator[Impression]:
        """
        Generator that yields one `Impression` at a time from the dataset.

        Parameters
        ----------
        shuffle: return rows in a random order (default True).
        """
        order = (
            self._rng.permutation(self._num_rows) if shuffle else np.arange(self._num_rows)
        )

        for absolute_idx in order:
            frag_idx, offset = self._fragment_for_index(absolute_idx)
            current_fragment = self._fragments[frag_idx]

            # Read the specific row efficiently from the fragment
            # fragment.head(N) reads the first N rows, slice extracts the desired row
            row_table = current_fragment.head(offset + 1).slice(offset, 1)
            row = row_table.to_pydict()

            # Convert single-value lists → scalars, excluding campaign_id
            features = {k: v[0] for k, v in row.items() if k not in {"campaign_id"}}

            # Assigns a user_id to the impression if num_users is specified
            if self._num_users is not None:
                features["user_id"] = self._rng.choice(range(self._num_users), size=1, replace=True)[0]

            yield Impression(
                features=features,
            )

    # –– helpers ––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def _fragment_for_index(self, absolute_idx: int) -> tuple[int, int]:
        """
        Map an absolute row index → (fragment index, offset inside fragment).
        """
        # Find the fragment index using the precomputed cumulative row counts
        frag_idx = np.searchsorted(self._cum_fragment_rows, absolute_idx, side="right") - 1
        # Calculate the offset within that fragment
        offset = absolute_idx - self._cum_fragment_rows[frag_idx]
        return frag_idx, offset


# TODO: Refacor this into a 'Network' class maybe? or figure out a better place for it.
class OnlinePreprocessor:
    """
    Re-creates the *same* transformations done in `apply_preprocessors_to_split.py`.
    But for a single impression I_t from ImpressionGenerator and a given campaign_id.
    Output tensors have the *exact* same shape as those produced by `AuctionDataset`.
    """

    def __init__(self, preprocessor_dir: str | Path):
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)

        p = lambda name: os.path.join(preprocessor_dir, name)  # noqa: E731
        # Load preprocessors 
        self.cat_enc = joblib.load(p("categorical_encoder.joblib"))
        self.num_scal = joblib.load(p("numerical_scaler.joblib"))

        self.CATEGORICAL_FEATURES = {
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

        self.BOOLEAN_FEATURES = [
            "ua_is_mobile",
            "ua_is_tablet",
            "ua_is_pc",
            "ua_is_bot",
        ]

        self.CYCLICAL_FEATURES = [
            "impression_hour",
            "impression_dayofweek",
        ]

    
    def __call__(self, impression: Impression, campaign_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Processes a single impression and campaign_id into tensors matching those from AuctionDataset.
        
        Args:
            impression: An Impression object from ImpressionGenerator
            campaign_id: The campaign ID to use
            
        Returns:
            Tuple of (categorical_tensor, numerical_tensor) for the input to the model
        """
        features = {'campaign_id': int(campaign_id)}
        features.update(impression.features)
        df = pd.DataFrame([features])
            
        # -------- categorical --------
        cat_df = df[list(self.CATEGORICAL_FEATURES)].copy()
        for col in self.CATEGORICAL_FEATURES:
            if col in {"campaign_id", "dma"}:
                cat_df[col] = pd.to_numeric(cat_df[col], errors="raise").astype("int64")
            else:
                cat_df[col] = cat_df[col].astype("string").fillna("-1")
        cat_df = cat_df.astype(self.CATEGORICAL_FEATURES)

        cat_np = self.cat_enc.transform(cat_df).astype(np.int64)
        cat_np[cat_np == -1] = 0  # reserve 0 for “unknown”

        # -------- numerical --------
        num_df = pd.DataFrame(index=df.index)
        for col in self.BOOLEAN_FEATURES:
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

        num_np = self.num_scal.transform(num_df).astype(np.float32)

        # -------- target --------
        y = df["conversion_flag"].to_numpy(dtype=np.float32)

        # -------- flatten into columns --------
        processed = pd.DataFrame(index=df.index)
        for i in range(cat_np.shape[1]):
            processed[f"cat_{i}"] = cat_np[:, i]
        for i in range(num_np.shape[1]):
            processed[f"num_{i}"] = num_np[:, i]
        processed["conversion_flag"] = y
        

        CAT_COLS = [f"cat_{i}" for i in range(9)]
        NUM_COLS = [f"num_{i}" for i in range(8)]
        LABEL_COL = "conversion_flag"
        cat   = np.stack([np.asarray(processed[c]) for c in CAT_COLS], axis=1).astype("int32")
        num   = np.stack([np.asarray(processed[c]) for c in NUM_COLS], axis=1).astype("float32")
        label = np.asarray(processed[LABEL_COL]).astype("float32")
        return (
            torch.from_numpy(cat),
            torch.from_numpy(num),
            torch.from_numpy(label),
        )


class Market:
    """
    Simulates the actual external ad exchange market. 
    - Samples a market price for each bid-request / impression.
    - Provides a 'simulate' method that, given bid amount; impression; and campaign, 
        returns the result of the auction.
    """

    def __init__(self, median_cpm: float = 5, # $5 CPM = $0.005 per impression
                 sigma: float = 0.5,  # log-normal distribution sigma
                 seed: int = 42,  # random seed for reproducibility
                 ) -> None:
        self._median = median_cpm # / 1000.0      # convert CPM → per-imp $
        self._sigma = sigma
        self._rng = np.random.default_rng(seed)

    def _sample_price(self) -> float:
        """
        Sample a price from a log-normal distribution with the given median and sigma.
        """
        mu = np.log(self._median)
        return float(self._rng.lognormal(mean=mu, sigma=self._sigma))
    
    def simulate(self, bid_amount: float) -> bool:
        """
        Simulates the market price and returns the market price and the result of the auction.

        Args:
            bid_amount: The amount the decision loop is bidding.

        Returns:
            A tuple of (market_price, result).
        """
        market_price = self._sample_price()
        return market_price, bid_amount > market_price
