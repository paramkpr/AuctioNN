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
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd


@dataclass(slots=True, frozen=True)
class Impression:
    """
    A single impression I_t. Does not include the campaign_id.
    """
    features: dict[str, int | float | str]



class ImpressionGenerator:
    """
    Yields Impression objects drawn from a Parquet file.
    The file may be arbitrarily large—as long as it has row groups.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        seed: int | None = 42,
        num_users: int | None = None, # number of users the market provides
    ) -> None:
        self._pf = pq.ParquetFile(parquet_path)
        self._num_rows = self._pf.metadata.num_rows
        self._rng = np.random.default_rng(seed)
        self._num_users = num_users

    # –– public –––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def stream(
        self,
        shuffle: bool = True,
        rowgroup_cache: bool = True,
    ) -> Iterator[Impression]:
        """
        Generator that yields one `Impression` at a time.

        Parameters
        ----------
        shuffle: return rows in a random order (default True).
        rowgroup_cache: keep the last materialised row group in memory.
        """
        order = (
            self._rng.permutation(self._num_rows) if shuffle else np.arange(self._num_rows)
        )

        current_rg_idx = -1
        current_rg_table: pa.Table | None = None

        for absolute_idx in order:
            rg_idx, offset = self._rowgroup_for_index(absolute_idx)

            # Load a new row-group only when needed
            if rg_idx != current_rg_idx:
                current_rg_table = self._pf.read_row_group(
                    rg_idx,
                )
                current_rg_idx = rg_idx

            assert current_rg_table is not None  # mypy
            # Arrow tables are columnar; pull the *row* efficiently
            row = current_rg_table.slice(offset, 1).to_pydict()
            # Convert single-value lists → scalars
            features = {k: v[0] for k, v in row.items() if k not in {"campaign_id"}}

            # assigns a user_id to the impression
            features["user_id"] = self._rng.choice(range(self._num_users), size=1, replace=True)[0]

            yield Impression(
                features=features,
            )

            # Optionally clear table if we do *not* want to keep it in memory
            if not rowgroup_cache:
                current_rg_table = None

    # –– helpers ––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def _rowgroup_for_index(self, absolute_idx: int) -> tuple[int, int]:
        """
        Map an absolute row index → (row-group index, offset inside group).
        """
        # ParquetFile API gives us row-group sizes → cumulative sums
        rg_sizes = [self._pf.metadata.row_group(i).num_rows for i in range(self._pf.num_row_groups)]
        cum = np.cumsum([0] + rg_sizes)
        rg_idx = np.searchsorted(cum, absolute_idx, side="right") - 1
        offset = absolute_idx - cum[rg_idx]
        return rg_idx, offset


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
        self.categorical_encoder = joblib.load(p("categorical_encoder.joblib"))
        self.numerical_scaler = joblib.load(p("numerical_scaler.joblib"))
        self.categorical_features = joblib.load(p("categorical_features.joblib"))
        self.boolean_features = joblib.load(p("boolean_features.joblib"))
        self.cyclical_features = joblib.load(p("cyclical_features.joblib"))
        self.numerical_features_to_scale = joblib.load(p("numerical_features_to_scale.joblib"))
    
    def __call__(self, impression: Impression, campaign_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Processes a single impression and campaign_id into tensors matching those from AuctionDataset.
        
        Args:
            impression: An Impression object from ImpressionGenerator
            campaign_id: The campaign ID to use
            
        Returns:
            Tuple of (categorical_tensor, numerical_tensor) for the input to the model
        """
        # Combine features with campaign_id for processing
        features = {**impression.features, 'campaign_id': campaign_id}
        
        # 1. Process categorical features
        # Create a pandas DataFrame with proper column names to avoid warnings
        categorical_features = self.categorical_features
        cat_data = {feature: [features.get(feature, None)] for feature in categorical_features}
        cat_df = pd.DataFrame(cat_data)
        
        # Encode
        encoded_cats = self.categorical_encoder.transform(cat_df)
        
        # Handle unknowns (-1 -> 0, shift others +1)
        encoded_cats[encoded_cats == -1] = 0
        encoded_cats[encoded_cats > -1] += 1
        categorical_tensor = torch.tensor(encoded_cats, dtype=torch.int64).to(self.device)
        
        # 2. Process numerical features (boolean + cyclical)
        # Create dictionary with feature values
        numerical_values = {}
        
        # Boolean features
        for feat in self.boolean_features:
            numerical_values[feat] = float(features.get(feat, False))
        
        # Cyclical features (hour, day of week)
        hour = features.get('impression_hour', 0)
        day = features.get('impression_dayofweek', 0)
        numerical_values['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
        numerical_values['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
        numerical_values['day_sin'] = np.sin(2 * np.pi * day / 7.0)
        numerical_values['day_cos'] = np.cos(2 * np.pi * day / 7.0)
        
        # Create DataFrame with proper column names
        num_df = pd.DataFrame({feat: [numerical_values.get(feat, 0.0)] for feat in self.numerical_features_to_scale})
        
        # Scale numerical features
        scaled_numerical = self.numerical_scaler.transform(num_df)
        numerical_tensor = torch.tensor(scaled_numerical, dtype=torch.float32).to(self.device)

        # Process target (conversion_flag that is always 0)
        target_tensor = torch.tensor(0.0, dtype=torch.float32).to(self.device)

        return (categorical_tensor, numerical_tensor, target_tensor)


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
