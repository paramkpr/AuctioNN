"""
campaign.py
This module provides a Campaign class that represents the campaigns that the
media company is running. It is also responsible for bootstrapping the campaigns
for simulation purposes.
"""

from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np


@dataclass(slots=True)
class Campaign:
    """
    Represents a campaign that the media company is running.
    """
    id: str
    value_per_conv: float
    target_cpa: float | None
    budget_remaining: float
    ad_stock: defaultdict[int, int]           # user_id → exposure count



def bootstrap_campaigns(
    clean_data_path: Path = Path("./data/clean_data.parquet"),
    seed: int = 42,
) -> list[Campaign]:
    """
    First, from clean_data we load the list of campaign ids that we have.
    Then, for each campaign, we create a Campaign object with synthetic,
    random values for the campaign parameters. We provide a seed for reproducibility.
    """
    # Load the campaign ids from the clean data
    df = pd.read_parquet(clean_data_path)
    campaign_ids = df["campaign_id"].unique()

    rng = np.random.default_rng(seed)

    # Create a list of Campaign objects
    campaigns = []
    for campaign_id in campaign_ids:
        value_per_conv = rng.uniform(1.0, 10.0) # $ per conversion
        target_cpa = rng.uniform(1.1 * value_per_conv, 3.0 * value_per_conv)
        budget_remaining = rng.uniform(500.0, 5000.0)  # total $ budget
        ad_stock = defaultdict(int)

        campaigns.append(
            Campaign(
                id=campaign_id,
                value_per_conv = value_per_conv,                
                target_cpa = target_cpa,
                budget_remaining = budget_remaining,          
                ad_stock = ad_stock,
            )
        )

    return campaigns
