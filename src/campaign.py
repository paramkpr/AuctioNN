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
    clean_data_path: Path = Path("./data/cleaned/test/"),
    seed: int = 42,
) -> list[Campaign]:
    """
    First, from clean_data we load the list of campaign ids that we have.
    Then, for each campaign, we create a Campaign object with synthetic,
    random values for the campaign parameters. We provide a seed for reproducibility.
    """
    # Load the campaign ids from the clean data
    campaign_ids = [8334, 13411, 13505, 14108, 14213, 14546, 16007, 17562, 18997,
        19441, 19442, 40582, 41142, 42252, 42300, 42388, 42485, 42488,
        42517, 42540, 42569, 42580, 42593, 42751, 42838, 42844, 42915,
        42943, 42993, 43013, 43015, 43102, 43247, 43249, 43423, 43633,
        43662, 43787, 43789, 43813, 44002, 44120, 44126, 44165, 44424,
        44584, 44729, 44736, 44806, 44867, 44923, 45363, 45432, 45457,
        45459, 45460, 45461, 45482, 45488, 45783, 46536, 46729, 46975,
        47009, 47068, 47086, 47118, 47120, 47170, 47191, 47193, 47205,
        47242, 47245, 47253, 47259, 47322, 47362, 47381, 47386, 47451,
        47455, 47462, 47465, 47548, 47586, 47589, 47663]

    rng = np.random.default_rng(seed)

    # Create a list of Campaign objects
    campaigns = []
    for campaign_id in campaign_ids:
        value_per_conv = rng.uniform(20.0, 30.0) # $ per conversion
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
