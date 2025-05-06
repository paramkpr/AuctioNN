"""
Per-impression decision engine (Algorithm 1).
"""
from __future__ import annotations

from typing import Mapping, Sequence, Dict, List, Any # Added List and Any

import numpy as np
import torch

# Corrected import paths based on file structure
from src.campaign import Campaign
from src.exchange import Impression, Market, OnlinePreprocessor
from src.results_logger import ResultsLogger # Corrected import path


class DecisionLoop:
    def __init__(
        self,
        model: torch.jit.ScriptModule,
        campaigns: Sequence[Campaign],
        preproc: OnlinePreprocessor,
        market: Market,
        logger: ResultsLogger | None = None, 
        beta: float = 1.0, # bid shading factor for first-price auction, 1.0 = no shading
        tau: int = 3,  # number of impressions a user can see for the same campaign
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.preproc = preproc
        self.market = market
        self.logger = logger
        self.beta = beta
        self.tau = tau
        self.device = torch.device(device)

        # Cast list → dict for O(1) lookup
        self.campaigns: Dict[str, Campaign] = {c.id: c for c in campaigns}

        self._imp_counter = 0  # monotonically increasing impression id

    # –– public ––––––––––––––––––––––––––––––––––––––––––––––––––––
    def process(self, imp: Impression) -> None:
        """
        Execute one full pass of Algorithm 1 and (optionally) log the result.
        """
        self._imp_counter += 1
        # Assuming user_id is always present and is int-castable
        user_id = int(imp.features["user_id"])

        # 1. Eligibility gate ──────────────────────────────────────
        elig: list[Campaign] = [
            c
            for c in self.campaigns.values()
            if c.budget_remaining > 0 and c.ad_stock[user_id] < self.tau
        ]
        if not elig:           # nothing to bid on → skip
            return

        # 2. Batch model inference ─────────────────────────────────
        cat_tensors, num_tensors, carriers = [], [], []
        for c in elig:
            # The preprocessor returns (cat, num, target). We only need cat and num here.
            cat, num, _ = self.preproc(imp, c.id)
            cat_tensors.append(cat)
            num_tensors.append(num)
            carriers.append(c) # Store the campaign object itself

        cat_batch = torch.cat(cat_tensors, dim=0).to(self.device)
        num_batch = torch.cat(num_tensors, dim=0).to(self.device)

        with torch.no_grad():
            # Assuming model output is [batch_size, 1] or [batch_size], squeeze removes the dim=1
            pconv_batch = self.model(cat_batch, num_batch).squeeze().cpu().numpy()
            # If batch size is 1, squeeze might remove the batch dim entirely, making it a scalar.
            # np.atleast_1d ensures it's always an array.
            pconv_batch = np.atleast_1d(pconv_batch)

        # 3. Score & pick campaign ────────────────────────────────
        scores = []
        for pconv_val, camp in zip(pconv_batch, carriers):
            score: float
            # if camp.target_cpa is not None and camp.target_cpa > 0: # Avoid division by zero
            #     score = pconv_val / camp.target_cpa
            # else:
            # score = pconv_val * camp.value_per_conv
            score = pconv_val
            scores.append(score)

        best_idx = int(np.argmax(scores))
        chosen: Campaign = carriers[best_idx]
        pconv = float(pconv_batch[best_idx])

        # 4. Bid calculation ──────────────────────────────────────
        expected_value = pconv * chosen.value_per_conv
        bid = max(0.0, self.beta * expected_value) # Ensure bid is non-negative

        # 5. Market simulation ────────────────────────────────────
        market_price, won = self.market.simulate(bid)

        # 6. State updates ────────────────────────────────────────
        if won:
            chosen.budget_remaining -= market_price
            chosen.ad_stock[user_id] += 1 # defaultdict handles new users

        # 7. Logging  (add/remove fields as you please) ───────────
        if self.logger:
            log_data: Dict[str, Any] = {
                "imp_id": self._imp_counter,
                "user_id": user_id,
                "campaign_id": chosen.id,
                "bid": bid,
                "market_price": market_price,
                "win": won,
                "pconv": pconv,
                "expected_value": expected_value,
                "budget_remaining": chosen.budget_remaining,
            }
            self.logger.log(log_data)

