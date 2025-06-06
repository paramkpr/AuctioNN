"""
Per-impression decision engine (Algorithm 1).
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence, Dict, List, Any  # Added Callable

import numpy as np
import torch

# Corrected import paths based on file structure
from src.campaign import Campaign
from src.exchange import Impression, Market, OnlinePreprocessor
from src.results_logger import ResultsLogger  # Corrected import path


class DecisionLoop:
    def __init__(
        self,
        predictor: torch.jit.ScriptModule
        | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        campaigns: Sequence[Campaign],
        preproc: OnlinePreprocessor,
        market: Market,
        logger: ResultsLogger | None = None,
        beta: float = 1.0,  # bid shading factor for first-price auction, 1.0 = no shading
        tau: int = 3,  # number of impressions a user can see for the same campaign
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        # Move predictor to device and set to eval mode if it's a ScriptModule
        if isinstance(predictor, torch.jit.ScriptModule):
            self.predictor = predictor.to(self.device).eval()
        else:
            # Assume the callable handles its own device placement if needed
            self.predictor = predictor
        self.preproc = preproc
        self.market = market
        self.logger = logger
        self.beta = beta
        self.tau = tau

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
        if not elig:  # nothing to bid on → skip
            # Consider adding logging here if needed
            # print(f"No eligible campaigns for user {user_id}")
            return

        # 2. Batch model inference / heuristic prediction ───────────
        cat_tensors, num_tensors, carriers = [], [], []
        for c in elig:
            # The preprocessor returns (cat, num, target). We only need cat and num here.
            cat, num, _ = self.preproc(imp, c.id)
            cat_tensors.append(cat)
            num_tensors.append(num)
            carriers.append(c)  # Store the campaign object itself

        # Ensure tensors are created even if empty (shouldn't happen due to eligibility check, but safer)
        if not cat_tensors or not num_tensors:
            return  # Or handle appropriately

        cat_batch = torch.cat(cat_tensors, dim=0).to(self.device)
        num_batch = torch.cat(num_tensors, dim=0).to(self.device)

        pconv_batch_tensor: torch.Tensor
        with torch.no_grad():  # Keep no_grad for consistency, harmless for heuristics
            # predictor could be the ScriptModule or a Callable
            pconv_batch_tensor = self.predictor(cat_batch, num_batch)

        # Ensure output is on CPU and converted to numpy
        pconv_batch = pconv_batch_tensor.squeeze().cpu().numpy()
        # If batch size is 1, squeeze might remove the batch dim entirely, making it a scalar.
        # np.atleast_1d ensures it's always an array.
        pconv_batch = np.atleast_1d(pconv_batch)

        # Ensure number of predictions matches number of eligible campaigns
        if len(pconv_batch) != len(carriers):
            # Handle error: mismatch between predictions and campaigns
            # This might indicate an issue with the predictor's output shape
            print(
                f"Error: Mismatch in prediction batch size ({len(pconv_batch)}) and eligible campaigns ({len(carriers)})"
            )
            return

        # 3. Score & pick campaign ────────────────────────────────
        scores = []
        for pconv_val, camp in zip(pconv_batch, carriers):
            score: float
            # pconv_val = camp.true_conv_rate
            score = pconv_val * camp.value_per_conv
            scores.append(score)

        # Handle case where scores might be empty if pconv_batch was empty or mismatched
        if not scores:
            return  # Or handle appropriately

        best_idx = int(np.argmax(scores))
        # best_idx = np.random.randint(len(scores))
        chosen: Campaign = carriers[best_idx]
        pconv = float(pconv_batch[best_idx])

        # 4. Bid calculation ──────────────────────────────────────
        expected_value = pconv * chosen.value_per_conv
        bid = max(0.0, self.beta * expected_value)  # Ensure bid is non-negative

        # 5. Market simulation ────────────────────────────────────
        market_price, won = self.market.simulate(bid)

        # 6. State updates ────────────────────────────────────────
        if won:
            chosen.budget_remaining -= market_price
            chosen.ad_stock[user_id] += 1  # defaultdict handles new users

        # 7. Logging  (add/remove fields as you please) ───────────
        if self.logger:
            log_data: Dict[str, Any] = {
                "imp_id": self._imp_counter,
                "imp_campaign_id_sampled_from": imp.campaign_id_sampled_from,
                "user_id": user_id,
                "campaign_id": chosen.id,
                "campaign_value_per_conv": chosen.value_per_conv,
                "expected_value": expected_value,
                "bid": bid,
                "market_price": market_price,
                "win": won,
                "pconv": pconv,
                "utility": (expected_value - market_price)
                if won
                else 0.0,  # Utility is 0 if not won
                "budget_remaining": chosen.budget_remaining,
                "eligible_campaigns": len(elig),  # Added for context
                "chosen_score": scores[best_idx],  # Added for context
            }
            self.logger.log(log_data)
