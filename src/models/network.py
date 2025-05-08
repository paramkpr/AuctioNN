import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Sequence


class ImpressionConversionNetwork(nn.Module):
    """
    A wide and deep network for conversion-rate prediction.

    Expected input per batch:
    ------------------------
    - categorical: LongTensor (B, N_cat)  # cat_0 ... cat_8 as *ordinal* ID
    - numerical: FloatTensor (B, N_num)   # num_0 ... num_7 already scaled

    Returns
    -------
    - p: FloatTensor (B,) predicted conversion rate
    """

    def __init__(
        self,
        categorical_cardinalities: list[int],
        numeric_dim: int = 8,
        deep_embedding_dim: int = 16,
        mlp_hidden: Sequence[int] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.n_cat = len(categorical_cardinalities)
        self.n_num = numeric_dim

        # Wide part
        self.wide_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, 1) for cardinality in categorical_cardinalities]
        )

        # Deep part
        self.deep_embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, deep_embedding_dim)
                for cardinality in categorical_cardinalities
            ]
        )

        # Deep MLP
        deep_input_dim = self.n_cat * deep_embedding_dim + self.n_num
        mlp_layers: list[nn.Module] = []
        prev = deep_input_dim
        for h in mlp_hidden:
            mlp_layers.append(nn.Linear(prev, h))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            prev = h
        mlp_layers.append(nn.Linear(prev, 1))  # final logit
        self.deep_mlp = nn.Sequential(*mlp_layers)

        # Initialise embeddings with small std so logits start near 0
        for emb in list(self.wide_embeddings) + list(self.deep_embeddings):
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)

    def forward(
        self,
        categorical: torch.Tensor,
        numerical: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        categorical : LongTensor (B, N_CAT)
        numeric     : FloatTensor (B, N_NUM)
        return_logits : set True to bypass sigmoid for BCEWithLogitsLoss

        Returns
        -------
        Tensor (B,) – probabilities or raw logits
        """

        # Wide part
        wide_logits = torch.zeros(
            categorical.size(0), 1, device=categorical.device, dtype=torch.float32
        )

        for i, emb in enumerate(self.wide_embeddings):
            wide_logits += emb(categorical[:, i])  # Shape: (B, 1)

        # Deep part
        deep_embeddings = [
            emb(categorical[:, i]) for i, emb in enumerate(self.deep_embeddings)
        ]  # List of (B, D) tensors
        deep_input = torch.cat(
            deep_embeddings + [numerical], dim=1
        )  # Shape: (B, N_cat * D + N_num)
        deep_logits = self.deep_mlp(deep_input)  # Shape: (B, 1)

        # Combine wide and deep parts
        logits = wide_logits + deep_logits  # Shape: (B, 1)

        # Apply sigmoid if not returning logits
        if not return_logits:
            logits = torch.sigmoid(logits)

        return logits.squeeze(-1)
