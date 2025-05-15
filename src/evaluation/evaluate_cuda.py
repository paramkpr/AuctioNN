# eval_wide_deep.py
"""
Evaluate a trained Wide & Deep model on the test split.
"""

import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryCalibrationError,
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryConfusionMatrix,
)
from tqdm.auto import tqdm
import pandas as pd
import argparse
from pathlib import Path

from models.network import ImpressionConversionNetwork

# ----------------------------------------------------------------------
# Config – adapt if your paths change
TEST_CACHE = "data/processed/test_tensor_cache_local.pt"
CHECKPOINTS = [
    f"runs/wad/20250505_234318/epoch_{i}.pt" for i in (4,)
]  # ← pick the best epoch
BATCH_SIZE = 2**8  # bigger ok for eval
NUM_WORKERS = 1
EXPORT_SCORES = False  # True → write scores.csv
# ----------------------------------------------------------------------


class InMemoryDataset(Dataset):
    """
    Wraps three pre-loaded tensors:
        cat  – LongTensor  (N, 9)
        num  – FloatTensor (N, 8)
        label – FloatTensor (N,)
    """

    def __init__(self, tensor_cache_path: str):
        obj = torch.load(tensor_cache_path, map_location="cpu")
        self.cat, self.num, self.label = obj["cat"], obj["num"], obj["label"]

    def __len__(self):
        return self.label.size(0)

    def __getitem__(self, idx):
        return self.cat[idx], self.num[idx], self.label[idx]


# ---- Dataset wrapper -------------------------------------------------


cat_enc = joblib.load("./preprocessors/categorical_encoder.joblib")
CARDINALITIES = [len(cat) for cat in cat_enc.categories_]  # +1 → reserve row for <UNK>


def load_model(ckpt_path: str, device: torch.device):
    model = ImpressionConversionNetwork(
        CARDINALITIES, numeric_dim=8, deep_embedding_dim=16
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device):
    criterion = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    n_samples = 0

    # --- metric objects ------------------------------------------------
    auroc = BinaryAUROC().to(device)
    ap = BinaryAveragePrecision().to(device)
    ece = BinaryCalibrationError(n_bins=15).to(device)

    acc = BinaryAccuracy().to(device)  # NEW
    prec = BinaryPrecision().to(device)  # NEW
    rec = BinaryRecall().to(device)  # NEW
    f1 = BinaryF1Score().to(device)  # NEW
    cm = BinaryConfusionMatrix().to(device)  # NEW
    # ------------------------------------------------------------------

    all_logits, all_labels = [], []

    for cat, num, y in tqdm(dataloader, desc="Evaluating"):
        cat, num, y = cat.to(device), num.to(device), y.to(device)

        logits = model(cat, num, return_logits=True)
        loss = criterion(logits, y)

        running_loss += loss.item() * y.size(0)
        n_samples += y.size(0)

        probs = torch.sigmoid(logits)

        # update metrics ----------------------------------------------
        auroc.update(probs, y.int())
        ap.update(probs, y.int())
        ece.update(probs, y.int())

        acc.update(probs, y.int())  # NEW
        prec.update(probs, y.int())  # NEW
        rec.update(probs, y.int())  # NEW
        f1.update(probs, y.int())  # NEW
        cm.update(probs, y.int())  # NEW
        # --------------------------------------------------------------

        if EXPORT_SCORES:
            all_logits.append(probs.cpu())
            all_labels.append(y.cpu())

    # final compute -----------------------------------------------------
    metrics = {
        "log_loss": running_loss / n_samples,
        "roc_auc": auroc.compute().item(),
        "pr_auc": ap.compute().item(),
        "ece": ece.compute().item(),
        "accuracy": acc.compute().item(),  # NEW
        "precision": prec.compute().item(),  # NEW
        "recall": rec.compute().item(),  # NEW
        "f1": f1.compute().item(),  # NEW
        "conf_mat": cm.compute().cpu().tolist(),  # NEW -- nice for printing / JSON
        "n_samples": n_samples,
    }
    # ------------------------------------------------------------------

    if EXPORT_SCORES:
        scores = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()
        pd.DataFrame({"prob": scores, "label": labels}).to_csv(
            "scores.csv", index=False
        )
        print("Saved scores.csv")

    return metrics


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Dataset & loader
    test_ds = InMemoryDataset(TEST_CACHE)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    for ckpt in CHECKPOINTS:
        # Model
        model = load_model(ckpt, device)

        # Eval
        metrics = evaluate(model, test_loader, device)
        print("\n=== Test metrics ===")
        for k, v in metrics.items():
            print(f"{k}: {v}" if isinstance(v, float) else f"{k}: {v}")

        # TensorBoard summary
        try:
            from torch.utils.tensorboard import SummaryWriter

            run_dir = Path("runs/wad_eval/" + ckpt)
            writer = SummaryWriter(run_dir)
            for k, v in metrics.items():
                if isinstance(v, float):
                    writer.add_scalar("test/" + k, v, 0)
            writer.close()
            print(f"TensorBoard logs → {run_dir}")
        except Exception as e:
            print("TensorBoard logging skipped:", e)


if __name__ == "__main__":
    main()
