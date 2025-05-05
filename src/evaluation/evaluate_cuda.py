# eval_wide_deep.py
"""
Evaluate a trained Wide & Deep model on the test split.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryCalibrationError,
)
from tqdm.auto import tqdm
import pandas as pd
import argparse
from pathlib import Path

from data_processing.datasets import InMemoryDataset
from models.network import ImpressionConversionNetwork

# ----------------------------------------------------------------------
# Config – adapt if your paths change
TEST_CACHE   = "processed/test_tensor_cache.pt"
CHECKPOINT   = "runs/wad/epoch_9.pt"     # ← pick the best epoch
BATCH_SIZE   = 131_072                   # bigger ok for eval
NUM_WORKERS  = 8
EXPORT_SCORES = True                    # True → write scores.csv
# ----------------------------------------------------------------------

# ---- Dataset wrapper -------------------------------------------------


CARDINALITIES = [88, 4, 211, 190, 69, 520, 27, 7678, 73]

def load_model(ckpt_path: str, device: torch.device):
    model = ImpressionConversionNetwork(CARDINALITIES, numeric_dim=8, deep_embedding_dim=16)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device):
    criterion = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    n_samples    = 0

    auroc = BinaryAUROC().to(device)
    ap    = BinaryAveragePrecision().to(device)
    ece   = BinaryCalibrationError(n_bins=15).to(device)

    all_logits, all_labels = [], []

    for cat, num, y in tqdm(dataloader, desc="Evaluating"):
        cat, num, y = cat.to(device), num.to(device), y.to(device)

        logits = model(cat, num, return_logits=True)
        loss   = criterion(logits, y)

        running_loss += loss.item() * y.size(0)
        n_samples    += y.size(0)

        preds = torch.sigmoid(logits)
        auroc.update(preds, y.int())
        ap.update(preds, y.int())
        ece.update(preds, y.int())

        if EXPORT_SCORES:
            all_logits.append(preds.cpu())
            all_labels.append(y.cpu())

    metrics = {
        "log_loss" : running_loss / n_samples,
        "roc_auc"  : auroc.compute().item(),
        "pr_auc"   : ap.compute().item(),
        "ece"      : ece.compute().item(),
        "n_samples": n_samples,
    }

    if EXPORT_SCORES:
        scores = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()
        pd.DataFrame({"prob": scores, "label": labels}).to_csv("scores.csv", index=False)
        print("Saved scores.csv")

    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=CHECKPOINT, help="Path to epoch_*.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & loader
    test_ds = InMemoryDataset(TEST_CACHE)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Model
    model = load_model(args.ckpt, device)

    # Eval
    metrics = evaluate(model, test_loader, device)
    print("\n=== Test metrics ===")
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.6f}" if isinstance(v, float) else f"{k:10s}: {v}")

    # TensorBoard summary
    try:
        from torch.utils.tensorboard import SummaryWriter
        run_dir = Path("runs/wad_eval")
        writer  = SummaryWriter(run_dir)
        for k, v in metrics.items():
            if isinstance(v, float):
                writer.add_scalar("test/"+k, v, 0)
        writer.close()
        print(f"TensorBoard logs → {run_dir}")
    except Exception as e:
        print("TensorBoard logging skipped:", e)

if __name__ == "__main__":
    main()
