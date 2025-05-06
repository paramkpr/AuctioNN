import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm.auto import tqdm
import os
from typing import Optional
from datetime import datetime

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


def make_ratio_sampler(labels: torch.Tensor, k: int = 4) -> WeightedRandomSampler:
    """Return a sampler that draws 1 pos for every k negs (approx)."""
    pos_weight = 1.0
    neg_weight = 1.0 / k
    weights = torch.where(labels == 1, pos_weight, neg_weight)
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    auc_metric: BinaryAUROC,
    scaler: torch.amp.GradScaler,
    copy_stream: torch.cuda.Stream,
    # ap_metric: BinaryAveragePrecision,
    writer: SummaryWriter,
    phase: str,
    epoch: int,
    device: torch.device,
    global_step: int,
    max_batches: Optional[int] = None,
):
    """One full pass over `dataloader` (train or val)."""
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    total_batches = len(dataloader)
    pbar = tqdm(enumerate(dataloader, 1), total=total_batches, disable=os.getenv("CI"))
    for batch_idx, (cat, num, y) in pbar:
        with torch.cuda.stream(copy_stream):
            cat = cat.to(device, non_blocking=True, dtype=torch.int32)
            num = num.to(device, non_blocking=True).half()   # fp16
            y   = y.to(device, non_blocking=True)
        copy_stream.wait_stream(torch.cuda.current_stream())

        with torch.amp.autocast("cuda"):
            logits = model(cat, num, return_logits=True)
            loss = criterion(logits, y)

        if is_train:
            scaler.scale(loss).backward()              # <-- use GradScaler if you’re in AMP
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # ---- metrics ----------------------------------------------------
        preds  = torch.sigmoid(logits.detach())
        auc_metric.update(preds, y.int())
        # ap_metric.update(preds_cpu, labels_cpu.int())

        # ---- tensorboard per-100 ------------------------------------------
        if global_step % max(1, total_batches // 100) == 0:
            running_loss += loss.item() * y.size(0)

            writer.add_scalar(f"{phase}/batch_loss", loss.item(), global_step)
            if is_train:
                for i, pg in enumerate(optimizer.param_groups):
                    writer.add_scalar(f"lr/group_{i}", pg["lr"], global_step)

            pbar.set_description(
                f"{phase}  Epoch {epoch}  Loss {loss.item():.4f}"
            )

        global_step += 1
        if max_batches and batch_idx >= max_batches:
            break

    # ---- epoch‑level summaries ----------------------------------------
    epoch_loss = running_loss / (len(dataloader.dataset))
    epoch_auc = auc_metric.compute().item()
    # epoch_ap = ap_metric.compute().item()

    writer.add_scalar(f"{phase}/epoch_loss", epoch_loss, epoch)
    writer.add_scalar(f"{phase}/epoch_auc", epoch_auc, epoch)
    # writer.add_scalar(f"{phase}/epoch_pr_auc", epoch_ap, epoch)

    auc_metric.reset()
    # ap_metric.reset()
    return epoch_loss, global_step


def train(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    batch_size: int = 65_536,
    num_epochs: int = 5,
    pos_neg_ratio: int = 4,
    log_dir: str = "./runs/wide_deep",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device)
    model.to(device)

    writer = SummaryWriter(log_dir)
    imbalance_train = (len(train_ds) - train_ds.label.sum()) / train_ds.label.sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(imbalance_train, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler("cuda")
    copy_stream = torch.cuda.Stream()


    # --- torchmetrics objects reused each epoch ------------------------
    auc_metric = BinaryAUROC(thresholds=256).to(device)
    # ap_metric  = BinaryAveragePrecision().to("cpu")

    # --- dataloaders ---------------------------------------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    global_step = 0
    for epoch in range(num_epochs):
        # ---- training --------------------------------------------------
        _, global_step = run_epoch(
            model, train_loader, criterion, optimizer,
            auc_metric, scaler, copy_stream, writer, "train",
            epoch, device, global_step
        )

        # ---- validation ------------------------------------------------
        with torch.no_grad():
            run_epoch(
                model, val_loader, criterion, None,
                auc_metric, scaler, copy_stream, writer, "val",
                epoch, device, global_step
            )

        scheduler.step()

        # ---- save checkpoint ------------------------------------------
        ckpt_path = os.path.join(log_dir, f"epoch_{epoch}.pt")
        torch.save({"epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict()}, ckpt_path)

    writer.close()

# ------------------------------------------------------------
# 4️⃣  Usage example
# ------------------------------------------------------------
if __name__ == "__main__":
    from models.network import ImpressionConversionNetwork   # replace with your module path
    import joblib

    cat_enc = joblib.load("./preprocessors/categorical_encoder.joblib")
    CARDINALITIES = [len(cat) for cat in cat_enc.categories_] # +1 → reserve row for <UNK>

    model = ImpressionConversionNetwork(
        categorical_cardinalities=CARDINALITIES,
        numeric_dim=8,
        deep_embedding_dim=16,
        mlp_hidden=(128, 64),
        dropout=0.2,
    )
    model = torch.compile(model, mode="reduce-overhead")

    print("Loadded model successfully")

    print("Loading train and val datasets, adding to memory...")

    train_ds = InMemoryDataset("./data/processed/train_tensor_cache.pt")
    val_ds   = InMemoryDataset("./data/processed/val_tensor_cache.pt")

    print("starting training...")

    train(
        model           = model,
        train_ds        = train_ds,
        val_ds          = val_ds,
        batch_size      = 2**14,    # fits on A100‑40GB with mixed precision
        num_epochs      = 10,
        pos_neg_ratio   = 4,         # 1 positive : 4 negatives sampler
        log_dir         = "./runs/wad/" + datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
