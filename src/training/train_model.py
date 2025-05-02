import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import joblib
import os
from tqdm import tqdm
import numpy as np
import pyarrow.dataset as ds
import pyarrow.compute as pc
# Import necessary components from your project
from data_processing.datasets import ParquetAuctionDataset
from models.network import AuctionNetwork

def count_pos_neg(processed_data_dir: str, split: str = "train",
                  target_col: str = "conversion_flag") -> tuple[int, int]:
    """
    Returns (#negatives, #positives) by scanning only Parquet footers +
    a fast Arrow aggregate – no full-table materialisation.
    """
    dset = ds.dataset(os.path.join(processed_data_dir, split), format="parquet")
    total = dset.count_rows()
    pos   = dset.count_rows(filter=pc.field(target_col) == 1)
    neg   = total - pos
    return neg, pos

def infer_feature_counts(processed_data_dir: str, split: str = "train") -> tuple[int, int]:
    """
    Inspects the schema once and returns (n_categorical, n_numerical).
    """
    schema = ds.dataset(os.path.join(processed_data_dir, split), format="parquet").schema
    n_cat  = sum(1 for n in schema.names if n.startswith("cat_"))
    n_num  = sum(1 for n in schema.names if n.startswith("num_"))
    return n_cat, n_num



def run_training(
    processed_data_dir: str,
    preprocessor_dir: str,
    save_model_path: str = './models/best_auction_network.pth',
    epochs: int = 10,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    embedding_dim: int = 32,
    hidden_dims: list[int] = [128, 64],
    dropout_rate: float = 0.3,
    random_state: int = 42,
    device: str | None = None
):
    """
    Loads preprocessors (for model dims), pre-processed data arrays,
    sets up datasets/dataloaders, defines the model, and runs the training loop.

    Args:
        processed_data_dir: Directory containing the processed .npy files for train/val splits.
        preprocessor_dir: Directory containing fitted preprocessors (used for category_sizes).
        save_model_path: Path to save the best trained model state dictionary.
        epochs: Number of training epochs.
        batch_size: Batch size for training and validation.
        learning_rate: Learning rate for the Adam optimizer.
        embedding_dim: Dimension for categorical embeddings.
        hidden_dims: List of hidden layer dimensions.
        dropout_rate: Dropout rate for the model.
        random_state: Seed for torch random number generation.
        device: Device to train on ('cuda', 'cpu', or None for auto-detect).
    """
    print("--- Starting Training Process ---")

    # --- Device Setup ---
    if device is None:
        resolved_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        resolved_device = device
    print(f"Using device: {resolved_device}")
    torch.manual_seed(random_state)
    if resolved_device == "cuda":
        torch.cuda.manual_seed_all(random_state)

    # --- Load *Necessary* Preprocessors (category sizes, feature count) ---
    print(f"Loading preprocessor info from: {preprocessor_dir}")
    try:
        # Load only what's needed for model init and loss weight
        category_sizes = joblib.load(os.path.join(preprocessor_dir, 'category_sizes.joblib'))
        _, num_numerical_features = infer_feature_counts(processed_data_dir, "train")

    except FileNotFoundError as e:
        print(f"ERROR: Failed to load necessary preprocessor/data files. Details: {e}")
        raise
    print("Preprocessor info loaded successfully.")


    # ---- Create Datasets & DataLoaders ---------------------------------
    train_dir = os.path.join(processed_data_dir, "train")
    val_dir   = os.path.join(processed_data_dir, "val")

    train_dataset = ParquetAuctionDataset(train_dir, batch_rows=batch_size*4)
    val_dataset   = ParquetAuctionDataset(val_dir,   batch_rows=batch_size*4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,         # IterableDataset must be False
        num_workers=min(4, os.cpu_count()),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count()),
        pin_memory=True,
    )

    # --- Initialize Model ---
    print("Initializing model...")
    model = AuctionNetwork(
        category_sizes=category_sizes,
        num_numerical_features=num_numerical_features,
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate
    ).to(resolved_device)
    print(model)

    # --- Define Loss Function and Optimizer ---
    print("Counting positives/negatives …")
    neg_count, pos_count = count_pos_neg(processed_data_dir, "train", "conversion_flag")
    pos_weight_value = float(neg_count / pos_count) if pos_count > 0 else 1.0
    print(f"Pos weight: {pos_weight_value:.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=resolved_device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training Loop ---
    best_val_loss = float('inf')
    print(f"\n--- Starting Training for {epochs} Epochs ---")

    for epoch in range(epochs):
        # --- Training Phase (largely unchanged) ---
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for cat_batch, num_batch, target_batch in train_loop:
             cat_batch = cat_batch.to(resolved_device)
             num_batch = num_batch.to(resolved_device)
             target_batch = target_batch.to(resolved_device)

             optimizer.zero_grad()
             outputs = model(cat_batch, num_batch)
             loss = criterion(outputs, target_batch)
             loss.backward()
             optimizer.step()

             train_loss += loss.item()
             train_loop.set_postfix(loss=loss.item())
        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        avg_val_loss = float('nan') # Default if no validation
        accuracy = float('nan')
        if val_loader: # Only run validation if loader exists
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
            with torch.no_grad():
                for cat_batch, num_batch, target_batch in val_loop:
                     cat_batch = cat_batch.to(resolved_device)
                     num_batch = num_batch.to(resolved_device)
                     target_batch = target_batch.to(resolved_device).unsqueeze(1)
                     val_targets.append(target_batch.cpu())

                     outputs = model(cat_batch, num_batch)
                     loss = criterion(outputs, target_batch)
                     val_loss += loss.item()

                     preds = torch.sigmoid(outputs)
                     val_preds.append(preds.cpu())
                     val_loop.set_postfix(loss=loss.item())

            avg_val_loss = val_loss / len(val_loader)
            all_preds = torch.cat(val_preds)
            all_targets = torch.cat(val_targets)
            accuracy = ((all_preds > 0.5).float() == all_targets).float().mean().item()

            print(f"Epoch {epoch+1}/{epochs} => Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.4f}")

            # --- Save Best Model (only if validation occurred) ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
                torch.save(model.state_dict(), save_model_path)
                print(f"  -> Saved best model to {save_model_path} (Val Loss: {best_val_loss:.4f})")
        else:
             # No validation, just print train loss
             print(f"Epoch {epoch+1}/{epochs} => Train Loss: {avg_train_loss:.4f} | (No validation)")
             # Optionally save model based on training loss or just save last epoch
             # if avg_train_loss < best_train_loss: # Example if saving on train loss
             #    best_train_loss = avg_train_loss
             #    torch.save(model.state_dict(), save_model_path)
             #    print(f"  -> Saved model based on train loss: {save_model_path}")


    print("--- Training Complete ---")
    if val_loader:
        print(f"Best validation loss achieved: {best_val_loss:.4f}")
        print(f"Best model saved to: {save_model_path}")
    else:
        print("Training finished (no validation performed).")
        # Consider saving the final model state if no validation
        final_model_path = save_model_path.replace('.pth', '_final.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model state saved to: {final_model_path}")


if __name__ == "__main__":
    run_training(
        processed_data_dir="./data/processed",
        preprocessor_dir="./preprocessors",
        save_model_path="./models/best_02052025_01.pth",
        epochs=1,
        batch_size=2048,
    )
