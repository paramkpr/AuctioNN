import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import joblib
import os
from tqdm import tqdm
import numpy as np

# Import necessary components from your project
from src.data_processing.datasets import AuctionDataset
from src.models.network import AuctionNetwork

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
        numerical_features_to_scale = joblib.load(os.path.join(preprocessor_dir, 'numerical_features_to_scale.joblib'))
        num_numerical_features = len(numerical_features_to_scale)
        # Load target data for pos_weight calculation
        train_tgt_path = os.path.join(processed_data_dir, "train_target_data.npy")
        train_targets_np = np.load(train_tgt_path)
    except FileNotFoundError as e:
        print(f"ERROR: Failed to load necessary preprocessor/data files. Details: {e}")
        raise
    print("Preprocessor info loaded successfully.")


    # --- Create Datasets from processed .npy files ---
    print("Creating Datasets from pre-processed files...")
    try:
        train_dataset = AuctionDataset(processed_data_dir=processed_data_dir, split_name='train')
        val_dataset = AuctionDataset(processed_data_dir=processed_data_dir, split_name='val')
        # Assuming test data would be loaded similarly if needed later
    except FileNotFoundError:
         print(f"ERROR: Failed to find train/val .npy files in {processed_data_dir}. Make sure preprocessing was run.")
         raise
    except ValueError as ve: # Catch shape mismatch error from Dataset init
         print(f"ERROR: {ve}")
         raise


    # --- Create DataLoaders ---
    print("Creating DataLoaders...")
    # Check if datasets are non-empty before creating loaders
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty.")
    if len(val_dataset) == 0:
        # If validation is empty, we might want to skip it or handle differently
        print("Warning: Validation dataset is empty. Training without validation.")
        val_loader = None # Set loader to None
    else:
         val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=min(4, os.cpu_count()), pin_memory=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=min(4, os.cpu_count()), pin_memory=True)


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
    print("Calculating positive weight for loss function...")
    # Use the loaded training targets
    neg_count = np.sum(train_targets_np == 0)
    pos_count = np.sum(train_targets_np == 1)
    pos_weight_value = float(neg_count / pos_count) if pos_count > 0 else 1.0
    print(f"Using pos_weight: {pos_weight_value:.4f}")
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
             target_batch = target_batch.to(resolved_device).unsqueeze(1) # Add dim for loss fn

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
