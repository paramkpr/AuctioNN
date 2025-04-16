import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import joblib
import os
from tqdm import tqdm
import numpy as np

# Import necessary components from your project
from src.data_processing.datasets import AuctionDataset
from src.models.network import AuctionNetwork

def run_training(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    preprocessor_dir: str,
    save_model_path: str = './models/best_auction_network.pth',
    target_column: str = 'conversion_flag',
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
    Loads preprocessors, sets up datasets/dataloaders from provided DataFrames,
    defines the model, and runs the training/validation loop.

    Args:
        train_df: DataFrame for the training set.
        val_df: DataFrame for the validation set.
        preprocessor_dir: Directory containing fitted preprocessors.
        save_model_path: Path to save the best trained model state dictionary.
        target_column: Name of the target variable.
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

    # --- Load Preprocessors ---
    print(f"Loading preprocessors from: {preprocessor_dir}")
    try:
        categorical_encoder = joblib.load(os.path.join(preprocessor_dir, 'categorical_encoder.joblib'))
        numerical_scaler = joblib.load(os.path.join(preprocessor_dir, 'numerical_scaler.joblib'))
        category_sizes = joblib.load(os.path.join(preprocessor_dir, 'category_sizes.joblib'))
        categorical_features = joblib.load(os.path.join(preprocessor_dir, 'categorical_features.joblib'))
        boolean_features = joblib.load(os.path.join(preprocessor_dir, 'boolean_features.joblib'))
        cyclical_features = joblib.load(os.path.join(preprocessor_dir, 'cyclical_features.joblib'))
        numerical_features_to_scale = joblib.load(os.path.join(preprocessor_dir, 'numerical_features_to_scale.joblib'))
        num_numerical_features = len(numerical_features_to_scale)
    except FileNotFoundError as e:
        print(f"ERROR: Failed to load preprocessor files from {preprocessor_dir}. Did you run 'fit-preprocessors'? Details: {e}")
        raise
    print("Preprocessors loaded successfully.")

    # --- DataFrames are now passed directly ---
    print(f"Using provided DataFrames: Train={train_df.shape}, Validation={val_df.shape}")

    # --- Create Datasets ---
    common_args = {
        'target_column': target_column,
        'categorical_encoder': categorical_encoder,
        'numerical_scaler': numerical_scaler,
        'categorical_features': categorical_features,
        'boolean_features': boolean_features,
        'cyclical_features': cyclical_features,
        'numerical_features_to_scale': numerical_features_to_scale
    }
    print("Creating Datasets...")
    train_dataset = AuctionDataset(dataframe=train_df, **common_args)
    val_dataset = AuctionDataset(dataframe=val_df, **common_args)

    # --- Create DataLoaders ---
    print("Creating DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=min(4, os.cpu_count()), pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=min(4, os.cpu_count()), pin_memory=True)

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
    # Calculate pos_weight based on the training data split only
    print("Calculating positive weight for loss function...")
    target_counts = train_df[target_column].value_counts() # Use train_df now
    neg_count = target_counts.get(0, 0)
    pos_count = target_counts.get(1, 0)
    pos_weight_value = float(neg_count / pos_count) if pos_count > 0 else 1.0
    print(f"Using pos_weight: {pos_weight_value:.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=resolved_device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training Loop ---
    best_val_loss = float('inf')
    print(f"\n--- Starting Training for {epochs} Epochs ---")

    for epoch in range(epochs):
        # --- Training Phase ---
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

                # Store predictions (sigmoid needed for accuracy/AUC calc)
                preds = torch.sigmoid(outputs)
                val_preds.append(preds.cpu())
                val_loop.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)

        # --- Calculate Validation Metrics (Example: Accuracy) ---
        all_preds = torch.cat(val_preds)
        all_targets = torch.cat(val_targets)
        # Simple accuracy - threshold at 0.5
        accuracy = ((all_preds > 0.5).float() == all_targets).float().mean().item()
        # TODO: Consider adding AUC calculation using sklearn.metrics.roc_auc_score(all_targets.numpy(), all_preds.numpy())

        print(f"Epoch {epoch+1}/{epochs} => Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.4f}")

        # --- Save Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
            torch.save(model.state_dict(), save_model_path)
            print(f"  -> Saved best model to {save_model_path} (Val Loss: {best_val_loss:.4f})")

    print("--- Training Complete ---")
    print(f"Best validation loss achieved: {best_val_loss:.4f}")
    print(f"Best model saved to: {save_model_path}")
