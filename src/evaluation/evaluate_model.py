# src/evaluation/evaluate_model.py

import torch
from torch.utils.data import DataLoader
import numpy as np
import joblib
import os
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import pandas as pd # For displaying confusion matrix nicely

# Import necessary components from your project
from data_processing.datasets import ParquetAuctionDataset
from models.network import AuctionNetwork
from training.train_model import count_pos_neg, infer_feature_counts

def run_evaluation(
    processed_data_dir: str,
    preprocessor_dir: str,
    model_path: str,
    batch_size: int = 1024,
    threshold: float = 0.5, # Threshold for secondary metrics
    device: str | None = None # Auto-detect if None
):
    """
    Loads a trained model and evaluates its performance on the test set.

    Args:
        processed_data_dir: Directory containing the processed test .npy files.
        preprocessor_dir: Directory containing preprocessor info (for model init).
        model_path: Path to the saved trained model state dictionary (.pth file).
        batch_size: Batch size for evaluation.
        threshold: Probability threshold for calculating accuracy, precision, etc.
        device: Device to run evaluation on ('cuda', 'cpu', or None for auto-detect).

    Returns:
        dict: A dictionary containing calculated evaluation metrics.
    """
    print("--- Starting Model Evaluation ---")

    # --- Device Setup ---
    if device is None:
        resolved_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        resolved_device = device
    print(f"Using device: {resolved_device}")

    # --- Load Necessary Preprocessor Info ---
    print(f"Loading preprocessor info from: {preprocessor_dir}")
    try:
        category_sizes = joblib.load(os.path.join(preprocessor_dir, 'category_sizes.joblib'))
        _, num_numerical_features = infer_feature_counts(processed_data_dir, "train")

    except FileNotFoundError as e:
        print(f"ERROR: Failed to load preprocessor files needed for model init. Details: {e}")
        raise
    print("Preprocessor info loaded.")

    # --- Initialize Model ---
    print("Initializing model architecture...")
    model = AuctionNetwork(
        category_sizes=category_sizes,
        num_numerical_features=num_numerical_features
        # Use default hyperparams or load them if saved separately
    )

    # --- Load Model Weights ---
    print(f"Loading model weights from: {model_path}")
    try:
        model.load_state_dict(torch.load(model_path, map_location=resolved_device))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}")
        raise
    except Exception as e:
        print(f"ERROR: Failed to load model weights. Details: {e}")
        raise
    model.to(resolved_device)
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully.")

    # --- Load Test Dataset & DataLoader ---
    print(f"Loading test data from: {processed_data_dir}")
    try:
        test_dataset = ParquetAuctionDataset(processed_data_dir=processed_data_dir + "/test", batch_rows=batch_size*4)
    except FileNotFoundError:
         print(f"ERROR: Failed to find test .npy files in {processed_data_dir}.")
         raise
    except ValueError as ve:
         print(f"ERROR: {ve}") # e.g., shape mismatch
         raise

    if len(test_dataset) == 0:
        print("Warning: Test dataset is empty. Skipping evaluation.")
        return {}

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=min(4, os.cpu_count()))
    print(f"Test dataset loaded: {len(test_dataset)} samples.")

    # --- Run Predictions ---
    print("Running predictions on test set...")
    all_targets = []
    all_probs = []
    eval_loop = tqdm(test_loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for cat_batch, num_batch, target_batch in eval_loop:
            cat_batch = cat_batch.to(resolved_device)
            num_batch = num_batch.to(resolved_device)
            # target_batch stays on CPU for easier aggregation

            outputs = model(cat_batch, num_batch)
            probs = torch.sigmoid(outputs).squeeze() # Get probabilities

            all_targets.append(target_batch.numpy())
            all_probs.append(probs.cpu().numpy())

    # Concatenate results from all batches
    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)
    # Ensure y_prob is 1D if it ended up with an extra dim
    if y_prob.ndim > 1 and y_prob.shape[1] == 1:
        y_prob = y_prob.flatten()

    # --- Calculate Metrics ---
    print("Calculating metrics...")
    metrics = {}
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        metrics['average_precision'] = average_precision_score(y_true, y_prob)
        metrics['log_loss'] = log_loss(y_true, y_prob)

        # Secondary metrics based on threshold
        y_pred_binary = (y_prob >= threshold).astype(int)
        metrics['threshold'] = threshold
        metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
        metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred_binary, zero_division=0)

        cm = confusion_matrix(y_true, y_pred_binary)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0) # Handle edge case of single class predictions
        metrics['confusion_matrix'] = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

        report = classification_report(y_true, y_pred_binary, zero_division=0, output_dict=True)
        metrics['classification_report'] = report # Store full report dict

    except Exception as e:
        print(f"ERROR during metric calculation: {e}")
        # Decide how to handle - return partial metrics or raise?

    # --- Print Results ---
    print("\n--- Evaluation Results ---")
    print(f" ROC AUC:           {metrics.get('roc_auc', 'N/A'):.4f}")
    print(f" Average Precision: {metrics.get('average_precision', 'N/A'):.4f}")
    print(f" Log Loss:          {metrics.get('log_loss', 'N/A'):.4f}")
    print("-" * 26)
    print(f" Threshold:         {metrics.get('threshold', 'N/A')}")
    print(f" Accuracy:          {metrics.get('accuracy', 'N/A'):.4f}")
    print(f" Precision:         {metrics.get('precision', 'N/A'):.4f}")
    print(f" Recall:            {metrics.get('recall', 'N/A'):.4f}")
    print(f" F1 Score:          {metrics.get('f1_score', 'N/A'):.4f}")
    print("-" * 26)
    cm = metrics.get('confusion_matrix', {})
    print(" Confusion Matrix:")
    if cm:
         # Simple text print
         print(f"   TN: {cm.get('tn', 0):<6}  FP: {cm.get('fp', 0):<6}")
         print(f"   FN: {cm.get('fn', 0):<6}  TP: {cm.get('tp', 0):<6}")
         # Optional: Pretty print with pandas
         # cm_df = pd.DataFrame([[cm.get('tn', 0), cm.get('fp', 0)], [cm.get('fn', 0), cm.get('tp', 0)]],
         #                      index=['Actual Neg', 'Actual Pos'], columns=['Pred Neg', 'Pred Pos'])
         # print(cm_df)
    else:
        print("  (Not available)")
    print("--------------------------")

    return metrics

# Example usage (if run directly, though typically called from main.py)
if __name__ == '__main__':
    results = run_evaluation(
        processed_data_dir='./data/processed',
        preprocessor_dir='./preprocessors',
        model_path='./models/best_02052025_01.pth'
    )
    print("\nEvaluation metrics dictionary:")
    import json
    print(json.dumps(results, indent=2)) 