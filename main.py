#!/usr/bin/env python

"""
main.py - CLI Entrypoint for the AuctioNN project

This script uses the Click library to provide a command-line interface
for data preprocessing, model training, ad allocation, and evaluation.

Usage:
  python main.py [COMMAND] [OPTIONS]

Commands: (example for now)
  preprocess           Clean and merge raw impression/conversion data.
  fit-preprocessors    Fit and save data preprocessors (encoders, scalers).
  train                Train the neural network model.
  allocate             Run the allocation mechanism using the trained model.
  evaluate             Evaluate the performance (conversions, revenue, etc.).
  help                 Show command usage and help.

Example:
  python main.py preprocess --impressions-file data/raw/impressions.parquet \
    --conversions-file data/raw/conversions.parquet \
    --output-file data/processed/clean_data.parquet
  
  python main.py fit-preprocessors --cleaned-data-file data/processed/clean_data.parquet \
    --output-dir ./preprocessors
"""

import click
import pandas as pd
import numpy as np # Import numpy for stratification check
from sklearn.model_selection import train_test_split # Needed for splitting
import os # Needed for checking path writability, creating dirs
import shutil # To potentially remove temp processing dir
import sys # For printing to stderr and exiting
import time # For timing the simulation
from pathlib import Path # For path handling
import torch # For model loading and device management
import joblib # For loading preprocessor info for fallback model load

# Import necessary functions from preprocess
from src.data_processing.preprocess import (
    # clean_and_merge_data,
    # save_dataframe_to_parquet,
    fit_and_save_preprocessors,
    # apply_preprocessors_to_split # <-- Import the new function
)
# Import the training function
# from src.training.train_model import run_training # Ensure correct path
# from src.evaluation.evaluate_model import run_evaluation # <-- Import evaluation function

# Import components for simulation
from src.exchange import ImpressionGenerator, Market, OnlinePreprocessor
from src.campaign import bootstrap_campaigns, Campaign # Import Campaign type
from src.decision_loop import DecisionLoop
from src.results_logger import ResultsLogger
from src.models.network import ImpressionConversionNetwork # Used for fallback model loading

@click.group()
def cli():
    """
    AuctioNN CLI - Use subcommands to preprocess data, train models,
    run allocation, or evaluate results.
    """
    pass # Keep pass here, the group itself doesn't do anything


@cli.command()
@click.option(
    "--impressions-file",
    "-i",
    default="data/raw/impressions.parquet", # Assuming parquet input now
    help="Path to raw impressions data file (parquet).",
    type=click.Path(exists=True, dir_okay=True, readable=True) # Check existence
)
@click.option(
    "--conversions-file",
    "-c",
    default="data/raw/conversions.parquet", # Assuming parquet input now
    help="Path to raw conversions data file (parquet).",
    type=click.Path(exists=True, dir_okay=True, readable=True)
)
@click.option(
    "--output-file",
    "-o",
    default="data/processed/clean_data.parquet", # Outputting parquet
    help="Path to save cleaned data (parquet format).",
    type=click.Path(dir_okay=False, writable=True)
)
def preprocess(impressions_file, conversions_file, output_file):
    """
    Clean raw impression/conversion data and save the merged result.
    """
    click.echo(f"Preprocessing data from {impressions_file} and {conversions_file} ...")
    try:
        # df_merged = clean_and_merge_data(impressions_path=impressions_file, conversions_path=conversions_file)
        # save_dataframe_to_parquet(df_merged, output_file)
        click.echo(f"Processed data saved successfully to {output_file}.")
    except Exception as e:
        click.echo(f"ERROR during preprocessing: {e}", err=True)


@cli.command()
@click.option(
    "--cleaned-data-file",
    "-d",
    required=True, # Make input data mandatory
    help="Path to the cleaned data Parquet file.",
    type=click.Path(exists=True, dir_okay=False, readable=True)
)
@click.option(
    "--output-dir",
    "-o",
    default="./preprocessors",
    help="Directory to save the fitted preprocessors.",
    type=click.Path(file_okay=False, writable=True) # Ensure path is writable
)
@click.option(
    "--test-split-ratio", # Define split ratios here for consistency
    default=0.15,
    type=click.FloatRange(0.0, 0.9), # Test can be 0
    help="Proportion of data to hold out for testing (0.0 to 0.9).",
)
@click.option(
    "--val-split-ratio",
    default=0.15,
    type=click.FloatRange(0.0, 0.9), # Val can be 0
    help="Proportion of non-test data to use for validation (0.0 to 0.9).",
)
@click.option(
    "--target-column",
    default="conversion_flag",
    help="Name of the target column for stratification.",
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    help="Random seed for the data splits.",
)
def fit_preprocessors(cleaned_data_file, output_dir, test_split_ratio, val_split_ratio, target_column, random_state):
    """
    Load data, split it, fit preprocessors ONLY on the training split,
    and save them.
    """
    click.echo(f"Loading data from {cleaned_data_file}...")
    try:
        full_df = pd.read_parquet(cleaned_data_file)

        # --- Perform Split (consistent logic) ---
        click.echo(f"Splitting data (Test={test_split_ratio*100:.0f}%, Val={val_split_ratio*100:.0f}% of remainder)...")
        if target_column not in full_df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")
        targets_np = full_df[target_column].to_numpy()

        # Split test set off first
        if test_split_ratio > 0:
            # Check if stratification is possible
            can_stratify_full = len(np.unique(targets_np)) > 1
            stratify_full = targets_np if can_stratify_full else None
            if not can_stratify_full:
                 click.echo("Warning: Cannot stratify initial test split (only one class in full dataset).")

            train_val_df, _ = train_test_split( # We only need train_val_df here
                full_df,
                test_size=test_split_ratio,
                random_state=random_state,
                stratify=stratify_full
            )
            targets_train_val_np = train_val_df[target_column].to_numpy() if not train_val_df.empty else np.array([])
        else:
            click.echo("Skipping test split (ratio is 0).")
            train_val_df = full_df
            targets_train_val_np = targets_np

        # Split validation set off from the rest
        if val_split_ratio > 0 and not train_val_df.empty:
            current_train_val_ratio = 1.0 - test_split_ratio
            # Avoid division by zero if test split was 100% (though disallowed by FloatRange)
            relative_val_ratio = val_split_ratio / current_train_val_ratio if current_train_val_ratio > 0 else val_split_ratio

            if relative_val_ratio >= 1.0:
                 click.echo("Warning: Validation split ratio results in empty training set. Using all non-test data for training.")
                 train_df = train_val_df
            else:
                # Check if stratification is possible for the validation split
                can_stratify_val = len(np.unique(targets_train_val_np)) > 1
                stratify_val = targets_train_val_np if can_stratify_val else None
                if not can_stratify_val:
                    click.echo("Warning: Cannot stratify validation split (only one class in train_val data). Performing non-stratified split.")

                try:
                    train_df, _ = train_test_split( # We only need train_df here
                        train_val_df,
                        test_size=relative_val_ratio,
                        random_state=random_state,
                        stratify=stratify_val
                    )
                except ValueError as e: # Catch sklearn stratification errors
                    click.echo(f"Warning: Stratified validation split failed ({e}). Performing non-stratified split.")
                    train_df, _ = train_test_split(
                        train_val_df,
                        test_size=relative_val_ratio,
                        random_state=random_state
                        # No stratification on fallback
                    )
        else:
            click.echo("Skipping validation split (ratio is 0 or no data left). Using all non-test data for training.")
            train_df = train_val_df

        if train_df.empty:
            raise ValueError("Training data split resulted in an empty DataFrame. Check split ratios.")

        click.echo(f"Fitting preprocessors using training split (shape: {train_df.shape})...")
        # --- Call the refactored function ---
        fit_and_save_preprocessors(
            train_df=train_df, # Pass the split DataFrame
            output_dir=output_dir
            # Feature lists will use defaults within the function
        )
        # fit_and_save_preprocessors prints its own success message

    except FileNotFoundError:
         click.echo(f"ERROR: Cleaned data file not found at {cleaned_data_file}", err=True)
    except ValueError as ve:
         click.echo(f"ERROR: {ve}", err=True)
    except Exception as e:
        click.echo(f"ERROR during preprocessor fitting: {e}", err=True)
        import traceback
        traceback.print_exc() # Print traceback for detailed debugging


@cli.command()
@click.option("--epochs", default=10, type=int, help="Number of training epochs.")
@click.option("--batch-size", default=1024, type=int, help="Training batch size.")
@click.option(
    "--cleaned-data-file", # Renamed from data-path for clarity
    "-d",
    required=True,
    help="Path to the cleaned data Parquet file.",
    type=click.Path(exists=True, dir_okay=False, readable=True)
)
@click.option(
    "--preprocessor-dir",
    "-p",
    default="./preprocessors",
    help="Directory containing fitted preprocessors.",
    type=click.Path(exists=True, file_okay=False, readable=True)
)
@click.option(
    "--save-model-path",
    "-s",
    default="./models/best_auction_network.pth",
    help="Path to save the best trained model.",
    type=click.Path(dir_okay=False, writable=True) # Ensure path is writable
)
@click.option(
    "--processed-data-dir",
    "-o",
    default="./data/processed_splits", # Default location for processed .npy files
    help="Directory to save/load pre-processed NumPy arrays for splits.",
    type=click.Path(file_okay=False, writable=True)
)
@click.option(
    "--force-reprocess",
    is_flag=True,
    default=False,
    help="Force reprocessing of data splits even if .npy files exist."
)
# Split parameters remain
@click.option("--test-split-ratio", default=0.15, type=click.FloatRange(0.0, 0.9), help="Proportion for test set.")
@click.option("--val-split-ratio", default=0.15, type=click.FloatRange(0.0, 0.9), help="Proportion of non-test for validation.")
@click.option("--target-column", default="conversion_flag", help="Target column for stratification.")
@click.option("--random-state", default=42, type=int, help="Random seed for splits.")
# Other training hyperparameters
@click.option("--learning-rate", default=1e-3, type=float, help="Optimizer learning rate.")
@click.option("--embedding-dim", default=32, type=int, help="Dimension for categorical embeddings.")
def train(
    epochs, batch_size, cleaned_data_file, preprocessor_dir, save_model_path,
    test_split_ratio, val_split_ratio, target_column, random_state, learning_rate, embedding_dim,
    processed_data_dir, force_reprocess # Add new args
):
    """
    Load data, split it, pre-process splits (if needed), load preprocessors,
    and train the model using processed NumPy arrays.
    """
    click.echo("--- Preparing for Training ---")
    click.echo(f"Loading data from {cleaned_data_file}...")
    try:
        full_df = pd.read_parquet(cleaned_data_file)

        # --- Perform Split (Identical logic to fit-preprocessors) ---
        click.echo(f"Splitting data (Test={test_split_ratio*100:.0f}%, Val={val_split_ratio*100:.0f}% of remainder)...")
        if target_column not in full_df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")
        targets_np = full_df[target_column].to_numpy()
        test_df = pd.DataFrame()
        val_df = pd.DataFrame()
        train_df = pd.DataFrame()

        if test_split_ratio > 0:
            can_stratify_full = len(np.unique(targets_np)) > 1
            stratify_full = targets_np if can_stratify_full else None
            train_val_df, test_df = train_test_split(
                full_df, test_size=test_split_ratio, random_state=random_state, stratify=stratify_full
            )
            targets_train_val_np = train_val_df[target_column].to_numpy() if not train_val_df.empty else np.array([])
        else:
            train_val_df = full_df
            targets_train_val_np = targets_np

        if val_split_ratio > 0 and not train_val_df.empty:
            current_train_val_ratio = 1.0 - test_split_ratio
            relative_val_ratio = val_split_ratio / current_train_val_ratio if current_train_val_ratio > 0 else val_split_ratio
            if relative_val_ratio >= 1.0:
                 train_df = train_val_df
            else:
                can_stratify_val = len(np.unique(targets_train_val_np)) > 1
                stratify_val = targets_train_val_np if can_stratify_val else None
                try:
                    train_df, val_df = train_test_split(
                        train_val_df, test_size=relative_val_ratio, random_state=random_state, stratify=stratify_val
                    )
                except ValueError: # Fallback if stratification fails
                     train_df, val_df = train_test_split(
                         train_val_df, test_size=relative_val_ratio, random_state=random_state
                     )
        else:
            train_df = train_val_df

        click.echo(f"Data splits created: Train={train_df.shape}, Val={val_df.shape}, Test={test_df.shape}")

        # --- Apply Preprocessing to Splits ---
        os.makedirs(processed_data_dir, exist_ok=True)
        # Check if files already exist (using one file as a proxy)
        train_tgt_exists = os.path.exists(os.path.join(processed_data_dir, "train_target_data.npy"))
        val_tgt_exists = os.path.exists(os.path.join(processed_data_dir, "val_target_data.npy"))
        test_tgt_exists = os.path.exists(os.path.join(processed_data_dir, "test_target_data.npy"))

        # Determine if reprocessing is needed for any split
        needs_reprocessing = force_reprocess or not train_tgt_exists or \
                             (not val_df.empty and not val_tgt_exists) or \
                             (not test_df.empty and not test_tgt_exists)

        if needs_reprocessing:
            click.echo(f"Preprocessing splits and saving to {processed_data_dir}...")
            if force_reprocess: click.echo("(Force reprocess enabled)")
            # apply_preprocessors_to_split(train_df, preprocessor_dir, processed_data_dir, 'train', target_column)
            # apply_preprocessors_to_split(val_df, preprocessor_dir, processed_data_dir, 'val', target_column)
            # apply_preprocessors_to_split(test_df, preprocessor_dir, processed_data_dir, 'test', target_column)
        else:
            click.echo(f"Found existing processed splits in {processed_data_dir}, skipping reprocessing.")


        # --- Ensure output directory for model exists ---
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

        # --- Call the refactored training function ---
        # run_training(
        #     processed_data_dir=processed_data_dir, # Pass directory of .npy files
        #     preprocessor_dir=preprocessor_dir,     # Pass directory for category_sizes etc.
        #     save_model_path=save_model_path,
        #     # target_column removed from run_training args
        #     epochs=epochs,
        #     batch_size=batch_size,
        #     learning_rate=learning_rate,
        #     embedding_dim=embedding_dim,
        #     random_state=random_state
        #     # hidden_dims/dropout use defaults from run_training
        # )
        # run_training prints its own completion messages

        # TODO: Add step to evaluate on the test split using the saved best model
        # test_loader = DataLoader(AuctionDataset(processed_data_dir, 'test'), ...)
        # model.load_state_dict(torch.load(save_model_path))
        # evaluate(model, test_loader, ...)

    except FileNotFoundError as e:
         click.echo(f"ERROR: Input file not found. {e}", err=True)
    except ValueError as ve:
         click.echo(f"ERROR: {ve}", err=True)
    except Exception as e:
        click.echo(f"ERROR during training preparation or execution: {e}", err=True)
        import traceback
        traceback.print_exc() # Print traceback for detailed debugging


@cli.command()
@click.option(
    "--processed-data-dir",
    "-d",
    default="./data/processed_splits",
    help="Directory containing the processed test split (.npy files).",
    type=click.Path(exists=True, file_okay=False, readable=True)
)
@click.option(
    "--preprocessor-dir",
    "-p",
    default="./preprocessors",
    help="Directory containing preprocessor info (for model initialization).",
    type=click.Path(exists=True, file_okay=False, readable=True)
)
@click.option(
    "--model-path",
    "-m",
    default="./models/best_auction_network.pth",
    help="Path to the saved trained model (.pth file).",
    type=click.Path(exists=True, dir_okay=False, readable=True)
)
@click.option(
    "--batch-size",
    default=1024,
    type=int,
    help="Batch size for evaluation."
)
@click.option(
    "--threshold",
    default=0.5,
    type=click.FloatRange(0.0, 1.0),
    help="Probability threshold for calculating secondary metrics (accuracy, etc.)."
)
def evaluate(processed_data_dir, preprocessor_dir, model_path, batch_size, threshold):
    """
    Evaluate the trained model performance on the test data.
    """
    click.echo("--- Starting Model Evaluation ---")
    click.echo(f"Evaluating model: {model_path}")
    click.echo(f"Using test data from: {processed_data_dir}")
    click.echo(f"Using preprocessor info from: {preprocessor_dir}")

    try:
        # Call the evaluation function
        # results = run_evaluation(
        #     processed_data_dir=processed_data_dir,
        #     preprocessor_dir=preprocessor_dir,
        #     model_path=model_path,
        #     batch_size=batch_size,
        #     threshold=threshold
        # )

        # if not results:
            click.echo("Evaluation did not produce results (e.g., test set empty).")
        # run_evaluation already prints the metrics

    except FileNotFoundError as e:
        click.echo(f"ERROR: Required file not found. {e}", err=True)
    except Exception as e:
        click.echo(f"ERROR during evaluation: {e}", err=True)
        import traceback
        traceback.print_exc()



@cli.command()
@click.pass_context
def help(ctx):
    """
    Show help message and list commands.
    """
    click.echo(ctx.parent.get_help())


@cli.command()
@click.option(
    "--data",
    "-d",
    default="data/cleaned/test/", # Adjusted default to align with preprocess output
    help="Parquet dir containing impression features.",
)
@click.option(
    "--model",
    "-m",
    default="models/best_auction_network.pth", # Adjusted default to align with train output
    help="Path to the trained model (TorchScript .pt or state_dict .pth).",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path)
)
@click.option(
    "--preproc",
    "-p",
    default="preprocessors/",
    help="Directory holding the fitted preprocessors.",
    type=click.Path(exists=True, file_okay=False, readable=True, path_type=Path)
)
@click.option(
    "--out",
    "-o",
    default="runs/simulation_results.parquet",
    help="Destination Parquet file for impression-level logs.",
    type=click.Path(dir_okay=False, writable=True, path_type=Path)
)
@click.option(
    "--num-imps",
    type=int,
    default=None,
    help="Number of impressions to simulate (None = all in data file)."
)
@click.option(
    "--num-users",
    type=int,
    default=10_000,
    help="Number of users to simulate (None = all in data file)."
)
@click.option(
    "--beta",
    type=float,
    default=1.0,
    help="Bid-shading factor β for first-price auction (1.0 = no shading)."
)
@click.option(
    "--tau",
    type=int,
    default=3,
    help="Per-user ad-stock cap τ (max impressions per user per campaign)."
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps", "auto"], case_sensitive=False),
    default="auto",
    help="Device for NN inference ('auto' selects best available)."
)
@click.option(
    "--flush-every",
    type=int,
    default=20_000,
    help="Number of log rows to buffer before flushing to Parquet."
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Global RNG seed for reproducibility."
)
def simulate(data, model, preproc, out, num_imps, num_users, beta, tau, device, flush_every, seed):
    """
    Run the offline ad allocation simulation using a trained model.

    Streams impressions from the DATA file, uses the MODEL and PREPROC info
    to make bidding decisions via the DecisionLoop, simulates the market,
    updates campaign states, and logs results to the OUT file.
    """
    click.echo("--- Starting Offline Simulation ---")

    # 0. Setup: Device, Output Dir, Seed
    if device == "auto":
        resolved_device_str = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        resolved_device_str = device
    torch_device = torch.device(resolved_device_str)
    click.echo(f"Using device: {torch_device}")

    # Ensure output directory exists
    os.makedirs(out.parent, exist_ok=True)
    click.echo(f"Setting random seed: {seed}")
    np.random.seed(seed) # Seed numpy for Market and ImpressionGenerator shuffling
    torch.manual_seed(seed)
    if resolved_device_str == "cuda":
        torch.cuda.manual_seed_all(seed)

    logger = None # Initialize logger to None for graceful shutdown
    try:
        # 1. Instantiate building blocks
        click.echo(f"Loading impression data from: {data}")
        gen = ImpressionGenerator(data, seed=seed, num_users=num_users)

        click.echo(f"Loading online preprocessor from: {preproc}")
        online_preprocessor = OnlinePreprocessor(preproc)

        click.echo("Initializing market simulation...")
        market = Market(seed=seed) # Pass seed here

        click.echo(f"Bootstrapping campaigns using data from: {data}")
        campaigns = bootstrap_campaigns(clean_data_path=data, seed=seed) # Pass seed
        if not campaigns:
            click.echo("ERROR: No campaigns were bootstrapped. Check the data file.", err=True)
            return

        click.echo(f"Setting up results logger to: {out}")
        logger = ResultsLogger(out, flush_every=flush_every)

        click.echo(f"Loading model from: {model}")
        cat_enc = joblib.load("./preprocessors/categorical_encoder.joblib")
        CARDINALITIES = [len(cat) for cat in cat_enc.categories_] # +1 → reserve row for <UNK>


        def load_model(ckpt_path: str, device: torch.device):
            model = ImpressionConversionNetwork(CARDINALITIES, numeric_dim=8, deep_embedding_dim=16)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            model.to(device)
            model.eval()
            return model

        click.echo("Initializing decision loop...")
        loop = DecisionLoop(model=load_model("runs/wad/20250505_234318/epoch_4.pt", torch_device),
                            campaigns=campaigns,
                            preproc=online_preprocessor,
                            market=market,
                            logger=logger,
                            beta=beta,
                            tau=tau,
                            device=torch_device) # Pass torch device object

        # 2. Stream impressions
        click.echo(f"Starting impression stream (limit: {'all' if num_imps is None else f'{num_imps:,}'})...")
        t0 = time.perf_counter()
        processed_count = 0
        for i, imp in enumerate(gen.stream(shuffle=True), start=1):
            if num_imps and i > num_imps:
                break
            loop.process(imp)
            processed_count = i
            if i % (num_imps / 10) == 0:
                 # Calculate current total spent for reporting
                 spent = sum(c_init.budget_remaining - c_loop.budget_remaining
                             for c_init, c_loop in zip(bootstrap_campaigns(data, seed), loop.campaigns.values()))
                 remaining_avg = sum(c.budget_remaining for c in loop.campaigns.values()) / len(campaigns)
                 print(f"\rProcessed {i:,} imps | Spent: ${spent:,.2f} | Avg Rem: ${remaining_avg:,.2f} | Elapsed: {time.perf_counter()-t0:,.1f}s", end="")

        if logger:
            logger.close() # Ensure final flush

        # 3. Final stats
        elapsed = time.perf_counter() - t0
        imps_per_sec = processed_count / elapsed if elapsed > 0 else 0
        final_spent = sum(c_init.budget_remaining - c_loop.budget_remaining
                           for c_init, c_loop in zip(bootstrap_campaigns(data, seed), loop.campaigns.values()))
        final_remaining = sum(c.budget_remaining for c in loop.campaigns.values())

        print(f"""
 
 --- Simulation Complete ---
 Processed {processed_count:,} impressions in {elapsed:,.1f}s ({imps_per_sec:,.0f} imps/sec).
 Total budget spent: ${final_spent:,.2f}
 Total budget remaining: ${final_remaining:,.2f}
 Results logged to: {out}
 """)

    except FileNotFoundError as e:
        click.echo(f"\nERROR: Required file not found. {e}", err=True)
        if logger: logger.close() # Attempt cleanup
    except KeyboardInterrupt:
         print("Simulation interrupted by user. Flushing logs...", file=sys.stderr)
         if logger: logger.close() # Attempt cleanup
         sys.exit(1) # Exit with error code
    except Exception as e:
        click.echo(f"\nERROR during simulation: {e}", err=True)
        import traceback
        traceback.print_exc()
        if logger: logger.close() # Attempt cleanup


if __name__ == "__main__":
    cli()
