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

# Import necessary functions from preprocess
from src.data_processing.preprocess import (
    clean_and_merge_data,
    save_dataframe_to_parquet,
    fit_and_save_preprocessors # <-- Import the new function
)
# Import the training function
from src.training.train_model import run_training # Ensure correct path

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
        df_merged = clean_and_merge_data(impressions_path=impressions_file, conversions_path=conversions_file)
        save_dataframe_to_parquet(df_merged, output_file)
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
# Add split parameters here as well to ensure consistent splitting
@click.option("--test-split-ratio", default=0.15, type=click.FloatRange(0.0, 0.9), help="Proportion for test set.")
@click.option("--val-split-ratio", default=0.15, type=click.FloatRange(0.0, 0.9), help="Proportion of non-test for validation.")
@click.option("--target-column", default="conversion_flag", help="Target column for stratification.")
@click.option("--random-state", default=42, type=int, help="Random seed for splits.")
# Add other relevant training hyperparameters
@click.option("--learning-rate", default=1e-3, type=float, help="Optimizer learning rate.")
@click.option("--embedding-dim", default=32, type=int, help="Dimension for categorical embeddings.")
# Note: hidden_dims/dropout use defaults in run_training, add CLI options if needed
def train(
    epochs, batch_size, cleaned_data_file, preprocessor_dir, save_model_path,
    test_split_ratio, val_split_ratio, target_column, random_state, learning_rate, embedding_dim
):
    """
    Load data, split it, load preprocessors, and train the model.
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

        # Split test set off first
        if test_split_ratio > 0:
            can_stratify_full = len(np.unique(targets_np)) > 1
            stratify_full = targets_np if can_stratify_full else None
            if not can_stratify_full:
                 click.echo("Warning: Cannot stratify initial test split (only one class in full dataset).")

            train_val_df, test_df = train_test_split( # Need test_df later for evaluation
                full_df,
                test_size=test_split_ratio,
                random_state=random_state,
                stratify=stratify_full
            )
            targets_train_val_np = train_val_df[target_column].to_numpy() if not train_val_df.empty else np.array([])
        else:
            train_val_df = full_df
            test_df = pd.DataFrame() # Empty DataFrame if no test split
            targets_train_val_np = targets_np

        # Split validation set off from the rest
        if val_split_ratio > 0 and not train_val_df.empty:
            current_train_val_ratio = 1.0 - test_split_ratio
            relative_val_ratio = val_split_ratio / current_train_val_ratio if current_train_val_ratio > 0 else val_split_ratio

            if relative_val_ratio >= 1.0:
                 train_df = train_val_df
                 val_df = pd.DataFrame() # No validation set if ratio too high
                 click.echo("Warning: Validation split resulted in empty training set or empty validation set.")
            else:
                can_stratify_val = len(np.unique(targets_train_val_np)) > 1
                stratify_val = targets_train_val_np if can_stratify_val else None
                if not can_stratify_val:
                    click.echo("Warning: Cannot stratify validation split (only one class in train_val data). Performing non-stratified split.")

                try:
                    train_df, val_df = train_test_split( # Need both train_df and val_df
                        train_val_df,
                        test_size=relative_val_ratio,
                        random_state=random_state,
                        stratify=stratify_val
                    )
                except ValueError as e:
                    click.echo(f"Warning: Stratified validation split failed ({e}). Performing non-stratified split.")
                    train_df, val_df = train_test_split(
                        train_val_df,
                        test_size=relative_val_ratio,
                        random_state=random_state
                    )
        else:
            train_df = train_val_df
            val_df = pd.DataFrame() # No validation set

        # --- Sanity Checks for Splits ---
        if train_df.empty:
            raise ValueError("Training data split resulted in an empty DataFrame. Check split ratios.")
        if val_df.empty:
             click.echo("Warning: Validation data split is empty. Proceeding without validation loop in run_training.")
             # Note: run_training needs logic to handle this gracefully.
             # For now, we'll proceed but it might fail or skip validation inside run_training.


        click.echo(f"Data splits created: Train={train_df.shape}, Val={val_df.shape}, Test={test_df.shape}")

        # --- Ensure output directory for model exists ---
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

        # --- Call the refactored training function ---
        # Pass only non-empty val_df to avoid issues inside run_training if it expects data
        if not val_df.empty:
            run_training(
                train_df=train_df,
                val_df=val_df,
                preprocessor_dir=preprocessor_dir,
                save_model_path=save_model_path,
                target_column=target_column,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                embedding_dim=embedding_dim,
                random_state=random_state
                # hidden_dims/dropout use defaults from run_training
            )
            # run_training prints its own completion message
        else:
             click.echo("Skipping training run because validation DataFrame is empty.")
             # TODO: Optionally implement training without validation in run_training
             # Or just accept that training requires a validation set with this setup.

        # TODO: Add step to evaluate on the test_df using the saved best model

    except FileNotFoundError:
         click.echo(f"ERROR: Cleaned data file not found at {cleaned_data_file}", err=True)
    except ValueError as ve:
         click.echo(f"ERROR: {ve}", err=True)
    except Exception as e:
        click.echo(f"ERROR during training preparation or execution: {e}", err=True)


@cli.command()
@click.option(
    "--method",
    default="nn",
    type=click.Choice(["nn", "traditional"]),
    help="Allocation method to use",
)
def allocate(method):
    """
    Run the allocation mechanism using a specified method.
    Methods:
      - nn: Use the trained neural network's predictions
      - traditional: Use a baseline (e.g., second-price auction)
    """
    click.echo(f"Allocating impressions using {method} allocation method...")
    # TODO: call allocation mechanism here
    click.echo(f"(Placeholder) Allocation process complete using {method}.")


@cli.command()
def evaluate():
    """
    Evaluate the system performance on test data.

    This may include metrics like:
    - Total conversions
    - Revenue
    - Fairness (distribution across advertisers)
    - Social welfare
    """
    click.echo("Evaluating system performance...")
    # TODO: call evaluation logic here
    click.echo("(Placeholder) Evaluation complete.")


@cli.command()
@click.pass_context
def help(ctx):
    """
    Show help message and list commands.
    """
    click.echo(ctx.parent.get_help())


if __name__ == "__main__":
    cli()
