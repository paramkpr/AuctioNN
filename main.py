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
import numpy as np  # Import numpy for stratification check
from sklearn.model_selection import train_test_split  # Needed for splitting
import os  # Needed for checking path writability, creating dirs
import sys  # For printing to stderr and exiting
import time  # For timing the simulation
from pathlib import Path  # For path handling
import torch  # For model loading and device management
import joblib  # For loading preprocessor info for fallback model load

# Import components for simulation
from src.exchange import ImpressionGenerator, Market, OnlinePreprocessor
from src.campaign import bootstrap_campaigns  # Import Campaign type
from src.decision_loop import DecisionLoop
from src.results_logger import ResultsLogger
from src.utils import load_model, constant_pconv_heuristic


@click.group()
def cli():
    """
    AuctioNN CLI - Use subcommands to preprocess data, train models,
    run allocation, or evaluate results.
    """
    pass  # Keep pass here, the group itself doesn't do anything


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
    default="data/cleaned/test/",  # Adjusted default to align with preprocess output
    help="Parquet dir containing impression features.",
)
@click.option(
    "--model",
    "-m",
    default="models/best_auction_network.pth",  # Adjusted default to align with train output
    help="Path to the trained model (TorchScript .pt or state_dict .pth).",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--preproc",
    "-p",
    default="preprocessors/",
    help="Directory holding the fitted preprocessors.",
    type=click.Path(exists=True, file_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--out",
    "-o",
    default="runs/simulation_results.parquet",
    help="Destination Parquet file for impression-level logs.",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
@click.option(
    "--num-imps",
    type=int,
    default=None,
    help="Number of impressions to simulate (None = all in data file).",
)
@click.option(
    "--num-users",
    type=int,
    default=10_000,
    help="Number of users to simulate (None = all in data file).",
)
@click.option(
    "--beta",
    type=float,
    default=1.0,
    help="Bid-shading factor β for first-price auction (1.0 = no shading).",
)
@click.option(
    "--tau",
    type=int,
    default=3,
    help="Per-user ad-stock cap τ (max impressions per user per campaign).",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps", "auto"], case_sensitive=False),
    default="auto",
    help="Device for NN inference ('auto' selects best available).",
)
@click.option(
    "--flush-every",
    type=int,
    default=20_000,
    help="Number of log rows to buffer before flushing to Parquet.",
)
@click.option(
    "--seed", type=int, default=42, help="Global RNG seed for reproducibility."
)
def simulate(
    data, model, preproc, out, num_imps, num_users, beta, tau, device, flush_every, seed
):
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
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        resolved_device_str = device
    torch_device = torch.device(resolved_device_str)
    click.echo(f"Using device: {torch_device}")

    # Ensure output directory exists
    os.makedirs(out.parent, exist_ok=True)
    click.echo(f"Setting random seed: {seed}")
    np.random.seed(seed)  # Seed numpy for Market and ImpressionGenerator shuffling
    torch.manual_seed(seed)
    if resolved_device_str == "cuda":
        torch.cuda.manual_seed_all(seed)

    logger = None  # Initialize logger to None for graceful shutdown
    try:
        # 1. Instantiate building blocks
        click.echo(f"Loading impression data from: {data}")
        gen = ImpressionGenerator(data, seed=seed, num_users=num_users)

        click.echo(f"Loading online preprocessor from: {preproc}")
        online_preprocessor = OnlinePreprocessor(preproc)

        click.echo("Initializing market simulation...")
        market = Market(seed=seed)  # Pass seed here

        click.echo(f"Bootstrapping campaigns using data from: {data}")
        campaigns = bootstrap_campaigns(clean_data_path=data, seed=seed)  # Pass seed
        if not campaigns:
            click.echo(
                "ERROR: No campaigns were bootstrapped. Check the data file.", err=True
            )
            return

        click.echo(f"Setting up results logger to: {out}")
        logger = ResultsLogger(out, flush_every=flush_every)

        click.echo(f"Loading model from: {model}")
        cat_enc = joblib.load("./preprocessors/categorical_encoder.joblib")
        CARDINALITIES = [
            len(cat) for cat in cat_enc.categories_
        ]  # +1 → reserve row for <UNK>

        model = load_model(model, torch_device, CARDINALITIES)
        # model = constant_pconv_heuristic

        click.echo("Initializing decision loop...")
        loop = DecisionLoop(
            predictor=model,
            campaigns=campaigns,
            preproc=online_preprocessor,
            market=market,
            logger=logger,
            beta=beta,
            tau=tau,
            device=torch_device,
        )  # Pass torch device object

        # 2. Stream impressions
        click.echo(
            f"Starting impression stream (limit: {'all' if num_imps is None else f'{num_imps:,}'})..."
        )
        t0 = time.perf_counter()
        processed_count = 0
        for i, imp in enumerate(gen.stream(shuffle=True), start=1):
            if num_imps and i > num_imps:
                break

            loop.process(imp)

            processed_count = i
            if i % (num_imps / 10) == 0:
                # Calculate current total spent for reporting
                spent = sum(
                    c_init.budget_remaining - c_loop.budget_remaining
                    for c_init, c_loop in zip(
                        bootstrap_campaigns(data, seed), loop.campaigns.values()
                    )
                )
                remaining_avg = sum(
                    c.budget_remaining for c in loop.campaigns.values()
                ) / len(campaigns)
                print(
                    f"\rProcessed {i:,} imps | Spent: ${spent:,.2f} | Avg Rem: ${remaining_avg:,.2f} | Elapsed: {time.perf_counter() - t0:,.1f}s",
                    end="",
                )

        if logger:
            logger.close()  # Ensure final flush

        # 3. Final stats
        elapsed = time.perf_counter() - t0
        imps_per_sec = processed_count / elapsed if elapsed > 0 else 0
        final_spent = sum(
            c_init.budget_remaining - c_loop.budget_remaining
            for c_init, c_loop in zip(
                bootstrap_campaigns(data, seed), loop.campaigns.values()
            )
        )
        final_remaining = sum(c.budget_remaining for c in loop.campaigns.values())

        print(
            f"""
                --- Simulation Complete ---
                Processed {processed_count:,} impressions in {elapsed:,.1f}s ({imps_per_sec:,.0f} imps/sec).
                Total budget spent: ${final_spent:,.2f}
                Total budget remaining: ${final_remaining:,.2f}
                Results logged to: {out}
            """
        )

    except FileNotFoundError as e:
        click.echo(f"\nERROR: Required file not found. {e}", err=True)
        if logger:
            logger.close()  # Attempt cleanup
    except KeyboardInterrupt:
        print("Simulation interrupted by user. Flushing logs...", file=sys.stderr)
        if logger:
            logger.close()  # Attempt cleanup
        sys.exit(1)  # Exit with error code
    except Exception as e:
        click.echo(f"\nERROR during simulation: {e}", err=True)
        import traceback

        traceback.print_exc()
        if logger:
            logger.close()  # Attempt cleanup


if __name__ == "__main__":
    cli()
