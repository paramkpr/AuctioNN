#!/usr/bin/env python

"""
main.py - CLI Entrypoint for the AuctioNN project

This script uses the Click library to provide a command-line interface
for data preprocessing, model training, ad allocation, and evaluation.

Usage:
  python main.py [COMMAND] [OPTIONS]

Commands: (example for now)
  preprocess   Clean and preprocess the raw dataset.
  train        Train the neural network model.
  allocate     Run the allocation mechanism using the trained model.
  evaluate     Evaluate the performance (conversions, revenue, etc.).
  help         Show command usage and help.

Example:
  python main.py preprocess --input-file data/raw/impressions.csv
    --output-file data/processed/impressions_clean.csv
"""

import click

from src.data_processing.preprocess import clean_and_merge_data, save_dataframe_to_parquet

@click.group()
def cli():
    """
    AuctioNN CLI - Use subcommands to preprocess data, train models,
    run allocation, or evaluate results.
    """


@cli.command()
@click.option(
    "--impressions-file",
    "-i",
    default="data/raw/impressions.csv",
    help="Path to raw data file",
)
@click.option(
    "--conversions-file",
    "-c",
    default="data/raw/conversions.csv",
    help="Path to conversions data file",
)
@click.option(
    "--output-file",
    "-o",
    default="data/processed/impressions_clean.csv",
    help="Path to save cleaned data",
)
def preprocess(impressions_file, conversions_file, output_file):
    """
    Clean and preprocess the raw dataset for model training.
    """
    click.echo(f"Preprocessing data from {impressions_file} and {conversions_file} ...")
    df_merged = clean_and_merge_data(impressions_path=impressions_file, conversions_path=conversions_file)
    save_dataframe_to_parquet(df_merged, output_file)       
    click.echo(f"Processed data saved to {output_file}.")


@cli.command()
@click.option("--epochs", default=5, help="Number of training epochs")
@click.option("--batch-size", default=32, help="Training batch size")
def train(epochs, batch_size):
    """
    Train the neural network model.
    """
    click.echo("Starting model training...")
    # TODO: call training logic here, e.g.:
    # from src.models import neural_net
    # neural_net.train_model(epochs=epochs, batch_size=batch_size)

    click.echo(f"Training completed with {epochs} epochs and batch size {batch_size}.")


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
    # TODO: call allocation mechanism here, e.g.:
    # from src.mechanism import allocation_mechanism
    # allocation_mechanism.run_allocation(method=method)

    click.echo(f"Allocation process complete using {method}.")


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
    # TODO: call evaluation logic here, e.g.:
    # from src.evaluation import evaluation_metrics
    # evaluation_metrics.run_evaluation()

    click.echo("Evaluation complete. Results are saved or printed to console.")


@cli.command()
@click.pass_context
def help(ctx):
    """
    Show help message and list commands.
    """
    click.echo(ctx.parent.get_help())


if __name__ == "__main__":
    cli()
