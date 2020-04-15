import os

import click
import pandas as pd

from config import *

import tensorflow as tf

from evaluation.evaluate import evaluate as real_evaluate
from evaluation.results import *


def _configure_tensorflow():
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # tf.debugging.set_log_device_placement(True)
    except:
        pass


@click.group()
def main():
    pass


@main.command()
def evaluate():
    _configure_tensorflow()

    if not os.path.exists(EVALUATION_OUTPUT_DIR):
        os.makedirs(EVALUATION_OUTPUT_DIR)

    if os.path.isfile(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    result_df = real_evaluate(RESULTS_FILE)


@main.command()
@click.argument("results_file", type=click.File("r"), default=RESULTS_FILE)
def prepare_results(results_file):
    results_df = pd.read_csv(results_file)
    results_df["dataset"] = results_df["dataset"].map(clean_dataset_names)
    results_df["true_positives"] = results_df["true_positives"].astype(int)
    results_df["false_positives"] = results_df["false_positives"].astype(int)
    results_df["true_negatives"] = results_df["true_negatives"].astype(int)
    results_df["false_negatives"] = results_df["false_negatives"].astype(int)
    generate_model_comparison_by_dataset(results_df, EVALUATION_OUTPUT_DIR)
    generate_dataset_contamination_model_comparison(results_df, EVALUATION_OUTPUT_DIR)


if __name__ == "__main__":
    main()
