import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    fbeta_score,
)
from sklearn.preprocessing import scale

from config import EVALUATION_OUTPUT_DIR
from .utils import remove_anomaly_labels


def _get_anomaly_label_for(column_name):
    return column_name.replace("value_", "anomaly_")


def _color_map(value):
    return {
        0: (0, 0, 1),
        1: (0, 1, 0),
        2: (0.5, 0, 0.5),
        3: (1, 140 / 255, 0),  # orange - #FF8C00
    }.get(value)


def visualise_predicted_anomalies(
    data,
    outputs,
    model_name,
    dataset_name,
    extra_label="",
    y_label="Value",
    x_label="Timestep",
    output_dir=EVALUATION_OUTPUT_DIR,
):
    pure_data = remove_anomaly_labels(data)

    for col_idx, column in enumerate(pure_data.columns):
        y_pred = outputs[:, col_idx]

        plt.clf()
        fig = plt.figure(figsize=(15, 3), dpi=300)
        ax = fig.add_subplot(111)

        col_true_anomalies = data[_get_anomaly_label_for(column)]
        # True Positives - anomalies that were detected
        true_positives = np.logical_and(col_true_anomalies == 1, y_pred == 1)
        # False Positives - non-anomalies that were flagged
        false_positives = np.logical_and(col_true_anomalies == 0, y_pred == 1)
        # False Negatives - anomalies that weren't detected
        false_negatives = np.logical_and(col_true_anomalies == 1, y_pred == 0)

        ax.title.set_text(
            f"{column} - {model_name} {'- ' if extra_label else ''}{extra_label}"
        )

        point_colors = col_true_anomalies.copy()
        point_colors = np.where(point_colors, 1, 0)
        point_colors = np.where(false_positives, 2, point_colors)
        point_colors = np.where(false_negatives, 3, point_colors)
        point_colors = list(map(_color_map, point_colors))

        ax.scatter(pure_data.index, pure_data[column], c=point_colors, s=1, marker=",")

        ax.fill_between(
            data.index,
            0,
            1,
            where=false_positives,
            color="purple",
            alpha=0.2,
            transform=ax.get_xaxis_transform(),
        )
        ax.fill_between(
            data.index,
            0,
            1,
            where=false_negatives,
            color="#FF8C00",
            alpha=0.9,
            transform=ax.get_xaxis_transform(),
        )

        ax.set_xticks([])
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)

        fig.tight_layout()
        plt.draw()

        out_path = (
            f"{output_dir}/{dataset_name}/{model_name}/{column}-{extra_label}.png"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        plt.close(fig)


def fbeta_from_precision_recall(precision, recall, beta):
    return (((beta ** 2) + 1) * precision * recall) / ((beta ** 2 * precision) + recall)


def plot_precision_recall_curve(
    y_true,
    anomaly_scores,
    chosen_threshold,
    model_name,
    dataset_name,
    beta,
    output_dir=EVALUATION_OUTPUT_DIR,
    identifier="_",
):
    # remove NaNs
    indices = np.argwhere(~np.isnan(anomaly_scores))
    y_true = y_true[indices]
    anomaly_scores = anomaly_scores[indices]

    prec, rec, thresholds = precision_recall_curve(y_true, anomaly_scores)

    fbeta_scores = fbeta_from_precision_recall(prec, rec, beta)
    true_thresholds = np.append(thresholds, [1], axis=0)

    # def _calculate_n_anomalies(threshold):
    #     return np.count_nonzero(np.where(anomaly_scores >= threshold, 1, 0))
    # n_anomalies_identified = np.vectorize(_calculate_n_anomalies)(true_thresholds)
    # anomaly_proportion = n_anomalies_identified / np.count_nonzero(y_true)

    # values at the chosen point
    outputs = np.where(anomaly_scores >= chosen_threshold, 1, 0)
    chosen_prec = precision_score(y_true, outputs)
    chosen_rec = recall_score(y_true, outputs)

    plt.clf()
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), dpi=300)
    fig.suptitle(f"Precision/Recall curve for {model_name} on {dataset_name}")

    # F1-score
    ax = axes[0]
    ax.plot(rec, prec, c="black", label="_nolegend_", zorder=1, linestyle="--")
    sc = ax.scatter(
        rec,
        prec,
        c=fbeta_scores,
        # s=anomaly_proportion*200,
        cmap="viridis",
        vmin=0,
        vmax=1,
        label=r"$F_\beta$ at threshold",
        zorder=2,
    )
    ax.plot(
        [chosen_rec],
        [chosen_prec],
        "xm",
        label="Chosen threshold",
        markersize=15,
        zorder=3,
    )
    ax.set_ylabel("Precision")
    ax.set_xlabel("Recall")
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.title.set_text(rf"$F_\beta$ ($\beta = {beta}$)")
    legend = ax.legend()
    legend.set_zorder(4)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(rf"$F_\beta$ score ($\beta = {beta}$)")

    # Threshold
    ax = axes[1]
    ax.plot(rec, prec, c="black", label="_nolegend_", zorder=1, linestyle="--")
    sc = ax.scatter(
        rec,
        prec,
        c=true_thresholds,
        # s=anomaly_proportion*200,
        cmap="inferno",
        label=r"Threshold",
        zorder=2,
    )
    ax.plot(
        [chosen_rec],
        [chosen_prec],
        "xm",
        label="Chosen threshold",
        markersize=15,
        zorder=3,
    )
    ax.set_ylabel("Precision")
    ax.set_xlabel("Recall")
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    legend = ax.legend()
    legend.set_zorder(4)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Threshold")
    ax.title.set_text("Threshold")

    plt.draw()

    out_path = (
        f"{output_dir}/{dataset_name}/{model_name}/precision_recall/{identifier}.png"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def plot_prec_rec_curves(
    actual_anomalies_train_set,
    anomaly_scores_train_set,
    anomaly_thresholds,
    model_label,
    dataset_name,
    beta,
    identifier="",
):
    for column_idx in range(actual_anomalies_train_set.shape[1]):
        plot_precision_recall_curve(
            actual_anomalies_train_set[:, column_idx],
            anomaly_scores_train_set[:, column_idx],
            anomaly_thresholds[column_idx],
            model_label,
            dataset_name,
            beta,
            identifier=identifier + f"_column_{column_idx}",
        )
