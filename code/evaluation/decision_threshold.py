import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from scipy.optimize import minimize_scalar
from .executors import get_nested_executor


def optmise_anomaly_threshold(actual_anomalies, anomaly_scores, beta):
    actual_anomalies = actual_anomalies.astype(int)

    # pick the threshold that results in the optimal $F_\beta$ score
    def fbeta_cost_function(threshold):
        outputs = np.where(anomaly_scores > threshold, 1, 0)
        outputs = np.nan_to_num(outputs, nan=0)

        # we want to maximise the Fbeta score, so minimise 1-Fbeta
        fbeta_component = 1 - fbeta_score(
            actual_anomalies, outputs, beta, pos_label=1, labels=[0, 1], zero_division=0
        )

        # # heavily penalise classifying more anomalies than there are anomalies in the dataset
        eager_penalty = (
            max(0, np.count_nonzero(outputs) - np.count_nonzero(actual_anomalies)) ** 2
        )

        # # really heavily penalise scoring EVERYTHING in the dataset as an anomaly - anomalies are < 5% of the dataset
        everything_penalty = (
            max(0, np.count_nonzero(outputs) - (actual_anomalies.shape[0] * 0.05)) ** 2
        )

        return (fbeta_component * 100 + everything_penalty + eager_penalty) * 10e4

    optimised = minimize_scalar(
        fbeta_cost_function,
        bounds=(0, 1),
        method="bounded",
        options=dict(maxiter=10e9),
    )

    return optimised.x


def optimise_anomaly_thresholds(actual_anomalies, anomaly_scores, beta):
    """
    :param actual_anomalies: m x n
    :param anomaly_scores: m x n
    """

    def _optimise_col(col_idx):
        return optmise_anomaly_threshold(
            actual_anomalies[:, col_idx], anomaly_scores[:, col_idx], beta
        )

    with get_nested_executor() as executor:
        thresholds = list(executor.map(_optimise_col, range(anomaly_scores.shape[1])))

    return np.stack(thresholds, axis=0)
