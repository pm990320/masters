import sys
import warnings
import traceback

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, fbeta_score
from scipy.optimize import minimize_scalar

from data.nab.real_tweets import nab_multivariate_tweet_volume
from data.nab.relabelled_tweets import nab_tweets_relabelled
from data.yahoo.a1_benchmark import yahoo_a1_benchmark
from models.simple.random import RandomPredictor
from models.simple.last_value import LastValueTrainedPredictor, LastValueWindowPredictor
from models.time_series_analysis.arima import ARIMAPredictor
from models.time_series_analysis.adsas import ADSaSPredictor
from models.lstm.simple_lstm import LSTMPredictor
from models.lstm.lstm_stl import LSTMSTLPredictor

from .decision_threshold import optimise_anomaly_thresholds
from .executors import get_executor
from .utils import *
from .visualisation import visualise_predicted_anomalies, plot_prec_rec_curves


TWEET_DF_WITH_WINDOWS = nab_multivariate_tweet_volume(labels="windows")
TWEET_DF_WITH_POINTS = nab_multivariate_tweet_volume(labels="points")
RELABELED_TWEETS = nab_tweets_relabelled()
YAHOO_DF = yahoo_a1_benchmark()
DATASETS = [
    ("Yahoo A1", upsample(YAHOO_DF, 10)),
    ("NAB Tweets", upsample(RELABELED_TWEETS, "2T30S")),
]
CONTAMINATIONS = [
    0.00,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
]
MODELS = [
    # BASELINE MODEL
    ("LastValue", LastValueWindowPredictor(7)),
    # LSTM MODELS
    ("LSTM + STL", LSTMSTLPredictor(20, num_ephochs=25)),
    ("LSTM", LSTMPredictor(20, num_ephochs=25)),
    # ADSaS MODEL
    ("ADSaS", ADSaSPredictor()),
]
RESULT_COLUMNS = [
    "dataset",
    "contamination",
    "model",
    "precision",
    "recall",
    "fbeta",
    "best_fbeta",
    "best_precision",
    "best_recall",
    "avg_fbeta",
    "avg_precision",
    "avg_recall",
    "n_anomalies_detected",
    "n_anomalies_actual",
    "true_positives",
    "false_positives",
    "true_negatives",
    "false_negatives",
]
BETA = 1


def get_split_for_dataset(dataset_name):
    DATASET_CV_SPLITS = {
        "NAB": 0.5,
        "Yahoo": 0.21,
    }
    prefix, _ = dataset_name.split(" ", 1)
    return DATASET_CV_SPLITS.get(prefix)


def save_results(results, results_file, append=True):
    results = [x for x in results if x is not None]
    result_df = pd.DataFrame(results, columns=RESULT_COLUMNS)
    result_df.set_index(["dataset", "model", "contamination"], inplace=True)
    with open(results_file, "a" if append else "w") as f:
        result_df.to_csv(f, header=f.tell() == 0 or not append)
    return result_df


def evaluate_model(
    results_file,
    model_info,
    training_set,
    cv_set,
    dataset_name,
    contamination_level,
    beta=BETA,
):
    model_label, model = model_info
    print(
        f"Training {model_label} on dataset {dataset_name} with contamination {contamination_level*100:.0f}%"
    )

    try:
        model.reset()
        model.fit(training_set)

        # pick the best anomaly threshold, based on the training set
        anomaly_scores_train_set = model.predict_proba(
            remove_anomaly_labels(training_set)
        )
        if isinstance(anomaly_scores_train_set, pd.DataFrame):
            anomaly_scores_train_set = anomaly_scores_train_set.values

        actual_anomalies_train_set = get_anomaly_cols(training_set).values.astype(int)
        anomaly_thresholds = optimise_anomaly_thresholds(
            actual_anomalies_train_set, anomaly_scores_train_set, beta
        )

        plot_prec_rec_curves(
            actual_anomalies_train_set,
            anomaly_scores_train_set,
            anomaly_thresholds,
            model_label,
            dataset_name,
            beta,
            identifier=f"{contamination_level*100}% contamination-",
        )

        outputs = np.where(anomaly_scores_train_set > anomaly_thresholds, 1, 0)
        outputs = np.nan_to_num(outputs, nan=0)
        visualise_predicted_anomalies(
            training_set,
            outputs,
            model_label,
            dataset_name,
            extra_label=f"training {contamination_level*100}% contamination",
        )

        # use the anomaly threshold to make binary classification on the Cross-Validation set
        anomaly_scores_cv_set = model.predict_proba(remove_anomaly_labels(cv_set))
        if isinstance(anomaly_scores_cv_set, pd.DataFrame):
            anomaly_scores_cv_set = anomaly_scores_cv_set.values

        actual_anomalies_cv_set = get_anomaly_cols(cv_set).values.astype(int)
        outputs = np.where(anomaly_scores_cv_set > anomaly_thresholds, 1, 0)
        outputs = np.nan_to_num(outputs, nan=0)
        visualise_predicted_anomalies(
            cv_set,
            outputs,
            model_label,
            dataset_name,
            extra_label=f"cv {contamination_level*100}% contamination",
        )

        # compute overall result by flattening arrays
        agg_cv_anomalies = actual_anomalies_cv_set.flatten()
        agg_outputs = outputs.flatten()
        precision, recall, fbeta, support = precision_recall_fscore_support(
            agg_cv_anomalies,
            agg_outputs,
            beta=beta,
            pos_label=1,
            average="binary",
            zero_division=0,
        )

        true_values = np.logical_and(agg_outputs == 1, agg_cv_anomalies == 1)
        false_values = np.logical_and(agg_outputs == 0, agg_cv_anomalies == 0)
        true_positives = np.count_nonzero(true_values)
        false_positives = np.count_nonzero(agg_outputs) - true_positives
        true_negatives = np.count_nonzero(false_values)
        false_negatives = (
            agg_outputs.shape[0] - np.count_nonzero(agg_outputs)
        ) - true_negatives

        # compute individual averages
        def _series_metrics(col_idx):
            y_true = actual_anomalies_cv_set[:, col_idx]
            y_pred = outputs[:, col_idx]
            precision, recall, fbeta, support = precision_recall_fscore_support(
                y_true,
                y_pred,
                beta=beta,
                pos_label=1,
                average="binary",
                zero_division=0,
            )
            return np.array([precision, recall, fbeta, support])

        series_metrics = np.stack(
            list(map(_series_metrics, range(outputs.shape[1]))), axis=0
        )

        best_fbeta = np.max(series_metrics[:, 2])
        best_precision = np.max(series_metrics[:, 0])
        best_recall = np.max(series_metrics[:, 1])

        avg_fbeta = np.mean(series_metrics[:, 2])
        avg_precision = np.mean(series_metrics[:, 0])
        avg_recall = np.mean(series_metrics[:, 1])

        result = [
            dataset_name,
            contamination_level,
            model_label,
            precision,
            recall,
            fbeta,
            best_fbeta,
            best_precision,
            best_recall,
            avg_fbeta,
            avg_precision,
            avg_recall,
            np.count_nonzero(agg_outputs),
            np.count_nonzero(agg_cv_anomalies),
            true_positives,
            false_positives,
            true_negatives,
            false_negatives,
        ]

        save_results([result], results_file, append=True)

        return result
    except Exception as error:
        warnings.warn(
            f"Failed to train {model_label} on {dataset_name} with contamination {contamination_level}. Error: {traceback.format_exc()}"
        )
        return None


def evaluate(
    results_file, models=MODELS, contaminations=CONTAMINATIONS, datasets=DATASETS
):
    result_futures = []

    with get_executor() as executor:
        for dataset_name, df in datasets:
            train_df, cv_df = cross_validation_split(
                extract_anomaly_labels_to_anomaly_column(df),
                frac_cv=get_split_for_dataset(dataset_name),
            )

            for contamination_level in contaminations:
                contaminated_train_df = contaminate(
                    train_df, contamination_factor=contamination_level
                )

                for model_label, model in models:
                    future = executor.submit(
                        evaluate_model,
                        results_file,
                        (model_label, model),
                        contaminated_train_df,
                        cv_df,
                        dataset_name,
                        contamination_level,
                    )
                    result_futures.append(future)

    results = list(map(lambda future: future.result(), result_futures))
    result_df = save_results(results, results_file, append=False)

    return result_df
