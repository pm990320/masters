from abc import ABCMeta, abstractmethod

import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.seasonal import STL

from .base import Model
from evaluation.utils import *
from evaluation.executors import get_nested_executor


class PredictionModel(Model):
    @abstractmethod
    def predict_values(self, df):
        raise NotImplementedError

    @abstractmethod
    def fit_predictor(self, df):
        """
        :returns: predicted values for training set
        """
        raise NotImplementedError


class SquaredErrorPredictor(PredictionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.column_distributions = []

    def reset(self):
        self.column_distributions = []

    def fit(self, df):
        predictions = self.fit_predictor(df)
        noanomalies_df = remove_anomaly_labels(df)
        squared_errors = (predictions - noanomalies_df) ** 2

        def _fit_normal_column(column):
            column_errors = squared_errors[column]
            mu, sigma = norm.fit(column_errors.dropna())
            return (mu, sigma)

        self.column_distributions = list(
            map(_fit_normal_column, squared_errors.columns)
        )

    def predict_proba(self, df):
        noanomalies_df = remove_anomaly_labels(df)
        predictions = self.predict_values(noanomalies_df)
        squared_errors = (predictions - noanomalies_df) ** 2
        if isinstance(squared_errors, pd.DataFrame):
            squared_errors = squared_errors.values

        def _predict_proba_column(col_idx):
            mu, sigma = self.column_distributions[col_idx]
            return norm.cdf(squared_errors[:, col_idx], loc=mu, scale=sigma)

        with get_nested_executor() as executor:
            results = list(
                executor.map(_predict_proba_column, range(noanomalies_df.shape[1]))
            )

        predicted_probas = np.stack(results, axis=1)
        return predicted_probas


class STLResidualSquaredErrorPredictor(PredictionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.column_distributions = []

    def reset(self):
        self.column_distributions = []

    def fit(self, df):
        predictions = self.fit_predictor(df)
        noanomalies_df = remove_anomaly_labels(df)
        errors = predictions - noanomalies_df

        def _fit_normal_column(column):
            column_errors = errors[column].fillna(0)
            errors_stl = STL(column_errors, period=7, robust=True, trend=13).fit()
            mu, sigma = norm.fit(errors_stl.resid ** 2)
            return (mu, sigma)

        self.column_distributions = list(map(_fit_normal_column, errors.columns))

    def predict_proba(self, df):
        noanomalies_df = remove_anomaly_labels(df)
        predictions = self.predict_values(noanomalies_df)
        errors = predictions - noanomalies_df
        if isinstance(errors, pd.DataFrame):
            errors = errors.values

        def _predict_proba_column(col_idx):
            column_errors = np.nan_to_num(errors[:, col_idx], nan=0)
            errors_stl = STL(column_errors, period=7, robust=True, trend=13).fit()
            residuals = errors_stl.resid ** 2
            mu, sigma = self.column_distributions[col_idx]
            return norm.cdf(residuals, loc=mu, scale=sigma)

        with get_nested_executor() as executor:
            results = list(
                executor.map(_predict_proba_column, range(noanomalies_df.shape[1]))
            )

        predicted_probas = np.stack(results, axis=1)
        return predicted_probas
