import warnings
import traceback

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.arima_model import ARIMA

from evaluation.utils import remove_anomaly_labels, roll

from ..base import Model
from .univariate_model import UnivariateModel


class ColumnARIMAPredictor:
    def __init__(self, column_name, order=None):
        self.order = order
        self.column_name = column_name
        self.arima_results = None
        self.err_mean = 0
        self.err_cov = 1
        self.train_max_index = None

    def fit(self, df):
        df = df[self.column_name]
        if not isinstance(df.index, pd.DatetimeIndex):
            self.train_max_index = max(df.index)

        model = ARIMA(df, self.order)

        try:
            self.arima_results = model.fit(disp=False)
        except (ValueError, np.linalg.LinAlgError) as error:
            warnings.warn(
                f"Failed to fit ARIMA model for column {self.column_name}: {traceback.format_exc()}"
            )
            return

        predictions = self.arima_results.predict()
        errors = (df - predictions) ** 2
        self.err_mean, self.err_cov = norm.fit(errors)

    def predict_values(self, df):
        if self.arima_results is None:
            return np.full(df.shape[0], 0.5)  # unsure? make it a coin toss!

        df = df[self.column_name].copy()

        if isinstance(df.index, pd.DatetimeIndex):
            predictions = self.arima_results.predict(
                start=df.index[0], end=df.index[-1]
            )
        else:
            predictions, stderr, conf_int = self.arima_results.forecast(steps=len(df))

        return predictions

    def predict_proba(self, df):
        predictions = self.predict_values(df)
        df = df[self.column_name]

        errors = (df.values - predictions) ** 2
        errors = np.nan_to_num(errors, nan=0)

        prob_of_anomaly = norm.cdf(errors, loc=self.err_mean, scale=self.err_cov)

        return prob_of_anomaly


class ARIMAPredictor(UnivariateModel):
    column_predictor_class = ColumnARIMAPredictor

    def __init__(self, order=(1, 0, 0)):
        super().__init__()
        self.order = order

    def get_column_predictor_kwargs(self):
        return dict(order=self.order)
