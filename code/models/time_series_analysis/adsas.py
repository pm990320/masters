import warnings
import traceback

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.signal import periodogram
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL

from evaluation.utils import remove_anomaly_labels, roll

from ..base import Model
from .univariate_model import UnivariateModel

DICKEY_FULLER_PVALUE_THRESHOLD = 0.0005


class ColumnADSaSPredictor:
    def __init__(self, column_name):
        self.column_name = column_name
        self.sarimax_results = None
        self.residual_mean = 0
        self.residual_cov = [1]
        self.data_frequency = None
        self.train_max_index = None

    def fit(self, df, assumed_day_rows=24):
        df = df[self.column_name]
        if not isinstance(df.index, pd.DatetimeIndex):
            self.train_max_index = max(df.index)

        # 1. Data analysis module
        adfstat, pvalue, usedlag, nobs, critvalues, icbest = adfuller(df.values)
        if pvalue > DICKEY_FULLER_PVALUE_THRESHOLD:
            # data is nonstationary
            is_stationary = False
            if isinstance(df.index, pd.DatetimeIndex):
                period_df = df.copy()
                period_df.index = period_df.index.to_period()
                num_rows_per_day = int(period_df.resample("D").count().max())
                self.data_frequency = num_rows_per_day
            else:
                self.data_frequency = assumed_day_rows
        else:
            # data is stationary
            is_stationary = True
            frequencies, Pxx_spec = periodogram(df.values, scaling="spectrum")
            freq = frequencies[np.argmax(Pxx_spec)]
            self.data_frequency = max(2, int(np.rint(freq)))

        # 2. Forecasing module
        model = SARIMAX(
            df,
            # p, d, q
            order=(1, 1, 0),
            # P, D, Q, s
            seasonal_order=(0, 1, 1, self.data_frequency),
        )

        try:
            self.sarimax_results = model.fit(disp=False)
        except (ValueError, np.linalg.LinAlgError) as error:
            warnings.warn(
                f"Failed to fit SARIMAX model for column {self.column_name}: {traceback.format_exc()}"
            )
            return

        predictions = self.sarimax_results.predict()

        # 3. Error processing module
        errors = df.values - predictions
        errors_stl = STL(errors, period=7, robust=True, trend=13).fit()
        residuals = errors_stl.resid ** 2

        self.residual_mean, self.residual_cov = norm.fit(residuals)

    def predict_values(self, df):
        if self.sarimax_results is None:
            return np.full(df.shape[0], 0.5)  # unsure? make it a coin toss!

        df = df[self.column_name]

        if isinstance(df.index, pd.DatetimeIndex):
            predictions = self.sarimax_results.predict(
                start=df.index[0], end=df.index[-1]
            )
        else:
            predictions = self.sarimax_results.forecast(steps=len(df))

        return predictions

    def predict_proba(self, df):
        predictions = self.predict_values(df)
        df = df[self.column_name]

        errors = df.values - predictions
        errors = np.nan_to_num(errors, nan=0)
        errors_stl = STL(errors, period=7, robust=True, trend=13).fit()
        residuals = errors_stl.resid ** 2

        prob_of_anomaly = norm.cdf(
            residuals, loc=self.residual_mean, scale=self.residual_cov
        )

        return prob_of_anomaly


class ADSaSPredictor(UnivariateModel):
    column_predictor_class = ColumnADSaSPredictor
