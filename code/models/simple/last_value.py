import numpy as np
import pandas as pd
from scipy.stats import norm

from ..base import Model
from ..prediction import SquaredErrorPredictor
from evaluation.executors import get_nested_executor
from evaluation.utils import remove_anomaly_labels, roll


class LastValueTrainedPredictor(SquaredErrorPredictor):
    """
    Fits a normal distribution using the errors on training, using the previous value as the prediction.
    """

    def fit_predictor(self, df):
        cleansed_df = remove_anomaly_labels(df)
        predictions = cleansed_df.shift(periods=1)
        return predictions

    def predict_values(self, df):
        cleansed_df = remove_anomaly_labels(df)
        predictions = cleansed_df.shift(periods=1)
        return predictions


class LastValueWindowPredictor(Model):
    """
    Fits a normal distribution over the errors of the last N terms, uses this to compute the PDF
    of the error term for next error, inversing this gives the probability of this point being anomalous
    """

    def __init__(self, n=5):
        self.N = n

    def reset(self):
        pass

    def fit(self, df):
        pass

    def predict_proba(self, df):
        diffs_df = df.diff().fillna(0) ** 2

        def predict_window(window_df):
            def predict_column(column):
                values = window_df[column].values
                mu, sigma = norm.fit(values[:-1])
                return norm.cdf(values[-1], loc=mu, scale=sigma)

            with get_nested_executor() as executor:
                result = list(executor.map(predict_column, window_df.columns))

            return pd.Series(np.stack(result, axis=0))

        res_df = roll(diffs_df, self.N).apply(predict_window).fillna(0)

        for time_step in df.index.difference(res_df.index).values:
            res_df.at[time_step] = 0

        return res_df
