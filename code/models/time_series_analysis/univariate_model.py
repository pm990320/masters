import numpy as np

from evaluation.executors import get_nested_executor
from evaluation.utils import remove_anomaly_labels
from models.prediction import PredictionModel


class UnivariateModel(PredictionModel):
    column_predictor_class = None  # override!

    def __init__(self):
        self.column_models = {}

    def reset(self):
        self.column_models = {}

    def get_column_predictor_kwargs(self):
        return {}

    def fit_predictor(self, df):
        raise NotImplementedError

    def fit(self, df):
        cleansed_df = remove_anomaly_labels(df)

        def fit_column(column):
            column_predictor = self.column_predictor_class(
                column, **self.get_column_predictor_kwargs()
            )
            column_predictor.fit(cleansed_df)
            return (column, column_predictor)

        with get_nested_executor() as executor:
            self.column_models = dict(executor.map(fit_column, cleansed_df.columns))

    def _map_over_columns(self, func):
        with get_nested_executor() as executor:
            prediction_results = list(executor.map(func, self.column_models.items()))

        prediction_results = np.nan_to_num(np.stack(prediction_results, axis=1), nan=0)
        return prediction_results

    def predict_proba(self, df):
        def predict_proba_column(tup):
            column, column_predictor = tup
            return column_predictor.predict_proba(df)

        return self._map_over_columns(predict_proba_column)

    def predict_values(self, df):
        def predict_values_column(tup):
            column, column_predictor = tup
            return column_predictor.predict_values(df)

        return self._map_over_columns(predict_values_column)
