from scipy.stats import multivariate_normal
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import numpy as np

from evaluation.utils import remove_anomaly_labels
from .lookback_model import LookbackModel
from ..prediction import PredictionModel


class LSTMBase(LookbackModel, PredictionModel):
    def __init__(self, look_back, look_forward=1, num_ephochs=200):
        super(LSTMBase, self).__init__(look_back, look_forward=look_forward)
        self.num_ephochs = num_ephochs
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.keras_model = None

    def reset(self):
        super().reset()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.keras_model = None

    def _fit_model(self, df):
        noanomalies_df = remove_anomaly_labels(df)
        noanomalies_scaled = self.scaler.fit_transform(noanomalies_df)
        x, y = self._create_lookback_dataframe(noanomalies_scaled)

        keras_model = Sequential()

        keras_model.add(
            LSTM(
                x.shape[2],
                input_shape=(self.look_back, x.shape[2]),
                return_sequences=True,
            )
        )
        middle_units = int(np.floor(x.shape[2] / 2))
        keras_model.add(LSTM(middle_units))
        keras_model.add(Dense(y.shape[-1], activation="relu"))

        keras_model.compile(loss="mean_squared_error", optimizer="adam")
        print("Model compiled")
        keras_model.fit(
            x,
            y,
            epochs=self.num_ephochs,
            batch_size=10,
            verbose=2,
            use_multiprocessing=True,
        )
        print("Trained")

        return x, y, keras_model

    def fit_predictor(self, df):
        x, y, keras_model = self._fit_model(df)
        self.keras_model = keras_model

        pred = keras_model.predict(x)
        return self._pad_answer(remove_anomaly_labels(df), pred)

    def _predict_lstm_values(self, df):
        noanomalies_df = remove_anomaly_labels(df)
        scaled_df = self.scaler.transform(noanomalies_df)
        x, y = self._create_lookback_dataframe(scaled_df)
        return self.keras_model.predict(x), y

    def predict_values(self, df):
        predictions, y = self._predict_lstm_values(df)
        return self._pad_answer(df, predictions, value=np.nan)
