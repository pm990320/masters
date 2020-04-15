import numpy as np

from ..base import Model


class RandomPredictor(Model):
    def reset(self):
        pass

    def fit(self, df):
        pass

    def predict_proba(self, df):
        return np.random.random(df.shape)
