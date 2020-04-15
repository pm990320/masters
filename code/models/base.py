from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):
    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def fit(self, df):
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, df):
        raise NotImplementedError
