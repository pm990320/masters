from ..prediction import SquaredErrorPredictor
from .base import LSTMBase


class LSTMPredictor(LSTMBase, SquaredErrorPredictor):
    pass
