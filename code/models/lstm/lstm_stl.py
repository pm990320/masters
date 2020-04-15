from ..prediction import STLResidualSquaredErrorPredictor
from .base import LSTMBase


class LSTMSTLPredictor(LSTMBase, STLResidualSquaredErrorPredictor):
    pass
