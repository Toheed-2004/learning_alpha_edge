# ml/learners/deep_placeholders.py
from .base_learner import BaseLearner

class BRNNLearner(BaseLearner):   # brnn
    def __init__(self):
        super().__init__("brnn")
    def build_model(self, **kwargs):
        raise NotImplementedError("BRNN (RNN) not implemented yet.")
    def train(self, X, y):
        raise NotImplementedError

class TIDELearner(BaseLearner):   # tide
    def __init__(self):
        super().__init__("tide")
    def build_model(self, **kwargs):
        raise NotImplementedError("TIDE not implemented yet.")
    def train(self, X, y):
        raise NotImplementedError

class NHITSLearner(BaseLearner):   # nhits
    def __init__(self):
        super().__init__("nhits")
    def build_model(self, **kwargs):
        raise NotImplementedError("NHITS not implemented yet.")
    def train(self, X, y):
        raise NotImplementedError

class TFTLearner(BaseLearner):    # tft
    def __init__(self):
        super().__init__("tft")
    def build_model(self, **kwargs):
        raise NotImplementedError("TFT not implemented yet.")
    def train(self, X, y):
        raise NotImplementedError

class TCNLearner(BaseLearner):    # tcn
    def __init__(self):
        super().__init__("tcn")
    def build_model(self, **kwargs):
        raise NotImplementedError("TCN not implemented yet.")
    def train(self, X, y):
        raise NotImplementedError

class TMLearner(BaseLearner):     # tm
    def __init__(self):
        super().__init__("tm")
    def build_model(self, **kwargs):
        raise NotImplementedError("TM/Transformer not implemented yet.")
    def train(self, X, y):
        raise NotImplementedError
