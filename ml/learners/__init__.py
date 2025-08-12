# ml/learners/__init__.py
from .sklearn_learners import (
    ABRLearner, ETRLearner, MLPRegressorLearner,
    SGDRegressorLearner, ARDLearner, KNeighborsLearner, SVRLearner
)
# from .bnbc_placeholder import BNBCLearner
from .deep_learning_learners import BRNNLearner, TIDELearner, NHITSLearner, TFTLearner, TCNLearner, TMLearner

_MODEL_REGISTRY = {
    "abr": ABRLearner,
    "etr": ETRLearner,
    "mlpr": MLPRegressorLearner,
    "sgdr": SGDRegressorLearner,
    "ardr": ARDLearner,
    "knc": KNeighborsLearner,
    "svc": SVRLearner,
    # "bnbc": BNBCLearner,
    "brnn": BRNNLearner,
    "tide": TIDELearner,
    "nhits": NHITSLearner,
    "tft": TFTLearner,
    "tcn": TCNLearner,
    "tm": TMLearner,
}

def get_learner_class(key: str):
    return _MODEL_REGISTRY.get(key)
