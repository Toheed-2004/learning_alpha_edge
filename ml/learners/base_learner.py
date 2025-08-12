# ml/learners/base_learner.py
from abc import ABC, abstractmethod
from pathlib import Path
import joblib

class BaseLearner(ABC):
    """
    Abstract base class for all learners.
    Concrete learners must implement:
      - build_model(**kwargs) -> self
      - train(X, y)
      - predict(X) -> np.array
      - suggest_params(trial) -> dict  (staticmethod for Optuna)
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    @abstractmethod
    def build_model(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model is not built/trained.")
        return self.model.predict(X)

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)

    @staticmethod
    def suggest_params(trial):
        """Default (no params). Override in subclasses."""
        return {}
