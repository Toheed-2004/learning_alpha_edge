from abc import ABC, abstractmethod

class BaseLearner(ABC):
    def __init__(self, params=None):
        self.params = params or {}

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        pass

    def save_model(self, path):
        import joblib
        joblib.dump(self.model, path)

    def load_model(self, path):
        import joblib
        self.model = joblib.load(path)
