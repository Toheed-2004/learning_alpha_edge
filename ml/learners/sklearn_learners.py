# ml/learners/sklearn_learners.py
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor, ARDRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from .base_learner import BaseLearner

# --- AdaBoost Regressor (abr) ---
class ABRLearner(BaseLearner):
    def __init__(self):
        super().__init__("abr")

    def build_model(self, **kwargs):
        self.model = AdaBoostRegressor(**kwargs)
        return self

    def train(self, X, y):
        self.model.fit(X, y)
        return self

    @staticmethod
    def suggest_params(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
            "loss": trial.suggest_categorical("loss", ["linear", "square", "exponential"]),
        }

# --- ExtraTrees Regressor (etr) ---
class ETRLearner(BaseLearner):
    def __init__(self):
        super().__init__("etr")

    def build_model(self, **kwargs):
        self.model = ExtraTreesRegressor(**kwargs)
        return self

    def train(self, X, y):
        self.model.fit(X, y)
        return self

    @staticmethod
    def suggest_params(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }

# --- MLP Regressor (mlpr) ---
class MLPRegressorLearner(BaseLearner):
    def __init__(self):
        super().__init__("mlpr")

    def build_model(self, **kwargs):
        # kwargs passed to sklearn MLPRegressor
        self.model = MLPRegressor(**kwargs)
        return self

    def train(self, X, y):
        self.model.fit(X, y)
        return self

    @staticmethod
    def suggest_params(trial):
        # define small search space (expand as needed)
        hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [(50,), (100,), (100,50)])
        alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
        learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True)
        return {
            "hidden_layer_sizes": hidden_layer_sizes,
            "alpha": alpha,
            "learning_rate_init": learning_rate_init,
            "max_iter": 500,
        }

# --- SGD Regressor (sgdr) ---
class SGDRegressorLearner(BaseLearner):
    def __init__(self):
        super().__init__("sgdr")

    def build_model(self, **kwargs):
        self.model = SGDRegressor(**kwargs)
        return self

    def train(self, X, y):
        self.model.fit(X, y)
        return self

    @staticmethod
    def suggest_params(trial):
        return {
            "loss": trial.suggest_categorical("loss", ["squared_loss", "huber", "epsilon_insensitive"]),
            "penalty": trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
            "learning_rate": trial.suggest_categorical("learning_rate", ["invscaling", "constant", "adaptive"]),
            "max_iter": 10000,
        }

# --- ARD Regression (ardr) ---
class ARDLearner(BaseLearner):
    def __init__(self):
        super().__init__("ardr")

    def build_model(self, **kwargs):
        self.model = ARDRegression(**kwargs)
        return self

    def train(self, X, y):
        self.model.fit(X, y)
        return self

    @staticmethod
    def suggest_params(trial):
        # ARD has few tunables; we can keep defaults or tune alpha_1/beta_1
        return {
            # typically leave defaults; provide a placeholder params map
        }

# --- KNeighbors Regressor (knc) ---
class KNeighborsLearner(BaseLearner):
    def __init__(self):
        super().__init__("knc")

    def build_model(self, **kwargs):
        self.model = KNeighborsRegressor(**kwargs)
        return self

    def train(self, X, y):
        self.model.fit(X, y)
        return self

    @staticmethod
    def suggest_params(trial):
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 50),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_categorical("p", [1, 2]),
        }

# --- SVR (svc key used but we implement SVR regressor) ---
class SVRLearner(BaseLearner):
    def __init__(self):
        super().__init__("svc")

    def build_model(self, **kwargs):
        self.model = SVR(**kwargs)
        return self

    def train(self, X, y):
        self.model.fit(X, y)
        return self

    @staticmethod
    def suggest_params(trial):
        return {
            "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
        }
