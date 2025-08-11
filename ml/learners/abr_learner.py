from sklearn.ensemble import AdaBoostRegressor
from .base_learner import BaseLearner

class AdaBoostRegressorLearner(BaseLearner):
    def __init__(self, params=None):
        super().__init__(params)
        self.model = AdaBoostRegressor(**(params or {}))

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        from sklearn.metrics import mean_squared_error
        preds = self.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        return mse
