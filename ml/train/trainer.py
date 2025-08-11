import os
from pathlib import Path
import pandas as pd
from ml.utils.config import Config
from ml.learners.abr_learner import AdaBoostRegressorLearner
from ml.learners.etr_learner import ExtraTreesRegressorLearner
from ml.learners.mlpr_learner import MLPRegressorLearner
# import other learners similarly...

MODEL_MAP = {
    "abr": AdaBoostRegressorLearner,
    "etr": ExtraTreesRegressorLearner,
    "mlpr": MLPRegressorLearner,
    # Add rest here...
}

class Trainer:
    def __init__(self, config_path, data_path, symbol, timeframe, exchange):
        self.config = Config(config_path)
        self.data_path = Path(data_path)
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = exchange

        # Prepare save dirs
        self.model_dir = self.data_path / symbol / timeframe / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        # Implement your own logic to load OHLCV and indicators dataframe for symbol/timeframe
        # For now dummy:
        df = pd.read_csv(self.data_path / f"{self.symbol}_{self.timeframe}.csv", parse_dates=["datetime"])
        return df

    def split_data(self, df):
        train_pct = self.config.get_train_split() / 100
        backtest_pct = self.config.get_backtest_split() / 100
        forwardtest_pct = self.config.get_forwardtest_split() / 100

        n = len(df)
        train_end = int(n * train_pct)
        backtest_end = train_end + int(n * backtest_pct)

        train_df = df.iloc[:train_end]
        backtest_df = df.iloc[train_end:backtest_end]
        forwardtest_df = df.iloc[backtest_end:]

        return train_df, backtest_df, forwardtest_df

    def train(self):
        models_to_train = self.config.get_models_to_train()
        trials_dict = self.config.get_optuna_trials()
        metric = self.config.get_optimization_metric()

        df = self.load_data()
        train_df, backtest_df, forwardtest_df = self.split_data(df)

        # Assuming 'close' is target
        target_col = "close"
        feature_cols = [c for c in df.columns if c not in ["datetime", target_col]]

        X_train, y_train = train_df[feature_cols], train_df[target_col]
        X_backtest, y_backtest = backtest_df[feature_cols], backtest_df[target_col]
        X_forwardtest, y_forwardtest = forwardtest_df[feature_cols], forwardtest_df[target_col]

        for model_code in models_to_train:
            learner_cls = MODEL_MAP.get(model_code)
            if learner_cls is None:
                print(f"[WARN] No learner found for code '{model_code}'")
                continue

            print(f"Training model {model_code}...")

            learner = learner_cls()
            # For now no hyperparam tuning; just train
            learner.train(X_train, y_train)

            # Evaluate on backtest & forwardtest
            backtest_score = learner.evaluate(X_backtest, y_backtest)
            forwardtest_score = learner.evaluate(X_forwardtest, y_forwardtest)

            print(f"{model_code} Backtest score ({metric}): {backtest_score}")
            print(f"{model_code} Forwardtest score ({metric}): {forwardtest_score}")

            # Save model
            model_filename = f"{self.symbol}_{self.timeframe}_{self.exchange}_{model_code}.pkl"
            save_path = self.model_dir / model_filename
            learner.save_model(save_path)
            print(f"Saved model to {save_path}")

            # TODO: Save training results metadata to db/json etc.

if __name__ == "__main__":
    # Example usage
    trainer = Trainer(
        config_path="config.ini",
        data_path="ml/train",
        symbol="btc",
        timeframe="4H",
        exchange="binance"
    )
    trainer.train()
