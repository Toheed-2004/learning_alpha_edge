import configparser
import json
from pathlib import Path

class Config:
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.parser = configparser.ConfigParser()
        self.parser.read(self.config_path)

    def get_models_to_train(self):
        val = self.parser.get("train", "models_to_train", fallback="")
        # Clean and split comma-separated string into list
        models = [m.strip() for m in val.split(",") if m.strip()]
        return models

    def get_train_split(self):
        return self.parser.getfloat("train", "train_split_percent", fallback=75.0)

    def get_backtest_split(self):
        return self.parser.getfloat("train", "backtest_split_percent", fallback=20.0)

    def get_forwardtest_split(self):
        return self.parser.getfloat("train", "forwardtest_split_percent", fallback=5.0)

    def get_optuna_trials(self):
        raw = self.parser.get("train", "optima_trials_per_model", fallback="{}")
        try:
            return json.loads(raw.replace("'", '"'))
        except json.JSONDecodeError:
            print("[WARN] Invalid JSON for optima_trials_per_model. Using empty dict.")
            return {}

    def get_optimization_metric(self):
        return self.parser.get("train", "optimization_metric", fallback="pnl")
