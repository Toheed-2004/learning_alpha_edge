# ml/learners/base_learner.py
import os, json
import pandas as pd
from configparser import ConfigParser
from learning_alpha_edge.utils.db_utils import get_pg_engine
import joblib
import sqlite3

class BaseLearner:
    def __init__(self, symbol:str, time_horizon, model_name, exchange):
        self.symbol = symbol.lower()
        self.time_horizon = time_horizon
        self.model_name = model_name
        self.exchange = exchange
        #PATH
        self.root_path=os.path.dirname(os.path.abspath(__file__))
        self.root_path = os.path.abspath(os.path.join(self.root_path, ".."))
        self.root_path = os.path.join(self.root_path, "train", symbol, time_horizon)
        os.makedirs(self.root_path, exist_ok=True)
        self.model_path = os.path.join(self.root_path, "models",self.model_name)
        os.makedirs(self.model_path, exist_ok=True)
        self.db_path = os.path.join(self.root_path, "dbs")
        os.makedirs(self.db_path, exist_ok=True)
        self.meta_path = os.path.join(self.root_path, "metadata")
        os.makedirs(self.meta_path, exist_ok=True)

        # Config
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")
        self.config = ConfigParser()
        self.config.read(config_path)

        # DB engine
    
        pg_cfg = self.config["postgres"]
        self.engine = get_pg_engine(pg_cfg["user"], pg_cfg["password"], pg_cfg["host"], pg_cfg["port"], pg_cfg["dbname"])

    def split_time_series(self, df: pd.DataFrame, train_pct, backtest_pct):
        n = len(df)
        train_end = int(n * (train_pct / 100))
        backtest_end = train_end + int(n * (backtest_pct / 100))

        return df.iloc[:train_end], df.iloc[train_end:backtest_end], df.iloc[backtest_end:]
    
        

    def save_model(self, model,model_name):
        """
        Save the trained model to the model_path.
        """
        # ensure directory exists
        model_file = os.path.join(self.model_path, f"{model_name}")
        
        with open(model_file, "wb") as f:
            joblib.dump(model, f)
        
        print(f"[INFO] Model saved at {model_file}")

    def save_metadata(self, params):
        """
        Save ML model metadata including input features and hyperparameters.
        
        params         : dict of hyperparameters used for this trial
        input_features : list of column names / feature identifiers used as model input
        trial_number   : optional int, trial id
        """
        meta_file=os.path.join(self.meta_path,"metadata.json")
        
        metadata = {
            
            "model_name": self.model_name,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "time_horizon": self.time_horizon,
            "params": params,
            "input_features": ["Open","High","Low","Close","Target(close.shift(-1))"]
        }
        
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=4)
        
        print(f"[INFO] Metadata for {self.model_name} saved at {self.meta_path}")

    def log_trial(self, trial_number, params, PnL, is_best=False):
                db_file=os.path.join(self.db_path,"training_results.db")
                conn = sqlite3.connect(db_file)
                cur = conn.cursor()
                params=json.dumps(params)
                df = pd.DataFrame([{
                    "trial_number": trial_number,
                    "model_name": self.model_name,
                    "exchange": self.exchange,
                    "symbol": self.symbol,
                    "time_horizon": self.time_horizon,
                    "params": params,
                    "pnl": PnL,
                    "is_best": is_best
                }])
                df.to_sql(
                    name="trials",
                    con=conn,
                    if_exists="append",
                    index=False
                )

               
                
                print(f"[trial {trial_number}] score={PnL:.4f}, params={params}, best={is_best}")
