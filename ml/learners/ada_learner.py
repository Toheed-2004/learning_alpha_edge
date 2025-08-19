# ml/learners/abr.py
import optuna
import pandas as pd
import psycopg2
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from learning_alpha_edge.data_updater.data_downloader import Data_Downloader
from learning_alpha_edge.ml.learners.base_learner import BaseLearner
# from utils.indicators import apply_indicators  
from configparser import ConfigParser
from learning_alpha_edge.utils.db_utils import get_pg_engine,load_data
from learning_alpha_edge.utils.data_utils import resample_ohlcv_data,generate_signals
from learning_alpha_edge.backtest.main_backtest import Backtester
import os

class AdaBoostLearner(BaseLearner):
    def __init__(self, symbol, time_horizon, exchange):
        
        super().__init__(symbol, time_horizon, "ada", exchange)

        # Load config
        config_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(config_path, "config.ini")        
        self.config = ConfigParser()
        self.config.read(config_path)
        print(os.path.exists(config_path))
        self.intervals=['1h']
        self.train_split = float(self.config["train"]["train_split_percent"]) / 100
        self.backtest_split = float(self.config["train"]["backtest_split_percent"]) / 100
        self.forwardtest_split = float(self.config["train"]["forwardtest_split_percent"]) / 100
        self.optuna_trials = eval(self.config["train"]["optuna_trials_per_model"])["ada"]
        self.pg_cfg=self.config["postgres"]
        self.user = self.pg_cfg["user"]
        self.password = self.pg_cfg["password"]
        self.host = self.pg_cfg["host"]
        self.port = self.pg_cfg["port"]
        self.dbname = self.pg_cfg["dbname"]
        self.engine=get_pg_engine(self.user,self.password,self.host,self.port,self.dbname)

        self.pg_cfg = self.config["postgres"]

    # ----------------- Optuna API -----------------
    def suggest_params(self, trial:optuna.Trial):
        """Suggest hyperparameters for AdaBoost."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
        }
    
    def split_time_series(self,df:pd.DataFrame, train_pct, backtest_pct, forward_pct):
       
        
        n = len(df)
        train_end = int(n * (train_pct / 100))
        backtest_end = train_end + int(n * (backtest_pct / 100))

        df_train = df.iloc[:train_end]
        df_backtest = df.iloc[train_end:backtest_end]
        df_forward = df.iloc[backtest_end:]

        return df_train, df_backtest, df_forward




    def build_model(self, params):
        return AdaBoostRegressor(**params)

    # ----------------- Data Fetch -----------------
    # def _load_data(self):
    #     user = self.pg_cfg["user"]
    #     password = self.pg_cfg["password"]
    #     host = self.pg_cfg["host"]
    #     port = self.pg_cfg["port"]
    #     dbname = self.pg_cfg["dbname"]
    #     engine=get_pg_engine(user,password,host,port,dbname)
    #     query = "SELECT * FROM binance.btc_1m ORDER BY datetime ASC;"
    #     df = pd.read_sql(query, engine)
    #     return df

    # ----------------- Training -----------------
    def train(self):
        df_base = load_data("binance","btc_1m",self.engine)
        for interval in self.intervals:
            print(f"\n[abr] Training {self.symbol} @ {interval} on {self.exchange}")
                  
            df=resample_ohlcv_data(df_base,interval)
            # Shift target: predict next candle's close
            df["target"] = df["close"].shift(-1)
            df = df.dropna()
            # Split time-series into train/backtest/forward
            df_train, df_backtest, df_forward = self.split_time_series(
                df, train_pct=75.0, backtest_pct=20.0, forward_pct=5.0
            )

            # Build train/val/forward sets
            X_train = df_train.drop(columns=["datetime", "target"])
            y_train = df_train["target"]

            X_val = df_backtest.drop(columns=["datetime", "target"])
            y_val = df_backtest["target"]

            X_forward = df_forward.drop(columns=["datetime", "target"])
            y_forward = df_forward["target"]

            def objective(trial:optuna.Trial):
                params = self.suggest_params(trial)
                model = self.build_model(params)
                model.fit(X_train, y_train)

                preds = model.predict(X_val)
                signals = generate_signals(df_backtest,preds)
                backtester=Backtester(signals,df_backtest)
                ledger=backtester.run_backtest()
                pnl=ledger["PnLSum"].iloc[-1]

                self.log_trial(trial.number, params, pnl)
                return pnl

            # Optuna optimization
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.optuna_trials)

            best_params = study.best_trial.params
            best_model = self.build_model(best_params)
            best_model.fit(X_train, y_train)

            # Retrain on all available data if desired
            model_name=f'ada_{self.time_horizon}_{self.exchange}_ml{study.best_trial._trial_id}.pkl'
            self.save_model(best_model,model_name)
            self.save_metadata(best_params)
            self.log_trial(study.best_trial.number, best_params, study.best_value, is_best=True)

            print(f"[abr][{interval}] Best PnL: {study.best_value:.2f} with {best_params}")

if __name__=="__main__":
    learner=AdaBoostLearner("BTCUSDT","1h","binance")
    learner.train()