# ml/learners/abr.py
import optuna
import pandas as pd
import psycopg2
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from learning_alpha_edge.data_updater.data_downloader import Data_Downloader
from learning_alpha_edge.ml.learners.base_learner import BaseLearner
from sklearn.preprocessing import MinMaxScaler
# from utils.indicators import apply_indicators  
from configparser import ConfigParser
import numpy as np
from learning_alpha_edge.utils.db_utils import get_pg_engine,load_data
from learning_alpha_edge.utils.data_utils import resample_ohlcv_data,generate_signals
from learning_alpha_edge.backtest.main_backtest import Backtester
from learning_alpha_edge.signals.technical_indicators_signals.main_signals import apply_indicators
import os

class AdaBoostLearner(BaseLearner):
    def __init__(self, symbol, time_horizon, exchange):
        
        super().__init__(symbol, time_horizon, "ada", exchange)

        # Load config
        config_path = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(config_path, "config.ini")        
        self.config = ConfigParser()
        self.config.read(config_path)
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
    def suggest_params(self, trial: optuna.Trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
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
        base = DecisionTreeRegressor(
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"]
    )
        return AdaBoostRegressor(
        estimator=base,   # sklearn >=1.2
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"]
    )

    # ----------------- Training -----------------
    def train(self):
        df_base = load_data("binance","btc_1m",self.engine)
        for interval in self.intervals:
            print(f"\n[abr] Training {self.symbol} @ {interval} on {self.exchange}")
                  
            df=resample_ohlcv_data(df_base,interval)
            # Shift target: predict next candle's close
            minmaxscaler=MinMaxScaler()
            df["target"] = df["close"].shift(-1)
            # --- Feature Engineering ---
           
            # 1. Returns
            df["return_1"] = df["close"].pct_change()
            df["log_return"] = np.log(df["close"] / df["close"].shift(1))

            # 2. Lag features
            for lag in range(1, 6):
                df[f"close_lag{lag}"] = df["close"].shift(lag)

            # 3. Rolling statistics
            df["rolling_mean_5"] = df["close"].rolling(5).mean()
            df["rolling_std_5"] = df["close"].rolling(5).std()
            df["volatility_10"] = df["return_1"].rolling(10).std()

            # 4. Volume features
            df["volume_ma_5"] = df["volume"].rolling(5).mean()
            df["volume_change"] = df["volume"].pct_change()
            df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

            # 5. TA-Lib indicators
            df=apply_indicators(df, ['sma', 'ema', 'macd', 'rsi', 'atr'])

        # Drop NaNs introduced by rolling/indicators
            df = df.dropna()
                # Split time-series into train/backtest/forward
            df_train, df_backtest, df_forward = self.split_time_series(
                df, train_pct=75.0, backtest_pct=20.0, forward_pct=5.0
            )
            
            # Build train/val/forward sets
            X_train = df_train.drop(columns=["datetime", "target"])
            X_train=minmaxscaler.fit_transform(X_train)
            y_train = df_train["target"]

            X_val = df_backtest.drop(columns=["datetime", "target"])
            X_val=minmaxscaler.fit_transform(X_val)
            y_val = df_backtest["target"]

            X_forward = df_forward.drop(columns=["datetime", "target"])
            X_forward=minmaxscaler.fit_transform(X_forward)
            y_forward = df_forward["target"]

            def objective(trial:optuna.Trial):
                params = self.suggest_params(trial)
                model = self.build_model(params)
                mean=df_train["target"].mean()
                model.fit(X_train, y_train)

                preds = model.predict(X_val)
                signals = generate_signals(df_backtest,preds)
                backtester=Backtester(signals,df_backtest)
                ledger=backtester.run_backtest()
                pnl=ledger["PnLSum"].iloc[-1]

                # self.log_trial(trial.number, params, pnl)
                return pnl

            # Optuna optimization
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=5)

            best_params = study.best_trial.params
            best_model = self.build_model(best_params)
            best_model.fit(X_train, y_train)
            prediction=best_model.predict(X_forward)
            signals = generate_signals(df_forward,prediction)
            backtester=Backtester(signals,df_forward)
            ledger=backtester.run_backtest()
            pnl=ledger["PnLSum"].iloc[-1]
            print(pnl)

            # Retrain on all available data if desired
            model_name=f'ada_{self.time_horizon}_{self.exchange}_ml{study.best_trial._trial_id}.pkl'
            self.save_model(best_model,model_name)
            self.save_metadata(best_params)
            self.log_trial(study.best_trial.number, best_params, study.best_value, is_best=True)

            print(f"[abr][{interval}] Best PnL: {study.best_value:.2f} with {best_params}")

if __name__=="__main__":
    learner=AdaBoostLearner("BTCUSDT","4h","binance")
    learner.train()