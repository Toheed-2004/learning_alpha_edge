# ml/learners/abr.py
import optuna
import pandas as pd
import psycopg2
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from learning_alpha_edge.utils.data_utils import engineer_features
from sklearn.tree import DecisionTreeRegressor
from learning_alpha_edge.data_updater.data_downloader import Data_Downloader
from learning_alpha_edge.ml.learners.base_learner import BaseLearner
from sklearn.preprocessing import StandardScaler
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
        df_base = load_data("binance", "btc_1m", self.engine)

        for interval in self.intervals:
            print(f"\n[abr] Training {self.symbol} @ {interval} on {self.exchange}")

            def objective(trial: optuna.Trial):
                # 1) Resample
                df = resample_ohlcv_data(df_base, interval)

                # 2) Target = next close
                df["target"] = df["close"].shift(-1)

                # 3) Trial-dependent features
                df = engineer_features(df, self.config, trial)

                # 4) Drop NaNs
                df = df.dropna()

                # 5) Split
                df_train, df_backtest, df_forward = self.split_time_series(
                    df, train_pct=75.0, backtest_pct=20.0, forward_pct=5.0
                )

                # 6) Scale (fit only on train)
                scaler =StandardScaler()
                X_train = (df_train.drop(columns=["datetime", "target"]))
                y_train = df_train["target"]

                X_val = (df_backtest.drop(columns=["datetime", "target"]))
                y_val = df_backtest["target"]

                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

                # 7) Build + fit model
                params = self.suggest_params(trial)
                model = self.build_model(params)
                model.fit(X_train, y_train)

                # 8) Backtest on validation
                preds = model.predict(X_val)
                signals = generate_signals(df_backtest, preds)
                backtester = Backtester(signals, df_backtest)
                pnl = backtester.run_backtest()["PnLSum"].iloc[-1]

                return pnl

            # Run optimization for this interval
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=5)

            #  Best model training on all data
            best_params = study.best_trial.params
            best_model = self.build_model(best_params)

            # Resample + feature engineering again with best params
            df = resample_ohlcv_data(df_base, interval)
            df["target"] = df["close"].shift(-1)
            df = engineer_features(df, self.config, study.best_trial)  # use best trial
            df = df.dropna()

            df_train, df_backtest, df_forward = self.split_time_series(
                df, train_pct=75.0, backtest_pct=20.0, forward_pct=5.0
            )

            scaler = StandardScaler()
            X_train = (df_train.drop(columns=["datetime", "target"]))
            y_train = df_train["target"]

            X_forward = (df_forward.drop(columns=["datetime", "target"]))
            y_forward = df_forward["target"]
            X_train = scaler.fit_transform(X_train)
            X_forward = scaler.transform(X_forward)

            best_model.fit(X_train, y_train)
            prediction = best_model.predict(X_forward)
            y_forward_pred = prediction
            mse_f = mean_squared_error(y_forward, y_forward_pred)
            rmse_f = np.sqrt(mse_f)
            mae_f = mean_absolute_error(y_forward, y_forward_pred)
            r2_f = r2_score(y_forward, y_forward_pred)
            signals = generate_signals(df_forward, prediction)
            pnl = Backtester(signals, df_forward).run_backtest()["PnLSum"].iloc[-1]
            print("\nForward Test Performance:")
            print(f" MSE : {mse_f:.6f}")
            print(f" RMSE: {rmse_f:.6f}")
            print(f" MAE : {mae_f:.6f}")
            print(f" R²  : {r2_f:.6f}")

            print(f"[abr][{interval}] Forward PnL: {pnl:.2f}")
            print(f"[abr][{interval}] Best trial params: {best_params}")

            # Save model + metadata
            model_name = f'ada_{self.time_horizon}_{self.exchange}_{interval}_ml{study.best_trial.number}.pkl'
            self.save_model(best_model, model_name)
            self.save_metadata(best_params)
            self.log_trial(study.best_trial.number, best_params, study.best_value, is_best=True)

if __name__=="__main__":
    learner=AdaBoostLearner("BTCUSDT","1h","binance")
    learner.train()