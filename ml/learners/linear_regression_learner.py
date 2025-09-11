import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error, r2_score
from darts import TimeSeries
from darts.models import LinearRegressionModel
from darts.dataprocessing.transformers import Scaler

from learning_alpha_edge.ml.learners.base_learner import BaseLearner
from learning_alpha_edge.utils.db_utils import load_data
from learning_alpha_edge.utils.data_utils import resample_ohlcv_data


class LinearRegressionLearner(BaseLearner):
    def __init__(self, symbol, time_horizon, exchange):
        super().__init__(symbol, time_horizon, "linreg", exchange)

        # Load and preprocess
        self.df = load_data("binance", "btc_1m", self.engine)
        self.df = resample_ohlcv_data(self.df, self.time_horizon)
        
        # Ensure datetime index is proper
        if 'datetime' not in self.df.columns and self.df.index.name == 'datetime':
            self.df = self.df.reset_index()
        
        # Remove any potential NaN values
        self.df = self.df.dropna()

        # Target = close
        self.series = TimeSeries.from_dataframe(self.df, time_col="datetime", value_cols="close")

        # Past covariates, features = open, high, low, volume
        if set(["open", "high", "low", "volume"]).issubset(self.df.columns):
            self.covariates = TimeSeries.from_dataframe(
                self.df, time_col="datetime", value_cols=["open", "high", "low", "volume"]
            )
        else:
            self.covariates = None

        # Scalers
        self.scaler_target = Scaler()
        self.scaler_covariates = Scaler() if self.covariates is not None else None

        self.model = None
        self.best_params = None

    def _ensure_sufficient_data(self, df, min_length=100):
        """Ensure we have sufficient data for training and prediction"""
        if len(df) < min_length:
            raise ValueError(f"Insufficient data: got {len(df)} rows, need at least {min_length}")
        return df

    def _align_covariates_with_target(self, target_series, covariate_series):
        """Ensure covariates are properly aligned with target series"""
        # Get the time index of the target series
        target_time_index = target_series.time_index
        
        # Filter covariates to match target time index
        covariate_df = covariate_series.pd_dataframe()
        aligned_covariate_df = covariate_df.loc[target_time_index]
        aligned_covariates = TimeSeries.from_dataframe(aligned_covariate_df)
        
        return aligned_covariates

    # ----------------- Training -----------------
    def train(self, train_series, train_covs, params: dict):
        # Ensure covariates are properly aligned
        if train_covs is not None:
            train_covs = self._align_covariates_with_target(train_series, train_covs)
        
        self.model = LinearRegressionModel(
            lags=params["lags"],
            lags_past_covariates=params["lags_past_covariates"] if train_covs is not None else None,
            output_chunk_length=params["output_chunk_length"],
            n_jobs=-1,
        )
        self.model.fit(train_series, past_covariates=train_covs)

    # ----------------- Metrics -----------------
    def compute_metrics(self, series_true: TimeSeries, series_pred: TimeSeries, dataset_name=""):
        y_true = series_true.values().flatten()
        y_pred = series_pred.values().flatten()

        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        pnl = self.evaluate_pnl(series_true, series_pred)

        print(f"{dataset_name} -> RMSE: {rmse:.4f}, R²: {r2:.4f}, PnL: {pnl:.4f}")
        return {"rmse": rmse, "r2": r2, "pnl": pnl}

    # ----------------- PnL -----------------
    def evaluate_pnl(self, series_true: TimeSeries, series_pred: TimeSeries, fees=0.0005):
        y_true = series_true.values().flatten()
        y_pred = series_pred.values().flatten()

        # Simple strategy: buy if predicted price > current price, sell otherwise
        signals = np.zeros_like(y_true)
        for i in range(1, len(y_pred)):
            if y_pred[i] > y_true[i-1]:
                signals[i] = 1  # Buy
            else:
                signals[i] = -1  # Sell

        returns = np.diff(y_true, prepend=y_true[0]) / y_true
        pnl = np.sum(signals * returns - fees * np.abs(signals))
        return pnl

    # ----------------- Optuna Objective -----------------
    def objective(self, trial: optuna.Trial):
        train_df, backtest_df, _ = self.split_time_series(
            self.df, self.train_split, self.backtest_split
        )
        
        # Ensure sufficient data
        train_df = self._ensure_sufficient_data(train_df)
        backtest_df = self._ensure_sufficient_data(backtest_df)

        # series
        train_series = TimeSeries.from_dataframe(train_df, "datetime", "close")
        backtest_series = TimeSeries.from_dataframe(backtest_df, "datetime", "close")

        # scaling
        train_series_scaled = self.scaler_target.fit_transform(train_series)
        backtest_series_scaled = self.scaler_target.transform(backtest_series)

        # covariates
        if self.covariates is not None:
            covs_train = TimeSeries.from_dataframe(train_df, "datetime", ["open", "high", "low", "volume"])
            covs_backtest = TimeSeries.from_dataframe(backtest_df, "datetime", ["open", "high", "low", "volume"])
            
            # Align covariates with target series
            covs_train = self._align_covariates_with_target(train_series, covs_train)
            covs_backtest = self._align_covariates_with_target(backtest_series, covs_backtest)
            
            covs_train_scaled = self.scaler_covariates.fit_transform(covs_train)
            covs_backtest_scaled = self.scaler_covariates.transform(covs_backtest)
        else:
            covs_train_scaled = covs_backtest_scaled = None

        # hyperparams - ensure reasonable values
        max_lags = min(50, len(train_series) // 2)  # Don't exceed half the training data
        params = {
            "lags": trial.suggest_int("lags", 5, max_lags),
            "output_chunk_length": trial.suggest_int("output_chunk_length", 1, 5),
            "lags_past_covariates": trial.suggest_int("lags_past_covariates", 1, 10) if self.covariates is not None else 0,
        }

        try:
            # train
            self.train(train_series_scaled, covs_train_scaled, params)

            # predict - ensure we don't predict more than available data
            n_predict = min(len(backtest_series_scaled), 1000)  # Limit prediction length
            
            preds_scaled = self.model.predict(
                n=n_predict, 
                series=train_series_scaled,  # Use training series for historical context
                past_covariates=covs_backtest_scaled
            )
            
            # Align predictions with actual backtest data
            pred_time_index = preds_scaled.time_index
            actual_backtest_aligned = backtest_series_scaled.slice(
                pred_time_index[0], pred_time_index[-1]
            )
            
            preds = self.scaler_target.inverse_transform(preds_scaled)
            actual_backtest = self.scaler_target.inverse_transform(actual_backtest_aligned)

            # metrics
            metrics = self.compute_metrics(actual_backtest, preds, "Validation")
            return metrics["pnl"]  # Optuna maximizes PnL
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return -float('inf')  # Return very poor score for failed trials

    # ----------------- Optimization -----------------
    def optimize(self, n_trials=30):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial), n_trials=n_trials)
        self.best_params = study.best_params
        print("Best params:", self.best_params)
        return study.best_value

    # ----------------- Final Pipeline -----------------
    def run_pipeline(self):
        # Optuna search
        best_pnl = self.optimize(n_trials=20)
        print(f"Best validation PnL: {best_pnl:.4f}")

        # Final split
        train_df, backtest_df, test_df = self.split_time_series(
            self.df, self.train_split, self.backtest_split
        )
        
        # Ensure sufficient data
        train_df = self._ensure_sufficient_data(train_df)
        backtest_df = self._ensure_sufficient_data(backtest_df)
        test_df = self._ensure_sufficient_data(test_df)

        # Combine train and backtest for final training
        train_all_df = pd.concat([train_df, backtest_df])
        
        # series
        train_all_series = TimeSeries.from_dataframe(train_all_df, "datetime", "close")
        test_series = TimeSeries.from_dataframe(test_df, "datetime", "close")

        # scaling
        train_all_series_scaled = self.scaler_target.fit_transform(train_all_series)
        test_series_scaled = self.scaler_target.transform(test_series)

        # covariates
        if self.covariates is not None:
            covs_train_all = TimeSeries.from_dataframe(train_all_df, "datetime", ["open", "high", "low", "volume"])
            covs_test = TimeSeries.from_dataframe(test_df, "datetime", ["open", "high", "low", "volume"])
            
            # Align covariates
            covs_train_all = self._align_covariates_with_target(train_all_series, covs_train_all)
            covs_test = self._align_covariates_with_target(test_series, covs_test)
            
            covs_train_all_scaled = self.scaler_covariates.fit_transform(covs_train_all)
            covs_test_scaled = self.scaler_covariates.transform(covs_test)
        else:
            covs_train_all_scaled = covs_test_scaled = None

        try:
            # retrain
            self.train(train_all_series_scaled, covs_train_all_scaled, self.best_params)

            # predict unseen test - limit prediction length
            n_predict = min(len(test_series_scaled), 1000)
            
            preds_scaled = self.model.predict(
                n=n_predict, 
                series=train_all_series_scaled,  # Use full training data for context
                past_covariates=covs_test_scaled
            )
            
            # Align predictions with test data
            pred_time_index = preds_scaled.time_index
            actual_test_aligned = test_series_scaled.slice(
                pred_time_index[0], pred_time_index[-1]
            )
            
            preds = self.scaler_target.inverse_transform(preds_scaled)
            actual_test = self.scaler_target.inverse_transform(actual_test_aligned)

            # metrics on final unseen
            final_metrics = self.compute_metrics(actual_test, preds, "Final Test")
            return final_metrics
            
        except Exception as e:
            print(f"Final prediction failed: {e}")
            return None


# ----------------- Example Run -----------------
if __name__ == "__main__":
    learner = LinearRegressionLearner("BTCUSDT", "1h", "binance")
    results = learner.run_pipeline()
    
    if results:
        print(f"\nFinal Results:")
        print(f"Test RMSE: {results['rmse']:.4f}")
        print(f"Test R²: {results['r2']:.4f}")
        print(f"Test PnL: {results['pnl']:.4f}")