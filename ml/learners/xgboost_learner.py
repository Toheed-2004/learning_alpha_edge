import pandas as pd
import numpy as np
import xgboost
from learning_alpha_edge.ml.learners.base_learner import BaseLearner
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from learning_alpha_edge.utils.db_utils import load_data
from learning_alpha_edge.utils.data_utils import resample_ohlcv_data
from typing import Tuple, Dict, Any
import optuna
from optuna.samplers import TPESampler


class xgboost_learner(BaseLearner):
    def __init__(self, symbol, time_horizon, exchange):
        super().__init__(symbol, time_horizon, "xgb", exchange)
        self.df = load_data("binance", "btc_1m", self.engine)
        self.df = resample_ohlcv_data(self.df, self.time_horizon)
        self.model = None
        self.scaler = StandardScaler()
        self.signals_df = None
        self.best_params = None

    def create_lagged_features(self, df:pd.DataFrame, N=10):
        """Create lagged OHLCV features to predict next close."""
        if 'datetime' in df.columns:
           df = df.set_index('datetime')
        features = ["open", "high", "low", "close", "volume"]
        data = df[features].to_numpy()
        datetime_index = df.index[N:]
        num_rows, num_features = data.shape

        X, y = [], []
        for end_idx in range(N - 1, num_rows - 1):
            window = data[end_idx - N + 1 : end_idx + 1].flatten()
            target = data[end_idx + 1, features.index("close")]
            X.append(window)
            y.append(target)

        return np.array(X), np.array(y), datetime_index

    def generate_signals(self, predictions, true_prices, datetime_index, threshold=0.01):
        """Generate trading signals based on predictions vs actual prices."""
        signals = []
        
        for pred, actual in zip(predictions, true_prices):
            pred_return = (pred - actual) / actual
            
            if pred_return > threshold:
                signal = 1  # Buy
            elif pred_return < -threshold:
                signal = -1  # Sell
            else:
                signal = 0  # Hold
            
            signals.append(signal)
        
        signals_df = pd.DataFrame({
            'signal': signals,
            'actual_price': true_prices,
            'predicted_price': predictions
        }, index=datetime_index)
        
        signals_df = signals_df.reset_index()
        signals_df = signals_df.rename(columns={'index': 'datetime'})
        return signals_df
    
    def prepare_backtest_data(self, df:pd.DataFrame):
        """Prepare DataFrame for backtester with datetime as column."""
        if 'datetime' not in df.columns and df.index.name == 'datetime':
            df = df.reset_index()
        return df

    def objective(self, trial:optuna.Trial, train_df, backtest_df, N, initial_train_size):
        """Optuna objective function to maximize PnL - TRAIN ONCE PER TRIAL."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 10)
        }
        
        X_train, y_train, train_datetime = self.create_lagged_features(train_df, N=N)
        X_backtest, y_backtest, backtest_datetime = self.create_lagged_features(backtest_df, N=N)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = xgboost.XGBRegressor(
            n_estimators=params.get('n_estimators', 100),
            learning_rate=params.get('learning_rate', 0.1),
            max_depth=params.get('max_depth', 5),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            gamma=params.get('gamma', 0),
            reg_alpha=params.get('reg_alpha', 0),
            reg_lambda=params.get('reg_lambda', 1),
            random_state=42,
            verbosity=0
        )
        
        model.fit(X_train_scaled, y_train)
        
        X_backtest_scaled = scaler.transform(X_backtest)
        predictions = model.predict(X_backtest_scaled)
        
        signals_df = self.generate_signals(predictions, y_backtest, backtest_datetime, threshold=0.005)
        
        try:
            backtest_data_prepared = self.prepare_backtest_data(backtest_df)
            pnl = self.run_backtest(backtest_data_prepared,signals_df )
            self.log_trial(trial.number, params, pnl, is_best=False)
            return pnl
        except Exception as e:
            print(f"Backtest failed in trial: {e}")
            return -float('inf')

    def optimize_hyperparameters(self, train_df, backtest_df, N=10, n_trials=50):
        """Find best hyperparameters using Optuna based on REAL PnL maximization."""
        initial_train_size = len(train_df) - N
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42) # Use Seed For Reproducibility
        )
        
        study.optimize(
            lambda trial: self.objective(trial, train_df, backtest_df, N, initial_train_size),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        self.best_params = study.best_params
        print(f"Best hyperparameters: {study.best_params}")
        print(f"Best REAL PnL: {study.best_value:.2f}")
        
        return study.best_params

    def run_backtest(self, df, signals_df):
        """Interface to your existing backtester."""
        from learning_alpha_edge.backtest.main_backtest import Backtester
        backtester = Backtester(signals_df, df)
        pnl = backtester.run_backtest()["PnLSum"].iloc[-1]
        return pnl

    def walk_forward_validation(self, X_full, y_full, initial_train_size, datetime_index, 
                              params=None, step_size=10):
        """
        Walk-forward validation with step size to reduce retraining frequency.
        Retrains only after every 'step_size' predictions instead of every prediction.
        """
        X_train = X_full[:initial_train_size]
        y_train = y_full[:initial_train_size]
        
        predictions = []
        true_values = []
        models = []
        
        num_predictions = len(X_full) - initial_train_size
        
        for i in range(0, num_predictions, step_size):
            # Determine the end index for this batch
            end_idx = min(initial_train_size + i + step_size, len(X_full))
            
            # Get the batch of data for prediction
            X_batch = X_full[initial_train_size + i : end_idx]
            y_batch = y_full[initial_train_size + i : end_idx]
            
            # Scale features and train model
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            model = xgboost.XGBRegressor(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 5),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.8),
                gamma=params.get('gamma', 0),
                reg_alpha=params.get('reg_alpha', 0),
                reg_lambda=params.get('reg_lambda', 1),
                random_state=42,
                verbosity=0
            )
            model.fit(X_train_scaled, y_train)
            
            # Predict on the entire batch
            X_batch_scaled = self.scaler.transform(X_batch)
            batch_predictions = model.predict(X_batch_scaled)
            
            # Store results
            predictions.extend(batch_predictions)
            true_values.extend(y_batch)
            models.append(model)  # Store the last model of each batch
            
            # Append the entire batch to training data for next iteration
            X_train = np.vstack([X_train, X_batch])
            y_train = np.append(y_train, y_batch)
            
            print(f"Processed batch {i//step_size + 1}/{(num_predictions + step_size - 1)//step_size}: "
                  f"{i + step_size}/{num_predictions} predictions")
        
        return np.array(predictions), np.array(true_values), models

    def train_and_generate_signals(self, train_df, backtest_df, N=10, 
                                 use_optuna=True, n_trials=30, threshold=0.01,
                                 step_size=10):
        """Complete training pipeline with step-based walk-forward."""
        if use_optuna:
            print("Optimizing hyperparameters with Optuna...")
            self.optimize_hyperparameters(train_df, backtest_df, N, n_trials)
        
        combined_df = pd.concat([train_df, backtest_df])
        X_combined, y_combined, combined_datetime = self.create_lagged_features(combined_df, N=N)
        initial_train_size = len(train_df) - N
        
        print(f"Running step-based walk-forward validation (step_size={step_size})...")
        predictions, true_values, models = self.walk_forward_validation(
            X_combined, y_combined, initial_train_size, combined_datetime, 
            self.best_params, step_size
        )
        
        # Store the final model
        self.model = models[-1] if models else None
        self.save_model(self.model, self.model_name)
        self.save_metadata(self.best_params)
        # Generate signals
        backtest_start_idx = initial_train_size
        backtest_datetime = combined_datetime[backtest_start_idx:]
        signals_df = self.generate_signals(predictions, true_values, backtest_datetime, threshold)
        
        # Calculate metrics using REAL backtest
        backtest_data_prepared = self.prepare_backtest_data(backtest_df)
        final_pnl = self.run_backtest(backtest_data_prepared,signals_df)
        
        metrics = {
            'mse': np.mean((predictions - true_values)**2),
            'r2': r2_score(true_values, predictions),
            'final_pnl': final_pnl,
            'num_retrainings': len(models),
            'step_size': step_size
        }
        
        return signals_df, metrics

    def forward_test(self, test_df, N=10, threshold=0.01):
        """Generate signals for forward testing using the trained model."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        X_test, y_test, datetime_index = self.create_lagged_features(test_df, N=N)
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        
        signals_df = self.generate_signals(predictions, y_test, datetime_index, threshold)
        
        return signals_df

    def run_full_pipeline(self, train_df:pd.DataFrame, backtest_df:pd.DataFrame, test_df:pd.DataFrame, N=10, 
                         use_optuna=True, n_trials=30, threshold=0.01,
                         step_size=10):
        """Complete pipeline with configurable step size."""
        train_df_prepared = self.prepare_backtest_data(train_df.copy())
        backtest_df_prepared = self.prepare_backtest_data(backtest_df.copy())
        test_df_prepared = self.prepare_backtest_data(test_df.copy())
        
        # print("=== BACKTEST PHASE ===")
        backtest_signals, metrics = self.train_and_generate_signals(
            train_df_prepared, backtest_df_prepared, N, use_optuna, n_trials, threshold, step_size
        )
        
        print("=== FORWARD TEST PHASE ===")
        forward_test_signals = self.forward_test(test_df_prepared, N, threshold)
        
        backtest_pnl = self.run_backtest( backtest_df_prepared,backtest_signals)
        forward_test_pnl = self.run_backtest( test_df_prepared,forward_test_signals)        
        print(f"\n=== RESULTS ===")
        print(f"Backtest PnL: {backtest_pnl:.2f}")
        print(f"Forward Test PnL: {forward_test_pnl:.2f}")
        print(f"Model MSE: {metrics['mse']:.6f}, R²: {metrics['r2']:.4f}")
        print(f"Number of retrainings: {metrics['num_retrainings']} (step_size={metrics['step_size']})")
        
        return backtest_signals, forward_test_signals, backtest_pnl, forward_test_pnl


# Usage
if __name__ == '__main__':
    learner = xgboost_learner("BTCUSDT", "1h", "binance")
   
    train_df, backtest_df, test_df = learner.split_time_series(learner.df,learner.train_split,learner.backtest_split)        
    
    # Run with step size - adjust based on your data size
    backtest_signals, forward_test_signals, backtest_pnl, test_pnl = learner.run_full_pipeline(
        train_df, backtest_df, test_df, 
        N=24, 
        use_optuna=True, 
        n_trials=20,  # Reduced for faster optimization
        threshold=0.005,
        step_size=200 # Retrain every 200 predictions instead of every prediction to reduce overhead
    )
    
    print(f"\nFinal Results:")
    print(f"Backtest PnL: {backtest_pnl:.2f}%")
    print(f"Forward Test PnL: {test_pnl:.2f}%")