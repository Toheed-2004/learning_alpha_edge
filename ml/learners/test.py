import joblib
import os
joblib.load
from learning_alpha_edge.data.binance.binance_fetcher import fetch_data
from learning_alpha_edge.utils.data_utils import resample_ohlcv_data
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler

if __name__=="__main__":
    start_date=datetime.datetime.strptime('2025-07-01', "%Y-%m-%d")
    end_date=datetime.datetime.strptime('2025-08-01', "%Y-%m-%d")
    df=fetch_data("BTCUSDT",start_date,end_date)
    df=resample_ohlcv_data(df,"1h")
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
    df=df.drop(columns=["datetime","target"])
    y=df["target"]
    # 5. TA-Lib indicators
    # df=apply_indicators(df, ['sma', 'ema', 'macd', 'rsi', 'atr'])
    model=joblib.load("D:\learning_alpha_edge\ml\train\BTCUSDT\4h\models\ada\ada_4h_binance_ml0.pkl")
    model.predict(df)