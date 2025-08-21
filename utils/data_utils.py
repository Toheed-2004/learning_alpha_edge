import pandas as pd
import inspect
from sklearn.preprocessing import MinMaxScaler
from learning_alpha_edge.technical_indicators.ti import indicator_map
import numpy as np
import configparser
from configparser import ConfigParser
import optuna

def preprocess_klines(df:pd.DataFrame, interpolate_method='linear', fill_zero_volume='ffill'):
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']

    # Interpolate missing values
    df[numeric_cols] = df[numeric_cols].interpolate(method=interpolate_method, limit_direction='both')

    # Handle zero volume
    if fill_zero_volume == 'ffill':
        df['volume'] = df['volume'].replace(0, pd.NA).ffill()
    elif fill_zero_volume == 'bfill':
        df['volume'] = df['volume'].replace(0, pd.NA).fillna(method='bfill')
    elif fill_zero_volume == 'drop':
        df = df[df['volume'] != 0]

    # Drop any remaining NaNs
    df.dropna(inplace=True)

    # Round and enforce float64
    df[numeric_cols] = df[numeric_cols].round(2).astype('float64')
    return df


def resample_ohlcv_data(df: pd.DataFrame, new_interval: str) -> pd.DataFrame:
    """
    Resample a clean OHLCV DataFrame to a higher timeframe.
    
    Args:
        df (pd.DataFrame): DataFrame with datetime index and OHLCV columns.
        new_interval (str): Pandas-compatible resample interval (e.g., '15min', '1H').

    Returns:
        pd.DataFrame: Resampled OHLCV data.
    """

    if df.empty:
        print("[WARN] Input DataFrame is empty.")
        return df

    # Ensure datetime column is datetime and set as index
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.set_index('datetime')

    # Sort and drop any NaT values
    df = df.sort_index()
    df = df[~df.index.isnull()]

    ohlcv_resampled = df.resample(new_interval).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    ohlcv_resampled = ohlcv_resampled.dropna().reset_index()
    return ohlcv_resampled

def generate_signals(df: pd.DataFrame, predictions, threshold=0.001, mode="trend"):
    """
    Generate trading signals from predicted next close.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'close' column (time T close) and 'datetime'.
    predictions : np.ndarray
        Predictions for close_{t+1}, aligned with df.
    threshold : float
        Minimum relative change to trigger a trade.
    mode : str
        'direct' → compare prediction vs current close
        'trend'  → compare prediction vs previous prediction

    Returns
    -------
    pd.Series
        Signal series aligned with df.index (1=long, -1=short, 0=hold).
    """
    df = df.copy()
    df.set_index("datetime", inplace=True)
    df["predicted"] = predictions  # already aligned with df

    if mode == "direct":
        rel_change = (df["predicted"] - df["close"]) / df["close"]
    elif mode == "trend":
        rel_change = df["predicted"].pct_change()
    else:
        raise ValueError("mode must be 'direct' or 'trend'")

    df["signal"] = np.where(rel_change > threshold, 1,
                     np.where(rel_change < -threshold, -1, 0))
    df.reset_index(inplace=True)

    return df[["datetime","signal"]].fillna(0)

def engineer_features(df,config:ConfigParser,trial:optuna.Trial):         
    from learning_alpha_edge.signals.technical_indicators_signals.main_signals import apply_indicators
    scaler=MinMaxScaler()
    selected_indicators=[]
    timeperiods={}
    ind_config=config["technical_indicators"]
    for ind in ind_config:
        if config.getboolean("technical_indicators", ind):
            selected_indicators.append(ind)
            func=indicator_map[ind]
            params=inspect.signature(func).parameters
            if 'timeperiod' in params:
                timeperiods[ind]=trial.suggest_int(f'Window_size_{ind}',5,50)
    df=apply_indicators(df,selected_indicators,timeperiods)
        
    return df

        