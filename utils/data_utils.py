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

def generate_signals(df, predictions, threshold=0.0015, use_dynamic_threshold=False):
    """
    Generate trading signals based on predicted price changes.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with datetime index and price data
    predictions : array-like
        Model predictions for the next period's close price (close@T+1)
    threshold : float
        Minimum expected return threshold to trigger a trade
    use_dynamic_threshold : bool
        Whether to use a dynamic threshold based on recent volatility
    
    Returns:
    --------
    pandas.DataFrame with datetime and signal columns
    """
    df = df.copy()
    
    # Ensure we have a datetime index
    if "datetime" in df.columns:
        df.set_index("datetime", inplace=True)
    
    # Add predictions to dataframe
    df["predicted"] = predictions
    
    # Calculate the expected return from current close to predicted next close
    # At time T, we know close@T and predict close@T+1
    expected_return = (df["predicted"] - df["close"]) / df["close"]
    
    # Optional: Dynamic threshold based on recent volatility
    if use_dynamic_threshold:
        volatility = df["close"].pct_change().rolling(window=20).std().fillna(0.001)
        dynamic_threshold = threshold * (1 + volatility * 10)  # Scale threshold with volatility
    else:
        dynamic_threshold = threshold
    
    # Generate signals
    df["signal"] = 0  # Default: no position
    
    # Long signal: expected return exceeds positive threshold
    df.loc[expected_return > dynamic_threshold, "signal"] = 1
    
    # Short signal: expected return below negative threshold
    df.loc[expected_return < -dynamic_threshold, "signal"] = -1
    
    # Reset index to return datetime as a column
    df.reset_index(inplace=True)
    
    return df[["datetime", "signal"]]
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
                timeperiods[ind]=trial.suggest_int(f'Window_size_{ind}',20,20)
    df=apply_indicators(df,selected_indicators,timeperiods)
        
    return df

        