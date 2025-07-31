import pandas as pd
from sqlalchemy import create_engine

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
