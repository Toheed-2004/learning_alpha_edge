import pandas as pd
import sqlite3


def preprocess_klines(df, interpolate_method='linear', fill_zero_volume='ffill'):
    """
    Applies interpolation and volume cleaning on already-processed kline DataFrame.
    Assumes datetime conversion and numeric casting are already done.
    """
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
    df.dropna(inplace=True)# setting inplace=true does not return a new data-frame and replaces the original one/caller data-frame

    # Optional: Round again if needed
    df[numeric_cols] = df[numeric_cols].round(2)
    df[numeric_cols] = df[numeric_cols].astype('float64')

    return df

def resample_ohlcv_data(db_path, table_name, new_interval):#ohlcv=open,high,low,close,volume
    """
    Resample OHLCV data from SQLite DB to a higher interval.
    
    Args:
        db_path (str): Path to the SQLite database.
        table_name (str): Table to fetch 1m data from.
        new_interval (str): Pandas-compatible resample interval (e.g., '4min', '15min', '1H').

    Returns:
        pd.DataFrame: Resampled OHLCV data.
    """
    # Load data
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(f"SELECT * FROM '{table_name}'", conn, parse_dates=['datetime'])

    if df.empty:
        print(f"[WARN] Table '{table_name}' is empty.")
        return df

    # Set datetime as index for resampling
    df.set_index('datetime', inplace=True)

    # Resample using appropriate aggregation
    ohlcv_resampled = df.resample(new_interval).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    # Drop NaN rows (may happen if interval causes gaps)
    ohlcv_resampled.dropna(inplace=True)

    ohlcv_resampled.reset_index(inplace=True)
    return ohlcv_resampled
