import pandas as pd

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
        df['volume'] = df['volume'].replace(0, pd.NA).fillna(method='ffill')
    elif fill_zero_volume == 'bfill':
        df['volume'] = df['volume'].replace(0, pd.NA).fillna(method='bfill')
    elif fill_zero_volume == 'drop':
        df = df[df['volume'] != 0]

    # Drop any remaining NaNs
    df.dropna(inplace=True)

    # Optional: Round again if needed
    df[numeric_cols] = df[numeric_cols].round(2)

    return df
