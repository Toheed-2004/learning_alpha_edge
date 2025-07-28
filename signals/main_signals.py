import configparser
import os
import random
import pandas as pd
from learning_alpha_edge.signals.technical_indicators_applier import TechnicalIndicatorApplier
from learning_alpha_edge.signals.technical_indicators_applier import TechnicalIndicatorApplier
from learning_alpha_edge.signals.technical_indicators_applier import TechnicalIndicatorApplier
from learning_alpha_edge.signals.signals_generator import  generate_signals
import sqlite3

def apply_random_indicators(df: pd.DataFrame, N: int = 10):
    """
    Randomly selects N indicators from the full indicator_map and applies them to the DataFrame.

    Args:
        df (pd.DataFrame): OHLCV DataFrame with at least ['open', 'high', 'low', 'close', 'volume'].
        N (int): Number of random indicators to apply.

    Returns:
        pd.DataFrame: DataFrame with original OHLCV + applied indicators.
        list: List of randomly selected indicator names.
    """
    # Step 1: Get all available indicators
    full_indicator_map = TechnicalIndicatorApplier([]).indicator_map
    all_indicators = list(full_indicator_map.keys())

    if N > len(all_indicators):
        raise ValueError(f"Requested {N} indicators, but only {len(all_indicators)} are available.")

    # Step 2: Randomly select N
    selected_indicators = random.sample(all_indicators, N)

    # Step 3: Initialize applier with selected indicators
    applier = TechnicalIndicatorApplier(selected_indicators)

    # Step 4: Apply them to the dataframe
    df_with_indicators = applier.apply(df)

    return df_with_indicators, selected_indicators


if __name__ == "__main__":   
    config_path = os.path.join(os.path.dirname(__file__), "signals_config.ini")
    config = configparser.ConfigParser()
    config.read(config_path)

    # STEP 1: Read config
    db_path=os.path.abspath(os.path.join(os.path.dirname(__file__),"../"))
    db_path=os.path.join(db_path,"db","market_data.db")
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(f"SELECT * FROM binance_btc_1min", conn, parse_dates=["datetime"])
    N = 10
    result_df, selected_indicators = apply_random_indicators(df, 15)
    print(result_df.columns.tolist())
    print(selected_indicators)
    signals_df=generate_signals(result_df,selected_indicators)
    signals_df.columns.str.lower()
    signals_df.to_csv("signals.csv")




    
      

    