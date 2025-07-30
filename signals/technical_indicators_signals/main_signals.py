import pandas as pd
import numpy as np
import random
import datetime
import os
import inspect
import hashlib
from learning_alpha_edge.utils.db_utils import *
from learning_alpha_edge.data.binance.main_binance import load_config
from learning_alpha_edge.data.binance import binance_fetcher
from learning_alpha_edge.data.bybit import bybit_fetcher
from learning_alpha_edge.signals.technical_indicators_signals.signals_generator import generate_signals
from learning_alpha_edge.technical_indicators.ti  import * 

# Randomly enable indicators (True/False)
def randomly_select_indicators(indicator_map):
    return [name for name in indicator_map if random.choice([True, False])]


# Apply only selected indicators
def apply_indicators( df,randomly_selected_indicators) -> pd.DataFrame:
        df = df.copy()

        # Start with just OHLCV once
        indicators_df = df[["open", "high", "low", "close", "volume"]].copy()
        existing_cols = set(indicators_df.columns)

        for name in randomly_selected_indicators:
            func = indicator_map.get(name)
            if func:
                sig = inspect.signature(func)
                params = sig.parameters

                # Call the function with or without timeperiod
                if len(params) >= 2:
                    result_df = func(df, 20)
                else:
                    result_df = func(df)

                # Find only the new columns added by the indicator function
                new_cols = set(result_df.columns) - existing_cols
                new_data = result_df[list(new_cols)]

                # Merge new indicator columns into indicators_df
                indicators_df = pd.concat([indicators_df, new_data], axis=1)

                # Update existing_cols so we don't keep collecting the same again
                existing_cols.update(new_cols)
            else:
                print(f"[WARN] No handler for indicator '{name}'")

        return indicators_df

def majority_vote(signals_df):
    signal_columns = signals_df.drop(columns=["datetime", "open", "high", "low", "close", "volume"])

    # Count how many columns have each signal per row
    long_count = (signal_columns == 1).sum(axis=1)
    short_count = (signal_columns == -1).sum(axis=1)
    hold_count = (signal_columns == 0).sum(axis=1)

    # Assign signal based on majority
    majority_signal = pd.Series(0, index=signals_df.index)  # Default to hold
    majority_signal[long_count > short_count] = 1
    majority_signal[short_count > long_count] = -1

    # Add majority signal column
    signals_df["final_signal"] = majority_signal

    # Keep only datetime, OHLCV, and final_signal
    return signals_df[["datetime", "open", "high", "low", "close", "volume", "final_signal"]]

def sanitize_table_name(name):
    return name.lower().replace("-", "_").replace(".", "_").replace(" ", "_")

def save_strategy_to_db(engine, strategy_name, symbol, time_horizon, data_exchange, selected_indicators, all_indicators):
    row = {
        "name": strategy_name,
        "symbol": symbol,
        "time_horizon": time_horizon,
        "data_exchange": data_exchange,
    }

    for ind in all_indicators:
        row[f'use_{ind}'] = ind in selected_indicators

    df = pd.DataFrame([row])

    save_to_db(df, engine, schema="public", table_name="config_strtg")


def save_signals(engine:Engine, strategy_name, signal_df:pd.DataFrame):
    # Add strategy name column
    signal_df["name"] = strategy_name
    save_to_db(signal_df, engine, schema="signals", table_name=strategy_name)



# Run the pipeline
if __name__ == "__main__":
    config_path=os.path.dirname(os.path.abspath(__file__))
    config_path=os.path.join(config_path,"signals_config.ini")
    config = load_config(config_path)
    print(config.sections())
    cfg = config['data']
    db_cfg = config['postgres']

    exchanges = [ex.strip().lower() for ex in cfg.get('exchanges').split(',')]
    symbols_base = [s.strip().upper() for s in cfg.get('symbols').split(',')]
    start_date = datetime.datetime.strptime(cfg.get('start_date'), "%Y-%m-%d")
    end_date = datetime.datetime.now() if cfg.get('end_date') == 'now' else datetime.datetime.strptime(cfg.get('end_date'), "%Y-%m-%d")
    interval = cfg.get('time_horizons', '1min')

    all_indicators = list(indicator_map.keys())

    # Create PostgreSQL engine
    engine = get_pg_engine(
        user=db_cfg.get('user'),
        password=db_cfg.get('password'),
        host=db_cfg.get('host'),
        port=db_cfg.get('port'),
        dbname=db_cfg.get('dbname')
    )

    
    for exchange in exchanges:
        
        

        for base_symbol in symbols_base:
            symbol = base_symbol + "USDT"

            # Fetch data
            if(exchange=="binance"):
                df = binance_fetcher.fetch_data(symbol, start_date, end_date, interval=interval)
            else:
                df=bybit_fetcher.fetch_data(symbol,start_date,end_date,interval=interval)
                continue

            for i in range(20):
                selected_indicators = randomly_select_indicators(indicator_map)
                raw_string = f"{exchange}_{symbol}_{interval}_{i}_{datetime.datetime.now().isoformat()}"
                strategy_hash = hashlib.sha256(raw_string.encode()).hexdigest()[:8]  # shorten for readability
                strategy_name = f"{exchange}_{symbol.lower()}_strategy_{i}"               
                print(type(df))
                df_with_indicators = apply_indicators(df.copy(), selected_indicators)
                signal_df = generate_signals(df_with_indicators, selected_indicators)
                signal_df = majority_vote(signal_df)

                save_strategy_to_db(
                    engine, strategy_name, symbol, interval, exchange,
                    selected_indicators, all_indicators
                )

                save_signals(engine, strategy_name, signal_df)

    
