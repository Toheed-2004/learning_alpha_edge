# main_bybit.py

import os
import sys
import datetime
import configparser
import pandas as pd

from learning_alpha_edge.utils.db_utils import (
    get_pg_engine,
    save_to_db,
    ensure_schema_exists
)
from learning_alpha_edge.data.bybit.bybit_fetcher import fetch_data
from learning_alpha_edge.utils.data_utils import preprocess_klines

# Load config
def load_config(path='config_bybit.ini'):
    config = configparser.ConfigParser()
    config.read(path)
    return config

if __name__ == '__main__':
    config = load_config()
    cfg = config['data']
    db_cfg = config['postgres']

    # Create PostgreSQL engine
    engine = get_pg_engine(
        user=db_cfg.get('user'),
        password=db_cfg.get('password'),
        host=db_cfg.get('host'),
        port=db_cfg.get('port'),
        dbname=db_cfg.get('dbname')
    )

    # Define schema for Bybit
    schema = 'bybit'
    ensure_schema_exists(engine, schema)

    # Parse config
    symbols = [s.strip().upper() + 'USDT' for s in cfg.get('symbols').split(',')]
    start_date = datetime.datetime.strptime(cfg.get('start_date'), "%Y-%m-%d")
    end_date = datetime.datetime.now() if cfg.get('end_date') == 'now' else datetime.datetime.strptime(cfg.get('end_date'), "%Y-%m-%d")
    interval = cfg.get('time_horizons')
    interpolate_method = cfg.get('interpolation_method')
    fill_zero_volume = cfg.get('fill_zero_volume')

    for symbol in symbols:
        print(f"[INFO] Fetching {symbol} from Bybit...")
        df = fetch_data(symbol, start_date, end_date)

        if df.empty:
            print(f"[WARN] No data for {symbol}, skipping.")
            continue

        # Prepare DataFrame
        df.rename(columns={'open_time': 'datetime'}, inplace=True)
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = preprocess_klines(df, interpolate_method, fill_zero_volume)

        base_symbol = symbol.replace('USDT', '').lower()
        table_name = f"{base_symbol}_{interval}"
        df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

        save_to_db(df, engine, schema, table_name)

        print(f"[INFO] {symbol} saved to {schema}.{table_name}")
