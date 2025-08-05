# main_binance.py

import os
import sys
import datetime
import configparser
import pandas as pd
from learning_alpha_edge.data.binance.binance_fetcher import fetch_data
from learning_alpha_edge.utils.data_utils import preprocess_klines
from learning_alpha_edge.utils.db_utils import (
    get_pg_engine,
    save_to_db,
    ensure_schema_exists
)

# Load config
def load_config(path='config_binance.ini'):
    config = configparser.ConfigParser()
    config.read(path)
    return config

if __name__ == '__main__':
    config = load_config()
    cfg = config['data']
    db_cfg = config['postgres']

    # Extract DB credentials
    engine = get_pg_engine(
        user=db_cfg.get('user'),
        password=db_cfg.get('password'),
        host=db_cfg.get('host'),
        port=db_cfg.get('port'),
        dbname=db_cfg.get('dbname')
    )

    # Define schema for Binance
    schema = 'binance'
    # ensure_schema_exists(engine, schema)

    # Parse symbols and dates
    symbols = [s.strip().upper() + 'USDT' for s in cfg.get('symbols').split(',')]
    start_date = datetime.datetime.strptime(cfg.get('start_date'), "%Y-%m-%d")
    end_date = datetime.datetime.now() if cfg.get('end_date') == 'now' else datetime.datetime.strptime(cfg.get('end_date'), "%Y-%m-%d")
    interval = cfg.get('time_horizons')
    interpolate_method = cfg.get('interpolation_method')
    fill_zero_volume = cfg.get('fill_zero_volume')

    for symbol in symbols:
        print(f"[INFO] Fetching {symbol}...")
        df = fetch_data(symbol, start_date, end_date)

        if df.empty:
            print(f"[WARN] No data for {symbol}, skipping.")
            continue

        # Preprocess
        df = preprocess_klines(df, interpolate_method, fill_zero_volume)

        # Create table name and save
        base_symbol = symbol.replace('USDT', '').lower()
        table_name = f"{base_symbol}_{interval}"
        df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        save_to_db(df, engine, schema, table_name)
        
