# main_binance.py
import os
import sys
import datetime
import configparser
import pandas as pd

from learning_alpha_edge.utils.db_utils import save_to_db,drop_table,drop_all_tables
from learning_alpha_edge.data.binance.binance_fetcher import fetch_data
from learning_alpha_edge.utils.data_utils import preprocess_klines,resample_ohlcv_data
from learning_alpha_edge.utils.data_utils import resample_ohlcv_data

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
db_path = os.path.join(BASE_DIR, 'db', 'market_data.db')
print(db_path)

def load_config(path='config_binance.ini'):
    config = configparser.ConfigParser()
    config.read(path)
    return config

if __name__ == '__main__':

    config = load_config()
    print(config.sections())
    cfg = config['data']

    symbols = [s.strip().upper() + 'USDT' for s in cfg.get('symbols').split(',')]
    start_date = datetime.datetime.strptime(cfg.get('start_date'), "%Y-%m-%d")
    end_date = datetime.datetime.now() if cfg.get('end_date') == 'now' else datetime.datetime.strptime(cfg.get('end_date'), "%Y-%m-%d")
    interval = cfg.get('time_horizons')
    interpolate_method = cfg.get('interpolation_method')
    fill_zero_volume = cfg.get('fill_zero_volume')

    for symbol in symbols:
        print(f"[INFO] Fetching {symbol}...")
        df= fetch_data(symbol, start_date, end_date)

        
        # Preprocess
        df = preprocess_klines(df, interpolate_method, fill_zero_volume)
        base_symbol = symbol.replace('USDT', '').lower()
        table_name = f"binance_{base_symbol}_1min"
        print(symbol.lower())
        save_to_db(df, db_path, table_name)
        