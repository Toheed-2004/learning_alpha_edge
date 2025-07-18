# main_binance.py
import os
import sys
sys.path.append(os.path.abspath('D:\LEARNING_ALPHA_EDGE'))#at the top to append root path in system-path before importing
import datetime
import configparser
import pandas as pd
from utils.db_utils import save_to_db,drop_table,drop_all_tables
from learning_alpha_edge.data.binance.binance_fetcher import fetch_data
from utils.data_utils import preprocess_klines,resample_ohlcv_data

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
        klines = fetch_data(symbol, start_date, end_date)

        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.rename(columns={'open_time': 'datetime'}, inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df[['datetime', 'open', 'low', 'high', 'close', 'volume']]

        # Preprocess
        df = preprocess_klines(df, interpolate_method, fill_zero_volume)

        table_name = f"{symbol.lower()}_binance_1min"
        print(symbol.lower())
        save_to_db(df, db_path, table_name)
        