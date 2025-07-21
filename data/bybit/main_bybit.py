import os
import sys
import datetime
import configparser
import pandas as pd

from learning_alpha_edge.utils.db_utils import save_to_db
from learning_alpha_edge.data.bybit.bybit_fetcher import fetch_data
from learning_alpha_edge.utils.data_utils import preprocess_klines, resample_ohlcv_data

# Base directory and DB path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
db_path = os.path.join(BASE_DIR, 'db', 'market_data.db')
print("[INFO] DB Path:", db_path)

def load_config(path='config_bybit.ini'):
    config = configparser.ConfigParser()
    config.read(path)
    return config

if __name__ == '__main__':
    config = load_config()
    cfg = config['data']

    symbols = [s.strip().upper() + 'USDT' for s in cfg.get('symbols').split(',')]
    start_date = datetime.datetime.strptime(cfg.get('start_date'), "%Y-%m-%d")
    end_date = datetime.datetime.now() if cfg.get('end_date') == 'now' else datetime.datetime.strptime(cfg.get('end_date'), "%Y-%m-%d")
    interval = cfg.get('time_horizons')
    interpolate_method = cfg.get('interpolation_method')
    fill_zero_volume = cfg.get('fill_zero_volume')

    for symbol in symbols:
        print(f"[INFO] Fetching {symbol} from Bybit...")
        df = fetch_data(symbol, start_date, end_date)
        
        print(df['open_time'].head())
        print(df['open_time'].dtype)
        # df['open_time'] = pd.to_datetime(pd.to_numeric(df['open_time'], errors='coerce'), unit='ms')
        print("[DEBUG] Columns before rename:", df.columns.tolist())

        df.rename(columns={'open_time': 'datetime'}, inplace=True)
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # print(df.shape)
        # print(df.head())   

        df = preprocess_klines(df, interpolate_method, fill_zero_volume)

        base_symbol = symbol.replace('USDT', '').lower()
        table_name = f"bybit_{base_symbol}_1min"
        

        save_to_db(df, db_path, table_name)

        print(f"[INFO] {symbol} saved to {table_name}")
