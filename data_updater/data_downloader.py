import sqlite3
import pandas as pd
import os
from datetime import datetime
from learning_alpha_edge.data.binance import binance_fetcher
from learning_alpha_edge.utils import data_utils
from learning_alpha_edge.data.bybit import bybit_fetcher

class Data_Downloader:
    def __init__(self, symbol: str, exchange: str, resample_to: str):
        print('[INFO]IN update class constructor')  
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        self.db_path=os.path.join(BASE_DIR,'db','market_data.db')
        print(self.db_path)
        self.symbol = f"{symbol.upper()}USDT"
        self.base_symbol=self.symbol.replace("USDT","").lower()
        self.exchange = exchange.lower()
        self.resample_to = resample_to
        self.table_name = f"{self.exchange}_{self.base_symbol}_1min"
        self.full_df=None 
        self.resampled_df=None
        self._update()
        klines=None
    
    def _update(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Get latest datetime from table
        cursor.execute(f"SELECT MAX(datetime) FROM {self.table_name}")
        result = cursor.fetchone()
        conn.close()
        new_start_date = pd.to_datetime(result[0]) # New start date is set to the current maximum date.
        # new_end_date = datetime.now() 
        new_end_date=datetime.now()
               
        # Fetch new data
        if(self.exchange=='binance'):
            binance=True
            klines = binance_fetcher.fetch_data(self.symbol,new_start_date,new_end_date)
            if not klines:
             print(f"[INFO] No new data for {self.symbol}")
             return
            # Convert to DataFrame
            df_binance = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            df_binance['open_time'] = pd.to_datetime(df_binance['open_time'], unit='ms')
            df_binance.rename(columns={'open_time': 'datetime'}, inplace=True)
            df_binance = df_binance[['datetime', 'open', 'low', 'high', 'close', 'volume']]

            for col in ['open', 'high', 'low', 'close', 'volume']:
               df_binance[col] = pd.to_numeric(df_binance[col], errors='coerce')

        else:
            binance=False
            df_bybit=bybit_fetcher.fetch_data(self.symbol,new_start_date,new_end_date)
            df_bybit.rename(columns={'open_time': 'datetime'}, inplace=True)
            df_bybit = df_bybit[['datetime', 'open', 'high', 'low', 'close', 'volume']]

            for col in ['open', 'high', 'low', 'close', 'volume']:
               df_bybit[col] = pd.to_numeric(df_bybit[col], errors='coerce')

        
        # Preprocess
        interpolate_method = 'linear'
        fill_zero_volume = 'ffill'
        
        if binance:
             df_binance = data_utils.preprocess_klines(df_binance, interpolate_method, fill_zero_volume)
        else:
             df_bybit = data_utils.preprocess_klines(df_bybit, interpolate_method, fill_zero_volume)


       
        # Save new data to DB
        conn = sqlite3.connect(self.db_path)
        
        df_binance.to_sql(self.table_name, conn, if_exists='append', index=False) if binance else df_bybit.to_sql(self.table_name, conn, if_exists='append', index=False)
        query = f"SELECT * FROM {self.table_name} ORDER BY datetime ASC"
        self.full_df = pd.read_sql(query, conn, parse_dates=['datetime'])
        self.resampled_df=data_utils.resample_ohlcv_data(self.db_path,self.table_name,self.resample_to)
        conn.close()
    def get_data(self):
        return self.full_df, self.resampled_df

        

        

if __name__ == '__main__':
    updater = Data_Downloader( 'btc', 'bybit', '3min') # Also updates the corresponding db table to datetime.Now()
    