import sqlite3
import pandas as pd
from datetime import datetime
from learning_alpha_edge.data.binance import binance_fetcher
from learning_alpha_edge.utils import data_utils

class UpdateData:
    def __init__(self, symbol: str, exchange: str, resample_to: str):
        print('[INFO]IN update class constructor')  
        self.db_path='D:\learning_alpha_edge\db\market_data.db'
        print(self.db_path)
        self.symbol = f"{symbol.upper()}USDT"
        self.exchange = exchange.lower()
        self.resample_to = resample_to
        self.table_name = f"{self.symbol.lower()}_{self.exchange}_1min"
        self._update()
    
    def _update(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Get latest datetime from table
        cursor.execute(f"SELECT MAX(datetime) FROM {self.table_name}")
        result = cursor.fetchone()
        conn.close()
        new_start_date = pd.to_datetime(result[0]) # New start date is set to the current maximum date.
        # new_end_date = datetime.now() 
        new_end_date=datetime.strptime('2024-5-23',"%Y-%m-%d")
               
        # Fetch new data
        klines = binance_fetcher.fetch_data(self.symbol,new_start_date,new_end_date)

        if not klines:
            print(f"[INFO] No new data for {self.symbol}")
            return

        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.rename(columns={'open_time': 'datetime'}, inplace=True)
        df = df[['datetime', 'open', 'low', 'high', 'close', 'volume']]

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Preprocess
        interpolate_method = 'linear'
        fill_zero_volume = 'ffill'

        df = data_utils.preprocess_klines(df, interpolate_method, fill_zero_volume)

        # Save new data to DB
        conn = sqlite3.connect(self.db_path)
        df.to_sql(self.table_name, conn, if_exists='append', index=False)
        query = f"SELECT * FROM {self.table_name} ORDER BY datetime ASC"
        self.full_df = pd.read_sql(query, conn, parse_dates=['datetime'])
        self.resampled_df=data_utils.resample_ohlcv_data(self.db_path,self.table_name,self.resample_to)
        conn.close()

        

if __name__ == '__main__':
    updater = UpdateData( 'eth', 'binance', '3min') # Also updates the corresponding db table to datetime.Now()
    # Access data like this:
    print(updater.full_df.head())
    print(updater.resampled_df.head())
