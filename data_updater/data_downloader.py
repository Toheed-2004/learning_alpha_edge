import pandas as pd
from datetime import datetime
from sqlalchemy import text
from learning_alpha_edge.data.binance import binance_fetcher
from learning_alpha_edge.data.bybit import bybit_fetcher
from learning_alpha_edge.utils import data_utils
import sqlalchemy
from sqlalchemy import Engine

class Data_Downloader:
    def __init__(self, symbol: str, exchange: str, resample_to: str):
        print('[INFO] In PostgreSQL Data_Downloader constructor')
        self.symbol = symbol
        self.base_symbol = self.symbol.replace("USDT", "").lower()
        self.exchange = exchange.lower()
        self.resample_to = resample_to
        # self.table_name = f"{self.base_symbol}_1m"
        # self.schema = schema
        # self.engine = engine
        self.full_df = None
        self.resampled_df = None
        self._update()

    def _update(self):
        # Get latest datetime from table
        # with self.engine.connect() as conn:
        #     result = conn.execute(
        #     text(f'SELECT MAX("datetime") FROM "{self.exchange}"."{self.table_name}"')
        #     ).fetchone()

            

        new_start_date = pd.Timestamp("2024-01-01")
        new_end_date = pd.Timestamp("2025-01-01")

        # Fetch new data
        if self.exchange == 'binance':
            df = binance_fetcher.fetch_data(self.symbol, new_start_date, new_end_date)
        else:
            df = bybit_fetcher.fetch_data(self.symbol, new_start_date, new_end_date)
            df.rename(columns={'open_time': 'datetime'}, inplace=True)
            df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if df.empty:
            print(f"[INFO] No new data for {self.symbol}")
            return

        # Preprocess
        df = data_utils.preprocess_klines(df, interpolate_method='linear', fill_zero_volume='ffill')
        # df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

        # Save to PostgreSQL
        # df.to_sql(
        #     self.table_name,
        #     self.engine,
        #     schema=self.schema,
        #     if_exists='append',
        #     index=False,
        #     method='multi',
        #     chunksize=1000
        # )

        # Read full and resampled data
        # query = f'SELECT * FROM "{self.schema}"."{self.table_name}" ORDER BY datetime ASC'
        self.full_df = df
        self.full_df['datetime'] = pd.to_datetime(self.full_df['datetime'])

        self.resampled_df = data_utils.resample_ohlcv_data(self.full_df, self.resample_to)

    def get_data(self):
        return self.full_df, self.resampled_df


if __name__ == '__main__':
    from sqlalchemy import create_engine
    # engine = create_engine("postgresql+psycopg2://postgres:Afridi11@localhost:5432/db")
    downloader = Data_Downloader('BTCUSDT', 'binance', '3min')
