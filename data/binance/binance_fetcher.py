import os
import time
import datetime
import pandas as pd
from binance.client import Client
print("starting point")

class binance_fetcher:
    def __init__(self, config_section):        
        self.exchange = config_section.get('exchange')
        self.symbols = [s.strip().upper() + 'USDT' for s in config_section.get('symbols', '').split(',')]
        self.start_date = config_section.get('start_date')
        self.end_date = config_section.get('end_date')
        self.time_horizon = config_section.get('time_horizons')
        self.filled_missing_method = config_section.get('filled_missing_method')
        self.interpolation_method = config_section.get('interpolation_method')
        self.fill_zero_volume = config_section.get('fill_zero_volume')
        self.retries = int(config_section.get('retries', 5))

        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        if not self.api_key or not self.api_secret:
            raise EnvironmentError("Missing Binance API credentials in environment variables")

        self.client = Client(self.api_key, self.api_secret)

        # Convert dates
        self.start_time = datetime.datetime.strptime(self.start_date, "%Y-%m-%d")
        self.end_time = datetime.datetime.now() if self.end_date == 'now' else datetime.datetime.strptime(self.end_date, "%Y-%m-%d")

    def date_to_milliseconds(self, dt):
        return int(dt.timestamp() * 1000)

    def fetch_all_klines(self, symbol):
        all_klines = []
        limit = 1000
        start_ts = self.date_to_milliseconds(self.start_time)
        end_ts = self.date_to_milliseconds(self.end_time)

        print(f"[INFO] Fetching {symbol} from {self.start_time} to {self.end_time}")

        while start_ts < end_ts:
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=self.time_horizon,
                start_str=start_ts,
                end_str=end_ts,
                limit=limit
            )

            if not klines:
                break

            all_klines.extend(klines)
            start_ts = klines[-1][0] + 60_000
            time.sleep(0.5)

        return all_klines

    def save_all_symbols(self):
        for symbol in self.symbols:
            data = self.fetch_all_klines(symbol)
            self.save_to_csv(data, symbol)

    def save_to_csv(self, klines, symbol):
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.rename(columns={'open_time': 'datetime'}, inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        df = df[['datetime', 'open', 'low', 'high', 'close', 'volume']]
        df.to_csv(f"{symbol.lower()}_{self.time_horizon}_data.csv", index=False)
        print(f"[INFO] Saved {symbol} data.")
