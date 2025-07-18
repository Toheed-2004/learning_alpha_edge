# binance_fetcher.py

import os
import time
import datetime
from binance.client import Client

# Set up client (assumes env vars are set)
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
if not api_key or not api_secret:
    raise EnvironmentError("Missing Binance API credentials.")
client = Client(api_key, api_secret)

def date_to_milliseconds(dt):
    return int(dt.timestamp() * 1000)

def fetch_data(symbol, start_date, end_date):
    """Fetch historical klines for a single symbol"""
    print(f"[INFO]Fetching Data from {start_date} TO {end_date}")
    all_klines = []
    limit = 1000
    start_ts = date_to_milliseconds(start_date)
    end_ts = date_to_milliseconds(end_date)

    while start_ts < end_ts:
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=Client.KLINE_INTERVAL_1MINUTE,
            start_str=start_ts,
            end_str=end_ts,
            limit=limit
        )

        if not klines:
            break

        all_klines.extend(klines)
        last_open_time = klines[-1][0]
        start_ts = last_open_time + 60_000
        time.sleep(0.5)

    return all_klines
