# binance_fetcher.py
import pandas as pd
import os
import time
import datetime
import psycopg2
from binance.client import Client


# Set up client (assumes env vars are set)
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
if not api_key or not api_secret:
    raise EnvironmentError("Missing Binance API credentials.")
client = Client(api_key, api_secret)

def date_to_milliseconds(dt:datetime.datetime):
    return int(dt.timestamp() * 1000)

def fetch_data(symbol, start_date, end_date,interval="1m"):
    """Fetch historical klines for a single symbol"""
    print(f"[INFO]Fetching Data from {start_date} TO {end_date}")
    all_klines = []
    limit = 1000
    start_ts = date_to_milliseconds(start_date)
    end_ts = date_to_milliseconds(end_date)

    while start_ts < end_ts:
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
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
    # Drop the last candle to avoid possibly incomplete data
    if all_klines:
        all_klines = all_klines[:-1]

    # Convert to DataFrame
    df = pd.DataFrame(all_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms',utc=True )
    df.rename(columns={'open_time': 'datetime'}, inplace=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    return df
if __name__=="__main__":
     start_date=datetime.datetime.strptime(('2025-07-01'), "%Y-%m-%d")
     end_date=datetime.datetime.strptime('2025-07-31', "%Y-%m-%d")
     df=fetch_data('BTCUSDT',start_date,end_date)
     df.to_csv('testoutput.csv',index=False)

