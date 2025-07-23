# bybit_fetcher.py
import os
import time
import pandas as pd
from datetime import datetime
from pybit.unified_trading import HTTP

api_key = os.getenv("BYBIT_API_KEY")
api_secret = os.getenv("BYBIT_SECRET_KEY")
session = HTTP(api_key=api_key, api_secret=api_secret)
def date_to_milliseconds(dt):
    return int(dt.timestamp() * 1000)

def fetch_data(symbol, start_date, end_date) -> pd.DataFrame:
    print(f"[INFO] Fetching Bybit data for {symbol} from {start_date} to {end_date}")

    start_ts = date_to_milliseconds(start_date)
    end_ts = date_to_milliseconds(end_date)

    all_klines = []

    while start_ts < end_ts:
        response = session.get_kline(
        
            category="linear",
            symbol=symbol.upper(),
            interval="1",
            start=start_ts,
            end=end_date,
            limit=1000
         )

        klines = response.get("result", {}).get("list", [])
        if not klines:
            break
        # Sort the klines ASC, As bybit follows DSC(reverse) our loop expects latest date at the end 
        klines = sorted(klines, key=lambda x: x[0])
        all_klines.extend(klines)
        last_open_time = int(klines[-1][0])
        # print(f"[DEBUG] Retrieved {len(all_klines)} total rows so far. Last timestamp: {last_open_time}")
        start_ts = last_open_time + 60 * 1000

    df = pd.DataFrame(all_klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])

    df['open_time'] = pd.to_datetime(pd.to_numeric(df['open_time'], errors='coerce'), unit='ms',utc=True)
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    return df
if __name__=='__main__':
    fetch_data()
