from binance.client import Client
import pandas as pd
import datetime

# Binance API credentials
api_key = 'sw6SivDKTzEF9WiP3gpVgdmQeqU4wzWRvQ6vmo4u5FIbNHz76Gkc3mfmoBoCAbBR'#pending identification approval from binance
api_secret = 'TzFKoSFRpRL2WFoYE4KO6TqkeFerblHDckD6LBiGPo9MAVfZ0edc4lyLuQTZePFC'

client = Client(api_key, api_secret)

# Define the symbol for BTC/USDT pair
symbol = 'BTCUSDT'

# Define custom start and end time
start_time = datetime.datetime(2025, 1, 1, 0, 0, 0)
end_time = datetime.datetime.now()

# Fetch historical 1-minute candlestick data
klines = client.get_historical_klines(
    symbol=symbol,
    interval=Client.KLINE_INTERVAL_1MINUTE,
    start_str=str(start_time),
    end_str=str(end_time)
)

# Convert to DataFrame
df_M = pd.DataFrame(klines, columns=[
    'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
    'Close Time', 'Quote Asset Volume', 'Number of Trades',
    'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
])

# Convert appropriate columns to float
columns_to_convert = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Quote Asset Volume', 'Number of Trades',
    'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume'
]

for col in columns_to_convert:
    df_M[col] = df_M[col].astype(float)

# Save to CSV
df_M.to_csv('BTCUSDT_1min_data.csv', index=False)
