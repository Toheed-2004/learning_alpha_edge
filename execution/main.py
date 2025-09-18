import pandas as pd
import os
import schedule
import time
from configparser import ConfigParser
from learning_alpha_edge.utils.db_utils import get_pg_engine
from pybit.unified_trading import HTTP
from learning_alpha_edge.data_updater.data_downloader import Data_Downloader
from learning_alpha_edge.signals.technical_indicators_signals.main_signals import apply_indicators   

class Executor:
    INTERVAL_MAP = {"4H": 240}

    def __init__(self):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")
        self.config = ConfigParser()
        self.config.read(config_path)

        db_credentials = self.config["postgres"]
        self.engine = get_pg_engine(
            db_credentials["user"],
            db_credentials["password"],
            db_credentials["host"],
            db_credentials["port"],
            db_credentials["dbname"],
        )

        self.strategies_df = self.load_strategies_from_db()
        self.strategies = [self.parse_strategy_row(strategy) for _, strategy in self.strategies_df.iterrows()]
        self.session = HTTP(
            api_key=os.getenv("BYBIT_TESTNET_KEY"),
            api_secret=os.getenv("BYBIT_TESTNET_SECRET"),
            testnet=True,
        )
        self.data_downloader = Data_Downloader
        self.candles = pd.DataFrame()
        self.flag=False

    def load_strategies_from_db(self):
        query = "SELECT * FROM public.config_Strategies"
        return pd.read_sql(query, self.engine)

    def parse_strategy_row(self, strategy: pd.Series) -> dict:
        enabled_indicators = [
            col
            for col in strategy.index
            if not col.endswith("_timeperiod")
            and col not in ["name", "symbol", "time_horizon", "data_exchange"]
            and strategy[col]
        ]

        timeperiods = {
            col.replace("_timeperiod", ""): int(strategy[col])
            for col in strategy.index
            if col.endswith("_timeperiod") and pd.notna(strategy[col])
        }

        return {
            "name": strategy["name"],
            "symbol": strategy["symbol"],
            "time_horizon": strategy["time_horizon"],
            "exchange": strategy["data_exchange"],
            "enabled_indicators": enabled_indicators,
            "timeperiods": timeperiods,
        }

    def fetch_ohlcv(self, strategy_idx: int = 0):
        strat = self.strategies[strategy_idx]

        # For now force 1m candles to simulate streaming
        interval = "1min"
        symbol = strat["symbol"].upper() + "USDT"
        exchange = strat["exchange"] if strat["exchange"] == "bybit" else "bybit"
        
        data_downloader = self.data_downloader(symbol, exchange, interval,limit=max(strat["timeperiods"].values())) if not self.flag else self.data_downloader(symbol, exchange, interval)
        self.flag=True
        df = data_downloader.get_data()[0]
        
        # Append new candles
        self.candles = pd.concat([self.candles, df], ignore_index=True)

        #  Apply indicators each time new candles are fetched
        df_with_indicators = apply_indicators(
            self.candles,
            strat["enabled_indicators"],
            strat["timeperiods"]
        )

        print(f"Fetched new {interval} candle for {symbol} at {df['datetime'].iloc[-1]}")
        print(f"DataFrame now has shape: {df_with_indicators.shape}")

        return df_with_indicators

    def schedule_strategy(self, strategy_idx: int = 0):
        strat = self.strategies[strategy_idx]
        interval = strat["time_horizon"]

        # For now: run every 1 minute
        minutes = 1

        schedule.every(minutes).minutes.do(self.fetch_ohlcv, strategy_idx)
        print(f"Scheduled {strat['symbol']} every {minutes} minutes ({interval})")

        while True:
            schedule.run_pending()
            time.sleep(1)


if __name__ == "__main__":
    executor = Executor()
    executor.schedule_strategy()
