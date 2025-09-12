import pandas as pd
import os
from configparser import ConfigParser
from learning_alpha_edge.utils.db_utils import get_pg_engine
from pybit.unified_trading import HTTP




class Executor ():
    def __init__(self):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")
        self.config = ConfigParser()
        self.config.read(config_path)
        db_credentials = self.config["postgres"]
        self.engine = get_pg_engine(db_credentials["user"], db_credentials["password"], db_credentials["host"], db_credentials["port"], db_credentials["dbname"])
        self.strategies_df=self.load_strategies_from_db()
        self.strategies=[self.parse_strategy_row(strategy) for _,strategy in self.strategies_df.iterrows()]
        self.session=HTTP(api_key=os.getenv("BYBIT_TESTNET_KEY"),api_secret=os.getenv("BYBIT_TESTNET_SECRET"),testnet=True)



    def load_strategies_from_db(self):
        query="SELECT * FROM public.config_Strategies"
        return pd.read_sql(query,self.engine)

    def parse_strategy_row(self,strategy: pd.Series) -> dict:
        """
        Convert one strategy row into a clean dictionary format
        ready for execution.
        """
        # Enabled indicators = any boolean column that's True
        enabled_indicators = [
            col
            for col in strategy.index
            if not col.endswith("_timeperiod")   # ignore timeperiod cols
            and col not in ["name", "symbol", "time_horizon", "data_exchange"]  # ignore metadata
            and strategy[col]             # only keep if True
        ]

        # Timeperiods = non-null values from *_timeperiod cols
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

    def fetch_ohlcv(self):
        response = self.session.get_kline(
        
            category="linear",
            symbol="BTCUSDT",
            interval=240,
            limit=1000
        )
        response=response.get("result").get("list")
        df = pd.DataFrame(response, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'
    ])

        df['open_time'] = pd.to_datetime(pd.to_numeric(df['open_time'], errors='coerce'), unit='ms',utc=True)
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
        df.sort_values(by="open_time",ascending=True,inplace=True)
        df.rename(columns={"open_time":"datetime"},inplace=True)
        print("break")
        


















if __name__=="__main__":
    executor=Executor()
    executor.fetch_ohlcv()
    
    