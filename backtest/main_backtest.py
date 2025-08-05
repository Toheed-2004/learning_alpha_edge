import pandas as pd
from learning_alpha_edge.data.binance.main_binance import load_config
from learning_alpha_edge.utils.db_utils import get_pg_engine
import os


class Backtester:
    def __init__(self, signal_df:pd.DataFrame, data_df:pd.DataFrame, start_balance=1000, takeprofit=0.05, stoploss=0.03, fees=0.0005):
        self.signal_df = signal_df.set_index("datetime")
        self.data_df = data_df.set_index("datetime")
        self.df = self.data_df.join(self.signal_df["signal"], how="inner").dropna()
        
        self.start_balance = start_balance
        self.takeprofit = takeprofit
        self.stoploss = stoploss
        self.fees = fees
        self.ledger = []

    def run_backtest(self):
        position = None
        entry_price = 0
        balance = self.start_balance
        cumulative_pnl = 0

        for dt, row in self.df.iterrows():
            signal = row["signal"]
            open_price = row["open"]
            high = row["high"]
            low = row["low"]
            close_price = row["close"]

            # Check if a position is already open
            if position == "long":
                tp_price = entry_price * (1 + self.takeprofit)
                sl_price = entry_price * (1 - self.stoploss)

                # Check for TP/SL hits
                if high >= tp_price:
                    sell_price = tp_price
                    pnl = (sell_price - entry_price) - (entry_price * self.fees) - (sell_price * self.fees)
                    balance += pnl
                    cumulative_pnl += pnl
                    self.ledger.append([dt, "TP-Sell", entry_price, sell_price, balance, pnl, cumulative_pnl])
                    position = None
                    entry_price = 0
                elif low <= sl_price:
                    sell_price = sl_price
                    pnl = (sell_price - entry_price) - (entry_price * self.fees) - (sell_price * self.fees)
                    balance += pnl
                    cumulative_pnl += pnl
                    self.ledger.append([dt, "SL-Sell", entry_price, sell_price, balance, pnl, cumulative_pnl])
                    position = None
                    entry_price = 0
                elif signal == -1:
                    sell_price = open_price
                    pnl = (sell_price - entry_price) - (entry_price * self.fees) - (sell_price * self.fees)
                    balance += pnl
                    cumulative_pnl += pnl
                    self.ledger.append([dt, "Signal-Sell", entry_price, sell_price, balance, pnl, cumulative_pnl])
                    position = "short"
                    entry_price = open_price
                    balance -= open_price * self.fees
                    self.ledger.append([dt, "Open-Short", None, open_price, balance, 0, cumulative_pnl])

            elif position == "short":
                tp_price = entry_price * (1 - self.takeprofit)
                sl_price = entry_price * (1 + self.stoploss)

                if low <= tp_price:
                    buy_price = tp_price
                    pnl = (entry_price - buy_price) - (entry_price * self.fees) - (buy_price * self.fees)
                    balance += pnl
                    cumulative_pnl += pnl
                    self.ledger.append([dt, "TP-Buy", buy_price, entry_price, balance, pnl, cumulative_pnl])
                    position = None
                    entry_price = 0
                elif high >= sl_price:
                    buy_price = sl_price
                    pnl = (entry_price - buy_price) - (entry_price * self.fees) - (buy_price * self.fees)
                    balance += pnl
                    cumulative_pnl += pnl
                    self.ledger.append([dt, "SL-Buy", buy_price, entry_price, balance, pnl, cumulative_pnl])
                    position = None
                    entry_price = 0
                elif signal == 1:
                    buy_price = open_price
                    pnl = (entry_price - buy_price) - (entry_price * self.fees) - (buy_price * self.fees)
                    balance += pnl
                    cumulative_pnl += pnl
                    self.ledger.append([dt, "Signal-Buy", buy_price, entry_price, balance, pnl, cumulative_pnl])
                    position = "long"
                    entry_price = open_price
                    balance -= open_price * self.fees
                    self.ledger.append([dt, "Open-Long", open_price, None, balance, 0, cumulative_pnl])

            else:
                # No position open
                if signal == 1:
                    position = "long"
                    entry_price = open_price
                    balance -= open_price * self.fees
                    self.ledger.append([dt, "Open-Long", open_price, None, balance, 0, cumulative_pnl])
                elif signal == -1:
                    position = "short"
                    entry_price = open_price
                    balance -= open_price * self.fees
                    self.ledger.append([dt, "Open-Short", None, open_price, balance, 0, cumulative_pnl])

        return pd.DataFrame(self.ledger, columns=["datetime", "Action", "Buy_price", "Sell_price", "balance", "pnl", "pnl_sum"])

if __name__ == "__main__":
        config_path=os.path.dirname(os.path.abspath(__file__))
        config_path=os.path.join(config_path,"backtest_config.ini")
        config = load_config(config_path)
        print(config.sections())
        db_cfg = config['postgres']
        engine = get_pg_engine(
            user=db_cfg.get('user'),
            password=db_cfg.get('password'),
            host=db_cfg.get('host'),
            port=db_cfg.get('port'),
            dbname=db_cfg.get('dbname')
        )
        with engine.begin() as conn:
         signal_df = pd.read_sql('SELECT * FROM "signals"."strategy_0_84c9f029"', conn)

         data_df = pd.read_sql('SELECT * FROM "binance"."btc_1m"', conn)
        backtester = Backtester(signal_df=signal_df, data_df=data_df, start_balance=1000)
        ledger=backtester.run_backtest()
        ledger.to_csv("ledger.csv")



