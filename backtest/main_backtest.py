import pandas as pd
from learning_alpha_edge.data.binance.main_binance import load_config
from learning_alpha_edge.utils.db_utils import get_pg_engine
import os


class Backtester:
    def __init__(self, signal_df:pd.DataFrame, data_df:pd.DataFrame, start_balance=1000, takeprofit=5, stoploss=3, fees=0.05):
        self.signal_df = signal_df.set_index("datetime")
        self.data_df = data_df.set_index("datetime")
        self.df = self.data_df.copy()
        self.df["signal"]=self.df["signal"].shift(1)
        self.df["signal"] = self.signal_df["signal"]
        # self.df["signal"] = self.df["signal"].ffill() # forward filling the NaNs with previous signal,untill new signal is encountered
        
        self.balance = start_balance
        self.takeprofit = takeprofit
        self.stoploss = stoploss
        self.fees = fees
        self.ledger = []

    def run_backtest(self):
        position = None
        entry_price = 0
        cumulative_pnl = 0

        for dt, row in self.df.iterrows():
            signal = row["signal"]
            open_price = row["open"]
            high = row["high"]
            low = row["low"]
            close_price = row["close"]

            predicted_direction = "Long" if signal == 1 else ("Short" if signal == -1 else None)

            fee_rate = self.fees / 100
            tp_pct = self.takeprofit / 100
            sl_pct = self.stoploss / 100

            # ---------------------- LONG POSITION ----------------------
            if position == "long":
                tp_price = entry_price * (1 + tp_pct)
                sl_price = entry_price * (1 - sl_pct)

                if high >= tp_price:
                    exit_price = high
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                    pnl_pct=pnl_pct-self.fees
                    profit = self.balance * (pnl_pct / 100)

                    # fee = profit * fee_rate  # Fee is charged on the trade profit, not full balance
                    # net_profit = profit - fee

                    self.balance += profit
                    cumulative_pnl += pnl_pct
                    self.ledger.append([dt, predicted_direction, "TP-Sell", entry_price, exit_price, round(self.balance, 2), round(pnl_pct, 2), round(cumulative_pnl, 2)])
                    position = None
                    entry_price = 0

                elif low <= sl_price:
                    exit_price = low
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                    pnl_pct=pnl_pct-self.fees
                    profit = self.balance * (pnl_pct / 100)

                    self.balance += profit
                    cumulative_pnl += pnl_pct
                    self.ledger.append([dt, predicted_direction, "SL-Sell", entry_price, exit_price, round(self.balance, 2), round(pnl_pct, 2), round(cumulative_pnl, 2)])
                    position = None
                    entry_price = 0

                elif signal == -1:
                    exit_price = open_price
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                    pnl_pct=pnl_pct-self.fees
                    profit = self.balance * (pnl_pct / 100)
                    self.balance += profit
                    cumulative_pnl += pnl_pct
                    self.ledger.append([dt, predicted_direction, "Signal-Sell-Direction-Change", entry_price, exit_price, round(self.balance, 2), round(pnl_pct, 2), round(cumulative_pnl, 2)])
                    # Open new short
                    entry_price = open_price
                    fee = self.balance * fee_rate
                    self.balance -= fee
                    pnl=0
                    pnl-=self.fees
                    cumulative_pnl=cumulative_pnl+pnl   
                    self.ledger.append([dt, predicted_direction, "Open-Short", entry_price, None, round(self.balance, 2), pnl, round(cumulative_pnl , 2)])
                    position = "short"

            # ---------------------- SHORT POSITION ----------------------
            elif position == "short":
                tp_price = entry_price * (1 - tp_pct)
                sl_price = entry_price * (1 + sl_pct)

                if low <= tp_price:
                    exit_price = low
                    pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                    pnl_pct=pnl_pct-self.fees
                    profit = self.balance * (pnl_pct / 100)
                    self.balance += profit
                    cumulative_pnl += pnl_pct
                    self.ledger.append([dt, predicted_direction, "TP-Buy", entry_price, exit_price, round(self.balance, 2), round(pnl_pct, 2), round(cumulative_pnl, 2)])
                    position = None
                    entry_price = 0

                elif high >= sl_price:
                    exit_price = high
                    pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                    pnl_pct=pnl_pct-self.fees
                    profit = self.balance * (pnl_pct / 100)
                    self.balance += profit
                    cumulative_pnl += pnl_pct
                    self.ledger.append([dt, predicted_direction, "SL-Buy", entry_price, exit_price, round(self.balance, 2), round(pnl_pct, 2), round(cumulative_pnl, 2)])
                    position = None
                    entry_price = 0

                elif signal == 1:
                    exit_price = open_price
                    pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                    pnl_pct=pnl_pct-self.fees
                    profit = self.balance * (pnl_pct / 100)
                    self.balance += profit
                    cumulative_pnl += pnl_pct
                    self.ledger.append([dt, predicted_direction, "Signal-Buy-Direction-Change", entry_price, exit_price, round(self.balance, 2), round(pnl_pct, 2), round(cumulative_pnl, 2)])
                    # Open new long
                    entry_price = open_price
                    fee = self.balance * fee_rate
                    self.balance -= fee
                    pnl=0
                    pnl-=self.fees
                    cumulative_pnl=cumulative_pnl+pnl
                    self.ledger.append([dt, predicted_direction, "Open-Long", entry_price, None, round(self.balance, 2), pnl, round(cumulative_pnl , 2)])
                    position = "long"

            # ---------------------- NO POSITION (Flat) ----------------------
            else:
                if signal == 1:
                    entry_price = open_price
                    fee = self.balance * fee_rate
                    self.balance -= fee
                    pnl=0
                    pnl-=self.fees
                    cumulative_pnl=cumulative_pnl+pnl
                    self.ledger.append([dt, predicted_direction, "Open-Long", entry_price, None, round(self.balance, 2), pnl, round(cumulative_pnl , 2)])
                    position = "long"
                elif signal == -1:
                    entry_price = open_price
                    fee = self.balance * fee_rate
                    self.balance -= fee
                    pnl=0
                    pnl-=self.fees
                    cumulative_pnl=cumulative_pnl+pnl
              
                    self.ledger.append([dt, predicted_direction, "Open-Short", entry_price, None, round(self.balance, 2), pnl, round(cumulative_pnl , 2)])
                    position = "short"

        return pd.DataFrame(self.ledger, columns=["datetime", "Predicted Direction", "Action", "Buy Price", "Sell Price", "Balance", "PnL", "PnLSum"])
    
config_path=os.path.dirname(os.path.abspath(__file__))
config_path=os.path.join(config_path,"backtest_config.ini")
config = load_config(config_path)
db_cfg = config['postgres']
engine = get_pg_engine(
            user=db_cfg.get('user'),
            password=db_cfg.get('password'),
            host=db_cfg.get('host'),
            port=db_cfg.get('port'),
            dbname=db_cfg.get('dbname')
        )

if __name__ == "__main__":
       
        with engine.begin() as conn:
         signal_path=os.path.dirname(__file__)
         signal_path=os.path.join(signal_path,"signals.csv")
         signal_df = pd.read_csv(signal_path)

         data_df = pd.read_sql("""
                SELECT 
                datetime,
                open, high, low, close, volume
                FROM "bybit"."btc_1m"
                WHERE datetime >= '2025-01-01 00:00:00'
                AND datetime <= '2025-07-22 04:00:00'
                ORDER BY datetime ASC

            """, conn)
         



        backtester = Backtester(signal_df=signal_df, data_df=data_df, start_balance=1000)
        ledger=backtester.run_backtest()
        numeric_cols=["Buy Price", "Sell Price", "Balance", "PnL", "PnL Sum"]
        ledger[numeric_cols]=ledger[numeric_cols].round(2)
        ledger.to_csv("ledger.csv")


