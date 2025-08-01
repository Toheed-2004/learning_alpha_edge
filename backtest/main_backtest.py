import pandas as pd

class Backtester:
 def __init__(self, signal_df: pd.DataFrame, data_df: pd.DataFrame, start_balance=1000, takeprofit=0.05, stoploss=0.03, fees=0.05):
    self.signal_df = signal_df.copy()
    self.data_df = data_df.copy()

    # Make sure both are datetime and localized
    self.signal_df['datetime'] = pd.to_datetime(self.signal_df['datetime']).dt.tz_localize('UTC')
    self.data_df['datetime'] = pd.to_datetime(self.data_df['datetime']).dt.tz_localize('UTC')

    # Set index
    self.signal_df = self.signal_df.set_index("datetime")
    self.data_df = self.data_df.set_index("datetime")

    # Now join works fine
    self.df = self.data_df.join(self.signal_df[["signal"]], how="inner")

    self.balance = start_balance
    self.takeprofit = takeprofit
    self.stoploss = stoploss
    self.fees = fees
    self.ledger = []

 def run(self):
    position = 0  # 1 = long, -1 = short
    entry_price = None
    entry_time = None
    cumulative_pnl = 0
    self.ledger = []

    for i in range(len(self.df)):
        row = self.df.iloc[i]
        signal = row["signal"]
        price = row["close"]
        timestamp = row.name

        # EXIT logic first (always evaluate if we're in a trade)
        if position != 0:
            change = (price - entry_price) / entry_price if position == 1 else (entry_price - price) / entry_price
            tp_hit = change >= self.takeprofit
            sl_hit = change <= -self.stoploss
            exit_signal = signal != position and signal != 0
            exit_condition = tp_hit or sl_hit or exit_signal

            if exit_condition:
                raw_pnl = change
                net_pnl = raw_pnl - 2 * self.fees
                pnl_amount = self.balance * net_pnl
                self.balance += pnl_amount
                cumulative_pnl += pnl_amount

                self.ledger.append({
                    "datetime": timestamp,
                    "action": "sell" if position == 1 else "cover",
                    "buy_price": entry_price if position == 1 else price,
                    "sell_price": price if position == 1 else entry_price,
                    "balance": round(self.balance, 2),
                    "pnl": round(pnl_amount, 2),
                    "cumulative_pnl": round(cumulative_pnl, 2)
                })

                # Reset position
                position = 0
                entry_price = None
                entry_time = None

        # ENTRY logic (after potential exit)
        if position == 0 and signal != 0:
            position = signal
            entry_price = price
            entry_time = timestamp

            self.ledger.append({
                "datetime": timestamp,
                "action": "buy" if signal == 1 else "short",
                "buy_price": price if signal == 1 else None,
                "sell_price": None if signal == 1 else price,
                "balance": round(self.balance, 2),
                "pnl": None,
                "cumulative_pnl": round(cumulative_pnl, 2)
            })

    # Final exit if still in position
    if position != 0:
        final_row = self.df.iloc[-1]
        final_price = final_row["close"]
        final_time = final_row.name

        change = (final_price - entry_price) / entry_price if position == 1 else (entry_price - final_price) / entry_price
        raw_pnl = change
        net_pnl = raw_pnl - 2 * self.fees
        pnl_amount = self.balance * net_pnl
        self.balance += pnl_amount
        cumulative_pnl += pnl_amount

        self.ledger.append({
            "datetime": final_time,
            "action": "final_sell" if position == 1 else "final_cover",
            "buy_price": entry_price if position == 1 else final_price,
            "sell_price": final_price if position == 1 else entry_price,
            "balance": round(self.balance, 2),
            "pnl": round(pnl_amount, 2),
            "cumulative_pnl": round(cumulative_pnl, 2)
        })

    return pd.DataFrame(self.ledger)
