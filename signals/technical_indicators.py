import talib
import pandas as pd

class TechnicalIndicatorApplier:
    def __init__(self, enabled_indicators):
        self.enabled_indicators = enabled_indicators
        self.indicator_map = {
            "SMA": self._apply_sma,
            "RSI": self._apply_rsi,
            "MACD": self._apply_macd,
            "Bollinger_BANDS": self._apply_bollinger_bands,
            "DOJI_PATTERN": self._apply_doji_pattern,
            # Add more mappings here
        }

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
     df = df.copy()
     indicators_df = pd.DataFrame(index=df.index)

     for name in self.enabled_indicators:
        func = self.indicator_map.get(name.upper())
        if func:
            temp_df = df.copy()  # <-- important: isolate each indicator
            result_df = func(temp_df)
            new_cols = [col for col in result_df.columns if col not in df.columns]
            if new_cols:
                indicators_df = pd.concat([indicators_df, result_df[new_cols]], axis=1)
        else:
            print(f"[WARN] No handler for indicator '{name}'")

        return indicators_df

    def _apply_sma(self, df):
        df["SMA_20"] = talib.SMA(df["close"], timeperiod=20)
        return df

    def _apply_rsi(self, df):
        df["RSI_14"] = talib.RSI(df["close"], timeperiod=14)
        return df

    def _apply_macd(self, df):
        macd, signal, hist = talib.MACD(df["close"])
        df["MACD"] = macd
        df["MACD_signal"] = signal
        df["MACD_hist"] = hist
        return df

    def _apply_bollinger_bands(self, df):
        upper, middle, lower = talib.BBANDS(df["close"], timeperiod=20)
        df["BB_upper"] = upper
        df["BB_middle"] = middle
        df["BB_lower"] = lower
        return df

    def _apply_doji_pattern(self, df):
        # TA-Lib candlestick functions return +100, -100, or 0
        df["Doji"] = talib.CDLDOJI(df["open"], df["high"], df["low"], df["close"])
        return df
