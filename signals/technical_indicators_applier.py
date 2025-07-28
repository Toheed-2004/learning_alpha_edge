import talib
import pandas as pd
import os
import inspect

class TechnicalIndicatorApplier:
    def __init__(self, enabled_indicators):
        self.randomly_enabled_indicators =enabled_indicators
        self.indicator_map = {
            # === Overlap Studies ===
            "sma": self._apply_sma,
            "ema": self._apply_ema,
            "wma": self._apply_wma,
            "dema": self._apply_dema,
            "tema": self._apply_tema,
            "trima": self._apply_trima,
            "kama": self._apply_kama,
            "mama": self._apply_mama,
            "mavp": self._apply_mavp,
            "t3": self._apply_t3,
            "sar": self._apply_sar,
            "sarext": self._apply_sarext,
            "midpoint": self._apply_midpoint,
            "midprice": self._apply_midprice,
            "ht_trendline": self._apply_ht_trendline,
            "bbands": self._apply_bollinger_bands,

            # === Momentum Indicators ===
            "adx": self._apply_adx,
            "adxr": self._apply_adxr,
            "apo": self._apply_apo,
            "aroon": self._apply_aroon,
            "aroonosc": self._apply_aroonosc,
            "bop": self._apply_bop,
            "cci": self._apply_cci,
            "cmo": self._apply_cmo,
            "dx": self._apply_dx,
            "macd": self._apply_macd,
            "macdext": self._apply_macdext,
            "macdfix": self._apply_macdfix,
            "mfi": self._apply_mfi,
            "minus_di": self._apply_minus_di,
            "minus_dm": self._apply_minus_dm,
            "mom": self._apply_mom,
            "plus_di": self._apply_plus_di,
            "plus_dm": self._apply_plus_dm,
            "ppo": self._apply_ppo,
            "roc": self._apply_roc,
            "rocp": self._apply_rocp,
            "rocr": self._apply_rocr,
            "rocr100": self._apply_rocr100,
            "rsi": self._apply_rsi,
            "stoch": self._apply_stoch,
            "stochf": self._apply_stochf,
            "stochrsi": self._apply_stochrsi,
            "trix": self._apply_trix,
            "ultosc": self._apply_ultosc,
            "willr": self._apply_willr,
            "ad":self._apply_ad,
            "adosc": lambda df: df.assign(ADOSC=talib.ADOSC(df["high"], df["low"], df["close"], df["volume"])),
            "obv": lambda df: df.assign(OBV=talib.OBV(df["close"], df["volume"])),

            
            "ht_dcperiod": self._apply_ht_dcperiod,
            "ht_dcphase": self._apply_ht_dcphase,
            "ht_phasor": self._apply_ht_phasor,
            "ht_sine": self._apply_ht_sine,
            "ht_trendmode": self._apply_ht_trendmode,
            # === Price Transform Indicators ===
            "avgprice": self._apply_avgprice,
            "medprice": self._apply_medprice,
            "typprice": self._apply_typprice,
            "wclprice": self._apply_wclprice,

            # === Volatility Indicators ===
            "atr": self._apply_atr,
            "natr": self._apply_natr,
            "trange": self._apply_trange,


            # === Pattern Recognition (example)
            "doji_pattern": self._apply_doji_pattern,
             # --- Pattern Recognition ---
            "cdl2crows": lambda df: df.assign(CDL2CROWS=talib.CDL2CROWS(df["open"], df["high"], df["low"], df["close"])),
            "cdl3blackcrows": lambda df: df.assign(CDL3BLACKCROWS=talib.CDL3BLACKCROWS(df["open"], df["high"], df["low"], df["close"])),
            "cdl3inside": lambda df: df.assign(CDL3INSIDE=talib.CDL3INSIDE(df["open"], df["high"], df["low"], df["close"])),
            "cdl3linestrike": lambda df: df.assign(CDL3LINESTRIKE=talib.CDL3LINESTRIKE(df["open"], df["high"], df["low"], df["close"])),
            "cdl3outside": lambda df: df.assign(CDL3OUTSIDE=talib.CDL3OUTSIDE(df["open"], df["high"], df["low"], df["close"])),
            "cdl3starsinsouth": lambda df: df.assign(CDL3STARSINSOUTH=talib.CDL3STARSINSOUTH(df["open"], df["high"], df["low"], df["close"])),
            "cdl3whitesoldiers": lambda df: df.assign(CDL3WHITESOLDIERS=talib.CDL3WHITESOLDIERS(df["open"], df["high"], df["low"], df["close"])),
            "cdlabandonedbaby": lambda df: df.assign(CDLABANDONEDBABY=talib.CDLABANDONEDBABY(df["open"], df["high"], df["low"], df["close"])),
            "cdladvanceblock": lambda df: df.assign(CDLADVANCEBLOCK=talib.CDLADVANCEBLOCK(df["open"], df["high"], df["low"], df["close"])),
            "cdlbelthold": lambda df: df.assign(CDLBELTHOLD=talib.CDLBELTHOLD(df["open"], df["high"], df["low"], df["close"])),
            "cdlbreakaway": lambda df: df.assign(CDLBREAKAWAY=talib.CDLBREAKAWAY(df["open"], df["high"], df["low"], df["close"])),
            "cdlclosingmarubozu": lambda df: df.assign(CDLCLOSINGMARUBOZU=talib.CDLCLOSINGMARUBOZU(df["open"], df["high"], df["low"], df["close"])),
            "cdlconcealbabyswall": lambda df: df.assign(CDLCONCEALBABYSWALL=talib.CDLCONCEALBABYSWALL(df["open"], df["high"], df["low"], df["close"])),
            "cdlcounterattack": lambda df: df.assign(CDLCOUNTERATTACK=talib.CDLCOUNTERATTACK(df["open"], df["high"], df["low"], df["close"])),
            "cdldarkcloudcover": lambda df: df.assign(CDLDARKCLOUDCOVER=talib.CDLDARKCLOUDCOVER(df["open"], df["high"], df["low"], df["close"])),
            "cdldoji": lambda df: df.assign(CDLDOJI=talib.CDLDOJI(df["open"], df["high"], df["low"], df["close"])),
            "cdldojistar": lambda df: df.assign(CDLDOJISTAR=talib.CDLDOJISTAR(df["open"], df["high"], df["low"], df["close"])),
            "cdldragonflydoji": lambda df: df.assign(CDLDRAGONFLYDOJI=talib.CDLDRAGONFLYDOJI(df["open"], df["high"], df["low"], df["close"])),
            "cdlengulfing": lambda df: df.assign(CDLENGULFING=talib.CDLENGULFING(df["open"], df["high"], df["low"], df["close"])),
            "cdleveningdojistar": lambda df: df.assign(CDLEVENINGDOJISTAR=talib.CDLEVENINGDOJISTAR(df["open"], df["high"], df["low"], df["close"])),
            "cdleveningstar": lambda df: df.assign(CDLEVENINGSTAR=talib.CDLEVENINGSTAR(df["open"], df["high"], df["low"], df["close"])),
            "cdlgapsidesidewhite": lambda df: df.assign(CDLGAPSIDESIDEWHITE=talib.CDLGAPSIDESIDEWHITE(df["open"], df["high"], df["low"], df["close"])),
            "cdlgravestonedoji": lambda df: df.assign(CDLGRAVESTONEDOJI=talib.CDLGRAVESTONEDOJI(df["open"], df["high"], df["low"], df["close"])),
            "cdlhammer": lambda df: df.assign(CDLHAMMER=talib.CDLHAMMER(df["open"], df["high"], df["low"], df["close"])),
            "cdlhangingman": lambda df: df.assign(CDLHANGINGMAN=talib.CDLHANGINGMAN(df["open"], df["high"], df["low"], df["close"])),
            "cdlharami": lambda df: df.assign(CDLHARAMI=talib.CDLHARAMI(df["open"], df["high"], df["low"], df["close"])),
            "cdlharamicross": lambda df: df.assign(CDLHARAMICROSS=talib.CDLHARAMICROSS(df["open"], df["high"], df["low"], df["close"])),
            "cdlhighwave": lambda df: df.assign(CDLHIGHWAVE=talib.CDLHIGHWAVE(df["open"], df["high"], df["low"], df["close"])),
            "cdlhikkake": lambda df: df.assign(CDLHIKKAKE=talib.CDLHIKKAKE(df["open"], df["high"], df["low"], df["close"])),
            "cdlhikkakemod": lambda df: df.assign(CDLHIKKAKEMOD=talib.CDLHIKKAKEMOD(df["open"], df["high"], df["low"], df["close"])),
            "cdlhomingpigeon": lambda df: df.assign(CDLHOMINGPIGEON=talib.CDLHOMINGPIGEON(df["open"], df["high"], df["low"], df["close"])),
            "cdlidentical3crows": lambda df: df.assign(CDLIDENTICAL3CROWS=talib.CDLIDENTICAL3CROWS(df["open"], df["high"], df["low"], df["close"])),
            "cdlinneck": lambda df: df.assign(CDLINNECK=talib.CDLINNECK(df["open"], df["high"], df["low"], df["close"])),
            "cdlinvertedhammer": lambda df: df.assign(CDLINVERTEDHAMMER=talib.CDLINVERTEDHAMMER(df["open"], df["high"], df["low"], df["close"])),
            "cdlkicking": lambda df: df.assign(CDLKICKING=talib.CDLKICKING(df["open"], df["high"], df["low"], df["close"])),
            "cdlkickingbylength": lambda df: df.assign(CDLKICKINGBYLENGTH=talib.CDLKICKINGBYLENGTH(df["open"], df["high"], df["low"], df["close"])),
            "cdlladderbottom": lambda df: df.assign(CDLLADDERBOTTOM=talib.CDLLADDERBOTTOM(df["open"], df["high"], df["low"], df["close"])),
            "cdllongleggeddoji": lambda df: df.assign(CDLLONGLEGGEDDOJI=talib.CDLLONGLEGGEDDOJI(df["open"], df["high"], df["low"], df["close"])),
            "cdllongline": lambda df: df.assign(CDLLONGLINE=talib.CDLLONGLINE(df["open"], df["high"], df["low"], df["close"])),
            "cdlmarubozu": lambda df: df.assign(CDLMARUBOZU=talib.CDLMARUBOZU(df["open"], df["high"], df["low"], df["close"])),
            "cdlmatchinglow": lambda df: df.assign(CDLMATCHINGLOW=talib.CDLMATCHINGLOW(df["open"], df["high"], df["low"], df["close"])),
            "cdlmathold": lambda df: df.assign(CDLMATHOLD=talib.CDLMATHOLD(df["open"], df["high"], df["low"], df["close"])),
            "cdlmorningdojistar": lambda df: df.assign(CDLMORNINGDOJISTAR=talib.CDLMORNINGDOJISTAR(df["open"], df["high"], df["low"], df["close"])),
            "cdlmorningstar": lambda df: df.assign(CDLMORNINGSTAR=talib.CDLMORNINGSTAR(df["open"], df["high"], df["low"], df["close"])),
            "cdlonneck": lambda df: df.assign(CDLONNECK=talib.CDLONNECK(df["open"], df["high"], df["low"], df["close"])),
            "cdlpiercing": lambda df: df.assign(CDLPIERCING=talib.CDLPIERCING(df["open"], df["high"], df["low"], df["close"])),
            "cdlrickshawman": lambda df: df.assign(CDLRICKSHAWMAN=talib.CDLRICKSHAWMAN(df["open"], df["high"], df["low"], df["close"])),
            "cdlrisefall3methods": lambda df: df.assign(CDLRISEFALL3METHODS=talib.CDLRISEFALL3METHODS(df["open"], df["high"], df["low"], df["close"])),
            "cdlseparatinglines": lambda df: df.assign(CDLSEPARATINGLINES=talib.CDLSEPARATINGLINES(df["open"], df["high"], df["low"], df["close"])),
            "cdlshootingstar": lambda df: df.assign(CDLSHOOTINGSTAR=talib.CDLSHOOTINGSTAR(df["open"], df["high"], df["low"], df["close"])),
            "cdlshortline": lambda df: df.assign(CDLSHORTLINE=talib.CDLSHORTLINE(df["open"], df["high"], df["low"], df["close"])),
            "cdlspinningtop": lambda df: df.assign(CDLSPINNINGTOP=talib.CDLSPINNINGTOP(df["open"], df["high"], df["low"], df["close"])),
            "cdlstalledpattern": lambda df: df.assign(CDLSTALLEDPATTERN=talib.CDLSTALLEDPATTERN(df["open"], df["high"], df["low"], df["close"])),
            "cdlsticksandwich": lambda df: df.assign(CDLSTICKSANDWICH=talib.CDLSTICKSANDWICH(df["open"], df["high"], df["low"], df["close"])),
            "cdltakuri": lambda df: df.assign(CDLTAKURI=talib.CDLTAKURI(df["open"], df["high"], df["low"], df["close"])),
            "cdltasukigap": lambda df: df.assign(CDLTASUKIGAP=talib.CDLTASUKIGAP(df["open"], df["high"], df["low"], df["close"])),
            "cdlthrusting": lambda df: df.assign(CDLTHRUSTING=talib.CDLTHRUSTING(df["open"], df["high"], df["low"], df["close"])),
            "cdltristar": lambda df: df.assign(CDLTRISTAR=talib.CDLTRISTAR(df["open"], df["high"], df["low"], df["close"])),
            "cdlunique3river": lambda df: df.assign(CDLUNIQUE3RIVER=talib.CDLUNIQUE3RIVER(df["open"], df["high"], df["low"], df["close"])),
            "cdlupsidegap2crows": lambda df: df.assign(CDLUPSIDEGAP2CROWS=talib.CDLUPSIDEGAP2CROWS(df["open"], df["high"], df["low"], df["close"])),
            "cdlxsidegap3methods": lambda df: df.assign(CDLXSIDEGAP3METHODS=talib.CDLXSIDEGAP3METHODS(df["open"], df["high"], df["low"], df["close"])),
             # Statistic Functions
            "beta": lambda df: df.assign(BETA=talib.BETA(df["high"], df["low"])),
            "correl": lambda df: df.assign(CORREL=talib.CORREL(df["high"], df["low"])),
            "linearreg": lambda df: df.assign(LINEARREG=talib.LINEARREG(df["close"])),
            "linearreg_angle": lambda df: df.assign(LINEARREG_ANGLE=talib.LINEARREG_ANGLE(df["close"])),
            "linearreg_intercept": lambda df: df.assign(LINEARREG_INTERCEPT=talib.LINEARREG_INTERCEPT(df["close"])),
            "linearreg_slope": lambda df: df.assign(LINEARREG_SLOPE=talib.LINEARREG_SLOPE(df["close"])),
            "stddev": lambda df: df.assign(STDDEV=talib.STDDEV(df["close"])),
            "tsf": lambda df: df.assign(TSF=talib.TSF(df["close"])),
            "var": lambda df: df.assign(VAR=talib.VAR(df["close"])),
        }

    def apply(self, df) -> pd.DataFrame:
        df = df.copy()

        # Start with just OHLCV once
        indicators_df = df[["open", "high", "low", "close", "volume"]].copy()
        existing_cols = set(indicators_df.columns)

        for name in self.randomly_enabled_indicators:
            func = self.indicator_map.get(name)
            if func:
                sig = inspect.signature(func)
                params = sig.parameters

                # Call the function with or without timeperiod
                if len(params) >= 3:
                    result_df = func(df.copy(), 20)
                else:
                    result_df = func(df.copy())

                # Find only the new columns added by the indicator function
                new_cols = set(result_df.columns) - existing_cols
                new_data = result_df[list(new_cols)]

                # Merge new indicator columns into indicators_df
                indicators_df = pd.concat([indicators_df, new_data], axis=1)

                # Update existing_cols so we don't keep collecting the same again
                existing_cols.update(new_cols)
            else:
                print(f"[WARN] No handler for indicator '{name}'")

        return indicators_df

    # === Overlap Studies ===
    def _apply_sma(self, df,timeperiod=20):
         df["SMA_20"] = talib.SMA(df["close"], timeperiod) 
         return df
    def _apply_ema(self, df,timeperiod=20):
         df["EMA_20"] = talib.EMA(df["close"], timeperiod)
         return df
    def _apply_wma(self, df,timeperiod=20):
         df["WMA_20"] = talib.WMA(df["close"], timeperiod)
         return df
    def _apply_dema(self, df,timeperiod=20): 
        df["DEMA_20"] = talib.DEMA(df["close"], timeperiod)
        return df
    def _apply_tema(self, df,timeperiod=20): 
        df["TEMA_20"] = talib.TEMA(df["close"], timeperiod)
        return df
    def _apply_trima(self, df,timeperiod=20):
         df["TRIMA_20"] = talib.TRIMA(df["close"], timeperiod)
         return df
    def _apply_kama(self, df,timeperiod=20):
         df["KAMA_20"] = talib.KAMA(df["close"], timeperiod)
         return df
    def _apply_mama(self, df):
         m, f = talib.MAMA(df["close"])
         df["MAMA"], df["FAMA"] = m, f
         return df
    def _apply_mavp(self, df):
         df["MAVP"] = talib.MAVP(df["close"], df["volume"], 2, 30)
         return df
    def _apply_t3(self, df,timeperiod=20):
         df["T3_20"] = talib.T3(df["close"], timeperiod)
         return df
    def _apply_sar(self, df):
         df["SAR"] = talib.SAR(df["high"], df["low"])
         return df
    def _apply_sarext(self, df):
         df["SAREXT"] = talib.SAREXT(df["high"], df["low"])
         return df
    def _apply_midpoint(self, df,timeperiod=20):
         df["MidPoint"] = talib.MIDPOINT(df["close"], timeperiod); return df
    def _apply_midprice(self, df,timeperiod=20):
         df["MidPrice"] = talib.MIDPRICE(df["high"], df["low"], timeperiod); return df
    def _apply_ht_trendline(self, df):
         df["HT_Trendline"] = talib.HT_TRENDLINE(df["close"]); return df
    def _apply_bollinger_bands(self, df,timeperiod=20):
        u, m, l = talib.BBANDS(df["close"], timeperiod)
        df["BB_upper"], df["BB_middle"], df["BB_lower"] = u, m, l
        return df

    # === Momentum Indicators ===
    def _apply_mfi(self, df, timeperiod=20):
     df["MFI"] = talib.MFI(df["high"], df["low"], df["close"], df["volume"], timeperiod)
     return df

    def _apply_minus_di(self, df, timeperiod=20):
        df["MINUS_DI"] = talib.MINUS_DI(df["high"], df["low"], df["close"], timeperiod)
        return df

    def _apply_minus_dm(self, df, timeperiod=20):
        df["MINUS_DM"] = talib.MINUS_DM(df["high"], df["low"], timeperiod)
        return df

    def _apply_mom(self, df, timeperiod=20):
        df["MOM"] = talib.MOM(df["close"], timeperiod)
        return df

    def _apply_plus_di(self, df, timeperiod=20):
        df["PLUS_DI"] = talib.PLUS_DI(df["high"], df["low"], df["close"], timeperiod)
        return df

    def _apply_plus_dm(self, df, timeperiod=20):
        df["PLUS_DM"] = talib.PLUS_DM(df["high"], df["low"], timeperiod)
        return df

    def _apply_ppo(self, df):
        df["PPO"] = talib.PPO(df["close"])
        return df  # PPO uses fast/slow periods by default internally

    def _apply_roc(self, df, timeperiod=20):
        df["ROC"] = talib.ROC(df["close"], timeperiod)
        return df

    def _apply_rocp(self, df, timeperiod=20):
        df["ROCP"] = talib.ROCP(df["close"], timeperiod)
        return df

    def _apply_rocr(self, df, timeperiod=20):
        df["ROCR"] = talib.ROCR(df["close"], timeperiod)
        return df

    def _apply_rocr100(self, df, timeperiod=20):
        df["ROCR100"] = talib.ROCR100(df["close"], timeperiod)
        return df

    def _apply_rsi(self, df, timeperiod=20):
        df["RSI"] = talib.RSI(df["close"], timeperiod)
        return df

    def _apply_stoch(self, df):
        slowk, slowd = talib.STOCH(df["high"], df["low"], df["close"])
        df["STOCH_k"], df["STOCH_d"] = slowk, slowd
        return df

    def _apply_stochf(self, df):
        fastk, fastd = talib.STOCHF(df["high"], df["low"], df["close"])
        df["STOCHF_k"], df["STOCHF_d"] = fastk, fastd
        return df

    def _apply_stochrsi(self, df):
        fastk, fastd = talib.STOCHRSI(df["close"])
        df["STOCHRSI_k"], df["STOCHRSI_d"] = fastk, fastd
        return df

    def _apply_trix(self, df, timeperiod=20):
        df["TRIX"] = talib.TRIX(df["close"], timeperiod)
        return df

    def _apply_ultosc(self, df):
        df["ULTOSC"] = talib.ULTOSC(df["high"], df["low"], df["close"])
        return df  # ULTOSC has 3 timeperiods, so we keep the default

    def _apply_willr(self, df, timeperiod=20):
        df["WILLR"] = talib.WILLR(df["high"], df["low"], df["close"], timeperiod)
        return df

    def _apply_doji_pattern(self, df):
        df["CDLDOJI"] = talib.CDLDOJI(df["open"], df["high"], df["low"], df["close"])
        return df

    # === Volume Indicators ===
    def _apply_ad(self, df):
       df["AD"] = talib.AD(df["high"], df["low"], df["close"], df["volume"])
       return df

    def _apply_adosc(self, df):
       df["ADOSC"] = talib.ADOSC(df["high"], df["low"], df["close"], df["volume"])
       return df

    def _apply_obv(self, df):
       df["OBV"] = talib.OBV(df["close"], df["volume"])
       return df
    # === Cycle Indicators ===
    def _apply_ht_dcperiod(self, df):
      df["HT_DCPERIOD"] = talib.HT_DCPERIOD(df["close"])
      return df

    def _apply_ht_dcphase(self, df):
       df["HT_DCPHASE"] = talib.HT_DCPHASE(df["close"])
       return df

    def _apply_ht_phasor(self, df):
       inphase, quadrature = talib.HT_PHASOR(df["close"])
       df["HT_PHASOR_inphase"], df["HT_PHASOR_quadrature"] = inphase, quadrature
       return df

    def _apply_ht_sine(self, df):
      sine, leadsine = talib.HT_SINE(df["close"])
      df["HT_SINE_sine"], df["HT_SINE_leadsine"] = sine, leadsine
      return df

    def _apply_ht_trendmode(self, df):
      df["HT_TRENDMODE"] = talib.HT_TRENDMODE(df["close"])
      return df
    # === Price Transform Indicators ===
    def _apply_avgprice(self, df):
        df["AVGPRICE"] = talib.AVGPRICE(df["open"], df["high"], df["low"], df["close"])
        return df

    def _apply_medprice(self, df):
        df["MEDPRICE"] = talib.MEDPRICE(df["high"], df["low"])
        return df

    def _apply_typprice(self, df):
        df["TYPPRICE"] = talib.TYPPRICE(df["high"], df["low"], df["close"])
        return df

    def _apply_wclprice(self, df):
        df["WCLPRICE"] = talib.WCLPRICE(df["high"], df["low"], df["close"])
        return df

    # === Volatility Indicators ===
    def _apply_atr(self, df,timeperiod=20):
        df["ATR_14"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod)
        return df

    def _apply_natr(self, df,timeperiod=20):
        df["NATR_14"] = talib.NATR(df["high"], df["low"], df["close"], timeperiod)
        return df

    def _apply_trange(self, df):
        df["TRANGE"] = talib.TRANGE(df["high"], df["low"], df["close"])
        return df
    def _apply_adx(self, df,timeperiod=20):
        df["ADX_14"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod)
        return df

    def _apply_adxr(self, df,timeperiod=20):
        df["ADXR_14"] = talib.ADXR(df["high"], df["low"], df["close"], timeperiod)
        return df

    def _apply_apo(self, df):
        df["APO"] = talib.APO(df["close"], fastperiod=12, slowperiod=26, matype=0)
        return df

    def _apply_aroon(self, df,timeperiod=20):
        aroon_down, aroon_up = talib.AROON(df["high"], df["low"], timeperiod)
        df["AROON_down"] = aroon_down
        df["AROON_up"] = aroon_up
        return df

    def _apply_aroonosc(self, df,timeperiod=20):
        df["AROONOSC"] = talib.AROONOSC(df["high"], df["low"], timeperiod)
        return df

    def _apply_bop(self, df):
        df["BOP"] = talib.BOP(df["open"], df["high"], df["low"], df["close"])
        return df

    def _apply_cci(self, df,timeperiod=20):
        df["CCI_14"] = talib.CCI(df["high"], df["low"], df["close"], timeperiod)
        return df

    def _apply_cmo(self, df,timeperiod=20):
        df["CMO_14"] = talib.CMO(df["close"], timeperiod)
        return df

    def _apply_dx(self, df,timeperiod=20):
        df["DX_14"] = talib.DX(df["high"], df["low"], df["close"], timeperiod)
        return df

    def _apply_macd(self, df):
        macd, signal, hist = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
        df["MACD"] = macd
        df["MACD_signal"] = signal
        df["MACD_hist"] = hist
        return df

    def _apply_macdext(self, df):
        macd, signal, hist = talib.MACDEXT(
            df["close"],
            fastperiod=12,
            fastmatype=0,
            slowperiod=26,
            slowmatype=0,
            signalperiod=9,
            signalmatype=0
        )
        df["MACDEXT"] = macd
        df["MACDEXT_signal"] = signal
        df["MACDEXT_hist"] = hist
        return df

    def _apply_macdfix(self, df):
        macd, signal, hist = talib.MACDFIX(df["close"], signalperiod=9)
        df["MACDFIX"] = macd
        df["MACDFIX_signal"] = signal
        df["MACDFIX_hist"] = hist
        return df


    # @staticmethod
    # def save_to_csv(indicators_df: pd.DataFrame, exchange: str, symbol: str, output_dir="ti_output"):
    #     os.makedirs(output_dir, exist_ok=True)
    #     filename = f"{exchange}_{symbol}.csv"
    #     path = os.path.join(output_dir, filename)
    #     indicators_df.to_csv(path, index=True)
    #     print(f"[INFO] Saved indicator data to {path}")
    #     print(indicators_df.head())
