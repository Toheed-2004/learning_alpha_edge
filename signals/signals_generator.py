import pandas as pd


def bbands_signal(df):
    long = (df["BB_middle"] > df["BB_lower"]) & (df["BB_middle"].shift(1) <= df["BB_lower"].shift(1))
    short = (df["BB_middle"] < df["BB_upper"]) & (df["BB_middle"].shift(1) >= df["BB_upper"].shift(1))
    return long, short

def dema_signal(df):
    long = df["DEMA_20"] > df["DEMA_20"].shift(1)
    short = df["DEMA_20"] < df["DEMA_20"].shift(1)
    return long, short

def ema_signal(df):
    long = df["EMA_20"] > df["EMA_20"].shift(1)
    short = df["EMA_20"] < df["EMA_20"].shift(1)
    return long, short

def ht_trendline_signal(df):
    long = df["HT_Trendline"] > df["HT_Trendline"].shift(1)
    short = df["HT_Trendline"] < df["HT_Trendline"].shift(1)
    return long, short

def kama_signal(df):
    long = df["KAMA_20"] > df["KAMA_20"].shift(1)
    short = df["KAMA_20"] < df["KAMA_20"].shift(1)
    return long, short

def ma_signal(df):
    long = df["MA"] > df["MA"].shift(1)
    short = df["MA"] < df["MA"].shift(1)
    return long, short

def mama_signal(df):
    long = df["MAMA"] > df["FAMA"]
    short = df["MAMA"] < df["FAMA"]
    return long, short

def mavp_signal(df):
    long = df["MAVP"] > df["MAVP"].shift(1)
    short = df["MAVP"] < df["MAVP"].shift(1)
    return long, short

def midpoint_signal(df):
    long = df["MidPoint"] > df["MidPoint"].shift(1)
    short = df["MidPoint"] < df["MidPoint"].shift(1)
    return long, short

def midprice_signal(df):
    long = df["MidPrice"] > df["MidPrice"].shift(1)
    short = df["MidPrice"] < df["MidPrice"].shift(1)
    return long, short

def sar_signal(df):
    long = df["SAR"] < df["close"]
    short = df["SAR"] > df["close"]
    return long, short

def sarext_signal(df):
    long = df["SAREXT"] < df["close"]
    short = df["SAREXT"] > df["close"]
    return long, short

def sma_signal(df):
    long = df["SMA_20"] > df["SMA_20"].shift(1)
    short = df["SMA_20"] < df["SMA_20"].shift(1)
    return long, short

def t3_signal(df):
    long = df["T3_20"] > df["T3_20"].shift(1)
    short = df["T3_20"] < df["T3_20"].shift(1)
    return long, short

def tema_signal(df):
    long = df["TEMA_20"] > df["TEMA_20"].shift(1)
    short = df["TEMA_20"] < df["TEMA_20"].shift(1)
    return long, short

def trima_signal(df):
    long = df["TRIMA_20"] > df["TRIMA_20"].shift(1)
    short = df["TRIMA_20"] < df["TRIMA_20"].shift(1)
    return long, short

def wma_signal(df):
    long = df["WMA_20"] > df["WMA_20"].shift(1)
    short = df["WMA_20"] < df["WMA_20"].shift(1)
    return long, short
def adx_signal(df):
    long = df["ADX_14"] > 25
    short = df["ADX_14"] < 20
    return long, short

def adxr_signal(df):
     long = df["ADXR_14"] > 25
     short = df["ADXR_14"] < 20
     return long,short
 

def apo_signal(df):
    long = df["APO"] > 0
    short = df["APO"] < 0
    return long,short

def aroon_signal(df):

    long = (df["AROON_up"] > 70) & (df["AROON_down"] < 30)
    short = (df["AROON_down"] > 70) & (df["AROON_up"] < 30)
    return long,short
def aroonosc_signal(df):
    long = df["AROONOSC"] > 50
    short = df["AROONOSC"] < -50
    return long, short

def bop_signal(df):
    long = df["BOP"] > 0.5
    short = df["BOP"] < -0.5
    return long, short

def cci_signal(df):
    long = df["CCI_14"] > 100
    short = df["CCI_14"] < -100
    return long, short

def cmo_signal(df):
    long = df["CMO_14"] > 50
    short = df["CMO_14"] < -50
    return long, short

def dx_signal(df):
    long = df["DX"] > 20
    short = df["DX"] < 20
    return long, short
def macd_signal(df):
    long = df["MACD"] > df["MACD_signal"]
    short = df["MACD"] < df["MACD_signal"]
    return long, short

def macdext_signal(df):
    long = df["MACDEXT"] > df["MACDEXT_signal"]
    short = df["MACDEXT"] < df["MACDEXT_signal"]
    return long, short

def macdfix_signal(df):
    long = df["MACDFIX"] > df["MACDFIX_signal"]
    short = df["MACDFIX"] < df["MACDFIX_signal"]
    return long, short

def mfi_signal(df):
    long = df["MFI"] < 20
    short = df["MFI"] > 80
    return long, short

def minus_di_signal(df):
    long = df["PLUS_DI"] > df["MINUS_DI"]
    short = df["MINUS_DI"] > df["PLUS_DI"]
    return long, short

def minus_dm_signal(df):
    long = df["PLUS_DM"] > df["MINUS_DM"]
    short = df["MINUS_DM"] > df["PLUS_DM"]
    return long, short

def mom_signal(df):
    long = df["MOM_10"] > 0
    short = df["MOM_10"] < 0
    return long, short

def plus_di_signal(df):
    long = df["PLUS_DI"] > 25
    short = df["PLUS_DI"] < 20
    return long, short

def plus_dm_signal(df):
    long = df["PLUS_DM"] > 25
    short = df["PLUS_DM"] < 20
    return long, short

def ppo_signal(df):
    long = df["PPO"] > 0
    short = df["PPO"] < 0
    return long, short

def roc_signal(df):
    long = df["ROC"] > 0
    short = df["ROC"] < 0
    return long, short

def rocp_signal(df):
    long = df["ROCP"] > 0
    short = df["ROCP"] < 0
    return long, short

def rocr_signal(df):
    long = df["ROCR"] > 1
    short = df["ROCR"] < 1
    return long, short

def rocr100_signal(df):
    long = df["ROCR100"] > 100
    short = df["ROCR100"] < 100
    return long, short

def rsi_signal(df):
    long = df["RSI"] < 30
    short = df["RSI"] > 70
    return long, short

def stoch_signal(df):
    long = (df["STOCH_k"] > df["STOCH_d"]) & (df["STOCH_k"] < 20)
    short = (df["STOCH_k"] < df["STOCH_d"]) & (df["STOCH_k"] > 80)
    return long, short

def stochf_signal(df):
    long = (df["STOCHF_k"] > df["STOCHF_d"]) & (df["STOCHF_k"] < 20)
    short = (df["STOCHF_k"] < df["STOCHF_d"]) & (df["STOCHF_k"] > 80)
    return long, short

def stochrsi_signal(df):
    long = (df["STOCHRSI_k"] > df["STOCHRSI_d"]) & (df["STOCHRSI_k"] < 20)
    short = (df["STOCHRSI_k"] < df["STOCHRSI_d"]) & (df["STOCHRSI_k"] > 80)
    return long, short

def trix_signal(df):
    long = df["TRIX"] > 0
    short = df["TRIX"] < 0
    return long, short

def ultosc_signal(df):
    long = df["ULTOSC"] < 30
    short = df["ULTOSC"] > 70
    return long, short

def willr_signal(df):
    long = df["WILLR"] < -80
    short = df["WILLR"] > -20
    return long, short
def ad_signal(df):
    long = df["AD"] > df["AD"].shift(1)
    short = df["AD"] < df["AD"].shift(1)
    return long, short

def adosc_signal(df):
    long = df["ADOSC"] > df["ADOSC"].shift(1)
    short = df["ADOSC"] < df["ADOSC"].shift(1)
    return long, short

def obv_signal(df):
    long = df["OBV"] > df["OBV"].shift(1)
    short = df["OBV"] < df["OBV"].shift(1)
    return long, short
def ht_dcperiod_signal(df):
    long = df["HT_DCPERIOD"].diff() > 2
    short = df["HT_DCPERIOD"].diff() < -2
    return long, short

def ht_dcphase_signal(df):
    long = df["HT_DCPHASE"].diff() < -300
    short = pd.Series([False] * len(df), index=df.index)  # always False
    return long, short

def ht_phasor_signal(df):
    long = (df["HT_PHASOR_inphase"] > df["HT_PHASOR_quadrature"]) & (df["HT_PHASOR_inphase"].shift(1) <= df["HT_PHASOR_quadrature"].shift(1))
    short = (df["HT_PHASOR_inphase"] < df["HT_PHASOR_quadrature"]) & (df["HT_PHASOR_inphase"].shift(1) >= df["HT_PHASOR_quadrature"].shift(1))
    return long, short

def ht_sine_signal(df):
    long = (df["HT_SINE_sine"] > df["HT_SINE_leadsine"]) & (df["HT_SINE_sine"].shift(1) <= df["HT_SINE_leadsine"].shift(1))
    short = (df["HT_SINE_sine"] < df["HT_SINE_leadsine"]) & (df["HT_SINE_sine"].shift(1) >= df["HT_SINE_leadsine"].shift(1))
    return long, short

def ht_trendmode_signal(df):
    long = (df["HT_TRENDMODE"] == 1) & (df["HT_TRENDMODE"].shift(1) == 0)
    short = (df["HT_TRENDMODE"] == 0) & (df["HT_TRENDMODE"].shift(1) == 1)
    return long, short
def avgprice_signal(df):
    ema10 = df["AVGPRICE"].ewm(span=10).mean()
    long = (df["AVGPRICE"] > ema10) & (df["AVGPRICE"].shift(1) <= ema10.shift(1))
    short = (df["AVGPRICE"] < ema10) & (df["AVGPRICE"].shift(1) >= ema10.shift(1))
    return long, short

def medprice_signal(df):
    long = df["MEDPRICE"].diff() > 0.5
    short = df["MEDPRICE"].diff() < -0.5
    return long, short

def typprice_signal(df):
    sma10 = df["TYPPRICE"].rolling(window=10).mean()
    long = (df["TYPPRICE"] > sma10) & (df["TYPPRICE"].shift(1) <= sma10.shift(1))
    short = (df["TYPPRICE"] < sma10) & (df["TYPPRICE"].shift(1) >= sma10.shift(1))
    return long, short

def wclprice_signal(df):
    long = (df["WCLPRICE"].diff() > 0) & (df["WCLPRICE"].diff(2) > 0)
    short = (df["WCLPRICE"].diff() < 0) & (df["WCLPRICE"].diff(2) < 0)
    return long, short
def atr_signal(df):
    sma14 = df["ATR_14"].rolling(window=14).mean()
    long = df["ATR_14"] > sma14
    short = df["ATR_14"] < sma14
    return long, short

def natr_signal(df):
    sma10 = df["NATR_14"].rolling(window=10).mean()
    long = df["NATR_14"] > sma10
    short = df["NATR_14"] < sma10
    return long, short

def trange_signal(df):
    sma10 = df["TRANGE"].rolling(window=10).mean()
    std10 = df["TRANGE"].rolling(window=10).std()
    long = df["TRANGE"] > (sma10 + std10)
    short = df["TRANGE"] < (sma10 - std10)
    return long, short
def cdl2crows_signal(df):
    long = pd.Series([0] * len(df), index=df.index)
    short = df["CDL2CROWS"] == -100
    return long, short

def cdl3blackcrows_signal(df):
    long = pd.Series([0] * len(df), index=df.index)
    short = df["CDL3BLACKCROWS"] == -100
    return long, short

def cdl3inside_signal(df):
    long = df["CDL3INSIDE"] == 100
    short = df["CDL3INSIDE"] == -100
    return long, short
def cdl3linestrike_signal(df):
    long=df["CDL3LINESTRIKE"] == 100
    short= df["CDL3LINESTRIKE"] == -100
    return long,short

def cdl3outside_signal(df):
    long,short=df["CDL3OUTSIDE"] == 100, df["CDL3OUTSIDE"] == -100
    return long,short

def cdl3starsinsouth_signal(df):
    long=df["CDL3STARSINSOUTH"] == 100
    short=df["CDL3STARSINSOUTH"] * 0  # short = 0
    return long,short

def cdl3whitesoldiers_signal(df):
    long=df["CDL3WHITESOLDIERS"] == 100
    short= df["CDL3WHITESOLDIERS"] * 0  # short = 0
    return long,short
def cdlabandonedbaby_signal(df):
    long= df["CDLABANDONEDBABY"] == 100
    short=df["CDLABANDONEDBABY"] == -100
    return long,short
def cdladvanceblock_signal(df):
    long=df["CDLADVANCEBLOCK"] * 0
    short=df["CDLADVANCEBLOCK"] == -100
    return long,short
def cdlbelthold_signal(df):
    long=df["CDLBELTHOLD"] == 100
    short=df["CDLBELTHOLD"] == -100
    return long,short
def cdlbreakaway_signal(df):
    long= df["CDLBREAKAWAY"] == 100
    short=df["CDLBREAKAWAY"] == -100
    return long,short
def cdlclosingmarubozu_signal(df):
    long=df["CDLCLOSINGMARUBOZU"] == 100
    short=df["CDLCLOSINGMARUBOZU"] == -100
    return long,short
def cdlconcealbabyswall_signal(df):
    long=df["CDLCONCEALBABYSWALL"] == 100
    short=df["CDLCONCEALBABYSWALL"] * 0
    return long,short
def cdlcounterattack_signal(df):
    long=df["CDLCOUNTERATTACK"] == 100
    short=df["CDLCOUNTERATTACK"] == -100
    return long,short

def cdldarkcloudcover_signal(df):
    long= df["CDLDARKCLOUDCOVER"] * 0
    short=df["CDLDARKCLOUDCOVER"] == -100
    return long,short
def cdldoji_signal(df):
    long= df["CDLDOJI"] * 0
    short=df["CDLDOJI"] * 0  # both long and short are 0
    return long,short
def cdldojistar_signal(df):
    long=df["CDLDOJISTAR"] == 100
    short=df["CDLDOJISTAR"] == -100
    return long,short
def cdldragonflydoji_signal(df):
    long= df["CDLDRAGONFLYDOJI"] == 100
    short=df["CDLDRAGONFLYDOJI"] * 0
    return long,short
def cdlengulfing_signal(df):
    long= df["CDLENGULFING"] == 100
    short=df["CDLENGULFING"] == -100
    return long,short
def cdleveningdojistar_signal(df):
    long= df["CDLEVENINGDOJISTAR"] * 0
    short=df["CDLEVENINGDOJISTAR"] == -100
    return long,short
def cdleveningstar_signal(df):
    long=df["CDLEVENINGSTAR"] * 0
    short=df["CDLEVENINGSTAR"] == -100
    return long,short
def cdlgapsidesidewhite_signal(df):
    long=df["CDLGAPSIDESIDEWHITE"] == 100
    short=df["CDLGAPSIDESIDEWHITE"] * 0
    return long,short
def cdlgravestonedoji_signal(df):
    long=df["CDLGRAVESTONEDOJI"] * 0
    short=df["CDLGRAVESTONEDOJI"] == -100
    return long,short
def cdlhammer_signal(df):
    long=df["CDLHAMMER"] == 100
    short=df["CDLHAMMER"] * 0
    return long,short
def cdlhangingman_signal(df):
    long=df["CDLHANGINGMAN"] * 0
    short=df["CDLHANGINGMAN"] == -100
    return long,short
def cdlharami_signal(df):
     long=df["CDLHARAMI"] == 100
     short=df["CDLHARAMI"] == -100
     return long,short
def cdlharamicross_signal(df):
    long= df["CDLHARAMICROSS"] == 100
    short=df["CDLHARAMICROSS"] == -100
    return long,short
def cdlhighwave_signal(df):
    long=df["CDLHIGHWAVE"] == 100
    short=df["CDLHIGHWAVE"] == -100
    return long,short
def cdlhikkake_signal(df):
    long=df["CDLHIKKAKE"] == 100
    short=df["CDLHIKKAKE"] == -100
    return long,short
def cdlhikkakemod_signal(df):
    long=df["CDLHIKKAKEMOD"] == 100
    short=df["CDLHIKKAKEMOD"] == -100
    return long,short
def cdlhomingpigeon_signal(df):
    long=df["CDLHOMINGPIGEON"] == 100
    short=df["CDLHOMINGPIGEON"] * 0
    return long,short
def cdlidentical3crows_signal(df):
    long=df["CDLIDENTICAL3CROWS"] * 0
    short= df["CDLIDENTICAL3CROWS"] == -100
    return long,short
def cdlinneck_signal(df):
    long= df["CDLINNECK"] * 0
    short=df["CDLINNECK"] == -100
    return long,short
def cdlinvertedhammer_signal(df):
    long= df["CDLINVERTEDHAMMER"] == 100
    short=df["CDLINVERTEDHAMMER"] * 0
    return long,short
def cdlkicking_signal(df):
    long=df["CDLKICKING"] == 100
    short=df["CDLKICKING"] == -100
    return long,short
def cdlkickingbylength_signal(df):
    long= df["CDLKICKINGBYLENGTH"] == 100
    short=df["CDLKICKINGBYLENGTH"] == -100
    return long,short
def cdlladderbottom_signal(df):
    long=df["CDLLADDERBOTTOM"] == 100
    short= df["CDLLADDERBOTTOM"] * 0
    return long,short
def cdllongleggeddoji_signal(df):
    long=df["CDLLONGLEGGEDDOJI"] == 100
    short=df["CDLLONGLEGGEDDOJI"] == -100
    return long,short
def cdllongline_signal(df):
    long=df["CDLLONGLINE"] == 100
    short=df["CDLLONGLINE"] == -100
    return long,short
def cdlmarubozu_signal(df):
    long= df["CDLMARUBOZU"] == 100 
    short=df["CDLMARUBOZU"] == -100
    return long,short
def cdlmatchinglow_signal(df):
    long=df["CDLMATCHINGLOW"] == 100
    short= df["CDLMATCHINGLOW"] * 0
    return long,short
def cdlmathold_signal(df):
    long = df["CDLMATHOLD"] == 100
    short = df["CDLMATHOLD"] == -100
    return long, short

def cdlmorningdojistar_signal(df):
    long = df["CDLMORNINGDOJISTAR"] == 100
    short = df["CDLMORNINGDOJISTAR"] * 0
    return long, short

def cdlmorningstar_signal(df):
    long = df["CDLMORNINGSTAR"] == 100
    short = df["CDLMORNINGSTAR"] * 0
    return long, short

def cdlonneck_signal(df):
    long = df["CDLONNECK"] * 0
    short = df["CDLONNECK"] == -100
    return long, short

def cdlpiercing_signal(df):
    long = df["CDLPIERCING"] == 100
    short = df["CDLPIERCING"] * 0
    return long, short

def cdlrickshawman_signal(df):
    long = df["CDLRICKSHAWMAN"] == 100
    short = df["CDLRICKSHAWMAN"] == -100
    return long, short

def cdlrisefall3methods_signal(df):
    long = df["CDLRISEFALL3METHODS"] == 100
    short = df["CDLRISEFALL3METHODS"] == -100
    return long, short

def cdlseparatinglines_signal(df):
    long = df["CDLSEPARATINGLINES"] == 100
    short = df["CDLSEPARATINGLINES"] == -100
    return long, short

def cdlshootingstar_signal(df):
    long = df["CDLSHOOTINGSTAR"] * 0
    short = df["CDLSHOOTINGSTAR"] == -100
    return long, short

def cdlshortline_signal(df):
    long = df["CDLSHORTLINE"] == 100
    short = df["CDLSHORTLINE"] == -100
    return long, short

def cdlspinningtop_signal(df):
    long = df["CDLSPINNINGTOP"] == 100
    short = df["CDLSPINNINGTOP"] == -100
    return long, short

def cdlstalledpattern_signal(df):
    long = df["CDLSTALLEDPATTERN"] == 100
    short = df["CDLSTALLEDPATTERN"] == -100
    return long, short

def cdlsticksandwich_signal(df):
    long = df["CDLSTICKSANDWICH"] == 100
    short = df["CDLSTICKSANDWICH"] * 0
    return long, short

def cdltakuri_signal(df):
    long = df["CDLTAKURI"] == 100
    short = df["CDLTAKURI"] * 0
    return long, short

def cdltasukigap_signal(df):
    long = df["CDLTASUKIGAP"] == 100
    short = df["CDLTASUKIGAP"] == -100
    return long, short

def cdlthrusting_signal(df):
    long = df["CDLTHRUSTING"] * 0
    short = df["CDLTHRUSTING"] == -100
    return long, short

def cdltristar_signal(df):
    long = df["CDLTRISTAR"] == 100
    short = df["CDLTRISTAR"] == -100
    return long, short

def cdlunique3river_signal(df):
    long = df["CDLUNIQUE3RIVER"] == 100
    short = df["CDLUNIQUE3RIVER"] * 0
    return long, short

def cdlupsidegap2crows_signal(df):
    long = df["CDLUPSIDEGAP2CROWS"] * 0
    short = df["CDLUPSIDEGAP2CROWS"] == -100
    return long, short

def cdlxsidegap3methods_signal(df):
    long = df["CDLXSIDEGAP3METHODS"] == 100
    short = df["CDLXSIDEGAP3METHODS"] == -100
    return long, short

def beta_signal(df):
    long = df["BETA"] > df["BETA"].shift(1)
    short = df["BETA"] < df["BETA"].shift(1)
    return long, short

def correl_signal(df):
    long = df["CORREL"] > 0.5
    short = df["CORREL"] < -0.5
    return long, short

def linearreg_signal(df):
    long = df["LINEARREG"] > df["LINEARREG"].shift(1)
    short = df["LINEARREG"] < df["LINEARREG"].shift(1)
    return long, short

def linearreg_angle_signal(df):
    long = df["LINEARREG_ANGLE"] > 0
    short = df["LINEARREG_ANGLE"] < 0
    return long, short

def linearreg_intercept_signal(df):
    long = df["LINEARREG_INTERCEPT"] > df["LINEARREG_INTERCEPT"].shift(1)
    short = df["LINEARREG_INTERCEPT"] < df["LINEARREG_INTERCEPT"].shift(1)
    return long, short

def linearreg_slope_signal(df):
    long = df["LINEARREG_SLOPE"] > 0
    short = df["LINEARREG_SLOPE"] < 0
    return long, short

def stddev_signal(df):
    mean_rolling = df["STDDEV"].rolling(10).mean()
    long = df["STDDEV"] > mean_rolling
    short = df["STDDEV"] < mean_rolling
    return long, short

def tsf_signal(df):
    long = df["TSF"] > df["TSF"].shift(1)
    short = df["TSF"] < df["TSF"].shift(1)
    return long, short

def var_signal(df):
    mean_rolling = df["VAR"].rolling(10).mean()
    long = df["VAR"] > mean_rolling
    short = df["VAR"] < mean_rolling
    return long, short

signals_map = {
            # === Overlap Studies ===
            "sma": sma_signal,
            "ema": ema_signal,
            "wma": sma_signal,
            "dema": dema_signal,
            "tema": tema_signal,
            "trima": trima_signal,
            "kama": kama_signal,
            "mama": mama_signal,
            "mavp": mavp_signal,
            "t3": t3_signal,
            "sar": sar_signal,
            "sarext": sarext_signal,
            "midpoint": midpoint_signal,
            "midprice": midprice_signal,
            "ht_trendline": ht_trendline_signal,
            "bbands": bbands_signal,

            # === Momentum Indicators ===
            "adx": adx_signal,
            "adxr": adxr_signal,
            "apo": apo_signal,
            "aroon": aroon_signal,
            "aroonosc": aroonosc_signal,
            "bop": bop_signal,
            "cci": cci_signal,
            "cmo": cmo_signal,
            "dx": dx_signal,
            "macd": macd_signal,
            "macdext": macdext_signal,
            "macdfix": macdfix_signal,
            "mfi": mfi_signal,
            "minus_di": minus_di_signal,
            "minus_dm": minus_dm_signal,
            "mom": mom_signal,
            "plus_di": plus_di_signal,
            "plus_dm": plus_dm_signal,
            "ppo": ppo_signal,
            "roc": roc_signal,
            "rocp": rocp_signal,
            "rocr": rocr_signal,
            "rocr100": rocr100_signal,
            "rsi": rsi_signal,
            "stoch": stoch_signal,
            "stochf": stochf_signal,
            "stochrsi": stochrsi_signal,
            "trix": trix_signal,
            "ultosc": ultosc_signal,
            "willr": willr_signal,
            "ad":ad_signal,
            "adosc": adosc_signal,
            "obv": obv_signal,

            
            "ht_dcperiod": ht_dcperiod_signal,
            "ht_dcphase": ht_dcphase_signal,
            "ht_phasor": ht_phasor_signal,
            "ht_sine": ht_sine_signal,
            "ht_trendmode": ht_trendmode_signal,
            # === Price Transform Indicators ===
            "avgprice": avgprice_signal,
            "medprice": medprice_signal,
            "typprice": typprice_signal,
            "wclprice": wclprice_signal,

            # === Volatility Indicators ===
            "atr": atr_signal,
            "natr": natr_signal,
            "trange": trange_signal,


            # === Pattern Recognition (example)
            "doji_pattern":cdldoji_signal,
             # --- Pattern Recognition ---
            "cdl3blackcrows": cdl3blackcrows_signal,
            "cdl3inside": cdl3inside_signal,
            "cdl3linestrike": cdl3linestrike_signal,
            "cdl3outside":cdl3outside_signal ,
            "cdl3starsinsouth":cdl3starsinsouth_signal,
            "cdl3whitesoldiers": cdl3whitesoldiers_signal,
            "cdlabandonedbaby":cdlabandonedbaby_signal ,
            "cdladvanceblock":cdladvanceblock_signal ,
            "cdlbelthold":cdlbelthold_signal ,
            "cdlbreakaway":cdlbreakaway_signal,
            "cdlclosingmarubozu":cdlclosingmarubozu_signal ,
            "cdlconcealbabyswall":cdlconcealbabyswall_signal,
             "cdlcounterattack" :cdlcounterattack_signal,
            "cdldarkcloudcover" :cdldarkcloudcover_signal ,
             "cdldoji": cdldoji_signal,
            "cdldojistar": cdldojistar_signal,
            "cdldragonflydoji": cdldragonflydoji_signal,
            "cdlengulfing": cdlengulfing_signal,
            "cdleveningdojistar": cdleveningdojistar_signal,
            "cdleveningstar": cdleveningstar_signal,
            "cdlgapsidesidewhite": cdlgapsidesidewhite_signal,
            "cdlgravestonedoji": cdlgravestonedoji_signal,
            "cdlhammer": cdlhammer_signal,
            "cdlhangingman": cdlhangingman_signal,
            "cdlharami": cdlharami_signal,
            "cdlharamicross": cdlharamicross_signal,
            "cdlhighwave": cdlhighwave_signal,
            "cdlhikkake": cdlhikkake_signal,
            "cdlhikkakemod": cdlhikkakemod_signal,
            "cdlhomingpigeon": cdlhomingpigeon_signal,
            "cdlidentical3crows": cdlidentical3crows_signal,
            "cdlinneck": cdlinneck_signal,
            "cdlinvertedhammer": cdlinvertedhammer_signal,
            "cdlkicking": cdlkicking_signal,
            "cdlkickingbylength": cdlkickingbylength_signal,
            "cdlladderbottom": cdlladderbottom_signal,
            "cdllongleggeddoji": cdllongleggeddoji_signal,
            "cdllongline": cdllongline_signal,
            "cdlmarubozu": cdlmarubozu_signal,
            "cdlmatchinglow": cdlmatchinglow_signal,
            "cdlmathold": cdlmathold_signal,
            "cdlmorningdojistar": cdlmorningdojistar_signal,
            "cdlmorningstar": cdlmorningstar_signal,
            "cdlonneck": cdlonneck_signal,
            "cdlpiercing": cdlpiercing_signal,
            "cdlrickshawman": cdlrickshawman_signal,
            "cdlrisefall3methods": cdlrisefall3methods_signal,
            "cdlseparatinglines": cdlseparatinglines_signal,
            "cdlshootingstar": cdlshootingstar_signal,
            "cdlshortline": cdlshortline_signal,
            "cdlspinningtop": cdlspinningtop_signal,
            "cdlstalledpattern": cdlstalledpattern_signal,
            "cdlsticksandwich": cdlsticksandwich_signal,
            "cdltakuri": cdltakuri_signal,
            "cdltasukigap": cdltasukigap_signal,
            "cdlthrusting": cdlthrusting_signal,
            "cdltristar": cdltristar_signal,
            "cdlunique3river": cdlunique3river_signal,
            "cdlupsidegap2crows": cdlupsidegap2crows_signal,
            "cdlxsidegap3methods": cdlxsidegap3methods_signal,

            # === Statistic Functions ===
            "beta": beta_signal,
            "correl": correl_signal,
            "linearreg": linearreg_signal,
            "linearreg_angle": linearreg_angle_signal,
            "linearreg_intercept": linearreg_intercept_signal,
            "linearreg_slope": linearreg_slope_signal,
            "stddev": stddev_signal,
            "tsf": tsf_signal,
            "var": var_signal,


        
        }
def generate_signals(df, randomly_enabled_indicators):
    # 1. Keep only datetime + OHLCV
    base_df = df[["open", "high", "low", "close", "volume"]].copy()
    if 'datetime' in df.columns:
        base_df.insert(0, "datetime", df["datetime"])

    # 2. Create an empty DataFrame to store signals
    signal_data = pd.DataFrame(index=df.index)

    # 3. Apply signal logic
    for ind in randomly_enabled_indicators:
        signal_func = signals_map.get(ind)
        if signal_func:
            long_cond, short_cond = signal_func(df)

            signal_series = pd.Series(0, index=df.index)
            signal_series[long_cond] = 1
            signal_series[short_cond] = -1

            signal_data[ind + "_signal"] = signal_series
        else:
            print(f"[WARN] No signal logic defined for: {ind}")

    # 4. Combine OHLCV + signal columns
    final_df = pd.concat([base_df, signal_data], axis=1)
    final_df.dropna(inplace=True)

    return final_df
