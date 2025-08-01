import talib

# Indicator Application Functions
def _applysma( df,timeperiod=20):
         df["SMA20"] = talib.SMA(df["close"], timeperiod) 
         return df
def _applysma( df,timeperiod=20):
        df["SMA20"] = talib.SMA(df["close"], timeperiod) 
        return df
def _apply_ema( df,timeperiod=20):
        df["EMA_20"] = talib.EMA(df["close"], timeperiod)
        return df
def _apply_wma( df,timeperiod=20):
        df["WMA_20"] = talib.WMA(df["close"], timeperiod)
        return df
def _apply_dema( df,timeperiod=20): 
    df["DEMA_20"] = talib.DEMA(df["close"], timeperiod)
    return df
def _apply_tema( df,timeperiod=20): 
    df["TEMA_20"] = talib.TEMA(df["close"], timeperiod)
    return df
def _apply_trima( df,timeperiod=20):
        df["TRIMA_20"] = talib.TRIMA(df["close"], timeperiod)
        return df
def _apply_kama( df,timeperiod=20):
        df["KAMA_20"] = talib.KAMA(df["close"], timeperiod)
        return df
def _apply_mama( df):
        m, f = talib.MAMA(df["close"])
        df["MAMA"], df["FAMA"] = m, f
        return df
def _apply_mavp( df):
        df["MAVP"] = talib.MAVP(df["close"], df["volume"], 2, 30)
        return df
def _apply_t3(df,timeperiod=20):
        df["T3_20"] = talib.T3(df["close"], timeperiod)
        return df
def _apply_sar( df):
        df["SAR"] = talib.SAR(df["high"], df["low"])
        return df
def _apply_sarext( df):
        df["SAREXT"] = talib.SAREXT(df["high"], df["low"])
        return df
def _apply_midpoint( df,timeperiod=20):
        df["MidPoint"] = talib.MIDPOINT(df["close"], timeperiod); return df
def _apply_midprice( df,timeperiod=20):
        df["MidPrice"] = talib.MIDPRICE(df["high"], df["low"], timeperiod); return df
def _apply_ht_trendline( df):
        df["HT_Trendline"] = talib.HT_TRENDLINE(df["close"]); return df
def _apply_bollinger_bands( df,timeperiod=20):
    u, m, l = talib.BBANDS(df["close"], timeperiod)
    df["BB_upper"], df["BB_middle"], df["BB_lower"] = u, m, l
    return df

# === Momentum Indicators ===
def _apply_mfi( df, timeperiod=20):
    df["MFI"] = talib.MFI(df["high"], df["low"], df["close"], df["volume"], timeperiod)
    return df

def _apply_minus_di( df, timeperiod=20):
    df["MINUS_DI"] = talib.MINUS_DI(df["high"], df["low"], df["close"], timeperiod)
    return df

def _apply_minus_dm( df, timeperiod=20):
    df["MINUS_DM"] = talib.MINUS_DM(df["high"], df["low"], timeperiod)
    return df

def _apply_mom( df, timeperiod=20):
    df["MOM"] = talib.MOM(df["close"], timeperiod)
    return df

# def _apply_plus_di( df, timeperiod=20):
#     df["PLUS_DI"] = talib.PLUS_DI(df["high"], df["low"], df["close"], timeperiod)
#     return df

def _apply_plus_dm( df, timeperiod=20):
    df["PLUS_DM"] = talib.PLUS_DM(df["high"], df["low"], timeperiod)
    return df

def _apply_ppo( df):
    df["PPO"] = talib.PPO(df["close"])
    return df  # PPO uses fast/slow periods by default internally

def _apply_roc( df, timeperiod=20):
    df["ROC"] = talib.ROC(df["close"], timeperiod)
    return df

def _apply_rocp( df, timeperiod=20):
    df["ROCP"] = talib.ROCP(df["close"], timeperiod)
    return df

def _apply_rocr( df, timeperiod=20):
    df["ROCR"] = talib.ROCR(df["close"], timeperiod)
    return df

def _apply_rocr100( df, timeperiod=20):
    df["ROCR100"] = talib.ROCR100(df["close"], timeperiod)
    return df

def _apply_rsi( df, timeperiod=20):
    df["RSI"] = talib.RSI(df["close"], timeperiod)
    return df

def _apply_stoch( df):
    slowk, slowd = talib.STOCH(df["high"], df["low"], df["close"])
    df["STOCH_k"], df["STOCH_d"] = slowk, slowd
    return df

def _apply_stochf( df):
    fastk, fastd = talib.STOCHF(df["high"], df["low"], df["close"])
    df["STOCHF_k"], df["STOCHF_d"] = fastk, fastd
    return df

def _apply_stochrsi( df):
    fastk, fastd = talib.STOCHRSI(df["close"])
    df["STOCHRSI_k"], df["STOCHRSI_d"] = fastk, fastd
    return df

def _apply_trix( df, timeperiod=20):
    df["TRIX"] = talib.TRIX(df["close"], timeperiod)
    return df

def _apply_ultosc( df):
    df["ULTOSC"] = talib.ULTOSC(df["high"], df["low"], df["close"])
    return df  # ULTOSC has 3 timeperiods, so we keep the default

def _apply_willr( df, timeperiod=20):
    df["WILLR"] = talib.WILLR(df["high"], df["low"], df["close"], timeperiod)
    return df

def _apply_doji_pattern( df):
    df["CDLDOJI"] = talib.CDLDOJI(df["open"], df["high"], df["low"], df["close"])
    return df

# === Volume Indicators ===
def _apply_ad( df):
    df["AD"] = talib.AD(df["high"], df["low"], df["close"], df["volume"])
    return df

def _apply_adosc( df):
    df["ADOSC"] = talib.ADOSC(df["high"], df["low"], df["close"], df["volume"])
    return df

def _apply_obv( df):
    df["OBV"] = talib.OBV(df["close"], df["volume"])
    return df
# === Cycle Indicators ===
def _apply_ht_dcperiod( df):
    df["HT_DCPERIOD"] = talib.HT_DCPERIOD(df["close"])
    return df

def _apply_ht_dcphase( df):
    df["HT_DCPHASE"] = talib.HT_DCPHASE(df["close"])
    return df

def _apply_ht_phasor( df):
    inphase, quadrature = talib.HT_PHASOR(df["close"])
    df["HT_PHASOR_inphase"], df["HT_PHASOR_quadrature"] = inphase, quadrature
    return df

def _apply_ht_sine( df):
    sine, leadsine = talib.HT_SINE(df["close"])
    df["HT_SINE_sine"], df["HT_SINE_leadsine"] = sine, leadsine
    return df

def _apply_ht_trendmode( df):
    df["HT_TRENDMODE"] = talib.HT_TRENDMODE(df["close"])
    return df
# === Price Transform Indicators ===
def _apply_avgprice( df):
    df["AVGPRICE"] = talib.AVGPRICE(df["open"], df["high"], df["low"], df["close"])
    return df

def _apply_medprice( df):
    df["MEDPRICE"] = talib.MEDPRICE(df["high"], df["low"])
    return df

def _apply_typprice( df):
    df["TYPPRICE"] = talib.TYPPRICE(df["high"], df["low"], df["close"])
    return df

def _apply_wclprice( df):
    df["WCLPRICE"] = talib.WCLPRICE(df["high"], df["low"], df["close"])
    return df

# === Volatility Indicators ===
def _apply_atr( df,timeperiod=20):
    df["ATR_14"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod)
    return df

def _apply_natr( df,timeperiod=20):
    df["NATR_14"] = talib.NATR(df["high"], df["low"], df["close"], timeperiod)
    return df

def _apply_trange( df):
    df["TRANGE"] = talib.TRANGE(df["high"], df["low"], df["close"])
    return df
def _apply_adx( df,timeperiod=20):
    df["ADX_14"] = talib.ADX(df["high"], df["low"], df["close"], timeperiod)
    return df

def _apply_adxr( df,timeperiod=20):
    df["ADXR_14"] = talib.ADXR(df["high"], df["low"], df["close"], timeperiod)
    return df

def _apply_apo( df):
    df["APO"] = talib.APO(df["close"], fastperiod=12, slowperiod=26, matype=0)
    return df

def _apply_aroon( df,timeperiod=20):
    aroon_down, aroon_up = talib.AROON(df["high"], df["low"], timeperiod)
    df["AROON_down"] = aroon_down
    df["AROON_up"] = aroon_up
    return df

def _apply_aroonosc( df,timeperiod=20):
    df["AROONOSC"] = talib.AROONOSC(df["high"], df["low"], timeperiod)
    return df

def _apply_bop( df):
    df["BOP"] = talib.BOP(df["open"], df["high"], df["low"], df["close"])
    return df

def _apply_cci( df,timeperiod=20):
    df["CCI_14"] = talib.CCI(df["high"], df["low"], df["close"], timeperiod)
    return df

def _apply_cmo( df,timeperiod=20):
    df["CMO_14"] = talib.CMO(df["close"], timeperiod)
    return df

def _apply_dx( df,timeperiod=20):
    df["DX_14"] = talib.DX(df["high"], df["low"], df["close"], timeperiod)
    return df

def _apply_macd( df):
    macd, signal, hist = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["MACD"] = macd
    df["MACD_signal"] = signal
    df["MACD_hist"] = hist
    return df

def _apply_macdext( df):
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

def _apply_macdfix( df):
    macd, signal, hist = talib.MACDFIX(df["close"], signalperiod=9)
    df["MACDFIX"] = macd
    df["MACDFIX_signal"] = signal
    df["MACDFIX_hist"] = hist
    return df

def apply_cdl2crows(df):
    return df.assign(CDL2CROWS=talib.CDL2CROWS(df["open"], df["high"], df["low"], df["close"]))

def apply_cdl3blackcrows(df):
    return df.assign(CDL3BLACKCROWS=talib.CDL3BLACKCROWS(df["open"], df["high"], df["low"], df["close"]))

def apply_cdl3inside(df):
    return df.assign(CDL3INSIDE=talib.CDL3INSIDE(df["open"], df["high"], df["low"], df["close"]))

def apply_cdl3linestrike(df):
    return df.assign(CDL3LINESTRIKE=talib.CDL3LINESTRIKE(df["open"], df["high"], df["low"], df["close"]))

def apply_cdl3outside(df):
    return df.assign(CDL3OUTSIDE=talib.CDL3OUTSIDE(df["open"], df["high"], df["low"], df["close"]))

def apply_cdl3starsinsouth(df):
    return df.assign(CDL3STARSINSOUTH=talib.CDL3STARSINSOUTH(df["open"], df["high"], df["low"], df["close"]))

def apply_cdl3whitesoldiers(df):
    return df.assign(CDL3WHITESOLDIERS=talib.CDL3WHITESOLDIERS(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlabandonedbaby(df):
    return df.assign(CDLABANDONEDBABY=talib.CDLABANDONEDBABY(df["open"], df["high"], df["low"], df["close"]))

def apply_cdladvanceblock(df):
    return df.assign(CDLADVANCEBLOCK=talib.CDLADVANCEBLOCK(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlbelthold(df):
    return df.assign(CDLBELTHOLD=talib.CDLBELTHOLD(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlbreakaway(df):
    return df.assign(CDLBREAKAWAY=talib.CDLBREAKAWAY(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlclosingmarubozu(df):
    return df.assign(CDLCLOSINGMARUBOZU=talib.CDLCLOSINGMARUBOZU(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlconcealbabyswall(df):
    return df.assign(CDLCONCEALBABYSWALL=talib.CDLCONCEALBABYSWALL(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlcounterattack(df):
    return df.assign(CDLCOUNTERATTACK=talib.CDLCOUNTERATTACK(df["open"], df["high"], df["low"], df["close"]))

def apply_cdldarkcloudcover(df):
    return df.assign(CDLDARKCLOUDCOVER=talib.CDLDARKCLOUDCOVER(df["open"], df["high"], df["low"], df["close"]))

def apply_cdldoji(df):
    return df.assign(CDLDOJI=talib.CDLDOJI(df["open"], df["high"], df["low"], df["close"]))

def apply_cdldojistar(df):
    return df.assign(CDLDOJISTAR=talib.CDLDOJISTAR(df["open"], df["high"], df["low"], df["close"]))

def apply_cdldragonflydoji(df):
    return df.assign(CDLDRAGONFLYDOJI=talib.CDLDRAGONFLYDOJI(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlengulfing(df):
    return df.assign(CDLENGULFING=talib.CDLENGULFING(df["open"], df["high"], df["low"], df["close"]))

def apply_cdleveningdojistar(df):
    return df.assign(CDLEVENINGDOJISTAR=talib.CDLEVENINGDOJISTAR(df["open"], df["high"], df["low"], df["close"]))

def apply_cdleveningstar(df):
    return df.assign(CDLEVENINGSTAR=talib.CDLEVENINGSTAR(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlgapsidesidewhite(df):
    return df.assign(CDLGAPSIDESIDEWHITE=talib.CDLGAPSIDESIDEWHITE(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlgravestonedoji(df):
    return df.assign(CDLGRAVESTONEDOJI=talib.CDLGRAVESTONEDOJI(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlhammer(df):
    return df.assign(CDLHAMMER=talib.CDLHAMMER(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlhangingman(df):
    return df.assign(CDLHANGINGMAN=talib.CDLHANGINGMAN(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlharami(df):
    return df.assign(CDLHARAMI=talib.CDLHARAMI(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlharamicross(df):
    return df.assign(CDLHARAMICROSS=talib.CDLHARAMICROSS(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlhighwave(df):
    return df.assign(CDLHIGHWAVE=talib.CDLHIGHWAVE(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlhikkake(df):
    return df.assign(CDLHIKKAKE=talib.CDLHIKKAKE(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlhikkakemod(df):
    return df.assign(CDLHIKKAKEMOD=talib.CDLHIKKAKEMOD(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlhomingpigeon(df):
    return df.assign(CDLHOMINGPIGEON=talib.CDLHOMINGPIGEON(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlidentical3crows(df):
    return df.assign(CDLIDENTICAL3CROWS=talib.CDLIDENTICAL3CROWS(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlinneck(df):
    return df.assign(CDLINNECK=talib.CDLINNECK(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlinvertedhammer(df):
    return df.assign(CDLINVERTEDHAMMER=talib.CDLINVERTEDHAMMER(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlkicking(df):
    return df.assign(CDLKICKING=talib.CDLKICKING(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlkickingbylength(df):
    return df.assign(CDLKICKINGBYLENGTH=talib.CDLKICKINGBYLENGTH(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlladderbottom(df):
    return df.assign(CDLLADDERBOTTOM=talib.CDLLADDERBOTTOM(df["open"], df["high"], df["low"], df["close"]))

def apply_cdllongleggeddoji(df):
    return df.assign(CDLLONGLEGGEDDOJI=talib.CDLLONGLEGGEDDOJI(df["open"], df["high"], df["low"], df["close"]))

def apply_cdllongline(df):
    return df.assign(CDLLONGLINE=talib.CDLLONGLINE(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlmarubozu(df):
    return df.assign(CDLMARUBOZU=talib.CDLMARUBOZU(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlmatchinglow(df):
    return df.assign(CDLMATCHINGLOW=talib.CDLMATCHINGLOW(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlmathold(df):
    return df.assign(CDLMATHOLD=talib.CDLMATHOLD(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlmorningdojistar(df):
    return df.assign(CDLMORNINGDOJISTAR=talib.CDLMORNINGDOJISTAR(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlmorningstar(df):
    return df.assign(CDLMORNINGSTAR=talib.CDLMORNINGSTAR(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlonneck(df):
    return df.assign(CDLONNECK=talib.CDLONNECK(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlpiercing(df):
    return df.assign(CDLPIERCING=talib.CDLPIERCING(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlrickshawman(df):
    return df.assign(CDLRICKSHAWMAN=talib.CDLRICKSHAWMAN(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlrisefall3methods(df):
    return df.assign(CDLRISEFALL3METHODS=talib.CDLRISEFALL3METHODS(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlseparatinglines(df):
    return df.assign(CDLSEPARATINGLINES=talib.CDLSEPARATINGLINES(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlshootingstar(df):
    return df.assign(CDLSHOOTINGSTAR=talib.CDLSHOOTINGSTAR(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlshortline(df):
    return df.assign(CDLSHORTLINE=talib.CDLSHORTLINE(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlspinningtop(df):
    return df.assign(CDLSPINNINGTOP=talib.CDLSPINNINGTOP(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlstalledpattern(df):
    return df.assign(CDLSTALLEDPATTERN=talib.CDLSTALLEDPATTERN(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlsticksandwich(df):
    return df.assign(CDLSTICKSANDWICH=talib.CDLSTICKSANDWICH(df["open"], df["high"], df["low"], df["close"]))

def apply_cdltakuri(df):
    return df.assign(CDLTAKURI=talib.CDLTAKURI(df["open"], df["high"], df["low"], df["close"]))

def apply_cdltasukigap(df):
    return df.assign(CDLTASUKIGAP=talib.CDLTASUKIGAP(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlthrusting(df):
    return df.assign(CDLTHRUSTING=talib.CDLTHRUSTING(df["open"], df["high"], df["low"], df["close"]))

def apply_cdltristar(df):
    return df.assign(CDLTRISTAR=talib.CDLTRISTAR(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlunique3river(df):
    return df.assign(CDLUNIQUE3RIVER=talib.CDLUNIQUE3RIVER(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlupsidegap2crows(df):
    return df.assign(CDLUPSIDEGAP2CROWS=talib.CDLUPSIDEGAP2CROWS(df["open"], df["high"], df["low"], df["close"]))

def apply_cdlxsidegap3methods(df):
    return df.assign(CDLXSIDEGAP3METHODS=talib.CDLXSIDEGAP3METHODS(df["open"], df["high"], df["low"], df["close"]))
def apply_beta(df):
    df["BETA"] = talib.BETA(df["high"], df["low"])
    return df

def apply_correl(df):
    df["CORREL"] = talib.CORREL(df["high"], df["low"])
    return df

def apply_linearreg(df):
    df["LINEARREG"] = talib.LINEARREG(df["close"])
    return df

def apply_linearreg_angle(df):
    df["LINEARREG_ANGLE"] = talib.LINEARREG_ANGLE(df["close"])
    return df

def apply_linearreg_intercept(df):
    df["LINEARREG_INTERCEPT"] = talib.LINEARREG_INTERCEPT(df["close"])
    return df

def apply_linearreg_slope(df):
    df["LINEARREG_SLOPE"] = talib.LINEARREG_SLOPE(df["close"])
    return df

def apply_stddev(df):
    df["STDDEV"] = talib.STDDEV(df["close"])
    return df

def apply_tsf(df):
    df["TSF"] = talib.TSF(df["close"])
    return df

def apply_var(df):
    df["VAR"] = talib.VAR(df["close"])
    return df
def apply_adosc(df):
    df["ADOSC"]=talib.ADOSC(df["high"], df["low"], df["close"], df["volume"])
    return df


def apply_obv(df):
     df["OBV"]=talib.OBV(df["close"], df["volume"])
     return df

indicator_map = {
            # === Overlap Studies ===
            "sma": _applysma,
            "ema": _apply_ema,
            "wma": _apply_wma,
            "dema": _apply_dema,
            "tema": _apply_tema,
            "trima": _apply_trima,
            "kama": _apply_kama,
            "mama": _apply_mama,
            "mavp": _apply_mavp,
            "t3": _apply_t3,
            "sar": _apply_sar,
            "sarext": _apply_sarext,
            "midpoint": _apply_midpoint,
            "midprice": _apply_midprice,
            "ht_trendline": _apply_ht_trendline,
            "bbands": _apply_bollinger_bands,

            # === Momentum Indicators ===
            "adx": _apply_adx,
            "adxr": _apply_adxr,
            "apo": _apply_apo,
            "aroon": _apply_aroon,
            "aroonosc": _apply_aroonosc,
            "bop": _apply_bop,
            "cci": _apply_cci,
            "cmo": _apply_cmo,
            "dx": _apply_dx,
            "macd": _apply_macd,
            "macdext": _apply_macdext,
            "macdfix": _apply_macdfix,
            "mfi": _apply_mfi,
            "minus_di": _apply_minus_di,
            "minus_dm": _apply_minus_dm,
            "mom": _apply_mom,
            # "plusdi": _apply_plus_di,
            "plus_dm": _apply_plus_dm,
            "ppo": _apply_ppo,
            "roc": _apply_roc,
            "rocp": _apply_rocp,
            "rocr": _apply_rocr,
            "rocr100": _apply_rocr100,
            "rsi": _apply_rsi,
            "stoch": _apply_stoch,
            "stochf": _apply_stochf,
            "stochrsi": _apply_stochrsi,
            "trix": _apply_trix,
            "ultosc": _apply_ultosc,
            "willr": _apply_willr,
            "ad":_apply_ad,
            "adosc":apply_adosc,
            "obv": apply_obv,

            
            "ht_dcperiod": _apply_ht_dcperiod,
            "ht_dcphase": _apply_ht_dcphase,
            "ht_phasor": _apply_ht_phasor,
            "ht_sine": _apply_ht_sine,
            "ht_trendmode": _apply_ht_trendmode,
            # === Price Transform Indicators ===
            "avgprice": _apply_avgprice,
            "medprice": _apply_medprice,
            "typprice": _apply_typprice,
            "wclprice": _apply_wclprice,

            # === Volatility Indicators ===
            "atr": _apply_atr,
            "natr": _apply_natr,
            "trange": _apply_trange,


            
            
             # --- Pattern Recognition ---
             "doji_pattern":_apply_doji_pattern,
            "cdl2crows": apply_cdl2crows,
            "cdl3blackcrows": apply_cdl3blackcrows,
            "cdl3inside": apply_cdl3inside,
            "cdl3linestrike": apply_cdl3linestrike,
            "cdl3outside": apply_cdl3outside,
            "cdl3starsinsouth": apply_cdl3starsinsouth,
            "cdl3whitesoldiers": apply_cdl3whitesoldiers,
            "cdlabandonedbaby": apply_cdlabandonedbaby,
            "cdladvanceblock": apply_cdladvanceblock,
            "cdlbelthold": apply_cdlbelthold,
            "cdlbreakaway": apply_cdlbreakaway,
            "cdlclosingmarubozu": apply_cdlclosingmarubozu,
            "cdlconcealbabyswall": apply_cdlconcealbabyswall,
            "cdlcounterattack": apply_cdlcounterattack,
            "cdldarkcloudcover": apply_cdldarkcloudcover,
            "cdldoji": apply_cdldoji,
            "cdldojistar": apply_cdldojistar,
            "cdldragonflydoji": apply_cdldragonflydoji,
            "cdlengulfing": apply_cdlengulfing,
            "cdleveningdojistar": apply_cdleveningdojistar,
            "cdleveningstar": apply_cdleveningstar,
            "cdlgapsidesidewhite": apply_cdlgapsidesidewhite,
            "cdlgravestonedoji": apply_cdlgravestonedoji,
            "cdlhammer": apply_cdlhammer,
            "cdlhangingman": apply_cdlhangingman,
            "cdlharami": apply_cdlharami,
            "cdlharamicross": apply_cdlharamicross,
            "cdlhighwave": apply_cdlhighwave,
            "cdlhikkake": apply_cdlhikkake,
            "cdlhikkakemod": apply_cdlhikkakemod,
            "cdlhomingpigeon": apply_cdlhomingpigeon,
            "cdlidentical3crows": apply_cdlidentical3crows,
            "cdlinneck": apply_cdlinneck,
            "cdlinvertedhammer": apply_cdlinvertedhammer,
            "cdlkicking": apply_cdlkicking,
            "cdlkickingbylength": apply_cdlkickingbylength,
            "cdlladderbottom": apply_cdlladderbottom,
            "cdllongleggeddoji": apply_cdllongleggeddoji,
            "cdllongline": apply_cdllongline,
            "cdlmarubozu": apply_cdlmarubozu,
            "cdlmatchinglow": apply_cdlmatchinglow,
            "cdlmathold": apply_cdlmathold,
            "cdlmorningdojistar": apply_cdlmorningdojistar,
            "cdlmorningstar": apply_cdlmorningstar,
            "cdlonneck": apply_cdlonneck,
            "cdlpiercing": apply_cdlpiercing,
            "cdlrickshawman": apply_cdlrickshawman,
            "cdlrisefall3methods": apply_cdlrisefall3methods,
            "cdlseparatinglines": apply_cdlseparatinglines,
            "cdlshootingstar": apply_cdlshootingstar,
            "cdlshortline": apply_cdlshortline,
            "cdlspinningtop": apply_cdlspinningtop,
            "cdlstalledpattern": apply_cdlstalledpattern,
            "cdlsticksandwich": apply_cdlsticksandwich,
            "cdltakuri": apply_cdltakuri,
            "cdltasukigap": apply_cdltasukigap,
            "cdlthrusting": apply_cdlthrusting,
            "cdltristar": apply_cdltristar,
            "cdlunique3river": apply_cdlunique3river,
            "cdlupsidegap2crows": apply_cdlupsidegap2crows,
            "cdlxsidegap3methods": apply_cdlxsidegap3methods,
                         # Statistic Functions
             "beta": apply_beta,
            "correl": apply_correl,
            "linearreg": apply_linearreg,
            "linearreg_angle": apply_linearreg_angle,
            "linearreg_intercept": apply_linearreg_intercept,
            "linearreg_slope": apply_linearreg_slope,
            "stddev": apply_stddev,
            "tsf": apply_tsf,
            "var": apply_var,
        }

timeperiods = {
    "adx": 20,
    "adxr": 20,
    "aroon": 20,
    "aroonosc": 20,
    "atr": 20,
    "bbands": 20,
    "cci": 20,
    "cmo": 20,
    "correl": 20,
    "dema": 20,
    "dx": 20,
    "ema": 20,
    "kama": 20,
    "linearreg": 20,
    "linearreg_angle": 20,
    "linearreg_intercept": 20,
    "linearreg_slope": 20,
    "mfi": 20,
    "MidPoint": 20,
    "MidPrice": 20,
    "minus_di": 20,
    "minus_dm": 20,
    "mom": 20,
    "natr": 20,
    "plus_di": 20,
    "plus_dm": 20,
    "roc": 20,
    "rocp": 20,
    "rocr": 20,
    "rocr100": 20,
    "rsi": 20,
    "sma": 20,
    "stddev": 20,
    "stochrsi": 20,
    "sum": 20,
    "t3": 20,
    "tema": 20,
    "trima": 20,
    "trix": 20,
    "tsf": 20,
    "tar": 20,
    "willr": 20,
    "wma": 20,
}


