import pandas as pd
import numpy as np
import random
import datetime
import os
import inspect
import hashlib
import optuna
from learning_alpha_edge.backtest.main_backtest import Backtester
from learning_alpha_edge.utils.db_utils import *
from learning_alpha_edge.data.binance.main_binance import load_config
from learning_alpha_edge.data_updater.data_downloader import Data_Downloader
from learning_alpha_edge.signals.technical_indicators_signals.signals_generator import generate_signals
from learning_alpha_edge.technical_indicators.ti  import * 
from learning_alpha_edge.utils.data_utils import resample_ohlcv_data
# from learning_alpha_edge.backtest.main_backtest import Backtester 

# Randomly enable indicators (True/False)
def randomly_select_indicators(indicator_map):
    return [name for name in indicator_map if random.choice([True, False])]


# Apply only selected indicators
def apply_indicators(df:pd.DataFrame, randomly_selected_indicators, timeperiods:dict):
    df = df.copy()
    indicators_df = df[["open", "high", "low", "close", "volume"]].copy()
    existing_cols = set(indicators_df.columns)

    for name in randomly_selected_indicators:
        func = indicator_map.get(name)
        if func:
            sig = inspect.signature(func)
            params = sig.parameters

            if 'timeperiod' in params:
                timeperiod=timeperiods.get(name,20)
                result_df=func(df,timeperiod)
            else:
                result_df:pd.DataFrame = func(df)

            new_cols = set(result_df.columns) - existing_cols
            new_data = result_df[list(new_cols)]
            indicators_df = pd.concat([indicators_df, new_data], axis=1)
            existing_cols.update(new_cols)
        else:
            print(f"[WARN] No handler for indicator '{name}'")

    return indicators_df

def majority_vote(signals_df:pd.DataFrame):
    signal_columns = signals_df.drop(columns=["datetime", "open", "high", "low", "close", "volume"])

    # Count how many columns have each signal per row
    long_count = (signal_columns == 1).sum(axis=1)
    short_count = (signal_columns == -1).sum(axis=1)
    hold_count = (signal_columns == 0).sum(axis=1)

    # Assign signal based on majority
    majority_signal = pd.Series(0, index=signals_df.index)  # Default to hold
    majority_signal[long_count > short_count] = 1
    majority_signal[short_count > long_count] = -1

    # Add majority signal column
    signals_df["signal"] = majority_signal

    # Keep only datetime, OHLCV, and final_signal
    return signals_df[["datetime", "signal"]]

def sanitize_table_name(name):
    return name.lower().replace("-", "_").replace(".", "_").replace(" ", "_")

def save_strategy_to_db(engine, strategy_name, symbol:str, time_horizon, data_exchange, selected_indicators, all_indicators, timeperiods):
    row = {
        "name": strategy_name,
        "symbol": symbol.lower(),
        "time_horizon": time_horizon,
        "data_exchange": data_exchange,
    }

    for ind in all_indicators:
        is_enabled = ind in selected_indicators
        row[f'{ind}'] = is_enabled
        if ind in timeperiods and ind in selected_indicators:
            row[f'{ind}_timeperiod'] = timeperiods[ind]
        if ind in timeperiods and  not ind  in selected_indicators:
            row[f'{ind}_timeperiod']=None
        

    df = pd.DataFrame([row])
    save_to_db(df, engine, schema="public", table_name="config_strategies")



def save_signals(engine:Engine, strategy_name, signal_df:pd.DataFrame):
    # Add strategy name column
    save_to_db(signal_df, engine, schema="signals", table_name=strategy_name)

def objective (trial:optuna.Trial):
        for exchange in exchanges:
                
            for base_symbol in symbols_base:
                symbol = base_symbol + "USDT"
                # Fetch data
                # schema = exchange.lower()  # 'binance' or 'bybit'
                # downloader = Data_Downloader(symbol=symbol, exchange=exchange, resample_to=interval, engine=engine, schema=schema)
                # df, resampled_df = downloader.get_data()                               
                        
                selected_indicators =[]
                indicator_timeperiods={}
                df=pd.read_sql(f'SELECT * FROM "binance"."btc_1m" ORDER BY datetime', engine)
                df['datetime']=pd.to_datetime(df['datetime'])
                resampled_df=resample_ohlcv_data(df,'4h')
                for name, func in indicator_map.items():
                    use_indicator = trial.suggest_categorical(f"use_{name}", [True, False])
                    if use_indicator:
                        selected_indicators.append(name)

                        # If indicator has 'timeperiod' parameter, suggest it
                    if 'timeperiod' in inspect.signature(func).parameters:
                        indicator_timeperiods[name] = trial.suggest_int(
                            f"timeperiod_{name}", 5, 50
                        )
                raw_string = f"{exchange}_{symbol}_{interval}_{trial.number}_{datetime.datetime.now().isoformat()}"
                strategy_hash = hashlib.sha256(raw_string.encode()).hexdigest()[:8]  # shorten for readability
                strategy_name = f"strategy_{trial.number}_{strategy_hash}"               
                df_with_indicators = apply_indicators(resampled_df.copy(), selected_indicators,indicator_timeperiods)
                signal_df = generate_signals(df_with_indicators, selected_indicators)
                signal_df = majority_vote(signal_df)
                signal_df["datetime"] = signal_df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

                save_strategy_to_db(
                    engine, strategy_name, base_symbol, interval, exchange,
                    selected_indicators, all_indicators,indicator_timeperiods
                )

                save_signals(engine, strategy_name, signal_df[['datetime','signal']])
                signal_df=pd.read_sql(f'SELECT * FROM "signals"."{strategy_name}" ORDER BY datetime', engine)
                signal_df['datetime']=pd.to_datetime(signal_df['datetime'])
                backtester = Backtester(signal_df, df, start_balance=1000)
                ledger=backtester.run_backtest() 
                ledger.to_sql(strategy_name, engine, schema="ledger", if_exists="replace", index=False)
                pnl=ledger["PnLSum"].iloc[-1]
        return pnl
    


# Run the pipeline
if __name__ == "__main__":
    config_path=os.path.dirname(os.path.abspath(__file__))
    config_path=os.path.join(config_path,"signals_config.ini")
    config = load_config(config_path)
    print(config.sections())
    cfg = config['data']
    db_cfg = config['postgres']
    engine = get_pg_engine(
        user=db_cfg.get('user'),
        password=db_cfg.get('password'),
        host=db_cfg.get('host'),
        port=db_cfg.get('port'),
        dbname=db_cfg.get('dbname')
    )

    
    exchanges = [ex.strip().lower() for ex in cfg.get('exchanges').split(',')]
    symbols_base = [s.strip().upper() for s in cfg.get('symbols').split(',')]
    start_date = datetime.datetime.strptime(cfg.get('start_date'), "%Y-%m-%d")
    end_date = datetime.datetime.now() if cfg.get('end_date') == 'now' else datetime.datetime.strptime(cfg.get('end_date'), "%Y-%m-%d")
    interval = cfg.get('time_horizons', '1min')

    all_indicators = list(indicator_map.keys())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)           


                
    
