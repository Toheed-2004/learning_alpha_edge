import configparser
import os
from learning_alpha_edge.data_updater.data_downloader import Data_Downloader
from learning_alpha_edge.signals.technical_indicators import TechnicalIndicatorApplier
from learning_alpha_edge.utils import db_utils

def get_enabled_indicators(config):
    enabled = []
    if "indicators" in config:
        for key, value in config["indicators"].items():
            try:
                if config.getboolean("indicators", key):
                    enabled.append(key)
            except ValueError:
                pass
    return enabled

def get_symbol_list(config):
    raw = config["data"].get("symbols", "")
    return [s.strip() for s in raw.split(",") if s.strip()]

def get_exchange_list(config):
    raw = config["data"].get("exchange", "")
    return [ex.strip().lower() for ex in raw.split(",") if ex.strip()]

if __name__ == "__main__":   
    config_path = os.path.join(os.path.dirname(__file__), "signals_config.ini")
    config = configparser.ConfigParser()
    config.read(config_path)

    # STEP 1: Read config
    symbols = get_symbol_list(config)
    exchanges = get_exchange_list(config)
    enabled_indicators = get_enabled_indicators(config)  

    # STEP 2: Download & update data
    symbol_data_map = {}

    for exchange in exchanges:
        for symbol in symbols:
            print(f"[INFO] Updating data for {symbol} on {exchange}...")
            downloader = Data_Downloader(symbol, exchange, "3min")
            full_df, resampled_df = downloader.get_data()
            indicator_applier=TechnicalIndicatorApplier(enabled_indicators)
            full_df=indicator_applier.apply(full_df)
            symbol_key = f"{exchange}_{symbol}"
            print(full_df.head())
            # full_df.to_csv(f"{symbol_key} indicators_data.csv",index=False)            
            # symbol_data_map[symbol_key] = {
            #     "full_df": full_df,
            #     "resampled_df": resampled_df
            # }
