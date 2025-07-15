import configparser
from binance_fetcher import binance_fetcher


def load_config(path='config_binance.ini'):
    config = configparser.ConfigParser()
    config.read(path)
    return config

if __name__ == '__main__':
    config = load_config()
    config_section = config['data']
    fetcher=binance_fetcher(config_section)
    fetcher.save_all_symbols()
    
