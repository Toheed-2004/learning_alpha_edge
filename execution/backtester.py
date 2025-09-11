#!/usr/bin/env python3
"""
bybit_live_executor.py

Enhanced Bybit live executor with historical gap filling for accurate performance evaluation.
Handles downtime by reconstructing missed trades using historical OHLCV data.

Features:
- Live signal processing with real order execution
- Historical gap filling for missed signals during downtime
- Hybrid simulation/live mode for accurate performance tracking
- Comprehensive state management and reconciliation
- Configurable trade sizing and risk management

Environment variables:
  DATABASE_URL, BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_TESTNET
  EXECUTOR_POLL_INTERVAL, TRADE_USDT, TRADE_RS, DEFAULT_TP, DEFAULT_SL
  STRATEGY_TIMEFRAME (e.g., '1h', '4h') - MUST be set to match signal interval
"""

import os
import time
import logging
import uuid
import math
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

import ccxt
import pandas as pd
from sqlalchemy import create_engine, text, Table, Column, Integer, String, Float, DateTime, MetaData, Boolean
from sqlalchemy.exc import OperationalError, SQLAlchemyError

# ---------------------------
# Configuration & Logging
# ---------------------------
LOG = logging.getLogger("BybitExecutor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATABASE_URL = os.getenv("DATABASE_URL")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
BYBIT_TESTNET = os.getenv("BYBIT_TESTNET", "0") == "1"

POLL_INTERVAL = float(os.getenv("EXECUTOR_POLL_INTERVAL", "5"))
TRADE_USDT = os.getenv("TRADE_USDT")
TRADE_RS = float(os.getenv("TRADE_RS", "0.01"))
DEFAULT_TP = float(os.getenv("DEFAULT_TP", "0.05"))
DEFAULT_SL = float(os.getenv("DEFAULT_SL", "0.03"))
STRATEGY_TIMEFRAME = os.getenv("STRATEGY_TIMEFRAME", "1h")  # Must match signal interval

# Signals source
SIGNALS_TABLE = None
SIGNALS_SCHEMA = None

# ---------------------------
# DB helper / Schema
# ---------------------------
def get_db_engine():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    return create_engine(DATABASE_URL, future=True)

def init_db(engine):
    """Create required tables if missing"""
    meta = MetaData()
    schema = "execution"

    orders = Table('orders', meta,
                   Column('id', Integer, primary_key=True, autoincrement=True),
                   Column('client_order_id', String, unique=True, nullable=False),
                   Column('exchange_order_id', String),
                   Column('strategy_name', String),
                   Column('symbol', String),
                   Column('side', String),
                   Column('qty', Float),
                   Column('price', Float),
                   Column('status', String),
                   Column('type', String),
                   Column('created_at', DateTime(timezone=True)),
                   Column('raw', String),
                   Column('simulated', Boolean, default=False),
                   schema=schema)

    ledger = Table('ledger', meta,
                   Column('id', Integer, primary_key=True, autoincrement=True),
                   Column('datetime', DateTime(timezone=True)),
                   Column('strategy_name', String),
                   Column('symbol', String),
                   Column('action', String),
                   Column('price', Float),
                   Column('qty', Float),
                   Column('pnl', Float, nullable=True),
                   Column('notes', String, nullable=True),
                   Column('simulated', Boolean, default=False),
                   schema=schema)

    state = Table('state', meta,
                  Column('key', String, primary_key=True),
                  Column('value', String),
                  schema=schema)

    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema};"))
        meta.create_all(conn)
    LOG.info("Database schema initialized")

# ---------------------------
# Exchange wrapper
# ---------------------------
class BybitClient:
    def __init__(self, api_key, api_secret, testnet=False, max_retries=3, retry_delay=1.0):
        self.testnet = testnet
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api_key = api_key
        self.api_secret = api_secret
        self._init_ccxt()

    def _init_ccxt(self):
        params = {'apiKey': self.api_key, 'secret': self.api_secret, 'enableRateLimit': True}
        self.ex = ccxt.bybit(params)
        if self.testnet:
            try:
                self.ex.set_sandbox_mode(True)
            except Exception:
                LOG.warning("Couldn't set sandbox mode; ensure ccxt supports testnet")

    def _call(self, fn, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return fn(*args, **kwargs)
            except ccxt.NetworkError as e:
                LOG.warning("NetworkError: %s (attempt %s/%s)", e, attempt+1, self.max_retries)
            except ccxt.ExchangeError as e:
                LOG.error("ExchangeError: %s", e)
                raise
            except Exception as e:
                LOG.exception("Unexpected error: %s", e)
                raise
            time.sleep(self.retry_delay * (1 + attempt))
        raise RuntimeError("Max retries exceeded")

    def fetch_balance(self):
        return self._call(self.ex.fetch_balance)

    def fetch_ticker(self, symbol):
        return self._call(self.ex.fetch_ticker, symbol)

    def fetch_open_orders(self, symbol=None):
        return self._call(self.ex.fetch_open_orders, symbol)

    def fetch_my_trades(self, symbol=None, since=None, limit=500):
        return self._call(self.ex.fetch_my_trades, symbol, since, limit)

    def fetch_positions(self, symbols=None):
        try:
            return self._call(self.ex.fetch_positions, symbols)
        except Exception:
            LOG.info("fetch_positions not supported")
            return []

    def fetch_ohlcv(self, symbol, timeframe='1h', since=None, limit=100):
        return self._call(self.ex.fetch_ohlcv, symbol, timeframe, since, limit)

    def create_market_order(self, symbol, side, amount, params=None):
        return self._call(self.ex.create_order, symbol, 'market', side, amount, None, params or {})

    def create_limit_order(self, symbol, side, amount, price, params=None):
        return self._call(self.ex.create_order, symbol, 'limit', side, amount, price, params or {})

    def cancel_order(self, order_id, symbol=None):
        return self._call(self.ex.cancel_order, order_id, symbol)

    def create_stop_order(self, symbol, side, amount, stop_price, params=None):
        try:
            return self._call(self.ex.create_order, symbol, 'stop', side, amount, None, 
                             {'stopPrice': stop_price, **(params or {})})
        except Exception:
            LOG.warning("create_stop_order fallback")
            return self._call(self.ex.create_order, symbol, 'market', side, amount, None, params or {})

# ---------------------------
# Execution Engine
# ---------------------------
class BybitExecutor:
    def __init__(self, db_engine, bybit_client: BybitClient, signals_table: str, 
                 signals_schema: str = 'public', tp=DEFAULT_TP, sl=DEFAULT_SL, 
                 trade_usdt=None, trade_rs=TRADE_RS, poll_interval=POLL_INTERVAL,
                 strategy_timeframe=STRATEGY_TIMEFRAME):
        self.engine = db_engine
        self.client = bybit_client
        self.tp = tp
        self.sl = sl
        self.trade_usdt = float(trade_usdt) if trade_usdt else None
        self.trade_rs = trade_rs
        self.poll_interval = poll_interval
        self.signals_table = signals_table
        self.signals_schema = signals_schema
        self.strategy_timeframe = strategy_timeframe
        self.last_signal_ts = None
        self.open_positions = {}
        self.virtual_positions = {}  # For simulated trades during downtime
        self._ensure_tables()

    def _ensure_tables(self):
        init_db(self.engine)

    # DB helpers
    def _save_order_db(self, client_order_id, exchange_order_id, strategy_name, 
                      symbol, side, qty, price, status, order_type, raw, simulated=False):
        with self.engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO execution.orders(client_order_id, exchange_order_id, strategy_name, "
                "symbol, side, qty, price, status, type, created_at, raw, simulated) "
                "VALUES(:coid, :exid, :strat, :sym, :side, :qty, :price, :status, :type, :created_at, :raw, :simulated)"),
                {"coid": client_order_id, "exid": exchange_order_id, "strat": strategy_name, 
                 "sym": symbol, "side": side, "qty": qty, "price": price, "status": status, 
                 "type": order_type, "created_at": datetime.now(timezone.utc), "raw": str(raw),
                 "simulated": simulated}
            )

    def _save_ledger_db(self, dt, strategy_name, symbol, action, price, qty, 
                       pnl=None, notes=None, simulated=False):
        with self.engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO execution.ledger(datetime, strategy_name, symbol, action, "
                "price, qty, pnl, notes, simulated) "
                "VALUES(:dt, :strat, :sym, :action, :price, :qty, :pnl, :notes, :simulated)"),
                {"dt": dt, "strat": strategy_name, "sym": symbol, "action": action, 
                 "price": price, "qty": qty, "pnl": pnl, "notes": notes, "simulated": simulated}
            )

    def _get_last_signal_ts_db(self):
        with self.engine.begin() as conn:
            r = conn.execute(text("SELECT value FROM execution.state WHERE key='last_signal_ts';")).fetchone()
            if r:
                return datetime.fromisoformat(r[0])
            return None

    def _set_last_signal_ts_db(self, ts: datetime):
        with self.engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO execution.state(key, value) VALUES ('last_signal_ts', :v) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;"
            ), {"v": ts.isoformat()})

    # Historical price fetching for gap filling
    def _get_historical_price_for_interval(self, symbol: str, signal_dt: datetime) -> Dict[str, float]:
        """Get OHLC data for the specific interval containing the signal timestamp"""
        since_timestamp = int(signal_dt.timestamp() * 1000)
        
        try:
            ohlcv = self.client.fetch_ohlcv(symbol, self.strategy_timeframe, 
                                           since=since_timestamp, limit=1)
            if ohlcv:
                candle = ohlcv[0]
                return {
                    'timestamp': candle[0],
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'volume': candle[5]
                }
        except Exception as e:
            LOG.warning("Failed to fetch historical OHLCV for %s at %s: %s", 
                       symbol, signal_dt, e)
        
        # Fallback: use current price if historical data unavailable
        ticker = self.client.fetch_ticker(symbol)
        return {'close': float(ticker['last']), 'timestamp': int(time.time() * 1000)}

    # Sizing calculation
    def _compute_qty(self, symbol, price, simulated=False):
        """Compute quantity based on configured sizing method"""
        if self.trade_usdt:
            qty = max(0.0, float(self.trade_usdt) / price)
            return round(qty, 6)  # Basic rounding for demonstration
        
        # Risk sizing - for simulation, use last known equity
        if simulated:
            # For simulation, we might not have real-time balance
            # Use a fixed simulation equity or track it separately
            simulation_equity = 10000.0  # Default simulation equity
            usdt_risk = simulation_equity * self.trade_rs
        else:
            try:
                bal = self.client.fetch_balance()
                equity = float(bal.get('USDT', {}).get('total', 10000))
                usdt_risk = equity * self.trade_rs
            except Exception:
                usdt_risk = 10000 * self.trade_rs  # Fallback
        
        qty = max(0.0, usdt_risk / price)
        return round(qty, 6)

    # Reconciliation
    def reconcile(self):
        """Reconcile with exchange state and handle downtime gaps"""
        LOG.info("Starting reconciliation...")
        
        # 1. Load last processed signal timestamp
        self.last_signal_ts = self._get_last_signal_ts_db()
        LOG.info("Last processed signal: %s", self.last_signal_ts)

        # 2. Fill historical gaps for missed signals during downtime
        if self.last_signal_ts:
            self._fill_historical_gaps()

        # 3. Reconcile with actual exchange state
        self._reconcile_exchange_state()

        LOG.info("Reconciliation complete. Positions: %s", self.open_positions)

    def _fill_historical_gaps(self):
        """Process missed signals during downtime using historical data"""
        LOG.info("Checking for missed signals during downtime...")
        
        # Get all signals that occurred during downtime
        missed_signals = self._fetch_signals_in_range(self.last_signal_ts, datetime.now(timezone.utc))
        
        if not missed_signals:
            LOG.info("No missed signals found during downtime period")
            return

        LOG.info("Found %d missed signals, processing with historical data...", len(missed_signals))
        
        for signal in missed_signals:
            try:
                # Get historical price for the signal timestamp
                historical_data = self._get_historical_price_for_interval(
                    signal['symbol'], signal['datetime']
                )
                historical_price = historical_data['close']
                
                # Process the signal in simulation mode
                self._process_signal_simulated(signal, historical_price)
                
            except Exception as e:
                LOG.error("Failed to process missed signal %s: %s", signal, e)

    def _reconcile_exchange_state(self):
        """Reconcile with actual exchange positions and orders"""
        try:
            # Get real positions from exchange
            positions = self.client.fetch_positions()
            for p in positions:
                if isinstance(p, dict):
                    symbol = p.get('symbol')
                    size = float(p.get('positionAmt') or p.get('contracts') or 0)
                    if size != 0:
                        side = 'long' if size > 0 else 'short'
                        entry = float(p.get('entryPrice') or p.get('avgEntryPrice') or 0)
                        self.open_positions[symbol] = {
                            "side": side, 
                            "size": abs(size), 
                            "entry": entry,
                            "real": True  # Mark as real exchange position
                        }
        except Exception as e:
            LOG.warning("Could not fetch positions from exchange: %s", e)

        # Cancel any stale orders
        try:
            open_orders = self.client.fetch_open_orders()
            for order in open_orders:
                try:
                    self.client.cancel_order(order['id'], order['symbol'])
                    LOG.info("Cancelled stale order: %s", order['id'])
                except Exception as e:
                    LOG.warning("Failed to cancel order %s: %s", order['id'], e)
        except Exception as e:
            LOG.warning("Could not fetch open orders: %s", e)

    def _fetch_signals_in_range(self, start_dt: datetime, end_dt: datetime) -> List[Dict]:
        """Fetch signals within a specific time range"""
        if not start_dt:
            return []
            
        with self.engine.begin() as conn:
            result = conn.execute(text(
                f"SELECT strategy_name, symbol, datetime, final_signal "
                f"FROM {self.signals_schema}.{self.signals_table} "
                f"WHERE datetime > :start AND datetime <= :end "
                f"ORDER BY datetime ASC"
            ), {"start": start_dt, "end": end_dt})
            
            return [dict(zip(['strategy_name', 'symbol', 'datetime', 'final_signal'], row)) 
                   for row in result.fetchall()]

    # Signal processing
    def _process_signal_simulated(self, sig_row: Dict, fill_price: float):
        """Process a signal in simulation mode (for historical gap filling)"""
        strategy_name = sig_row['strategy_name']
        symbol = sig_row['symbol']
        sig_time = pd.to_datetime(sig_row['datetime']).to_pydatetime().astimezone(timezone.utc)
        signal = int(sig_row['final_signal'])
        
        LOG.info("Simulating signal: %s %s %s at price %s", 
                strategy_name, symbol, signal, fill_price)
        
        current_pos = self.virtual_positions.get(symbol)
        qty = self._compute_qty(symbol, fill_price, simulated=True)
        
        if signal == 1:  # BUY
            if current_pos and current_pos['side'] == 'long':
                return  # Already long
                
            if current_pos and current_pos['side'] == 'short':
                self._close_virtual_position(symbol, current_pos, fill_price, strategy_name)
                
            if qty > 0:
                self._save_ledger_db(sig_time, strategy_name, symbol, 'BUY', 
                                   fill_price, qty, notes="simulated_entry", simulated=True)
                self.virtual_positions[symbol] = {
                    'side': 'long', 'size': qty, 'entry': fill_price, 'real': False
                }
                
        elif signal == -1:  # SELL
            if current_pos and current_pos['side'] == 'short':
                return  # Already short
                
            if current_pos and current_pos['side'] == 'long':
                self._close_virtual_position(symbol, current_pos, fill_price, strategy_name)
                
            if qty > 0:
                self._save_ledger_db(sig_time, strategy_name, symbol, 'SELL', 
                                   fill_price, qty, notes="simulated_entry_short", simulated=True)
                self.virtual_positions[symbol] = {
                    'side': 'short', 'size': qty, 'entry': fill_price, 'real': False
                }
                
        elif signal == 0:  # CLOSE
            if current_pos:
                self._close_virtual_position(symbol, current_pos, fill_price, strategy_name)

    def _close_virtual_position(self, symbol: str, position: Dict, price: float, strategy_name: str):
        """Close a virtual/simulated position"""
        side = position['side']
        size = position['size']
        action = 'SELL' if side == 'long' else 'BUY'
        
        self._save_ledger_db(datetime.now(timezone.utc), strategy_name, symbol, 
                           f"CLOSE_{action}", price, size, notes="simulated_close", simulated=True)
        self.virtual_positions.pop(symbol, None)

    def _process_signal_live(self, sig_row: Dict):
        """Process a signal with live trading"""
        strategy_name = sig_row['strategy_name']
        symbol = sig_row['symbol']
        sig_time = pd.to_datetime(sig_row['datetime']).to_pydatetime().astimezone(timezone.utc)
        signal = int(sig_row['final_signal'])

        if self.last_signal_ts and sig_time <= self.last_signal_ts:
            return  # Already processed

        ticker = self.client.fetch_ticker(symbol)
        price = float(ticker['last'])
        current_pos = self.open_positions.get(symbol)

        if signal == 1:  # BUY
            if current_pos and current_pos['side'] == 'long':
                return
                
            if current_pos and current_pos['side'] == 'short':
                self._close_position(symbol, current_pos)

            qty = self._compute_qty(symbol, price)
            if qty <= 0:
                return
                
            coid = f"{strategy_name}-{symbol}-{int(sig_time.timestamp())}-{uuid.uuid4().hex[:8]}"
            try:
                order = self.client.create_market_order(symbol, 'buy', qty, 
                                                       params={"clientOrderId": coid})
                ex_id = order.get('id')
                self._save_order_db(coid, ex_id, strategy_name, symbol, 'buy', 
                                  qty, price, 'filled', 'market', order)
                self._save_ledger_db(sig_time, strategy_name, symbol, 'BUY', 
                                   price, qty, notes="live_entry")
                self.open_positions[symbol] = {
                    'side': 'long', 'size': qty, 'entry': price, 'real': True
                }
                self._place_tp_sl(symbol, 'long', qty, price, strategy_name)
            except Exception as e:
                LOG.error("Live BUY failed: %s", e)

        elif signal == -1:  # SELL
            if current_pos and current_pos['side'] == 'short':
                return
                
            if current_pos and current_pos['side'] == 'long':
                self._close_position(symbol, current_pos)

            qty = self._compute_qty(symbol, price)
            if qty <= 0:
                return
                
            coid = f"{strategy_name}-{symbol}-{int(sig_time.timestamp())}-{uuid.uuid4().hex[:8]}"
            try:
                order = self.client.create_market_order(symbol, 'sell', qty, 
                                                       params={"clientOrderId": coid})
                ex_id = order.get('id')
                self._save_order_db(coid, ex_id, strategy_name, symbol, 'sell', 
                                  qty, price, 'filled', 'market', order)
                self._save_ledger_db(sig_time, strategy_name, symbol, 'SELL', 
                                   price, qty, notes="live_entry_short")
                self.open_positions[symbol] = {
                    'side': 'short', 'size': qty, 'entry': price, 'real': True
                }
                self._place_tp_sl(symbol, 'short', qty, price, strategy_name)
            except Exception as e:
                LOG.error("Live SELL failed: %s", e)

        elif signal == 0:  # CLOSE
            if current_pos:
                self._close_position(symbol, current_pos)

        self.last_signal_ts = sig_time
        self._set_last_signal_ts_db(sig_time)

    def _close_position(self, symbol: str, position: Dict):
        """Close a live position"""
        side = position['side']
        size = position['size']
        close_side = 'sell' if side == 'long' else 'buy'
        
        try:
            order = self.client.create_market_order(symbol, close_side, size, 
                                                   params={"reduceOnly": True})
            fill_price = float(order.get('price') or order.get('info', {}).get('price') or 0)
            self._save_ledger_db(datetime.now(timezone.utc), "system", symbol, 
                               "CLOSE", fill_price, size, notes="live_close")
            self.open_positions.pop(symbol, None)
        except Exception as e:
            LOG.error("Position close failed: %s", e)

    def _place_tp_sl(self, symbol: str, side: str, qty: float, entry_price: float, strategy_name: str):
        """Place take-profit and stop-loss orders"""
        if side == 'long':
            tp_price = entry_price * (1 + self.tp)
            sl_price = entry_price * (1 - self.sl)
            tp_side, sl_side = 'sell', 'sell'
        else:
            tp_price = entry_price * (1 - self.tp)
            sl_price = entry_price * (1 + self.sl)
            tp_side, sl_side = 'buy', 'buy'

        try:
            # TP order
            coid_tp = f"{strategy_name}-{symbol}-TP-{uuid.uuid4().hex[:8]}"
            tp_order = self.client.create_limit_order(symbol, tp_side, qty, tp_price, 
                                                    params={"reduceOnly": True, "clientOrderId": coid_tp})
            self._save_order_db(coid_tp, tp_order.get('id'), strategy_name, symbol, 
                              tp_side, qty, tp_price, 'open', 'limit', tp_order)

            # SL order
            coid_sl = f"{strategy_name}-{symbol}-SL-{uuid.uuid4().hex[:8]}"
            sl_order = self.client.create_stop_order(symbol, sl_side, qty, sl_price, 
                                                   params={"reduceOnly": True, "clientOrderId": coid_sl})
            self._save_order_db(coid_sl, sl_order.get('id'), strategy_name, symbol, 
                              sl_side, qty, sl_price, 'open', 'stop', sl_order)
        except Exception as e:
            LOG.warning("TP/SL placement failed: %s", e)

    def _monitor_loop_once(self):
        """Monitor positions and enforce TP/SL if needed"""
        for symbol, pos in list(self.open_positions.items()):
            if not pos.get('real', True):
                continue  # Skip simulated positions
                
            try:
                ticker = self.client.fetch_ticker(symbol)
                last = float(ticker['last'])
                entry = pos['entry']
                
                if pos['side'] == 'long':
                    if last >= entry * (1 + self.tp) or last <= entry * (1 - self.sl):
                        self._close_position(symbol, pos)
                        action = "TP" if last >= entry * (1 + self.tp) else "SL"
                        self._save_ledger_db(datetime.now(timezone.utc), "system", 
                                           symbol, action, last, pos['size'], 
                                           notes=f"{action}_enforced")
                else:  # short
                    if last <= entry * (1 - self.tp) or last >= entry * (1 + self.sl):
                        self._close_position(symbol, pos)
                        action = "TP" if last <= entry * (1 - self.tp) else "SL"
                        self._save_ledger_db(datetime.now(timezone.utc), "system", 
                                           symbol, action, last, pos['size'], 
                                           notes=f"{action}_enforced_short")
            except Exception as e:
                LOG.error("Monitoring failed for %s: %s", symbol, e)

    def _fetch_new_signals(self):
        """Fetch new signals since last processing"""
        sql = f"SELECT strategy_name, symbol, datetime, final_signal FROM {self.signals_schema}.{self.signals_table} "
        params = {}
        
        if self.last_signal_ts:
            sql += "WHERE datetime > :since ORDER BY datetime ASC"
            params = {"since": self.last_signal_ts}
        else:
            sql += "ORDER BY datetime ASC LIMIT 100"
            
        with self.engine.begin() as conn:
            rows = conn.execute(text(sql), params).fetchall()
            
        return [{"strategy_name": r[0], "symbol": r[1], "datetime": r[2], "final_signal": r[3]} 
               for r in rows]

    def run(self):
        """Main execution loop"""
        LOG.info("Starting BybitExecutor with gap filling")
        self.reconcile()
        
        while True:
            try:
                # Process new live signals
                new_signals = self._fetch_new_signals()
                for sig in new_signals:
                    try:
                        self._process_signal_live(sig)
                    except Exception as e:
                        LOG.error("Live signal processing failed: %s", e)
                
                # Monitor for TP/SL
                self._monitor_loop_once()
                
            except Exception as e:
                LOG.error("Main loop error: %s", e)
                
            time.sleep(self.poll_interval)

# ---------------------------
# Entrypoint
# ---------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals-table", default=os.getenv("SIGNALS_TABLE") or "signals")
    parser.add_argument("--signals-schema", default=os.getenv("SIGNALS_SCHEMA") or "public")
    args = parser.parse_args()

    if not STRATEGY_TIMEFRAME:
        raise ValueError("STRATEGY_TIMEFRAME environment variable must be set (e.g., '1h', '4h')")

    engine = get_db_engine()
    client = BybitClient(BYBIT_API_KEY, BYBIT_API_SECRET, testnet=BYBIT_TESTNET)
    
    executor = BybitExecutor(
        db_engine=engine,
        bybit_client=client,
        signals_table=args.signals_table,
        signals_schema=args.signals_schema,
        tp=DEFAULT_TP,
        sl=DEFAULT_SL,
        trade_usdt=TRADE_USDT,
        trade_rs=TRADE_RS,
        poll_interval=POLL_INTERVAL,
        strategy_timeframe=STRATEGY_TIMEFRAME
    )
    
    executor.run()

if __name__ == "__main__":
    main()