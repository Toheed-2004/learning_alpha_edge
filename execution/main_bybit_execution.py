import asyncio
import logging
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import ccxt
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple
import time
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"

@dataclass
class Strategy:
    id: int
    name: str
    symbol: str
    time_horizon: str
    data_exchange: str
    indicators: Dict[str, bool]
    indicator_periods: Dict[str, int]
    is_active: bool = True

@dataclass
class Position:
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime

@dataclass
class TradeRecord:
    strategy_id: int
    datetime: datetime
    predicted_direction: str
    action: str
    buy_price: Optional[float]
    sell_price: Optional[float]
    quantity: float
    balance: float
    pnl: float
    pnl_sum: float
    fees: float
    order_id: Optional[str] = None

class BybitExecutor:
    def __init__(self, config: dict):
        """
        Initialize the Bybit execution system
        
        config should contain:
        - bybit_api_key, bybit_secret, bybit_testnet
        - db_config (host, database, user, password)
        - risk_management settings
        """
        self.config = config
        self.exchange = self._initialize_exchange()
        self.db_config = config['db_config']
        
        # Risk management parameters
        self.max_position_size = config.get('max_position_size', 0.1)  # % of balance
        self.stop_loss_pct = config.get('stop_loss_pct', 0.02)  # 2%
        self.take_profit_pct = config.get('take_profit_pct', 0.04)  # 4%
        self.max_daily_trades = config.get('max_daily_trades', 10)
        
        # Tracking variables
        self.active_positions: Dict[str, Position] = {}
        self.daily_trade_count: Dict[str, int] = {}
        self.strategy_balances: Dict[int, float] = {}
        self.running = False
        
    def _initialize_exchange(self) -> ccxt.bybit:
        """Initialize Bybit exchange connection"""
        try:
            exchange = ccxt.bybit({
                'apiKey': self.config['bybit_api_key'],
                'secret': self.config['bybit_secret'],
                'sandbox': self.config.get('bybit_testnet', True),
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'linear'  # USDT perpetual
                }
            })
            
            # Test connection
            balance = exchange.fetch_balance()
            logger.info(f"Successfully connected to Bybit. USDT Balance: {balance['USDT']['free']}")
            return exchange
            
        except Exception as e:
            logger.error(f"Failed to initialize Bybit exchange: {e}")
            raise

    def get_db_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    def load_strategies(self) -> List[Strategy]:
        """Load active strategies from database"""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT * FROM strategies WHERE is_active = true
                    """)
                    
                    strategies = []
                    for row in cursor.fetchall():
                        # Parse indicator columns (assuming boolean columns for each indicator)
                        indicators = {}
                        indicator_periods = {}
                        
                        for key, value in row.items():
                            if key.startswith('indicator_') and not key.endswith('_timeperiod'):
                                indicator_name = key.replace('indicator_', '')
                                indicators[indicator_name] = bool(value)
                                
                                # Check for corresponding timeperiod
                                period_key = f"{key}_timeperiod"
                                if period_key in row and row[period_key]:
                                    indicator_periods[indicator_name] = row[period_key]
                        
                        strategy = Strategy(
                            id=row['id'],
                            name=row['name'],
                            symbol=row['symbol'],
                            time_horizon=row['time_horizon'],
                            data_exchange=row['data_exchange'],
                            indicators=indicators,
                            indicator_periods=indicator_periods
                        )
                        strategies.append(strategy)
                        
                    logger.info(f"Loaded {len(strategies)} active strategies")
                    return strategies
                    
        except Exception as e:
            logger.error(f"Failed to load strategies: {e}")
            return []

    def get_market_data(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """Fetch market data from Bybit"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            df.drop('timestamp', axis=1, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch market data for {symbol}: {e}")
            return pd.DataFrame()

    def calculate_position_size(self, strategy_id: int, signal_strength: float = 1.0) -> float:
        """Calculate position size based on risk management rules"""
        try:
            balance = self.exchange.fetch_balance()
            available_balance = balance['USDT']['free']
            
            # Calculate position size as percentage of available balance
            base_size = available_balance * self.max_position_size * signal_strength
            
            # Additional risk checks can be added here
            return base_size
            
        except Exception as e:
            logger.error(f"Failed to calculate position size: {e}")
            return 0.0

    def place_market_order(self, symbol: str, side: str, amount: float, 
                          stop_loss: Optional[float] = None, 
                          take_profit: Optional[float] = None) -> Optional[dict]:
        """Place market order with optional SL/TP"""
        try:
            # Place main market order
            order = self.exchange.create_market_order(symbol, side, amount)
            logger.info(f"Market order placed: {order['id']} - {side} {amount} {symbol}")
            
            if order['status'] == 'closed' and (stop_loss or take_profit):
                # Place SL/TP orders after main order is filled
                filled_price = float(order['price'])
                
                if stop_loss:
                    sl_side = 'sell' if side == 'buy' else 'buy'
                    try:
                        sl_order = self.exchange.create_order(
                            symbol, 'stop_market', sl_side, amount,
                            None, None, {'stopPrice': stop_loss, 'reduceOnly': True}
                        )
                        logger.info(f"Stop loss placed: {sl_order['id']}")
                    except Exception as e:
                        logger.warning(f"Failed to place stop loss: {e}")
                
                if take_profit:
                    tp_side = 'sell' if side == 'buy' else 'buy'
                    try:
                        tp_order = self.exchange.create_limit_order(
                            symbol, tp_side, amount, take_profit,
                            None, {'reduceOnly': True}
                        )
                        logger.info(f"Take profit placed: {tp_order['id']}")
                    except Exception as e:
                        logger.warning(f"Failed to place take profit: {e}")
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            return None

    def calculate_sl_tp_prices(self, side: str, entry_price: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit prices"""
        if side.lower() == 'buy':
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
        else:  # sell
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 - self.take_profit_pct)
            
        return stop_loss, take_profit

    def execute_signal(self, strategy: Strategy, signal: str, current_price: float, 
                      signal_data: dict) -> Optional[TradeRecord]:
        """Execute trading signal"""
        try:
            symbol = strategy.symbol
            strategy_id = strategy.id
            
            # Check daily trade limit
            today = datetime.now().date()
            daily_key = f"{strategy_id}_{today}"
            if self.daily_trade_count.get(daily_key, 0) >= self.max_daily_trades:
                logger.warning(f"Daily trade limit reached for strategy {strategy_id}")
                return None
            
            # Check if we already have a position
            if symbol in self.active_positions:
                logger.info(f"Position already exists for {symbol}, skipping signal")
                return None
            
            if signal in [SignalType.BUY.value, SignalType.SELL.value]:
                # Calculate position size
                position_size = self.calculate_position_size(strategy_id)
                if position_size <= 0:
                    logger.warning("Position size is 0 or negative, skipping trade")
                    return None
                
                # Calculate SL/TP prices
                side = signal.lower()
                stop_loss, take_profit = self.calculate_sl_tp_prices(side, current_price)
                
                # Place order
                order = self.place_market_order(
                    symbol, side, position_size, stop_loss, take_profit
                )
                
                if order and order.get('status') == 'closed':
                    # Update daily trade count
                    self.daily_trade_count[daily_key] = self.daily_trade_count.get(daily_key, 0) + 1
                    
                    # Create trade record
                    filled_price = float(order['price'])
                    fees = float(order.get('fee', {}).get('cost', 0))
                    
                    # Get current balance
                    balance = self.exchange.fetch_balance()
                    current_balance = balance['USDT']['free']
                    
                    trade_record = TradeRecord(
                        strategy_id=strategy_id,
                        datetime=datetime.now(),
                        predicted_direction=signal,
                        action="OPEN_POSITION",
                        buy_price=filled_price if side == 'buy' else None,
                        sell_price=filled_price if side == 'sell' else None,
                        quantity=position_size,
                        balance=current_balance,
                        pnl=0.0,  # Will be calculated when position closes
                        pnl_sum=self.get_strategy_pnl_sum(strategy_id),
                        fees=fees,
                        order_id=order['id']
                    )
                    
                    # Update active positions
                    self.active_positions[symbol] = Position(
                        symbol=symbol,
                        side=side,
                        size=position_size,
                        entry_price=filled_price,
                        current_price=current_price,
                        unrealized_pnl=0.0,
                        realized_pnl=0.0,
                        timestamp=datetime.now()
                    )
                    
                    # Save trade record to database
                    self.save_trade_record(trade_record)
                    
                    logger.info(f"Successfully executed {signal} for {symbol} at {filled_price}")
                    return trade_record
                    
        except Exception as e:
            logger.error(f"Failed to execute signal: {e}")
            
        return None

    def save_trade_record(self, trade_record: TradeRecord):
        """Save trade record to database"""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO trade_ledger 
                        (strategy_id, datetime, predicted_direction, action, buy_price, 
                         sell_price, quantity, balance, pnl, pnl_sum, fees, order_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        trade_record.strategy_id,
                        trade_record.datetime,
                        trade_record.predicted_direction,
                        trade_record.action,
                        trade_record.buy_price,
                        trade_record.sell_price,
                        trade_record.quantity,
                        trade_record.balance,
                        trade_record.pnl,
                        trade_record.pnl_sum,
                        trade_record.fees,
                        trade_record.order_id
                    ))
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Failed to save trade record: {e}")

    def get_strategy_pnl_sum(self, strategy_id: int) -> float:
        """Get cumulative PnL for a strategy"""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT COALESCE(SUM(pnl), 0) as total_pnl 
                        FROM trade_ledger 
                        WHERE strategy_id = %s
                    """, (strategy_id,))
                    
                    result = cursor.fetchone()
                    return float(result[0]) if result else 0.0
                    
        except Exception as e:
            logger.error(f"Failed to get strategy PnL sum: {e}")
            return 0.0

    def update_positions(self):
        """Update active positions with current prices"""
        try:
            for symbol, position in self.active_positions.items():
                # Get current price
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = float(ticker['last'])
                
                # Calculate unrealized PnL
                if position.side == 'buy':
                    unrealized_pnl = (current_price - position.entry_price) * position.size
                else:
                    unrealized_pnl = (position.entry_price - current_price) * position.size
                
                position.current_price = current_price
                position.unrealized_pnl = unrealized_pnl
                
        except Exception as e:
            logger.error(f"Failed to update positions: {e}")

    def process_strategy(self, strategy: Strategy, apply_indicators_func, generate_signals_func):
        """Process a single strategy"""
        try:
            # Get market data
            df = self.get_market_data(strategy.symbol, strategy.time_horizon)
            if df.empty:
                logger.warning(f"No market data available for {strategy.symbol}")
                return
            
            # Get enabled indicators and their periods
            enabled_indicators = [k for k, v in strategy.indicators.items() if v]
            if not enabled_indicators:
                logger.warning(f"No indicators enabled for strategy {strategy.name}")
                return
            
            # Apply indicators
            df_with_indicators = apply_indicators_func(df, enabled_indicators, strategy.indicator_periods)
            
            # Generate signals
            df_with_signals = generate_signals_func(df_with_indicators)
            
            # Get the latest signal
            if len(df_with_signals) > 0:
                latest_row = df_with_signals.iloc[-1]
                current_signal = latest_row.get('signal', SignalType.HOLD.value)
                current_price = float(latest_row['close'])
                
                logger.info(f"Strategy {strategy.name} - Signal: {current_signal}, Price: {current_price}")
                
                # Execute signal if it's not HOLD
                if current_signal != SignalType.HOLD.value:
                    signal_data = {
                        'indicators': {col: latest_row[col] for col in df_with_indicators.columns if col not in ['open', 'high', 'low', 'close', 'volume']},
                        'price_data': {
                            'open': float(latest_row['open']),
                            'high': float(latest_row['high']),
                            'low': float(latest_row['low']),
                            'close': float(latest_row['close']),
                            'volume': float(latest_row['volume'])
                        }
                    }
                    
                    self.execute_signal(strategy, current_signal, current_price, signal_data)
                    
        except Exception as e:
            logger.error(f"Failed to process strategy {strategy.name}: {e}")

    async def run_trading_loop(self, apply_indicators_func, generate_signals_func, 
                              loop_interval: int = 60):
        """Main trading loop"""
        logger.info("Starting trading loop...")
        self.running = True
        
        while self.running:
            try:
                # Load current strategies
                strategies = self.load_strategies()
                
                # Update positions
                self.update_positions()
                
                # Process each strategy
                for strategy in strategies:
                    self.process_strategy(strategy, apply_indicators_func, generate_signals_func)
                    
                # Wait for next iteration
                await asyncio.sleep(loop_interval)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, stopping...")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying
                
        self.running = False
        logger.info("Trading loop stopped")

    def stop(self):
        """Stop the trading system"""
        logger.info("Stopping trading system...")
        self.running = False
        
        # Close any remaining positions (optional)
        # self.close_all_positions()

    def get_performance_summary(self, strategy_id: Optional[int] = None) -> dict:
        """Get performance summary"""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    where_clause = "WHERE strategy_id = %s" if strategy_id else ""
                    params = (strategy_id,) if strategy_id else ()
                    
                    cursor.execute(f"""
                        SELECT 
                            strategy_id,
                            COUNT(*) as total_trades,
                            SUM(pnl) as total_pnl,
                            AVG(pnl) as avg_pnl,
                            SUM(fees) as total_fees,
                            COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades,
                            COUNT(CASE WHEN pnl < 0 THEN 1 END) as losing_trades
                        FROM trade_ledger 
                        {where_clause}
                        GROUP BY strategy_id
                        ORDER BY strategy_id
                    """, params)
                    
                    results = cursor.fetchall()
                    
                    performance = {}
                    for row in results:
                        win_rate = (row['winning_trades'] / row['total_trades'] * 100) if row['total_trades'] > 0 else 0
                        performance[row['strategy_id']] = {
                            'total_trades': row['total_trades'],
                            'total_pnl': float(row['total_pnl']) if row['total_pnl'] else 0,
                            'avg_pnl': float(row['avg_pnl']) if row['avg_pnl'] else 0,
                            'total_fees': float(row['total_fees']) if row['total_fees'] else 0,
                            'winning_trades': row['winning_trades'],
                            'losing_trades': row['losing_trades'],
                            'win_rate': win_rate
                        }
                    
                    return performance
                    
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}


# Example usage and database schema
"""
Required database tables:

1. strategies table:
CREATE TABLE strategies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    time_horizon VARCHAR(10) NOT NULL,
    data_exchange VARCHAR(50) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    -- Add boolean columns for each TA-Lib indicator
    indicator_sma BOOLEAN DEFAULT false,
    indicator_sma_timeperiod INTEGER,
    indicator_ema BOOLEAN DEFAULT false,
    indicator_ema_timeperiod INTEGER,
    indicator_rsi BOOLEAN DEFAULT false,
    indicator_rsi_timeperiod INTEGER,
    indicator_macd BOOLEAN DEFAULT false,
    indicator_bollinger BOOLEAN DEFAULT false,
    indicator_bollinger_timeperiod INTEGER,
    -- Add more indicators as needed
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

2. trade_ledger table:
CREATE TABLE trade_ledger (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES strategies(id),
    datetime TIMESTAMP NOT NULL,
    predicted_direction VARCHAR(10) NOT NULL,
    action VARCHAR(20) NOT NULL,
    buy_price DECIMAL(20, 8),
    sell_price DECIMAL(20, 8),
    quantity DECIMAL(20, 8) NOT NULL,
    balance DECIMAL(20, 8) NOT NULL,
    pnl DECIMAL(20, 8) NOT NULL,
    pnl_sum DECIMAL(20, 8) NOT NULL,
    fees DECIMAL(20, 8) DEFAULT 0,
    order_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

Example usage:

# Configuration
# config = {
#     'bybit_api_key': 'your_api_key',
#     'bybit_secret': 'your_secret',
#     'bybit_testnet': True,  # Set to False for live trading
#     'db_config': {
#         'host': 'localhost',
#         'database': 'trading_db',
#         'user': 'postgres',
#         'password': 'password'
#     },
#     'max_position_size': 0.1,  # 10% of balance
#     'stop_loss_pct': 0.02,     # 2%
#     'take_profit_pct': 0.04,   # 4%
#     'max_daily_trades': 10
# }

# Initialize executor
executor = BybitExecutor(config)

# Run trading loop
import asyncio
asyncio.run(executor.run_trading_loop(apply_indicators, generate_signals, loop_interval=300))
"""