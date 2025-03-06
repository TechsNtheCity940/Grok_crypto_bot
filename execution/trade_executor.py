import ccxt
import os
from config import TRADING_PAIRS
from utils.log_setup import logger

class TradeExecutor:
    def __init__(self):
        self.exchange = ccxt.kraken({
            'apiKey': os.environ.get('KRAKEN_API_KEY'),
            'secret': os.environ.get('KRAKEN_API_SECRET'),
            'enableRateLimit': True
        })
        self.min_trade_sizes = {'DOGE/USD': 10.0, 'SHIB/USD': 1000000.0, 'XRP/USD': 1.0}

    def get_balance(self, symbol):
        try:
            # Force refresh of balance from exchange
            balance = self.exchange.fetch_balance({'recvWindow': 60000})
            logger.info(f"Raw Kraken balance response: {balance}")
            
            # Handle both 'USD' and 'ZUSD' formats for different exchange configurations
            usd = balance['total'].get('USD', balance['total'].get('ZUSD', 0.0))
            
            asset = symbol.split('/')[0]
            crypto = balance['total'].get(asset, 0.0)
            
            # Log the parsed balance
            logger.info(f"Parsed balance for {symbol}: USD={usd}, {asset}={crypto}")
            
            # Store the latest balance for tracking
            self._last_balance = {
                'USD': usd,
                asset: crypto,
                'timestamp': self.exchange.milliseconds()
            }
            
            return usd, crypto
        except Exception as e:
            logger.error(f"Error fetching balance for {symbol}: {e}")
            # If we have a previous balance, use that instead of returning zeros
            if hasattr(self, '_last_balance') and self._last_balance:
                asset = symbol.split('/')[0]
                logger.warning(f"Using cached balance due to error: USD={self._last_balance.get('USD', 0.0)}, {asset}={self._last_balance.get(asset, 0.0)}")
                return self._last_balance.get('USD', 0.0), self._last_balance.get(asset, 0.0)
            return 0.0, 0.0

    def fetch_current_price(self, symbol):
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return 0.0

    def execute(self, action, symbol, amount):
        try:
            # Get balance before trade
            pre_usd, pre_crypto = self.get_balance(symbol)
            current_price = self.fetch_current_price(symbol)
            pre_total_value = pre_usd + (pre_crypto * current_price)
            
            # Execute the trade
            if action == 1:  # Buy
                order = self.exchange.create_market_buy_order(symbol, amount)
                logger.info(f"Buy order executed: {order}")
                
                # Wait for order to settle
                self._wait_for_order_settlement()
                
                # Get balance after trade
                post_usd, post_crypto = self.get_balance(symbol)
                post_total_value = post_usd + (post_crypto * current_price)
                
                # Calculate actual trade impact
                usd_change = post_usd - pre_usd
                crypto_change = post_crypto - pre_crypto
                value_change = post_total_value - pre_total_value
                
                logger.info(f"Trade impact - USD change: {usd_change}, {symbol.split('/')[0]} change: {crypto_change}, Value change: {value_change}")
                
                # Log the trade to performance tracker if available
                self._log_trade_to_performance_tracker(symbol, action, amount, current_price, order)
                
                return order, False
                
            elif action == 2:  # Sell
                order = self.exchange.create_market_sell_order(symbol, amount)
                logger.info(f"Sell order executed: {order}")
                
                # Wait for order to settle
                self._wait_for_order_settlement()
                
                # Get balance after trade
                post_usd, post_crypto = self.get_balance(symbol)
                post_total_value = post_usd + (post_crypto * current_price)
                
                # Calculate actual trade impact
                usd_change = post_usd - pre_usd
                crypto_change = post_crypto - pre_crypto
                value_change = post_total_value - pre_total_value
                
                logger.info(f"Trade impact - USD change: {usd_change}, {symbol.split('/')[0]} change: {crypto_change}, Value change: {value_change}")
                
                # Log the trade to performance tracker if available
                self._log_trade_to_performance_tracker(symbol, action, amount, current_price, order)
                
                return order, False
        except ccxt.InsufficientFunds as e:
            logger.error(f"Insufficient funds for {symbol}: {e}")
            return None, True
        except Exception as e:
            logger.error(f"Execution error for {symbol}: {e}")
            return None, True

    def _wait_for_order_settlement(self, wait_time_ms=2000):
        """Wait for order to settle on the exchange"""
        import time
        time.sleep(wait_time_ms / 1000)  # Convert ms to seconds
    
    def _log_trade_to_performance_tracker(self, symbol, action, amount, price, order):
        """Log trade to performance tracker if available"""
        try:
            # Import here to avoid circular imports
            from monitoring.performance_tracker import PerformanceTracker
            
            # Try to get the performance tracker instance
            import sys
            import os
            sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
            
            # Check if we have a performance tracker in the main module
            import inspect
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals:
                    instance = frame.frame.f_locals['self']
                    if hasattr(instance, 'performance_tracker') and instance.performance_tracker:
                        tracker = instance.performance_tracker
                        action_str = 'buy' if action == 1 else 'sell'
                        tracker.log_trade(symbol, action, amount, price, strategy='executed')
                        logger.info(f"Trade logged to performance tracker: {action_str} {amount} {symbol} @ {price}")
                        return
            
            logger.warning("Could not find performance tracker instance to log trade")
        except Exception as e:
            logger.error(f"Error logging trade to performance tracker: {e}")
    
    def update_trading_pairs(self):
        try:
            markets = self.exchange.load_markets()
            return [pair for pair in markets.keys() if pair.endswith('/USD')]
        except Exception as e:
            logger.error(f"Error updating trading pairs: {e}")
            return TRADING_PAIRS
            
    def get_active_orders(self, symbol=None):
        """Get active orders for a symbol or all symbols"""
        try:
            params = {}
            if symbol:
                params['symbol'] = symbol
            
            orders = self.exchange.fetch_open_orders(symbol, params=params)
            logger.info(f"Active orders for {symbol or 'all symbols'}: {len(orders)}")
            return orders
        except Exception as e:
            logger.error(f"Error fetching active orders: {e}")
            return []
    
    def get_order_history(self, symbol=None, limit=50):
        """Get order history for a symbol or all symbols"""
        try:
            params = {'limit': limit}
            if symbol:
                params['symbol'] = symbol
            
            orders = self.exchange.fetch_closed_orders(symbol, params=params)
            logger.info(f"Order history for {symbol or 'all symbols'}: {len(orders)}")
            return orders
        except Exception as e:
            logger.error(f"Error fetching order history: {e}")
            return []
    
    def get_account_summary(self):
        """Get a summary of the account including balances and open positions"""
        try:
            balance = self.exchange.fetch_balance()
            
            # Get only non-zero balances
            non_zero = {k: v for k, v in balance['total'].items() if v > 0}
            
            # Get current prices for crypto assets
            prices = {}
            for asset, amount in non_zero.items():
                if asset != 'USD' and asset != 'ZUSD':
                    try:
                        symbol = f"{asset}/USD"
                        price = self.fetch_current_price(symbol)
                        prices[asset] = price
                    except:
                        # Skip if we can't get price
                        pass
            
            # Calculate total value
            total_value = non_zero.get('USD', non_zero.get('ZUSD', 0))
            for asset, amount in non_zero.items():
                if asset != 'USD' and asset != 'ZUSD' and asset in prices:
                    total_value += amount * prices[asset]
            
            return {
                'balances': non_zero,
                'prices': prices,
                'total_value': total_value
            }
        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return {'balances': {}, 'prices': {}, 'total_value': 0}
