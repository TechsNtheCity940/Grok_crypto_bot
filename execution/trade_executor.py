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
            balance = self.exchange.fetch_balance()
            logger.info(f"Raw Kraken balance response: {balance}")
            usd = balance['total'].get('USD', 0.0)  # Changed from 'ZUSD' to 'USD'
            asset = symbol.split('/')[0]
            crypto = balance['total'].get(asset, 0.0)
            logger.info(f"Parsed balance for {symbol}: USD={usd}, {asset}={crypto}")
            return usd, crypto
        except Exception as e:
            logger.error(f"Error fetching balance for {symbol}: {e}")
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
            if action == 1:  # Buy
                order = self.exchange.create_market_buy_order(symbol, amount)
                logger.info(f"Buy order executed: {order}")
                return order, False
            elif action == 2:  # Sell
                order = self.exchange.create_market_sell_order(symbol, amount)
                logger.info(f"Sell order executed: {order}")
                return order, False
        except ccxt.InsufficientFunds as e:
            logger.error(f"Insufficient funds for {symbol}: {e}")
            return None, True
        except Exception as e:
            logger.error(f"Execution error for {symbol}: {e}")
            return None, True

    def update_trading_pairs(self):
        try:
            markets = self.exchange.load_markets()
            return [pair for pair in markets.keys() if pair.endswith('/USD')]
        except Exception as e:
            logger.error(f"Error updating trading pairs: {e}")
            return TRADING_PAIRS