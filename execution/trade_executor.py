import ccxt
import logging
from config import KRAKEN_API_KEY, KRAKEN_API_SECRET, ACTIVE_EXCHANGE

class TradeExecutor:
    def __init__(self):
        self.logger = logging.getLogger('TradeExecutor')
        self.exchange = ccxt.kraken({
            'apiKey': KRAKEN_API_KEY,
            'secret': KRAKEN_API_SECRET,
            'enableRateLimit': True,
        })
        self.trading_pairs = self.update_trading_pairs()
        self.min_trade_sizes = {
            'DOGE/USD': 10.0,  # Kraken minimums
            'SHIB/USD': 100000.0,
            'XRP/USD': 0.1
        }

    def update_trading_pairs(self):
        try:
            markets = self.exchange.load_markets()
            self.trading_pairs = [pair for pair in markets if pair.endswith('/USD')]
            self.logger.info(f"Updated trading pairs: {self.trading_pairs}")
            return self.trading_pairs
        except Exception as e:
            self.logger.error(f"Error updating trading pairs: {str(e)}")
            return ['DOGE/USD']

    def fetch_current_price(self, symbol):
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            self.logger.error(f"Error fetching price for {symbol}: {str(e)}")
            return 0

    def get_balance(self, symbol):
        try:
            balance = self.exchange.fetch_balance()
            base = symbol.split('/')[0]
            quote = symbol.split('/')[1]
            base_balance = float(balance.get(base, {}).get('free', 0))
            quote_balance = float(balance.get(quote, {}).get('free', 0))
            self.logger.debug(f"Raw balance response: {balance}")
            self.logger.info(f"Balance for {symbol}: {quote}={quote_balance}, {base}={base_balance}")
            return quote_balance, base_balance
        except Exception as e:
            self.logger.error(f"Error fetching balance: {str(e)}")
            return 0, 0

    def execute(self, action, symbol, amount):
        try:
            min_trade_size = self.min_trade_sizes.get(symbol, 10.0)
            if amount < min_trade_size:
                self.logger.warning(f"Amount {amount} below minimum {min_trade_size} for {symbol}")
                return None, True
            if action == 1:  # Buy
                order = self.exchange.create_market_buy_order(symbol, amount)
            elif action == 2:  # Sell
                order = self.exchange.create_market_sell_order(symbol, amount)
            else:
                return None, False
            self.logger.info(f"Order executed: {order}")
            return order, False
        except Exception as e:
            self.logger.error(f"Error executing order: {str(e)}")
            return None, True