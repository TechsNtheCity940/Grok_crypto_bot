import ccxt
from config import KRAKEN_API_KEY, KRAKEN_API_SECRET, COINBASE_API_KEY, COINBASE_API_SECRET, TRADING_PAIR, ACTIVE_EXCHANGE

class TradeExecutor:
    def __init__(self):
        if ACTIVE_EXCHANGE == 'kraken':
            self.exchange = ccxt.kraken({
                'apiKey': KRAKEN_API_KEY,
                'secret': KRAKEN_API_SECRET,
                'enableRateLimit': True,
            })
        else:  # Coinbase
            self.exchange = ccxt.coinbasepro({
                'apiKey': COINBASE_API_KEY,
                'secret': COINBASE_API_SECRET,
                'enableRateLimit': True,
            })
        self.symbol = TRADING_PAIR

    def execute(self, action, amount=0.001):  # Default 0.001 BTC
        if action == 1:  # Buy
            order = self.exchange.create_market_buy_order(self.symbol, amount)
            return order
        elif action == 2:  # Sell
            order = self.exchange.create_market_sell_order(self.symbol, amount)
            return order
        return None

    def get_balance(self):
        balance = self.exchange.fetch_balance()
        if ACTIVE_EXCHANGE == 'kraken':
            return balance['total']['USD'], balance['total']['XBT']
        else:  # Coinbase
            return balance['total']['USD'], balance['total']['BTC']