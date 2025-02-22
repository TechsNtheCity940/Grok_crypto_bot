import ccxt
from config import KRAKEN_API_KEY, KRAKEN_API_SECRET, ACTIVE_EXCHANGE

class TradeExecutor:
    def __init__(self):
        self.exchange = ccxt.kraken({
            'apiKey': KRAKEN_API_KEY,
            'secret': KRAKEN_API_SECRET,
            'enableRateLimit': True,
        })
        self.exchange.load_markets()
        self.trading_pairs = self.update_trading_pairs()

    def update_trading_pairs(self):
        balance = self.get_total_balance()
        markets = self.exchange.markets
        trading_pairs = [pair for pair in markets if pair.endswith('/USD') and balance.get(pair.split('/')[0], 0) > 0]
        print(f"Updated trading pairs: {trading_pairs}")
        return trading_pairs

    def execute(self, action, symbol, amount):
        balance_usd, balance_asset = self.get_balance(symbol)
        current_price = self.fetch_current_price(symbol)
        market = self.exchange.markets[symbol]
        min_amount = market['limits']['amount']['min']
        min_cost = market['limits']['cost']['min'] if 'cost' in market['limits'] else 0

        if action == 1:  # Buy
            cost = amount * current_price
            if amount < min_amount:
                print(f"Amount {amount} below minimum {min_amount} for {symbol}")
                return None
            if min_cost > 0 and cost < min_cost:
                print(f"Cost {cost} below minimum {min_cost} for {symbol}")
                return None
            try:
                order = self.exchange.create_market_buy_order(symbol, amount)
                print(f"Buy order executed for {symbol}: {order}")
                return order
            except Exception as e:
                print(f"Buy order failed for {symbol}: {e}")
                return None
        elif action == 2:  # Sell
            if amount < min_amount:
                print(f"Amount {amount} below minimum {min_amount} for {symbol}")
                return None
            try:
                order = self.exchange.create_market_sell_order(symbol, amount)
                print(f"Sell order executed for {symbol}: {order}")
                return order
            except Exception as e:
                print(f"Sell order failed for {symbol}: {e}")
                return None
        return None

    def get_balance(self, symbol):
        try:
            balance = self.exchange.fetch_balance()
            print(f"Raw balance response: {balance}")
            usd = balance['total'].get('USD', 0.0)  # Correct key for USD
            asset = symbol.split('/')[0]
            asset = 'BTC' if asset == 'XBT' else asset  # Normalize XBT to BTC
            asset_balance = balance['total'].get(asset, 0.0)
            print(f"Balance for {symbol}: USD={usd}, {asset}={asset_balance}")
            return usd, asset_balance
        except Exception as e:
            print(f"Balance fetch failed: {e}")
            return 0.0, 0.0

    def get_total_balance(self):
        try:
            return self.exchange.fetch_balance()['total']
        except Exception as e:
            print(f"Total balance fetch failed: {e}")
            return {}

    def fetch_current_price(self, symbol):
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            print(f"Failed to fetch price for {symbol}: {e}")
            return 98700  # Fallback price