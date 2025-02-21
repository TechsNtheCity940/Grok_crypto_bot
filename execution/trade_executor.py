import ccxt
from config import KRAKEN_API_KEY, KRAKEN_API_SECRET, COINBASE_API_KEY, COINBASE_API_SECRET, TRADING_PAIRS, ACTIVE_EXCHANGE

class TradeExecutor:
    def __init__(self):
        if ACTIVE_EXCHANGE == 'kraken':
            self.exchange = ccxt.kraken({
                'apiKey': KRAKEN_API_KEY,
                'secret': KRAKEN_API_SECRET,
                'enableRateLimit': True,
            })
        else:
            self.exchange = ccxt.coinbase({
                'apiKey': COINBASE_API_KEY,
                'secret': COINBASE_API_SECRET,
                'enableRateLimit': True,
            })
        self.trading_pairs = TRADING_PAIRS

    def execute(self, action, symbol, amount=0.0001):
        balance_usd, balance_asset = self.get_balance(symbol)
        current_price = self.fetch_current_price(symbol)
        if action == 1:  # Buy
            if balance_usd < amount * current_price:
                print(f"Insufficient USD: need {amount * current_price}, have {balance_usd} for {symbol}")
                return None
            try:
                order = self.exchange.create_market_buy_order(symbol, amount)
                print(f"Buy order executed for {symbol}: {order}")
                return order
            except Exception as e:
                print(f"Buy order failed for {symbol}: {e}")
                return None
        elif action == 2:  # Sell
            if balance_asset < amount:
                print(f"Insufficient asset: need {amount}, have {balance_asset} for {symbol}")
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
            if ACTIVE_EXCHANGE == 'kraken':
                usd = balance['total'].get('ZUSD', 0.0)  # Kraken USD
                asset = symbol.split('/')[0]
                asset = 'XXBT' if asset == 'XBT' else asset  # Normalize XBT to XXBT
                asset_balance = balance['total'].get(asset, 0.0)
                print(f"Kraken balance for {symbol}: USD={usd}, {asset}={asset_balance}")
                return usd, asset_balance
            else:
                usd = balance['total'].get('USD', 0.0)
                asset = symbol.split('/')[0]
                asset_balance = balance['total'].get(asset, 0.0)
                print(f"Coinbase balance for {symbol}: USD={usd}, {asset}={asset_balance}")
                return usd, asset_balance
        except Exception as e:
            print(f"Balance fetch failed: {e}")
            return 0.0, 0.0

    def fetch_current_price(self, symbol):
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            print(f"Failed to fetch price for {symbol}: {e}")
            return 98700  # Fallback