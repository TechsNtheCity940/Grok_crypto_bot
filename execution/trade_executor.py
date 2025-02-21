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
            self.exchange = ccxt.coinbase({
                'apiKey': COINBASE_API_KEY,
                'secret': COINBASE_API_SECRET,
                'enableRateLimit': True,
            })
        self.symbol = TRADING_PAIR

    def execute(self, action, amount=0.0001):  # Micro-trading: 0.0001 BTC
        balance_usd, balance_btc = self.get_balance()
        current_price = 98290  # Approx price; should fetch real-time ideally
        if action == 1:  # Buy
            if balance_usd < amount * current_price:
                print(f"Insufficient USD: need {amount * current_price}, have {balance_usd}")
                return None
            try:
                order = self.exchange.create_market_buy_order(self.symbol, amount)
                print(f"Buy order executed: {order}")
                return order
            except Exception as e:
                print(f"Buy order failed: {e}")
                return None
        elif action == 2:  # Sell
            if balance_btc < amount:
                print(f"Insufficient BTC: need {amount}, have {balance_btc}")
                return None
            try:
                order = self.exchange.create_market_sell_order(self.symbol, amount)
                print(f"Sell order executed: {order}")
                return order
            except Exception as e:
                print(f"Sell order failed: {e}")
                return None
        return None

    def get_balance(self):
        try:
            balance = self.exchange.fetch_balance()
            print(f"Raw balance response: {balance}")  # Debug full response
            if ACTIVE_EXCHANGE == 'kraken':
                usd = balance['total'].get('USD', 0.0)  # Default to 0 if not present
                xbt = balance['total'].get('XBT', 0.0)  # Default to 0 if not present
                print(f"Kraken balance: USD={usd}, XBT={xbt}")
                return usd, xbt
            else:  # Coinbase
                usd = balance['total'].get('USD', 0.0)
                btc = balance['total'].get('BTC', 0.0)
                print(f"Coinbase balance: USD={usd}, BTC={btc}")
                return usd, btc
        except Exception as e:
            print(f"Balance fetch failed: {e}")
            return 0.0, 0.0