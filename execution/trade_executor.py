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

    def execute(self, action, amount=0.0001):
        balance_usd, balance_btc = self.get_balance()
        current_price = 98290  # Replace with real-time price fetch
        if action == 1:  # Buy
            max_buy_amount = balance_usd / current_price
            if max_buy_amount < amount:
                amount = max_buy_amount * 0.99  # 1% buffer
            if amount <= 0:
                print("Insufficient USD to buy")
                return None
            # Execute buy order
        elif action == 2:  # Sell
            if balance_btc < amount:
                amount = balance_btc * 0.99  # 1% buffer
            if amount <= 0:
                print("Insufficient BTC to sell")
                return None
            # Execute sell order

    def get_balance(self):
        try:
            balance = self.exchange.fetch_balance()
            print(f"Raw balance response: {balance}")
            usd = balance['total'].get('ZUSD', 0.0)  # Kraken uses 'ZUSD' for USD
            xbt = balance['total'].get('XXBT', 0.0)  # Use 'XXBT' for BTC on Kraken
            print(f"Kraken balance: USD={usd}, XBT={xbt}")
            return usd, xbt
        except Exception as e:
            print(f"Balance fetch failed: {e}")
            return 0.0, 0.0