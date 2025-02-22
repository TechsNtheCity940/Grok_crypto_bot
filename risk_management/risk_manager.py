class RiskManager:
    def __init__(self, max_loss=0.5):  # 50% for aggression
        self.max_loss = max_loss
        self.initial_balances = {}

    def set_initial_balance(self, symbol, balance_usd, balance_asset, current_price):
        self.initial_balances[symbol] = balance_usd + (balance_asset * current_price)

    def is_safe(self, action, symbol, balance_usd, balance_asset, current_price):
        if symbol not in self.initial_balances:
            self.set_initial_balance(symbol, balance_usd, balance_asset, current_price)
        initial = self.initial_balances[symbol]
        total_value = balance_usd + (balance_asset * current_price)
        loss = (initial - total_value) / initial if initial > 0 else 0
        safe = abs(loss) < self.max_loss
        if not safe:
            print(f"Trade unsafe for {symbol}: loss={loss*100:.2f}% exceeds {self.max_loss*100}%")
        return safe