class RiskManager:
    def __init__(self, max_loss=0.1):
        self.max_loss = max_loss
        self.initial_balance = None  # Set dynamically

    def set_initial_balance(self, balance_usd, balance_btc, current_price):
        self.initial_balance = balance_usd + (balance_btc * current_price)

    def is_safe(self, action, balance_usd, balance_btc, current_price):
        if self.initial_balance is None:
            self.set_initial_balance(balance_usd, balance_btc, current_price)
        total_value = balance_usd + (balance_btc * current_price)
        loss = (self.initial_balance - total_value) / self.initial_balance if self.initial_balance > 0 else 0
        safe = abs(loss) < self.max_loss  # Allow small gains/losses
        if not safe:
            print(f"Trade unsafe: loss={loss*100:.2f}% exceeds {self.max_loss*100}%")
        return safe