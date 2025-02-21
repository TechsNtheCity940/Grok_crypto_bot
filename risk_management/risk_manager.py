class RiskManager:
    def __init__(self, max_loss=0.1):
        self.max_loss = max_loss
        self.initial_balance = 7.4729  # Your actual starting USD balance

    def is_safe(self, action, balance_usd, balance_btc, current_price):
        total_value = balance_usd + (balance_btc * current_price)
        loss = (self.initial_balance - total_value) / self.initial_balance
        safe = loss < self.max_loss
        if not safe:
            print(f"Trade unsafe: loss={loss*100:.2f}% exceeds {self.max_loss*100}%")
        return safe