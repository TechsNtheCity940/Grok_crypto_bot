class RiskManager:
    def __init__(self, max_loss=0.1):  # 10% max loss
        self.max_loss = max_loss
        self.initial_balance = 10000

    def is_safe(self, action, balance, position_value):
        total_value = balance + position_value
        loss = (self.initial_balance - total_value) / self.initial_balance
        return loss < self.max_loss