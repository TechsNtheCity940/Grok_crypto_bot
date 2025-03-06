import time
import logging
from utils.log_setup import logger

class RiskManager:
    def __init__(self, max_loss=0.5):
        self.max_loss = max_loss
        self.initial_balances = {}
        self.trade_cooldowns = {}  # Track last trade times per symbol

    def set_initial_balance(self, symbol, balance_usd, balance_asset, current_price):
        self.initial_balances[symbol] = balance_usd + (balance_asset * current_price)

    def is_safe(self, action, symbol, balance_usd, balance_asset, current_price):
        # Check trading cooldown - don't allow rapid consecutive trades for the same symbol
        cooldown_period = 15  # seconds
        current_time = time.time()
        if symbol in self.trade_cooldowns:
            time_since_last_trade = current_time - self.trade_cooldowns[symbol]
            if time_since_last_trade < cooldown_period:
                logger.info(f"Trade skipped for {symbol}: Cooldown period ({cooldown_period - time_since_last_trade:.1f}s remaining)")
                return False
        
        # Check sufficient balance
        if action == 1:  # Buy
            if balance_usd < 10.0:  # Minimum $10 USD to buy
                logger.warning(f"Trade unsafe for {symbol}: Insufficient USD balance (${balance_usd:.2f})")
                return False
        elif action == 2:  # Sell
            min_asset_value = 10.0  # Minimum $10 worth to sell
            asset_value = balance_asset * current_price
            if asset_value < min_asset_value:
                logger.warning(f"Trade unsafe for {symbol}: Insufficient asset balance (${asset_value:.2f})")
                return False
        
        # Initialize balance tracking if needed
        if symbol not in self.initial_balances:
            self.set_initial_balance(symbol, balance_usd, balance_asset, current_price)
        
        # Check max loss
        initial = self.initial_balances[symbol]
        total_value = balance_usd + (balance_asset * current_price)
        loss = (initial - total_value) / initial if initial > 0 else 0
        
        if abs(loss) > self.max_loss:
            logger.warning(f"Trade unsafe for {symbol}: loss={loss*100:.2f}% exceeds max allowed {self.max_loss*100:.2f}%")
            return False
        
        # Update trade cooldown
        self.trade_cooldowns[symbol] = current_time
        return True
