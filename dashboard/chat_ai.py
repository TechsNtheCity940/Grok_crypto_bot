import random
import re
from datetime import datetime, timedelta

class ChatAI:
    """
    Simple AI chat assistant for the crypto trading bot dashboard.
    Provides insights, summaries, and responses to user queries.
    """
    
    def __init__(self):
        self.personality = "analytical"
        self.log_history = []
        self.last_summary_time = None
        self.last_trade_time = None
    
    def add_log(self, log_entry):
        """Add a log entry to the AI's history"""
        self.log_history.append(log_entry)
        # Keep only the last 1000 log entries
        if len(self.log_history) > 1000:
            self.log_history = self.log_history[-1000:]
    
    def change_personality(self, personality):
        """Change the AI's personality"""
        valid_personalities = ["analytical", "friendly", "technical", "cautious"]
        if personality in valid_personalities:
            self.personality = personality
            return True
        return False
    
    def get_summary(self, bot_status, force=False):
        """Generate a summary of the bot's current status"""
        current_time = datetime.now()
        
        # Only generate a new summary if forced or it's been more than 5 minutes
        if not force and self.last_summary_time and (current_time - self.last_summary_time) < timedelta(minutes=5):
            return self.last_summary
        
        # Check if there are recent trades
        recent_trades = bot_status.get('recent_trades', [])
        new_trade = False
        
        if recent_trades:
            latest_trade_time = datetime.fromisoformat(recent_trades[0].get('timestamp'))
            if not self.last_trade_time or latest_trade_time > self.last_trade_time:
                new_trade = True
                self.last_trade_time = latest_trade_time
        
        # Generate summary based on bot status
        if bot_status.get('running', False):
            if new_trade:
                trade = recent_trades[0]
                summary = self._generate_trade_message(trade, bot_status)
            else:
                summary = self._generate_status_message(bot_status)
        else:
            summary = self._generate_idle_message(bot_status)
        
        self.last_summary = summary
        self.last_summary_time = current_time
        
        return summary
    
    def generate_message(self, bot_status):
        """Generate a message based on the bot's status"""
        return self.get_summary(bot_status, True)
    
    def _generate_trade_message(self, trade, bot_status):
        """Generate a message about a recent trade"""
        trade_type = trade.get('type', 'unknown')
        pair = trade.get('pair', 'unknown')
        price = trade.get('price', 0)
        amount = trade.get('amount', 0)
        value = trade.get('value', 0)
        
        if self.personality == "analytical":
            return f"I've just executed a {trade_type} order for {amount:.6f} {pair.split('/')[0]} at ${price:.2f}, with a total value of ${value:.2f}. The portfolio is now valued at ${bot_status.get('portfolio_value', 0):.2f}."
        
        elif self.personality == "friendly":
            if trade_type == "buy":
                return f"Good news! I just bought {amount:.6f} {pair.split('/')[0]} at ${price:.2f}. Your portfolio is now worth ${bot_status.get('portfolio_value', 0):.2f}."
            else:
                return f"Just letting you know, I sold {amount:.6f} {pair.split('/')[0]} at ${price:.2f}. Your portfolio is now worth ${bot_status.get('portfolio_value', 0):.2f}."
        
        elif self.personality == "technical":
            return f"EXECUTED: {trade_type.upper()} {pair} | AMOUNT: {amount:.6f} | PRICE: ${price:.2f} | VALUE: ${value:.2f} | PORTFOLIO: ${bot_status.get('portfolio_value', 0):.2f}"
        
        elif self.personality == "cautious":
            if trade_type == "buy":
                return f"I've carefully analyzed the market and executed a buy of {amount:.6f} {pair.split('/')[0]} at ${price:.2f}. I'll continue to monitor this position closely. Current portfolio value: ${bot_status.get('portfolio_value', 0):.2f}."
            else:
                return f"After careful consideration, I've sold {amount:.6f} {pair.split('/')[0]} at ${price:.2f}. This seemed like the prudent move based on current indicators. Current portfolio value: ${bot_status.get('portfolio_value', 0):.2f}."
    
    def _generate_status_message(self, bot_status):
        """Generate a message about the bot's current status"""
        portfolio_value = bot_status.get('portfolio_value', 0)
        balance_usd = bot_status.get('balance_usd', 0)
        active_pairs = bot_status.get('active_pairs', [])
        
        if self.personality == "analytical":
            return f"The trading bot is currently active, monitoring {len(active_pairs)} trading pairs. Current portfolio value is ${portfolio_value:.2f}, with ${balance_usd:.2f} in available USD."
        
        elif self.personality == "friendly":
            return f"I'm working hard for you! Watching {len(active_pairs)} different cryptocurrencies and ready to trade. Your portfolio is worth ${portfolio_value:.2f} right now, with ${balance_usd:.2f} ready to invest."
        
        elif self.personality == "technical":
            pairs_str = ", ".join(active_pairs)
            return f"STATUS: ACTIVE | MONITORING: {pairs_str} | PORTFOLIO VALUE: ${portfolio_value:.2f} | USD BALANCE: ${balance_usd:.2f}"
        
        elif self.personality == "cautious":
            return f"I'm carefully monitoring {len(active_pairs)} trading pairs, looking for safe opportunities. Your portfolio value is ${portfolio_value:.2f}, with ${balance_usd:.2f} in USD that I can use for prudent investments if conditions are right."
    
    def _generate_idle_message(self, bot_status):
        """Generate a message when the bot is idle"""
        portfolio_value = bot_status.get('portfolio_value', 0)
        
        if self.personality == "analytical":
            return f"The trading bot is currently inactive. Portfolio value is ${portfolio_value:.2f}. Start the bot to begin automated trading."
        
        elif self.personality == "friendly":
            return f"I'm taking a break right now! Your portfolio is worth ${portfolio_value:.2f}. Just hit the start button when you want me to start trading again."
        
        elif self.personality == "technical":
            return f"STATUS: INACTIVE | PORTFOLIO VALUE: ${portfolio_value:.2f} | AWAITING ACTIVATION"
        
        elif self.personality == "cautious":
            return f"The trading bot is currently paused. Your portfolio value is ${portfolio_value:.2f}. I'll wait for your instruction before making any trades."

# Create a singleton instance
chat_ai = ChatAI()
