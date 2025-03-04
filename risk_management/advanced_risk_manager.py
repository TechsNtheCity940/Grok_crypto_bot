import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

from risk_management.risk_manager import RiskManager
from utils.log_setup import logger
from config_manager import config

class AdvancedRiskManager(RiskManager):
    """
    Enhanced risk management system that extends the base RiskManager
    with advanced risk management techniques like Kelly Criterion,
    Value at Risk (VaR), and dynamic position sizing.
    """
    def __init__(self, max_loss=None):
        # Initialize with max_loss from config if not provided
        max_loss = max_loss or config.get('max_loss', 0.5)
        super().__init__(max_loss=max_loss)
        
        # Additional parameters
        self.use_kelly = config.get('use_kelly_criterion', False)
        self.kelly_fraction = config.get('kelly_fraction', 0.5)  # Half-Kelly for more conservative sizing
        self.max_position_size = config.get('max_position_size', 0.2)  # Max 20% of portfolio in one position
        self.var_confidence = 0.95  # 95% confidence for VaR
        self.var_days = 1  # 1-day VaR
        
        # Track trade history for performance metrics
        self.trade_history = {}
        
        # Track portfolio value history
        self.portfolio_history = []
        
        # Track drawdowns
        self.max_portfolio_value = 0
        self.current_drawdown = 0
        self.max_drawdown = 0
        
        # Track correlation between assets
        self.correlation_matrix = None
        self.returns_data = {}
    
    def is_safe(self, action, symbol, balance_usd, balance_asset, current_price):
        """
        Enhanced safety check that considers multiple risk factors
        """
        # First, check basic safety with parent method
        basic_safe = super().is_safe(action, symbol, balance_usd, balance_asset, current_price)
        
        if not basic_safe:
            return False
        
        # Calculate current portfolio value
        portfolio_value = balance_usd + (balance_asset * current_price)
        
        # Update max portfolio value and drawdown
        if portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value
        
        if self.max_portfolio_value > 0:
            self.current_drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Add to portfolio history
        self.portfolio_history.append({
            'timestamp': datetime.now(),
            'value': portfolio_value,
            'drawdown': self.current_drawdown
        })
        
        # Trim history to last 30 days
        cutoff_time = datetime.now() - timedelta(days=30)
        self.portfolio_history = [
            item for item in self.portfolio_history
            if item['timestamp'] >= cutoff_time
        ]
        
        # Check drawdown limit
        if self.current_drawdown > self.max_loss:
            logger.warning(f"Trade unsafe: Current drawdown {self.current_drawdown:.2%} exceeds max loss {self.max_loss:.2%}")
            return False
        
        # Check VaR if we have enough history
        if len(self.portfolio_history) > 10:
            var = self.calculate_var(symbol)
            max_var_pct = 0.05  # 5% VaR limit
            
            if var > max_var_pct * portfolio_value:
                logger.warning(f"Trade unsafe: VaR {var:.2f} exceeds {max_var_pct:.2%} of portfolio value")
                return False
        
        return True
    
    def calculate_position_size(self, symbol, balance_usd, current_price, win_rate=0.55, risk_reward=1.5):
        """
        Calculate optimal position size using Kelly Criterion or fixed percentage
        """
        if not self.use_kelly:
            # Use fixed percentage of portfolio
            return balance_usd * self.max_position_size / current_price
        
        # Get win rate and risk/reward ratio from historical data if available
        if symbol in self.trade_history and len(self.trade_history[symbol]) >= 10:
            trades = self.trade_history[symbol]
            wins = sum(1 for trade in trades if trade['profit'] > 0)
            win_rate = wins / len(trades)
            
            if wins > 0 and len(trades) - wins > 0:
                avg_win = sum(trade['profit'] for trade in trades if trade['profit'] > 0) / wins
                avg_loss = abs(sum(trade['profit'] for trade in trades if trade['profit'] <= 0)) / (len(trades) - wins)
                risk_reward = avg_win / avg_loss if avg_loss > 0 else 1.5
        
        # Calculate Kelly percentage
        kelly_pct = win_rate - ((1 - win_rate) / risk_reward)
        
        # Apply Kelly fraction for more conservative sizing
        kelly_pct = kelly_pct * self.kelly_fraction
        
        # Limit to max position size
        kelly_pct = min(kelly_pct, self.max_position_size)
        
        # Ensure non-negative
        kelly_pct = max(0, kelly_pct)
        
        # Calculate position size in crypto units
        position_size = balance_usd * kelly_pct / current_price
        
        logger.info(f"Kelly position size for {symbol}: {kelly_pct:.2%} of portfolio")
        return position_size
    
    def record_trade(self, symbol, action, amount, price, timestamp=None):
        """
        Record trade for performance tracking
        """
        if symbol not in self.trade_history:
            self.trade_history[symbol] = []
        
        timestamp = timestamp or datetime.now()
        
        # Find previous trade to calculate profit
        profit = 0
        if self.trade_history[symbol]:
            prev_trade = self.trade_history[symbol][-1]
            if prev_trade['action'] != action:  # If previous action was opposite
                if action == 2:  # Sell
                    profit = (price - prev_trade['price']) * amount
                else:  # Buy
                    profit = (prev_trade['price'] - price) * prev_trade['amount']
        
        trade = {
            'timestamp': timestamp,
            'action': action,
            'amount': amount,
            'price': price,
            'profit': profit
        }
        
        self.trade_history[symbol].append(trade)
        
        # Update returns data for correlation calculation
        if symbol not in self.returns_data:
            self.returns_data[symbol] = []
        
        self.returns_data[symbol].append({
            'timestamp': timestamp,
            'price': price,
            'return': 0 if len(self.returns_data[symbol]) == 0 else 
                     (price - self.returns_data[symbol][-1]['price']) / self.returns_data[symbol][-1]['price']
        })
        
        # Trim to last 100 data points
        if len(self.returns_data[symbol]) > 100:
            self.returns_data[symbol] = self.returns_data[symbol][-100:]
        
        # Update correlation matrix
        self._update_correlation_matrix()
        
        return trade
    
    def _update_correlation_matrix(self):
        """
        Update correlation matrix between assets
        """
        if len(self.returns_data) < 2:
            return
        
        # Create DataFrame with returns
        returns_dict = {}
        for symbol, data in self.returns_data.items():
            if len(data) > 5:  # Need at least a few data points
                returns = [item['return'] for item in data]
                returns_dict[symbol] = returns[-min(len(returns), 30):]  # Use last 30 returns at most
        
        # Ensure all arrays have the same length
        min_length = min(len(returns) for returns in returns_dict.values())
        for symbol in returns_dict:
            returns_dict[symbol] = returns_dict[symbol][-min_length:]
        
        if min_length > 5:  # Need at least a few data points
            df = pd.DataFrame(returns_dict)
            self.correlation_matrix = df.corr()
    
    def calculate_var(self, symbol, confidence=None, days=None):
        """
        Calculate Value at Risk (VaR) for a given symbol
        """
        confidence = confidence or self.var_confidence
        days = days or self.var_days
        
        if symbol not in self.returns_data or len(self.returns_data[symbol]) < 10:
            return 0
        
        # Extract returns
        returns = [item['return'] for item in self.returns_data[symbol]]
        
        # Calculate VaR
        returns = np.array(returns)
        var_percentile = np.percentile(returns, (1 - confidence) * 100)
        
        # Scale to the number of days
        var_scaled = var_percentile * np.sqrt(days)
        
        return abs(var_scaled)
    
    def calculate_portfolio_var(self, portfolio_value, positions):
        """
        Calculate portfolio VaR considering correlations
        
        Args:
            portfolio_value: Total portfolio value
            positions: Dict of {symbol: position_value}
        """
        if not self.correlation_matrix is not None or len(positions) < 2:
            # Fall back to simple sum of individual VaRs
            total_var = sum(self.calculate_var(symbol) * value for symbol, value in positions.items())
            return total_var
        
        # Calculate weighted VaR
        weights = {}
        vars = {}
        symbols = []
        
        for symbol, value in positions.items():
            if symbol in self.correlation_matrix and self.correlation_matrix.index.contains(symbol):
                weights[symbol] = value / portfolio_value
                vars[symbol] = self.calculate_var(symbol)
                symbols.append(symbol)
        
        if not symbols:
            return 0
        
        # Create weight vector and VaR vector
        weight_vector = np.array([weights[s] for s in symbols])
        var_vector = np.array([vars[s] for s in symbols])
        
        # Extract correlation matrix for these symbols
        corr_matrix = self.correlation_matrix.loc[symbols, symbols].values
        
        # Calculate portfolio VaR
        portfolio_var = np.sqrt(weight_vector @ corr_matrix @ weight_vector) * np.mean(var_vector) * portfolio_value
        
        return portfolio_var
    
    def get_diversification_recommendation(self, current_positions):
        """
        Get recommendation for portfolio diversification
        
        Args:
            current_positions: Dict of {symbol: position_value}
        """
        if self.correlation_matrix is None or len(current_positions) < 2:
            return None
        
        # Find highly correlated pairs
        high_corr_pairs = []
        
        for i, symbol1 in enumerate(self.correlation_matrix.index):
            for j, symbol2 in enumerate(self.correlation_matrix.index):
                if i < j and symbol1 in current_positions and symbol2 in current_positions:
                    corr = self.correlation_matrix.loc[symbol1, symbol2]
                    if corr > 0.7:  # High correlation threshold
                        high_corr_pairs.append((symbol1, symbol2, corr))
        
        if not high_corr_pairs:
            return {"status": "diversified", "message": "Portfolio is well diversified"}
        
        # Sort by correlation (highest first)
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        
        recommendations = []
        for symbol1, symbol2, corr in high_corr_pairs:
            recommendations.append({
                "pair": (symbol1, symbol2),
                "correlation": corr,
                "recommendation": f"Consider reducing exposure to either {symbol1} or {symbol2} (correlation: {corr:.2f})"
            })
        
        return {
            "status": "high_correlation",
            "message": "Some assets are highly correlated",
            "recommendations": recommendations
        }
    
    def get_risk_metrics(self):
        """
        Get comprehensive risk metrics for the portfolio
        """
        if not self.portfolio_history:
            return {
                "max_drawdown": 0,
                "current_drawdown": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "win_rate": 0,
                "profit_factor": 0
            }
        
        # Calculate returns
        values = [item['value'] for item in self.portfolio_history]
        returns = [0]
        for i in range(1, len(values)):
            returns.append((values[i] - values[i-1]) / values[i-1])
        
        returns = returns[1:]  # Remove the first 0
        
        if not returns:
            return {
                "max_drawdown": self.max_drawdown,
                "current_drawdown": self.current_drawdown,
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "win_rate": 0,
                "profit_factor": 0
            }
        
        # Calculate metrics
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        risk_free_rate = 0.02 / 365  # Assuming 2% annual risk-free rate
        
        # Sharpe ratio
        sharpe_ratio = (avg_return - risk_free_rate) / std_return if std_return > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = [r for r in returns if r < 0]
        downside_deviation = np.std(downside_returns) if downside_returns else 0
        sortino_ratio = (avg_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Win rate and profit factor across all trades
        all_trades = []
        for symbol_trades in self.trade_history.values():
            all_trades.extend(symbol_trades)
        
        if all_trades:
            wins = sum(1 for trade in all_trades if trade['profit'] > 0)
            losses = sum(1 for trade in all_trades if trade['profit'] < 0)
            win_rate = wins / len(all_trades) if len(all_trades) > 0 else 0
            
            gross_profit = sum(trade['profit'] for trade in all_trades if trade['profit'] > 0)
            gross_loss = abs(sum(trade['profit'] for trade in all_trades if trade['profit'] < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            win_rate = 0
            profit_factor = 0
        
        return {
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "win_rate": win_rate,
            "profit_factor": profit_factor
        }
