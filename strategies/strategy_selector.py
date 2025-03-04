import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

from strategies.grid_trading import GridTrader
from utils.log_setup import logger
from utils.enhanced_data_utils import detect_market_regime
from config_manager import config

# Import strategies (will add more as they're implemented)
try:
    from strategies.momentum_strategy import MomentumStrategy
except ImportError:
    MomentumStrategy = None

class StrategySelector:
    """
    Strategy selector that dynamically chooses the best trading strategy
    based on current market conditions and performance history.
    """
    def __init__(self, available_strategies=None):
        self.available_strategies = available_strategies or ['grid_trading']
        
        # Initialize strategy instances
        self.strategies = {}
        self.initialize_strategies()
        
        # Track strategy performance
        self.performance_history = {}
        
        # Default weights for strategy combination
        self.strategy_weights = {
            'grid_trading': 0.5,
            'momentum': 0.5,
            'mean_reversion': 0.5,
            'breakout': 0.5
        }
        
        # Market regime preferences for each strategy
        self.regime_preferences = {
            'grid_trading': 'ranging',
            'momentum': 'trending',
            'mean_reversion': 'ranging',
            'breakout': 'volatile'
        }
    
    def initialize_strategies(self):
        """Initialize available strategy instances"""
        if 'grid_trading' in self.available_strategies:
            grid_config = config.get('grid_trading', {})
            self.strategies['grid_trading'] = GridTrader({'grid_trading': grid_config})
        
        if 'momentum' in self.available_strategies and MomentumStrategy is not None:
            try:
                self.strategies['momentum'] = MomentumStrategy()
            except Exception as e:
                logger.error(f"Error initializing momentum strategy: {e}")
        
        # Add more strategies as they're implemented
    
    def select_strategy(self, market_data, symbol=None):
        """
        Select the best strategy based on current market conditions
        
        Args:
            market_data: DataFrame with market data
            symbol: Trading pair symbol
        
        Returns:
            Selected strategy instance or dict of weighted strategies
        """
        # Detect market regime
        regime_data = detect_market_regime(market_data)
        current_regime = regime_data['regime'].iloc[-1] if not regime_data.empty else 'ranging'
        
        logger.info(f"Detected market regime: {current_regime}")
        
        # Update strategy weights based on regime
        self._update_weights_by_regime(current_regime)
        
        # Update weights based on recent performance if available
        if symbol and symbol in self.performance_history:
            self._update_weights_by_performance(symbol)
        
        # Select strategy with highest weight
        best_strategy = max(self.strategy_weights.items(), key=lambda x: x[1] if x[0] in self.strategies else 0)
        
        if best_strategy[0] in self.strategies:
            logger.info(f"Selected strategy: {best_strategy[0]} with weight {best_strategy[1]:.2f}")
            return self.strategies[best_strategy[0]]
        else:
            # Fallback to grid trading if best strategy not available
            logger.warning(f"Best strategy {best_strategy[0]} not available, falling back to grid trading")
            return self.strategies.get('grid_trading')
    
    def get_weighted_strategies(self, market_data, symbol=None):
        """
        Get all available strategies with their weights
        
        Args:
            market_data: DataFrame with market data
            symbol: Trading pair symbol
        
        Returns:
            Dict of {strategy_name: (strategy_instance, weight)}
        """
        # Detect market regime
        regime_data = detect_market_regime(market_data)
        current_regime = regime_data['regime'].iloc[-1] if not regime_data.empty else 'ranging'
        
        # Update strategy weights
        self._update_weights_by_regime(current_regime)
        
        if symbol and symbol in self.performance_history:
            self._update_weights_by_performance(symbol)
        
        # Return strategies with weights
        weighted_strategies = {}
        for strategy_name, strategy in self.strategies.items():
            weight = self.strategy_weights.get(strategy_name, 0)
            if weight > 0:
                weighted_strategies[strategy_name] = (strategy, weight)
        
        return weighted_strategies
    
    def _update_weights_by_regime(self, current_regime):
        """Update strategy weights based on current market regime"""
        # Boost strategies that work well in the current regime
        for strategy_name, preferred_regime in self.regime_preferences.items():
            if strategy_name in self.strategy_weights:
                if preferred_regime == current_regime:
                    # Boost weight for strategies that work well in this regime
                    self.strategy_weights[strategy_name] *= 1.5
                else:
                    # Reduce weight for strategies that don't work well in this regime
                    self.strategy_weights[strategy_name] *= 0.7
        
        # Normalize weights
        total_weight = sum(self.strategy_weights.values())
        if total_weight > 0:
            self.strategy_weights = {k: v / total_weight for k, v in self.strategy_weights.items()}
    
    def _update_weights_by_performance(self, symbol):
        """Update strategy weights based on recent performance"""
        if symbol not in self.performance_history:
            return
        
        # Get recent performance data
        recent_performance = {}
        for strategy_name, performance_data in self.performance_history[symbol].items():
            # Filter to last 7 days
            cutoff_time = datetime.now() - timedelta(days=7)
            recent_data = [p for p in performance_data if p['timestamp'] >= cutoff_time]
            
            if recent_data:
                # Calculate average profit
                avg_profit = sum(p['profit'] for p in recent_data) / len(recent_data)
                recent_performance[strategy_name] = avg_profit
        
        if not recent_performance:
            return
        
        # Adjust weights based on performance
        min_profit = min(recent_performance.values())
        max_profit = max(recent_performance.values())
        profit_range = max_profit - min_profit
        
        if profit_range > 0:
            for strategy_name, avg_profit in recent_performance.items():
                if strategy_name in self.strategy_weights:
                    # Normalize profit to 0-1 range and adjust weight
                    normalized_profit = (avg_profit - min_profit) / profit_range
                    self.strategy_weights[strategy_name] *= (1 + normalized_profit)
        
        # Normalize weights
        total_weight = sum(self.strategy_weights.values())
        if total_weight > 0:
            self.strategy_weights = {k: v / total_weight for k, v in self.strategy_weights.items()}
    
    def record_performance(self, symbol, strategy_name, profit, timestamp=None):
        """
        Record strategy performance for future selection
        
        Args:
            symbol: Trading pair symbol
            strategy_name: Name of the strategy
            profit: Profit/loss from the trade
            timestamp: Timestamp of the trade (default: now)
        """
        if symbol not in self.performance_history:
            self.performance_history[symbol] = {}
        
        if strategy_name not in self.performance_history[symbol]:
            self.performance_history[symbol][strategy_name] = []
        
        timestamp = timestamp or datetime.now()
        
        self.performance_history[symbol][strategy_name].append({
            'timestamp': timestamp,
            'profit': profit
        })
        
        # Trim history to last 30 days
        cutoff_time = datetime.now() - timedelta(days=30)
        self.performance_history[symbol][strategy_name] = [
            p for p in self.performance_history[symbol][strategy_name]
            if p['timestamp'] >= cutoff_time
        ]
        
        logger.info(f"Recorded {strategy_name} performance for {symbol}: {profit:.2f}")
    
    def get_strategy_recommendations(self, market_data, symbol):
        """
        Get strategy recommendations based on current market conditions
        
        Args:
            market_data: DataFrame with market data
            symbol: Trading pair symbol
        
        Returns:
            Dict with strategy recommendations
        """
        # Detect market regime
        regime_data = detect_market_regime(market_data)
        current_regime = regime_data['regime'].iloc[-1] if not regime_data.empty else 'ranging'
        
        # Get volatility and trend strength
        volatility = regime_data['volatility'].iloc[-1] if not regime_data.empty else 0
        trend_strength = regime_data['adx'].iloc[-1] if not regime_data.empty else 0
        
        # Get weighted strategies
        weighted_strategies = self.get_weighted_strategies(market_data, symbol)
        
        # Prepare recommendations
        recommendations = {
            'market_regime': current_regime,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'recommended_strategies': [
                {
                    'name': strategy_name,
                    'weight': weight,
                    'reason': f"Works well in {current_regime} markets"
                }
                for strategy_name, (_, weight) in weighted_strategies.items()
            ],
            'primary_strategy': max(weighted_strategies.items(), key=lambda x: x[1][1])[0] if weighted_strategies else 'grid_trading'
        }
        
        return recommendations
