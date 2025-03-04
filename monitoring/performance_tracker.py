import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

from utils.log_setup import logger
from config_manager import config

class PerformanceTracker:
    """
    Performance tracking system that records trades, calculates metrics,
    and provides visualization and analysis of trading performance.
    """
    def __init__(self, data_dir='data/performance'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize data structures
        self.trades = []
        self.portfolio_history = []
        self.metrics_history = []
        
        # Load existing data if available
        self._load_data()
    
    def _load_data(self):
        """Load existing performance data if available"""
        trades_file = os.path.join(self.data_dir, 'trades.json')
        portfolio_file = os.path.join(self.data_dir, 'portfolio_history.json')
        metrics_file = os.path.join(self.data_dir, 'metrics_history.json')
        
        try:
            if os.path.exists(trades_file):
                with open(trades_file, 'r') as f:
                    self.trades = json.load(f)
                logger.info(f"Loaded {len(self.trades)} trades from {trades_file}")
            
            if os.path.exists(portfolio_file):
                with open(portfolio_file, 'r') as f:
                    self.portfolio_history = json.load(f)
                logger.info(f"Loaded portfolio history from {portfolio_file}")
            
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    self.metrics_history = json.load(f)
                logger.info(f"Loaded metrics history from {metrics_file}")
        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
    
    def _save_data(self):
        """Save performance data to disk"""
        trades_file = os.path.join(self.data_dir, 'trades.json')
        portfolio_file = os.path.join(self.data_dir, 'portfolio_history.json')
        metrics_file = os.path.join(self.data_dir, 'metrics_history.json')
        
        try:
            with open(trades_file, 'w') as f:
                json.dump(self.trades, f, indent=2)
            
            with open(portfolio_file, 'w') as f:
                json.dump(self.portfolio_history, f, indent=2)
            
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            
            logger.info("Saved performance data to disk")
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    def log_trade(self, symbol, action, amount, price, timestamp=None, strategy=None, metadata=None):
        """
        Log a trade for performance tracking
        
        Args:
            symbol: Trading pair symbol
            action: Trade action (1=buy, 2=sell)
            amount: Trade amount
            price: Trade price
            timestamp: Trade timestamp (default: now)
            strategy: Strategy that generated the trade
            metadata: Additional trade metadata
        
        Returns:
            dict: The logged trade
        """
        timestamp = timestamp or datetime.now().isoformat()
        
        # Convert action to string for readability
        action_str = 'buy' if action == 1 else 'sell' if action == 2 else 'unknown'
        
        trade = {
            'id': len(self.trades) + 1,
            'symbol': symbol,
            'action': action_str,
            'amount': float(amount),
            'price': float(price),
            'value': float(amount * price),
            'timestamp': timestamp,
            'strategy': strategy,
            'metadata': metadata or {}
        }
        
        # Calculate profit if this is a closing trade
        prev_trades = [t for t in reversed(self.trades) if t['symbol'] == symbol]
        if prev_trades and prev_trades[0]['action'] != action_str:
            prev_trade = prev_trades[0]
            
            if action_str == 'sell' and prev_trade['action'] == 'buy':
                # Buy then sell
                profit = (price - prev_trade['price']) * min(amount, prev_trade['amount'])
                trade['profit'] = float(profit)
                trade['profit_percent'] = float((price / prev_trade['price'] - 1) * 100)
            elif action_str == 'buy' and prev_trade['action'] == 'sell':
                # Sell then buy
                profit = (prev_trade['price'] - price) * min(amount, prev_trade['amount'])
                trade['profit'] = float(profit)
                trade['profit_percent'] = float((prev_trade['price'] / price - 1) * 100)
        
        self.trades.append(trade)
        logger.info(f"Logged {action_str} trade for {symbol}: {amount} @ {price}")
        
        # Save data
        self._save_data()
        
        return trade
    
    def update_portfolio_value(self, portfolio_value, balances=None, timestamp=None):
        """
        Update portfolio value history
        
        Args:
            portfolio_value: Total portfolio value in USD
            balances: Dict of {symbol: amount} balances
            timestamp: Timestamp (default: now)
        """
        timestamp = timestamp or datetime.now().isoformat()
        
        entry = {
            'timestamp': timestamp,
            'value': float(portfolio_value),
            'balances': balances or {}
        }
        
        # Calculate change from previous
        if self.portfolio_history:
            prev_value = self.portfolio_history[-1]['value']
            entry['change'] = float(portfolio_value - prev_value)
            entry['change_percent'] = float((portfolio_value / prev_value - 1) * 100) if prev_value > 0 else 0
        
        self.portfolio_history.append(entry)
        
        # Calculate and update metrics
        self.calculate_metrics()
        
        # Save data
        self._save_data()
    
    def calculate_metrics(self):
        """
        Calculate performance metrics
        
        Returns:
            dict: Performance metrics
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_trades': len(self.trades),
            'win_rate': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'avg_profit_per_trade': 0,
            'total_profit': 0
        }
        
        # Calculate win rate and profit metrics
        if self.trades:
            profit_trades = [t for t in self.trades if t.get('profit', 0) > 0]
            loss_trades = [t for t in self.trades if t.get('profit', 0) < 0]
            
            metrics['win_rate'] = len(profit_trades) / len(self.trades) if self.trades else 0
            
            total_profit = sum(t.get('profit', 0) for t in profit_trades)
            total_loss = abs(sum(t.get('profit', 0) for t in loss_trades))
            
            metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf')
            metrics['avg_profit_per_trade'] = sum(t.get('profit', 0) for t in self.trades) / len(self.trades) if self.trades else 0
            metrics['total_profit'] = sum(t.get('profit', 0) for t in self.trades)
        
        # Calculate portfolio metrics
        if len(self.portfolio_history) > 1:
            # Extract portfolio values and convert timestamps
            values = []
            timestamps = []
            
            for entry in self.portfolio_history:
                try:
                    if isinstance(entry['timestamp'], str):
                        dt = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                    else:
                        dt = entry['timestamp']
                    timestamps.append(dt)
                    values.append(entry['value'])
                except Exception as e:
                    logger.error(f"Error parsing portfolio history entry: {e}")
            
            if len(values) > 1:
                # Calculate returns
                returns = []
                for i in range(1, len(values)):
                    returns.append((values[i] - values[i-1]) / values[i-1])
                
                # Calculate Sharpe ratio (assuming risk-free rate of 0)
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                metrics['sharpe_ratio'] = avg_return / std_return if std_return > 0 else 0
                
                # Calculate Sortino ratio (downside deviation)
                downside_returns = [r for r in returns if r < 0]
                downside_deviation = np.std(downside_returns) if downside_returns else 0
                metrics['sortino_ratio'] = avg_return / downside_deviation if downside_deviation > 0 else 0
                
                # Calculate maximum drawdown
                peak = values[0]
                max_dd = 0
                
                for value in values:
                    if value > peak:
                        peak = value
                    dd = (peak - value) / peak if peak > 0 else 0
                    max_dd = max(max_dd, dd)
                
                metrics['max_drawdown'] = max_dd
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_latest_metrics(self):
        """Get the latest performance metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return self.calculate_metrics()
    
    def get_trades_by_symbol(self, symbol):
        """Get all trades for a specific symbol"""
        return [t for t in self.trades if t['symbol'] == symbol]
    
    def get_trades_by_strategy(self, strategy):
        """Get all trades for a specific strategy"""
        return [t for t in self.trades if t.get('strategy') == strategy]
    
    def get_trades_in_timeframe(self, start_time, end_time=None):
        """Get all trades within a specific timeframe"""
        end_time = end_time or datetime.now().isoformat()
        
        return [t for t in self.trades if start_time <= t['timestamp'] <= end_time]
    
    def get_portfolio_value_history(self, start_time=None, end_time=None):
        """Get portfolio value history within a specific timeframe"""
        if not start_time:
            # Default to last 30 days
            start_time = (datetime.now() - timedelta(days=30)).isoformat()
        
        end_time = end_time or datetime.now().isoformat()
        
        return [p for p in self.portfolio_history if start_time <= p['timestamp'] <= end_time]
    
    def generate_performance_report(self, timeframe='all'):
        """
        Generate a comprehensive performance report
        
        Args:
            timeframe: 'all', 'day', 'week', 'month', 'year'
        
        Returns:
            dict: Performance report
        """
        # Determine start time based on timeframe
        if timeframe == 'day':
            start_time = (datetime.now() - timedelta(days=1)).isoformat()
        elif timeframe == 'week':
            start_time = (datetime.now() - timedelta(days=7)).isoformat()
        elif timeframe == 'month':
            start_time = (datetime.now() - timedelta(days=30)).isoformat()
        elif timeframe == 'year':
            start_time = (datetime.now() - timedelta(days=365)).isoformat()
        else:
            start_time = None
        
        # Get trades and portfolio history for the timeframe
        if start_time:
            trades = self.get_trades_in_timeframe(start_time)
            portfolio_history = self.get_portfolio_value_history(start_time)
        else:
            trades = self.trades
            portfolio_history = self.portfolio_history
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Calculate additional metrics for the timeframe
        if trades:
            profit_trades = [t for t in trades if t.get('profit', 0) > 0]
            loss_trades = [t for t in trades if t.get('profit', 0) < 0]
            
            win_rate = len(profit_trades) / len(trades) if trades else 0
            
            total_profit = sum(t.get('profit', 0) for t in profit_trades)
            total_loss = abs(sum(t.get('profit', 0) for t in loss_trades))
            
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            avg_profit_per_trade = sum(t.get('profit', 0) for t in trades) / len(trades) if trades else 0
            total_profit = sum(t.get('profit', 0) for t in trades)
        else:
            win_rate = 0
            profit_factor = 0
            avg_profit_per_trade = 0
            total_profit = 0
        
        # Calculate portfolio performance
        if len(portfolio_history) > 1:
            start_value = portfolio_history[0]['value']
            end_value = portfolio_history[-1]['value']
            portfolio_return = (end_value - start_value) / start_value if start_value > 0 else 0
        else:
            portfolio_return = 0
        
        # Generate report
        report = {
            'timeframe': timeframe,
            'start_time': start_time or (self.trades[0]['timestamp'] if self.trades else None),
            'end_time': datetime.now().isoformat(),
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_profit_per_trade': avg_profit_per_trade,
            'total_profit': total_profit,
            'portfolio_return': portfolio_return,
            'sharpe_ratio': metrics['sharpe_ratio'],
            'sortino_ratio': metrics['sortino_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'trades_by_symbol': {},
            'trades_by_strategy': {}
        }
        
        # Group trades by symbol
        symbols = set(t['symbol'] for t in trades)
        for symbol in symbols:
            symbol_trades = [t for t in trades if t['symbol'] == symbol]
            report['trades_by_symbol'][symbol] = {
                'count': len(symbol_trades),
                'profit': sum(t.get('profit', 0) for t in symbol_trades),
                'win_rate': sum(1 for t in symbol_trades if t.get('profit', 0) > 0) / len(symbol_trades) if symbol_trades else 0
            }
        
        # Group trades by strategy
        strategies = set(t.get('strategy') for t in trades if t.get('strategy'))
        for strategy in strategies:
            strategy_trades = [t for t in trades if t.get('strategy') == strategy]
            report['trades_by_strategy'][strategy] = {
                'count': len(strategy_trades),
                'profit': sum(t.get('profit', 0) for t in strategy_trades),
                'win_rate': sum(1 for t in strategy_trades if t.get('profit', 0) > 0) / len(strategy_trades) if strategy_trades else 0
            }
        
        return report
    
    def plot_portfolio_performance(self, output_file=None, timeframe='month'):
        """
        Plot portfolio performance
        
        Args:
            output_file: Path to save the plot (default: None, display only)
            timeframe: 'all', 'day', 'week', 'month', 'year'
        """
        # Determine start time based on timeframe
        if timeframe == 'day':
            start_time = (datetime.now() - timedelta(days=1)).isoformat()
        elif timeframe == 'week':
            start_time = (datetime.now() - timedelta(days=7)).isoformat()
        elif timeframe == 'month':
            start_time = (datetime.now() - timedelta(days=30)).isoformat()
        elif timeframe == 'year':
            start_time = (datetime.now() - timedelta(days=365)).isoformat()
        else:
            start_time = None
        
        # Get portfolio history for the timeframe
        if start_time:
            portfolio_history = self.get_portfolio_value_history(start_time)
        else:
            portfolio_history = self.portfolio_history
        
        if not portfolio_history:
            logger.warning("No portfolio history available for plotting")
            return
        
        # Extract data for plotting
        timestamps = []
        values = []
        
        for entry in portfolio_history:
            try:
                if isinstance(entry['timestamp'], str):
                    dt = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                else:
                    dt = entry['timestamp']
                timestamps.append(dt)
                values.append(entry['value'])
            except Exception as e:
                logger.error(f"Error parsing portfolio history entry for plotting: {e}")
        
        if not timestamps:
            logger.warning("No valid timestamps in portfolio history")
            return
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, values, 'b-', linewidth=2)
        plt.title(f'Portfolio Performance ({timeframe})')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value (USD)')
        plt.grid(True)
        
        # Add annotations for key metrics
        if len(values) > 1:
            start_value = values[0]
            end_value = values[-1]
            total_return = (end_value - start_value) / start_value
            
            # Calculate drawdown
            peak = values[0]
            max_dd = 0
            max_dd_idx = 0
            
            for i, value in enumerate(values):
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
                    max_dd_idx = i
            
            # Add annotations
            plt.annotate(f'Total Return: {total_return:.2%}', 
                        xy=(0.02, 0.95), xycoords='axes fraction')
            plt.annotate(f'Max Drawdown: {max_dd:.2%}', 
                        xy=(0.02, 0.90), xycoords='axes fraction')
            
            # Mark max drawdown on chart
            if max_dd > 0:
                plt.plot(timestamps[max_dd_idx], values[max_dd_idx], 'ro')
                plt.annotate('Max DD', 
                            xy=(timestamps[max_dd_idx], values[max_dd_idx]),
                            xytext=(10, -30),
                            textcoords='offset points',
                            arrowprops=dict(arrowstyle='->'))
        
        # Save or display
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Portfolio performance plot saved to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def export_to_csv(self, output_dir=None):
        """
        Export performance data to CSV files
        
        Args:
            output_dir: Directory to save CSV files (default: data_dir)
        """
        output_dir = output_dir or self.data_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Export trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_file = os.path.join(output_dir, 'trades.csv')
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"Exported trades to {trades_file}")
        
        # Export portfolio history
        if self.portfolio_history:
            portfolio_df = pd.DataFrame(self.portfolio_history)
            portfolio_file = os.path.join(output_dir, 'portfolio_history.csv')
            portfolio_df.to_csv(portfolio_file, index=False)
            logger.info(f"Exported portfolio history to {portfolio_file}")
        
        # Export metrics history
        if self.metrics_history:
            metrics_df = pd.DataFrame(self.metrics_history)
            metrics_file = os.path.join(output_dir, 'metrics_history.csv')
            metrics_df.to_csv(metrics_file, index=False)
            logger.info(f"Exported metrics history to {metrics_file}")
    
    def load_from_csv(self, input_dir=None):
        """
        Load performance data from CSV files
        
        Args:
            input_dir: Directory to load CSV files from (default: data_dir)
        """
        input_dir = input_dir or self.data_dir
        
        # Load trades
        trades_file = os.path.join(input_dir, 'trades.csv')
        if os.path.exists(trades_file):
            try:
                trades_df = pd.read_csv(trades_file)
                self.trades = trades_df.to_dict('records')
                logger.info(f"Loaded {len(self.trades)} trades from {trades_file}")
            except Exception as e:
                logger.error(f"Error loading trades from CSV: {e}")
        
        # Load portfolio history
        portfolio_file = os.path.join(input_dir, 'portfolio_history.csv')
        if os.path.exists(portfolio_file):
            try:
                portfolio_df = pd.read_csv(portfolio_file)
                self.portfolio_history = portfolio_df.to_dict('records')
                logger.info(f"Loaded portfolio history from {portfolio_file}")
            except Exception as e:
                logger.error(f"Error loading portfolio history from CSV: {e}")
        
        # Load metrics history
        metrics_file = os.path.join(input_dir, 'metrics_history.csv')
        if os.path.exists(metrics_file):
            try:
                metrics_df = pd.read_csv(metrics_file)
                self.metrics_history = metrics_df.to_dict('records')
                logger.info(f"Loaded metrics history from {metrics_file}")
            except Exception as e:
                logger.error(f"Error loading metrics history from CSV: {e}")
    
    def get_portfolio_history(self, timeframe='all'):
        """
        Get portfolio value history for a specific timeframe
        
        Args:
            timeframe: 'all', 'day', 'week', 'month', 'year'
        
        Returns:
            dict: Dictionary of {timestamp: value} pairs
        """
        # Determine start time based on timeframe
        if timeframe == 'day':
            start_time = (datetime.now() - timedelta(days=1)).isoformat()
        elif timeframe == 'week':
            start_time = (datetime.now() - timedelta(days=7)).isoformat()
        elif timeframe == 'month':
            start_time = (datetime.now() - timedelta(days=30)).isoformat()
        elif timeframe == 'year':
            start_time = (datetime.now() - timedelta(days=365)).isoformat()
        else:
            start_time = None
        
        # Get portfolio history for the timeframe
        if start_time:
            history = self.get_portfolio_value_history(start_time)
        else:
            history = self.portfolio_history
        
        # Convert to dictionary of {timestamp: value} pairs
        result = {}
        for entry in history:
            try:
                if isinstance(entry['timestamp'], str):
                    timestamp = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                else:
                    timestamp = entry['timestamp']
                result[timestamp] = entry['value']
            except Exception as e:
                logger.error(f"Error parsing portfolio history entry: {e}")
        
        return result
    
    def get_recent_trades(self, limit=20):
        """
        Get the most recent trades
        
        Args:
            limit: Maximum number of trades to return
        
        Returns:
            list: List of recent trades
        """
        # Sort trades by timestamp (newest first)
        sorted_trades = sorted(self.trades, key=lambda t: t['timestamp'], reverse=True)
        
        # Return the most recent trades
        return sorted_trades[:limit]
