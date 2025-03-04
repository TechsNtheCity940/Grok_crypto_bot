import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import logging
from tqdm import tqdm
import talib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set up logging
logger = logging.getLogger('backtesting')

class BacktestEngine:
    """
    Comprehensive backtesting framework for evaluating trading strategies
    """
    def __init__(self, data_dir='data/historical', results_dir='models/backtest_results'):
        self.data_dir = data_dir
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize metrics
        self.metrics = {}
        
        # Default parameters
        self.initial_balance = 10000.0  # USD
        self.trading_fee = 0.001  # 0.1%
        self.slippage = 0.001  # 0.1%
        
    def load_data(self, symbol, timeframe='1h', start_date=None, end_date=None):
        """
        Load historical data for backtesting
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD')
            timeframe: Timeframe (e.g., '1h', '1d')
            start_date: Start date (optional)
            end_date: End date (optional)
        
        Returns:
            DataFrame with historical data
        """
        # Construct file path
        file_path = os.path.join(self.data_dir, f"{symbol.replace('/', '_')}_{timeframe}.csv")
        
        if not os.path.exists(file_path):
            logger.error(f"Data file not found: {file_path}")
            return None
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by date range
        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df['timestamp'] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df['timestamp'] <= end_date]
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        return df
    
    def add_indicators(self, df):
        """
        Add technical indicators to the dataframe
        
        Args:
            df: DataFrame with price data
        
        Returns:
            DataFrame with added indicators
        """
        # Make a copy to avoid modifying the original
        df_indicators = df.copy()
        
        # Add basic indicators
        df_indicators['ma_short'] = df_indicators['close'].rolling(window=5).mean()
        df_indicators['ma_long'] = df_indicators['close'].rolling(window=20).mean()
        df_indicators['rsi'] = talib.RSI(df_indicators['close'], timeperiod=14)
        df_indicators['macd'], df_indicators['macd_signal'], _ = talib.MACD(
            df_indicators['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        df_indicators['atr'] = talib.ATR(
            df_indicators['high'], df_indicators['low'], df_indicators['close'], timeperiod=14
        )
        df_indicators['bb_upper'], df_indicators['bb_middle'], df_indicators['bb_lower'] = talib.BBANDS(
            df_indicators['close'], timeperiod=20
        )
        
        # Add advanced indicators
        df_indicators['adx'] = talib.ADX(
            df_indicators['high'], df_indicators['low'], df_indicators['close'], timeperiod=14
        )
        df_indicators['cci'] = talib.CCI(
            df_indicators['high'], df_indicators['low'], df_indicators['close'], timeperiod=14
        )
        
        # Add volatility indicators
        df_indicators['historical_vol'] = df_indicators['close'].pct_change().rolling(window=20).std() * np.sqrt(365)
        
        # Add trend indicators
        df_indicators['psar'] = talib.SAR(df_indicators['high'], df_indicators['low'], acceleration=0.02, maximum=0.2)
        
        # Fill missing values
        df_indicators = df_indicators.fillna(0)
        
        return df_indicators
    
    def backtest_strategy(self, strategy, df, symbol, params=None, plot=True, save_results=True):
        """
        Backtest a trading strategy
        
        Args:
            strategy: Strategy object or function
            df: DataFrame with price data
            symbol: Trading pair symbol
            params: Strategy parameters (optional)
            plot: Whether to plot results (default: True)
            save_results: Whether to save results (default: True)
        
        Returns:
            Dictionary with backtest results
        """
        # Add indicators
        df_indicators = self.add_indicators(df)
        
        # Initialize strategy
        if params:
            strategy_instance = strategy(params)
        else:
            strategy_instance = strategy()
        
        # Initialize backtest variables
        balance_usd = self.initial_balance
        balance_asset = 0.0
        trades = []
        equity_curve = []
        
        # Get strategy name
        strategy_name = strategy_instance.__class__.__name__
        
        # Run backtest
        logger.info(f"Running backtest for {strategy_name} on {symbol}...")
        
        for i in tqdm(range(50, len(df_indicators))):
            # Get current price
            current_price = df_indicators.iloc[i]['close']
            current_time = df_indicators.iloc[i]['timestamp']
            
            # Get signal from strategy
            if hasattr(strategy_instance, 'get_signal'):
                # Standard strategy
                signal = strategy_instance.get_signal(df_indicators.iloc[:i+1])
                
                # Execute trade based on signal
                if signal == 1 and balance_usd > 0:  # Buy
                    # Calculate amount
                    amount = strategy_instance.calculate_position_size(
                        signal, current_price, balance_usd, balance_asset
                    )
                    
                    # Apply trading fee and slippage
                    effective_price = current_price * (1 + self.slippage)
                    fee = amount * effective_price * self.trading_fee
                    
                    # Execute trade if viable
                    if amount * effective_price + fee <= balance_usd:
                        balance_usd -= (amount * effective_price + fee)
                        balance_asset += amount
                        
                        # Record trade
                        trades.append({
                            'timestamp': current_time,
                            'type': 'buy',
                            'price': effective_price,
                            'amount': amount,
                            'fee': fee,
                            'balance_usd': balance_usd,
                            'balance_asset': balance_asset,
                            'equity': balance_usd + balance_asset * effective_price
                        })
                
                elif signal == 2 and balance_asset > 0:  # Sell
                    # Calculate amount
                    amount = strategy_instance.calculate_position_size(
                        signal, current_price, balance_usd, balance_asset
                    )
                    
                    # Apply trading fee and slippage
                    effective_price = current_price * (1 - self.slippage)
                    fee = amount * effective_price * self.trading_fee
                    
                    # Execute trade if viable
                    if amount <= balance_asset:
                        balance_usd += (amount * effective_price - fee)
                        balance_asset -= amount
                        
                        # Record trade
                        trades.append({
                            'timestamp': current_time,
                            'type': 'sell',
                            'price': effective_price,
                            'amount': amount,
                            'fee': fee,
                            'balance_usd': balance_usd,
                            'balance_asset': balance_asset,
                            'equity': balance_usd + balance_asset * effective_price
                        })
            
            elif hasattr(strategy_instance, 'get_grid_orders'):
                # Grid trading strategy
                grid_orders = strategy_instance.get_grid_orders(
                    current_price, balance_usd + balance_asset * current_price
                )
                
                # Execute grid orders
                for order in grid_orders:
                    order_type = order['type']
                    amount = order['amount']
                    
                    if order_type == 'buy' and balance_usd > 0:
                        # Apply trading fee and slippage
                        effective_price = current_price * (1 + self.slippage)
                        fee = amount * effective_price * self.trading_fee
                        
                        # Execute trade if viable
                        if amount * effective_price + fee <= balance_usd:
                            balance_usd -= (amount * effective_price + fee)
                            balance_asset += amount
                            
                            # Record trade
                            trades.append({
                                'timestamp': current_time,
                                'type': 'buy',
                                'price': effective_price,
                                'amount': amount,
                                'fee': fee,
                                'balance_usd': balance_usd,
                                'balance_asset': balance_asset,
                                'equity': balance_usd + balance_asset * effective_price
                            })
                    
                    elif order_type == 'sell' and balance_asset > 0:
                        # Apply trading fee and slippage
                        effective_price = current_price * (1 - self.slippage)
                        fee = amount * effective_price * self.trading_fee
                        
                        # Execute trade if viable
                        if amount <= balance_asset:
                            balance_usd += (amount * effective_price - fee)
                            balance_asset -= amount
                            
                            # Record trade
                            trades.append({
                                'timestamp': current_time,
                                'type': 'sell',
                                'price': effective_price,
                                'amount': amount,
                                'fee': fee,
                                'balance_usd': balance_usd,
                                'balance_asset': balance_asset,
                                'equity': balance_usd + balance_asset * effective_price
                            })
                
                # Update grids
                strategy_instance.update_grids_by_price(current_price)
            
            # Record equity
            equity = balance_usd + balance_asset * current_price
            equity_curve.append({
                'timestamp': current_time,
                'equity': equity,
                'price': current_price
            })
        
        # Calculate final results
        final_equity = balance_usd + balance_asset * df_indicators.iloc[-1]['close']
        total_return = (final_equity - self.initial_balance) / self.initial_balance
        
        # Calculate trade metrics
        trade_returns = []
        winning_trades = 0
        losing_trades = 0
        
        for i in range(1, len(trades)):
            if trades[i]['type'] != trades[i-1]['type']:
                if trades[i]['type'] == 'sell' and trades[i-1]['type'] == 'buy':
                    # Buy then sell
                    trade_return = (trades[i]['price'] - trades[i-1]['price']) / trades[i-1]['price']
                    trade_returns.append(trade_return)
                    
                    if trade_return > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
                
                elif trades[i]['type'] == 'buy' and trades[i-1]['type'] == 'sell':
                    # Sell then buy
                    trade_return = (trades[i-1]['price'] - trades[i]['price']) / trades[i]['price']
                    trade_returns.append(trade_return)
                    
                    if trade_return > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
        
        # Calculate performance metrics
        total_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate Sharpe ratio
        if trade_returns:
            avg_return = np.mean(trade_returns)
            std_return = np.std(trade_returns)
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        else:
            avg_return = 0
            std_return = 0
            sharpe_ratio = 0
        
        # Calculate drawdown
        equity_values = [point['equity'] for point in equity_curve]
        max_drawdown = self._calculate_max_drawdown(equity_values)
        
        # Calculate Sortino ratio
        negative_returns = [r for r in trade_returns if r < 0]
        downside_deviation = np.std(negative_returns) if negative_returns else 0
        sortino_ratio = avg_return / downside_deviation if downside_deviation > 0 else 0
        
        # Calculate Calmar ratio
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum([r for r in trade_returns if r > 0])
        gross_loss = abs(sum([r for r in trade_returns if r < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate average trade metrics
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0
        avg_winning_trade = np.mean([r for r in trade_returns if r > 0]) if winning_trades > 0 else 0
        avg_losing_trade = np.mean([r for r in trade_returns if r < 0]) if losing_trades > 0 else 0
        
        # Calculate expectancy
        expectancy = (win_rate * avg_winning_trade) - ((1 - win_rate) * abs(avg_losing_trade))
        
        # Calculate recovery factor
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else float('inf')
        
        # Calculate risk-adjusted return
        risk_adjusted_return = total_return / (std_return * np.sqrt(len(trade_returns))) if std_return > 0 and trade_returns else 0
        
        # Calculate buy & hold return
        buy_hold_return = (df_indicators.iloc[-1]['close'] - df_indicators.iloc[0]['close']) / df_indicators.iloc[0]['close']
        
        # Compile results
        results = {
            'strategy': strategy_name,
            'symbol': symbol,
            'timeframe': df_indicators.iloc[1]['timestamp'] - df_indicators.iloc[0]['timestamp'],
            'start_date': df_indicators.iloc[0]['timestamp'],
            'end_date': df_indicators.iloc[-1]['timestamp'],
            'initial_balance': self.initial_balance,
            'final_balance': final_equity,
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'recovery_factor': recovery_factor,
            'risk_adjusted_return': risk_adjusted_return,
            'buy_hold_return': buy_hold_return,
            'trades': trades,
            'equity_curve': equity_curve,
            'params': params
        }
        
        # Save results
        if save_results:
            self._save_backtest_results(results)
        
        # Plot results
        if plot:
            self._plot_backtest_results(results)
        
        return results
    
    def compare_strategies(self, strategies, df, symbol, params_list=None, plot=True, save_results=True):
        """
        Compare multiple trading strategies
        
        Args:
            strategies: List of strategy objects or functions
            df: DataFrame with price data
            symbol: Trading pair symbol
            params_list: List of strategy parameters (optional)
            plot: Whether to plot results (default: True)
            save_results: Whether to save results (default: True)
        
        Returns:
            Dictionary with comparison results
        """
        # Initialize results
        results = []
        
        # Run backtest for each strategy
        for i, strategy in enumerate(strategies):
            params = params_list[i] if params_list and i < len(params_list) else None
            result = self.backtest_strategy(strategy, df, symbol, params, plot=False, save_results=False)
            results.append(result)
        
        # Compile comparison results
        comparison = {
            'symbol': symbol,
            'start_date': df.iloc[0]['timestamp'],
            'end_date': df.iloc[-1]['timestamp'],
            'strategies': [r['strategy'] for r in results],
            'total_returns': [r['total_return'] for r in results],
            'sharpe_ratios': [r['sharpe_ratio'] for r in results],
            'max_drawdowns': [r['max_drawdown'] for r in results],
            'win_rates': [r['win_rate'] for r in results],
            'profit_factors': [r['profit_factor'] for r in results],
            'total_trades': [r['total_trades'] for r in results],
            'results': results
        }
        
        # Save comparison results
        if save_results:
            self._save_comparison_results(comparison)
        
        # Plot comparison results
        if plot:
            self._plot_comparison_results(comparison)
        
        return comparison
    
    def optimize_strategy(self, strategy, df, symbol, param_grid, metric='sharpe_ratio', n_splits=5):
        """
        Optimize strategy parameters using grid search
        
        Args:
            strategy: Strategy object or function
            df: DataFrame with price data
            symbol: Trading pair symbol
            param_grid: Dictionary with parameter grid
            metric: Metric to optimize (default: 'sharpe_ratio')
            n_splits: Number of time series splits for validation
        
        Returns:
            Dictionary with optimization results
        """
        # Generate parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        
        # Initialize results
        optimization_results = []
        
        # Split data for time series cross-validation
        split_indices = self._time_series_split(len(df), n_splits)
        
        # Run optimization
        logger.info(f"Optimizing {strategy.__name__} on {symbol} with {len(param_combinations)} parameter combinations...")
        
        for params in tqdm(param_combinations):
            # Initialize metrics for this parameter set
            metrics_values = []
            
            # Run backtest on each split
            for train_idx, test_idx in split_indices:
                train_df = df.iloc[train_idx]
                test_df = df.iloc[test_idx]
                
                # Run backtest on test set
                result = self.backtest_strategy(
                    strategy, test_df, symbol, params, plot=False, save_results=False
                )
                
                # Get metric value
                metric_value = result.get(metric, 0)
                metrics_values.append(metric_value)
            
            # Calculate average metric value
            avg_metric = np.mean(metrics_values)
            
            # Add to results
            optimization_results.append({
                'params': params,
                'avg_metric': avg_metric,
                'metric_values': metrics_values
            })
        
        # Sort results by average metric value
        optimization_results.sort(key=lambda x: x['avg_metric'], reverse=True)
        
        # Get best parameters
        best_params = optimization_results[0]['params']
        best_metric = optimization_results[0]['avg_metric']
        
        # Run final backtest with best parameters
        final_result = self.backtest_strategy(
            strategy, df, symbol, best_params, plot=True, save_results=True
        )
        
        # Compile optimization results
        results = {
            'strategy': strategy.__name__,
            'symbol': symbol,
            'metric': metric,
            'best_params': best_params,
            'best_metric': best_metric,
            'all_results': optimization_results,
            'final_result': final_result
        }
        
        # Save optimization results
        self._save_optimization_results(results)
        
        return results
    
    def _calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown from equity curve"""
        max_dd = 0
        peak = equity_curve[0]
        
        for value in equity_curve:
            if value > peak:
                peak = value
            
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _time_series_split(self, n_samples, n_splits):
        """Generate indices for time series cross-validation"""
        indices = []
        
        # Calculate split size
        split_size = n_samples // (n_splits + 1)
        
        for i in range(n_splits):
            # Calculate train/test indices
            train_start = 0
            train_end = split_size * (i + 1)
            test_start = train_end
            test_end = min(test_start + split_size, n_samples)
            
            # Create index arrays
            train_indices = list(range(train_start, train_end))
            test_indices = list(range(test_start, test_end))
            
            indices.append((train_indices, test_indices))
        
        return indices
    
    def _save_backtest_results(self, results):
        """Save backtest results to file"""
        # Create directory for this symbol if it doesn't exist
        symbol_dir = os.path.join(self.results_dir, results['symbol'].replace('/', '_'))
        os.makedirs(symbol_dir, exist_ok=True)
        
        # Create filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{results['strategy']}_{timestamp}.json"
        
        # Create file path
        file_path = os.path.join(symbol_dir, filename)
        
        # Convert datetime objects to strings
        results_copy = results.copy()
        results_copy['start_date'] = results_copy['start_date'].isoformat()
        results_copy['end_date'] = results_copy['end_date'].isoformat()
        
        for trade in results_copy['trades']:
            trade['timestamp'] = trade['timestamp'].isoformat()
        
        for point in results_copy['equity_curve']:
            point['timestamp'] = point['timestamp'].isoformat()
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        logger.info(f"Saved backtest results to {file_path}")
    
    def _save_comparison_results(self, comparison):
        """Save strategy comparison results to file"""
        # Create directory for this symbol if it doesn't exist
        symbol_dir = os.path.join(self.results_dir, comparison['symbol'].replace('/', '_'))
        os.makedirs(symbol_dir, exist_ok=True)
        
        # Create filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        strategies_str = '_'.join(comparison['strategies'])
        filename = f"comparison_{strategies_str}_{timestamp}.json"
        
        # Create file path
        file_path = os.path.join(symbol_dir, filename)
        
        # Convert datetime objects to strings
        comparison_copy = comparison.copy()
        comparison_copy['start_date'] = comparison_copy['start_date'].isoformat()
        comparison_copy['end_date'] = comparison_copy['end_date'].isoformat()
        
        # Process results
        for result in comparison_copy['results']:
            result['start_date'] = result['start_date'].isoformat()
            result['end_date'] = result['end_date'].isoformat()
            
            for trade in result['trades']:
                trade['timestamp'] = trade['timestamp'].isoformat()
            
            for point in result['equity_curve']:
                point['timestamp'] = point['timestamp'].isoformat()
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(comparison_copy, f, indent=2)
        
        logger.info(f"Saved comparison results to {file_path}")
    
    def _save_optimization_results(self, results):
        """Save optimization results to file"""
        # Create directory for this symbol if it doesn't exist
        symbol_dir = os.path.join(self.results_dir, results['symbol'].replace('/', '_'))
        os.makedirs(symbol_dir, exist_ok=True)
        
        # Create filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"optimization_{results['strategy']}_{timestamp}.json"
        
        # Create file path
        file_path = os.path.join(symbol_dir, filename)
        
        # Convert datetime objects to strings in final result
        results_copy = results.copy()
        final_result = results_copy['final_result']
        
        final_result['start_date'] = final_result['start_date'].isoformat()
        final_result['end_date'] = final_result['end_date'].isoformat()
        
        for trade in final_result['trades']:
            trade['timestamp'] = trade['timestamp'].isoformat()
        
        for point in final_result['equity_curve']:
            point['timestamp'] = point['timestamp'].isoformat()
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        logger.info(f"Saved optimization results to {file_path}")
    
    def _plot_backtest_results(self, results):
        """Plot backtest results"""
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Extract data
        timestamps = [point['timestamp'] for point in results['equity_curve']]
        equity = [point['equity'] for point in results['equity_curve']]
        prices = [point['price'] for point in results['equity_curve']]
        
        # Normalize equity and price for comparison
        equity_norm = [e / equity[0] for e in equity]
        prices_norm = [p / prices[0] for p in prices]
        
        # Plot equity curve
        axs[0].plot(timestamps, equity, label='Portfolio Value', color='#6c5ce7', linewidth=2)
        axs[0].set_title(f"Backtest Results: {results['strategy']} on {results['symbol']}")
        axs[0].set_ylabel('Portfolio Value (USD)')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()
        
        # Plot normalized comparison
        axs[1].plot(timestamps, equity_norm, label='Strategy Return', color='#6c5ce7', linewidth=2)
        axs[1].plot(timestamps, prices_norm, label='Buy & Hold Return', color='#fdcb6e', linewidth=2)
        axs[1].set_ylabel('Normalized Return')
        axs[1].grid(True, alpha=0.3)
        axs[1].legend()
        
        # Plot drawdown
        max_equity = np.maximum.accumulate(equity)
        drawdown = [(max_eq - eq) / max_eq for max_eq, eq in zip(max_equity, equity)]
        
        axs[2].fill_between(timestamps, drawdown, color='#d63031', alpha=0.3)
        axs[2].plot(timestamps, drawdown, color='#d63031', linewidth=1)
        axs[2].set_ylabel('Drawdown')
        axs[2].set_xlabel('Date')
        axs[2].grid(True, alpha=0.3)
        
        # Add buy/sell markers
        for trade in results['trades']:
            if trade['type'] == 'buy':
                axs[0].scatter(trade['timestamp'], trade['equity'], color='#00b894', marker='^', s=100)
            else:
                axs[0].scatter(trade['timestamp'], trade['equity'], color='#d63031', marker='v', s=100)
        
        # Add text with key metrics
        metrics_text = (
            f"Total Return: {results['total_return']:.2%}\n"
            f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {results['max_drawdown']:.2%}\n"
            f"Win Rate: {results['win_rate']:.2%}\n"
            f"Profit Factor: {results['profit_factor']:.2f}\n"
            f"Total Trades: {results['total_trades']}"
        )
        
        # Add text box
        axs[0].text(
            0.02, 0.05, metrics_text,
            transform=axs[0].transAxes,
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'),
            fontsize=10,
            verticalalignment='bottom'
        )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        symbol_dir = os.path.join(self.results_dir, results['symbol'].replace('/', '_'))
        os.makedirs(symbol_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{results['strategy']}_{timestamp}.png"
        file_path = os.path.join(symbol_dir, filename)
        
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved backtest plot to {file_path}")
        
        # Show plot
        plt.show()
    
    def _plot_comparison_results(self, comparison):
        """Plot strategy comparison results"""
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract data
        strategies = comparison['strategies']
        returns = comparison['total_returns']
        sharpes = comparison['sharpe_ratios']
        drawdowns = comparison['max_drawdowns']
        win_rates = comparison['win_rates']
        
        # Plot returns
        axs[0, 0].bar(strategies, returns, color='#6c5ce7')
        axs[0, 0].set_title('Total Return')
        axs[0, 0].set_ylabel('Return')
        axs[0, 0].grid(True, alpha=0.3)
        
        # Plot Sharpe ratios
        axs[0, 1].bar(strategies, sharpes, color='#00b894')
        axs[0, 1].set_title('Sharpe Ratio')
        axs[0, 1].set_ylabel('Sharpe Ratio')
        axs[0, 1].grid(True, alpha=0.3)
        
        # Plot drawdowns
        axs[1, 0].bar(strategies, drawdowns, color='#d63031')
        axs[1, 0].set_title('Max Drawdown')
        axs[1, 0].set_ylabel('Drawdown')
        axs[1, 0].grid(True, alpha=0.3)
        
        # Plot win rates
        axs[1, 1].bar(strategies, win_rates, color='#fdcb6e')
        axs[1, 1].set_title('Win Rate')
        axs[1, 1].set_ylabel('Win Rate')
        axs[1, 1].grid(True, alpha=0.3)
        
        # Add title
        fig.suptitle(f"Strategy Comparison: {comparison['symbol']}", fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save figure
        symbol_dir = os.path.join(self.results_dir, comparison['symbol'].replace('/', '_'))
        os.makedirs(symbol_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        strategies_str = '_'.join(comparison['strategies'])
        filename = f"comparison_{strategies_str}_{timestamp}.png"
        file_path = os.path.join(symbol_dir, filename)
        
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {file_path}")
        
        # Show plot
        plt.show()
    
    def plot_equity_curves(self, results_list, title=None):
        """
        Plot equity curves for multiple backtest results
        
        Args:
            results_list: List of backtest results
            title: Plot title (optional)
        """
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot equity curves
        for results in results_list:
            # Extract data
            timestamps = [point['timestamp'] for point in results['equity_curve']]
            equity = [point['equity'] for point in results['equity_curve']]
            
            # Normalize equity
            equity_norm = [e / equity[0] for e in equity]
            
            # Plot equity curve
            plt.plot(timestamps, equity_norm, label=f"{results['strategy']} on {results['symbol']}", linewidth=2)
        
        # Add buy & hold reference
        if results_list:
            # Use first results for reference
            results = results_list[0]
            timestamps = [point['timestamp'] for point in results['equity_curve']]
            prices = [point['price'] for point in results['equity_curve']]
            
            # Normalize prices
            prices_norm = [p / prices[0] for p in prices]
            
            # Plot buy & hold
            plt.plot(timestamps, prices_norm, label='Buy & Hold', color='#fdcb6e', linewidth=2, linestyle='--')
        
        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Normalized Return')
        plt.title(title or 'Strategy Comparison')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Show plot
        plt.tight_layout()
        plt.show()
    
    def plot_drawdowns(self, results_list, title=None):
        """
        Plot drawdowns for multiple backtest results
        
        Args:
            results_list: List of backtest results
            title: Plot title (optional)
        """
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot drawdowns
        for results in results_list:
            # Extract data
            timestamps = [point['timestamp'] for point in results['equity_curve']]
            equity = [point['equity'] for point in results['equity_curve']]
            
            # Calculate drawdown
            max_equity = np.maximum.accumulate(equity)
            drawdown = [(max_eq - eq) / max_eq for max_eq, eq in zip(max_equity, equity)]
            
            # Plot drawdown
            plt.plot(timestamps, drawdown, label=f"{results['strategy']} on {results['symbol']}", linewidth=2)
        
        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.title(title or 'Drawdown Comparison')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Show plot
        plt.tight_layout()
        plt.show()
    
    def plot_monthly_returns(self, results, title=None):
        """
        Plot monthly returns heatmap
        
        Args:
            results: Backtest results
            title: Plot title (optional)
        """
        # Extract data
        equity_curve = results['equity_curve']
        
        # Convert to DataFrame
        df = pd.DataFrame(equity_curve)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate daily returns
        df['daily_return'] = df['equity'].pct_change()
        
        # Group by month and year
        monthly_returns = df['daily_return'].groupby([df.index.year, df.index.month]).sum()
        
        # Reshape for heatmap
        monthly_returns = monthly_returns.unstack()
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(monthly_returns, annot=True, fmt='.2%', cmap='RdYlGn', center=0)
        
        # Add labels and title
        plt.title(title or f"Monthly Returns: {results['strategy']} on {results['symbol']}")
        plt.xlabel('Month')
        plt.ylabel('Year')
        
        # Show plot
        plt.tight_layout()
        plt.show()
    
    def plot_trade_distribution(self, results, title=None):
        """
        Plot trade return distribution
        
        Args:
            results: Backtest results
            title: Plot title (optional)
        """
        # Extract trades
        trades = results['trades']
        
        # Calculate trade returns
        trade_returns = []
        
        for i in range(1, len(trades)):
            if trades[i]['type'] != trades[i-1]['type']:
                if trades[i]['type'] == 'sell' and trades[i-1]['type'] == 'buy':
                    # Buy then sell
                    trade_return = (trades[i]['price'] - trades[i-1]['price']) / trades[i-1]['price']
                    trade_returns.append(trade_return)
                
                elif trades[i]['type'] == 'buy' and trades[i-1]['type'] == 'sell':
                    # Sell then buy
                    trade_return = (trades[i-1]['price'] - trades[i]['price']) / trades[i]['price']
                    trade_returns.append(trade_return)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot distribution
        sns.histplot(trade_returns, kde=True, bins=30)
        
        # Add mean and median lines
        if trade_returns:
            mean_return = np.mean(trade_returns)
            median_return = np.median(trade_returns)
            
            plt.axvline(mean_return, color='red', linestyle='--', label=f'Mean: {mean_return:.2%}')
            plt.axvline(median_return, color='green', linestyle='--', label=f'Median: {median_return:.2%}')
            plt.axvline(0, color='black', linestyle='-', label='Breakeven')
        
        # Add labels and title
        plt.xlabel('Trade Return')
        plt.ylabel('Frequency')
        plt.title(title or f"Trade Return Distribution: {results['strategy']} on {results['symbol']}")
        plt.legend()
        
        # Show plot
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, results, output_file=None):
        """
        Generate a comprehensive backtest report
        
        Args:
            results: Backtest results
            output_file: Output file path (optional)
        
        Returns:
            Report as a string
        """
        # Format metrics
        total_return = f"{results['total_return']:.2%}"
        annualized_return = f"{results['total_return'] / ((results['end_date'] - results['start_date']).days / 365):.2%}"
        sharpe_ratio = f"{results['sharpe_ratio']:.2f}"
        sortino_ratio = f"{results['sortino_ratio']:.2f}"
        calmar_ratio = f"{results['calmar_ratio']:.2f}"
        max_drawdown = f"{results['max_drawdown']:.2%}"
        win_rate = f"{results['win_rate']:.2%}"
        profit_factor = f"{results['profit_factor']:.2f}"
        expectancy = f"{results['expectancy']:.4f}"
        recovery_factor = f"{results['recovery_factor']:.2f}"
        risk_adjusted_return = f"{results['risk_adjusted_return']:.2f}"
        buy_hold_return = f"{results['buy_hold_return']:.2%}"
        
        # Create report
        report = f"""
        # Backtest Report: {results['strategy']} on {results['symbol']}
        
        ## Overview
        
        - **Strategy:** {results['strategy']}
        - **Symbol:** {results['symbol']}
        - **Period:** {results['start_date']} to {results['end_date']}
        - **Initial Balance:** ${results['initial_balance']:.2f}
        - **Final Balance:** ${results['final_balance']:.2f}
        
        ## Performance Metrics
        
        | Metric | Value |
        |--------|-------|
        | Total Return | {total_return} |
        | Annualized Return | {annualized_return} |
        | Sharpe Ratio | {sharpe_ratio} |
        | Sortino Ratio | {sortino_ratio} |
        | Calmar Ratio | {calmar_ratio} |
        | Max Drawdown | {max_drawdown} |
        | Win Rate | {win_rate} |
        | Profit Factor | {profit_factor} |
        | Expectancy | {expectancy} |
        | Recovery Factor | {recovery_factor} |
        | Risk-Adjusted Return | {risk_adjusted_return} |
        | Buy & Hold Return | {buy_hold_return} |
        
        ## Trade Statistics
        
        - **Total Trades:** {results['total_trades']}
        - **Winning Trades:** {results['winning_trades']}
        - **Losing Trades:** {results['losing_trades']}
        - **Average Trade Return:** {results['avg_return']:.2%}
        - **Average Winning Trade:** {results['avg_winning_trade']:.2%}
        - **Average Losing Trade:** {results['avg_losing_trade']:.2%}
        
        ## Strategy Parameters
        
        ```
        {results['params']}
        ```
        
        ## Conclusion
        
        The {results['strategy']} strategy {('outperformed' if results['total_return'] > results['buy_hold_return'] else 'underperformed')} the buy & hold approach by {abs(results['total_return'] - results['buy_hold_return']):.2%}.
        """
        
        # Save report if output file is provided
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            
            logger.info(f"Saved backtest report to {output_file}")
        
        return report
