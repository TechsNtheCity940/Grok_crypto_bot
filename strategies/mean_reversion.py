import numpy as np
import pandas as pd
import talib
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

from utils.log_setup import logger
from config_manager import config

class MeanReversionStrategy:
    """
    Mean reversion trading strategy that identifies overbought and oversold conditions
    and trades on the assumption that prices will revert to their mean.
    """
    def __init__(self, config_params=None):
        self.config = config_params or {}
        
        # Strategy parameters
        self.rsi_period = self.config.get('rsi_period', 14)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        
        self.bollinger_period = self.config.get('bollinger_period', 20)
        self.bollinger_std = self.config.get('bollinger_std', 2.0)
        
        self.ema_short = self.config.get('ema_short', 5)
        self.ema_long = self.config.get('ema_long', 20)
        
        # Minimum price movement required to trigger a trade (% of price)
        self.min_price_movement = self.config.get('min_price_movement', 0.01)
        
        # Position sizing
        self.position_size = self.config.get('position_size', 0.1)  # 10% of available balance
        
        # Track active trades
        self.active_trades = {}
        
        # Track strategy performance
        self.performance_history = []
    
    def calculate_indicators(self, df):
        """
        Calculate technical indicators for mean reversion
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with added indicators
        """
        # Make a copy to avoid modifying the original
        df_indicators = df.copy()
        
        # RSI
        df_indicators['rsi'] = talib.RSI(df_indicators['close'], timeperiod=self.rsi_period)
        
        # Bollinger Bands
        df_indicators['bb_upper'], df_indicators['bb_middle'], df_indicators['bb_lower'] = talib.BBANDS(
            df_indicators['close'], 
            timeperiod=self.bollinger_period, 
            nbdevup=self.bollinger_std, 
            nbdevdn=self.bollinger_std
        )
        
        # Bollinger Band Width
        df_indicators['bb_width'] = (df_indicators['bb_upper'] - df_indicators['bb_lower']) / df_indicators['bb_middle']
        
        # Bollinger Band %B (position within bands)
        df_indicators['bb_pct_b'] = (df_indicators['close'] - df_indicators['bb_lower']) / (df_indicators['bb_upper'] - df_indicators['bb_lower'])
        
        # EMAs
        df_indicators['ema_short'] = talib.EMA(df_indicators['close'], timeperiod=self.ema_short)
        df_indicators['ema_long'] = talib.EMA(df_indicators['close'], timeperiod=self.ema_long)
        
        # Distance from EMAs
        df_indicators['ema_distance'] = (df_indicators['close'] - df_indicators['ema_long']) / df_indicators['ema_long']
        
        # Stochastic Oscillator
        df_indicators['slowk'], df_indicators['slowd'] = talib.STOCH(
            df_indicators['high'], 
            df_indicators['low'], 
            df_indicators['close'], 
            fastk_period=14, 
            slowk_period=3, 
            slowd_period=3
        )
        
        return df_indicators
    
    def get_signal(self, df):
        """
        Generate trading signals based on mean reversion indicators
        
        Args:
            df: DataFrame with price data and indicators
        
        Returns:
            int: 1 for buy, 2 for sell, 0 for hold
        """
        if len(df) < self.bollinger_period:
            return 0  # Not enough data
        
        # Get the latest data point
        latest = df.iloc[-1]
        
        # Initialize signal
        signal = 0
        
        # Check for oversold conditions (buy signals)
        oversold_conditions = [
            latest['rsi'] < self.rsi_oversold,
            latest['close'] < latest['bb_lower'],
            latest['bb_pct_b'] < 0.05,
            latest['slowk'] < 20 and latest['slowk'] > latest['slowd']
        ]
        
        # Check for overbought conditions (sell signals)
        overbought_conditions = [
            latest['rsi'] > self.rsi_overbought,
            latest['close'] > latest['bb_upper'],
            latest['bb_pct_b'] > 0.95,
            latest['slowk'] > 80 and latest['slowk'] < latest['slowd']
        ]
        
        # Count how many conditions are met
        oversold_count = sum(oversold_conditions)
        overbought_count = sum(overbought_conditions)
        
        # Generate signal based on conditions
        if oversold_count >= 2:
            signal = 1  # Buy
        elif overbought_count >= 2:
            signal = 2  # Sell
        
        # Check if price movement is significant enough
        if signal != 0:
            # Get previous close
            prev_close = df.iloc[-2]['close'] if len(df) > 1 else latest['close']
            price_change = abs(latest['close'] - prev_close) / prev_close
            
            if price_change < self.min_price_movement:
                signal = 0  # Not enough price movement
        
        return signal
    
    def calculate_position_size(self, signal, price, balance_usd, balance_asset):
        """
        Calculate position size based on available balance and signal strength
        
        Args:
            signal: Trading signal (1=buy, 2=sell)
            price: Current price
            balance_usd: Available USD balance
            balance_asset: Available asset balance
        
        Returns:
            float: Position size in asset units
        """
        if signal == 1:  # Buy
            # Use a percentage of available USD
            return balance_usd * self.position_size / price
        elif signal == 2:  # Sell
            # Use a percentage of available asset
            return balance_asset * self.position_size
        else:
            return 0
    
    def update_active_trades(self, symbol, action, amount, price, timestamp):
        """
        Update active trades tracking
        
        Args:
            symbol: Trading pair symbol
            action: Trade action (1=buy, 2=sell)
            amount: Trade amount
            price: Trade price
            timestamp: Trade timestamp
        """
        if symbol not in self.active_trades:
            self.active_trades[symbol] = []
        
        self.active_trades[symbol].append({
            'action': action,
            'amount': amount,
            'price': price,
            'timestamp': timestamp
        })
        
        # Calculate profit if this is a closing trade
        if len(self.active_trades[symbol]) >= 2:
            last_trade = self.active_trades[symbol][-1]
            prev_trade = self.active_trades[symbol][-2]
            
            if last_trade['action'] != prev_trade['action']:
                # Calculate profit
                if last_trade['action'] == 2:  # Sell after buy
                    profit = (last_trade['price'] - prev_trade['price']) * min(last_trade['amount'], prev_trade['amount'])
                else:  # Buy after sell
                    profit = (prev_trade['price'] - last_trade['price']) * min(last_trade['amount'], prev_trade['amount'])
                
                # Record performance
                self.performance_history.append({
                    'symbol': symbol,
                    'entry_price': prev_trade['price'],
                    'exit_price': last_trade['price'],
                    'profit': profit,
                    'timestamp': timestamp
                })
                
                logger.info(f"Mean reversion trade completed for {symbol}: profit={profit:.2f}")
    
    def get_performance_metrics(self):
        """
        Calculate performance metrics for the strategy
        
        Returns:
            dict: Performance metrics
        """
        if not self.performance_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'total_profit': 0
            }
        
        total_trades = len(self.performance_history)
        winning_trades = sum(1 for trade in self.performance_history if trade['profit'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = sum(trade['profit'] for trade in self.performance_history)
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'total_profit': total_profit
        }
    
    def optimize_parameters(self, df, symbol):
        """
        Optimize strategy parameters based on historical data
        
        Args:
            df: DataFrame with historical price data
            symbol: Trading pair symbol
        
        Returns:
            dict: Optimized parameters
        """
        # Define parameter ranges to test
        rsi_periods = [7, 14, 21]
        rsi_overbought_levels = [65, 70, 75, 80]
        rsi_oversold_levels = [20, 25, 30, 35]
        bollinger_periods = [10, 20, 30]
        bollinger_stds = [1.5, 2.0, 2.5, 3.0]
        
        best_params = {}
        best_profit = -float('inf')
        
        # Simple grid search (could be improved with more sophisticated methods)
        for rsi_period in rsi_periods:
            for rsi_ob in rsi_overbought_levels:
                for rsi_os in rsi_oversold_levels:
                    for bb_period in bollinger_periods:
                        for bb_std in bollinger_stds:
                            # Set parameters
                            self.rsi_period = rsi_period
                            self.rsi_overbought = rsi_ob
                            self.rsi_oversold = rsi_os
                            self.bollinger_period = bb_period
                            self.bollinger_std = bb_std
                            
                            # Calculate indicators
                            df_indicators = self.calculate_indicators(df)
                            
                            # Simulate trading
                            balance = 1000  # Start with $1000
                            asset = 0
                            trades = []
                            
                            for i in range(self.bollinger_period, len(df_indicators)):
                                signal = self.get_signal(df_indicators.iloc[:i+1])
                                price = df_indicators.iloc[i]['close']
                                
                                if signal == 1 and balance > 0:  # Buy
                                    amount = balance / price
                                    asset += amount
                                    balance = 0
                                    trades.append({'action': 'buy', 'price': price, 'amount': amount})
                                elif signal == 2 and asset > 0:  # Sell
                                    balance += asset * price
                                    asset = 0
                                    trades.append({'action': 'sell', 'price': price, 'amount': asset})
                            
                            # Calculate final value
                            final_value = balance + asset * df_indicators.iloc[-1]['close']
                            profit = final_value - 1000
                            
                            if profit > best_profit:
                                best_profit = profit
                                best_params = {
                                    'rsi_period': rsi_period,
                                    'rsi_overbought': rsi_ob,
                                    'rsi_oversold': rsi_os,
                                    'bollinger_period': bb_period,
                                    'bollinger_std': bb_std
                                }
        
        # Update parameters with best values
        self.rsi_period = best_params['rsi_period']
        self.rsi_overbought = best_params['rsi_overbought']
        self.rsi_oversold = best_params['rsi_oversold']
        self.bollinger_period = best_params['bollinger_period']
        self.bollinger_std = best_params['bollinger_std']
        
        logger.info(f"Optimized mean reversion parameters for {symbol}: {best_params}")
        return best_params
