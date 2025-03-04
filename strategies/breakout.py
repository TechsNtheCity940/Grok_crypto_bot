import numpy as np
import pandas as pd
import talib
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

from utils.log_setup import logger
from config_manager import config

class BreakoutStrategy:
    """
    Breakout trading strategy that identifies key support/resistance levels
    and trades on the breakout of these levels with momentum confirmation.
    """
    def __init__(self, config_params=None):
        self.config = config_params or {}
        
        # Strategy parameters
        self.lookback_period = self.config.get('lookback_period', 20)
        self.volume_factor = self.config.get('volume_factor', 1.5)  # Volume increase required for valid breakout
        self.atr_period = self.config.get('atr_period', 14)
        self.atr_multiplier = self.config.get('atr_multiplier', 1.0)  # For stop loss
        
        # Breakout confirmation parameters
        self.min_touches = self.config.get('min_touches', 2)  # Minimum touches of support/resistance
        self.confirmation_candles = self.config.get('confirmation_candles', 1)  # Candles to confirm breakout
        
        # Position sizing
        self.position_size = self.config.get('position_size', 0.1)  # 10% of available balance
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 0.02)  # 2% max risk per trade
        
        # Track active trades
        self.active_trades = {}
        
        # Track identified levels
        self.support_levels = {}
        self.resistance_levels = {}
        
        # Track strategy performance
        self.performance_history = []
    
    def calculate_indicators(self, df):
        """
        Calculate technical indicators for breakout strategy
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with added indicators
        """
        # Make a copy to avoid modifying the original
        df_indicators = df.copy()
        
        # ATR for volatility measurement
        df_indicators['atr'] = talib.ATR(
            df_indicators['high'], 
            df_indicators['low'], 
            df_indicators['close'], 
            timeperiod=self.atr_period
        )
        
        # ADX for trend strength
        df_indicators['adx'] = talib.ADX(
            df_indicators['high'], 
            df_indicators['low'], 
            df_indicators['close'], 
            timeperiod=14
        )
        
        # MACD for momentum confirmation
        df_indicators['macd'], df_indicators['macd_signal'], df_indicators['macd_hist'] = talib.MACD(
            df_indicators['close'], 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        
        # Volume moving average
        df_indicators['volume_ma'] = df_indicators['volume'].rolling(window=self.lookback_period).mean()
        
        # Relative volume
        df_indicators['relative_volume'] = df_indicators['volume'] / df_indicators['volume_ma']
        
        # Donchian channels
        df_indicators['dc_upper'] = df_indicators['high'].rolling(window=self.lookback_period).max()
        df_indicators['dc_lower'] = df_indicators['low'].rolling(window=self.lookback_period).min()
        df_indicators['dc_middle'] = (df_indicators['dc_upper'] + df_indicators['dc_lower']) / 2
        
        return df_indicators
    
    def identify_support_resistance(self, df):
        """
        Identify support and resistance levels
        
        Args:
            df: DataFrame with price data
        
        Returns:
            tuple: (support_levels, resistance_levels)
        """
        if len(df) < self.lookback_period:
            return [], []
        
        # Use the last lookback_period candles
        recent_df = df.iloc[-self.lookback_period:]
        
        # Find swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(recent_df) - 2):
            # Swing high
            if (recent_df.iloc[i]['high'] > recent_df.iloc[i-1]['high'] and 
                recent_df.iloc[i]['high'] > recent_df.iloc[i-2]['high'] and
                recent_df.iloc[i]['high'] > recent_df.iloc[i+1]['high'] and
                recent_df.iloc[i]['high'] > recent_df.iloc[i+2]['high']):
                swing_highs.append((recent_df.index[i], recent_df.iloc[i]['high']))
            
            # Swing low
            if (recent_df.iloc[i]['low'] < recent_df.iloc[i-1]['low'] and 
                recent_df.iloc[i]['low'] < recent_df.iloc[i-2]['low'] and
                recent_df.iloc[i]['low'] < recent_df.iloc[i+1]['low'] and
                recent_df.iloc[i]['low'] < recent_df.iloc[i+2]['low']):
                swing_lows.append((recent_df.index[i], recent_df.iloc[i]['low']))
        
        # Group nearby levels
        support_levels = self._group_levels([level for _, level in swing_lows])
        resistance_levels = self._group_levels([level for _, level in swing_highs])
        
        # Add Donchian channel levels
        if not recent_df.empty:
            support_levels.append(recent_df['dc_lower'].iloc[-1])
            resistance_levels.append(recent_df['dc_upper'].iloc[-1])
        
        return support_levels, resistance_levels
    
    def _group_levels(self, levels, tolerance=0.01):
        """Group nearby price levels"""
        if not levels:
            return []
        
        # Sort levels
        sorted_levels = sorted(levels)
        
        # Group nearby levels
        grouped_levels = []
        current_group = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # If level is within tolerance of the average of current group
            if abs(level - sum(current_group) / len(current_group)) / level < tolerance:
                current_group.append(level)
            else:
                # Add average of current group to grouped levels
                grouped_levels.append(sum(current_group) / len(current_group))
                current_group = [level]
        
        # Add the last group
        if current_group:
            grouped_levels.append(sum(current_group) / len(current_group))
        
        return grouped_levels
    
    def count_level_touches(self, df, level, tolerance=0.01):
        """Count how many times price has touched a level"""
        touches = 0
        
        for i in range(len(df)):
            # Check if high or low is within tolerance of level
            if (abs(df.iloc[i]['high'] - level) / level < tolerance or
                abs(df.iloc[i]['low'] - level) / level < tolerance):
                touches += 1
        
        return touches
    
    def get_signal(self, df):
        """
        Generate trading signals based on breakout indicators
        
        Args:
            df: DataFrame with price data and indicators
        
        Returns:
            tuple: (signal, stop_loss, take_profit)
                signal: 1 for buy, 2 for sell, 0 for hold
                stop_loss: Price level for stop loss
                take_profit: Price level for take profit
        """
        if len(df) < self.lookback_period:
            return 0, None, None  # Not enough data
        
        # Get the latest data
        latest = df.iloc[-1]
        
        # Identify support and resistance levels
        support_levels, resistance_levels = self.identify_support_resistance(df)
        
        # Initialize signal
        signal = 0
        stop_loss = None
        take_profit = None
        
        # Check for breakouts
        if support_levels and resistance_levels:
            # Find nearest support and resistance
            nearest_support = max([s for s in support_levels if s < latest['close']], default=None)
            nearest_resistance = min([r for r in resistance_levels if r > latest['close']], default=None)
            
            # Check for resistance breakout (buy signal)
            if nearest_resistance and latest['close'] > nearest_resistance:
                # Confirm with volume and momentum
                if (latest['relative_volume'] > self.volume_factor and
                    latest['macd_hist'] > 0 and
                    latest['adx'] > 25):  # Strong trend
                    
                    # Check if resistance has been tested multiple times
                    if self.count_level_touches(df.iloc[-self.lookback_period:-1], nearest_resistance) >= self.min_touches:
                        signal = 1  # Buy
                        stop_loss = latest['close'] - latest['atr'] * self.atr_multiplier
                        take_profit = latest['close'] + (latest['close'] - stop_loss) * 2  # 2:1 reward-risk ratio
            
            # Check for support breakdown (sell signal)
            elif nearest_support and latest['close'] < nearest_support:
                # Confirm with volume and momentum
                if (latest['relative_volume'] > self.volume_factor and
                    latest['macd_hist'] < 0 and
                    latest['adx'] > 25):  # Strong trend
                    
                    # Check if support has been tested multiple times
                    if self.count_level_touches(df.iloc[-self.lookback_period:-1], nearest_support) >= self.min_touches:
                        signal = 2  # Sell
                        stop_loss = latest['close'] + latest['atr'] * self.atr_multiplier
                        take_profit = latest['close'] - (stop_loss - latest['close']) * 2  # 2:1 reward-risk ratio
        
        return signal, stop_loss, take_profit
    
    def calculate_position_size(self, signal, price, stop_loss, balance_usd, balance_asset):
        """
        Calculate position size based on risk management rules
        
        Args:
            signal: Trading signal (1=buy, 2=sell)
            price: Current price
            stop_loss: Stop loss price level
            balance_usd: Available USD balance
            balance_asset: Available asset balance
        
        Returns:
            float: Position size in asset units
        """
        if signal == 0 or stop_loss is None:
            return 0
        
        if signal == 1:  # Buy
            # Calculate risk amount (USD)
            risk_amount = balance_usd * self.max_risk_per_trade
            
            # Calculate position size based on stop loss
            price_risk = price - stop_loss
            if price_risk <= 0:
                return 0
            
            # Position size = risk amount / price risk
            position_size = risk_amount / price_risk
            
            # Limit to max position size
            max_position = balance_usd * self.position_size / price
            position_size = min(position_size, max_position)
            
            return position_size
            
        elif signal == 2:  # Sell
            # Calculate risk amount (USD)
            portfolio_value = balance_asset * price
            risk_amount = portfolio_value * self.max_risk_per_trade
            
            # Calculate position size based on stop loss
            price_risk = stop_loss - price
            if price_risk <= 0:
                return 0
            
            # Position size = risk amount / price risk
            position_size = risk_amount / price_risk
            
            # Limit to max position size
            max_position = balance_asset * self.position_size
            position_size = min(position_size, max_position)
            
            return position_size
    
    def update_active_trades(self, symbol, action, amount, price, stop_loss, take_profit, timestamp=None):
        """
        Update active trades tracking
        
        Args:
            symbol: Trading pair symbol
            action: Trade action (1=buy, 2=sell)
            amount: Trade amount
            price: Trade price
            stop_loss: Stop loss price
            take_profit: Take profit price
            timestamp: Trade timestamp
        """
        timestamp = timestamp or datetime.now()
        
        if symbol not in self.active_trades:
            self.active_trades[symbol] = []
        
        self.active_trades[symbol].append({
            'action': action,
            'amount': amount,
            'price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'timestamp': timestamp,
            'status': 'open'
        })
        
        logger.info(f"Breakout trade opened for {symbol}: action={action}, price={price}, stop_loss={stop_loss}, take_profit={take_profit}")
    
    def check_trade_exits(self, symbol, current_price, timestamp=None):
        """
        Check if any active trades should be closed based on stop loss or take profit
        
        Args:
            symbol: Trading pair symbol
            current_price: Current price
            timestamp: Current timestamp
        
        Returns:
            list: Trades to close
        """
        timestamp = timestamp or datetime.now()
        trades_to_close = []
        
        if symbol not in self.active_trades:
            return trades_to_close
        
        for i, trade in enumerate(self.active_trades[symbol]):
            if trade['status'] != 'open':
                continue
            
            # Check stop loss
            if trade['action'] == 1 and current_price <= trade['stop_loss']:  # Buy trade
                trade['exit_price'] = current_price
                trade['exit_timestamp'] = timestamp
                trade['status'] = 'stopped'
                trade['profit'] = (current_price - trade['price']) * trade['amount']
                trades_to_close.append(trade)
                logger.info(f"Breakout trade stopped for {symbol}: price={current_price}, profit={trade['profit']:.2f}")
            
            elif trade['action'] == 2 and current_price >= trade['stop_loss']:  # Sell trade
                trade['exit_price'] = current_price
                trade['exit_timestamp'] = timestamp
                trade['status'] = 'stopped'
                trade['profit'] = (trade['price'] - current_price) * trade['amount']
                trades_to_close.append(trade)
                logger.info(f"Breakout trade stopped for {symbol}: price={current_price}, profit={trade['profit']:.2f}")
            
            # Check take profit
            elif trade['action'] == 1 and current_price >= trade['take_profit']:  # Buy trade
                trade['exit_price'] = current_price
                trade['exit_timestamp'] = timestamp
                trade['status'] = 'profit'
                trade['profit'] = (current_price - trade['price']) * trade['amount']
                trades_to_close.append(trade)
                logger.info(f"Breakout trade profit for {symbol}: price={current_price}, profit={trade['profit']:.2f}")
            
            elif trade['action'] == 2 and current_price <= trade['take_profit']:  # Sell trade
                trade['exit_price'] = current_price
                trade['exit_timestamp'] = timestamp
                trade['status'] = 'profit'
                trade['profit'] = (trade['price'] - current_price) * trade['amount']
                trades_to_close.append(trade)
                logger.info(f"Breakout trade profit for {symbol}: price={current_price}, profit={trade['profit']:.2f}")
        
        # Update performance history
        for trade in trades_to_close:
            self.performance_history.append({
                'symbol': symbol,
                'action': trade['action'],
                'entry_price': trade['price'],
                'exit_price': trade['exit_price'],
                'stop_loss': trade['stop_loss'],
                'take_profit': trade['take_profit'],
                'profit': trade['profit'],
                'status': trade['status'],
                'entry_timestamp': trade['timestamp'],
                'exit_timestamp': trade['exit_timestamp']
            })
        
        return trades_to_close
    
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
                'total_profit': 0,
                'profit_factor': 0
            }
        
        total_trades = len(self.performance_history)
        winning_trades = sum(1 for trade in self.performance_history if trade['profit'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_profit = sum(trade['profit'] for trade in self.performance_history)
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum(trade['profit'] for trade in self.performance_history if trade['profit'] > 0)
        gross_loss = abs(sum(trade['profit'] for trade in self.performance_history if trade['profit'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'total_profit': total_profit,
            'profit_factor': profit_factor
        }
