import time
import pandas as pd
import numpy as np
import os
import sys
import argparse
import torch
import tensorflow as tf
import gym
from gym import spaces
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import configuration
from config_manager import config

# Import utilities
from utils.log_setup import logger
from utils.enhanced_data_utils import fetch_and_process_enhanced_data, detect_market_regime

# Import execution components
from execution.trade_executor import TradeExecutor

# Import risk management
from risk_management import RiskManager, AdvancedRiskManager

# Import models
from models.model_factory import ModelFactory

# Import strategies
from strategies import GridTrader, MeanReversionStrategy, BreakoutStrategy, StrategySelector

# Import sentiment analysis
from sentiment_analyzer import SentimentAnalyzer
from sentiment import AdvancedSentimentAnalyzer

# Import performance monitoring
from monitoring import PerformanceTracker

# Import reinforcement learning
from stable_baselines3 import PPO

# Try to use GPU if available, otherwise use CPU
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch using device: {device}")
except Exception as e:
    print(f"Error setting PyTorch device: {e}")
    device = torch.device("cpu")
    print("PyTorch using CPU device")

# TensorFlow GPU setup
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"TensorFlow using GPU: {physical_devices[0]}")
    else:
        print("TensorFlow using CPU device")
except Exception as e:
    print(f"Error setting TensorFlow device: {e}")
    print("TensorFlow using CPU device")

class TradingEnv(gym.Env):
    """
    Enhanced trading environment for reinforcement learning
    """
    def __init__(self, df, symbol, executor, models, strategy_selector, risk_manager, performance_tracker=None):
        super(TradingEnv, self).__init__()
        self.df = df
        self.symbol = symbol
        self.executor = executor
        self.models = models  # Dict of models
        self.strategy_selector = strategy_selector
        self.risk_manager = risk_manager
        self.performance_tracker = performance_tracker
        
        self.current_step = 0
        self.balance_usd, self.balance_asset = self.executor.get_balance(self.symbol)
        self.initial_value = self.balance_usd + (self.balance_asset * self.df['close'].iloc[0])
        
        # Initialize active strategy
        self.active_strategy = self.strategy_selector.select_strategy(self.df, self.symbol)
        if hasattr(self.active_strategy, 'setup_grids'):
            self.active_strategy.setup_grids(self.df['close'].iloc[0], price_range=10.0)
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell

    def reset(self):
        self.current_step = 0
        
        # Get fresh balance from exchange
        self.balance_usd, self.balance_asset = self.executor.get_balance(self.symbol)
        
        # Store initial value for this episode
        current_price = self.df.iloc[0]['close'] if not self.df.empty else self.executor.fetch_current_price(self.symbol)
        self.initial_value = self.balance_usd + (self.balance_asset * current_price)
        
        # Log the initial state
        logger.info(f"TradingEnv reset - Initial USD: {self.balance_usd}, Initial {self.symbol.split('/')[0]}: {self.balance_asset}, Initial value: {self.initial_value}")
        
        return self._get_observation()

    def step(self, action):
        internal_action = action + 1 if action < 2 else 0  # Map to 1: buy, 2: sell, 0: hold
        
        # Get current price from dataframe or fetch from exchange if needed
        if self.current_step < len(self.df):
            price = self.df.iloc[self.current_step]['close']
        else:
            price = self.executor.fetch_current_price(self.symbol)
            
        reward = 0
        
        # Get fresh balance before any actions
        self.balance_usd, self.balance_asset = self.executor.get_balance(self.symbol)
        pre_value = self.balance_usd + (self.balance_asset * price)
        
        # Log current state
        logger.info(f"Step {self.current_step} - USD: {self.balance_usd}, {self.symbol.split('/')[0]}: {self.balance_asset}, Value: {pre_value}")
        
        # Get market regime
        regime_data = detect_market_regime(self.df.iloc[:self.current_step+1])
        current_regime = regime_data['regime'].iloc[-1] if not regime_data.empty else 'ranging'
        logger.info(f"Current market regime: {current_regime}")
        
        # Select best strategy for current regime
        self.active_strategy = self.strategy_selector.select_strategy(
            self.df.iloc[:self.current_step+1], 
            self.symbol
        )
        logger.info(f"Selected strategy: {self.active_strategy.__class__.__name__}")
        
        # Execute strategy-specific orders with improved balance tracking
        if hasattr(self.active_strategy, 'get_grid_orders'):
            # Grid trading strategy
            grid_orders = self.active_strategy.get_grid_orders(price, self.balance_usd + self.balance_asset * price)
            logger.info(f"Grid strategy generated {len(grid_orders)} orders")
            
            for order in grid_orders:
                order_type = 1 if order['type'] == 'buy' else 2
                
                # Get fresh balance before each order
                self.balance_usd, self.balance_asset = self.executor.get_balance(self.symbol)
                
                # Check if trade is safe with risk manager
                if self.risk_manager.is_safe(order_type, self.symbol, self.balance_usd, self.balance_asset, price):
                    logger.info(f"Executing grid order: {order['type']} {order['amount']} {self.symbol} @ {price}")
                    order_result, retry = self.executor.execute(order_type, self.symbol, order['amount'])
                    
                    if order_result:
                        self.active_strategy.update_grids(order)
                        
                        # Get fresh balance after order
                        self.balance_usd, self.balance_asset = self.executor.get_balance(self.symbol)
                        current_value = self.balance_usd + (self.balance_asset * price)
                        value_change = current_value - pre_value
                        reward += value_change / pre_value if pre_value > 0 else 0
                        
                        logger.info(f"Grid order executed - New USD: {self.balance_usd}, New {self.symbol.split('/')[0]}: {self.balance_asset}, Value change: {value_change}")
                        
                        # Log trade in performance tracker
                        if self.performance_tracker:
                            self.performance_tracker.log_trade(
                                self.symbol, 
                                order_type, 
                                order['amount'], 
                                price, 
                                strategy='grid_trading'
                            )
                    else:
                        logger.warning(f"Grid order execution failed: {order['type']} {order['amount']} {self.symbol}")
                else:
                    logger.warning(f"Grid order rejected by risk manager: {order['type']} {order['amount']} {self.symbol}")
        
        elif hasattr(self.active_strategy, 'get_signal'):
            # Mean reversion or breakout strategy
            if isinstance(self.active_strategy, BreakoutStrategy):
                signal, stop_loss, take_profit = self.active_strategy.get_signal(self.df.iloc[:self.current_step+1])
                if signal > 0:
                    # Calculate position size based on risk
                    amount = self.active_strategy.calculate_position_size(
                        signal, price, stop_loss, self.balance_usd, self.balance_asset
                    )
                    
                    # Execute trade if safe
                    if amount > 0 and self.risk_manager.is_safe(signal, self.symbol, self.balance_usd, self.balance_asset, price):
                        order_result, retry = self.executor.execute(signal, self.symbol, amount)
                        if order_result:
                            self.active_strategy.update_active_trades(
                                self.symbol, signal, amount, price, stop_loss, take_profit
                            )
                            self.balance_usd, self.balance_asset = self.executor.get_balance(self.symbol)
                            
                            # Log trade in performance tracker
                            if self.performance_tracker:
                                self.performance_tracker.log_trade(
                                    self.symbol, 
                                    signal, 
                                    amount, 
                                    price, 
                                    strategy='breakout',
                                    metadata={'stop_loss': stop_loss, 'take_profit': take_profit}
                                )
            else:
                # Mean reversion strategy
                signal = self.active_strategy.get_signal(self.df.iloc[:self.current_step+1])
                if signal > 0:
                    amount = self.active_strategy.calculate_position_size(
                        signal, price, self.balance_usd, self.balance_asset
                    )
                    
                    if amount > 0 and self.risk_manager.is_safe(signal, self.symbol, self.balance_usd, self.balance_asset, price):
                        order_result, retry = self.executor.execute(signal, self.symbol, amount)
                        if order_result:
                            self.active_strategy.update_active_trades(
                                self.symbol, signal, amount, price, datetime.now()
                            )
                            self.balance_usd, self.balance_asset = self.executor.get_balance(self.symbol)
                            
                            # Log trade in performance tracker
                            if self.performance_tracker:
                                self.performance_tracker.log_trade(
                                    self.symbol, 
                                    signal, 
                                    amount, 
                                    price, 
                                    strategy='mean_reversion'
                                )
        
        # Execute RL model action if provided with improved balance tracking
        if internal_action > 0:
            # Get fresh balance before RL action
            self.balance_usd, self.balance_asset = self.executor.get_balance(self.symbol)
            pre_rl_value = self.balance_usd + (self.balance_asset * price)
            
            # Calculate position size
            if isinstance(self.risk_manager, AdvancedRiskManager) and self.risk_manager.use_kelly:
                amount = self.risk_manager.calculate_position_size(
                    internal_action, self.symbol, self.balance_usd, price
                )
            else:
                # Default position sizing
                if internal_action == 1:  # Buy
                    amount = self.balance_usd * 0.1 / price  # 10% of USD balance
                else:  # Sell
                    amount = self.balance_asset * 0.1  # 10% of asset balance
            
            logger.info(f"RL model action: {internal_action} (1=buy, 2=sell), amount: {amount}")
            
            # Check if trade is safe
            if amount > 0 and self.risk_manager.is_safe(internal_action, self.symbol, self.balance_usd, self.balance_asset, price):
                order_result, retry = self.executor.execute(internal_action, self.symbol, amount)
                
                if order_result:
                    # Get fresh balance after RL action
                    self.balance_usd, self.balance_asset = self.executor.get_balance(self.symbol)
                    post_rl_value = self.balance_usd + (self.balance_asset * price)
                    rl_value_change = post_rl_value - pre_rl_value
                    
                    logger.info(f"RL action executed - New USD: {self.balance_usd}, New {self.symbol.split('/')[0]}: {self.balance_asset}, Value change: {rl_value_change}")
                    
                    # Add to reward
                    reward += rl_value_change / pre_rl_value if pre_rl_value > 0 else 0
                    
                    # Log trade in performance tracker
                    if self.performance_tracker:
                        self.performance_tracker.log_trade(
                            self.symbol, 
                            internal_action, 
                            amount, 
                            price, 
                            strategy='reinforcement_learning'
                        )
                else:
                    logger.warning(f"RL action execution failed: {internal_action} {amount} {self.symbol}")
            else:
                logger.warning(f"RL action rejected by risk manager: {internal_action} {amount} {self.symbol}")
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        if done:
            self.current_step = 0
        
        # Get fresh balance for final state
        self.balance_usd, self.balance_asset = self.executor.get_balance(self.symbol)
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate final portfolio value
        current_value = self.balance_usd + (self.balance_asset * price)
        
        # Calculate reward based on portfolio value change from initial value
        value_change = current_value - self.initial_value
        reward += value_change / self.initial_value if self.initial_value > 0 else 0
        
        # Log final state
        logger.info(f"Step complete - Final USD: {self.balance_usd}, Final {self.symbol.split('/')[0]}: {self.balance_asset}, Final value: {current_value}, Reward: {reward}")
        
        # Add risk-based penalties
        if current_value < self.initial_value * 0.9:  # More than 10% loss
            logger.warning(f"Large loss detected: {(current_value - self.initial_value) / self.initial_value:.2%}")
            reward -= 10
            done = True
        elif current_value > self.initial_value * 1.5:  # More than 50% gain
            logger.info(f"Large gain achieved: {(current_value - self.initial_value) / self.initial_value:.2%}")
            reward += 10
            done = True
        
        # Update portfolio value in performance tracker
        if self.performance_tracker:
            self.performance_tracker.update_portfolio_value(
                current_value,
                {self.symbol.split('/')[0]: self.balance_asset}
            )
            
            # Generate periodic performance report
            if self.current_step % 10 == 0:
                report = self.performance_tracker.get_latest_metrics()
                logger.info(f"Performance metrics - Win rate: {report.get('win_rate', 0):.2%}, Profit factor: {report.get('profit_factor', 0):.2f}")
        
        return obs, reward, done, {}

    def _get_observation(self):
        """Get enhanced observation with more features"""
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
            
        row = self.df.iloc[self.current_step]
        
        # Prepare feature data for models
        feature_columns = ['momentum', 'rsi', 'macd', 'atr', 'sentiment', 
                          'arbitrage_spread', 'whale_activity', 'bb_upper', 'defi_apr']
        
        X = self.df[feature_columns].iloc[max(0, self.current_step-49):self.current_step+1].values
        if len(X) < 50:
            X = np.pad(X, ((50 - len(X), 0), (0, 0)), mode='edge')
        
        # Get predictions from all models
        model_predictions = {}
        for model_type, model in self.models.items():
            try:
                if model_type == 'hybrid':
                    pred = model.predict(np.expand_dims(X, axis=0))[0][0].item()
                else:
                    pred = model.predict(np.expand_dims(X, axis=0))[0][0].item()
                model_predictions[model_type] = pred
            except Exception as e:
                print(f"Error getting prediction from {model_type} model: {e}")
                model_predictions[model_type] = 0.5
        
        # Get ensemble prediction with dynamic weighting
        if config.get('use_ensemble_weighting', False):
            # Weight models based on recent performance
            weights = {
                'hybrid': 0.4,
                'lstm': 0.3,
                'transformer': 0.3
            }
            ensemble_pred = sum(pred * weights.get(model_type, 0.33) 
                               for model_type, pred in model_predictions.items())
            ensemble_pred /= sum(weights.get(model_type, 0.33) 
                                for model_type in model_predictions.keys())
        else:
            # Simple average
            ensemble_pred = sum(model_predictions.values()) / len(model_predictions) if model_predictions else 0.5
        
        # Add volatility filter
        atr = row['atr']
        if atr < 0.005:  # Skip low volatility
            ensemble_pred = 0.5  # Neutral
        
        # Create observation with more features
        return np.array([
            ensemble_pred,  # Model prediction
            self.balance_usd,  # USD balance
            self.balance_asset,  # Asset balance
            row['sentiment'],  # Market sentiment
            row['arbitrage_spread']  # Arbitrage opportunity
        ], dtype=np.float32)

def optimize_portfolio(dataframes, risk_manager=None):
    """
    Enhanced portfolio optimization with risk considerations
    """
    # Calculate returns and volatilities
    returns = {s: df['close'].pct_change().mean() for s, df in dataframes.items()}
    vols = {s: df['close'].pct_change().std() for s, df in dataframes.items()}
    
    # Calculate correlation matrix if we have an advanced risk manager
    if isinstance(risk_manager, AdvancedRiskManager) and risk_manager.correlation_matrix is not None:
        # Use correlation-aware optimization
        weights = {}
        for symbol in dataframes.keys():
            # Start with Sharpe-based weight
            sharpe = returns[symbol] / vols[symbol] if vols[symbol] > 0 else 0
            
            # Adjust for correlation
            if symbol in risk_manager.correlation_matrix.index:
                # Reduce weight for highly correlated assets
                correlations = risk_manager.correlation_matrix[symbol].abs()
                avg_correlation = correlations.mean()
                correlation_factor = 1 - avg_correlation
                sharpe *= correlation_factor
            
            weights[symbol] = max(0, sharpe)  # Ensure non-negative
    else:
        # Traditional Sharpe ratio optimization
        total_sharpe = sum(r / v for r, v in zip(returns.values(), vols.values()) if v != 0)
        weights = {s: (r / v) / total_sharpe if v != 0 else 0 
                  for s, r, v in zip(returns.keys(), returns.values(), vols.values())}
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    else:
        # Equal weights if no valid Sharpe ratios
        weights = {s: 1.0 / len(dataframes) for s in dataframes.keys()}
    
    return weights

def main(args):
    """
    Enhanced main function with all improvements
    """
    # Load configuration
    active_exchange = config.get('active_exchange', 'kraken')
    trading_pairs = config.get('trading_pairs', ['DOGE/USD', 'SHIB/USD', 'XRP/USD'])
    model_dir = config.get('model_dir', 'models/trained_models')
    
    print(f"Starting Enhanced AI Crypto Trading Bot on {active_exchange}")
    logger.info(f"Starting Enhanced AI Crypto Trading Bot on {active_exchange}")
    
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize components
    executor = TradeExecutor()
    
    # Use advanced risk manager if enabled
    if config.get('advanced_risk', False):
        risk_manager = AdvancedRiskManager(max_loss=config.get('max_loss', 0.5))
        logger.info("Using advanced risk management")
    else:
        risk_manager = RiskManager(max_loss=config.get('max_loss', 0.5))
    
    # Use advanced sentiment analyzer if enabled
    if config.get('advanced_sentiment', False):
        sentiment_analyzer = AdvancedSentimentAnalyzer()
        logger.info("Using advanced sentiment analysis")
    else:
        sentiment_analyzer = SentimentAnalyzer()
    
    # Initialize performance tracker
    performance_tracker = PerformanceTracker() if config.get('use_performance_tracking', False) else None
    if performance_tracker:
        logger.info("Using performance tracking")
    
    # Initialize strategy selector
    available_strategies = ['grid_trading']
    if config.get('use_strategy_selector', False):
        available_strategies.extend(['mean_reversion', 'breakout'])
        logger.info(f"Using strategy selector with strategies: {available_strategies}")
    
    strategy_selector = StrategySelector(available_strategies)
    
    # Initialize data storage
    dataframes = {}
    models = {}
    ppo_models = {}
    
    # Initialize timing variables
    last_retrain_time = time.time()
    last_update_time = time.time()
    
    # Initial setup or load for each trading pair
    for symbol in trading_pairs:
        logger.info(f"Initializing {symbol}...")
        
        # Fetch and process data
        if config.get('use_enhanced_data', False):
            df_processed = fetch_and_process_enhanced_data(symbol)
            logger.info(f"Using enhanced data for {symbol}")
        else:
            from utils.data_utils import fetch_historical_data, augment_data, process_data
            df = fetch_historical_data(symbol)
            df_augmented = augment_data(df)
            df_processed = process_data(df_augmented, symbol)
        
        dataframes[symbol] = df_processed
        print(f"Initial data processed for {symbol}: {len(df_processed)} rows")
        
        # Initialize models for this symbol
        symbol_models = {}
        
        # Determine which model types to use
        model_types = config.get('model_types', ['hybrid', 'lstm'])
        
        # Train or load models
        for model_type in model_types:
            if args.trade_only:
                # Load existing model
                model = ModelFactory.load_model(model_type, symbol, model_dir)
                if model:
                    symbol_models[model_type] = model
                    logger.info(f"Loaded {model_type} model for {symbol}")
            else:
                # Train model
                model, accuracy = ModelFactory.train_model(model_type, symbol, df_processed, model_dir)
                symbol_models[model_type] = model
                logger.info(f"Trained {model_type} model for {symbol} with accuracy {accuracy:.2f}")
        
        models[symbol] = symbol_models
        
        # Initialize PPO model if needed
        if not args.trade_only:
            # Create trading environment
            env = TradingEnv(
                df_processed, 
                symbol, 
                executor, 
                symbol_models, 
                strategy_selector, 
                risk_manager,
                performance_tracker
            )
            
            # Train PPO model
            ppo_path = f'{model_dir}/ppo_{symbol.replace("/", "_")}'
            ppo_model = PPO('MlpPolicy', env, verbose=1)
            
            if os.path.exists(f"{ppo_path}.zip"):
                try:
                    ppo_model = PPO.load(f"{ppo_path}.zip", env=env)
                    logger.info(f"Loaded existing PPO model for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading PPO model: {e}")
                    ppo_model.learn(total_timesteps=1000)
                    ppo_model.save(ppo_path)
                    logger.info(f"Trained new PPO model for {symbol}")
            else:
                ppo_model.learn(total_timesteps=1000)
                ppo_model.save(ppo_path)
                logger.info(f"Trained new PPO model for {symbol}")
                
            ppo_models[symbol] = ppo_model
        else:
            # Load existing PPO model
            ppo_path = f'{model_dir}/ppo_{symbol.replace("/", "_")}'
            if os.path.exists(f"{ppo_path}.zip"):
                try:
                    env = TradingEnv(
                        df_processed, 
                        symbol, 
                        executor, 
                        symbol_models, 
                        strategy_selector, 
                        risk_manager,
                        performance_tracker
                    )
                    ppo_models[symbol] = PPO.load(f"{ppo_path}.zip", env=env)
                    logger.info(f"Loaded PPO model for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading PPO model: {e}")
                    ppo_models[symbol] = None
            else:
                ppo_models[symbol] = None
    
    # Main trading loop
    iteration = 0
    try:
        while True:
            # Full retraining every 24 hours (if not trade-only)
            retrain_interval = config.get('retrain_interval_hours', 24) * 3600
            if not args.trade_only and time.time() - last_retrain_time > retrain_interval:
                logger.info("Starting full model retraining...")
                
                for symbol in trading_pairs:
                    # Fetch and process fresh data
                    if config.get('use_enhanced_data', False):
                        df_processed = fetch_and_process_enhanced_data(symbol)
                    else:
                        from utils.data_utils import fetch_historical_data, augment_data, process_data
                        df = fetch_historical_data(symbol)
                        df_augmented = augment_data(df)
                        df_processed = process_data(df_augmented, symbol)
                    
                    dataframes[symbol] = df_processed
                    
                    # Retrain all models
                    symbol_models = {}
                    for model_type in config.get('model_types', ['hybrid', 'lstm']):
                        model, accuracy = ModelFactory.train_model(model_type, symbol, df_processed, model_dir)
                        symbol_models[model_type] = model
                        logger.info(f"Retrained {model_type} model for {symbol} with accuracy {accuracy:.2f}")
                    
                    models[symbol] = symbol_models
                    
                    # Retrain PPO model
                    env = TradingEnv(
                        df_processed, 
                        symbol, 
                        executor, 
                        symbol_models, 
                        strategy_selector, 
                        risk_manager,
                        performance_tracker
                    )
                    
                    ppo_path = f'{model_dir}/ppo_{symbol.replace("/", "_")}'
                    if symbol in ppo_models and ppo_models[symbol]:
                        # Continue training existing model
                        ppo_models[symbol].set_env(env)
                        ppo_models[symbol].learn(total_timesteps=1000)
                    else:
                        # Create new model
                        ppo_models[symbol] = PPO('MlpPolicy', env, verbose=1)
                        ppo_models[symbol].learn(total_timesteps=1000)
                    
                    ppo_models[symbol].save(ppo_path)
                    logger.info(f"Retrained PPO model for {symbol}")
                
                last_retrain_time = time.time()
                logger.info("Full model retraining completed")
            
            # Incremental update every hour
            update_interval = config.get('update_interval_hours', 1) * 3600
            if time.time() - last_update_time > update_interval:
                logger.info("Starting incremental model update...")
                
                for symbol in trading_pairs:
                    # Fetch new data
                    if config.get('use_enhanced_data', False):
                        from utils.enhanced_data_utils import fetch_real_time_data
                        new_data = fetch_real_time_data(symbol)
                    else:
                        from utils.data_utils import fetch_real_time_data
                        new_data = fetch_real_time_data(symbol)
                    
                    # Update dataframe
                    df = pd.concat([dataframes[symbol], new_data]).tail(100)
                    
                    # Process data
                    if config.get('use_enhanced_data', False):
                        from utils.enhanced_data_utils import enrich_data
                        df = enrich_data(df, symbol)
                    else:
                        from utils.data_utils import process_data
                        df = process_data(df, symbol)
                    
                    dataframes[symbol] = df
                    
                    # Fine-tune models with new data
                    feature_columns = ['momentum', 'rsi', 'macd', 'atr', 'sentiment', 
                                      'arbitrage_spread', 'whale_activity', 'bb_upper', 'defi_apr']
                    
                    X = [df[feature_columns].iloc[-50:].values]
                    y = [1 if (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] > 0.02 else 0]
                    
                    for model_type, model in models[symbol].items():
                        try:
                            if model_type in ['hybrid', 'lstm']:
                                model.model.fit(np.array(X), np.array(y), epochs=1, verbose=0)
                            elif model_type == 'transformer':
                                model.model.fit(np.array(X), np.array(y), epochs=1, verbose=0)
                        except Exception as e:
                            logger.error(f"Error fine-tuning {model_type} model: {e}")
                    
                    # Update PPO model
                    if symbol in ppo_models and ppo_models[symbol]:
                        try:
                            env = TradingEnv(
                                df, 
                                symbol, 
                                executor, 
                                models[symbol], 
                                strategy_selector, 
                                risk_manager,
                                performance_tracker
                            )
                            ppo_models[symbol].set_env(env)
                            ppo_models[symbol].learn(total_timesteps=100, reset_num_timesteps=False)
                        except Exception as e:
                            logger.error(f"Error updating PPO model: {e}")
                
                last_update_time = time.time()
                logger.info("Incremental model update completed")
            
            # Calculate total portfolio value
            total_usd, total_assets = 0, {}
            for symbol in trading_pairs:
                balance_usd, balance_asset = executor.get_balance(symbol)
                current_price = executor.fetch_current_price(symbol)
                total_usd += balance_usd
                total_assets[symbol.split('/')[0]] = balance_asset * current_price
            
            total_value = total_usd + sum(total_assets.values())
            print(f"Total account value: ${total_value:.2f}")
            logger.info(f"Total account value: ${total_value:.2f}")
            
            # Update portfolio value in performance tracker
            if performance_tracker:
                performance_tracker.update_portfolio_value(total_value, total_assets)
                
                # Generate performance report every 10 iterations
                if iteration % 10 == 0:
                    report = performance_tracker.generate_performance_report('day')
                    logger.info(f"Performance report: Win rate: {report['win_rate']:.2%}, "
                               f"Profit factor: {report['profit_factor']:.2f}, "
                               f"Total profit: ${report['total_profit']:.2f}")
            
            # Optimize portfolio allocation
            weights = optimize_portfolio(dataframes, risk_manager)
            
            # Get diversification recommendations if using advanced risk management
            if isinstance(risk_manager, AdvancedRiskManager):
                positions = {symbol: total_assets.get(symbol.split('/')[0], 0) for symbol in trading_pairs}
                diversification = risk_manager.get_diversification_recommendation(positions)
                if diversification and diversification['status'] == 'high_correlation':
                    logger.warning(f"Portfolio diversification warning: {diversification['message']}")
                    for rec in diversification.get('recommendations', []):
                        logger.warning(f"  {rec['recommendation']}")
            
            # Process each trading pair
            retry_pairs = []
            processed_pairs = set()
            
            for symbol in trading_pairs:
                if symbol in processed_pairs:
                    continue
                
                print(f"Processing {symbol}...")
                try:
                    # Fetch latest data
                    if config.get('use_enhanced_data', False):
                        from utils.enhanced_data_utils import fetch_real_time_data
                        new_data = fetch_real_time_data(symbol)
                    else:
                        from utils.data_utils import fetch_real_time_data
                        new_data = fetch_real_time_data(symbol)
                    
                    print(f"New data for {symbol}: {new_data}")
                    
                    # Update dataframe
                    df = pd.concat([dataframes[symbol], new_data]).tail(100)
                    
                    # Process data
                    if config.get('use_enhanced_data', False):
                        from utils.enhanced_data_utils import enrich_data
                        df = enrich_data(df, symbol)
                    else:
                        from utils.data_utils import process_data
                        df = process_data(df, symbol)
                    
                    dataframes[symbol] = df
                    latest = df.iloc[-1]
                    
                    # Get account balances
                    balance_usd, balance_asset = executor.get_balance(symbol)
                    current_price = latest['close']
                    
                    # Create trading environment
                    env = TradingEnv(
                        df, 
                        symbol, 
                        executor, 
                        models[symbol], 
                        strategy_selector, 
                        risk_manager,
                        performance_tracker
                    )
                    
                    # Get observation and determine action
                    obs = env.reset()
                    
                    # Use PPO model if available
                    if symbol in ppo_models and ppo_models[symbol]:
                        ppo_action, _ = ppo_models[symbol].predict(obs, deterministic=True)
                        action = int(ppo_action) + 1  # Convert to 1=buy, 2=sell
                    else:
                        # Fallback to simple threshold
                        action = 1 if obs[0] > 0.5 else 2
                    
                    print(f"Action chosen for {symbol}: {action} (1=buy, 2=sell)")
                    logger.info(f"Action for {symbol}: {action}, Balance USD: {balance_usd}, Balance {symbol.split('/')[0]}: {balance_asset}")
                    
                    # Get minimum trade size
                    min_trade_size = executor.min_trade_sizes.get(symbol, 10.0)
                    
                    # Execute trade based on action
                    if action == 1 and balance_usd > 0:  # Buy
                        # Calculate amount based on portfolio weights and risk
                        if isinstance(risk_manager, AdvancedRiskManager) and risk_manager.use_kelly:
                            amount = risk_manager.calculate_position_size(
                                action, symbol, balance_usd, current_price
                            )
                        else:
                            amount = min(balance_usd * weights[symbol] / current_price, balance_usd / current_price)
                        
                        # Check minimum trade size
                        if amount * current_price >= min_trade_size:
                            # Check if trade is safe
                            if risk_manager.is_safe(action, symbol, balance_usd, balance_asset, current_price):
                                order_result, retry = executor.execute(action, symbol, amount)
                                if order_result:
                                    logger.info(f"Executed order for {symbol}: {order_result}")
                                    
                                    # Log trade in performance tracker
                                    if performance_tracker:
                                        performance_tracker.log_trade(
                                            symbol, action, amount, current_price, 
                                            strategy='ensemble'
                                        )
                                    
                                    # Update environment
                                    env.step(action-1)
                                elif retry:
                                    retry_pairs.append(symbol)
                                    print(f"Added {symbol} to retry list due to execution issue")
                            else:
                                print(f"Trade skipped for {symbol}: Risk management prevented trade")
                        else:
                            print(f"Trade skipped for {symbol}: Amount {amount * current_price} below min size {min_trade_size}")
                    
                    elif action == 2 and balance_asset > 0:  # Sell
                        # Calculate amount based on portfolio weights and risk
                        if isinstance(risk_manager, AdvancedRiskManager):
                            amount = risk_manager.calculate_position_size(
                                action, symbol, balance_usd, current_price, balance_asset
                            )
                        else:
                            amount = min(balance_asset * weights[symbol], balance_asset)
                        
                        # Check minimum trade size
                        if amount >= min_trade_size / current_price:
                            # Check if trade is safe
                            if risk_manager.is_safe(action, symbol, balance_usd, balance_asset, current_price):
                                order_result, retry = executor.execute(action, symbol, amount)
                                if order_result:
                                    logger.info(f"Executed order for {symbol}: {order_result}")
                                    
                                    # Log trade in performance tracker
                                    if performance_tracker:
                                        performance_tracker.log_trade(
                                            symbol, action, amount, current_price, 
                                            strategy='ensemble'
                                        )
                                    
                                    # Update environment
                                    env.step(action-1)
                                elif retry:
                                    retry_pairs.append(symbol)
                                    print(f"Added {symbol} to retry list due to execution issue")
                            else:
                                print(f"Trade skipped for {symbol}: Risk management prevented trade")
                        else:
                            print(f"Trade skipped for {symbol}: Amount {amount} below min size {min_trade_size / current_price}")
                    else:
                        print(f"Trade skipped for {symbol}: No {['USD', symbol.split('/')[0]][action-1]} to trade")
                    
                    processed_pairs.add(symbol)
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
                    logger.error(f"Error processing {symbol}: {e}")
            
            # Retry failed trades
            for symbol in retry_pairs:
                print(f"Retrying {symbol} with adjusted parameters...")
                try:
                    balance_usd, balance_asset = executor.get_balance(symbol)
                    current_price = executor.fetch_current_price(symbol)
                    
                    # Create trading environment
                    env = TradingEnv(
                        dataframes[symbol], 
                        symbol, 
                        executor, 
                        models[symbol], 
                        strategy_selector, 
                        risk_manager,
                        performance_tracker
                    )
                    
                    # Get observation and determine action
                    obs = env.reset()
                    action = 1 if obs[0] > 0.5 else 2
                    
                    # Get minimum trade size
                    min_trade_size = executor.min_trade_sizes.get(symbol, 10.0)
                    
                    # Calculate amount with more aggressive sizing
                    if action == 1:  # Buy
                        amount = min(balance_usd * weights[symbol] * 1.5 / current_price, balance_usd / current_price)
                    else:  # Sell
                        amount = min(balance_asset * weights[symbol] * 1.5, balance_asset)
                    
                    # Ensure minimum trade size
                    amount = max(amount, min_trade_size / current_price if action == 1 else min_trade_size)
                    
                    # Execute trade if safe
                    if risk_manager.is_safe(action, symbol, balance_usd, balance_asset, current_price) and amount > 0:
                        order_result, retry = executor.execute(action, symbol, amount)
                        if order_result:
                            logger.info(f"Executed retry order for {symbol}: {order_result}")
                            
                            # Log trade in performance tracker
                            if performance_tracker:
                                performance_tracker.log_trade(
                                    symbol, action, amount, current_price, 
                                    strategy='retry'
                                )
                            
                            # Update environment
                            env.step(action-1)
                        else:
                            print(f"Retry failed for {symbol}: Insufficient funds or execution error")
                    else:
                        print(f"Retry skipped for {symbol}: Amount {amount} not viable or risk not safe")
                except Exception as e:
                    print(f"Error retrying {symbol}: {e}")
                    logger.error(f"Error retrying {symbol}: {e}")
            
            # Increment iteration counter
            iteration += 1
            
            # Update trading pairs periodically
            if iteration % 10 == 0:
                executor.trading_pairs = executor.update_trading_pairs()
            
            # Wait before next iteration
            print("Waiting 60 seconds...")
            time.sleep(60)
    except KeyboardInterrupt:
        print("Trading bot stopped by user")
        logger.info("Trading bot stopped by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}")
    finally:
        # Save final performance report if using performance tracking
        if performance_tracker:
            report = performance_tracker.generate_performance_report('all')
            logger.info(f"Final performance report: Win rate: {report['win_rate']:.2%}, "
                       f"Profit factor: {report['profit_factor']:.2f}, "
                       f"Total profit: ${report['total_profit']:.2f}")
            
            # Export data to CSV
            performance_tracker.export_to_csv()
            
            # Generate performance plot
            try:
                performance_tracker.plot_portfolio_performance(
                    output_file='data/performance/portfolio_performance.png',
                    timeframe='all'
                )
            except Exception as e:
                logger.error(f"Error generating performance plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the enhanced trading bot")
    parser.add_argument('--trade-only', action='store_true', help="Load existing models and trade without retraining")
    parser.add_argument('--config', type=str, default='config.json', help="Path to configuration file")
    args = parser.parse_args()
    
    main(args)
