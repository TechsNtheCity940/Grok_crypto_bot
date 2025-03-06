import time
import pandas as pd
import numpy as np
import os
import shutil
from utils.data_utils import fetch_real_time_data, process_data, fetch_historical_data, augment_data
from utils.log_setup import logger
from execution.trade_executor import TradeExecutor
from risk_management.risk_manager import RiskManager
from config import TRADING_PAIRS, ACTIVE_EXCHANGE
from models.hybrid_model import HybridCryptoModel
from models.lstm_model import LSTMModel
from models.backtest import backtest_model
from strategies.grid_trading import GridTrader
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from sentiment_analyzer import SentimentAnalyzer
from stable_baselines3 import PPO
import gym
from gym import spaces
import tensorflow as tf
import argparse
import torch

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

def train_hybrid_model(symbol, df, model_path):
    model = HybridCryptoModel(sequence_length=50, n_features=9)
    # TensorFlow models don't use PyTorch's .to(device) method
    # GPU usage is handled by TensorFlow's configuration
    if os.path.exists(model_path):
        model.model.load_weights(model_path)
        logger.info(f"Loaded existing hybrid model for {symbol} from {model_path}")
    else:
        X = []
        y_price = []
        for i in range(len(df) - 50):
            X.append(df[['momentum', 'rsi', 'macd', 'atr', 'sentiment', 'arbitrage_spread', 'whale_activity', 'bb_upper', 'defi_apr']].iloc[i:i+50].values)
            price_change = (df['close'].iloc[i+50] - df['close'].iloc[i+49]) / df['close'].iloc[i+49]
            y_price.append(1 if price_change > 0.02 else 0)
        X = np.array(X)
        y_price = np.array(y_price)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y_price[:split_idx], y_price[split_idx:]
        history = model.train(X_train, y_train, np.zeros_like(y_train), (X_val, y_val, np.zeros_like(y_val)), model_path=model_path)
        logger.info(f"Trained new hybrid model for {symbol}")
    accuracy = backtest_model(model, symbol, df)
    logger.info(f"Backtest accuracy for {symbol}: {accuracy:.2f}")
    return model, None

def train_lstm_model(symbol, df, model_path):
    model = LSTMModel(sequence_length=50, n_features=9)
    # TensorFlow models don't use PyTorch's .to(device) method
    # GPU usage is handled by TensorFlow's configuration
    if os.path.exists(model_path):
        model.model.load_weights(model_path)
        logger.info(f"Loaded existing LSTM model for {symbol} from {model_path}")
    else:
        X = []
        y_price = []
        for i in range(len(df) - 50):
            X.append(df[['momentum', 'rsi', 'macd', 'atr', 'sentiment', 'arbitrage_spread', 'whale_activity', 'bb_upper', 'defi_apr']].iloc[i:i+50].values)
            price_change = (df['close'].iloc[i+50] - df['close'].iloc[i+49]) / df['close'].iloc[i+49]
            y_price.append(1 if price_change > 0.02 else 0)
        X = np.array(X)
        y_price = np.array(y_price)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y_price[:split_idx], y_price[split_idx:]
        history = model.train(X_train, y_train, X_val, y_val, model_path=model_path)
        logger.info(f"Trained new LSTM model for {symbol}")
    accuracy = backtest_model(model, symbol, df)
    logger.info(f"LSTM backtest accuracy for {symbol}: {accuracy:.2f}")
    return model, None

def train_ppo_model(env, symbol, model_path):
    model = PPO('MlpPolicy', env, verbose=1)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    except Exception as e:
        print(f"Error setting PPO model device: {e}")
        print("Using default device for PPO model")
    
    if os.path.exists(f"{model_path}.zip"):
        try:
            model = PPO.load(f"{model_path}.zip", env=env)
            logger.info(f"Loaded existing PPO model for {symbol} from {model_path}.zip")
        except Exception as e:
            print(f"Error loading PPO model: {e}")
            print("Training new PPO model")
            model.learn(total_timesteps=1000)
            model.save(model_path)
            logger.info(f"Trained new PPO model for {symbol}")
    else:
        model.learn(total_timesteps=1000)
        model.save(model_path)
        logger.info(f"Trained new PPO model for {symbol}")
    return model

class TradingEnv(gym.Env):
    def __init__(self, df, symbol, executor, hybrid_model, lstm_model, ppo_model, grid_trader):
        super(TradingEnv, self).__init__()
        self.df = df
        self.symbol = symbol
        self.executor = executor
        self.hybrid_model = hybrid_model
        self.lstm_model = lstm_model
        self.ppo_model = ppo_model
        self.grid_trader = grid_trader
        self.current_step = 0
        self.balance_usd, self.balance_asset = self.executor.get_balance(self.symbol)
        self.initial_value = self.balance_usd + (self.balance_asset * self.df['close'].iloc[0])
        self.grid_trader.setup_grids(self.df['close'].iloc[0], price_range=10.0)
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self):
        self.current_step = 0
        
        # Get fresh balance from exchange
        self.balance_usd, self.balance_asset = self.executor.get_balance(self.symbol)
        
        # Store initial value for this episode
        current_price = self.df['close'].iloc[0] if len(self.df) > 0 else self.executor.fetch_current_price(self.symbol)
        self.initial_value = self.balance_usd + (self.balance_asset * current_price)
        
        # Log the initial state
        logger.info(f"TradingEnv reset - Initial USD: {self.balance_usd}, Initial {self.symbol.split('/')[0]}: {self.balance_asset}, Initial value: {self.initial_value}")
        
        return self._get_observation()

    def step(self, action):
        internal_action = action + 1
        
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
        
        # Execute grid trading orders with improved balance tracking
        grid_orders = self.grid_trader.get_grid_orders(price, self.balance_usd + self.balance_asset * price)
        logger.info(f"Grid strategy generated {len(grid_orders)} orders")
        
        for order in grid_orders:
            order_type = 1 if order['type'] == 'buy' else 2
            
            # Get fresh balance before each order
            self.balance_usd, self.balance_asset = self.executor.get_balance(self.symbol)
            
            logger.info(f"Executing grid order: {order['type']} {order['amount']} {self.symbol} @ {price}")
            order_result, retry = self.executor.execute(order_type, self.symbol, order['amount'])
            
            if order_result:
                self.grid_trader.update_grids(order)
                
                # Get fresh balance after order
                self.balance_usd, self.balance_asset = self.executor.get_balance(self.symbol)
                current_value = self.balance_usd + (self.balance_asset * price)
                value_change = current_value - pre_value
                reward += value_change / pre_value if pre_value > 0 else 0
                
                logger.info(f"Grid order executed - New USD: {self.balance_usd}, New {self.symbol.split('/')[0]}: {self.balance_asset}, Value change: {value_change}")
            else:
                logger.warning(f"Grid order execution failed: {order['type']} {order['amount']} {self.symbol}")
        
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
        if current_value < self.initial_value * 0.9:
            logger.warning(f"Large loss detected: {(current_value - self.initial_value) / self.initial_value:.2%}")
            reward -= 100
            done = True
        elif current_value > self.initial_value * 2:
            logger.info(f"Large gain achieved: {(current_value - self.initial_value) / self.initial_value:.2%}")
            reward += 200
            done = True
        
        return obs, reward, done, {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        X = self.df[['momentum', 'rsi', 'macd', 'atr', 'sentiment', 'arbitrage_spread', 'whale_activity', 'bb_upper', 'defi_apr']].iloc[max(0, self.current_step-49):self.current_step+1].values
        if len(X) < 50:
            X = np.pad(X, ((50 - len(X), 0), (0, 0)), mode='edge')
        hybrid_pred = self.hybrid_model.predict(np.expand_dims(X, axis=0))[0][0].item()
        lstm_pred = self.lstm_model.predict(np.expand_dims(X, axis=0))[0][0].item()
        ppo_pred = self.ppo_model.predict(observation=np.array([self.balance_usd, self.balance_asset, hybrid_pred]), deterministic=True)[0] if self.ppo_model else 0
        ensemble_pred = np.mean([hybrid_pred, lstm_pred, float(ppo_pred)])
        atr = row['atr']
        if atr < 0.005:  # Skip low volatility
            ensemble_pred = 0.5  # Neutral
        return np.array([ensemble_pred, self.balance_usd, self.balance_asset], dtype=np.float32)

def optimize_portfolio(dataframes):
    returns = {s: df['close'].pct_change().mean() for s, df in dataframes.items()}
    vols = {s: df['close'].pct_change().std() for s, df in dataframes.items()}
    total_sharpe = sum(r / v for r, v in zip(returns.values(), vols.values()) if v != 0)
    weights = {s: (r / v) / total_sharpe if v != 0 else 0 for s, r, v in zip(returns.keys(), returns.values(), vols.values())}
    return weights

def main(trade_only=False):
    print(f"Starting AI Crypto Trading Bot on {ACTIVE_EXCHANGE}")
    logger.info(f"Starting AI Crypto Trading Bot on {ACTIVE_EXCHANGE}")
    
    model_dir = 'models/trained_models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    executor = TradeExecutor()
    risk_manager = RiskManager(max_loss=0.5)
    
    dataframes = {}
    hybrid_models = {}
    lstm_models = {}
    ppo_models = {}
    grid_traders = {}
    last_retrain_time = time.time()
    last_update_time = time.time()

    # Initial setup or load
    for symbol in TRADING_PAIRS:
        df = fetch_historical_data(symbol)
        df_augmented = augment_data(df)
        df_processed = process_data(df_augmented, symbol)
        dataframes[symbol] = df_processed
        print(f"Initial historical data fetched and processed for {symbol}: {len(df_processed)} rows")
        
        hybrid_path = f'{model_dir}/hybrid_{symbol.replace("/", "_")}.h5'
        lstm_path = f'{model_dir}/lstm_{symbol.replace("/", "_")}.h5'
        ppo_path = f'{model_dir}/ppo_{symbol.replace("/", "_")}'
        
        hybrid_model, _ = train_hybrid_model(symbol, df_processed, hybrid_path)
        lstm_model, _ = train_lstm_model(symbol, df_processed, lstm_path)
        hybrid_models[symbol] = hybrid_model
        lstm_models[symbol] = lstm_model
        
        env = TradingEnv(df_processed, symbol, executor, hybrid_model, lstm_model, None, GridTrader({'grid_trading': {}}))
        ppo_models[symbol] = train_ppo_model(env, symbol, ppo_path) if not trade_only else PPO.load(f"{ppo_path}.zip", env=env)
        if trade_only:
            logger.info(f"Loaded models for {symbol} without retraining")
        
        grid_config = {'grid_trading': {'num_grids': 10, 'grid_spread': 0.05, 'max_position': 1.0, 'min_profit': 0.2}}
        grid_traders[symbol] = GridTrader(grid_config)

    iteration = 0
    while True:
        # Full retraining every 24 hours (if not trade-only)
        if not trade_only and time.time() - last_retrain_time > 24 * 3600:
            for symbol in TRADING_PAIRS:
                df = fetch_historical_data(symbol)
                df_augmented = augment_data(df)
                df_processed = process_data(df_augmented, symbol)
                dataframes[symbol] = df_processed
                
                hybrid_path = f'{model_dir}/hybrid_{symbol.replace("/", "_")}.h5'
                lstm_path = f'{model_dir}/lstm_{symbol.replace("/", "_")}.h5'
                ppo_path = f'{model_dir}/ppo_{symbol.replace("/", "_")}'
                
                hybrid_model, _ = train_hybrid_model(symbol, df_processed, hybrid_path)
                lstm_model, _ = train_lstm_model(symbol, df_processed, lstm_path)
                hybrid_models[symbol] = hybrid_model
                lstm_models[symbol] = lstm_model
                
                env = TradingEnv(df_processed, symbol, executor, hybrid_model, lstm_model, ppo_models[symbol], grid_traders[symbol])
                ppo_models[symbol] = train_ppo_model(env, symbol, ppo_path)
            last_retrain_time = time.time()
            logger.info("Models retrained with new data")

        # Incremental update every hour
        if time.time() - last_update_time > 3600:
            for symbol in TRADING_PAIRS:
                new_data = fetch_real_time_data(symbol)
                df = pd.concat([dataframes[symbol], new_data]).tail(100)
                df = process_data(df, symbol)
                dataframes[symbol] = df
                
                X = [df[['momentum', 'rsi', 'macd', 'atr', 'sentiment', 'arbitrage_spread', 'whale_activity', 'bb_upper', 'defi_apr']].iloc[-50:].values]
                y = [1 if (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] > 0.02 else 0]
                hybrid_models[symbol].model.fit(np.array(X), np.array(y), epochs=1, verbose=0)
                lstm_models[symbol].model.fit(np.array(X), np.array(y), epochs=1, verbose=0)
                env = TradingEnv(df, symbol, executor, hybrid_models[symbol], lstm_models[symbol], ppo_models[symbol], grid_traders[symbol])
                ppo_models[symbol].learn(total_timesteps=100, reset_num_timesteps=False)
            last_update_time = time.time()
            logger.info("Models fine-tuned with new data")

        total_usd, total_assets = 0, {}
        for symbol in TRADING_PAIRS:
            balance_usd, balance_asset = executor.get_balance(symbol)
            current_price = executor.fetch_current_price(symbol)
            total_usd += balance_usd
            total_assets[symbol.split('/')[0]] = balance_asset * current_price
        
        total_value = total_usd + sum(total_assets.values())
        print(f"Total account value: ${total_value:.2f}")
        logger.info(f"Total account value: ${total_value:.2f}")

        weights = optimize_portfolio(dataframes)
        retry_pairs = []
        processed_pairs = set()

        for symbol in TRADING_PAIRS:
            if symbol in processed_pairs:
                continue
            print(f"Processing {symbol}...")
            try:
                new_data = fetch_real_time_data(symbol)
                print(f"New data for {symbol}: {new_data}")
                
                df = pd.concat([dataframes[symbol], new_data]).tail(100)
                df = process_data(df, symbol)
                dataframes[symbol] = df
                latest = df.iloc[-1]
                
                balance_usd, balance_asset = executor.get_balance(symbol)
                current_price = latest['close']
                
                env = TradingEnv(df, symbol, executor, hybrid_models[symbol], lstm_models[symbol], ppo_models[symbol], grid_traders[symbol])
                obs = env.reset()
                action = 1 if obs[0] > 0.5 else 2
                print(f"Action chosen for {symbol}: {action} (1=buy, 2=sell)")
                logger.info(f"Action for {symbol}: {action}, Balance USD: {balance_usd}, Balance {symbol.split('/')[0]}: {balance_asset}")
                
                min_trade_size = executor.min_trade_sizes.get(symbol, 10.0)
                if action == 1 and balance_usd > 0:  # Buy only if USD available
                    amount = min(balance_usd * weights[symbol] / current_price, balance_usd / current_price)
                    if amount * current_price >= min_trade_size:
                        order, retry = executor.execute(action, symbol, amount)
                        if order:
                            logger.info(f"Executed order for {symbol}: {order}")
                            env.step(action-1)
                        elif retry:
                            retry_pairs.append(symbol)
                            print(f"Added {symbol} to retry list due to execution issue")
                    else:
                        print(f"Trade skipped for {symbol}: Amount {amount * current_price} below min size {min_trade_size}")
                elif action == 2 and balance_asset > 0:  # Sell only if crypto available
                    amount = min(balance_asset * weights[symbol], balance_asset)
                    if amount >= min_trade_size / current_price:
                        order, retry = executor.execute(action, symbol, amount)
                        if order:
                            logger.info(f"Executed order for {symbol}: {order}")
                            env.step(action-1)
                        elif retry:
                            retry_pairs.append(symbol)
                            print(f"Added {symbol} to retry list due to execution issue")
                    else:
                        print(f"Trade skipped for {symbol}: Amount {amount} below min size {min_trade_size / current_price}")
                else:
                    print(f"Trade skipped for {symbol}: No {['USD', symbol.split('/')[0]][action-1]} to trade")
                processed_pairs.add(symbol)
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                logger.error(f"Error processing {symbol}: {e}")

        for symbol in retry_pairs:
            print(f"Retrying {symbol} with full balance...")
            try:
                balance_usd, balance_asset = executor.get_balance(symbol)
                current_price = executor.fetch_current_price(symbol)
                env = TradingEnv(dataframes[symbol], symbol, executor, hybrid_models[symbol], lstm_models[symbol], ppo_models[symbol], grid_traders[symbol])
                obs = env.reset()
                action = 1 if obs[0] > 0.5 else 2
                
                min_trade_size = executor.min_trade_sizes.get(symbol, 10.0)
                if action == 1:
                    amount = min(balance_usd * weights[symbol] / current_price, balance_usd / current_price)
                else:
                    amount = min(balance_asset * weights[symbol], balance_asset)
                amount = max(min_trade_size, amount)
                if risk_manager.is_safe(action, symbol, balance_usd, balance_asset, current_price) and amount >= min_trade_size:
                    order, retry = executor.execute(action, symbol, amount)
                    if order:
                        logger.info(f"Executed retry order for {symbol}: {order}")
                        env.step(action-1)
                    else:
                        print(f"Retry failed for {symbol}: Insufficient funds or execution error")
                else:
                    print(f"Retry skipped for {symbol}: Amount {amount} not viable or risk not safe")
            except Exception as e:
                print(f"Error retrying {symbol}: {e}")
                logger.error(f"Error retrying {symbol}: {e}")
        
        iteration += 1
        if iteration % 10 == 0:
            executor.trading_pairs = executor.update_trading_pairs()
        print("Waiting 60 seconds...")
        time.sleep(60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the trading bot with or without retraining")
    parser.add_argument('--trade-only', action='store_true', help="Load existing models and trade without retraining")
    args = parser.parse_args()
    main(trade_only=args.trade_only)
