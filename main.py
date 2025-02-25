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

def train_hybrid_model(symbol, df):
    model = HybridCryptoModel(sequence_length=50, n_features=9)
    X = []
    y_price = []
    for i in range(len(df) - 50):
        X.append(df[['momentum', 'rsi', 'macd', 'atr', 'sentiment', 'arbitrage_spread', 'whale_activity', 'bb_upper']].iloc[i:i+50].values)
        price_change = (df['close'].iloc[i+50] - df['close'].iloc[i+49]) / df['close'].iloc[i+49]
        y_price.append(1 if price_change > 0.02 else 0)
    X = np.array(X)
    y_price = np.array(y_price)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y_price[:split_idx], y_price[split_idx:]
    history = model.train(X_train, y_train, np.zeros_like(y_train), (X_val, y_val, np.zeros_like(y_val)), model_path=f'models/trained_models/hybrid_{symbol.replace("/", "_")}.h5')
    accuracy = backtest_model(model, symbol, df)
    logger.info(f"Backtest accuracy for {symbol}: {accuracy:.2f}")
    return model, history

def train_lstm_model(symbol, df):
    model = LSTMModel(sequence_length=50, n_features=9)
    X = []
    y_price = []
    for i in range(len(df) - 50):
        X.append(df[['momentum', 'rsi', 'macd', 'atr', 'sentiment', 'arbitrage_spread', 'whale_activity', 'bb_upper']].iloc[i:i+50].values)
        price_change = (df['close'].iloc[i+50] - df['close'].iloc[i+49]) / df['close'].iloc[i+49]
        y_price.append(1 if price_change > 0.02 else 0)
    X = np.array(X)
    y_price = np.array(y_price)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y_price[:split_idx], y_price[split_idx:]
    history = model.train(X_train, y_train, X_val, y_val, model_path=f'models/trained_models/lstm_{symbol.replace("/", "_")}.h5')
    accuracy = backtest_model(model, symbol, df)
    logger.info(f"LSTM backtest accuracy for {symbol}: {accuracy:.2f}")
    return model, history

class TradingEnv:
    def __init__(self, df, symbol, executor, hybrid_model, lstm_model, ppo_model, grid_trader):
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

    def reset(self):
        self.current_step = 0
        self.balance_usd, self.balance_asset = self.executor.get_balance(self.symbol)
        return self._get_observation()

    def step(self, action):
        price = self.df.iloc[self.current_step]['close']
        reward = 0
        
        grid_orders = self.grid_trader.get_grid_orders(price, self.balance_usd + self.balance_asset * price)
        for order in grid_orders:
            order_type = 1 if order['type'] == 'buy' else 2
            order_result, retry = self.executor.execute(order_type, self.symbol, order['amount'])
            if order_result:
                self.grid_trader.update_grids(order)
                self.balance_usd, self.balance_asset = self.executor.get_balance(self.symbol)
                current_value = self.balance_usd + (self.balance_asset * price)
                reward += current_value - self.initial_value
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        if done:
            self.current_step = 0
        obs = self._get_observation()
        
        current_value = self.balance_usd + (self.balance_asset * price)
        if current_value < self.initial_value * 0.9:
            reward -= 100
            done = True
        elif current_value > self.initial_value * 2:
            reward += 200
            done = True
        
        return obs, reward, done, False

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        X = self.df[['momentum', 'rsi', 'macd', 'atr', 'sentiment', 'arbitrage_spread', 'whale_activity', 'bb_upper']].iloc[max(0, self.current_step-49):self.current_step+1].values
        if len(X) < 50:
            X = np.pad(X, ((50 - len(X), 0), (0, 0)), mode='edge')
        hybrid_pred = self.hybrid_model.predict(np.expand_dims(X, axis=0))[0][0]
        lstm_pred = self.lstm_model.predict(np.expand_dims(X, axis=0))[0]
        ppo_pred = self.ppo_model.predict(obs=np.array([self.balance_usd, self.balance_asset, hybrid_pred]), deterministic=True)[0]
        ensemble_pred = np.mean([hybrid_pred, lstm_pred, 1 if ppo_pred == 1 else 0])
        return np.array([ensemble_pred, self.balance_usd, self.balance_asset], dtype=np.float32)

def main():
    print(f"Starting AI Crypto Trading Bot on {ACTIVE_EXCHANGE}")
    logger.info(f"Starting AI Crypto Trading Bot on {ACTIVE_EXCHANGE}")
    
    model_dir = 'models/trained_models'
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)
    
    executor = TradeExecutor()
    risk_manager = RiskManager(max_loss=0.5)
    
    dataframes = {}
    hybrid_models = {}
    lstm_models = {}
    ppo_models = {}
    grid_traders = {}
    for symbol in TRADING_PAIRS:
        df = fetch_historical_data(symbol)
        df_augmented = augment_data(df)
        df_processed = process_data(df_augmented, symbol)
        dataframes[symbol] = df_processed
        print(f"Initial historical data fetched and processed for {symbol}: {len(df_processed)} rows")
        
        hybrid_model, _ = train_hybrid_model(symbol, df_processed)
        lstm_model, _ = train_lstm_model(symbol, df_processed)
        hybrid_models[symbol] = hybrid_model
        lstm_models[symbol] = lstm_model
        
        env = TradingEnv(df_processed, symbol, executor, hybrid_model, lstm_model, None, GridTrader({'grid_trading': {}}))
        ppo_models[symbol] = PPO('MlpPolicy', env, verbose=0)
        ppo_models[symbol].learn(total_timesteps=10000)
        ppo_models[symbol].save(f'{model_dir}/ppo_{symbol.replace("/", "_")}')
        
        grid_config = {
            'grid_trading': {
                'num_grids': 10,
                'grid_spread': 0.05,
                'max_position': 1.0,
                'min_profit': 0.2
            }
        }
        grid_traders[symbol] = GridTrader(grid_config)

    iteration = 0
    while True:
        total_usd, total_assets = 0, {}
        for symbol in TRADING_PAIRS:
            balance_usd, balance_asset = executor.get_balance(symbol)
            current_price = executor.fetch_current_price(symbol)
            total_usd += balance_usd
            total_assets[symbol.split('/')[0]] = balance_asset * current_price
        
        total_value = total_usd + sum(total_assets.values())
        print(f"Total account value: ${total_value:.2f}")

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
                action = 1 if obs[0] > 0.5 else 2  # Ensemble vote
                print(f"Action chosen for {symbol}: {action} (1=buy, 2=sell)")
                logger.info(f"Action for {symbol}: {action}, Balance USD: {balance_usd}, Balance {symbol.split('/')[0]}: {balance_asset}")
                
                min_trade_size = executor.min_trade_sizes.get(symbol, 10.0)
                if action == 1:
                    amount = balance_usd / current_price
                else:
                    amount = balance_asset
                amount = max(min_trade_size, amount)
                
                if risk_manager.is_safe(action, symbol, balance_usd, balance_asset, current_price) and amount >= min_trade_size:
                    order, retry = executor.execute(action, symbol, amount)
                    if order:
                        logger.info(f"Executed order for {symbol}: {order}")
                        env.step(action)
                    elif retry:
                        retry_pairs.append(symbol)
                        print(f"Added {symbol} to retry list due to execution issue")
                else:
                    print(f"Trade skipped for {symbol}: Amount {amount} not viable or risk not safe")
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
                    amount = balance_usd / current_price
                else:
                    amount = balance_asset
                amount = max(min_trade_size, amount)
                if risk_manager.is_safe(action, symbol, balance_usd, balance_asset, current_price) and amount >= min_trade_size:
                    order, retry = executor.execute(action, symbol, amount)
                    if order:
                        logger.info(f"Executed retry order for {symbol}: {order}")
                        env.step(action)
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
    main()