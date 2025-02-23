import time
import pandas as pd
import numpy as np
import os
from utils.data_utils import fetch_real_time_data, process_data, fetch_historical_data
from utils.log_setup import logger
from execution.trade_executor import TradeExecutor
from risk_management.risk_manager import RiskManager
from config import TRADING_PAIRS, ACTIVE_EXCHANGE
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import Env, spaces
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from sentiment_analyzer import SentimentAnalyzer

class TradingEnv(Env):
    def __init__(self, df, symbol, executor):
        super().__init__()
        self.df = df
        self.symbol = symbol
        self.executor = executor
        self.current_step = 0
        self.balance_usd, self.balance_asset = self.executor.get_balance(self.symbol)
        self.initial_value = self.balance_usd + (self.balance_asset * self.df['close'].iloc[0])
        self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.balance_usd, self.balance_asset = self.executor.get_balance(self.symbol)
        return self._get_observation(), {}

    def step(self, action):
        price = self.df.iloc[self.current_step]['close']
        reward = 0
        
        if action == 1 and self.balance_usd > 0:
            amount = self.balance_usd / price * 0.5
            order = self.executor.execute(1, self.symbol, amount)
            if order:
                self.balance_usd, self.balance_asset = self.executor.get_balance(self.symbol)
        elif action == 2 and self.balance_asset > 0:
            amount = self.balance_asset * 0.5
            order = self.executor.execute(2, self.symbol, amount)
            if order:
                self.balance_usd, self.balance_asset = self.executor.get_balance(self.symbol)
                current_value = self.balance_usd + (self.balance_asset * price)
                reward = current_value - self.initial_value  # Reward based on portfolio value change
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        if done:
            self.current_step = 0
        obs = self._get_observation()
        
        # Stop-loss and take-profit logic
        current_value = self.balance_usd + (self.balance_asset * price)
        if current_value < self.initial_value * 0.9:  # 10% stop-loss
            reward -= 100  # Penalize significant loss
            done = True
        elif current_value > self.initial_value * 1.1:  # 10% take-profit
            reward += 100  # Reward significant gain
            done = True
        
        return obs, reward, done, False, {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        return np.array([
            row.get('momentum', 0.0),
            row.get('rsi', 0.0),
            row.get('macd', 0.0),
            row.get('atr', 0.0),
            row.get('sentiment', 0.0),
            self.balance_usd,
            self.balance_asset
        ], dtype=np.float32)

def main():
    print(f"Starting AI Crypto Trading Bot on {ACTIVE_EXCHANGE}")
    logger.info(f"Starting AI Crypto Trading Bot on {ACTIVE_EXCHANGE}")
    
    os.makedirs('models/trained_models', exist_ok=True)
    
    executor = TradeExecutor()
    risk_manager = RiskManager(max_loss=0.5)
    
    dataframes = {}
    models = {}
    for symbol in executor.trading_pairs:
        df = fetch_historical_data(symbol, limit=50)
        df = process_data(df, symbol)
        print(f"Initial historical data fetched and processed for {symbol}: {len(df)} rows")
        dataframes[symbol] = df
        model_path = f'models/trained_models/ppo_trading_model_{symbol.replace("/", "_")}'
        try:
            models[symbol] = PPO.load(model_path)
        except FileNotFoundError:
            env = DummyVecEnv([lambda: TradingEnv(df, symbol, executor)])
            models[symbol] = PPO('MlpPolicy', env, verbose=0)
            models[symbol].learn(total_timesteps=10000)
            models[symbol].save(model_path)

    iteration = 0
    while True:
        total_usd, total_assets = 0, {}
        for symbol in executor.trading_pairs:
            balance_usd, balance_asset = executor.get_balance(symbol)
            current_price = executor.fetch_current_price(symbol)
            total_usd += balance_usd
            total_assets[symbol.split('/')[0]] = balance_asset * current_price
        
        total_value = total_usd + sum(total_assets.values())
        print(f"Total account value: ${total_value:.2f}")

        for symbol in executor.trading_pairs:
            print(f"Processing {symbol}...")
            try:
                new_data = fetch_real_time_data(symbol)
                print(f"New data for {symbol}: {new_data}")
                
                df = pd.concat([dataframes[symbol], new_data]).tail(50)
                df = process_data(df, symbol)
                dataframes[symbol] = df
                latest = df.iloc[-1]
                
                balance_usd, balance_asset = executor.get_balance(symbol)
                current_price = latest['close']
                obs = np.array([
                    latest['momentum'],
                    latest['rsi'],
                    latest['macd'],
                    latest['atr'],
                    latest['sentiment'],
                    balance_usd,
                    balance_asset
                ])
                asset_name = symbol.split('/')[0]
                print(f"Observation for {symbol}: momentum={latest['momentum']}, rsi={latest['rsi']}, macd={latest['macd']}, atr={latest['atr']}, sentiment={latest['sentiment']}, balance_usd={balance_usd}, balance_{asset_name}={balance_asset}")
                
                action = models[symbol].predict(obs)[0]
                print(f"Action chosen for {symbol}: {action} (0=hold, 1=buy, 2=sell)")
                logger.info(f"Action for {symbol}: {action}, Balance USD: {balance_usd}, Balance {asset_name}: {balance_asset}")
                
                amount = (total_value * 0.2) / current_price if action != 0 else 0
                amount = min(amount, balance_asset if action == 2 else balance_usd / current_price)
                
                if risk_manager.is_safe(action, symbol, balance_usd, balance_asset, current_price) and amount > 0:
                    order = executor.execute(action, symbol, amount)
                    if order:
                        logger.info(f"Executed order for {symbol}: {order}")
                        env = DummyVecEnv([lambda: TradingEnv(df, symbol, executor)])
                        models[symbol].set_env(env)
                        models[symbol].learn(total_timesteps=1000, reset_num_timesteps=False)
                        models[symbol].save(model_path)
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        iteration += 1
        if iteration % 10 == 0:
            executor.trading_pairs = executor.update_trading_pairs()
        print("Waiting 60 seconds...")
        time.sleep(60)

if __name__ == "__main__":
    main() 