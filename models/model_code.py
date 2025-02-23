import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from config import TRADING_PAIRS

class TradingEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.current_step = 0
        self.balance = 10000  # Starting balance in USD
        self.position = 0     # Crypto held
        self.action_space = gym.spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.balance = 10000
        self.position = 0
        return self._get_observation(), {}

    def step(self, action):
        price = self.df.iloc[self.current_step]['close']
        reward = 0

        if action == 1 and self.balance > 0:  # Buy
            self.position += self.balance / price
            self.balance = 0
        elif action == 2 and self.position > 0:  # Sell
            self.balance += self.position * price
            reward = self.balance - 10000  # Profit/loss
            self.position = 0

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        if done:
            self.current_step = 0  # Reset to prevent out-of-bounds
        obs = self._get_observation()
        return obs, reward, done, False, {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        return np.array([row.get('momentum', 0.0), self.balance, self.position], dtype=np.float32)

def process_data(df):
    df = df.dropna()
    df['ma_short'] = df['close'].rolling(window=5).mean()  # Faster reaction
    df['ma_long'] = df['close'].rolling(window=20).mean()
    df['momentum'] = df['ma_short'] - df['ma_long']
    return df.dropna()

def train_model(df, symbol):
    df = process_data(df)
    env = DummyVecEnv([lambda: TradingEnv(df)])
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save(f'models/trained_models/ppo_trading_model_{symbol.replace("/", "_")}')
    return model

if __name__ == "__main__":                                                                                                      
    import os
    os.makedirs('models/trained_models', exist_ok=True)
    for symbol in TRADING_PAIRS:
        try:
            data_path = f'data/historical/{symbol.replace("/", "_")}_1h.csv'
            df = pd.read_csv(data_path)
            print("Data loaded and processed successfully:")
            print(df.tail())
            train_model(df, symbol)
            print(f"Model training completed for {symbol}")
        except FileNotFoundError:
            print(f"Error: Historical data file '{data_path}' not found. Please run data_utils.py first.")
        except Exception as e:
            print(f"An error occurred for {symbol}: {str(e)}")