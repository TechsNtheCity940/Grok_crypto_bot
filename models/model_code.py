import sys
import os
# Ensure the project directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

import gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from config import TRADING_PAIRS  # Import TRADING_PAIR from config.py

class TradingEnv(gym.Env):
    """Custom RL environment for trading."""
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.balance = 20  # Starting balance in USD
        self.position = 1     # Crypto held
        self.action_space = gym.spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = 20
        self.position = 0
        return self._get_observation()

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
        obs = self._get_observation()
        return obs, reward, done, {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        return np.array([row['momentum'], self.balance, self.position], dtype=np.float32)

def train_model(df):
    """Train the RL agent."""
    env = DummyVecEnv([lambda: TradingEnv(df)])
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    # Save to the same directory as model_code.py
    model.save('F:/newrepos/Grok_crypto_bot/models/trained_models/ppo_trading_model')  # Adjusted to relative path
    return model

def process_data(df):
    """Clean and engineer features."""
    df = df.dropna()
    df['ma_short'] = df['close'].rolling(window=10).mean()  # 10-period MA
    df['ma_long'] = df['close'].rolling(window=50).mean()   # 50-period MA
    df['momentum'] = df['ma_short'] - df['ma_long']
    return df.dropna()

if __name__ == "__main__":
    # Load historical data
    data_path = 'f:/newrepos/Grok_crypto_bot/data/historical/BTC_USD_1h.csv'  # Adjusted for directory structure
    try:
        df = pd.read_csv(data_path)
        processed_df = process_data(df)
        print("Data loaded and processed successfully:")
        print(processed_df.tail())
        # Train the model
        train_model(processed_df)
        print("Model training completed.")
    except FileNotFoundError:
        print(f"Error: Historical data file '{data_path}' not found. Please run data_utils.py to fetch data first.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")