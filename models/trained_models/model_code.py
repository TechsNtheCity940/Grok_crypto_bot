import gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class TradingEnv(gym.Env):
    """Custom RL environment for trading."""
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.balance = 10000  # Starting balance in USD
        self.position = 0     # Crypto held
        self.action_space = gym.spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = 10000
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
    model.save('models/trained_models/ppo_trading_model')
    return model

if __name__ == "__main__":
    df = pd.read_csv(f'data/historical/{TRADING_PAIR.replace("/", "_")}_1h.csv')
    df = process_data(df)
    train_model(df)