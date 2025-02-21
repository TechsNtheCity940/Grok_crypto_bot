import time
import pandas as pd
import numpy as np
from utils.data_utils import fetch_real_time_data, process_data, fetch_historical_data
from utils.log_setup import logger
from strategies.momentum_strategy import MomentumStrategy
from execution.trade_executor import TradeExecutor
from risk_management.risk_manager import RiskManager
from config import TRADING_PAIRS, ACTIVE_EXCHANGE
from stable_baselines3 import PPO

def main():
    print(f"Starting AI Crypto Trading Bot on {ACTIVE_EXCHANGE}")
    logger.info(f"Starting AI Crypto Trading Bot on {ACTIVE_EXCHANGE}")
    
    # Initialize components
    executor = TradeExecutor()
    risk_manager = RiskManager(max_loss=0.3)  # More aggressive
    strategy = MomentumStrategy()
    
    # Initialize data and models for all trading pairs
    dataframes = {}
    models = {}
    for symbol in executor.trading_pairs:
        df = fetch_historical_data(symbol, limit=50)
        print(f"Initial historical data fetched for {symbol}: {len(df)} rows")
        dataframes[symbol] = df
        try:
            models[symbol] = PPO.load(f'models/trained_models/ppo_trading_model_{symbol.replace("/", "_")}')
        except FileNotFoundError:
            env = DummyVecEnv([lambda: TradingEnv(df)])
            models[symbol] = PPO('MlpPolicy', env, verbose=0)
            models[symbol].learn(total_timesteps=10000)
            models[symbol].save(f'models/trained_models/ppo_trading_model_{symbol.replace("/", "_")}')

    # Main trading loop
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
                dataframes[symbol] = df
                processed_df = process_data(df)
                latest = processed_df.iloc[-1] if len(processed_df) > 0 else df.iloc[-1]
                momentum = latest.get('momentum', 0.0)
                
                balance_usd, balance_asset = executor.get_balance(symbol)
                current_price = latest['close']
                obs = np.array([momentum, balance_usd, balance_asset])
                asset_name = symbol.split('/')[0]
                print(f"Observation for {symbol}: momentum={momentum}, balance_usd={balance_usd}, balance_{asset_name}={balance_asset}")
                
                action = models[symbol].predict(obs)[0]
                print(f"Action chosen for {symbol}: {action} (0=hold, 1=buy, 2=sell)")
                logger.info(f"Action for {symbol}: {action}, Balance USD: {balance_usd}, Balance {asset_name}: {balance_asset}")
                
                # Dynamic trade size: 20% of total value per trade
                amount = (total_value * 0.2) / current_price if action != 0 else 0
                amount = min(amount, balance_asset if action == 2 else balance_usd / current_price)
                
                if risk_manager.is_safe(action, symbol, balance_usd, balance_asset, current_price) and amount > 0:
                    order = executor.execute(action, symbol, amount)
                    if order:
                        logger.info(f"Executed order for {symbol}: {order}")
                        # Online learning after successful trade
                        env = DummyVecEnv([lambda: TradingEnv(df)])
                        models[symbol].set_env(env)
                        models[symbol].learn(total_timesteps=1000, reset_num_timesteps=False)
                        models[symbol].save(f'models/trained_models/ppo_trading_model_{symbol.replace("/", "_")}')
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        iteration += 1
        if iteration % 10 == 0:  # Update pairs every 10 iterations
            executor.update_trading_pairs()
        print("Waiting 60 seconds...")
        time.sleep(60)

class TradingEnv:
    def __init__(self, df):
        self.df = df
        self.current_step = 0
        self.balance_usd = 7.4729  # Initial from your account
        self.positions = {'XBT': 0.000100009, 'STX': 0.00097, 'AVAX': 6.44e-06}
        self.action_space = [0, 1, 2]  # hold, buy, sell
        self.observation_space = np.array([0.0, 0.0, 0.0])  # momentum, usd, asset

    def reset(self):
        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        price = self.df.iloc[self.current_step]['close']
        reward = 0
        asset = 'XBT'  # Simplified; adjust per pair in real use
        
        if action == 1 and self.balance_usd > 0:
            self.positions[asset] += self.balance_usd / price
            self.balance_usd = 0
        elif action == 2 and self.positions[asset] > 0:
            self.balance_usd += self.positions[asset] * price
            reward = self.balance_usd - 7.4729  # Profit/loss
            self.positions[asset] = 0

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        obs = self._get_observation()
        return obs, reward, done, {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        return np.array([row.get('momentum', 0.0), self.balance_usd, self.positions['XBT']])

from stable_baselines3.common.vec_env import DummyVecEnv
if __name__ == "__main__":
    main()