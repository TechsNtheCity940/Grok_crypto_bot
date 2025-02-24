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
from hybrid_model import HybridCryptoModel  # Import your hybrid model
from grid_trading import GridTrader
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from sentiment_analyzer import SentimentAnalyzer

def train_hybrid_model(symbol, df):
    model = HybridCryptoModel(sequence_length=50, n_features=9)
    X = []
    y_price = []
    for i in range(len(df) - 50):
        X.append(df[['momentum', 'rsi', 'macd', 'atr', 'sentiment', 'arbitrage_spread', 'whale_activity', 'defi_apr']].iloc[i:i+50].values)
        price_change = (df['close'].iloc[i+50] - df['close'].iloc[i+49]) / df['close'].iloc[i+49]
        y_price.append(1 if price_change > 0.005 else 0)  # 0.5% threshold for binary classification
    X = np.array(X)
    y_price = np.array(y_price)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y_price[:split_idx], y_price[split_idx:]
    history = model.train(
        X_train, y_train, np.zeros_like(y_train),  # Volatility unused for simplicity
        (X_val, y_val, np.zeros_like(y_val)),
        batch_size=32, epochs=50, model_path=f'models/trained_models/hybrid_{symbol.replace("/", "_")}.h5'
    )
    return model, history

class TradingEnv:
    def __init__(self, df, symbol, executor, hybrid_model, grid_trader):
        self.df = df
        self.symbol = symbol
        self.executor = executor
        self.hybrid_model = hybrid_model
        self.grid_trader = grid_trader
        self.current_step = 0
        self.balance_usd, self.balance_asset = self.executor.get_balance(self.symbol)
        self.initial_value = self.balance_usd + (self.balance_asset * self.df['close'].iloc[0])
        self.grid_trader.setup_grids(self.df['close'].iloc[0], price_range=5.0)

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
        if current_value < self.initial_value * 0.95:
            reward -= 50
            done = True
        elif current_value > self.initial_value * 1.20:
            reward += 50
            done = True
        
        return obs, reward, done, False

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        X = self.df[['momentum', 'rsi', 'macd', 'atr', 'sentiment', 'arbitrage_spread', 'whale_activity', 'defi_apr']].iloc[max(0, self.current_step-49):self.current_step+1].values
        if len(X) < 50:
            X = np.pad(X, ((50 - len(X), 0), (0, 0)), mode='edge')
        price_pred, _ = self.hybrid_model.predict(np.expand_dims(X, axis=0))
        return np.array([price_pred[0][0], self.balance_usd, self.balance_asset], dtype=np.float32)

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
    grid_traders = {}
    for symbol in executor.trading_pairs:
        df = fetch_historical_data(symbol)
        df_augmented = augment_data(df)
        df_processed = process_data(df_augmented, symbol)
        dataframes[symbol] = df_processed
        print(f"Initial historical data fetched and processed for {symbol}: {len(df_processed)} rows")
        
        hybrid_model, history = train_hybrid_model(symbol, df_processed)
        hybrid_models[symbol] = hybrid_model
        
        grid_config = {
            'grid_trading': {
                'num_grids': 10,
                'grid_spread': 0.5,
                'max_position': 0.1,
                'min_profit': 0.2
            }
        }
        grid_traders[symbol] = GridTrader(grid_config)
        
        model_path = f'{model_dir}/hybrid_{symbol.replace("/", "_")}.h5'
        hybrid_model.model.save(model_path)

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

        retry_pairs = []
        processed_pairs = set()

        for symbol in executor.trading_pairs:
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
                
                env = TradingEnv(df, symbol, executor, hybrid_models[symbol], grid_traders[symbol])
                obs = env.reset()
                action = 1 if hybrid_models[symbol].predict(np.expand_dims(df.iloc[-50:].values, axis=0))[0][0] > 0.5 else 2
                print(f"Action chosen for {symbol}: {action} (1=buy, 2=sell)")
                logger.info(f"Action for {symbol}: {action}, Balance USD: {balance_usd}, Balance DOGE: {balance_asset}")
                
                amount = (total_value * 0.2) / current_price if action != 0 else 0
                amount = min(amount, balance_asset if action == 2 else balance_usd / current_price)
                
                if risk_manager.is_safe(action, symbol, balance_usd, balance_asset, current_price) and amount > 0:
                    order, retry = executor.execute(action, symbol, amount)
                    if order:
                        logger.info(f"Executed order for {symbol}: {order}")
                        env.step(action)  # Update environment state
                    elif retry:
                        retry_pairs.append(symbol)
                        print(f"Added {symbol} to retry list due to minimum trade size")
                processed_pairs.add(symbol)
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                logger.error(f"Error processing {symbol}: {e}")

        for symbol in retry_pairs:
            print(f"Retrying {symbol} with adjusted pair selection...")
            for alt_symbol in [s for s in executor.trading_pairs if s not in processed_pairs]:
                print(f"Switching to {alt_symbol}")
                try:
                    new_data = fetch_real_time_data(alt_symbol)
                    df = pd.concat([dataframes.get(alt_symbol, fetch_historical_data(alt_symbol)), new_data]).tail(100)
                    df = process_data(df, alt_symbol)
                    dataframes[alt_symbol] = df
                    latest = df.iloc[-1]
                    
                    balance_usd, balance_asset = executor.get_balance(alt_symbol)
                    current_price = latest['close']
                    env = TradingEnv(df, alt_symbol, executor, hybrid_models[alt_symbol], grid_traders[alt_symbol])
                    obs = env.reset()
                    action = 1 if hybrid_models[alt_symbol].predict(np.expand_dims(df.iloc[-50:].values, axis=0))[0][0] > 0.5 else 2
                    print(f"Action chosen for {alt_symbol}: {action} (1=buy, 2=sell)")
                    logger.info(f"Action for {alt_symbol}: {action}, Balance USD: {balance_usd}, Balance DOGE: {balance_asset}")
                    
                    amount = (total_value * 0.2) / current_price if action != 0 else 0
                    amount = min(amount, balance_asset if action == 2 else balance_usd / current_price)
                    
                    if risk_manager.is_safe(action, alt_symbol, balance_usd, balance_asset, current_price) and amount > 0:
                        order, retry = executor.execute(action, alt_symbol, amount)
                        if order:
                            logger.info(f"Executed order for {alt_symbol}: {order}")
                            env.step(action)
                            processed_pairs.add(alt_symbol)
                            break
                except Exception as e:
                    print(f"Error processing {alt_symbol}: {e}")
                    logger.error(f"Error processing {alt_symbol}: {e}")
        
        iteration += 1
        if iteration % 10 == 0:
            executor.trading_pairs = executor.update_trading_pairs()
        print("Waiting 60 seconds...")
        time.sleep(60)

if __name__ == "__main__":
    main()