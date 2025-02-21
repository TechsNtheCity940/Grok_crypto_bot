import time
import pandas as pd
import numpy as np
from utils.data_utils import fetch_real_time_data, process_data, fetch_historical_data
from utils.log_setup import logger
from strategies.momentum_strategy import MomentumStrategy
from execution.trade_executor import TradeExecutor
from risk_management.risk_manager import RiskManager
from config import TRADING_PAIR, ACTIVE_EXCHANGE

def main():
    print(f"Starting AI Crypto Trading Bot on {ACTIVE_EXCHANGE}")
    logger.info(f"Starting AI Crypto Trading Bot on {ACTIVE_EXCHANGE}")
    
    # Fetch initial historical data (50 rows)
    df = fetch_historical_data(limit=50)
    print(f"Initial historical data fetched: {len(df)} rows")
    print(df.tail())
    
    # Initialize components
    executor = TradeExecutor()
    risk_manager = RiskManager()
    strategy = MomentumStrategy()
    
    # Initial balance check and sell BTC if no USD
    balance_usd, balance_btc = executor.get_balance()
    if balance_usd == 0 and balance_btc > 0:
        print(f"No USD, selling 0.0001 BTC to start (BTC balance: {balance_btc})")
        executor.execute(2)  # Sell 0.0001 BTC to get USD

    # Main trading loop
    while True:
        print("Fetching real-time data...")
        new_data = fetch_real_time_data()
        print(f"New data: {new_data}")
        
        # Append new data and keep the last 50 rows
        df = pd.concat([df, new_data]).tail(50)
        print(f"Current DataFrame size: {len(df)} rows")
        
        # Process data
        processed_df = process_data(df)
        if len(processed_df) == 0:
            print("Not enough data for momentum yet, using raw close price")
            latest = df.iloc[-1]
            momentum = 0.0
        else:
            latest = processed_df.iloc[-1]
            momentum = latest['momentum']
        
        # Get current balance and position
        balance_usd, balance_btc = executor.get_balance()
        current_price = latest['close']  # Use latest close as current price
        obs = np.array([momentum, balance_usd, balance_btc])
        print(f"Observation: momentum={momentum}, balance_usd={balance_usd}, balance_btc={balance_btc}")
        
        # Get action from strategy
        action = strategy.get_action(obs)
        print(f"Action chosen: {action} (0=hold, 1=buy, 2=sell)")
        logger.info(f"Action: {action}, Balance USD: {balance_usd}, Balance BTC: {balance_btc}")
        
        # Calculate position value
        position_value = balance_btc * current_price
        
        # Execute trade if safe
        if risk_manager.is_safe(action, balance_usd, balance_btc, current_price):
            order = executor.execute(action)
            if order:
                logger.info(f"Executed order: {order}")
        else:
            print("Trade skipped due to risk management")
        
        print("Waiting 60 seconds...")
        time.sleep(60)

if __name__ == "__main__":
    main()