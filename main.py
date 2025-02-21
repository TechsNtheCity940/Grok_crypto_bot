import time
import pandas as pd
import numpy as np
from utils.data_utils import fetch_real_time_data, process_data, fetch_historical_data
from utils.log_setup import logger
from strategies.momentum_strategy import MomentumStrategy
from execution.trade_executor import TradeExecutor
from risk_management.risk_manager import RiskManager
from config import TRADING_PAIRS, ACTIVE_EXCHANGE

def main():
    print(f"Starting AI Crypto Trading Bot on {ACTIVE_EXCHANGE}")
    logger.info(f"Starting AI Crypto Trading Bot on {ACTIVE_EXCHANGE}")
    
    # Initialize components
    executor = TradeExecutor()
    risk_manager = RiskManager()
    strategy = MomentumStrategy()
    
    # Initialize data for all trading pairs
    dataframes = {}
    for symbol in TRADING_PAIRS:
        df = fetch_historical_data(symbol, limit=50)
        print(f"Initial historical data fetched for {symbol}: {len(df)} rows")
        dataframes[symbol] = df
    
    # Initial balance check and sell if no USD (for XBT/USD only)
    balance_usd, balance_btc = executor.get_balance('XBT/USD')
    if balance_usd == 0 and balance_btc > 0:
        print(f"No USD, selling 0.0001 XBT to start (XBT balance: {balance_btc})")
        executor.execute(2, 'XBT/USD')

    # Main trading loop
    while True:
        for symbol in TRADING_PAIRS:
            print(f"Fetching real-time data for {symbol}...")
            new_data = fetch_real_time_data(symbol)
            print(f"New data for {symbol}: {new_data}")
            
            # Append and process data
            df = pd.concat([dataframes[symbol], new_data]).tail(50)
            dataframes[symbol] = df
            print(f"Current DataFrame size for {symbol}: {len(df)} rows")
            
            processed_df = process_data(df)
            if len(processed_df) == 0:
                print(f"Not enough data for momentum yet for {symbol}, using raw close price")
                latest = df.iloc[-1]
                momentum = 0.0
            else:
                latest = processed_df.iloc[-1]
                momentum = latest['momentum']
            
            # Get balance and execute trades
            balance_usd, balance_asset = executor.get_balance(symbol)
            current_price = latest['close']
            obs = np.array([momentum, balance_usd, balance_asset])
            print(f"Observation for {symbol}: momentum={momentum}, balance_usd={balance_usd}, balance_{symbol.split('/')[0]}={balance_asset}")
            
            action = strategy.get_action(obs)
            print(f"Action chosen for {symbol}: {action} (0=hold, 1=buy, 2=sell)")
            logger.info(f"Action for {symbol}: {action}, Balance USD: {balance_usd}, Balance {symbol.split('/')[0]}: {balance_asset}")
            
            if risk_manager.is_safe(action, balance_usd, balance_asset, current_price):
                order = executor.execute(action, symbol)
                if order:
                    logger.info(f"Executed order for {symbol}: {order}")
            else:
                print(f"Trade skipped for {symbol} due to risk management")
        
        print("Waiting 60 seconds...")
        time.sleep(60)

if __name__ == "__main__":
    main()