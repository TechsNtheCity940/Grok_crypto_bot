import time
import pandas as pd
import numpy as np
from utils.data_utils import fetch_real_time_data, process_data
from utils.log_setup import logger
from strategies.momentum_strategy import MomentumStrategy
from execution.trade_executor import TradeExecutor
from risk_management.risk_manager import RiskManager
from config import TRADING_PAIR, ACTIVE_EXCHANGE

def main():
    print(f"Starting AI Crypto Trading Bot on {ACTIVE_EXCHANGE}")
    logger.info(f"Starting AI Crypto Trading Bot on {ACTIVE_EXCHANGE}")
    executor = TradeExecutor()
    risk_manager = RiskManager()
    strategy = MomentumStrategy()
    df = pd.DataFrame(columns=['timestamp', 'close', 'ma_short', 'ma_long', 'momentum'])

    # Initial balance check and sell BTC if no USD
    balance_usd, balance_btc = executor.get_balance()
    if balance_usd == 0 and balance_btc > 0:
        print(f"No USD, selling 0.0001 BTC to start (BTC balance: {balance_btc})")
        executor.execute(2)  # Sell 0.0001 BTC to get USD

    while True:
        print("Fetching real-time data...")
        new_data = fetch_real_time_data()
        print(f"New data: {new_data}")
        df = pd.concat([df, new_data]).tail(50)
        print(f"Current DataFrame size: {len(df)} rows")

        processed_df = process_data(df)
        if len(processed_df) == 0:
            print("Not enough data for momentum yet, using raw close price")
            latest = df.iloc[-1]  # Use raw data until enough for processing
            momentum = 0.0  # Default momentum until calculated
        else:
            latest = processed_df.iloc[-1]
            momentum = latest['momentum']

        balance, position = executor.get_balance()
        obs = np.array([momentum, balance, position])
        print(f"Observation: momentum={momentum}, balance={balance}, position={position}")

        action = strategy.get_action(obs)
        print(f"Action chosen: {action} (0=hold, 1=buy, 2=sell)")
        logger.info(f"Action: {action}, Balance: {balance}, Position: {position}")

        position_value = position * latest['close']
        if risk_manager.is_safe(action, balance, position_value):
            order = executor.execute(action)
            if order:
                logger.info(f"Executed order: {order}")
        else:
            print("Trade skipped due to risk management")

        print("Waiting 60 seconds...")
        time.sleep(60)

if __name__ == "__main__":
    main()