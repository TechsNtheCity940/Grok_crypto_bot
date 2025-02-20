import time
import pandas as pd
import numpy as np
from utils.data_utils import fetch_real_time_data, process_data
from utils.logging import logger
from strategies.momentum_strategy import MomentumStrategy
from execution.trade_executor import TradeExecutor
from risk_management.risk_manager import RiskManager
from config import TRADING_PAIR, ACTIVE_EXCHANGE

def main():
    logger.info(f"Starting AI Crypto Trading Bot on {ACTIVE_EXCHANGE}")
    executor = TradeExecutor()
    risk_manager = RiskManager()
    strategy = MomentumStrategy()
    df = pd.DataFrame(columns=['timestamp', 'close', 'ma_short', 'ma_long', 'momentum'])

    while True:
        # Fetch and process real-time data
        new_data = fetch_real_time_data()
        df = pd.concat([df, new_data]).tail(50)  # Keep last 50 rows
        processed_df = process_data(df)
        latest = processed_df.iloc[-1]

        # Get observation
        balance, position = executor.get_balance()
        obs = np.array([latest['momentum'], balance, position])

        # Get action
        action = strategy.get_action(obs)
        logger.info(f"Action: {action}, Balance: {balance}, Position: {position}")

        # Execute trade if safe
        position_value = position * latest['close']
        if risk_manager.is_safe(action, balance, position_value):
            order = executor.execute(action)
            if order:
                logger.info(f"Executed order: {order}")
        
        time.sleep(60)  # Wait 1 minute

if __name__ == "__main__":
    main()