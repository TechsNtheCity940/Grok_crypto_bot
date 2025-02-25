import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from utils.data_utils import process_data

def backtest_model(model, symbol, df, sequence_length=50):
    df_processed = process_data(df, symbol)
    X, y = [], []
    for i in range(len(df_processed) - sequence_length):
        X.append(df_processed[['momentum', 'rsi', 'macd', 'atr', 'sentiment', 'arbitrage_spread', 'whale_activity', 'bb_upper']].iloc[i:i+sequence_length].values)
        price_change = (df_processed['close'].iloc[i+sequence_length] - df_processed['close'].iloc[i+sequence_length-1]) / df_processed['close'].iloc[i+sequence_length-1]
        y.append(1 if price_change > 0.02 else 0)
    X = np.array(X)
    y = np.array(y)
    preds = model.predict(X)
    accuracy = np.mean((preds.flatten() > 0.5) == y)
    return accuracy