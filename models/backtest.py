import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

import pandas as pd
import numpy as np
from data_utils import process_data

def backtest_model(model, symbol, df, sequence_length=50):
    df_processed = process_data(df, symbol)
    X, y = [], []
    for i in range(len(df_processed) - sequence_length):
        X.append(df_processed[['momentum', 'rsi', 'macd', 'atr', 'sentiment', 'arbitrage_spread', 'whale_activity', 'bb_upper', 'defi_apr']].iloc[i:i+sequence_length].values)
        price_change = (df_processed['close'].iloc[i+sequence_length] - df_processed['close'].iloc[i+sequence_length-1]) / df_processed['close'].iloc[i+sequence_length-1]
        y.append(1 if price_change > 0.02 else 0)
    X = np.array(X)
    y = np.array(y)
    preds = model.predict(X)[0]  # Extract price predictions (first element of tuple)
    accuracy = np.mean((preds.flatten() > 0.5) == y)
    return accuracy

if __name__ == "__main__":
    from data_utils import fetch_historical_data
    symbol = 'DOGE/USD'
    df = fetch_historical_data(symbol)
    from hybrid_model import HybridCryptoModel
    model = HybridCryptoModel()
    model.model.load_weights('models/trained_models/hybrid_DOGE_USD.h5')
    accuracy = backtest_model(model, symbol, df)
    print(f"Backtest accuracy for {symbol}: {accuracy:.2f}")