import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

import ccxt
import pandas as pd
import numpy as np
from websocket import create_connection
import json
from config import KRAKEN_API_KEY, KRAKEN_API_SECRET, COINBASE_API_KEY, COINBASE_API_SECRET, TRADING_PAIRS, ACTIVE_EXCHANGE

kraken = ccxt.kraken({
    'apiKey': KRAKEN_API_KEY,
    'secret': KRAKEN_API_SECRET,
    'enableRateLimit': True,
})

coinbase = ccxt.coinbase({
    'apiKey': COINBASE_API_KEY,
    'secret': COINBASE_API_SECRET,
    'enableRateLimit': True,
})

exchange = kraken if ACTIVE_EXCHANGE == 'kraken' else coinbase

def fetch_historical_data(symbol, timeframe='1h', limit=1000):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.to_csv(f'data/historical/{symbol.replace("/", "_")}_{timeframe}.csv', index=False)
    return df

def process_data(df):
    df = df.copy()
    df['ma_short'] = df['close'].rolling(window=min(10, len(df))).mean()
    df['ma_long'] = df['close'].rolling(window=min(50, len(df))).mean()
    df['momentum'] = df['ma_short'] - df['ma_long']
    df['momentum'] = df['momentum'].fillna(0)
    return df

def fetch_real_time_data(symbol):
    timeout = 60  # Increased timeout
    start_time = time.time()
    if ACTIVE_EXCHANGE == 'kraken':
        ws = create_connection('wss://ws.kraken.com')
        ws.send(json.dumps({
            "event": "subscribe",
            "pair": [symbol],
            "subscription": {"name": "trade"}
        }))
        while time.time() - start_time < timeout:
            message = ws.recv()
            print(f"Kraken WebSocket message: {message}")
            data = json.loads(message)
            if isinstance(data, list) and len(data) > 2 and data[2] == "trade":
                price = float(data[1][0][0])
                timestamp = pd.to_datetime(float(data[1][0][2]), unit='s')
                ws.close()
                print(f"Kraken trade detected: price={price}, timestamp={timestamp}")
                return pd.DataFrame([[timestamp, price]], columns=['timestamp', 'close'])
        ws.close()
        raise TimeoutError(f"No trade data received from Kraken within {timeout} seconds")
    # Coinbase logic omitted for brevity, add if needed