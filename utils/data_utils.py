import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

import ccxt
import pandas as pd
import numpy as np
from websocket import create_connection
import json
from config import KRAKEN_API_KEY, KRAKEN_API_SECRET, TRADING_PAIRS, ACTIVE_EXCHANGE

kraken = ccxt.kraken({
    'apiKey': KRAKEN_API_KEY,
    'secret': KRAKEN_API_SECRET,
    'enableRateLimit': True,
})

exchange = kraken if ACTIVE_EXCHANGE == 'kraken' else ccxt.coinbasepro()

def fetch_historical_data(symbol, timeframe='1h', limit=50):
    retries = 3
    for _ in range(retries):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.to_csv(f'data/historical/{symbol.replace("/", "_")}_{timeframe}.csv', index=False)
            return df
        except Exception as e:
            print(f"Failed to fetch historical data for {symbol}: {e}")
            time.sleep(5)
    return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

def process_data(df):
    df = df.copy()
    df['ma_short'] = df['close'].rolling(window=min(5, len(df))).mean()  # Reduced for faster reaction
    df['ma_long'] = df['close'].rolling(window=min(20, len(df))).mean()
    df['momentum'] = df['ma_short'] - df['ma_long']
    df['momentum'] = df['momentum'].fillna(0)
    return df

def fetch_real_time_data(symbol):
    timeout = 60
    start_time = time.time()
    ws = create_connection('wss://ws.kraken.com')
    ws.send(json.dumps({
        "event": "subscribe",
        "pair": [symbol],
        "subscription": {"name": "trade"}
    }))
    while time.time() - start_time < timeout:
        try:
            message = ws.recv()
            print(f"Kraken WebSocket message: {message}")
            data = json.loads(message)
            if isinstance(data, list) and len(data) > 2 and data[2] == "trade":
                price = float(data[1][0][0])
                timestamp = pd.to_datetime(float(data[1][0][2]), unit='s')
                ws.close()
                print(f"Kraken trade detected for {symbol}: price={price}, timestamp={timestamp}")
                return pd.DataFrame([[timestamp, price]], columns=['timestamp', 'close'])
        except Exception as e:
            print(f"WebSocket error for {symbol}: {e}")
            break
    ws.close()
    raise TimeoutError(f"No trade data received from Kraken within {timeout} seconds")