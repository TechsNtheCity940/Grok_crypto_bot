import ccxt
import pandas as pd
import numpy as np
from websocket import create_connection
import json
import os
from config import KRAKEN_API_KEY, KRAKEN_API_SECRET, COINBASE_API_KEY, COINBASE_API_SECRET, TRADING_PAIR, ACTIVE_EXCHANGE

# Initialize exchanges
kraken = ccxt.kraken({
    'apiKey': KRAKEN_API_KEY,
    'secret': KRAKEN_API_SECRET,
    'enableRateLimit': True,
})

coinbase = ccxt.coinbasepro({
    'apiKey': COINBASE_API_KEY,
    'secret': COINBASE_API_SECRET,
    'enableRateLimit': True,
})

# Select active exchange
exchange = kraken if ACTIVE_EXCHANGE == 'kraken' else coinbase

def fetch_historical_data(symbol=TRADING_PAIR, timeframe='1h', limit=1000):
    """Fetch historical OHLCV data."""
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.to_csv(f'data/historical/{symbol.replace("/", "_")}_{timeframe}.csv', index=False)
    return df

def process_data(df):
    """Clean and engineer features."""
    df = df.dropna()
    df['ma_short'] = df['close'].rolling(window=10).mean()  # 10-period MA
    df['ma_long'] = df['close'].rolling(window=50).mean()   # 50-period MA
    df['momentum'] = df['ma_short'] - df['ma_long']
    return df.dropna()

def fetch_real_time_data(symbol=TRADING_PAIR):
    """Fetch real-time price via WebSocket."""
    if ACTIVE_EXCHANGE == 'kraken':
        ws = create_connection('wss://ws.kraken.com')
        ws.send(json.dumps({
            "event": "subscribe",
            "pair": [symbol],
            "subscription": {"name": "trade"}
        }))
        while True:
            message = ws.recv()
            data = json.loads(message)
            if isinstance(data, list) and len(data) > 0:
                price = float(data[0][0])  # Kraken trade price
                timestamp = pd.to_datetime(float(data[0][2]), unit='s')  # Trade timestamp
                ws.close()
                return pd.DataFrame([[timestamp, price]], columns=['timestamp', 'close'])
    else:  # Coinbase
        ws = create_connection('wss://ws-feed.pro.coinbase.com')
        ws.send(json.dumps({
            "type": "subscribe",
            "product_ids": [symbol],
            "channels": ["matches"]
        }))
        while True:
            message = ws.recv()
            data = json.loads(message)
            if data.get('type') == 'match':
                price = float(data['price'])
                timestamp = pd.to_datetime(data['time'])
                ws.close()
                return pd.DataFrame([[timestamp, price]], columns=['timestamp', 'close'])

if __name__ == "__main__":
    # Test data fetching
    df = fetch_historical_data()
    processed_df = process_data(df)
    print(processed_df.tail())