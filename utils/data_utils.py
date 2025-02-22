import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

import ccxt
import pandas as pd
import numpy as np
from websocket import create_connection
import json
import talib
from sentiment_analyzer import SentimentAnalyzer
from config import KRAKEN_API_KEY, KRAKEN_API_SECRET, ACTIVE_EXCHANGE

kraken = ccxt.kraken({
    'apiKey': KRAKEN_API_KEY,
    'secret': KRAKEN_API_SECRET,
    'enableRateLimit': True,
})

exchange = kraken if ACTIVE_EXCHANGE == 'kraken' else ccxt.coinbasepro()
sentiment_analyzer = SentimentAnalyzer()

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

def process_data(df, symbol):
    df = df.copy()
    # Moving Averages for momentum
    df['ma_short'] = df['close'].rolling(window=min(5, len(df))).mean()
    df['ma_long'] = df['close'].rolling(window=min(20, len(df))).mean()
    df['momentum'] = df['ma_short'] - df['ma_long']
    # RSI
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    # MACD
    df['macd'], df['macd_signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    # ATR
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    # Sentiment
    sentiment_result = sentiment_analyzer.analyze_social_media(symbol.split('/')[0], '1h')
    df['sentiment'] = sentiment_result['sentiment_score']
    # Fill NaNs
    df = df.fillna(0)
    # Keep 'close' alongside features for TradingEnv
    selected_columns = ['close', 'momentum', 'rsi', 'macd', 'atr', 'sentiment']
    return df[selected_columns]

def fetch_real_time_data(symbol):
    timeout = 60
    start_time = time.time()
    for _ in range(3):
        try:
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
                    print(f"Kraken trade detected for {symbol}: price={price}, timestamp={timestamp}")
                    return pd.DataFrame([[timestamp, price]], columns=['timestamp', 'close'])
            ws.close()
        except Exception as e:
            print(f"WebSocket error for {symbol}: {e}")
        time.sleep(5)
    print(f"Falling back to historical data for {symbol}")
    df = fetch_historical_data(symbol, limit=1)
    return df.tail(1) if not df.empty else pd.DataFrame([[pd.Timestamp.now(), 98700]], columns=['timestamp', 'close'])