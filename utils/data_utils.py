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
from models.train import CryptoGAN  # Corrected import

kraken = ccxt.kraken({
    'apiKey': KRAKEN_API_KEY,
    'secret': KRAKEN_API_SECRET,
    'enableRateLimit': True,
})

exchange = kraken if ACTIVE_EXCHANGE == 'kraken' else ccxt.coinbasepro()
sentiment_analyzer = SentimentAnalyzer()

def fetch_historical_data(symbol, timeframe='1h', limit=8760):  # 1 year of hourly data
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

def augment_data(df):
    gan = CryptoGAN(input_dim=5)  # Updated to match OHLCV (5 features: open, high, low, close, volume)
    real_data = df[['open', 'high', 'low', 'close', 'volume']].values
    synthetic_data = gan.generate(real_data)
    augmented_data = np.concatenate([real_data, synthetic_data])
    augmented_df = pd.DataFrame(augmented_data, columns=['open', 'high', 'low', 'close', 'volume'])
    augmented_df['timestamp'] = pd.date_range(start=df['timestamp'].iloc[0], periods=len(augmented_data), freq='1h')
    return augmented_df

def process_data(df, symbol):
    df = df.copy()
    for col in ['open', 'high', 'low']:
        if col not in df.columns:
            df[col] = df['close']
    df['volume'] = df.get('volume', 0)
    
    df['ma_short'] = df['close'].rolling(window=5).mean()
    df['ma_long'] = df['close'].rolling(window=20).mean()
    df['momentum'] = df['ma_short'] - df['ma_long']
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['macd'], df['macd_signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    sentiment_result = sentiment_analyzer.analyze_social_media(symbol.split('/')[0], '1h')
    df['sentiment'] = sentiment_result['sentiment_score']
    df['arbitrage_spread'] = 0  # Placeholder; implement if arbitrage data available
    df['whale_activity'] = 0  # Placeholder; requires ChainAnalyzer
    df['defi_apr'] = 0  # Placeholder; requires DeFiManager
    df = df.fillna(0)
    return df[['close', 'momentum', 'rsi', 'macd', 'atr', 'sentiment', 'arbitrage_spread', 'whale_activity', 'defi_apr']]

def fetch_real_time_data(symbol):
    timeout = 60
    start_time = time.time()
    retries = 5
    for attempt in range(retries):
        try:
            ws = create_connection('wss://ws.kraken.com')
            ws.send(json.dumps({
                "event": "subscribe",
                "pair": [symbol],
                "subscription": {"name": "ohlc", "interval": 1}
            }))
            while time.time() - start_time < timeout:
                message = ws.recv()
                print(f"Kraken WebSocket message: {message}")
                data = json.loads(message)
                if isinstance(data, list) and len(data) > 2 and data[2] == "ohlc":
                    ohlc = data[1]
                    timestamp = pd.to_datetime(float(ohlc[1]), unit='s')
                    df = pd.DataFrame([[
                        timestamp,
                        float(ohlc[2]),  # Open
                        float(ohlc[3]),  # High
                        float(ohlc[4]),  # Low
                        float(ohlc[5]),  # Close
                        float(ohlc[6])   # Volume
                    ]], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    ws.close()
                    print(f"Kraken OHLC detected for {symbol}: {df.iloc[-1].to_dict()}")
                    return df
            ws.close()
        except Exception as e:
            print(f"WebSocket error for {symbol} (attempt {attempt + 1}/{retries}): {e}")
        time.sleep(5)
    print(f"Falling back to historical data for {symbol}")
    df = fetch_historical_data(symbol, limit=1)
    return df.tail(1) if not df.empty else pd.DataFrame([[pd.Timestamp.now(), 98700, 98700, 98700, 98700, 0]], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])