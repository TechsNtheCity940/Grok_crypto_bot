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
from models.train import CryptoGAN

kraken = ccxt.kraken({
    'apiKey': KRAKEN_API_KEY,
    'secret': KRAKEN_API_SECRET,
    'enableRateLimit': True,
})

exchange = kraken if ACTIVE_EXCHANGE == 'kraken' else ccxt.coinbasepro()
sentiment_analyzer = SentimentAnalyzer()

def fetch_historical_data(symbol, timeframe='1h', limit=8760):
    retries = 3
    df = pd.DataFrame()
    for _ in range(retries):
        try:
            since = exchange.milliseconds() - (limit * 3600 * 1000)  # 1 year back
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            new_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
            df = pd.concat([df, new_df]).drop_duplicates().sort_values('timestamp')
            if len(df) >= limit:
                break
        except Exception as e:
            print(f"Failed to fetch historical data for {symbol}: {e}")
            time.sleep(5)
    df.to_csv(f'data/historical/{symbol.replace("/", "_")}_{timeframe}.csv', index=False)
    return df

def augment_data(df):
    gan = CryptoGAN(input_dim=5)
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
    df['atr'] = talib.ATR(df['high'], df['low'], 'close', timeperiod=14)
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'], timeperiod=20)
    sentiment_result = sentiment_analyzer.analyze_social_media(symbol.split('/')[0], '1h')
    df['sentiment'] = sentiment_result['sentiment_score']
    df['arbitrage_spread'] = 0  # Placeholder for arbitrage
    df['whale_activity'] = df['volume'].rolling(window=24).mean() * 0.1  # Simulated whale proxy
    df['defi_apr'] = 0  # Placeholder, kept as 0 for now
    df = df.fillna(0)
    # Include defi_apr in the return list
    return df[['close', 'momentum', 'rsi', 'macd', 'atr', 'sentiment', 'arbitrage_spread', 'whale_activity', 'bb_upper', 'defi_apr']]

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
                data = json.loads(message)
                if isinstance(data, list) and len(data) > 2 and data[2] == "ohlc":
                    ohlc = data[1]
                    timestamp = pd.to_datetime(float(ohlc[1]), unit='s')
                    df = pd.DataFrame([[
                        timestamp,
                        float(ohlc[2]), float(ohlc[3]), float(ohlc[4]), float(ohlc[5]), float(ohlc[6])
                    ]], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    ws.close()
                    return df
            ws.close()
        except Exception as e:
            print(f"WebSocket error for {symbol}: {e}")
        time.sleep(5)
    df = fetch_historical_data(symbol, limit=1)
    return df.tail(1) if not df.empty else pd.DataFrame([[pd.Timestamp.now(), 0.232, 0.232, 0.232, 0.232, 0]], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])