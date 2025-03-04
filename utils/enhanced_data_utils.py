import sys
import os
import time
import json
import requests
import pandas as pd
import numpy as np
import ccxt
from websocket import create_connection
import talib
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

from utils.data_utils import fetch_historical_data, process_data
from utils.log_setup import logger
from config_manager import config
from sentiment_analyzer import SentimentAnalyzer

class MultiExchangeDataFetcher:
    """
    Enhanced data fetcher that collects data from multiple exchanges
    and enriches it with on-chain metrics and market indicators.
    """
    def __init__(self):
        # Initialize exchange connections
        self.exchanges = {
            'kraken': ccxt.kraken({
                'apiKey': config.get('kraken_api_key'),
                'secret': config.get('kraken_api_secret'),
                'enableRateLimit': True
            })
        }
        
        # Add more exchanges if needed
        if config.get('use_binance', False):
            self.exchanges['binance'] = ccxt.binance({
                'enableRateLimit': True
            })
            
        if config.get('use_coinbase', False):
            self.exchanges['coinbase'] = ccxt.coinbasepro({
                'enableRateLimit': True
            })
            
        self.sentiment_analyzer = SentimentAnalyzer()
        
    def fetch_multi_exchange(self, symbol, timeframe='1h', limit=1000):
        """Fetch data from multiple exchanges and merge"""
        all_dfs = []
        
        # Use ThreadPoolExecutor for parallel fetching
        with ThreadPoolExecutor(max_workers=len(self.exchanges)) as executor:
            futures = {
                exchange_name: executor.submit(
                    self._fetch_from_exchange, 
                    exchange, 
                    symbol, 
                    timeframe, 
                    limit
                )
                for exchange_name, exchange in self.exchanges.items()
            }
            
            # Collect results
            for exchange_name, future in futures.items():
                try:
                    df = future.result()
                    if not df.empty:
                        df['exchange'] = exchange_name
                        all_dfs.append(df)
                        logger.info(f"Fetched {len(df)} rows from {exchange_name} for {symbol}")
                except Exception as e:
                    logger.error(f"Error fetching from {exchange_name}: {e}")
        
        if not all_dfs:
            logger.warning(f"No data fetched for {symbol} from any exchange, falling back to historical data")
            return fetch_historical_data(symbol, timeframe, limit)
            
        # Merge dataframes
        merged_df = pd.concat(all_dfs).sort_values('timestamp')
        
        # Remove duplicates (same timestamp)
        merged_df = merged_df.drop_duplicates(subset=['timestamp'])
        
        # Calculate arbitrage opportunities
        if len(all_dfs) > 1:
            merged_df = self._calculate_arbitrage(merged_df, symbol)
            
        return merged_df
    
    def _fetch_from_exchange(self, exchange, symbol, timeframe, limit):
        """Fetch data from a specific exchange"""
        try:
            since = exchange.milliseconds() - (limit * 3600 * 1000)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error in _fetch_from_exchange: {e}")
            return pd.DataFrame()
    
    def _calculate_arbitrage(self, df, symbol):
        """Calculate arbitrage opportunities between exchanges"""
        exchanges = df['exchange'].unique()
        
        # Create pivot table with prices from different exchanges
        pivot = df.pivot_table(
            index='timestamp', 
            columns='exchange', 
            values='close', 
            aggfunc='last'
        ).ffill()
        
        # Calculate max price difference as percentage
        pivot['max_price'] = pivot.max(axis=1)
        pivot['min_price'] = pivot.min(axis=1)
        pivot['arbitrage_spread'] = (pivot['max_price'] - pivot['min_price']) / pivot['min_price']
        
        # Merge arbitrage data back to original dataframe
        arbitrage_df = pivot[['arbitrage_spread']].reset_index()
        df = pd.merge(df, arbitrage_df, on='timestamp', how='left')
        
        # Fill missing values
        df['arbitrage_spread'] = df['arbitrage_spread'].fillna(0)
        
        return df
    
    def fetch_on_chain_metrics(self, symbol, timeframe='1h'):
        """Fetch on-chain metrics for the given symbol"""
        # Extract the base currency (e.g., 'BTC' from 'BTC/USD')
        base_currency = symbol.split('/')[0]
        
        # Initialize metrics dataframe
        metrics_df = pd.DataFrame()
        
        try:
            # For Bitcoin
            if base_currency in ['BTC', 'XBT']:
                metrics_df = self._fetch_btc_metrics()
            # For Ethereum
            elif base_currency == 'ETH':
                metrics_df = self._fetch_eth_metrics()
            # For other coins, try generic approach
            else:
                metrics_df = self._fetch_generic_metrics(base_currency)
                
            if not metrics_df.empty:
                logger.info(f"Fetched on-chain metrics for {base_currency}")
                
            return metrics_df
        except Exception as e:
            logger.error(f"Error fetching on-chain metrics for {base_currency}: {e}")
            return pd.DataFrame()
    
    def _fetch_btc_metrics(self):
        """Fetch Bitcoin-specific on-chain metrics"""
        # This would normally use APIs like Glassnode, Coinmetrics, etc.
        # For now, we'll simulate the data
        
        # Create date range for the last 30 days
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
        
        # Simulate metrics
        df = pd.DataFrame({
            'timestamp': dates,
            'active_addresses': np.random.randint(800000, 1200000, size=30),
            'transaction_count': np.random.randint(250000, 350000, size=30),
            'avg_transaction_value': np.random.uniform(0.5, 2.0, size=30),
            'hash_rate': np.random.uniform(100, 150, size=30),  # EH/s
            'difficulty': np.random.uniform(25, 30, size=30),  # T
            'miner_revenue': np.random.uniform(15, 25, size=30),  # Million USD
            'utxo_count': np.random.randint(60000000, 80000000, size=30),
            'sopr': np.random.uniform(0.95, 1.05, size=30)  # Spent Output Profit Ratio
        })
        
        return df
    
    def _fetch_eth_metrics(self):
        """Fetch Ethereum-specific on-chain metrics"""
        # Simulate Ethereum metrics
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
        
        df = pd.DataFrame({
            'timestamp': dates,
            'active_addresses': np.random.randint(500000, 700000, size=30),
            'transaction_count': np.random.randint(1000000, 1500000, size=30),
            'gas_used': np.random.uniform(50, 100, size=30),  # Billion
            'avg_gas_price': np.random.uniform(30, 100, size=30),  # Gwei
            'defi_tvl': np.random.uniform(40, 80, size=30),  # Billion USD
            'eth_staked': np.random.uniform(10, 20, size=30),  # Million ETH
            'eth_burned': np.random.uniform(5000, 10000, size=30)  # ETH per day
        })
        
        return df
    
    def _fetch_generic_metrics(self, currency):
        """Fetch generic on-chain metrics for other cryptocurrencies"""
        # Simulate generic metrics
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
        
        df = pd.DataFrame({
            'timestamp': dates,
            'active_addresses': np.random.randint(10000, 100000, size=30),
            'transaction_count': np.random.randint(5000, 50000, size=30),
            'avg_transaction_value': np.random.uniform(100, 1000, size=30),
            'large_transactions': np.random.randint(100, 1000, size=30)  # Transactions > $100k
        })
        
        return df
    
    def fetch_whale_activity(self, symbol, days=30):
        """Fetch whale activity metrics"""
        base_currency = symbol.split('/')[0]
        
        # Simulate whale activity data
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
        
        df = pd.DataFrame({
            'timestamp': dates,
            'whale_transaction_count': np.random.randint(10, 100, size=days),
            'whale_transaction_volume': np.random.uniform(1000000, 10000000, size=days),
            'whale_accumulation': np.random.uniform(-0.1, 0.1, size=days)  # Negative means distribution
        })
        
        return df
    
    def fetch_defi_metrics(self, symbol):
        """Fetch DeFi-related metrics"""
        base_currency = symbol.split('/')[0]
        
        # Simulate DeFi metrics
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
        
        df = pd.DataFrame({
            'timestamp': dates,
            'lending_rate': np.random.uniform(0.01, 0.1, size=30),  # APR
            'borrowing_rate': np.random.uniform(0.03, 0.15, size=30),  # APR
            'liquidity_pool_size': np.random.uniform(1000000, 10000000, size=30),
            'total_value_locked': np.random.uniform(10000000, 100000000, size=30)
        })
        
        return df

def enrich_data(df, symbol):
    """
    Enrich the price data with additional metrics and indicators
    """
    # Create a copy to avoid modifying the original
    enriched_df = df.copy()
    
    # Make sure we have all required columns
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in enriched_df.columns:
            if col == 'volume':
                enriched_df[col] = 0
            else:
                enriched_df[col] = enriched_df['close']
    
    # Add standard technical indicators
    enriched_df['ma_short'] = enriched_df['close'].rolling(window=5).mean()
    enriched_df['ma_long'] = enriched_df['close'].rolling(window=20).mean()
    enriched_df['momentum'] = enriched_df['ma_short'] - enriched_df['ma_long']
    enriched_df['rsi'] = talib.RSI(enriched_df['close'], timeperiod=14)
    enriched_df['macd'], enriched_df['macd_signal'], _ = talib.MACD(
        enriched_df['close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    enriched_df['atr'] = talib.ATR(
        enriched_df['high'], enriched_df['low'], enriched_df['close'], timeperiod=14
    )
    enriched_df['bb_upper'], enriched_df['bb_middle'], enriched_df['bb_lower'] = talib.BBANDS(
        enriched_df['close'], timeperiod=20
    )
    
    # Add advanced indicators
    enriched_df['adx'] = talib.ADX(
        enriched_df['high'], enriched_df['low'], enriched_df['close'], timeperiod=14
    )
    enriched_df['cci'] = talib.CCI(
        enriched_df['high'], enriched_df['low'], enriched_df['close'], timeperiod=14
    )
    enriched_df['obv'] = talib.OBV(enriched_df['close'], enriched_df['volume'])
    
    # Add volatility indicators
    enriched_df['historical_vol'] = enriched_df['close'].pct_change().rolling(window=20).std() * np.sqrt(365)
    
    # Add trend indicators
    enriched_df['psar'] = talib.SAR(enriched_df['high'], enriched_df['low'], acceleration=0.02, maximum=0.2)
    
    # Add sentiment data
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_result = sentiment_analyzer.analyze_social_media(symbol.split('/')[0], '1h')
    enriched_df['sentiment'] = sentiment_result['sentiment_score']
    
    # Add arbitrage spread if available, otherwise set to 0
    if 'arbitrage_spread' not in enriched_df.columns:
        enriched_df['arbitrage_spread'] = 0
    
    # Add whale activity (simulated)
    if 'whale_activity' not in enriched_df.columns:
        enriched_df['whale_activity'] = enriched_df['volume'].rolling(window=24).mean() * 0.1
    
    # Add DeFi APR (simulated)
    if 'defi_apr' not in enriched_df.columns:
        enriched_df['defi_apr'] = 0.05  # 5% APR as placeholder
    
    # Fill missing values
    enriched_df = enriched_df.fillna(0)
    
    return enriched_df

def fetch_and_process_enhanced_data(symbol, timeframe='1h', limit=1000):
    """
    Fetch and process enhanced data from multiple sources
    """
    # Initialize multi-exchange fetcher
    fetcher = MultiExchangeDataFetcher()
    
    # Fetch price data from multiple exchanges
    df = fetcher.fetch_multi_exchange(symbol, timeframe, limit)
    
    # Enrich with additional metrics
    enriched_df = enrich_data(df, symbol)
    
    # Process data using the standard processor
    processed_df = process_data(enriched_df, symbol)
    
    return processed_df

def detect_market_regime(df, window=20):
    """
    Detect the current market regime (trending, ranging, volatile)
    """
    # Calculate metrics for regime detection
    returns = df['close'].pct_change()
    volatility = returns.rolling(window=window).std()
    adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=window)
    
    # Determine regime
    is_volatile = volatility > volatility.rolling(window=window*2).mean() * 1.5
    is_trending = adx > 25  # ADX > 25 indicates a trend
    
    # Create regime column
    regimes = []
    for vol, trend in zip(is_volatile, is_trending):
        if vol:
            regimes.append('volatile')
        elif trend:
            regimes.append('trending')
        else:
            regimes.append('ranging')
    
    regime_df = pd.DataFrame({
        'timestamp': df['timestamp'],
        'regime': regimes,
        'volatility': volatility,
        'adx': adx
    })
    
    return regime_df
