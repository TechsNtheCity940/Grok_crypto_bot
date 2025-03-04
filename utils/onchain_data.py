import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger('onchain_data')

class OnChainDataProvider:
    """
    Provider for real on-chain data from various APIs like Glassnode, CoinMetrics, etc.
    """
    def __init__(self):
        # API keys from environment variables
        self.glassnode_api_key = os.getenv('GLASSNODE_API_KEY', '')
        self.coinmetrics_api_key = os.getenv('COINMETRICS_API_KEY', '')
        self.santiment_api_key = os.getenv('SANTIMENT_API_KEY', '')
        
        # Base URLs
        self.glassnode_base_url = 'https://api.glassnode.com/v1/metrics'
        self.coinmetrics_base_url = 'https://api.coinmetrics.io/v4'
        self.santiment_base_url = 'https://api.santiment.net/graphql'
        
        # Cache for API responses
        self.cache = {}
        self.cache_expiry = 3600  # 1 hour in seconds
    
    def _get_from_cache(self, key):
        """Get data from cache if available and not expired"""
        if key in self.cache:
            timestamp, data = self.cache[key]
            if time.time() - timestamp < self.cache_expiry:
                return data
        return None
    
    def _add_to_cache(self, key, data):
        """Add data to cache"""
        self.cache[key] = (time.time(), data)
    
    def fetch_glassnode_data(self, asset, metric, since=None, until=None, resolution='24h'):
        """
        Fetch data from Glassnode API
        
        Args:
            asset: Asset symbol (e.g., 'BTC')
            metric: Metric name (e.g., 'sopr')
            since: Start timestamp (optional)
            until: End timestamp (optional)
            resolution: Data resolution (default: '24h')
        
        Returns:
            DataFrame with the requested data
        """
        if not self.glassnode_api_key:
            logger.warning("Glassnode API key not found. Using simulated data.")
            return self._simulate_glassnode_data(asset, metric)
        
        # Create cache key
        cache_key = f"glassnode_{asset}_{metric}_{since}_{until}_{resolution}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Prepare parameters
        params = {
            'a': asset,
            'api_key': self.glassnode_api_key,
            'i': resolution
        }
        
        if since:
            params['s'] = since
        if until:
            params['u'] = until
        
        # Make API request
        try:
            url = f"{self.glassnode_base_url}/{metric}"
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            df = pd.DataFrame(data)
            
            # Rename columns
            df.rename(columns={'t': 'timestamp', 'v': metric}, inplace=True)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Add to cache
            self._add_to_cache(cache_key, df)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching data from Glassnode: {e}")
            return self._simulate_glassnode_data(asset, metric)
    
    def _simulate_glassnode_data(self, asset, metric):
        """Simulate Glassnode data for testing"""
        # Create date range for the last 30 days
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        # Simulate different metrics
        if metric == 'sopr':
            values = np.random.uniform(0.95, 1.05, size=30)
        elif metric == 'active_addresses':
            if asset == 'BTC':
                values = np.random.randint(800000, 1200000, size=30)
            elif asset == 'ETH':
                values = np.random.randint(500000, 700000, size=30)
            else:
                values = np.random.randint(10000, 100000, size=30)
        elif metric == 'difficulty':
            values = np.random.uniform(25, 30, size=30)
        elif metric == 'hash_rate':
            values = np.random.uniform(100, 150, size=30)
        else:
            values = np.random.uniform(0, 100, size=30)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            metric: values
        })
        
        return df
    
    def fetch_coinmetrics_data(self, asset, metric, start_date=None, end_date=None):
        """
        Fetch data from CoinMetrics API
        
        Args:
            asset: Asset symbol (e.g., 'btc')
            metric: Metric name (e.g., 'TxCnt')
            start_date: Start date (optional)
            end_date: End date (optional)
        
        Returns:
            DataFrame with the requested data
        """
        if not self.coinmetrics_api_key:
            logger.warning("CoinMetrics API key not found. Using simulated data.")
            return self._simulate_coinmetrics_data(asset, metric)
        
        # Create cache key
        cache_key = f"coinmetrics_{asset}_{metric}_{start_date}_{end_date}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Prepare parameters
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Make API request
        try:
            url = f"{self.coinmetrics_base_url}/timeseries/asset-metrics"
            params = {
                'assets': asset,
                'metrics': metric,
                'start_time': start_date,
                'end_time': end_date,
                'api_key': self.coinmetrics_api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            if 'data' in data:
                df = pd.DataFrame(data['data'])
                
                # Rename columns
                df.rename(columns={'time': 'timestamp'}, inplace=True)
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Add to cache
                self._add_to_cache(cache_key, df)
                
                return df
            else:
                logger.error(f"Invalid response from CoinMetrics: {data}")
                return self._simulate_coinmetrics_data(asset, metric)
        except Exception as e:
            logger.error(f"Error fetching data from CoinMetrics: {e}")
            return self._simulate_coinmetrics_data(asset, metric)
    
    def _simulate_coinmetrics_data(self, asset, metric):
        """Simulate CoinMetrics data for testing"""
        # Create date range for the last 30 days
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        # Simulate different metrics
        if metric == 'TxCnt':
            if asset == 'btc':
                values = np.random.randint(250000, 350000, size=30)
            elif asset == 'eth':
                values = np.random.randint(1000000, 1500000, size=30)
            else:
                values = np.random.randint(5000, 50000, size=30)
        elif metric == 'AdrActCnt':
            if asset == 'btc':
                values = np.random.randint(800000, 1200000, size=30)
            elif asset == 'eth':
                values = np.random.randint(500000, 700000, size=30)
            else:
                values = np.random.randint(10000, 100000, size=30)
        elif metric == 'FeeTotUSD':
            if asset == 'btc':
                values = np.random.uniform(1000000, 2000000, size=30)
            elif asset == 'eth':
                values = np.random.uniform(2000000, 5000000, size=30)
            else:
                values = np.random.uniform(10000, 100000, size=30)
        else:
            values = np.random.uniform(0, 100, size=30)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'asset': asset,
            metric: values
        })
        
        return df
    
    def fetch_santiment_data(self, asset, metric, from_date=None, to_date=None):
        """
        Fetch data from Santiment API
        
        Args:
            asset: Asset symbol (e.g., 'bitcoin')
            metric: Metric name (e.g., 'social_volume_total')
            from_date: Start date (optional)
            to_date: End date (optional)
        
        Returns:
            DataFrame with the requested data
        """
        if not self.santiment_api_key:
            logger.warning("Santiment API key not found. Using simulated data.")
            return self._simulate_santiment_data(asset, metric)
        
        # Create cache key
        cache_key = f"santiment_{asset}_{metric}_{from_date}_{to_date}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Prepare dates
        if not from_date:
            from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not to_date:
            to_date = datetime.now().strftime('%Y-%m-%d')
        
        # Prepare GraphQL query
        query = """
        {
          getMetric(metric: "%s") {
            timeseriesData(
              slug: "%s"
              from: "%s"
              to: "%s"
              interval: "1d"
            ) {
              datetime
              value
            }
          }
        }
        """ % (metric, asset, from_date, to_date)
        
        # Make API request
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f"Bearer {self.santiment_api_key}"
            }
            
            response = requests.post(
                self.santiment_base_url,
                json={'query': query},
                headers=headers
            )
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            if 'data' in data and 'getMetric' in data['data'] and 'timeseriesData' in data['data']['getMetric']:
                timeseries = data['data']['getMetric']['timeseriesData']
                df = pd.DataFrame(timeseries)
                
                # Rename columns
                df.rename(columns={'datetime': 'timestamp', 'value': metric}, inplace=True)
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Add to cache
                self._add_to_cache(cache_key, df)
                
                return df
            else:
                logger.error(f"Invalid response from Santiment: {data}")
                return self._simulate_santiment_data(asset, metric)
        except Exception as e:
            logger.error(f"Error fetching data from Santiment: {e}")
            return self._simulate_santiment_data(asset, metric)
    
    def _simulate_santiment_data(self, asset, metric):
        """Simulate Santiment data for testing"""
        # Create date range for the last 30 days
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        # Simulate different metrics
        if metric == 'social_volume_total':
            values = np.random.randint(1000, 10000, size=30)
        elif metric == 'dev_activity':
            values = np.random.randint(10, 100, size=30)
        elif metric == 'github_activity':
            values = np.random.randint(5, 50, size=30)
        else:
            values = np.random.uniform(0, 100, size=30)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            metric: values
        })
        
        return df
    
    def fetch_combined_metrics(self, symbol, days=30):
        """
        Fetch and combine multiple on-chain metrics for a given symbol
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD')
            days: Number of days of data to fetch
        
        Returns:
            DataFrame with combined metrics
        """
        # Extract base currency
        base_currency = symbol.split('/')[0]
        
        # Map to API-specific asset names
        glassnode_asset = base_currency
        coinmetrics_asset = base_currency.lower()
        santiment_asset = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'DOGE': 'dogecoin',
            'XRP': 'ripple',
            'SHIB': 'shiba-inu'
        }.get(base_currency, base_currency.lower())
        
        # Fetch data from different sources
        dfs = []
        
        # Glassnode metrics
        if base_currency in ['BTC', 'ETH']:
            metrics = ['sopr', 'difficulty', 'hash_rate'] if base_currency == 'BTC' else ['active_addresses']
            for metric in metrics:
                df = self.fetch_glassnode_data(glassnode_asset, metric)
                if not df.empty:
                    dfs.append(df)
        
        # CoinMetrics metrics
        metrics = ['TxCnt', 'AdrActCnt', 'FeeTotUSD']
        for metric in metrics:
            df = self.fetch_coinmetrics_data(coinmetrics_asset, metric)
            if not df.empty:
                dfs.append(df)
        
        # Santiment metrics
        metrics = ['social_volume_total', 'dev_activity']
        for metric in metrics:
            df = self.fetch_santiment_data(santiment_asset, metric)
            if not df.empty:
                dfs.append(df)
        
        # Combine all dataframes
        if not dfs:
            logger.warning(f"No on-chain data available for {symbol}. Using simulated data.")
            return self._simulate_combined_metrics(base_currency, days)
        
        # Merge dataframes on timestamp
        result_df = dfs[0]
        for df in dfs[1:]:
            result_df = pd.merge(result_df, df, on='timestamp', how='outer')
        
        # Sort by timestamp
        result_df = result_df.sort_values('timestamp')
        
        # Fill missing values
        result_df = result_df.ffill().bfill()
        
        return result_df
    
    def _simulate_combined_metrics(self, base_currency, days=30):
        """Simulate combined metrics for testing"""
        # Create date range
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Create base DataFrame
        df = pd.DataFrame({'timestamp': dates})
        
        # Add metrics based on currency
        if base_currency == 'BTC':
            df['active_addresses'] = np.random.randint(800000, 1200000, size=days)
            df['transaction_count'] = np.random.randint(250000, 350000, size=days)
            df['avg_transaction_value'] = np.random.uniform(0.5, 2.0, size=days)
            df['hash_rate'] = np.random.uniform(100, 150, size=days)
            df['difficulty'] = np.random.uniform(25, 30, size=days)
            df['miner_revenue'] = np.random.uniform(15, 25, size=days)
            df['sopr'] = np.random.uniform(0.95, 1.05, size=days)
        elif base_currency == 'ETH':
            df['active_addresses'] = np.random.randint(500000, 700000, size=days)
            df['transaction_count'] = np.random.randint(1000000, 1500000, size=days)
            df['gas_used'] = np.random.uniform(50, 100, size=days)
            df['avg_gas_price'] = np.random.uniform(30, 100, size=days)
            df['defi_tvl'] = np.random.uniform(40, 80, size=days)
            df['eth_staked'] = np.random.uniform(10, 20, size=days)
        else:
            df['active_addresses'] = np.random.randint(10000, 100000, size=days)
            df['transaction_count'] = np.random.randint(5000, 50000, size=days)
            df['avg_transaction_value'] = np.random.uniform(100, 1000, size=days)
            df['large_transactions'] = np.random.randint(100, 1000, size=days)
        
        # Add common metrics
        df['social_volume'] = np.random.randint(1000, 10000, size=days)
        df['dev_activity'] = np.random.randint(10, 100, size=days)
        
        return df
