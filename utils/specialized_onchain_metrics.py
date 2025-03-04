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
logger = logging.getLogger('specialized_onchain_metrics')

class SpecializedOnChainMetrics:
    """
    Provider for specialized on-chain metrics for specific cryptocurrencies.
    Extends the basic OnChainDataProvider with more advanced and coin-specific metrics.
    """
    def __init__(self):
        # API keys from environment variables
        self.glassnode_api_key = os.getenv('GLASSNODE_API_KEY', '')
        self.coinmetrics_api_key = os.getenv('COINMETRICS_API_KEY', '')
        self.santiment_api_key = os.getenv('SANTIMENT_API_KEY', '')
        self.cryptoquant_api_key = os.getenv('CRYPTOQUANT_API_KEY', '')
        self.intotheblock_api_key = os.getenv('INTOTHEBLOCK_API_KEY', '')
        self.messari_api_key = os.getenv('MESSARI_API_KEY', '')
        
        # Base URLs
        self.glassnode_base_url = 'https://api.glassnode.com/v1/metrics'
        self.coinmetrics_base_url = 'https://api.coinmetrics.io/v4'
        self.santiment_base_url = 'https://api.santiment.net/graphql'
        self.cryptoquant_base_url = 'https://api.cryptoquant.com/v1'
        self.intotheblock_base_url = 'https://api.intotheblock.com/v1'
        self.messari_base_url = 'https://data.messari.io/api/v1'
        
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
    
    # Bitcoin-specific metrics
    def fetch_bitcoin_metrics(self, days=30):
        """
        Fetch specialized Bitcoin metrics
        
        Args:
            days: Number of days of data to fetch
        
        Returns:
            DataFrame with Bitcoin-specific metrics
        """
        # Create cache key
        cache_key = f"bitcoin_metrics_{days}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Initialize metrics dataframe
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        df = pd.DataFrame({'timestamp': dates})
        
        # Fetch metrics from various sources
        try:
            # UTXO Age Distribution (Glassnode)
            if self.glassnode_api_key:
                utxo_age = self._fetch_glassnode_metric('btc', 'utxo_age_distribution')
                if not utxo_age.empty:
                    df = pd.merge(df, utxo_age, on='timestamp', how='left')
            
            # SOPR by Address Type (Glassnode)
            if self.glassnode_api_key:
                sopr_metrics = self._fetch_glassnode_metric('btc', 'sopr_address_type')
                if not sopr_metrics.empty:
                    df = pd.merge(df, sopr_metrics, on='timestamp', how='left')
            
            # Miner Outflow (CryptoQuant)
            if self.cryptoquant_api_key:
                miner_outflow = self._fetch_cryptoquant_metric('btc', 'miner_outflow')
                if not miner_outflow.empty:
                    df = pd.merge(df, miner_outflow, on='timestamp', how='left')
            
            # Exchange Reserves (CryptoQuant)
            if self.cryptoquant_api_key:
                exchange_reserves = self._fetch_cryptoquant_metric('btc', 'exchange_reserves')
                if not exchange_reserves.empty:
                    df = pd.merge(df, exchange_reserves, on='timestamp', how='left')
            
            # Stablecoin Supply Ratio (Glassnode)
            if self.glassnode_api_key:
                ssr = self._fetch_glassnode_metric('btc', 'ssr')
                if not ssr.empty:
                    df = pd.merge(df, ssr, on='timestamp', how='left')
            
            # RHODL Ratio (Glassnode)
            if self.glassnode_api_key:
                rhodl = self._fetch_glassnode_metric('btc', 'rhodl_ratio')
                if not rhodl.empty:
                    df = pd.merge(df, rhodl, on='timestamp', how='left')
            
            # Realized Cap HODL Waves (Glassnode)
            if self.glassnode_api_key:
                hodl_waves = self._fetch_glassnode_metric('btc', 'realized_cap_hodl_waves')
                if not hodl_waves.empty:
                    df = pd.merge(df, hodl_waves, on='timestamp', how='left')
            
            # Lightning Network Metrics (1ML API)
            lightning_metrics = self._fetch_lightning_network_metrics()
            if not lightning_metrics.empty:
                df = pd.merge(df, lightning_metrics, on='timestamp', how='left')
            
            # Add simulated metrics if real data is missing
            if len(df.columns) < 5:  # If we have less than 5 metrics, add simulated ones
                logger.warning("Limited real Bitcoin metrics available. Adding simulated data.")
                df = self._simulate_bitcoin_metrics(df, days)
            
            # Fill missing values
            df = df.ffill().bfill()
            
            # Add to cache
            self._add_to_cache(cache_key, df)
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching Bitcoin metrics: {e}")
            return self._simulate_bitcoin_metrics(pd.DataFrame({'timestamp': dates}), days)
    
    def _fetch_glassnode_metric(self, asset, metric):
        """Fetch a specific metric from Glassnode"""
        try:
            url = f"{self.glassnode_base_url}/{metric}"
            params = {
                'a': asset,
                'api_key': self.glassnode_api_key,
                'i': '24h'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data)
            
            # Rename columns
            df.rename(columns={'t': 'timestamp', 'v': metric}, inplace=True)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            return df
        except Exception as e:
            logger.error(f"Error fetching Glassnode metric {metric}: {e}")
            return pd.DataFrame()
    
    def _fetch_cryptoquant_metric(self, asset, metric):
        """Fetch a specific metric from CryptoQuant"""
        try:
            url = f"{self.cryptoquant_base_url}/{asset}/{metric}"
            headers = {
                'Authorization': f"Bearer {self.cryptoquant_api_key}"
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data['data'])
            
            # Rename columns
            df.rename(columns={'date': 'timestamp', 'value': metric}, inplace=True)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
        except Exception as e:
            logger.error(f"Error fetching CryptoQuant metric {metric}: {e}")
            return pd.DataFrame()
    
    def _fetch_lightning_network_metrics(self):
        """Fetch Lightning Network metrics"""
        try:
            url = "https://1ml.com/statistics?json=true"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            # Create dataframe with today's date
            df = pd.DataFrame([{
                'timestamp': datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
                'ln_node_count': data.get('nodeCount', 0),
                'ln_channel_count': data.get('channelCount', 0),
                'ln_capacity_btc': data.get('capacity', 0) / 100000000,  # Convert sats to BTC
                'ln_avg_channel_size': data.get('avgCapacity', 0) / 100000000  # Convert sats to BTC
            }])
            
            return df
        except Exception as e:
            logger.error(f"Error fetching Lightning Network metrics: {e}")
            return pd.DataFrame()
    
    def _simulate_bitcoin_metrics(self, df, days):
        """Simulate Bitcoin-specific metrics"""
        # Add UTXO age distribution
        df['utxo_1d_1w'] = np.random.uniform(0.05, 0.15, size=days)  # 1d-1w
        df['utxo_1w_1m'] = np.random.uniform(0.1, 0.2, size=days)    # 1w-1m
        df['utxo_1m_3m'] = np.random.uniform(0.1, 0.2, size=days)    # 1m-3m
        df['utxo_3m_6m'] = np.random.uniform(0.1, 0.2, size=days)    # 3m-6m
        df['utxo_6m_12m'] = np.random.uniform(0.1, 0.15, size=days)  # 6m-12m
        df['utxo_1y_2y'] = np.random.uniform(0.1, 0.15, size=days)   # 1y-2y
        df['utxo_2y_3y'] = np.random.uniform(0.05, 0.1, size=days)   # 2y-3y
        df['utxo_3y_5y'] = np.random.uniform(0.05, 0.1, size=days)   # 3y-5y
        df['utxo_5y_plus'] = np.random.uniform(0.05, 0.1, size=days) # 5y+
        
        # Add SOPR metrics
        df['sopr'] = np.random.uniform(0.95, 1.05, size=days)
        df['sopr_short_term'] = np.random.uniform(0.9, 1.1, size=days)
        df['sopr_long_term'] = np.random.uniform(0.98, 1.02, size=days)
        
        # Add miner metrics
        df['miner_outflow'] = np.random.uniform(500, 1500, size=days)
        df['miner_first_spend'] = np.random.uniform(100, 500, size=days)
        df['hash_rate'] = np.random.uniform(100, 150, size=days)  # EH/s
        df['difficulty'] = np.random.uniform(25, 30, size=days)   # T
        
        # Add exchange metrics
        df['exchange_reserves'] = np.random.uniform(2000000, 2500000, size=days)
        df['exchange_netflow'] = np.random.normal(0, 1000, size=days)
        
        # Add stablecoin metrics
        df['ssr'] = np.random.uniform(2, 6, size=days)
        df['stablecoin_supply_usd'] = np.random.uniform(80, 100, size=days)  # Billions
        
        # Add market cycle metrics
        df['rhodl_ratio'] = np.random.uniform(500, 2000, size=days)
        df['puell_multiple'] = np.random.uniform(0.5, 2.0, size=days)
        df['pi_cycle_top'] = np.random.uniform(0.5, 1.5, size=days)
        
        # Add Lightning Network metrics
        df['ln_node_count'] = np.random.randint(15000, 20000, size=days)
        df['ln_channel_count'] = np.random.randint(50000, 70000, size=days)
        df['ln_capacity_btc'] = np.random.uniform(3000, 5000, size=days)
        df['ln_avg_channel_size'] = np.random.uniform(0.03, 0.05, size=days)
        
        return df
    
    # Ethereum-specific metrics
    def fetch_ethereum_metrics(self, days=30):
        """
        Fetch specialized Ethereum metrics
        
        Args:
            days: Number of days of data to fetch
        
        Returns:
            DataFrame with Ethereum-specific metrics
        """
        # Create cache key
        cache_key = f"ethereum_metrics_{days}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Initialize metrics dataframe
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        df = pd.DataFrame({'timestamp': dates})
        
        # Fetch metrics from various sources
        try:
            # Gas metrics (Etherscan API)
            gas_metrics = self._fetch_etherscan_gas_metrics()
            if not gas_metrics.empty:
                df = pd.merge(df, gas_metrics, on='timestamp', how='left')
            
            # ETH 2.0 Staking metrics (Beaconcha.in API)
            staking_metrics = self._fetch_eth2_staking_metrics()
            if not staking_metrics.empty:
                df = pd.merge(df, staking_metrics, on='timestamp', how='left')
            
            # DeFi metrics (DeFi Pulse API)
            defi_metrics = self._fetch_defi_metrics()
            if not defi_metrics.empty:
                df = pd.merge(df, defi_metrics, on='timestamp', how='left')
            
            # NFT metrics (OpenSea API)
            nft_metrics = self._fetch_nft_metrics()
            if not nft_metrics.empty:
                df = pd.merge(df, nft_metrics, on='timestamp', how='left')
            
            # Layer 2 metrics (L2Beat API)
            l2_metrics = self._fetch_l2_metrics()
            if not l2_metrics.empty:
                df = pd.merge(df, l2_metrics, on='timestamp', how='left')
            
            # EIP-1559 metrics (Glassnode)
            if self.glassnode_api_key:
                eip1559_metrics = self._fetch_glassnode_metric('eth', 'eip1559_fees')
                if not eip1559_metrics.empty:
                    df = pd.merge(df, eip1559_metrics, on='timestamp', how='left')
            
            # Add simulated metrics if real data is missing
            if len(df.columns) < 5:  # If we have less than 5 metrics, add simulated ones
                logger.warning("Limited real Ethereum metrics available. Adding simulated data.")
                df = self._simulate_ethereum_metrics(df, days)
            
            # Fill missing values
            df = df.ffill().bfill()
            
            # Add to cache
            self._add_to_cache(cache_key, df)
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching Ethereum metrics: {e}")
            return self._simulate_ethereum_metrics(pd.DataFrame({'timestamp': dates}), days)
    
    def _fetch_etherscan_gas_metrics(self):
        """Fetch gas metrics from Etherscan"""
        try:
            etherscan_api_key = os.getenv('ETHERSCAN_API_KEY', '')
            if not etherscan_api_key:
                return pd.DataFrame()
            
            url = f"https://api.etherscan.io/api?module=gastracker&action=gasoracle&apikey={etherscan_api_key}"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            if data['status'] == '1':
                result = data['result']
                
                # Create dataframe with today's date
                df = pd.DataFrame([{
                    'timestamp': datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
                    'gas_safe': int(result['SafeGasPrice']),
                    'gas_standard': int(result['ProposeGasPrice']),
                    'gas_fast': int(result['FastGasPrice']),
                    'base_fee': int(result['suggestBaseFee'])
                }])
                
                return df
            else:
                logger.error(f"Etherscan API error: {data['message']}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching Etherscan gas metrics: {e}")
            return pd.DataFrame()
    
    def _fetch_eth2_staking_metrics(self):
        """Fetch ETH 2.0 staking metrics from Beaconcha.in"""
        try:
            url = "https://beaconcha.in/api/v1/epoch/latest"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            if data['status'] == 'OK':
                epoch_data = data['data']
                
                # Create dataframe with today's date
                df = pd.DataFrame([{
                    'timestamp': datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
                    'eth2_validators': epoch_data['validatorscount'],
                    'eth2_staked': epoch_data['totalvalidatorbalance'] / 1e9,  # Convert Gwei to ETH
                    'eth2_participation_rate': epoch_data['globalparticipationrate'],
                    'eth2_current_epoch': epoch_data['epoch']
                }])
                
                return df
            else:
                logger.error("Beaconcha.in API error")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching ETH 2.0 staking metrics: {e}")
            return pd.DataFrame()
    
    def _fetch_defi_metrics(self):
        """Fetch DeFi metrics from DeFi Pulse"""
        try:
            defipulse_api_key = os.getenv('DEFIPULSE_API_KEY', '')
            if not defipulse_api_key:
                return pd.DataFrame()
            
            url = f"https://data-api.defipulse.com/api/v1/defipulse/api/GetHistory?api-key={defipulse_api_key}"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to dataframe
            records = []
            for entry in data:
                records.append({
                    'timestamp': datetime.fromtimestamp(entry['timestamp']).replace(hour=0, minute=0, second=0, microsecond=0),
                    'defi_tvl': entry['totalLocked']['ETH'],
                    'defi_tvl_usd': entry['totalLocked']['USD'],
                    'defi_dominance_maker': entry['dominance']['maker'],
                    'defi_dominance_aave': entry['dominance']['aave']
                })
            
            df = pd.DataFrame(records)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching DeFi metrics: {e}")
            return pd.DataFrame()
    
    def _fetch_nft_metrics(self):
        """Fetch NFT metrics from OpenSea"""
        try:
            opensea_api_key = os.getenv('OPENSEA_API_KEY', '')
            if not opensea_api_key:
                return pd.DataFrame()
            
            url = "https://api.opensea.io/api/v1/collections/stats"
            headers = {
                'X-API-KEY': opensea_api_key
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            # Create dataframe with today's date
            df = pd.DataFrame([{
                'timestamp': datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
                'nft_total_volume': data['total_volume'],
                'nft_total_sales': data['total_sales'],
                'nft_total_supply': data['total_supply'],
                'nft_num_owners': data['num_owners'],
                'nft_average_price': data['average_price'],
                'nft_market_cap': data['market_cap']
            }])
            
            return df
        except Exception as e:
            logger.error(f"Error fetching NFT metrics: {e}")
            return pd.DataFrame()
    
    def _fetch_l2_metrics(self):
        """Fetch Layer 2 metrics from L2Beat"""
        try:
            url = "https://api.l2beat.com/api/tvl"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            # Create dataframe with today's date
            df = pd.DataFrame([{
                'timestamp': datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
                'l2_total_tvl': data['tvl']['total']['value'],
                'l2_arbitrum_tvl': next((p['tvl'] for p in data['projects'] if p['name'] == 'Arbitrum'), 0),
                'l2_optimism_tvl': next((p['tvl'] for p in data['projects'] if p['name'] == 'Optimism'), 0),
                'l2_zksync_tvl': next((p['tvl'] for p in data['projects'] if p['name'] == 'zkSync'), 0),
                'l2_starknet_tvl': next((p['tvl'] for p in data['projects'] if p['name'] == 'StarkNet'), 0)
            }])
            
            return df
        except Exception as e:
            logger.error(f"Error fetching Layer 2 metrics: {e}")
            return pd.DataFrame()
    
    def _simulate_ethereum_metrics(self, df, days):
        """Simulate Ethereum-specific metrics"""
        # Add gas metrics
        df['gas_safe'] = np.random.randint(20, 40, size=days)
        df['gas_standard'] = np.random.randint(30, 60, size=days)
        df['gas_fast'] = np.random.randint(50, 100, size=days)
        df['base_fee'] = np.random.uniform(1, 5, size=days)
        
        # Add ETH 2.0 staking metrics
        df['eth2_validators'] = np.random.randint(300000, 400000, size=days)
        df['eth2_staked'] = np.random.uniform(10, 20, size=days)  # Million ETH
        df['eth2_participation_rate'] = np.random.uniform(0.95, 0.99, size=days)
        df['eth2_current_epoch'] = np.random.randint(100000, 150000, size=days)
        
        # Add DeFi metrics
        df['defi_tvl'] = np.random.uniform(20, 40, size=days)  # Million ETH
        df['defi_tvl_usd'] = np.random.uniform(40, 80, size=days)  # Billion USD
        df['defi_dominance_maker'] = np.random.uniform(0.1, 0.2, size=days)
        df['defi_dominance_aave'] = np.random.uniform(0.1, 0.2, size=days)
        
        # Add NFT metrics
        df['nft_total_volume'] = np.random.uniform(1, 5, size=days)  # Billion USD
        df['nft_total_sales'] = np.random.randint(10000, 50000, size=days)
        df['nft_total_supply'] = np.random.randint(1000000, 2000000, size=days)
        df['nft_num_owners'] = np.random.randint(500000, 1000000, size=days)
        df['nft_average_price'] = np.random.uniform(0.1, 0.5, size=days)  # ETH
        df['nft_market_cap'] = np.random.uniform(10, 30, size=days)  # Billion USD
        
        # Add Layer 2 metrics
        df['l2_total_tvl'] = np.random.uniform(5, 10, size=days)  # Billion USD
        df['l2_arbitrum_tvl'] = np.random.uniform(1, 3, size=days)  # Billion USD
        df['l2_optimism_tvl'] = np.random.uniform(0.5, 2, size=days)  # Billion USD
        df['l2_zksync_tvl'] = np.random.uniform(0.2, 1, size=days)  # Billion USD
        df['l2_starknet_tvl'] = np.random.uniform(0.1, 0.5, size=days)  # Billion USD
        
        # Add EIP-1559 metrics
        df['eip1559_base_fee'] = np.random.uniform(10, 50, size=days)  # Gwei
        df['eip1559_priority_fee'] = np.random.uniform(1, 5, size=days)  # Gwei
        df['eip1559_burned'] = np.random.uniform(5000, 10000, size=days)  # ETH per day
        
        return df
    
    # Solana-specific metrics
    def fetch_solana_metrics(self, days=30):
        """
        Fetch specialized Solana metrics
        
        Args:
            days: Number of days of data to fetch
        
        Returns:
            DataFrame with Solana-specific metrics
        """
        # Create cache key
        cache_key = f"solana_metrics_{days}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Initialize metrics dataframe
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        df = pd.DataFrame({'timestamp': dates})
        
        # Fetch metrics from various sources
        try:
            # Solana network stats (Solana Beach API)
            network_stats = self._fetch_solana_network_stats()
            if not network_stats.empty:
                df = pd.merge(df, network_stats, on='timestamp', how='left')
            
            # Solana DeFi metrics (Step Finance API)
            defi_stats = self._fetch_solana_defi_stats()
            if not defi_stats.empty:
                df = pd.merge(df, defi_stats, on='timestamp', how='left')
            
            # Add simulated metrics if real data is missing
            if len(df.columns) < 5:  # If we have less than 5 metrics, add simulated ones
                logger.warning("Limited real Solana metrics available. Adding simulated data.")
                df = self._simulate_solana_metrics(df, days)
            
            # Fill missing values
            df = df.ffill().bfill()
            
            # Add to cache
            self._add_to_cache(cache_key, df)
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching Solana metrics: {e}")
            return self._simulate_solana_metrics(pd.DataFrame({'timestamp': dates}), days)
    
    def _fetch_solana_network_stats(self):
        """Fetch Solana network stats"""
        try:
            url = "https://api.solanabeach.io/v1/network-stats"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            # Create dataframe with today's date
            df = pd.DataFrame([{
                'timestamp': datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
                'sol_tps': data['tps']['current'],
                'sol_validators': data['validators']['total'],
                'sol_stake_percentage': data['stake']['percentage'],
                'sol_epoch': data['epoch']['current'],
                'sol_slot': data['slot']['current']
            }])
            
            return df
        except Exception as e:
            logger.error(f"Error fetching Solana network stats: {e}")
            return pd.DataFrame()
    
    def _fetch_solana_defi_stats(self):
        """Fetch Solana DeFi stats"""
        try:
            url = "https://api.step.finance/v1/tvl"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            # Create dataframe with today's date
            df = pd.DataFrame([{
                'timestamp': datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
                'sol_defi_tvl': data['total'],
                'sol_serum_tvl': next((p['tvl'] for p in data['projects'] if p['name'] == 'Serum'), 0),
                'sol_raydium_tvl': next((p['tvl'] for p in data['projects'] if p['name'] == 'Raydium'), 0),
                'sol_saber_tvl': next((p['tvl'] for p in data['projects'] if p['name'] == 'Saber'), 0)
            }])
            
            return df
        except Exception as e:
            logger.error(f"Error fetching Solana DeFi stats: {e}")
            return pd.DataFrame()
    
    def _simulate_solana_metrics(self, df, days):
        """Simulate Solana-specific metrics"""
        # Add network metrics
        df['sol_tps'] = np.random.uniform(2000, 4000, size=days)
        df['sol_validators'] = np.random.randint(1000, 1500, size=days)
        df['sol_stake_percentage'] = np.random.uniform(0.7, 0.8, size=days)
        df['sol_epoch'] = np.random.randint(200, 300, size=days)
        df['sol_slot'] = np.random.randint(100000000, 150000000, size=days)
        
        # Add DeFi metrics
        df['sol_defi_tvl'] = np.random.uniform(5, 10, size=days)  # Billion USD
        df['sol_serum_tvl'] = np.random.uniform(0.5, 1.5, size=days)  # Billion USD
        df['sol_raydium_tvl'] = np.random.uniform(0.5, 1.5, size=days)  # Billion USD
        df['sol_saber_tvl'] = np.random.uniform(0.3, 1.0, size=days)  # Billion USD
        
        # Add NFT metrics
        df['sol_nft_volume'] = np.random.uniform(10, 50, size=days)  # Million USD
        df['sol_nft_sales'] = np.random.randint(5000, 20000, size=days)
        
        # Add token metrics
        df['sol_active_addresses'] = np.random.randint(50000, 200000, size=days)
        df['sol_new_addresses'] = np.random.randint(1000, 5000, size=days)
        df['sol_transactions'] = np.random.randint(20000000, 50000000, size=days)
        
        return df
    
    # Fetch metrics for any cryptocurrency
    def fetch_metrics_for_symbol(self, symbol, days=30):
        """
        Fetch specialized metrics for a specific cryptocurrency
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USD')
            days: Number of days of data to fetch
        
        Returns:
            DataFrame with cryptocurrency-specific metrics
        """
        # Extract base currency
        base_currency = symbol.split('/')[0]
        
        # Fetch metrics based on currency
        if base_currency in ['BTC', 'XBT']:
            return self.fetch_bitcoin_metrics(days)
        elif base_currency == 'ETH':
            return self.fetch_ethereum_metrics(days)
        elif base_currency == 'SOL':
            return self.fetch_solana_metrics(days)
        else:
            # For other currencies, fetch generic metrics
            return self._fetch_generic_metrics(base_currency, days)
    
    def _fetch_generic_metrics(self, currency, days=30):
        """Fetch generic metrics for other cryptocurrencies"""
        # Create cache key
        cache_key = f"generic_metrics_{currency}_{days}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Initialize metrics dataframe
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        df = pd.DataFrame({'timestamp': dates})
        
        # Try to fetch from Messari API
        try:
            if self.messari_api_key:
                messari_metrics = self._fetch_messari_metrics(currency.lower())
                if not messari_metrics.empty:
                    df = pd.merge(df, messari_metrics, on='timestamp', how='left')
            
            # Try to fetch from IntoTheBlock API
            if self.intotheblock_api_key:
                itb_metrics = self._fetch_intotheblock_metrics(currency.lower())
                if not itb_metrics.empty:
                    df = pd.merge(df, itb_metrics, on='timestamp', how='left')
            
            # Add simulated metrics if real data is missing
            if len(df.columns) < 5:  # If we have less than 5 metrics, add simulated ones
                logger.warning(f"Limited real metrics available for {currency}. Adding simulated data.")
                df = self._simulate_generic_metrics(currency, df, days)
            
            # Fill missing values
            df = df.ffill().bfill()
            
            # Add to cache
            self._add_to_cache(cache_key, df)
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching generic metrics for {currency}: {e}")
            return self._simulate_generic_metrics(currency, pd.DataFrame({'timestamp': dates}), days)
    
    def _fetch_messari_metrics(self, currency):
        """Fetch metrics from Messari API"""
        try:
            url = f"{self.messari_base_url}/assets/{currency}/metrics"
            headers = {
                'x-messari-api-key': self.messari_api_key
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            metrics = data['data']
            
            # Create dataframe with today's date
            df = pd.DataFrame([{
                'timestamp': datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
                'active_addresses': metrics['addresses']['active_count'],
                'supply_pct_issued': metrics['supply']['percent_issued'],
                'market_cap_dominance': metrics['marketcap']['marketcap_dominance_percent'],
                'volatility_30d': metrics['volatility_stats']['volatility_30d'],
                'sharpe_ratio': metrics['risk_metrics']['sharpe_ratio'],
                'realized_cap': metrics['marketcap']['realized_marketcap_usd']
            }])
            
            return df
        except Exception as e:
            logger.error(f"Error fetching Messari metrics for {currency}: {e}")
            return pd.DataFrame()
    
    def _fetch_intotheblock_metrics(self, currency):
        """Fetch metrics from IntoTheBlock API"""
        try:
            url = f"{self.intotheblock_base_url}/summary/{currency}"
            headers = {
                'x-api-key': self.intotheblock_api_key
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            # Create dataframe with today's date
            df = pd.DataFrame([{
                'timestamp': datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
                'itb_concentration': data['concentration'],
                'itb_in_the_money': data['inOutMoney']['inTheMoney'],
                'itb_large_txs': data['largeTransactions']['largeTxs'],
                'itb_east_west_flow': data['eastWestFlow']['netFlow'],
                'itb_addresses_change': data['addresses']['change']
            }])
            
            return df
        except Exception as e:
            logger.error(f"Error fetching IntoTheBlock metrics for {currency}: {e}")
            return pd.DataFrame()
    
    def _simulate_generic_metrics(self, currency, df, days):
        """Simulate generic metrics for other cryptocurrencies"""
        # Add address metrics
        df['active_addresses'] = np.random.randint(10000, 100000, size=days)
        df['new_addresses'] = np.random.randint(1000, 10000, size=days)
        
        # Add transaction metrics
        df['transaction_count'] = np.random.randint(5000, 50000, size=days)
        df['avg_transaction_value'] = np.random.uniform(100, 1000, size=days)
        df['large_transactions'] = np.random.randint(100, 1000, size=days)
        
        # Add market metrics
        df['market_cap_dominance'] = np.random.uniform(0.01, 0.1, size=days)
        df['volatility_30d'] = np.random.uniform(0.05, 0.2, size=days)
        df['sharpe_ratio'] = np.random.uniform(-0.5, 2.0, size=days)
        
        # Add on-chain metrics
        df['supply_pct_issued'] = np.random.uniform(0.5, 1.0, size=days)
        df['realized_cap'] = np.random.uniform(1, 10, size=days)  # Billion USD
        
        # Add IntoTheBlock-style metrics
        df['itb_concentration'] = np.random.uniform(0.3, 0.7, size=days)
        df['itb_in_the_money'] = np.random.uniform(0.4, 0.8, size=days)
        df['itb_large_txs'] = np.random.randint(100, 500, size=days)
        df['itb_east_west_flow'] = np.random.normal(0, 1000000, size=days)
        df['itb_addresses_change'] = np.random.uniform(-0.05, 0.1, size=days)
        
        return df
