import os
import json
from dotenv import load_dotenv

class ConfigManager:
    """
    Enhanced configuration management system with feature flags
    to enable/disable new features without breaking existing functionality.
    """
    _instance = None
    _config = None

    @classmethod
    def get_instance(cls):
        """Singleton pattern to ensure only one config instance exists"""
        if cls._instance is None:
            cls._instance = ConfigManager()
        return cls._instance

    def __init__(self):
        """Initialize with default configuration and load from file if available"""
        # Load environment variables
        load_dotenv()
        
        # Default configuration with feature flags
        self._config = {
            # Exchange settings
            'active_exchange': os.getenv('ACTIVE_EXCHANGE', 'kraken'),
            'trading_pairs': ['DOGE/USD', 'SHIB/USD', 'XRP/USD'],
            
            # API Keys (from environment)
            'kraken_api_key': os.getenv('KRAKEN_API_KEY', ''),
            'kraken_api_secret': os.getenv('KRAKEN_API_SECRET', ''),
            'reddit_client_id': os.getenv('REDDIT_CLIENT_ID', ''),
            'reddit_client_secret': os.getenv('REDDIT_CLIENT_SECRET', ''),
            'mem0_api_key': os.getenv('MEM0_API_KEY', ''),
            
            # Feature flags for new enhancements
            'use_enhanced_data': False,  # Enable multi-exchange and on-chain data
            'advanced_sentiment': False,  # Enable advanced sentiment analysis
            'advanced_risk': False,       # Enable advanced risk management
            'use_transformer_model': False, # Enable transformer model
            'use_ensemble_weighting': False, # Enable dynamic ensemble weighting
            'use_strategy_selector': False,  # Enable adaptive strategy selection
            'use_performance_tracking': False, # Enable advanced performance metrics
            
            # Model settings
            'model_types': ['hybrid', 'lstm'],  # Default to existing models
            'sequence_length': 50,
            'n_features': 9,
            
            # Training settings
            'batch_size': 32,
            'epochs': 50,
            'train_test_split': 0.8,
            
            # Trading settings
            'max_position_size': 0.2,  # Max 20% of portfolio in one position
            'min_trade_sizes': {'DOGE/USD': 10.0, 'SHIB/USD': 1000000.0, 'XRP/USD': 1.0},
            
            # Risk management
            'max_loss': 0.5,  # 50% max drawdown
            'use_kelly_criterion': False,
            'kelly_fraction': 0.5,  # Half-Kelly for more conservative sizing
            
            # Grid trading settings
            'grid_trading': {
                'num_grids': 10,
                'grid_spread': 0.05,
                'max_position': 1.0,
                'min_profit': 0.2
            },
            
            # Paths
            'model_dir': 'models/trained_models',
            'data_dir': 'data/historical',
            
            # Operational settings
            'retrain_interval_hours': 24,
            'update_interval_hours': 1,
            'log_level': 'INFO'
        }
        
        # Try to load from config file if it exists
        self._load_from_file()

    def _load_from_file(self, filepath='config.json'):
        """Load configuration from JSON file if it exists"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    file_config = json.load(f)
                    # Update default config with file values
                    self._config.update(file_config)
                print(f"Loaded configuration from {filepath}")
        except Exception as e:
            print(f"Error loading configuration from {filepath}: {e}")

    def save_to_file(self, filepath='config.json'):
        """Save current configuration to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self._config, f, indent=4)
            print(f"Saved configuration to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving configuration to {filepath}: {e}")
            return False

    def get(self, key, default=None):
        """Get configuration value by key with optional default"""
        return self._config.get(key, default)

    def set(self, key, value):
        """Set configuration value"""
        self._config[key] = value
        return True

    def update(self, config_dict):
        """Update multiple configuration values at once"""
        self._config.update(config_dict)
        return True

    @property
    def all(self):
        """Get entire configuration dictionary"""
        return self._config.copy()  # Return a copy to prevent direct modification

# Create a global instance for easy import
config = ConfigManager.get_instance()
