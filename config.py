import os
from dotenv import load_dotenv

load_dotenv()

# Kraken API credentials
KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY')
KRAKEN_API_SECRET = os.getenv('KRAKEN_API_SECRET')

# Coinbase API credentials
COINBASE_API_KEY = os.getenv('COINBASE_API_KEY')
COINBASE_API_SECRET = os.getenv('COINBASE_API_SECRET')

# Trading pair (BTC/USD is supported by both)
TRADING_PAIR = 'BTC/USD'

# Choose exchange (toggle between 'kraken' and 'coinbase')
ACTIVE_EXCHANGE = 'kraken'  # Change to 'coinbase' as needed