import os
from dotenv import load_dotenv

load_dotenv()

# Kraken API credentials
KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY')
KRAKEN_API_SECRET = os.getenv('KRAKEN_API_SECRET')

# Coinbase API credentials
COINBASE_API_KEY = os.getenv('COINBASE_API_KEY')
COINBASE_API_SECRET = os.getenv('COINBASE_API_SECRET')
MEM0_API_KEY = os.getenv('MEM0_API_KEY')
# Twitter API credentials
TWITTER_CONSUMER_KEY = os.getenv('TWITTER_CONSUMER_KEY')
TWITTER_CONSUMER_SECRET = os.getenv('TWITTER_CONSUMER_SECRET')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')

# Trading pair (BTC/USD is supported by both)
TRADING_PAIRS = ['DOGE/USD']

# Choose exchange (toggle between 'kraken' and 'coinbase')
ACTIVE_EXCHANGE = 'kraken'  # Change to 'coinbase' as needed