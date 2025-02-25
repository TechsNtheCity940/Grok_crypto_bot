import os
from dotenv import load_dotenv

load_dotenv()

KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY')
KRAKEN_API_SECRET = os.getenv('KRAKEN_API_SECRET')
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
MEM0_API_KEY = os.getenv('MEM0_API_KEY')

TRADING_PAIRS = ['DOGE/USD', 'SHIB/USD', 'XRP/USD']  # Added SHIB, XRP
ACTIVE_EXCHANGE = 'kraken'