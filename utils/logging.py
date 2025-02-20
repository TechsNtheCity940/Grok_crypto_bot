import logging

def setup_logging():
    logging.basicConfig(
        filename='crypto_trading_bot.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger()

logger = setup_logging()