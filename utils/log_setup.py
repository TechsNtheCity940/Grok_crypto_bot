import logging
import time

def setup_logging():
    logging.basicConfig(
        filename='crypto_trading_bot.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    try:
        new_data = fetch_real_time_data()
    except TimeoutError as e:
        logging.error(f"Data fetch failed: {e}")
        time.sleep(10)  # Wait before retrying
        return logging.getLogger()


logger = setup_logging()