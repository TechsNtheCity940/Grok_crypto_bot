import logging
import numpy as np

logger = logging.getLogger(__name__)

class GridTrader:
    def __init__(self, config):
        self.config = config
        self.grid_config = config.get('grid_trading', {})
        self.num_grids = self.grid_config.get('num_grids', 10)
        self.grid_spread = self.grid_config.get('grid_spread', 0.05)  # 5% spread
        self.max_position = self.grid_config.get('max_position', 1.0)
        self.min_profit = self.grid_config.get('min_profit', 0.2)
        self.grids = []

    def setup_grids(self, current_price, price_range=10.0):
        lower_bound = current_price * (1 - price_range / 100)
        upper_bound = current_price * (1 + price_range / 100)
        grid_prices = np.linspace(lower_bound, upper_bound, self.num_grids)
        self.grids = [{'price': float(p), 'type': 'buy' if p < current_price else 'sell'} for p in grid_prices]
        logger.info(f"Set up {len(self.grids)} grids from {lower_bound:.4f} to {upper_bound:.4f}")

    def get_grid_orders(self, current_price, available_balance):
        orders = []
        position_size = available_balance * self.max_position / self.num_grids
        for grid in self.grids:
            grid_price = grid['price']
            if (grid['type'] == 'buy' and current_price <= grid_price) or (grid['type'] == 'sell' and current_price >= grid_price):
                amount = position_size / current_price if grid['type'] == 'buy' else position_size / grid_price
                orders.append({'type': grid['type'], 'price': grid_price, 'amount': amount})
        return orders

    def update_grids(self, executed_order):
        for grid in self.grids:
            if abs(grid['price'] - executed_order['price']) < 0.01:
                grid['type'] = 'sell' if executed_order['type'] == 'buy' else 'buy'

    def check_arbitrage(self, kraken_price, other_exchange_price):
        spread = (other_exchange_price - kraken_price) / kraken_price
        return spread if spread > 0.01 else 0  # 1% threshold