import logging
import numpy as np

logger = logging.getLogger(__name__)

class GridTrader:
    def __init__(self, config):
        self.config = config
        self.num_grids = config['grid_trading']['num_grids']
        self.grid_spread = config['grid_trading']['grid_spread']
        self.grids = []

    def setup_grids(self, current_price, price_range=5.0):
        lower_bound = current_price * (1 - price_range / 100)
        upper_bound = current_price * (1 + price_range / 100)
        grid_prices = np.linspace(lower_bound, upper_bound, self.num_grids)
        self.grids = [{'price': float(p), 'type': 'buy' if p < current_price else 'sell'} for p in grid_prices]

    def get_grid_orders(self, current_price, available_balance):
        orders = []
        position_size = available_balance / self.num_grids * 0.2
        for grid in self.grids:
            grid_price = grid['price']
            if (grid['type'] == 'buy' and current_price <= grid_price) or (grid['type'] == 'sell' and current_price >= grid_price):
                orders.append({'type': grid['type'], 'price': grid_price, 'amount': position_size / grid_price})
        return orders

    def update_grids(self, executed_order):
        for grid in self.grids:
            if abs(grid['price'] - executed_order['price']) < 0.01:  # Approximate match
                grid['type'] = 'sell' if executed_order['type'] == 'buy' else 'buy'