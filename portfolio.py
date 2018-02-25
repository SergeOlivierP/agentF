import numpy as np


class Portfolio:

    def __init__(self, init_cash, market_size):
        # (T,N) matrices where t in [0,T] is the day and n in [0,N] is the asset.
        self.weights = np.zeros((1, market_size))
        self.weights[0] = 1
        self.quantity = np.zeros((1, market_size))
        self.quantity[0] = init_cash

        # (T,m) index to compute sharpe ratio in the reward function
        self.sharpe_cache = None

    def adjust_quantities(target, prices):
        # append a new row to the weights and quantity as close as the target as possible,
        # adjusted by the transaction cost.
        pass
