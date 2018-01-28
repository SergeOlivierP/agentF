import numpy as np


class Agent:

    def __init__(self, c, q):
        self.cash = c
        self.stock = q
        self.state = np.array([self.cash, self.stock])

    def transaction(self, action, stock_price):
        if action == 1:
            # action: "buy"
            if stock_price <= self.cash:
                self.cash -= stock_price
                self.stock += 1
            else:
                raise ValueError('Out of funds')
        elif action == 0:
            # action: "sell"
            if self.stock > 0:
                self.cash += stock_price
                self.stock -= 1
            else:
                raise ValueError('Out of stock')
        else:
            pass
