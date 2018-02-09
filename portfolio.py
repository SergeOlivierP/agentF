import numpy as np


class Portfolio:

    def __init__(self, c, q):
        self.cash = c
        self.stock = q
        self.state = self.compute_state(self.cash, self.stock)

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
        self.state = self.compute_state(self.cash, self.stock)

    def compute_state(self, cash, stock):
        got_money = 1 if cash > 0 else 0
        got_stock = 1 if stock > 0 else 0
        return np.array([got_money, got_stock])
