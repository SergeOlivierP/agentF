import numpy as np


class Environment:

    def __init__(self, portfolio, agent, market,
                 start_day=0,
                 trade_cost=0.002,
                 sanction=0.05,
                 duration=500):
        self.portfolio = portfolio
        self.agent = agent
        self.market = market
        self.init_cash = self.portfolio.cash
        self.start_day = start_day
        self.end_day = self.start_day + duration
        self.trade_cost = trade_cost
        self.sanction = sanction

    def run_transaction(self, y, stock_price, future):

        try:
            self.portfolio.transaction(y, stock_price)
            cost = self.trade_cost
        except ValueError:
            cost = self.sanction

        profit = future - cost - stock_price

        return 1 / (1 + np.exp(-profit))

    def run(self):

        for i in range(self.start_day, self.end_day-1):

            state = np.concatenate((self.market.indices[i], self.portfolio.state), axis=0)
            action_prob = self.agent.decide(state)

            if np.random.uniform() < action_prob:
                # transaction: "buy"
                y = 1
            else:
                # transaction: "sell"
                y = 0

            y_hat = self.run_transaction(y, self.market.stock_price[i], self.market.stock_price[i+1])

            self.agent.train(state, y_hat, i)

        asset = self.portfolio.cash+self.portfolio.stock*self.market.stock_price[i]
        return self.agent, asset