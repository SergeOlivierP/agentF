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
        self.start_day = start_day
        self.end_day = self.start_day + duration
        self.trade_cost = trade_cost
        self.sanction = sanction

    def run(self):
        for i in range(self.end_day):
            state = np.concatenat(self.market.signals[i][:], self.portfolio.weights[i][:])
            decision = self.agent.decide(state)
            self.portfolio.update_transaction(decision, self.market.prices[i][:])
            reward = self.compute_reward(self.portfolio, self.market)
            self.agent.train(decision, reward)

        asset = self.portfolio.get_total_value(self.market.prices[i][:])
        return self.policy, asset

    def compute_reward(portfolio_updated, market):
        reward = None
        return reward
