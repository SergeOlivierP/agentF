import numpy as np


class Session:

    DURATION = 20

    def __init__(self, agent, policy, market, start_day=0, trade_cost=0.002, sanction=0.05):
        self.agent = agent
        self.policy = policy
        self.market = market
        self.init_cash = self.agent.cash
        self.start_day = start_day
        self.end_day = self.start_day + self.DURATION
        self.trade_cost = trade_cost
        self.sanction = sanction

    def run_transaction(self, action, stock_price):
        try:
            self.agent.transaction(action, stock_price)
            cost = self.trade_cost
        except ValueError:
            cost = self.sanction

        transaction_cost = cost
        assets = self.agent.cash + self.agent.stock*stock_price
        assets -= transaction_cost
        reward = ((assets - self.init_cash) / self.init_cash)

        return reward, assets

    def compute_cost_function(self, y, action_prob, reward):
        return (y - action_prob) * reward

    def run(self):

        for i in range(self.start_day, self.end_day):

            state = np.concatenate((self.market.indices[i], self.agent.state), axis=0)
            action_prob = self.policy.decide(state)

            if np.random.uniform() < action_prob:
                action = "buy"
                y = 1
            else:
                action = "sell"
                y = 0

            reward, assets = self.run_transaction(action, self.market.stock_price[i])
            y_hat = self.compute_cost_function(y, action_prob, reward)

            self.policy.train(state, y_hat, i)

        return self.policy, assets
