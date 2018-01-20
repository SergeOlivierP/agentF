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

    def run_transaction(self, action, stock_price, future_stock_price):
        try:
            self.agent.transaction(action, stock_price)
            cost = self.trade_cost
        except ValueError:
            cost = self.sanction

        transaction_cost = cost
        reward = ((future_stock_price - transaction_cost - stock_price) / stock_price)

        if action == "sell":
            reward = -reward

        return reward

    def compute_cost_function(self, y, action_prob, reward):
        return action_prob + (y - action_prob) * reward

    def run(self):

        for i in range(self.start_day, self.end_day-1):

            state = np.concatenate((self.market.indices[i], self.agent.state), axis=0)
            action_prob = self.policy.decide(state)

            if np.random.uniform() < action_prob:
                action = "buy"
                y = 1
            else:
                action = "sell"
                y = 0

            reward = self.run_transaction(action, self.market.stock_price[i], self.market.stock_price[i+1])
            y_hat = self.compute_cost_function(y, action_prob, reward)

            self.policy.train(state, y_hat, i)

        asset = self.agent.cash+self.agent.stock*self.market.stock_price[-1]
        return self.policy, asset
