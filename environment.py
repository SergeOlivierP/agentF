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
        for i in range(1, self.end_day):
            state = np.concatenate(self.market.signals[i][:], self.portfolio.weights[i][:])
            decision = self.agent.decide(state)
            self.portfolio.update_transaction(decision, self.market.prices[i][:])
            
            asset_returns = self.market.prices[i+1]/self.market.prices[i] - 1
            self.portfolio.set_portfolio_return(asset_returns)
            sharpe_derivative = self.portfolio.process_sharpe_ratio()
            reward = self.compute_reward(i, asset_returns, sharpe_derivative)
            self.agent.train(decision, reward)

        asset = self.portfolio.get_total_value(self.market.prices[i][:])
        return self.policy, asset

    def compute_reward(self, i, asset_returns, sharpe_derivative, reward_learning_rate=0.01):
        reward = (self.portfolio.weights[-1] + reward_learning_rate*sharpe_derivative*asset_returns)
        return np.exp(reward)/np.sum(np.exp(reward))
