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

    def agent_decide(market, portfolio):
        # The model outputs a new set of weights from the state of the day
        decision = None
        return decision  # weights

    def run_transaction(decision, portfolio):
        # append new state to portfolio from the decision of the agent
        portfolio_updated = None
        return portfolio_updated

    def compute_reward(portfolio, asset_returns):
        # parameter for the geometric means
        geo_parameter = .9
        # save the running geometric mean of returns
        self.portfolio.geo_mean_returns = np.append(self.portfolio.geo_mean_returns,
                                             geo_parameter*self.portfolio.returns[-1] + (1-geo_parameter)*self.portfolio.geo_mean_returns[-1])
        # save the running geometric mean of returns squared
        self.portfolio.geo_mean2_returns = np.append(self.portfolio.geo_mean2_returns,
                                             geo_parameter*self.portfolio.returns[-1]**2 + (1-geo_parameter)*self.portfolio.geo_mean2_returns[-1])
        
        self.portfolio.differential_sharpe = np.append(self.portfolio.differential_sharpe,
                                            ((self.portfolio.returns[-1] - self.portfolio.geo_mean_returns[-2])*self.portfolio.geo_mean2_returns[-2]
                                            - 0.5*self.portfolio.geo_mean_returns[-2]*(self.portfolio.returns[-1]**2 - self.portfolio.geo_mean2_returns[-2]))
                                            / (((self.portfolio.geo_mean2_returns[-2]) - self.portfolio.geo_mean_returns[-2]**2)**(3/2)) )
        self.portfolio.differential_sharpe_derivative = np.append(self.portfolio.differential_sharpe_derivative,
                                            (self.portfolio.geo_mean2_returns[-2]-(self.portfolio.geo_mean_returns[-2])*self.portfolio.returns[-1])
                                             / (((self.portfolio.geo_mean2_returns[-2]) - self.portfolio.geo_mean_returns[-2]**2)**(3/2)) )
                                         
        reward = (self.portfolio.weights[-1] + reward_learning_rate*self.portfolio.differential_sharpe_derivative[-1]*asset_returns )

        reward = np.exp(reward)/np.sum(np.exp(reward))

        return reward

    def agent_update(portfolio_updated, reward):
        pass
