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

    def agent_decide(market, portfolio):
        # The model outputs a new set of weights from the state of the day
        decision = None
        return decision  # weights

    def run_transaction(decision, portfolio):
        # append new state to portfolio from the decision of the agent
        portfolio_updated = None
        return portfolio_updated

    def compute_reward(portfolio_updated, market):
        reward = None
        return reward

    def agent_update(portfolio_updated, reward):
        pass
