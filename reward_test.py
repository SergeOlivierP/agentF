import unittest
from unittest.mock import MagicMock
from portfolio import Portfolio
from environment import Environment
from agent import Agent
import numpy as np


class EnvironmentTest(unittest.TestCase):

    def setUp(self):
        self.portfolio = Portfolio(100, 10)

        self.portfolio.weights = np.array([.1,.9])
        self.agent = MagicMock()
        self.market = MagicMock()
        self.market.asset_prices = np.array([[1,1,1,1],[10, 15, 20, 25]]).T
        self.environment = Environment(self.portfolio, self.agent, self.market)

    def test_compute_reward(self):
        
        reward = self.environment.compute_reward(self.portfolio, self.market)

        self.assertTrue(np.sum(reward)==1)
        self.assertTrue(np.all(reward) > 0)

if __name__ == '__main__':
    unittest.main()
