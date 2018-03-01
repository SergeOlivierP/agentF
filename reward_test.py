import unittest
from unittest.mock import MagicMock
from portfolio import Portfolio
from environment import Environment
import numpy as np


class EnvironmentTest(unittest.TestCase):

    def setUp(self):
        self.portfolio = Portfolio(100, 2)

        self.portfolio.weights = np.array([0.1, 0.9])
        self.agent = MagicMock()
        self.market = MagicMock()
        self.market.prices = np.array([[1, 1, 1, 1], [10, 15, 20, 25]]).T
        self.environment = Environment(self.portfolio, self.agent, self.market)

    def test_compute_reward(self):

        reward = self.environment.compute_reward(1, sharpe_derivative=0.1)

        self.assertTrue(np.sum(reward) == 1)
        self.assertTrue(np.all(reward) > 0)


if __name__ == '__main__':
    unittest.main()
