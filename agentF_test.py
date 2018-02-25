import unittest
from unittest.mock import MagicMock
from portfolio import Portfolio
from environment import Environment
from agent import Agent
import numpy as np


class PortfolioTest(unittest.TestCase):



# class EnvironmentTest(unittest.TestCase):
#
#     def setUp(self):
#         self.portfolio = Portfolio(100, 10)
#         self.agent = MagicMock()
#         self.market = MagicMock()
#         self.market.stock_price = [10, 15, 20, 25]
#         self.environment = Environment(self.portfolio, self.agent, self.market)



# 
# class AgentTest(unittest.TestCase):
#
#     def setUp(self):
#         self.network = Agent(input_dim=10)
#
#     def test_agent_decide(self):
#         data = np.random.random((10))
#         action = self.network.decide(data)
#         self.assertTrue(action > 0 and action < 1)


if __name__ == '__main__':
    unittest.main()
