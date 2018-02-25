import unittest
from unittest.mock import MagicMock
from portfolio import Portfolio
# from environment import Environment
# from agent import Agent
import numpy as np


class PortfolioTest(unittest.TestCase):

    def setUp(self):
        self.money = 10
        self.assets = 5
        self.pf = Portfolio(cash=self.money, market_size=self.assets)

    def test_portoflio_init(self):

        self.assertEqual(self.pf.weights.shape, (1, self.assets))
        self.assertEqual(self.pf.weights[0][0], 1)
        self.assertEqual(self.pf.weights.shape, self.pf.quantities.shape)
        self.assertEqual(self.pf.quantities[0][0], self.money)

    def test_portfolio_update_respect_shape(self):
        target = np.array([[0.20, 0.20, 0.20, 0.20, 0.20]])
        prices = np.array([[1, 1, 1, 1, 1]])
        prev_shape = self.pf.weights.shape
        self.pf.update_transaction(target, prices)

        self.assertEqual(self.pf.weights.shape, (prev_shape[0]+1, prev_shape[1]))

    def test_portfolio_update_respect_weights(self):
        target = np.array([[0.20, 0.20, 0.20, 0.20, 0.20]])
        prices = np.array([[1, 1, 1, 1, 1]])
        self.pf.update_transaction(target, prices)
        self.assertEqual(np.sum(self.pf.weights[-1][:]), 1)

    def test_portfolio_update_respect_quantities(self):
        target = np.array([[0.20, 0.20, 0.20, 0.20, 0.20]])
        prices = np.array([[1, 1, 1, 1, 1]])
        self.pf.update_transaction(target, prices)
        self.assertEqual(self.pf.quantities[-1][1], self.pf.quantities[-1][2])

    def test_portfolio_update_works(self):
        pf = Portfolio(cash=10, market_size=3)
        target = np.array([[0.2, 0.4, 0.4]])
        prices = np.array([[1, 2, 3]])
        pf.update_transaction(target, prices)
        self.assertTrue((pf.quantities == np.array([[10, 0, 0], [3, 2, 1]])).all())


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
