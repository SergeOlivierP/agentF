import unittest
from unittest.mock import MagicMock
from portfolio import Portfolio
from environment import Environment
from agent import Agent
import numpy as np


class PortfolioTest(unittest.TestCase):

    def test_portfolio_buy_stock(self):
        portfolio = Portfolio(100, 10)
        portfolio.transaction(1, 10)
        self.assertEqual(portfolio.stock, 11)
        self.assertEqual(portfolio.cash, 90)

    def test_portfolio_buy_stock_no_cash(self):
        portfolio = Portfolio(0, 10)
        with self.assertRaises(ValueError):
            portfolio.transaction(1, 10)

    def test_portfolio_sells_no_stock(self):
        portfolio = Portfolio(10, 0)
        with self.assertRaises(ValueError):
            portfolio.transaction(0, 10)

    def test_portfolio_state_updates(self):
        portfolio = Portfolio(100, 0)
        self.assertIn(0, portfolio.state)
        self.assertIn(1, portfolio.state)
        portfolio.transaction(1, 10)
        self.assertIn(1, portfolio.state)
        self.assertIn(1, portfolio.state)


class EnvironmentTest(unittest.TestCase):

    def setUp(self):
        self.portfolio = Portfolio(100, 10)
        self.agent = MagicMock()
        self.market = MagicMock()
        self.market.stock_price = [10, 15, 20, 25]
        self.environment = Environment(self.portfolio, self.agent, self.market)

    def test_run_transaction_rewards_buy(self):
        self.environment.trade_cost = 0.1
        self.environment.end_day = 1

        reward1 = self.environment.run_transaction(y=1,
                                                   stock_price=10,
                                                   future=15,
                                                   )
        reward2 = self.environment.run_transaction(y=1,
                                                   stock_price=10,
                                                   future=12,
                                                   )
        reward3 = self.environment.run_transaction(y=1,
                                                   stock_price=15,
                                                   future=10,
                                                   )
        reward4 = self.environment.run_transaction(y=1,
                                                   stock_price=15,
                                                   future=12,
                                                   )

        self.assertGreater(reward1, 0.5)
        self.assertLess(reward3, 0.5)
        self.assertGreater(reward1, reward2)
        self.assertGreater(reward4, reward3)
        self.assertGreater(reward1, reward3)
        self.assertTrue(reward1 > 0 and reward1 < 1)
        self.assertTrue(reward2 > 0 and reward2 < 1)

    def run_transaction_rewards_sell(self):
        self.environment.trade_cost = 0.1
        self.environment.end_day = 1

        reward1 = self.environment.run_transaction(y=0,
                                                   stock_price=10,
                                                   future=15,
                                                   )
        reward2 = self.environment.run_transaction(y=0,
                                                   stock_price=10,
                                                   future=12,
                                                   )

        reward3 = self.environment.run_transaction(y=0,
                                                   stock_price=15,
                                                   future=10,
                                                   )
        reward4 = self.environment.run_transaction(y=0,
                                                   stock_price=15,
                                                   future=12,
                                                   )

        self.assertGreater(reward3, reward1)
        self.assertGreater(reward3, reward4)
        self.assertGreater(reward2, reward1)
        self.assertTrue(reward1 > 0 and reward2 < 1)
        self.assertTrue(reward1 > 0 and reward2 < 1)

    def test_run_cost_is_prohibitive(self):
        self.environment.trade_cost = 0.01
        self.environment.sanction = 0.1
        self.environment.end_day = 1

        reward1 = self.environment.run_transaction(y=1,
                                                   stock_price=10,
                                                   future=10)
        self.environment.portfolio = Portfolio(100, 0)
        reward2 = self.environment.run_transaction(y=0,
                                                   stock_price=10,
                                                   future=10)

        self.assertLess(reward1, 0.5)
        self.assertLess(reward2, 0.5)
        self.assertLess(reward2, reward1)


class AgentTest(unittest.TestCase):

    def setUp(self):
        self.network = Agent(input_dim=10)

    def test_agent_decide(self):
        data = np.random.random((10))
        action = self.network.decide(data)
        self.assertTrue(action > 0 and action < 1)


if __name__ == '__main__':
    unittest.main()
