import unittest
from unittest.mock import MagicMock
from agent import Agent
from market import Market
from session import Session
from policy_keras import Policy
import numpy as np


class AgentTest(unittest.TestCase):

    def test_agent_buy_stock(self):
        agent = Agent(100, 10)
        agent.transaction("buy", 10)
        self.assertEqual(agent.stock, 11)
        self.assertEqual(agent.cash, 90)

    def test_agent_buy_stock_no_cash(self):
        agent = Agent(0, 10)
        with self.assertRaises(ValueError):
            agent.transaction("buy", 10)

    def test_agent_sells_no_stock(self):
        agent = Agent(10, 0)
        with self.assertRaises(ValueError):
            agent.transaction("sell", 10)


class SessionTest(unittest.TestCase):

    def setUp(self):
        self.agent = Agent(10, 0)
        self.policy = MagicMock()
        self.market = MagicMock()
        self.market.stock_price = [10, 10]
        self.session = Session(self.agent, self.policy, self.market)

    def test_run_transaction(self):
        self.session.trade_cost = 0.1
        self.session.end_day = 1

        self.session.agent = Agent(100, 0)
        price1 = 10
        reward1, assets = self.session.run_transaction("buy", price1)

        self.session.agent = Agent(100, 0)
        price2 = 20
        reward2, assets = self.session.run_transaction("buy", price2)

        self.assertEqual(reward2, reward1)


class PolicyTest(unittest.TestCase):

    def setUp(self):
        self.network = Policy(input_dim=10)

    def test_policy_decide(self):
        data = np.random.random((10))
        action = self.network.decide(data)
        self.assertEqual(len(action), 1)



if __name__ == '__main__':
    unittest.main()
