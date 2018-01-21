import unittest
from unittest.mock import MagicMock
from agent import Agent
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
        self.agent = Agent(100, 10)
        self.policy = MagicMock()
        self.market = MagicMock()
        self.market.stock_price = [10, 15, 20, 25]
        self.session = Session(self.agent, self.policy, self.market)

    def test_run_transaction_rewards_buy(self):
        self.session.trade_cost = 0.1
        self.session.end_day = 1

        reward1 = self.session.run_transaction("buy", 10, 15)
        reward2 = self.session.run_transaction("buy", 10, 20)

        self.assertGreater(reward2, reward1)
        self.assertTrue(reward1 > 0 and reward1 < 1)
        self.assertTrue(reward2 > 0 and reward2 < 1)

    def run_transaction_rewards_sell(self):
        self.session.trade_cost = 0.1
        self.session.end_day = 1

        reward3 = self.session.run_transaction("sell", 15, 10)
        reward4 = self.session.run_transaction("sell", 15, 12)

        self.assertGreater(reward3, reward4)
        self.assertTrue(reward3 > 0 and reward3 < 1)
        self.assertTrue(reward4 > 0 and reward4 < 1)

    def test_run_cost_is_prohibitive(self):
        self.session.trade_cost = 0.01
        self.session.sanction = 0.1
        self.session.end_day = 1

        reward1 = self.session.run_transaction("buy", 10, 10)
        self.session.agent = Agent(100, 0)
        reward2 = self.session.run_transaction("sell", 10, 10)

        print(reward1)
        print(reward2)
        self.assertLess(reward1, 0)
        self.assertLess(reward2, 0)
        self.assertLess(reward2, reward1)
        self.assertTrue(False)


class PolicyTest(unittest.TestCase):

    def setUp(self):
        self.network = Policy(input_dim=10)

    def test_policy_decide(self):
        data = np.random.random((10))
        action = self.network.decide(data)
        self.assertGreater(action, 0)
        self.assertLess(action, 1)


if __name__ == '__main__':
    unittest.main()
