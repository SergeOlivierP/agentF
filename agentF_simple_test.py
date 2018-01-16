import unittest
from agent import Agent
from market import Market
from simulation import Simulation

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
            self.assertEqual(agent.stock, 10)
            self.assertEqual(agent.cash, 0)

    def test_agent_sells_no_stock(self):
        agent = Agent(10, 0)
        with self.assertRaises(ValueError):
            agent.transaction("sell", 10)
            self.assertEqual(agent.stock, 0)
            self.assertEqual(agent.cash, 10)

if __name__ == '__main__':
    unittest.main()
