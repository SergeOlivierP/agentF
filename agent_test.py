from agent import Agent
import numpy as np
import unittest


class AgentTest(unittest.TestCase):

    def setUp(self):
        self.network = Agent(input_dim=10, output_dim=4)

    def test_agent_decide(self):
        data = np.random.random((10))
        action = self.network.decide(data)
        self.assertEqual(action.shape, (1, 4))


if __name__ == '__main__':
    unittest.main()
