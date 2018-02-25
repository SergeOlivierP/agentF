import unittest
from portfolio import Portfolio
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


if __name__ == '__main__':
    unittest.main()
