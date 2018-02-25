import numpy as np


class Portfolio:

    def __init__(self, cash, market_size):
        # (T,N) matrices where t in [0,T] is the day and n in [0,N] is the asset.
        self.weights = np.zeros((1, market_size))
        self.weights[0][0] = 1
        self.quantities = np.zeros((1, market_size))
        self.quantities[0][0] = cash

        # (T,m) stats to compute sharpe ratio in the reward function
        self.geo_mean_returns = 0
        self.geo_mean2_returns = 0
        self.differential_sharpe = 0
        self.differential_sharpe_derivative = 0
        

    def update_sharpe_stat(self, geo_parameter):
        # save the running geometric mean of returns
        self.geo_mean_returns = np.append(self.geo_mean_returns,
                                             geo_parameter*self.returns[-1] + (1-geo_parameter)*self.geo_mean_returns[-1])

        # save the running geometric mean of returns squared
        self.geo_mean2_returns = np.append(self.geo_mean2_returns,
                                             geo_parameter*self.returns[-1]**2 + (1-geo_parameter)*self.geo_mean2_returns[-1])
        
        self.differential_sharpe = np.append(self.differential_sharpe,
                                            ((self.returns[-1] - self.geo_mean_returns[-2])*self.geo_mean2_returns[-2]
                                            - 0.5*self.geo_mean_returns[-2]*(self.returns[-1]**2 - self.geo_mean2_returns[-2]))
                                            / (((self.geo_mean2_returns[-2]) - self.geo_mean_returns[-2]**2)**(3/2)) )
        self.differential_sharpe_derivative = np.append(self.differential_sharpe_derivative,
                                            (self.geo_mean2_returns[-2]-(self.geo_mean_returns[-2])*self.returns[-1])
                                             / (((self.geo_mean2_returns[-2]) - self.geo_mean_returns[-2]**2)**(3/2)) )

    def update_transaction(self, target, prices):
        # append a new row to the weights and quantities as close as the target as possible,
        # adjusted by the transaction cost.

        # Compute total value of assets
        total_budget = np.sum(self.quantities[-1][:]*prices)

        # Compute weights drift, delta and transaction cost
        drifted_weights = self.quantities[-1][:]*prices/total_budget
        delta_weights = target-drifted_weights

        # Compute amount of money worth of each stock this delta allows
        assets_budget = delta_weights*total_budget

        # Round quantities to the floor integer according to allocated money
        delta_quantities = np.floor(assets_budget/prices)

        # adjust quantities accordingly, transaction costs will be here
        new_quantities = self.quantities+delta_quantities
        self.quantities = np.append(self.quantities, new_quantities, axis=0)

        # sum unalocated money
        unalocated_money = total_budget-np.sum(new_quantities*prices)
        self.quantities[-1][0] = self.quantities[-1][0] + unalocated_money

        # Compute new real weights
        assets_values = self.quantities[-1][:]*prices
        new_weights = assets_values/np.sum(assets_values)
        self.weights = np.append(self.weights, new_weights, axis=0)
