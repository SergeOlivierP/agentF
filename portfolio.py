import numpy as np


class Portfolio:

    def __init__(self, cash, market_size, geo_parameter=0.9):

        self.geo_parameter = geo_parameter

        # (T,N) matrices where t in [0,T] is the day and n in [0,N] is the asset.
        self.weights = np.zeros((1, market_size))
        self.weights[0][0] = 1
        self.quantities = np.zeros((1, market_size))
        self.quantities[0][0] = cash

        # (T,m) stats to compute sharpe ratio in the reward function
        self.returns = [0]
        self.geo_mean_returns = [0]
        self.geo_mean2_returns = [0]
        self.differential_sharpe = [0]
        self.differential_sharpe_derivative = [0]

    def process_sharpe_ratio(self):
        past_returns = self.returns[-1]
        # save the running geometric mean of returns
        new_geo_mean_returns = self.geo_parameter*past_returns + (1-self.geo_parameter)*self.geo_mean_returns[-1]
        self.geo_mean_returns = np.append(self.geo_mean_returns, new_geo_mean_returns)

        # save the running geometric mean of returns squared
        new_geo_mean2_returns = self.geo_parameter*past_returns**2+(1-self.geo_parameter)*self.geo_mean2_returns[-1]
        self.geo_mean2_returns = np.append(self.geo_mean2_returns, new_geo_mean2_returns)

        new_differential_sharpe = (((past_returns-self.geo_mean_returns[-2])*self.geo_mean2_returns[-2]
                                    - 0.5*self.geo_mean_returns[-2]*(past_returns**2 - self.geo_mean2_returns[-2]))
                                   / (((self.geo_mean2_returns[-2]) - self.geo_mean_returns[-2]**2)**(3/2)))
        self.differential_sharpe = np.append(self.differential_sharpe, new_differential_sharpe)

        new_sharpe_derivative = ((self.geo_mean2_returns[-2]-(self.geo_mean_returns[-2])*past_returns)
                                 / (((self.geo_mean2_returns[-2]) - self.geo_mean_returns[-2]**2)**(3/2)))
        self.differential_sharpe_derivative = np.append(self.differential_sharpe_derivative, new_sharpe_derivative)

        return self.differential_sharpe_derivative[-1]

    def get_total_value(self, prices):
        return np.sum(self.quantities[-1][:]*prices)

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
