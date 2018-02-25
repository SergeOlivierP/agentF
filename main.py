from agent import Agent
from portfolio import Portfolio
from environment import Environment
from market import Market
# from math import floor
# from random import randint
import numpy as np


# Model hyperparameters

num_iterations = 500000
market = Market('IntelDataSet.csv')
signals = market.signals.shape[0]
market_size = market.asset_prices.shape[0]
agent = Agent(input_dim=signals+market_size, output_dim=market_size, learning_rate=1e-4)
running_reward = []


# Output for further analysis, should also output model parameters
# output = open("{}".format(datetime.now().strftime('%Y/%m/%d %H:%M:%S')), "w")

for j in range(num_iterations):

    session = Environment(portfolio=Portfolio(cash=5000, market_size=market_size),
                          agent=agent,
                          market=market,
                          start_day=2,
                          duration=100,
                          )
    agent, asset = session.run()

    running_reward.append(asset)

    if j % 100 == 0:
        mean = np.mean(running_reward)
        std = np.std(running_reward)
        running_reward = []
        print("Average cumulated asset value (round {}):\nmean: {:10.2f}\nstd: {:10.2f}".format(j, mean, std))
        # output.write("{:10.2f}\n".format(mean))

# output.close()
