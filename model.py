from agent import Agent
from portfolio import Portfolio
from environment import Environment
from market import Market
from math import floor
from random import randint
import numpy as np


# Model hyperparameters

num_iterations = 500000
market = Market('IntelDataSet.csv')
D = np.shape(market.indices)[1]+2
agent = Agent(input_dim=D, learning_rate=1e-4)
running_reward = []


# Output for further analysis, should also output model parameters
# output = open("{}".format(datetime.now().strftime('%Y/%m/%d %H:%M:%S')), "w")

for j in range(num_iterations):

    session = Environment(portfolio=Portfolio(c=5000, q=0),
                          agent=agent,
                          market=market,
                          start_day=randint(0, floor(market.duration/2)),
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
