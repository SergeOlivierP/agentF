from policy_keras import Policy
from agent import Agent
from session import Session
from market import Market
from datetime import datetime
from random import randint
from math import floor
import numpy as np


# Model hyperparameters

num_iterations = 1
market = Market('IntelDataSet.csv')
D = np.shape(market.indices)[1]+2
policy = Policy(D)
running_reward = []

# Output for further analysis, should also output model parameters
# output = open("{}".format(datetime.now().strftime('%Y/%m/%d %H:%M:%S')), "w")

for j in range(num_iterations):

    session = Session(agent=Agent(c=5000, q=0),
                      policy=policy,
                      market=market,
                      start_day=0,
                      )
    policy, assets = session.run()

    running_reward.append(assets)

    if j % 10 == 0:
        mean = np.mean(running_reward)
        running_reward = []
        print("Average cumulated asset value (round {}): {:10.2f}".format(j, mean))
        # output.write("{:10.2f}\n".format(mean))

# output.close()
