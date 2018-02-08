from policy_keras import Policy
from agent import Agent
from session import Session
from market import Market
from datetime import datetime
from math import floor
from random import randint
import numpy as np


# Model hyperparameters

num_iterations = 500000
market = Market('IntelDataSet.csv')
D = np.shape(market.indices)[1]+2
policy = Policy(input_dim=D,
                learning_rate=1e-4)
running_reward = []


# Output for further analysis, should also output model parameters
# output = open("{}".format(datetime.now().strftime('%Y/%m/%d %H:%M:%S')), "w")

for j in range(num_iterations):

    session = Session(agent=Agent(c=5000, q=0),
                      policy=policy,
                      market=market,
                      start_day=randint(0, floor(market.duration/2)),
                      duration=100,
                      )
    policy, asset = session.run()

    running_reward.append(asset)

    if j % 100 == 0:
        mean = np.mean(running_reward)
        std = np.std(running_reward)
        running_reward = []
        print("Average cumulated asset value (round {}):\nmean: {:10.2f}\nstd: {:10.2f}".format(j, mean, std))
        # output.write("{:10.2f}\n".format(mean))

# output.close()
