from policy import Policy
from agent import Agent
from simulation import Simulation
from market import Market
from datetime import datetime
import numpy as np


# Model hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2

num_iterations = 5000000
market = Market('IntelDataSet.csv')
D = np.shape(market.indices)[1]+2
policy = Policy(H, D, gamma, batch_size, decay_rate, learning_rate)
running_reward = []
reward_sum = 0

# Output for further analysis, should also output model parameters
output = open("{}".format(datetime.now().strftime('%Y/%m/%d %H:%M:%S')), "w")

for j in range(num_iterations):

    agent = Agent(c=5000, q=0)
    sim = Simulation(agent, policy, market)
    observed_states, hidden_states, diff, rewards = sim.run()

    observed_states = np.vstack(observed_states)
    hidden_states = np.vstack(hidden_states)
    diff = np.vstack(diff)
    rewards = np.vstack(rewards)

    discounted_reward = policy.discount_rewards(rewards)
    gradient = discounted_reward * diff

    grad = policy.backward(hidden_states, gradient, observed_states)
    policy.update(grad, j)

    running_reward.append(rewards[-1])

    if j % 100 == 0:
        mean = np.mean(running_reward)
        running_reward = []
        print("Average return rate (round {}): {:10.2f}".format(j, mean))
        output.write("{:10.2f}\n".format(mean))

output.close()
