from math import floor
from random import randint
import numpy as np


class Simulation:

    DURATION = 20

    def __init__(self, agent, policy, market):
        self.agent = agent
        self.policy = policy
        self.market = market
        self.init_cash = self.agent.cash
        self.start_day = randint(0, floor(market.duration/2))
        self.end_day = self.start_day + self.DURATION

    def run_transaction(self, action, stock_price):
        try:
            self.agent.transaction(action, stock_price)
            reward = 0.1
        except ValueError:
            reward = -0.1
        return reward

    def run(self):

        reward_sum = 0
        observed_states = []
        hidden_states = []
        rewards = []
        diff = []

        for i in range(self.start_day, self.end_day):

            immediate_reward = 0

            state = np.concatenate((self.market.indices[i], self.agent.state), axis=0)
            action_prob, hidden = self.policy.forward(state)

            if np.random.uniform() < action_prob:
                action = "buy"
                y = 1
            else:
                action = "sell"
                y = 0  # a "fake label"

            immediate_reward = self.run_transaction(action, self.market.stock_price[i])
            reward_sum += immediate_reward

            observed_states.append(state)  # observation
            hidden_states.append(hidden)  # hidden state
            diff.append(y - action_prob)
            rewards.append(immediate_reward)

            assets = self.agent.cash + self.agent.stock*self.market.stock_price[self.end_day-1]
            reward = ((assets - self.init_cash) / self.init_cash)
            rewards[-1] = rewards[-1] + reward

        return observed_states, hidden_states, diff, rewards
