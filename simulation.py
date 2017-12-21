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

    def run(self):

        immediate_reward_sum = 0
        xs, hs, dlogps, drs = [], [], [], []

        for i in range(self.start_day, self.end_day):

            immediate_reward = 0
            # Appending the current state of the agent
            state = np.concatenate((self.market.indices[i], self.agent.state), axis=0)

            aprob, h = self.policy._forward(state)  # compute bias for coin toss from decision policy
            action = "buy" if np.random.uniform() < aprob else "sell"  # randomly decide what to do from policy!
            self.agent.transaction(action, self.market.stock_price[i])

            # THE REGULATOR
            if action == "sell":
                if self.agent.stock <= 0:
                    immediate_reward -= 0.1
                    immediate_reward_sum -= 0.1
                if self.agent.stock >= 0:
                    immediate_reward += 0.1
                    immediate_reward_sum += 0.1
            elif action == "buy":
                if self.agent.cash <= 0:
                    immediate_reward -= 0.1
                    immediate_reward_sum -= 0.1
                if self.agent.cash >= 0:
                    immediate_reward += 0.1
                    immediate_reward_sum += 0.1

            self.agent.transaction(action, self.market.stock_price[i])

            y = 1 if action == "buy" else 0  # a "fake label"

            xs.append(state)  # observation
            hs.append(h)  # hidden state
            dlogps.append(y - aprob)
            drs.append(immediate_reward)

            assets = self.agent.cash + self.agent.stock*self.market.stock_price[self.end_day-1]
            reward = ((assets - self.init_cash) / self.init_cash)
            drs[-1] = drs[-1] + reward

        return xs, hs, dlogps, drs
