import numpy as np


class Policy:

    def __init__(self, H, D, gamma, batch_size, decay_rate, learning_rate):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.gamma = gamma
        self.model = {}
        self.model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
        self.model['W2'] = np.random.randn(H) / np.sqrt(H)
        self.grad_buffer = {k: np.zeros_like(v) for k, v in self.model.items()}
        self.rmsprop_cache = {k: np.zeros_like(v) for k, v in self.model.items()}

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def _forward(self, x):
        h = np.dot(self.model['W1'], x)
        h[h < 0] = 0  # ReLU nonlinearity
        logp = np.dot(self.model['W2'], h)
        p = self.sigmoid(logp)
        return p, h  # return probability of taking action 2, and hidden state

    def _backward(self, eph, epdlogp, epx):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, self.model['W2'])
        dh[eph <= 0] = 0  # backpro prelu
        dW1 = np.dot(dh.T, epx)
        return {'W1': dW1, 'W2': dW2}

    def _update(self, grad, episode_number):
        for k in self.model:
            self.grad_buffer[k] += grad[k]

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % self.batch_size == 0:
            for k, v in self.model.items():
                g = self.grad_buffer[k]
                self.rmsprop_cache[k] = self.decay_rate * self.rmsprop_cache[k] + (1 - self.decay_rate) * g**2
                self.model[k] += self.learning_rate * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)
                self.grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer
