from keras.models import Sequential
from keras.layers import Dense, Activation


class Policy:
    def __init__(self, H, D, gamma, batch_size, decay_rate, learning_rate):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.gamma = gamma
        self.model = Sequential([
            Dense(H, input_dim=D, kernel_initializer='random_uniform'),
            Activation('relu'),
            Dense(H, kernel_initializer='random_uniform'),
            Activation('sigmoid'),
            ])
        self.model.compile(optimizer='sgd', loss="WATUP")

    def forward(self, x):
        action_prob = self.model.evaluate(x)
        return action_prob

    def update(self, state, y_hat, i):
        self.model.fit(x, y)
        pass
