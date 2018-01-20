import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D


class Policy:

    def __init__(self, H, D, gamma, batch_size, decay_rate, learning_rate):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.gamma = gamma
        self.input_dimension = D
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(80, input_shape=(self.input_dimension,)))
        model.add(Dense(64, activation='relu', init='he_uniform'))
        model.add(Dense(32, activation='relu', init='he_uniform'))
        model.add(Dense(2, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    def decide(self, state):
        state = state.reshape(1, state.shape[0])
        action_prob = self.model.predict(state, batch_size=1).flatten()
        return action_prob[0]

    def train(self, state, y_hat, i):
        y_hat = np.array([y_hat, 1-y_hat])
        state = np.vstack([state])
        self.model.fit(state, y_hat)
        pass
