import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Policy:

    def __init__(self, input_dim, learning_rate=1e-4):
        self.learning_rate = learning_rate
        self.input_dimension = input_dim
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(100, input_shape=(self.input_dimension,)))
        model.add(Dense(64, activation='relu', init='he_uniform'))
        model.add(Dense(32, activation='relu', init='he_uniform'))
        model.add(Dense(1, activation='sigmoid'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    def decide(self, state):
        state = state.reshape(1, state.shape[0])
        action_prob = self.model.predict(state, batch_size=1).flatten()
        return action_prob

    def train(self, state, y_hat, i):
        y_hat = np.array([y_hat])
        y_hat = np.vstack([y_hat])
        state = np.vstack([state])
        self.model.fit(state, y_hat, verbose=0)
        pass
