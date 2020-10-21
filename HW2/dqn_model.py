import json

import munch
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential


class Model:
    def __init__(self, history_length, action_size, lr=1e-4):
        # Define network
        inputs = tf.keras.Input(shape=(96, 96, history_length), name="input")
        l1 = Conv2D(8, (7, 7), strides=(3, 3), activation="relu")(inputs)
        l2 = MaxPool2D(pool_size=(2, 2))(l1)
        l3 = Conv2D(16, (4, 4), activation="relu")(l2)
        l4 = MaxPool2D(pool_size=(2, 2))(l3)
        l5 = Flatten()(l4)
        l6 = Dense(256, activation="relu")(l5)
        l7 = Dense(action_size, activation="linear")(l6)
        self.net = tf.keras.Model(inputs=inputs, outputs=l7)

        # Loss and optimizer
        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-7)

    def load(self, file_name):
        self.net.load_weights(file_name)

    def save(self, file_name):
        self.net.save_weights(file_name)


class ReplayBuffer:
    def __init__(self, config):
        self.max_size = config.max_buffer_size
        self.burn_in_size = config.burn_in_size
        self.buffer = []
        self.is_burn_in = False

    def add_sample(self, sample):
        self.buffer.append(sample)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def get_samples(self, batch_size):
        sample_array = np.arange(len(self.buffer))
        idx = np.random.choice(sample_array, batch_size, replace=False)
        samples = [self.buffer[i] for i in idx.tolist()]
        return samples


class DQN:
    def __init__(self, config, env):
        self.history_length = config.history_length
        self.lr = config.lr
        self.env = env
        self.action_size = 5
        self.eval_net = Model(self.history_length, self.action_size, self.lr)
        self.target_net = Model(self.history_length, self.action_size, self.lr)
        self.replay_buffer = ReplayBuffer(config)
