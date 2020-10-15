import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
import json
import munch
import numpy as np


class Model:
    def __init__(self, history_length, action_size):
        # Define network
        inputs = tf.keras.Input(shape=(96, 96, history_length), name='input')
        l1 = Conv2D(8, (7, 7), strides=(3, 3), activation='relu')(inputs)
        l2 = MaxPool2D(pool_size=(2, 2), strides=2)(l1)
        l3 = Conv2D(16, (3, 3), activation='relu')(l2)
        l4 = MaxPool2D(pool_size=(2, 2), strides=2)(l3)
        l5 = Flatten()(l4)
        l6 = Dense(256, activation='relu')(l5)
        l7 = Dense(action_size, activation='linear')(l6)
        self.net = tf.keras.Model(inputs=inputs, outputs=l7)


class ReplayBuffer:
    def __init__(self, config):
        self.max_size = config.max_buffer_size
        self.burn_in_size = config.burn_in_size
        self.buffer = []
        self.sample_array = np.arange(self.max_size)
        self.is_burn_in = False

    def add_sample(self, sample):
        self.buffer.append(sample)
        if len(self.buffer) > self.max_size:
            self.buffer.pop()

    def get_samples(self, batch_size):
        idx = np.random.choice(self.sample_array, batch_size, replace=False)
        samples = [self.buffer[i] in idx.tolist()]
        return samples


class DQN:
    def __init__(self, config, env):
        self.history_length = config.history_length
        self.lr = config.lr
        self.env = env
        self.eval_net = Model(self.history_length,
                              self.env.action_space.shape[0])
        self.target_net = Model(self.history_length,
                                self.env.action_space.shape[0])
        self.replay_buffer = ReplayBuffer(config)


def main():
    config_path = 'config.json'
    with open(config_path) as json_file:
        config = json.load(json_file)
    config = munch.munchify(config)
    dqn = DQN(config)


if __name__ == "__main__":
    main()
