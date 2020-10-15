import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import json
import munch
import numpy as np


class Model:
    def __init__(self, action_size, lr=1e-4):
        # Define network
        inputs = tf.keras.Input(shape=(4, ), name='input')
        l1 = Dense(32, activation='relu')(inputs)
        l2 = Dense(32, activation='relu')(l1)
        l3 = Dense(action_size, activation='softmax')(l2)
        self.net = tf.keras.Model(inputs=inputs, outputs=l3)

        # Loss and optimizer
        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def load(self, file_name):
        self.net.load_weights(file_name)

    def save(self, file_name):
        self.net.save_weights(file_name)


class REINFORCE:
    def __init__(self, env, config):
        self.env = env
        self.lr = config.lr
        self.net = Model(self.env.action_space.n, lr=self.lr)
