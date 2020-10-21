import json

import munch
import scipy.signal
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Model:
    def __init__(self, input_size, output_size, activation_func="linear", lr=1e-4):
        # Define network
        inputs = tf.keras.Input(shape=(input_size,), name="input")
        l1 = Dense(64, activation="relu")(inputs)
        l2 = Dense(64, activation="relu")(l1)
        l3 = Dense(output_size, activation=activation_func)(l2)
        self.net = tf.keras.Model(inputs=inputs, outputs=l3)

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def load(self, file_name):
        self.net.load_weights(file_name)

    def save(self, file_name):
        self.net.save_weights(file_name)


class PPO:
    def __init__(self, config):
        self.config = config
        self.actor = Model(
            config.actor_input_size,
            config.actor_output_size,
            "softmax",
            lr=config.actor_lr,
        )
        self.critic = Model(
            config.critic_input_size,
            config.critic_output_size,
            "linear",
            lr=config.critic_lr,
        )
        self.buffer = []
        self.gamma = config.gamma
        self.lam = config.lam


def main():
    config_path = "ppo_config.json"
    with open(config_path) as json_file:
        config = json.load(json_file)
    config = munch.munchify(config)
    ppo = PPO(config)


if __name__ == "__main__":
    main()
