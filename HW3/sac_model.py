import json
from pdb import set_trace

import gym
import munch
import numpy as np
import scipy.signal
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Concatenate, Dense
from tensorflow.keras.models import Sequential

LOG_STD_MAX = 2
LOG_STD_MIN = -20
tfd = tfp.distributions


class CriticModel:
    def __init__(self, obs_size, act_size, output_size, activation_func="relu", lr=1e-4):
        # Define network
        obs_inputs = tf.keras.Input(shape=(obs_size,), name="obs_input")
        act_inputs = tf.keras.Input(shape=(act_size,), name="act_input")
        inputs = Concatenate()([obs_inputs, act_inputs])
        l1 = Dense(256, activation="relu")(inputs)
        l2 = Dense(256, activation="relu")(l1)
        l3 = Dense(output_size, activation=activation_func)(l2)
        self.net = tf.keras.Model(inputs=[obs_inputs, act_inputs], outputs=l3)

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def load(self, file_name):
        self.net.load_weights(file_name)

    def save(self, file_name):
        self.net.save_weights(file_name)


class ActorModel:
    def __init__(self, input_size, output_size, action_lim, activation_func="linear", lr=1e-4):
        # Define network
        inputs = tf.keras.Input(shape=(input_size,), name="input")
        l1 = Dense(256, activation="relu")(inputs)
        l2 = Dense(256, activation="relu")(l1)
        mu = Dense(output_size, activation=activation_func)(l2)
        log_std = Dense(output_size, activation=activation_func)(l2)
        self.net = tf.keras.Model(inputs=inputs, outputs=[mu, log_std])

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Action limit
        self.action_lim = action_lim

    def get_action(self, obs):
        mu, log_std = self.net(obs)
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = tf.math.exp(log_std)
        dist = tfd.Normal(loc=mu, scale=std)
        action = dist.sample()

        # get log prob of sampled action
        log_prob = tf.reduce_sum(dist.log_prob(action))
        log_prob -= tf.reduce_sum(2*(np.log(2) -
                                     action - tf.math.softplus(-2*action)))

        # get action
        action = tf.math.tanh(action)*self.action_lim

        return action, log_prob

    def load(self, file_name):
        self.net.load_weights(file_name)

    def save(self, file_name):
        self.net.save_weights(file_name)


class SAC:
    def __init__(self, config):
        self.config = config
        self.actor = ActorModel(
            config.actor_input_size,
            config.actor_output_size,
            config.action_lim,
            lr=config.actor_lr
        )
        self.critic1_eval = CriticModel(
            config.critic_obs_input_size,
            config.critic_act_input_size,
            config.critic_output_size,
            lr=config.critic_lr,
        )
        self.critic1_target = CriticModel(
            config.critic_obs_input_size,
            config.critic_act_input_size,
            config.critic_output_size,
            lr=config.critic_lr,
        )
        self.critic2_eval = CriticModel(
            config.critic_obs_input_size,
            config.critic_act_input_size,
            config.critic_output_size,
            lr=config.critic_lr,
        )
        self.critic2_target = CriticModel(
            config.critic_obs_input_size,
            config.critic_act_input_size,
            config.critic_output_size,
            lr=config.critic_lr,
        )


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


def main():
    config_path = "sac_config.json"
    with open(config_path) as json_file:
        config = json.load(json_file)
    config = munch.munchify(config)
    sac = SAC(config)
    # actor = ActorModel(config.actor_input_size, config.actor_output_size, 1)

    # env = gym.make("BipedalWalkerHardcore-v3")
    # obs = env.reset()
    # obs = np.expand_dims(obs, axis=0)
    # action, log_prob = actor.get_action(obs)


if __name__ == "__main__":
    main()
