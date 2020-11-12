import json
from pdb import set_trace

import gym
import munch
import numpy as np
import scipy.signal
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense
from tensorflow.keras.models import Sequential

LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPS = 1e-8


def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.math.exp(log_std)+EPS))
                      ** 2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


class CriticModel:
    def __init__(self, obs_size, act_size, output_size, activation_func="relu", lr=1e-4):
        # Define network
        obs_inputs = tf.keras.Input(shape=(obs_size,), name="obs_input")
        act_inputs = tf.keras.Input(shape=(act_size,), name="act_input")
        inputs = Concatenate()([obs_inputs, act_inputs])
        l1 = Dense(400, activation="relu")(inputs)
        l2 = Dense(300, activation="relu")(l1)
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
        l1 = Dense(400, activation="relu")(inputs)
        l2 = Dense(300, activation="relu")(l1)
        mu = Dense(output_size, activation=activation_func)(l2)
        log_std = Dense(output_size, activation=activation_func)(l2)
        self.net = tf.keras.Model(inputs=inputs, outputs=[mu, log_std])

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Action limit
        self.action_lim = action_lim

    def get_action(self, obs, test=False):
        mu, log_std = self.net(obs)
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = tf.math.exp(log_std)
        # Reparameterization trick
        action = mu + tf.random.normal(tf.shape(mu))*std

        # get log prob of sampled action
        log_prob = gaussian_likelihood(action, mu, log_std)
        log_prob -= tf.reduce_sum(2*(np.log(2) -
                                     action - tf.math.softplus(-2*action)))

        # get action
        action = tf.math.tanh(action)*self.action_lim

        if test:
            action = mu

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


class BipedalWalkerHardcoreWrapper(object):
    def __init__(self, env, action_repeat=3):
        self._env = env
        self.action_repeat = action_repeat
        self.act_noise = 0.3
        self.reward_scale = 5.0

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        obs = self._env.reset()
        return obs

    def step(self, action):
        action += self.act_noise * (-2 * np.random.random(4) + 1)
        r = 0.0
        for _ in range(self.action_repeat):
            obs_, reward_, done_, info_ = self._env.step(action)
            r = r + reward_
            if done_ and self.action_repeat != 1:
                return obs_, 0.0, done_, info_
            if self.action_repeat == 1:
                return obs_, r, done_, info_
        return obs_, self.reward_scale * r, done_, info_


class ReplayBuffer:
    def __init__(self, config):
        self.max_size = int(config.max_buffer_size)
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


if __name__ == "__main__":
    main()
