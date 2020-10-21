import json
from datetime import datetime
from pdb import set_trace
from time import time

import gym
import munch
import numpy as np
import tensorflow as tf

from reinforce_model import REINFORCE


class Agent:
    def __init__(self, env, config):
        self.env = env
        self.reinforce = REINFORCE(self.env, config)
        self.iter_num = config.iter_num
        self.gamma = config.gamma
        self.stamp = datetime.fromtimestamp(time()).strftime("%Y%m%d-%H%M%S")
        self.max_iter = config.max_iter
        self.log_freq = config.log_freq
        self.max_test_iter = config.max_test_iter

    def train(self):
        train_log_dir = "logs/reinforce/" + self.stamp + "/train"
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        for episode in range(self.iter_num):
            cummulative_reward = 0
            is_terminal = False
            current_state = self.env.reset()
            step_num = 0

            self.rewards = []
            self.states = []
            self.actions = []

            while not is_terminal:
                action_prob = self.reinforce.net.net(
                    np.expand_dims(current_state, axis=0)
                )
                action = np.random.choice(
                    self.env.action_space.n, 1, p=action_prob[0].numpy()
                )[0]
                next_state, reward, is_terminal, _ = self.env.step(action)

                self.states.append(current_state)
                self.rewards.append(reward)
                self.actions.append(action)

                current_state = next_state
                cummulative_reward += reward
                step_num += 1
                if step_num >= self.max_iter:
                    break

            returns = np.zeros_like(self.rewards, dtype=np.float32)
            G = 0
            for i in reversed(range(0, len(self.rewards))):
                G = G * self.gamma + self.rewards[i]
                returns[i] = G

            one_hot_actions = tf.keras.utils.to_categorical(
                self.actions, self.env.action_space.n, dtype=np.float32
            )
            states = np.array([state for state in self.states])

            loss_value = self.train_step(states, one_hot_actions, returns)
            print(
                "Iteration: {}, Reward: {}, Loss: {}".format(
                    episode, cummulative_reward, loss_value
                )
            )

            with train_summary_writer.as_default():
                tf.summary.scalar("loss", loss_value, step=episode)
                tf.summary.scalar("reward", cummulative_reward, step=episode)

            if (episode + 1) % self.log_freq == 0:
                filename = "reinforce_models/{}/{}".format(self.stamp, episode + 1)
                self.reinforce.net.save(filename)
                print("Model saved at {}".format(filename))

    @tf.function
    def train_step(self, states, one_hot_actions, returns):
        with tf.GradientTape() as tape:
            policy_val = self.reinforce.net.net(states)
            log_policy = tf.math.log(
                tf.reduce_sum(policy_val * one_hot_actions, axis=1)
            )
            loss_value = -tf.reduce_mean(returns * log_policy)

        grads = tape.gradient(loss_value, self.reinforce.net.net.trainable_weights)
        self.reinforce.net.optimizer.apply_gradients(
            zip(grads, self.reinforce.net.net.trainable_weights)
        )
        return loss_value

    def test(self, filename, test_iters, render=False):
        self.reinforce.net.load(filename)
        test_log_dir = "logs/reinforce/" + self.stamp + "/test"
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        for episode in range(test_iters):
            cummulative_reward = 0
            is_terminal = False
            current_state = self.env.reset()
            step_num = 0

            while not is_terminal:
                action_prob = self.reinforce.net.net(
                    np.expand_dims(current_state, axis=0)
                )
                action = np.random.choice(
                    self.env.action_space.n, 1, p=action_prob[0].numpy()
                )[0]
                next_state, reward, is_terminal, _ = self.env.step(action)
                if render:
                    self.env.render()

                current_state = next_state
                cummulative_reward += reward
                step_num += 1

                if step_num >= self.max_iter:
                    break

            self.env.close()

            with test_summary_writer.as_default():
                tf.summary.scalar("reward", cummulative_reward, step=episode)


def main():
    env = gym.make("CartPole-v1").unwrapped
    # env = gym.wrappers.Monitor(env, "reinforce_recording", force=True)
    config_path = "reinforce_config.json"
    with open(config_path) as json_file:
        config = json.load(json_file)
    config = munch.munchify(config)
    reinforce_agent = Agent(env, config)
    reinforce_agent.train()
    # reinforce_agent.test(
    #     "reinforce_models/20201015-171524/6000", 20, render=False)


if __name__ == "__main__":
    main()
