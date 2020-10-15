import tensorflow as tf
import gym
from pdb import set_trace
from model import DQN
import json
import munch
import numpy as np
from datetime import datetime
from time import time
import matplotlib.pyplot as plt


def get_action(action_num):
    if action_num == 0:
        # LEFT
        action = np.array([-1.0, 0.0, 0.0])
    elif action_num == 1:
        # RIGHT
        action = np.array([1.0, 0.0, 0.0])
    elif action_num == 2:
        # ACCELERATE
        action = np.array([0.0, 1.0, 0.0])
    elif action_num == 3:
        # BRAKE
        action = np.array([0.0, 0.0, 0.2])
    elif action_num == 4:
        # STRAIGHT
        action = np.array([0.0, 0.0, 0.0])
    return action


def rgb2gray(rgb):
    """
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[..., :3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32')


class DQNAgent:
    def __init__(self, config, env):
        self.env = env
        self.lr = config.lr
        self.dqn = DQN(config, self.env)
        self.epsilon = config.epsilon
        self.iter_num = config.iter_num
        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.stamp = datetime.fromtimestamp(time()).strftime('%Y%m%d-%H%M%S')
        self.max_iter = config.max_iter
        self.log_freq = config.log_freq
        self.target_update_freq = config.target_update_freq
        self.window_close_freq = config.window_close_freq

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        if not self.dqn.replay_buffer.is_burn_in:
            print("Burning in memory ...")
            current_state = self.env.reset()

            # Initialize history
            gray_state = np.expand_dims(rgb2gray(current_state), axis=2)
            state_memory = np.tile(gray_state, (1, 1, self.dqn.history_length))
            state_memory = np.expand_dims(state_memory, axis=0)

            for episode in range(self.dqn.replay_buffer.burn_in_size):
                action_num = np.random.choice(
                    [0, 1, 2, 3, 4], 1, replace=False)[0]
                action = get_action(action_num)
                next_state, reward, is_terminal, _ = self.env.step(action)
                gray_next_state = np.expand_dims(rgb2gray(next_state), axis=2)
                next_state_memory = np.concatenate(
                    (state_memory[:, :, :, 1:], np.expand_dims(gray_next_state, axis=0)), axis=3)

                # Append sample to replay buffer
                sample = {
                    "state": state_memory,
                    "next_state": next_state_memory,
                    "reward": reward,
                    "action": action_num,
                    "terminal": is_terminal,
                }
                self.dqn.replay_buffer.add_sample(sample)

                if is_terminal:
                    current_state = self.env.reset()

                    # Initialize history
                    gray_state = np.expand_dims(
                        rgb2gray(current_state), axis=2)
                    state_memory = np.tile(
                        gray_state, (1, 1, self.dqn.history_length))
                    state_memory = np.expand_dims(state_memory, axis=0)
                else:
                    current_state = next_state
                    state_memory = next_state_memory

                if (episode+1) % 1000 == 0:
                    print("Burned in {} pieces of memory ...".format(episode+1))

            self.dqn.replay_buffer.is_burn_in = True
            print("Burn in memory complete...")
        else:
            print("Memory already burned in...")

    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        epsilon = np.random.rand()
        if epsilon <= self.epsilon_true:
            action_num = np.random.choice(
                [0, 1, 2, 3, 4], 1, replace=False)[0]
            action = get_action(action_num)
        else:
            action, action_num = self.greedy_policy(q_values)
        return action, action_num

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        action_num = np.argmax(q_values)
        action = get_action(action_num)
        return action, action_num

    def train(self):
        self.burn_in_memory()
        train_log_dir = 'logs/gradient_tape/' + self.stamp + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.env.viewer = None

        # Assign eval_net weights to target_net
        self.dqn.target_net.net.set_weights(
            self.dqn.eval_net.net.get_weights())

        for episode in range(self.iter_num):
            cummulative_reward = 0
            is_terminal = False
            current_state = self.env.reset()
            self.epsilon_true = self.epsilon - \
                (self.epsilon - 0.01) * episode / (self.iter_num * 1.0)

            # Initialize history
            gray_state = np.expand_dims(rgb2gray(current_state), axis=2)
            state_memory = np.tile(gray_state, (1, 1, self.dqn.history_length))
            state_memory = np.expand_dims(state_memory, axis=0)

            step_num = 0

            while not is_terminal:
                q_values = self.dqn.eval_net.net(state_memory)
                action, action_num = self.epsilon_greedy_policy(q_values)
                next_state, reward, is_terminal, _ = self.env.step(action)
                gray_next_state = np.expand_dims(rgb2gray(next_state), axis=2)
                next_state_memory = np.concatenate(
                    (state_memory[:, :, :, 1:], np.expand_dims(gray_next_state, axis=0)), axis=3)

                sample = {
                    "state": state_memory,
                    "next_state": next_state_memory,
                    "reward": reward,
                    "action": action_num,
                    "terminal": is_terminal,
                }

                self.dqn.replay_buffer.add_sample(sample)

                cummulative_reward += reward
                current_state = next_state
                state_memory = next_state_memory

                step_num += 1
                if step_num >= self.max_iter:
                    break
            

            loss_value = self.optimize_step()

            print("Iteration: {}, Reward: {}, Loss: {}, Replay Buffer Size: {}".format(
                episode, cummulative_reward, loss_value, len(self.dqn.replay_buffer.buffer)))
            
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss_value, step=episode)
                tf.summary.scalar('reward', cummulative_reward, step=episode)

            # Deals with memory leak
            if (episode + 1) % self.window_close_freq == 0:
                self.env.close()

            if (episode + 1) % self.target_update_freq == 0:
                self.dqn.target_net.net.set_weights(
                    self.dqn.eval_net.net.get_weights())

            if (episode + 1) % self.log_freq == 0:
                filename = 'models/{}/{}'.format(self.stamp, episode + 1)
                self.dqn.eval_net.save(filename)
                print("Model saved at {}".format(filename))

    def optimize_step(self):
        batch = self.dqn.replay_buffer.get_samples(self.batch_size)
        # Shape [batch_size, image_h, image_w, history_length]
        batch_state_memory = np.array([sample["state"] for sample in batch])[
            :, 0, :, :, :]
        # Shape [batch_size, image_h, image_w, history_length]
        batch_next_state_memory = np.array(
            [sample["next_state"] for sample in batch])[:, 0, :, :, :]
        # Shape [batch_size, ]
        batch_reward = np.array([sample["reward"] for sample in batch])
        # Shape [batch_size, 3]
        batch_action = np.array([sample["action"] for sample in batch])
        # Shape [batch_size, ]
        batch_terminal = np.array([sample["terminal"] for sample in batch])

        next_q_vals = self.dqn.target_net.net(batch_next_state_memory)
        y_vals = batch_reward + self.gamma * \
            (1 - batch_terminal) * tf.reduce_max(next_q_vals, axis=1)
        one_hot_actions = tf.keras.utils.to_categorical(
            batch_action, self.dqn.action_size, dtype=np.float32)

        loss_value = self.train_step(
            y_vals, batch_state_memory, one_hot_actions)
        return loss_value

    @tf.function
    def train_step(self, y_vals, batch_state_memory, one_hot_actions):
        with tf.GradientTape() as tape:
            q_vals = self.dqn.eval_net.net(batch_state_memory, training=True)
            q_action_vals = tf.reduce_sum(
                tf.multiply(q_vals, one_hot_actions), axis=1)
            loss_value = self.dqn.eval_net.loss(y_vals, q_action_vals)

        grads = tape.gradient(
            loss_value, self.dqn.eval_net.net.trainable_weights)
        self.dqn.eval_net.optimizer.apply_gradients(
            zip(grads, self.dqn.eval_net.net.trainable_weights))
        return loss_value


def main():
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[2], True)

    with tf.device('/device:GPU:2'):
        env = gym.make('CarRacing-v0').unwrapped
        config_path = 'config.json'
        with open(config_path) as json_file:
            config = json.load(json_file)
        config = munch.munchify(config)
        agent = DQNAgent(config, env)
        agent.train()


if __name__ == "__main__":
    main()
