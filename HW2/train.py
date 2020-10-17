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

# When running in a headless server use the command:
# xvfb-run python3 train.py


def get_action(action_num):
    action_space = [
        [-1, 1, 0.2], [0, 1, 0.2], [1, 1, 0.2],  # Action Space Structure
        [-1, 1,   0], [0, 1,   0], [1, 1,   0],  # (Steering Wheel, Gas, Break)
        # Range        -1~1       0~1   0~1
        [-1, 0, 0.2], [0, 0, 0.2], [1, 0, 0.2],
        [-1, 0,   0], [0, 0,   0], [1, 0,   0]
    ]
    action = np.array(action_space[action_num])
#     if action_num == 0:
#         # LEFT
#         action = np.array([-1.0, 0.0, 0.0])
#     elif action_num == 1:
#         # RIGHT
#         action = np.array([1.0, 0.0, 0.0])
#     elif action_num == 2:
#         # ACCELERATE
#         action = np.array([0.0, 1.0, 0.0])
#     elif action_num == 3:
#         # BRAKE
#         action = np.array([0.0, 0.0, 0.2])
#     elif action_num == 4:
#         # STRAIGHT
#         action = np.array([0.0, 0.0, 0.0])
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
        self.epsilon_decay = config.epsilon_decay
        self.iter_num = config.iter_num
        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.stamp = datetime.fromtimestamp(time()).strftime('%Y%m%d-%H%M%S')
        self.max_iter = config.max_iter
        self.test_iter_num = config.test_iter_num
        self.log_freq = config.log_freq
        self.target_update_freq = config.target_update_freq
        self.window_close_freq = config.window_close_freq
        self.epsilon_true = self.epsilon
        self.polyak_constant = config.polyak_constant

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
                    np.arange(self.dqn.action_size), 1, replace=False)[0]
                action = get_action(action_num)

                for _ in range(3):
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
                np.arange(self.dqn.action_size), 1, replace=False)[0]
            action = get_action(action_num)
        else:
            action, action_num = self.greedy_policy(q_values)
        return action, action_num

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        action_num = np.argmax(q_values)
        action = get_action(action_num)
        return action, action_num

    def train(self, filename=None):
        if filename is not None:
            self.dqn.eval_net.load(filename)

        self.burn_in_memory()
        train_log_dir = 'logs/gradient_tape/' + self.stamp + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        reward_log = []

        # Assign eval_net weights to target_net
        self.dqn.target_net.net.set_weights(
            self.dqn.eval_net.net.get_weights())

        for episode in range(self.iter_num):
            cummulative_reward = 0
            is_terminal = False
            current_state = self.env.reset()
            if self.epsilon_true > 0.1:
                self.epsilon_true *= self.epsilon_decay

            # Initialize history
            gray_state = np.expand_dims(rgb2gray(current_state), axis=2)
            state_memory = np.tile(gray_state, (1, 1, self.dqn.history_length))
            state_memory = np.expand_dims(state_memory, axis=0)

            step_num = 0
            negative_reward_counter = 0

            while not is_terminal:
                q_values = self.dqn.eval_net.net(state_memory)
                action, action_num = self.epsilon_greedy_policy(q_values)

                # Make action change less frequent
                r_counter = 0
                for _ in range(3):
                    next_state, reward, is_terminal, _ = self.env.step(action)

                    step_num += 1
                    cummulative_reward += reward
                    r_counter += reward
                    if is_terminal:
                        break

                gray_next_state = np.expand_dims(rgb2gray(next_state), axis=2)
                next_state_memory = np.concatenate(
                    (state_memory[:, :, :, 1:], np.expand_dims(gray_next_state, axis=0)), axis=3)
                loss_value_tmp = self.optimize_step()

                # Attempt to not include to many negative examples
                if step_num > 400:
                    if r_counter < 0:
                        negative_reward_counter += 1
                    if negative_reward_counter >= 25:
                        break

                if cummulative_reward < 0:
                    break

                # Encourage driving with full gas in right direction
#                 if action[1] == 1 and action[2] == 0:
#                     reward *= 1.5

                sample = {
                    "state": state_memory,
                    "next_state": next_state_memory,
                    "reward": reward,
                    "action": action_num,
                    "terminal": is_terminal,
                }

                self.dqn.replay_buffer.add_sample(sample)
                current_state = next_state
                state_memory = next_state_memory

                if step_num >= self.max_iter:
                    break

            loss_value = 0
#             optimize_runs = 10
#             for i in range(optimize_runs):
#                 loss_value_tmp = self.optimize_step()
#                 loss_value += loss_value_tmp / optimize_runs
            reward_log.append(cummulative_reward)

            print("Iteration: {}, Reward: {}, Loss: {}, Replay Buffer Size: {}".format(
                episode, cummulative_reward, loss_value, len(self.dqn.replay_buffer.buffer)))

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss_value, step=episode)
                tf.summary.scalar('reward', cummulative_reward, step=episode)

            # Deals with memory leak
            if (episode + 1) % self.window_close_freq == 0:
                self.env.close()

            # Polyak Averaging
            eval_weights = self.dqn.eval_net.net.get_weights()
            target_weights = self.dqn.target_net.net.get_weights()
            update_weights = []
            for eval_weight, target_weight in zip(eval_weights, target_weights):
                update_weights.append(
                    self.polyak_constant*eval_weight + (1-self.polyak_constant)*target_weight)
            self.dqn.target_net.net.set_weights(update_weights)

#             if (episode + 1) % self.target_update_freq == 0:
#                 self.dqn.target_net.net.set_weights(
#                     self.dqn.eval_net.net.get_weights())

            if (episode + 1) % self.log_freq == 0:
                filename = 'models/{}/{}'.format(self.stamp, episode + 1)
                self.dqn.eval_net.save(filename)
                print("Model saved at {}".format(filename))

        numpy.savetxt("./logs/reward/reward_{}.csv".format(self.stamp),
                      np.array(reward_log), delimiter=",")

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

    def test(self, filename, render=False):
        test_log_dir = 'logs/gradient_tape/' + self.stamp + '/test'
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        self.dqn.eval_net.load(filename)

        for episode in range(self.test_iter_num):
            cummulative_reward = 0
            is_terminal = False
            current_state = self.env.reset()

            # Initialize history
            gray_state = np.expand_dims(rgb2gray(current_state), axis=2)
            state_memory = np.tile(gray_state, (1, 1, self.dqn.history_length))
            state_memory = np.expand_dims(state_memory, axis=0)

            step_num = 0

            while not is_terminal:
                q_values = self.dqn.eval_net.net(state_memory)
                action, action_num = self.epsilon_greedy_policy(q_values)

                # Make action change less frequent
                for _ in range(3):
                    next_state, reward, is_terminal, _ = self.env.step(action)

                    step_num += 1
                    cummulative_reward += reward
                    if is_terminal:
                        break

                    if render:
                        self.env.render()

                gray_next_state = np.expand_dims(rgb2gray(next_state), axis=2)
                next_state_memory = np.concatenate(
                    (state_memory[:, :, :, 1:], np.expand_dims(gray_next_state, axis=0)), axis=3)

                current_state = next_state
                state_memory = next_state_memory

                if step_num >= self.max_iter:
                    break

            print("Iteration: {}, Reward: {}".format(
                episode, cummulative_reward))

            with test_summary_writer.as_default():
                tf.summary.scalar('reward', cummulative_reward, step=episode)

            # Deals with memory leak
            if (episode + 1) % self.window_close_freq == 0:
                self.env.close()


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
        # agent.train()
        agent.test("models/3800", render=True)


if __name__ == "__main__":
    main()
