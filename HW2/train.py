import tensorflow as tf
import gym
from pdb import set_trace
from model import DQN
import json
import munch
import numpy as np


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
                action = self.env.action_space.sample()
                next_state, reward, is_terminal, _ = self.env.step(action)
                gray_next_state = np.expand_dims(rgb2gray(next_state), axis=2)
                next_state_memory = np.concatenate(
                    (state_memory[:, :, :, 1:], np.expand_dims(gray_next_state, axis=0)), axis=3)

                # Append sample to replay buffer
                sample = {
                    "state": state_memory,
                    "next_state": next_state_memory,
                    "reward": reward,
                    "action": action,
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
                    print("Burned in {} pieces of memory ...".format(episode))

            self.dqn.replay_buffer.is_burn_in = True
            print("Burn in memory complete...")
        else:
            print("Memory already burned in...")

    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        epsilon = np.random.rand()
        if epsilon <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.greedy_policy(q_values)

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        action_num = np.argmax(q_values)
        action = get_action(action_num)
        return action

    def train(self):
        self.reward_log = []
        self.burn_in_memory()

        for episode in range(self.iter_num):
            cummulative_reward = 0
            is_terminal = False
            current_state = self.env.reset()
            self.epsilon_true = self.epsilon - \
                (self.epsilon - 0.001) * episode / (self.iter_num * 1.0)

            # Initialize history
            gray_state = np.expand_dims(rgb2gray(current_state), axis=2)
            state_memory = np.tile(gray_state, (1, 1, self.dqn.history_length))
            state_memory = np.expand_dims(state_memory, axis=0)

            while not is_terminal:
                q_values = self.dqn.eval_net.net(state_memory)
                action = self.epsilon_greedy_policy(q_values)
                next_state, reward, is_terminal, _ = self.env.step(action)
                gray_next_state = np.expand_dims(rgb2gray(next_state), axis=2)
                next_state_memory = np.concatenate(
                    (state_memory[:, :, :, 1:], np.expand_dims(gray_next_state, axis=0)), axis=3)

                sample = {
                    "state": state_memory,
                    "next_state": next_state_memory,
                    "reward": reward,
                    "action": action,
                    "terminal": is_terminal,
                }

                self.dqn.replay_buffer.add_sample(sample)

                cummulative_reward += reward
                current_state = next_state
                state_memory = next_state_memory
                set_trace()

            print("Iteration: {}, Reward: {}".format(
                episode, cummulative_reward))


def main():
    env = gym.make('CarRacing-v0').unwrapped

    config_path = 'config.json'
    with open(config_path) as json_file:
        config = json.load(json_file)
    config = munch.munchify(config)
    agent = DQNAgent(config, env)

    state = env.reset()
    agent.train()
    set_trace()
    # agent.dqn.eval_net.net(state)

    while True:
        action = env.action_space.sample()
        set_trace()
        next_state, r, done, _ = env.step(action)


if __name__ == "__main__":
    main()
