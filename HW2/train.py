import tensorflow as tf
import gym
from pdb import set_trace
from model import DQN
import json
import munch


def get_action(action_num):
    if action_num == 0:
        # LEFT
        action = [-1.0, 0.0, 0.0]
    elif action_num == 1:
        # RIGHT
        action = [1.0, 0.0, 0.0]
    elif action_num == 2:
        # ACCELERATE
        action = [0.0, 1.0, 0.0]
    elif action_num == 3:
        # BRAKE
        action = [0.0, 0.0, 0.2]
    return action


def main():
    env = gym.make('CarRacing-v0').unwrapped

    config_path = 'config.json'
    with open(config_path) as json_file:
        config = json.load(json_file)
    config = munch.munchify(config)
    dqn = DQN(config, env)

    state = env.reset()

    while True:
        action = env.action_space.sample()
        next_state, r, done, _ = env.step(action)

        samples = {
            "state": state,
            "next_state": next_state,
            "reward": r,
            "action": action,
            "terminal": done,
        }

        set_trace()


if __name__ == "__main__":
    main()
