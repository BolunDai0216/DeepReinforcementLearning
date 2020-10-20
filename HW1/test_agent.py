from __future__ import print_function

from datetime import datetime
import numpy as np
import gym
import os
import json

from model import Model
from utils import *
from pdb import set_trace
from drive_manually import store_data


def run_episode(env, agent, rendering=True, max_timesteps=1000, history_length=4):

    episode_reward = 0
    step = 0

    state = env.reset()
    state_memory = []

    test_result = {
        "state": [],
        "next_state": [],
        "reward": [],
        "action": [],
        "terminal": [],
    }

    while True:

        # preprocess the state in the same way than in in your preprocessing in train_agent.py
        true_state = state
        state = np.expand_dims(rgb2gray(state), axis=2)

        # TODO: get the action from your agent! If you use discretized actions you need to transform them to continuous
        # actions again. a needs to have a shape like np.array([0.0, 0.0, 0.0])
        if step == 0:
            state_memory = np.tile(state, (1, 1, history_length))
            state_memory = np.expand_dims(state_memory, axis=0)
        else:
            state_memory = np.concatenate(
                (state_memory[:, :, :, 1:], np.expand_dims(state, axis=0)), axis=3)

        action = agent.model(state_memory, training=False)
        action = action.numpy()[0]

        next_state, r, done, info = env.step(action)
        episode_reward += r
        state = next_state
        step += 1

        # state has shape (96, 96, 3)
        test_result["state"].append(true_state)
        test_result["action"].append(np.array(action))
        test_result["next_state"].append(next_state)
        test_result["reward"].append(r)
        test_result["terminal"].append(done)

        if rendering:
            env.render()

        if done or step > max_timesteps:
            break

    return episode_reward, test_result


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True

    n_test_episodes = 15
    history_length = 10

    # load agent
    agent = Model(history_length)
    agent.load("models/20201004-224733/8")

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward, test_result = run_episode(
            env, agent, rendering=rendering, history_length=history_length)
        episode_rewards.append(episode_reward)
        store_data(test_result, "./data")

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(fname, 'w') as fp:
        json.dump(results, fp)

    env.close()
    print('... finished')
