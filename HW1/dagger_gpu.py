from __future__ import print_function

from datetime import datetime
import numpy as np
import gym
import os
import json

from model import Model
from utils import *
from pdb import set_trace
from pyglet.window import key
from train_agent import preprocessing, TrainModel, read_data
import time
import tensorflow as tf


def key_press(k, mod):
    global restart
    if k == 0xff0d:
        restart = True
    if k == key.LEFT:
        a[0] = -1.0
        a[3] = 1.0
    if k == key.RIGHT:
        a[0] = +1.0
        a[3] = 1.0
    if k == key.UP:
        a[1] = +1.0
        a[3] = 1.0
    if k == key.DOWN:
        a[2] = +0.2
        a[3] = 1.0
    if k == key.SPACE:
        a[0] = 0.0
        a[1] = 0.0
        a[2] = 0.0
        a[3] = 1.0


def key_release(k, mod):
    if k == key.LEFT and a[0] == -1.0:
        a[0] = 0.0
        a[3] = 0.0
    if k == key.RIGHT and a[0] == +1.0:
        a[0] = 0.0
        a[3] = 0.0
    if k == key.UP:
        a[1] = 0.0
        a[3] = 0.0
    if k == key.DOWN:
        a[2] = 0.0
        a[3] = 0.0
    if k == key.SPACE:
        a[0] = 0.0
        a[1] = 0.0
        a[2] = 0.0
        a[3] = 0.0


def process_data(data, frac=0.1, history_length=4):
    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)
                         ], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples)
                             :], y[int((1-frac) * n_samples):]

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(
        X_train, y_train, X_valid, y_valid, history_length=history_length)

    return X_train, y_train, X_valid, y_valid


def run_episode(env, agent, rendering=True, max_timesteps=1000, history_length=4):

    episode_reward = 0
    step = 0
    state = env.reset()

    samples = {
        "state": [],
        "next_state": [],
        "reward": [],
        "action": [],
        "terminal": [],
    }
    while True:

        # preprocess the state in the same way than in in your preprocessing in train_agent.py
        state_gray = np.expand_dims(rgb2gray(state), axis=2)

        # TODO: get the action from your agent! If you use discretized actions you need to transform them to continuous
        # actions again. a needs to have a shape like np.array([0.0, 0.0, 0.0])
        if step == 0:
            state_memory = np.tile(state_gray, (1, 1, history_length))
            state_memory = np.expand_dims(state_memory, axis=0)
        else:
            state_memory = np.concatenate(
                (state_memory[:, :, :, 1:], np.expand_dims(state_gray, axis=0)), axis=3)

        action = agent.agent.model(state_memory, training=False)
        action = action.numpy()[0]

        if a[3] == 0.0:
            next_state, r, done, info = env.step(action)
        else:
            action = np.array([a[0], a[1], a[2]])
            next_state, r, done, info = env.step(action)

        print(action)

        samples["state"].append(state)
        samples["action"].append(np.array(action))
        samples["next_state"].append(next_state)
        samples["reward"].append(r)
        samples["terminal"].append(done)

        episode_reward += r
        state = next_state
        step += 1

        if rendering:
            env.render()

        if done or step > max_timesteps:
            break

        time.sleep(0.08)

    return episode_reward, samples


if __name__ == "__main__":
    # read data
    X_train_1, y_train_1, X_valid_1, y_valid_1 = read_data(
        "data-20200924-230400.pkl.gzip")
    # X_train_2, y_train_2, X_valid_2, y_valid_2 = read_data(
    #     "data-20200930-204613.pkl.gzip")
    # X_train_3, y_train_3, X_valid_3, y_valid_3 = read_data(
    #     "data-20200923-103146.pkl.gzip")

    # X_train = np.concatenate((X_train_1, X_train_2), axis=0)
    # y_train = np.concatenate((y_train_1, y_train_2), axis=0)
    # X_valid = np.concatenate((X_valid_1, X_valid_2), axis=0)
    # y_valid = np.concatenate((y_valid_1, y_valid_2), axis=0)

    X_train = X_train_1
    y_train = y_train_1
    X_valid = X_valid_1
    y_valid = y_valid_1

    # preprocess data
    history_length = 10
    X_train, y_train, X_valid, y_valid = preprocessing(
        X_train, y_train, X_valid, y_valid, history_length=history_length)

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True

    iterations = 5

    # load agent
    agent = TrainModel(history_length=history_length)
    agent.agent.load("models/5000_data")

    env = gym.make('CarRacing-v0').unwrapped
    env.reset()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    a = np.array([0.0, 0.0, 0.0, 0.0]).astype('float32')

    episode_rewards = []
    for i in range(iterations):
        for _ in range(1):
            episode_reward, samples = run_episode(
                env, agent, rendering=rendering, history_length=history_length)
            episode_rewards.append(episode_reward)
            X_train_tmp, y_train_tmp, X_valid_tmp, y_valid_tmp = process_data(
                samples, history_length=history_length)
            X_train = np.concatenate((X_train, X_train_tmp), axis=0)
            y_train = np.concatenate((y_train, y_train_tmp), axis=0)
            X_valid = np.concatenate((X_valid, X_valid_tmp), axis=0)
            y_valid = np.concatenate((y_valid, y_valid_tmp), axis=0)
        agent.train(X_train, y_train, X_valid, y_valid,
                    epochs=10, batch_size=128)
        agent.agent.save(
            'models/{}/{}'.format(agent.stamp, i))

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
