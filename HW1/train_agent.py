from __future__ import print_function

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

from model import Model
from utils import *
from pdb import set_trace
import tensorflow as tf
from datetime import datetime
from time import time
# from tensorboard_evaluation import Evaluation


def read_data(name, datasets_dir="./data", frac=0.1):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, name)

    f = gzip.open(data_file, 'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)
                         ], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples)
                             :], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=4):
    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can either train your model with continous actions (as you get them from read_data) using regression
    #    or you discretize the action space using action_to_id() from utils.py. If you discretize them, you'll maybe find one_hot()
    #    useful and you may want to return X_train_unhot ... as well.

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96,1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    X_train_gray = np.expand_dims(rgb2gray(X_train), axis=3)
    X_valid_gray = np.expand_dims(rgb2gray(X_valid), axis=3)

    if history_length != 1:
        X_train = np.zeros(
            (X_train_gray.shape[0]-history_length+1, X_train_gray.shape[1], X_train_gray.shape[2], history_length))
        X_valid = np.zeros(
            (X_valid_gray.shape[0]-history_length+1, X_valid_gray.shape[1], X_valid_gray.shape[2], history_length))
        y_train = y_train[history_length-1:, :]
        y_valid = y_valid[history_length-1:, :]
        for i in range(history_length):
            # The data are stacked as [state_t0; state_t1; state_t2; ...]
            X_train[:, :, :, i] = X_train_gray[i:X_train.shape[0]+i, :, :, 0]
            X_valid[:, :, :, i] = X_valid_gray[i:X_valid.shape[0]+i, :, :, 0]

    return X_train, y_train, X_valid, y_valid


class TrainModel:
    def __init__(self, model_dir="./models",
                 tensorboard_dir="./tensorboard", history_length=4, lr=1e-4):
        # create result and model folders
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        self.lr = lr
        self.agent = Model(history_length, self.lr)
        self.test_acc_metric = tf.keras.metrics.MeanSquaredError()
        self.stamp = datetime.fromtimestamp(time()).strftime('%Y%m%d-%H%M%S')

    @tf.function
    def train_step(self, x_batch_train, y_batch_train):
        with tf.GradientTape() as tape:
            output = self.agent.model(x_batch_train, training=True)
            loss_value = self.agent.loss(y_batch_train, output)

        grads = tape.gradient(loss_value, self.agent.model.trainable_weights)
        self.agent.optimizer.apply_gradients(
            zip(grads, self.agent.model.trainable_weights))
        return loss_value

    @tf.function
    def test_step(self, x_batch_test, y_batch_test):
        output = self.agent.model(x_batch_test, training=False)
        self.test_acc_metric.update_state(y_batch_test, output)

    def train(self, X_train, y_train, X_valid, y_valid, epochs, batch_size):
        # Build Training Dataset
        self.batch_size = batch_size
        self.epochs = epochs

        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train))
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            (X_valid, y_valid))
        self.train_dataset = self.train_dataset.shuffle(
            buffer_size=1024).batch(self.batch_size)
        self.test_dataset = self.test_dataset.batch(self.batch_size)

        print("... train model")

        for epoch in range(epochs):
            # Each Epoch
            for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
                # Each Iteration
                loss_value = self.train_step(x_batch_train, y_batch_train)
                # if epoch % 10 == 0:
                # tensorboard_eval.write_episode_data(...)

            for x_batch_test, y_batch_test in self.test_dataset:
                self.test_step(x_batch_test, y_batch_test)

            test_acc = self.test_acc_metric.result()
            self.test_acc_metric.reset_states()

            print("epoch: {}, loss: {}, test_acc: {}".format(
                epoch, float(loss_value), float(test_acc)))

            # TODO: save your agent
            if epoch % 2 == 0:
                self.agent.model.save_weights(
                    'models/{}/{}'.format(self.stamp, epoch))
            # print("Model saved in file: %s" % model_dir)


if __name__ == "__main__":
    # data filenames
    list_of_data = ["data-20200923-102355.pkl.gzip", "data-20200923-102709.pkl.gzip",
                    "data-20200923-103146.pkl.gzip", "data-20200922-094550.pkl.gzip",
                    "data-20200924-230400.pkl.gzip", "data-20200924-230102.pkl.gzip",
                    "data-20200924-225749.pkl.gzip", "data-20200924-225011.pkl.gzip"]
    # read data
    X_train_list = []
    y_train_list = []
    X_valid_list = []
    y_valid_list = []

    for data in list_of_data:
        X_train_tmp, y_train_tmp, X_valid_tmp, y_valid_tmp = read_data(
            data)
        X_train_list.append(X_train_tmp)
        y_train_list.append(y_train_tmp)
        X_valid_list.append(X_valid_tmp)
        y_valid_list.append(y_valid_tmp)

    X_train = np.concatenate((X_train_list), axis=0)
    y_train = np.concatenate((y_train_list), axis=0)
    X_valid = np.concatenate((X_valid_list), axis=0)
    y_valid = np.concatenate((y_valid_list), axis=0)

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(
        X_train, y_train, X_valid, y_valid)

    # train model (you can change the parameters!)
    train = TrainModel()
    train.train(X_train, y_train, X_valid, y_valid,
                epochs=1000, batch_size=64)
