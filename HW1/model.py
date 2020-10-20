import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D


class Model:
    def __init__(self, history_length, lr=1e-4):

        # Define network
        inputs = tf.keras.Input(shape=(96, 96, history_length), name='input')
        l1 = Conv2D(32, (8, 8), strides=(2, 2), activation='relu')(inputs)
        l2 = Conv2D(64, (8, 8), strides=(2, 2), activation='relu')(l1)
        l3 = Conv2D(128, (5, 5), activation='relu')(l2)
        l4 = Conv2D(128, (5, 5), activation='relu')(l3)
        l5 = Flatten()(l4)
        l6 = Dense(1024, activation='relu')(l5)
        l7 = Dense(128, activation='relu')(l6)
        l8 = Dense(3, activation='linear')(l7)
        self.model = tf.keras.Model(inputs=inputs, outputs=l8)

        # Loss and optimizer
        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def load(self, file_name):
        self.model.load_weights(file_name)

    def save(self, file_name):
        self.model.save_weights(file_name)
