import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense
import math


def critic_model(obs_size, act_size, output_size, hidden_size=(256, 256)):
    obs_inputs = tf.keras.Input(shape=(obs_size,), name="obs_input")
    act_inputs = tf.keras.Input(shape=(act_size,), name="act_input")
    inputs = Concatenate()([obs_inputs, act_inputs])

    # First layer
    l1_int_size = obs_size + act_size
    l1_k = math.sqrt(1 / l1_int_size)
    l1_int = tf.keras.initializers.RandomUniform(minval=-l1_k, maxval=l1_k)
    l1 = Dense(
        256, activation="relu", kernel_initializer=l1_int, bias_initializer=l1_int
    )(inputs)
