from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import pandas as pd
import numpy as np

#MNIST:local network layers
def local_mnist(x):
    convFilter1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], mean=0, stddev=0.1))
    convBias1 = tf.Variable(tf.truncated_normal([32], mean=0, stddev=0.1))
    convLayer1 = tf.nn.conv2d(input=x, filter=convFilter1, strides=[1, 1, 1, 1], padding='SAME')
    convLayer1 = tf.add(convLayer1, convBias1)
    convLayer1 = tf.nn.relu(convLayer1)
    poolLayer1 = tf.nn.max_pool(value=convLayer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return poolLayer1

#SVHN:local network layers
def local_svhn(x):
    mu = 0
    sigma = 0.1
    reg_loss = tf.zeros(1)
    conv1_w = tf.Variable(tf.truncated_normal((5, 5, 3, 9), mu, sigma))

    conv1_b = tf.Variable(tf.zeros(9))
    conv1 = tf.nn.conv2d(x, conv1_w, [1, 1, 1, 1], 'VALID') + conv1_b
    conv1 = tf.nn.relu(conv1_w)
    pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

    return pool1