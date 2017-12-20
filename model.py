import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

class batchnorm(object):

    # Taken from DCGAN-tensorflow on Github. In future, rewrite for multi-backend batchnorm.

    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name
        
    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, is_training=train)

def leaky_relu(x, leak=0.2, backend='tf', name="lrelu"):
    
    if backend == 'tf':
        return tf.maximum(x, leak*x)

def relu(backend='tf'):

    if backend == 'tf':
        return tf.nn.relu


def tanh(backend='tf'):

    if backend == 'tf':
        return tf.nn.tanh


def sigmoid(backend='tf'):
    
    if backend == 'tf':
        return tf.nn.sigmoid

def dense(tensor, output_size, stddev=0.02, bias_start=0.0, with_w=False, name='dense'):

    with tf.variable_scope(name):

        shape = tensor.get_shape().as_list()

        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable("bias", [output_size], initializer=tf.zeros_initializer())

        if with_w:
            return tf.matmul(tensor, matrix) + bias, matrix, bias
        else:
            return tf.matmul(tensor, matrix) + bias

def conv3d(input_, output_dim, kernel_size=(8,8,2), stride_size=(1,1,1), stddev=0.02, name="conv3d", padding='SAME'):

    with tf.variable_scope(name):

        w = tf.get_variable('w', [kernel_size[0], kernel_size[1], kernel_size[2], input_.get_shape()[-1], output_dim], initializer=tf.contrib.layers.xavier_initializer())

        conv = tf.layers.conv3d(input_, output_dim, kernel_size=kernel_size, strides=[stride_size[0], stride_size[1], stride_size[2]], padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.zeros_initializer())
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def stacked_dense(state, layer_num=2, hidden_num=16, output_size=3, batch_norm=True):

    with tf.variable_scope("stacked_dense") as scope:

        # I still hate try/except checks.
        try:
            len(hidden_num)
        except:
            hidden_num = [hidden_num] * layer_num

        hidden_num[-1] = output_size

        layers = []

        for level in xrange(layer_num):

            if level == 0:
                layers += [dense(state, hidden_num[level], name='dense' + str(level))]
            else:
                layers += [dense(layers[-1], hidden_num[level], name='dense' + str(level))]

            if level == layer_num - 1:
                layers[-1] = tanh()(layers[-1])
            else:
                layers[-1] = leaky_relu((layers[-1]))

            if batch_norm and level != layer_num - 1:
                layers[-1] = batchnorm()(layers[-1]) # Going to hell for this

        return layers[-1]

