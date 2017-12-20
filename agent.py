
import numpy as np
import tensorflow as tf

from scipy.spatial import distance

from model import stacked_dense

class Agent(object):

    def __init__(self, state):

        self.state = state
        # self.actions = actions

        pass

    def cost(self):

        return 0

    def advance(self):

        return

class Scribbler(Agent):

    def build_model(self, layer_num, hidden_num, output_size=3, batch_norm=True):

        # self.layer_1 = tf.tanh(tf.matmul(self.state['position'], self.state['weights']['input']) + self.state['biases']['input'])
        # self.layer_out = tf.matmul(self.layer_1, self.state['weights']['output']) + self.state['biases']['output']
        # self.layer_tanh = tf.tanh(self.layer_out)

        self.layer_out = stacked_dense(self.state['position'], layer_num=layer_num, hidden_num=hidden_num, output_size=output_size, batch_norm=batch_norm)

        self.calc_position = tf.add(self.state['position'], self.prediction()*16)
        self.cost_func = tf.reduce_sum(tf.square(tf.subtract(self.calc_position, self.state['goal'])))

    def prediction(self):

        return self.layer_out

    def cost(self):

        # a, b = 360*a, 360*b
        # displacement = np.array(r * sin(a) * cos(b), r * sin(a) * sin(b), r * cos(a))

        return self.cost_func

        # return distance.euclidean(self.state['position'], self.state['goal'])

    def advance(self):

        # Calculate cartesian distance (why though?)
        r, a, b = self.actions
        a, b = 360*a, 360*b
        displacement = np.array(r * sin(a) * cos(b), r * sin(a) * sin(b), r * cos(a))

        self.state['position'] = self.state['position'] + displacement.astype(int)




