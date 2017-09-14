
import numpy as np

from scipy.spatial import distance

class Agent(object):

    def __init__(self, state, actions):

        self.state = state
        self.actions = actions

        pass

    def cost(self):

        return 0

    def advance(self):

        return

class Scribbler(Agent):

    def cost(self):

        # a, b = 360*a, 360*b
        # displacement = np.array(r * sin(a) * cos(b), r * sin(a) * sin(b), r * cos(a))

        new_position = tf.add(self.state['position'], self.actions)

        # return distance.euclidean(self.state['position'], self.state['goal'])
        return tf.square(tf.subtract(new_position, self.state['goal']))



    def advance(self):

        # Calculate cartesian distance (why though?)
        r, a, b = self.actions
        a, b = 360*a, 360*b
        displacement = np.array(r * sin(a) * cos(b), r * sin(a) * sin(b), r * cos(a))

        self.state['position'] = self.state['position'] + displacement.astype(int)





