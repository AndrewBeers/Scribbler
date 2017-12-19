
import csv
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from qtim_tools.qtim_utilities.format_util import convert_input_2_numpy

from agent import Agent, Scribbler
# from self.actions import Move

class Learner(object):

    def __init__(self, volume_filepath, label_filepath, radius=16, hidden_num=32, learning_rate=.001, random_rate=.1,
        epochs=1000, display_step=50, output_results='results.csv'):

        self.volume_filepath = volume_filepath
        self.label_filepath = label_filepath

        self.radius = radius

        self.hidden_num = hidden_num

        self.learning_rate = learning_rate
        self.random_rate = random_rate

        self.epochs = epochs
        self.display_step = display_step

        self.output_results = output_results

        self.sess = None

    def run(self):

        self.load_data()
        self.load_model()
        self.run_model()

    def load_data(self):

        self.volume = convert_input_2_numpy(self.volume_filepath)
        self.label = convert_input_2_numpy(self.label_filepath)

        self.label_centroid = np.array(np.argwhere(self.label==1).sum(0) / (self.label == 1).sum()).astype(np.float32).reshape(1,3)
        self.starting_point = np.array([np.random.randint(0, i-16) for i in self.volume.shape]).astype(np.float32).reshape(1,3)

        self.patch = extract_patch(self.volume, self.starting_point, self.radius)

        print 'starting_point', self.starting_point.shape, self.starting_point.dtype
        print 'centroid', self.label_centroid.shape, self.label_centroid.dtype, self.label_centroid

    def load_model(self):

        # [distance, angle0, angle1
        self.actions = [np.random.randint(0, self.radius-1), np.random.random(), np.random.random()]
        # self.weights = tf.Variable(tf.ones(len(self.actions)))

        # 'position' = np.array([np.random.randint(0, max_dim) for max_dim in self.volume.shape]), 

        self.weights = {
            'input': tf.Variable(tf.random_normal([len(self.actions), self.hidden_num])),
            'output': tf.Variable(tf.random_normal([self.hidden_num, len(self.actions)]))
        }

        self.biases = {
            'input': tf.Variable(tf.random_normal([self.hidden_num])),
            'output': tf.Variable(tf.random_normal([len(self.actions)]))
        }

        state = {'patch': tf.placeholder(shape=3*(int(self.radius),), dtype=tf.float32), 
                'weights': self.weights,
                'biases': self.biases,
                'position': tf.placeholder(shape=(1,3), dtype=tf.float32),
                'goal': tf.placeholder(shape=(1,3), dtype=tf.float32), 
                'radius': self.radius}

        self.scribbler_agent = Scribbler(actions=self.weights, state=state)
        self.scribbler_agent.init_vars()

        self.loss = self.scribbler_agent.cost()

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        self.gradients = tf.gradients(self.loss,tvars)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.update = self.optimizer.minimize(self.loss)
        self.update_batch = self.optimizer.apply_gradients(zip(self.gradient_holders,tvars))

        self.init_op = tf.initialize_all_variables()

    def run_model(self, mode='episodic'):

        if mode == 'basic':
            self.run_model_basic()

        if mode == 'episodic':
            self.run_model_episode()

    def run_model_episode(self):

        total_episodes = 5000 #Set total number of episodes to train agent on.
        max_ep = 999
        update_frequency = 1

        init = tf.global_variables_initializer()

        if self.sess == None:
            self.sess = tf.Session()
            self.saver = tf.train.Saver()  
            self.sess.run(self.init_op)
            
        gradBuffer = self.sess.run(tf.trainable_variables())
        for idx,grad in enumerate(gradBuffer):
            gradBuffer[idx] = grad * 0

        i = 0
        total_reward = []
        total_length = []


        with open(self.output_results, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            while i < total_episodes:

                self.starting_point = np.array([np.random.randint(0, i-16) for i in self.volume.shape]).astype(np.float32).reshape(1,3)
                running_reward = 0
                episode_history = []

                print self.starting_point
                for j in range(20):

                    action = self.sess.run(self.scribbler_agent.prediction(), feed_dict={self.scribbler_agent.state['position']:self.starting_point,self.scribbler_agent.state['goal']:self.label_centroid})
                    new_pos = advance_position(self.starting_point, action)

                    reward = self.sess.run(self.scribbler_agent.cost(), feed_dict={self.scribbler_agent.state['position']:new_pos,self.scribbler_agent.state['goal']:self.label_centroid})
                    episode_history.append([self.starting_point,action,reward,new_pos])
                    self.starting_point = new_pos

                    for idx in xrange(3):
                        self.starting_point[:,idx] = np.clip(self.starting_point[:,idx], 0, self.volume.shape[idx])

                    layer_out = self.sess.run(self.scribbler_agent.layer_out, feed_dict={self.scribbler_agent.state['position']:self.starting_point,self.scribbler_agent.state['goal']:self.label_centroid})

                    running_reward += reward
                    # print 'LAYER_OUT', layer_out
                    # print 'ACTION', action
                    # print 'CHANGE', (np.squeeze(action)*16).astype(int)
                    # print 'POSITION', self.starting_point
                    # print 'GOAL', self.label_centroid
                    print 'COST', self.sess.run(self.scribbler_agent.cost(), feed_dict={self.scribbler_agent.state['position']:self.starting_point,self.scribbler_agent.state['goal']:self.label_centroid})
                    # print '\n'
                    writer.writerow(self.starting_point[0,:])

                #Update the network.
                episode_history = np.array(episode_history)
                episode_history[:,2] = discount_rewards(episode_history[:,2])

                # feed_dict={myAgent.reward_holder:episode_history[:,2],
                        # myAgent.action_holder:episode_history[:,1],myAgent.state_in:np.vstack(episode_history[:,0])}

                grads = self.sess.run(self.gradients, feed_dict={self.scribbler_agent.state['position']:self.starting_point,self.scribbler_agent.state['goal']:self.label_centroid})

                for idx,grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    feed_dict= dictionary = dict(zip(self.gradient_holders, gradBuffer))
                    _ = self.sess.run(self.update_batch, feed_dict=feed_dict)
                    for ix,grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0
                
                total_reward.append(running_reward)
                total_length.append(j)

                i += 1

    def run_model_basic(self):

        if self.sess == None:
            self.sess = tf.Session()
            self.saver = tf.train.Saver()  
            self.sess.run(self.init_op)

        with open(self.output_results, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            i = 0
            while i < self.epochs:
                
                #Choose either a random action or one from our network.
                if np.random.rand(1) < self.random_rate:
                    action = (np.array([np.random.randint(0, self.radius-1) for i in xrange(3)]) / self.radius).reshape(1,3)
                else:
                    action = self.sess.run(self.scribbler_agent.prediction(), feed_dict={self.scribbler_agent.state['position']:self.starting_point,self.scribbler_agent.state['goal']:self.label_centroid})

                layer_out = self.sess.run(self.scribbler_agent.layer_out, feed_dict={self.scribbler_agent.state['position']:self.starting_point,self.scribbler_agent.state['goal']:self.label_centroid})

                self.sess.run(self.update, feed_dict={self.scribbler_agent.state['position']:self.starting_point,self.scribbler_agent.state['goal']:self.label_centroid})
                
                if i % self.display_step == 0:
                    print 'LAYER_OUT', layer_out
                    print 'ACTION', action
                    print 'CHANGE', (np.squeeze(action)*16).astype(int)
                    print 'POSITION', self.starting_point
                    print 'GOAL', self.label_centroid
                    print 'COST', self.sess.run(self.scribbler_agent.cost(), feed_dict={self.scribbler_agent.state['position']:self.starting_point,self.scribbler_agent.state['goal']:self.label_centroid})
                    print '\n'
                    writer.writerow(self.starting_point[0,:])

                self.starting_point = advance_position(self.starting_point, action)

                for idx in xrange(3):
                    self.starting_point[:,idx] = np.clip(self.starting_point[:,idx], 0, self.volume.shape[idx])

                i+=1


def discount_rewards(r, gamma=.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def extract_patch(patch, index, radius):

    index = np.squeeze(index.astype(int))

    print index
    print patch.shape

    return patch[index[0]:index[0]+radius, index[1]:index[1]+radius, index[2]:index[2]+radius]

def advance_position(position, action):

    return np.array(position + (np.squeeze(action)*16).astype(int))

if __name__ == '__main__':

    pass