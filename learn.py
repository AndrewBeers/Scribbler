
import numpy as np
import tensorflow as tf

from qtim_tools.qtim_utilities.format_util import convert_input_2_numpy

from agent import Agent, Scribbler
# from actions import Move

def learner(volume_filepath, label_filepath, radius=16, learning_rate=.001):

    volume = convert_input_2_numpy(volume_filepath)
    label = convert_input_2_numpy(label_filepath)

    label_centroid = np.argwhere(label==1).sum(0) / (label == 1).sum() 
    starting_point = [np.rand.randint(0, volume.shape[i]-16) for i in volume.shape]

    # [distance, angle0, angle1]
    actions = [np.rand.randint(0, radius-1), np.rand.random(), np.rand.random()]
    weights = tf.Variable(tf.ones(len(actions)))

    # data = extract_patch(volume, starting_point, radius)
    data = volume

    state = {'data': data, 
            # 'position' = np.array([np.rand.randint(0, max_dim) for max_dim in volume.shape]), 
            'position' = tf.placeholder(shape[3], dtype=tf.float32),
            'goal' = tf.placeholder(shape[3], dtype=tf.float32), 
            'radius' = radius}

    scribbler_agent = Scribbler(actions=weights, state=state)
    
    loss = scribbler_agent.cost()

    optimizer = tf.train.Adam(lr=learning_rate)
    update = optimizer.minimize(loss)

    total_episodes = 1000 #Set total number of episodes to train agent on.
    total_reward = np.zeros(num_bandits) #Set scoreboard for bandits to 0.
    e = 0.1 #Set the chance of taking a random action.

    init = tf.initialize_all_variables()

    # Launch the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)

        i = 0
        while i < total_episodes:
            
            #Choose either a random action or one from our network.
            if np.random.rand(1) < e:
                action = [np.rand.randint(0, radius-1) for i in xrange(3)]
            else:
                action = sess.run(weights)
            
            #Update the network.
            sess.run(update, feed_dict={state['position']:starting_point,state['goal']:label_centroid})
            
            print sess.run(scribbler_agent.cost())

            if i % 50 == 0:
                print "Running reward for the " + str(num_bandits) + " bandits: " + str(total_reward)
            i+=1
    print "The agent thinks bandit " + str(np.argmax(ww)+1) + " is the most promising...."
    if np.argmax(ww) == np.argmax(-np.array(bandits)):
        print "...and it was right!"
    else:
        print "...and it was wrong!"

    # time_steps = 100

    # for t in xrange(time_steps):
    #     pass

def extract_patch(data, index, radius):

    return data[index[0]:index[0]+radius, index[1]:index[1]+radius, index[2]:index[2]+radius]

if __name__ == '__main__':

    volume_filepath = 'sample_volume.nii.gz'
    label_filepath = 'sample_volume-label.nii.gz'

    learner(volume_filepath, label_filepath,
        radius = 16,
        learning_rate=.001)