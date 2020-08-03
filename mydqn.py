import gym
import random
from collections import deque
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt

class DQN:
    REPLAY_MEMORY_SIZE = 10000  # number of tuples in experience replay
    EPSILON = 0.5  # epsilon of epsilon-greedy exploation
    EPSILON_DECAY = 0.99  # exponential decay multiplier for epsilon
    HIDDEN1_SIZE = 128  # size of hidden layer 1
    HIDDEN2_SIZE = 128  # size of hidden layer 2
    EPISODES_NUM = 2000  # number of episodes to train on. Ideally shouldn't take longer than 2000
    MAX_STEPS = 200  # maximum number of steps in an episode
    LEARNING_RATE = 0.01  # learning rate and other parameters for SGD/RMSProp/Adam
    MINIBATCH_SIZE = 10  # size of minibatch sampled from the experience replay
    GAMMA = 0.9  # MDP's gamma
    TARGET_UPDATE_FREQ = 100  # number of steps (not episodes) after which to update the target networks
    LOG_DIR = './logs'  # directory wherein logging takes place
    MINEPSILON = 0.1
    # Create and initialize the environment
    def __init__(self, env):
        self.env = gym.make(env)
        assert len(self.env.observation_space.shape) == 1
        self.input_size = self.env.observation_space.shape[0]  # In case of cartpole, 4 state features
        self.output_size = self.env.action_space.n  # In case of cartpole, 2 actions (right/left)
        self.replay_memory = deque()
    # Create the Q-network
    def initialize_network(self):

        # placeholder for the state-space input to the q-network
        self.x = tf.placeholder(tf.float32, [None, self.input_size])
        self.W1 = tf.Variable(tf.truncated_normal([self.input_size, self.HIDDEN1_SIZE], stddev = 0.01), name = 'W1')
        self.b1 = tf.Variable(tf.truncated_normal([1, self.HIDDEN1_SIZE], stddev = 0.01), name = 'b1')
        self.W2 = tf.Variable(tf.truncated_normal([self.HIDDEN1_SIZE, self.HIDDEN2_SIZE], stddev=0.01), name='W2')
        self.b2 = tf.Variable(tf.truncated_normal([1, self.HIDDEN2_SIZE], stddev=0.01), name='b2')
        self.W3 = tf.Variable(tf.truncated_normal([self.HIDDEN2_SIZE, self.output_size], stddev=0.01), name='W3')
        self.b3 = tf.Variable(tf.truncated_normal([1, self.output_size], stddev=0.01), name='b3')

        # Target Network

        self.Wt1 = tf.Variable(tf.truncated_normal([self.input_size, self.HIDDEN1_SIZE], stddev=0.01), name='Wt1')
        self.bt1 = tf.Variable(tf.truncated_normal([1, self.HIDDEN1_SIZE], stddev=0.01), name='bt1')
        self.Wt2 = tf.Variable(tf.truncated_normal([self.HIDDEN1_SIZE, self.HIDDEN2_SIZE], stddev=0.01), name='Wt2')
        self.bt2 = tf.Variable(tf.truncated_normal([1, self.HIDDEN2_SIZE], stddev=0.01), name='bt2')
        self.Wt3 = tf.Variable(tf.truncated_normal([self.HIDDEN2_SIZE, self.output_size], stddev=0.01), name='Wt3')
        self.bt3 = tf.Variable(tf.truncated_normal([1, self.output_size], stddev=0.01), name='bt3')

        h1 = tf.matmul(self.x, self.W1) + self.b1
        a1 = tf.nn.relu(h1)
        h2 = tf.matmul(a1, self.W2) + self.b2
        a2 = tf.nn.relu(h2)
        self.q = tf.matmul(a2, self.W3) + self.b3

        # Target Q values

        ht1 = tf.matmul(self.x, self.Wt1) + self.bt1
        at1 = tf.nn.relu(ht1)
        ht2 = tf.matmul(at1, self.Wt2) + self.bt2
        at2 = tf.nn.relu(ht2)
        self.qt = tf.matmul(at2, self.Wt3) + self.bt3
    ############################################################
    # Design your q-network here.
    #
    # Add hidden layers and the output layer. For instance:
    #
    # with tf.name_scope('output'):
    #	W_n = tf.Variable(
    # 			 tf.truncated_normal([self.HIDDEN_n-1_SIZE, self.output_size],
    # 			 stddev=0.01), name='W_n')
    # 	b_n = tf.Variable(tf.zeros(self.output_size), name='b_n')
    # 	self.Q = tf.matmul(h_n-1, W_n) + b_n
    #
    #############################################################

    # Your code here

        self.curaction = tf.placeholder(tf.int64, [None], name = 'curaction')
        self.cur_one_hot = tf.one_hot(self.curaction, self.output_size, 1, 0, name = 'onehot',dtype=tf.float32)

        self.qvalues = tf.reduce_sum(tf.multiply(self.q, self.cur_one_hot, name = 'onehotq'), reduction_indices = [1])

        self.target = tf.placeholder(tf.float32, [None], name = 'target')
        self.error = self.target - self.qvalues
        self.loss = ((tf.square(self.error)) + (tf.nn.l2_loss(self.W1)) + (tf.nn.l2_loss(self.W2)) + (tf.nn.l2_loss(self.W3)))

    ############################################################
    # Next, compute the loss.
    #
    # First, compute the q-values. Note that you need to calculate these
    # for the actions in the (s,a,s',r) tuples from the experience replay's minibatch
    #
    # Next, compute the l2 loss between these estimated q-values and
    # the target (which is computed using the frozen target network)
    #
    ############################################################

    # Your code here
        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.LEARNING_RATE)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = optimizer.minimize(self.loss, global_step=global_step)
    ############################################################
    # Finally, choose a gradient descent algorithm : SGD/RMSProp/Adam.
    #
    # For instance:
    # optimizer = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)
    # global_step = tf.Variable(0, name='global_step', trainable=False)
    # self.train_op = optimizer.minimize(self.loss, global_step=global_step)
    #
    ############################################################

    # Your code here

    ############################################################

    def act(self, state):
        if (random.random()) <= self.EPSILON:
            return np.random.binomial(1, 0.5)
        qvalues = self.session.run(self.q, feed_dict = {self.x : state.reshape(1, state.shape[0])})
        return qvalues.argmax()

    def forward_prop(self, x):
        # print(x,'before')
        x = tf.linalg.normalize(x)[0]
        # print(x)
        x = x.numpy()
        # y = tf.make_ndarray(x)
        x = x.transpose()
        # x = x.numpy()
        # print(x,'y')
        # x = x.numpy()
        # x = tf.transpose(x)
        x = tf.Variable(x, dtype = tf.float32)
        # print(x, 'x')
        a1 = tf.tanh(tf.matmul(self.W1, x) + self.b1)
        output = tf.matmul(self.W2, a1) + self.b2

        return output
    def play(self):
        if len(self.replay_memory) >= self.MINIBATCH_SIZE:
            sample = np.array(random.sample(self.replay_memory, self.MINIBATCH_SIZE))

            states, actions = [], []
            for i in range(self.MINIBATCH_SIZE):
                    states.append(sample[i][0])
            for i in range(self.MINIBATCH_SIZE):
                    actions.append(sample[i][1])
                # states = [list(x) for x in sample[:, 0]]
                # actions = [list(x) for x in sample[:, 1]]
            rewards,next_states,dones = [],[],[]
            for i in range(self.MINIBATCH_SIZE):
                    rewards.append(sample[i][2])
            for i in range(self.MINIBATCH_SIZE):
                    next_states.append(sample[i][3])
                # print(next_states)
            for i in range(self.MINIBATCH_SIZE):
                    dones.append(sample[i][4])
                # next_states = [list(x) for x in sample[:, 3]]
                # dones = [list(x) for x in sample[:, 4]]


            q = np.max(self.session.run(self.q, feed_dict={self.x: next_states}), axis=1)
            maxact = np.where(dones, rewards, rewards + self.GAMMA * q)

            self.session.run([self.train_op, self.loss],
                             feed_dict={self.x: states, self.target: maxact, self.curaction: actions})

    def train(self, episodes_num=EPISODES_NUM):

        # Initialize summary for TensorBoard
        summary_writer = tf.summary.FileWriter(self.LOG_DIR)
        summary = tf.Summary()
        # Alternatively, you could use animated real-time plots from matplotlib
        # (https://stackoverflow.com/a/24228275/3284912)

        # Initialize the TF session
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        ############################################################
        # Initialize other variables (like the replay memory)
        ############################################################

        # Your code here

        ############################################################
        # Main training loop
        #
        # In each episode,
        #	pick the action for the given state,
        #	perform a 'step' in the environment to get the reward and next state,
        #	update the replay buffer,
        #	sample a random minibatch from the replay buffer,
        # 	perform Q-learning,
        #	update the target network, if required.
        #
        #
        #
        # You'll need to write code in various places in the following skeleton
        #
        ############################################################
        self.count = []
        for episode in range(episodes_num):

            state = self.env.reset()

            steps = 0
            score = 0
            finish = True
            ############################################################
            # Episode-specific initializations go here.
            ############################################################
            #
            # Your code here
            #
            ############################################################

            while finish:

            ############################################################
            # Pick the next action using epsilon greedy and and execute it
            ############################################################
                action = self.act(state)
                self.EPSILON = max(self.MINEPSILON, self.EPSILON*self.EPSILON_DECAY)
                steps += 1
                next_state, reward, done, _ = self.env.step(action)
                score += reward
                finish = not done

                exp = [state,action,reward, next_state, done]

                if len(self.replay_memory) == self.MINIBATCH_SIZE:
                    self.replay_memory.popleft()
                    self.replay_memory.append(exp)
                    pass
                else:
                    self.replay_memory.append(exp)
            # Your code here

            ############################################################
            # Step in the environment. Something like:
            # next_state, reward, done, _ = self.env.step(action)
            ############################################################

            # Your code here

            ############################################################
            # Update the (limited) replay buffer.
            #
            # Note : when the replay buffer is full, you'll need to
            # remove an entry to accommodate a new one.
            ############################################################

            # Your code here

            ############################################################
            # Sample a random minibatch and perform Q-learning (fetch max Q at s')
            #
            # Remember, the target (r + gamma * max Q) is computed
            # with the help of the target network.
            # Compute this target and pass it to the network for computing
            # and minimizing the loss with the current estimates
            #
            ############################################################
                reward -= abs(next_state[0])
                self.play()


            # Your code here

            ############################################################
            # Update target weights.
            #
            # Something along the lines of:
            # if total_steps % self.TARGET_UPDATE_FREQ == 0:
            # 	target_weights = self.session.run(self.weights)
            ############################################################
                if steps % self.TARGET_UPDATE_FREQ  == 0:
                    self.Wt1 = self.W1
                    self.Wt2 = self.W2
                    self.Wt3 = self.W3
                    self.bt1 = self.b1
                    self.bt2 = self.b2
                    self.bt3 = self.b3

                state = next_state

            # Your code here

            ############################################################
            # Break out of the loop if the episode ends
            #
            # Something like:
            # if done or (episode_length == self.MAX_STEPS):
            # 	break
            #
            ############################################################

            # Your code here

            ############################################################
            # Logging.
            #
            # Very important. This is what gives an idea of how good the current
            # experiment is, and if one should terminate and re-run with new parameters
            # The earlier you learn how to read and visualize experiment logs quickly,
            # the faster you'll be able to prototype and learn.
            #
            # Use any debugging information you think you need.
            # For instance :
            self.count.append(steps)
            print(episode, 'episode', steps, 'steps')
            # print("Training: Episode = %d, Length = %d, Global step = %d" % (episode, episode_length, total_steps))
            # summary.value.add(tag="episode length", simple_value=episode_length)
            # summary_writer.add_summary(summary, episode)

    # Simple function to visually 'test' a policy
    def playPolicy(self):

        done = False
        steps = 0
        state = self.env.reset()

        # we assume the CartPole task to be solved if the pole remains upright for 200 steps
        while not done and steps < 200:
            self.env.render()
            q_vals = self.session.run(self.q, feed_dict={self.x: [state]})
            action = q_vals.argmax()
            state, _, done, _ = self.env.step(action)
            steps += 1

        return steps


if __name__ == '__main__':

    # Create and initialize the model
    dqn = DQN('CartPole-v0')
    dqn.initialize_network()

    print("\nStarting training...\n")
    dqn.train()
    print("\nFinished training...\nCheck out some demonstrations\n")

    # Visualize the learned behaviour for a few episodes
    x = [i+1  for i in range(2000)]
    plt.plot(x,dqn.count)
    plt.show()
    results = []
    for i in range(50):
        episode_length = dqn.playPolicy()
        print("Test steps = ", episode_length)
        results.append(episode_length)
    print("Mean steps = ", sum(results) / len(results))

    print("\nFinished.")
    print("\nCiao, and hasta la vista...\n")