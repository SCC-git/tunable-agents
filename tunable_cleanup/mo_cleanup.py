# -*- coding: utf-8 -*-

from envs.custom_cleanup_env import CleanupEnv
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import random
import os
import sys
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense

from collections import deque # Used for replay buffer and reward tracking
from datetime import datetime # Used for timing script
import time


class ReplayMemory(deque):
    """
    Inherits from the 'deque' class to add a method called 'sample' for
    sampling batches from the deque.
    """
    def sample(self, batch_size):
        """
        Sample a minibatch from the replay buffer.
        """
        # Random sample of indices
        indices = np.random.randint(len(self),
                                    size=batch_size)
        # Filter the batch from the deque
        batch = [self[index] for index in indices]
        # Unpack and create numpy arrays for each element type in the batch
        states, actions, rewards, next_states, dones, weightss = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(6)]
        return states, actions, rewards, next_states, dones, weightss


class RewardTracker:
    """
    Class for tracking mean rewards and storing all episode rewards for
    analysis.
    """
    def __init__(self, maxlen):
        self.moving_average = deque([-np.inf for _ in range(maxlen)],
                                    maxlen=maxlen)
        self.maxlen = maxlen
        self.episode_rewards  = []

    def __repr__(self):
        # For printing
        return self.moving_average.__repr__()

    def append(self, reward):
        self.moving_average.append(reward)
        self.episode_rewards.append(reward)

    def mean(self):
        return sum(self.moving_average) / self.maxlen

    def get_reward_data(self):
        episodes = np.array(
            [i for i in range(len(self.episode_rewards ))]).reshape(-1,1)

        rewards = np.array(self.episode_rewards ).reshape(-1,1)
        return np.concatenate((episodes, rewards), axis=1)


class PreferenceSpace:

    def __init__(self):
        w0 = 0.05 # Fire
        w1 = 0.55 # Hit
        w2_range = np.linspace(0,0.4,5) # Apple
        # w3 = 0 # Clean
        self.distribution = [np.asarray([w0, w1, w2, 0.4-w2], dtype=np.float32) for w2 in w2_range]
        # w0 = 0.005 # Time penalty
        # w1 = 5 * w0 # Wall penalty : 5x time penalty
        # w2_range = np.linspace(0,0.97,5)
        # self.distribution = [np.array([w0, w1, w2, 0.97 - w2], dtype=np.float32) for w2 in w2_range]

    def sample(self):
        return random.choice(self.distribution)

    # # Uncomment below as necessary for training fixed agents
    # def sample(self):
    #     return np.array([-1, -5, +10, +20, +10, -20], dtype=np.float32) # Competitive
    #     # return np.array([-1, -5, +10, +20, +10, +20], dtype=np.float32) # Cooperative
    #     # return np.array([-1, -5, +20, +15, +20, +20], dtype=np.float32) # Fair
    #     # return np.array([-1, -5, +20,   0, +20, +20], dtype=np.float32) # Generous


class MovingAverage(deque):
    def mean(self):
        return sum(self) / len(self)


SEED = 42
IMAGE = True

BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 6_000

GAMMA = 0.99
ALPHA = 1e-4

TRAINING_EPISODES = 80_000

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 1 / (25000 * 0.85)

COPY_TO_TARGET_EVERY = 1000 # Steps
START_TRAINING_AFTER = 50 # Episodes
MEAN_REWARD_EVERY = 300 # Episodes

FRAME_STACK_SIZE = 3

now = datetime.now()
date_and_time = f'{now.year}-{now.month}-{now.day}_{now.hour}_{now.minute}'
PATH_ID = 'tunable_' + str(date_and_time)
PATH_DIR = './'
VIDEO_DIR = PATH_DIR + 'videos/' + str(date_and_time) + '/'

LOGS_DIR = PATH_DIR + 'logs/' + str(date_and_time) + '/'
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
log_file = open(f'{LOGS_DIR}output.txt', 'w')
sys.stdout = log_file


steps = 0 # Messy but it's basically operating as a static variable anyways

NUM_WEIGHTS = 4

class DQNAgent:

    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.actions = [i for i in range(env.action_space.n)]

        self.gamma = GAMMA # Discount
        self.eps0 = 1.0 # Epsilon greedy init

        self.batch_size = BATCH_SIZE
        self.replay_memory = ReplayMemory(maxlen=REPLAY_MEMORY_SIZE)
        self.reward_tracker = RewardTracker(maxlen=MEAN_REWARD_EVERY)

        if IMAGE:
            image_size = env.observation_space.shape
            self.input_size = (*image_size[:2],image_size[-1]*FRAME_STACK_SIZE)
        else:
            self.input_size = env.observation_space.shape

        self.output_size = env.action_space.n

        # Build both models
        self.model = self.build_model()
        self.target_model = self.build_model()
        # Make weights the same
        self.target_model.set_weights(self.model.get_weights())

        self.learning_plot_initialised = False

    def build_model(self):
        """
        Construct the DQN model.
        """
        if IMAGE:
            # image of size 8x8 with 3 channels (RGB)
            image_input = Input(shape=self.input_size)
            # preference weights
            weights_input = Input(shape=(NUM_WEIGHTS,))

            # Define Layers
            x = image_input
            x = Conv2D(256, (3, 3), activation='relu')(x)
            x = Dropout(0.2)(x)
            x = Conv2D(256, (3, 3), activation='relu')(x)
            x = Dropout(0.2)(x)
            x = Flatten()(x)
            x = Concatenate()([x, weights_input])
            x = Dense(64, activation='relu')(x)
            x = Dense(64, activation='relu')(x)

            x = Dense(env.action_space.n)(x)
            outputs = x

            # Build full model
            model = keras.Model(inputs=[image_input, weights_input], outputs=outputs)

            # Define optimizer and loss function
            self.optimizer = keras.optimizers.Adam(lr=ALPHA)
            self.loss_fn = keras.losses.Huber()
        else:
            # agent locations and distances between them
            state_input = Input(shape=self.input_size)
            # preference weights
            weights_input = Input(shape=(NUM_WEIGHTS,))

            # Define Layers
            x = Concatenate()([state_input, weights_input])
            x = Dense(64, activation='relu')(x)
            x = Dense(64, activation='relu')(x)
            x = Dense(env.action_space.n)(x)
            outputs = x

            # Build full model
            model = keras.Model(inputs=[state_input, weights_input], outputs=outputs)

            # Define optimizer and loss function
            self.optimizer = keras.optimizers.Adam(lr=ALPHA)
            self.loss_fn = keras.losses.Huber()

        return model

    def epsilon_greedy_policy(self, state, epsilon, weights):
        """
        Select greedy action from model output based on current state with
        probability epsilon. With probability 1 - epsilon select random action.
        """
        if np.random.rand() < epsilon:
            return random.choice(self.actions)
        else:
            Q_values = self.model([state[np.newaxis], weights[np.newaxis]])
            return np.argmax(Q_values)

    def training_step(self):
        """
        Train the DQN on a batch from the replay buffer.
        Adapted from:
            https://github.com/ageron/handson-ml2/blob/master/18_reinforcement_learning.ipynb
            [Accessed: 15/06/2020]
        """
        # Sample a batch of S A R S' from replay memory
        experiences = self.replay_memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones, weightss = experiences

        # Compute target Q values from 'next_states'
        next_Q_values = self.target_model.predict([next_states, weightss])

        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards +
                           (1 - dones) * self.gamma * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1) # Make column vector

        # Mask to only consider action taken
        mask = tf.one_hot(actions, self.output_size) # Number of actions
        # Compute loss and gradient for predictions on 'states'
        with tf.GradientTape() as tape:
            all_Q_values = self.model([states, weightss])
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1,
                                     keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        # Apply gradients
        self.optimizer.apply_gradients(zip(grads,
                                           self.model.trainable_variables))

    def load_model(self, path):
        self.model = keras.models.load_model(path)
        self.target_model = keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def plot_learning_curve(self, image_path=None, csv_path=None):
        """
        Plot the rewards per episode collected during training
        """

        colour_palette = get_cmap(name='Set1').colors
        if self.learning_plot_initialised == False:
            self.fig, self.ax = plt.subplots()
            self.learning_plot_initialised = True
        self.ax.clear()

        reward_data = self.reward_tracker.get_reward_data()
        x = reward_data[:,0]
        y = reward_data[:,1]

        # Save raw reward data
        if csv_path:
            np.savetxt(csv_path, reward_data, delimiter=",")

        # Compute moving average
        tracker = MovingAverage(maxlen=MEAN_REWARD_EVERY)
        mean_rewards = np.zeros(len(reward_data))
        for i, (_, reward) in enumerate(reward_data):
            tracker.append(reward)
            mean_rewards[i] = tracker.mean()

        # Create plot
        self.ax.plot(x, y, alpha=0.2, c=colour_palette[0])
        self.ax.plot(x[MEAN_REWARD_EVERY//2:], mean_rewards[MEAN_REWARD_EVERY//2:],
                     c=colour_palette[0])
        self.ax.set_xlabel('episode')
        self.ax.set_ylabel('reward per episode')
        self.ax.grid(True, ls=':')

        # Save plot
        if image_path:
            self.fig.savefig(image_path)


def training_episode(render=False):
    # Decay epsilon
    eps = max(EPSILON_START - episode * EPSILON_DECAY, EPSILON_END)

    # Sample preference space
    pref1 = pref_space.sample()
    pref2 = pref_space.sample()
    weights1 = pref1
    weights2 = pref2

    # Reset env
    observations = env.reset()
    agent1_state = observations['agent-0']
    agent2_state = observations['agent-1']

    if IMAGE:
        # Create deque for storing stack of N frames
        # Agent 1
        agent1_initial_stack = [agent1_state for _ in range(FRAME_STACK_SIZE)]
        agent1_frame_stack = deque(agent1_initial_stack, maxlen=FRAME_STACK_SIZE)
        agent1_state = np.concatenate(agent1_frame_stack, axis=2) # State is now a stack of frames
        # Agent 2
        agent2_initial_stack = [agent2_state for _ in range(FRAME_STACK_SIZE)]
        agent2_frame_stack = deque(agent2_initial_stack, maxlen=FRAME_STACK_SIZE)
        agent2_state = np.concatenate(agent2_frame_stack, axis=2) # State is now a stack of frames
    else:
        # Normalise states between 0 and 1
        agent1_state = agent1_state / max_state
        agent2_state = agent2_state / max_state


    episode_reward = np.zeros(N_AGENT)

    while True:
        # Get actions
        agent1_action = agent1.epsilon_greedy_policy(agent1_state, eps, weights1)
        agent2_action = agent2.epsilon_greedy_policy(agent2_state, eps, weights2)
        # actions = [agent1_action]
        actions = [agent1_action, agent2_action]

        # Take actions, observe next states and rewards
        next_observations, reward_vectors, done, _ = env.step(actions)
        next_agent1_state = next_observations['agent-0']
        next_agent2_state = next_observations['agent-1']
        # next_agent1_state, next_agent2_state = next_observations
        agent1_rewards = reward_vectors['agent-0']
        agent2_rewards = reward_vectors['agent-1']
        # _, agent1_rewards, agent2_rewards = reward_vectors

        # A dict now
        done = done['__all__']

        # Linear scalarisation
        agent1_reward = np.dot(agent1_rewards, pref1)
        agent2_reward = np.dot(agent2_rewards, pref2)
        # rewards = [agent1_reward]
        rewards = [agent1_reward, agent2_reward]

        global steps
        # Render if true
        if render:
            if not os.path.exists(VIDEO_DIR):
                os.makedirs(VIDEO_DIR)
            frame_number = steps % 1000
            env.render(filename=VIDEO_DIR + str(frame_number))

        # Store in replay buffers
        # Agent 1
        if IMAGE:
            agent1_frame_stack.append(next_agent1_state)
            next_agent1_state = np.concatenate(agent1_frame_stack, axis=2)
        else:
            next_agent1_state = next_agent1_state / max_state
        agent1.replay_memory.append((agent1_state, agent1_action, agent1_reward, next_agent1_state, done, weights1))

        # Store in replay buffers
        # Agent 2
        if IMAGE:
            agent2_frame_stack.append(next_agent2_state)
            next_agent2_state = np.concatenate(agent2_frame_stack, axis=2)
        else:
            next_agent2_state = next_agent2_state / max_state
        agent2.replay_memory.append((agent2_state, agent2_action, agent2_reward, next_agent2_state, done, weights2))

        # Assign next state to current state !!
        agent1_state = next_agent1_state
        agent2_state = next_agent2_state

        steps += 1
        episode_reward += np.array(rewards)

        if done:
            break

        # Copy weights from main model to target model
        if steps % COPY_TO_TARGET_EVERY == 0:
            agent1.update_target_model()
            agent2.update_target_model()

    agent1.reward_tracker.append(episode_reward[agent1.agent_id-1])
    agent2.reward_tracker.append(episode_reward[agent2.agent_id-1])

    ep_rewards = [np.round(episode_reward[agent1.agent_id-1], 2),
                  np.round(episode_reward[agent2.agent_id-1], 2)]
    av_rewards = [np.round(agent1.reward_tracker.mean(), 2),
                  np.round(agent2.reward_tracker.mean(), 2)]
    # print("\rEpisode: {}, Time: {}, Reward1: {}, Avg Reward1: {}, eps: {:.3f}".format(
    #     episode, datetime.now() - start_time, ep_rewards[0], av_rewards[0], eps), end="")
    print("\rEpisode: {}, Time: {}, Reward1: {}, Reward2: {}, Avg Reward1: {}, Avg Reward2: {}, eps: {:.3f}".format(
        episode, datetime.now() - start_time, ep_rewards[0], ep_rewards[1],  av_rewards[0], av_rewards[1], eps), end="")

    if episode > START_TRAINING_AFTER: # Wait for buffer to fill up a bit
        agent1.training_step()
        agent2.training_step()

    if episode % 250 == 0:
        agent1.model.save(f'{PATH_DIR}/models/cleanup_model_agent1_{PATH_ID}.h5')
        agent1.plot_learning_curve(image_path=f'{PATH_DIR}/plots/cleanup_plot_agent1_{PATH_ID}.png',
                                   csv_path=f'{PATH_DIR}/plots/cleanup_rewards_agent1_{PATH_ID}.csv')
        agent2.model.save(f'{PATH_DIR}/models/cleanup_model_agent2_{PATH_ID}.h5')
        agent2.plot_learning_curve(image_path=f'{PATH_DIR}/plots/cleanup_plot_agent2_{PATH_ID}.png',
                                   csv_path=f'{PATH_DIR}/plots/cleanup_rewards_agent2_{PATH_ID}.csv')


N_AGENT = 2
env = CleanupEnv(num_agents=N_AGENT)

if __name__ == '__main__':

    PATH_DIR = './'

    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)

    # For timing the script
    start_time = datetime.now()

    # Initialise agents
    # prey1 = DQNAgent(0)
    agent1 = DQNAgent(1)
    agent2 = DQNAgent(2)

    # Uncomment to load pre-trained models
    # agent1.load_model(f'{PATH_DIR}/models/wolfpack_model_tunable_agent1_seed1.h5')
    # agent2.load_model(f'{PATH_DIR}/models/wolfpack_model_tunable_agent2_seed1.h5')

    if not IMAGE:
        x, y = env.base_gridmap_array.shape[0] - 1, env.base_gridmap_array.shape[1] - 1
        max_state = np.array([x, y, *[z for _ in range(2) for z in [x, y, x+y]]], dtype=np.float32)

    pref_space = PreferenceSpace()

    for episode in range(1, TRAINING_EPISODES):
        training_episode() # Don't save images/render
    training_episode(True) # Render last image

    agent1.model.save(f'{PATH_DIR}/models/cleanup_model_agent1_{PATH_ID}.h5')
    agent1.plot_learning_curve(image_path=f'{PATH_DIR}/plots/cleanup_plot_agent1_{PATH_ID}.png',
                              csv_path=f'{PATH_DIR}/plots/cleanup_rewards_agent1_{PATH_ID}.csv')
    agent2.model.save(f'{PATH_DIR}/models/cleanup_model_agent2_{PATH_ID}.h5')
    agent2.plot_learning_curve(image_path=f'{PATH_DIR}/plots/cleanup_plot_agent2_{PATH_ID}.png',
                              csv_path=f'{PATH_DIR}/plots/cleanup_rewards_agent2_{PATH_ID}.csv')


    run_time = datetime.now() - start_time
    print(f'\nRun time: {run_time} s')