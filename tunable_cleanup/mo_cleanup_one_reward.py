# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 09:07:50 2020

Author: David O'Callaghan
"""

import numpy as np
import random
import pandas as pd

import tensorflow as tf
import sys
import os
from datetime import datetime # Used for timing script

from collections import deque # Used for replay buffer and reward tracking

from so_cleanup import DQNAgent, env, FRAME_STACK_SIZE, N_AGENT

PATH_DIR = "./models/"

MODEL_PATHS = [f'{PATH_DIR}/cleanup_model_dqn1_single_2021-3-31_16_38.h5',
               f'{PATH_DIR}/cleanup_model_dqn2_single_2021-3-31_16_38.h5']

now = datetime.now()
date_and_time = f'{now.year}-{now.month}-{now.day}_{now.hour}_{now.minute}'

LOGS_DIR = './logs/' + str(date_and_time) + '/'
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
log_file = open(f'{LOGS_DIR}output.txt', 'w')
sys.stdout = sys.__stdout__

EPISODES = 1000

if __name__ == '__main__':

    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    # Initialise agents
    agent1 = DQNAgent(1)
    agent2 = DQNAgent(2)
    agent3 = DQNAgent(3)
    agent4 = DQNAgent(4)
    agent5 = DQNAgent(5)

    agent1.load_model(MODEL_PATHS[0])
    agent2.load_model(MODEL_PATHS[1])
    agent3.load_model(MODEL_PATHS[0])
    agent4.load_model(MODEL_PATHS[1])
    agent5.load_model(MODEL_PATHS[0])

    steps = 0

    results = []
    steps = 0


    for episode in range(1, EPISODES+1):

        # Decay epsilon
        eps = 0.01

        # Reset env
        observations = env.reset()
        agent1_state = observations['agent-0']
        agent2_state = observations['agent-1']
        agent3_state = observations['agent-2']
        agent4_state = observations['agent-3']
        agent5_state = observations['agent-4']

        # Create deque for storing stack of N frames
        # Agent 1
        agent1_initial_stack = [agent1_state for _ in range(FRAME_STACK_SIZE)]
        agent1_frame_stack = deque(agent1_initial_stack, maxlen=FRAME_STACK_SIZE)
        agent1_state = np.concatenate(agent1_frame_stack, axis=2) # State is now a stack of frames
        # Agent 2
        agent2_initial_stack = [agent2_state for _ in range(FRAME_STACK_SIZE)]
        agent2_frame_stack = deque(agent2_initial_stack, maxlen=FRAME_STACK_SIZE)
        agent2_state = np.concatenate(agent2_frame_stack, axis=2) # State is now a stack of frames
        # Agent 3
        agent3_initial_stack = [agent3_state for _ in range(FRAME_STACK_SIZE)]
        agent3_frame_stack = deque(agent3_initial_stack, maxlen=FRAME_STACK_SIZE)
        agent3_state = np.concatenate(agent3_frame_stack, axis=2) # State is now a stack of frames
        # Agent 4
        agent4_initial_stack = [agent4_state for _ in range(FRAME_STACK_SIZE)]
        agent4_frame_stack = deque(agent4_initial_stack, maxlen=FRAME_STACK_SIZE)
        agent4_state = np.concatenate(agent4_frame_stack, axis=2) # State is now a stack of frames
        # Agent 5
        agent5_initial_stack = [agent5_state for _ in range(FRAME_STACK_SIZE)]
        agent5_frame_stack = deque(agent5_initial_stack, maxlen=FRAME_STACK_SIZE)
        agent5_state = np.concatenate(agent5_frame_stack, axis=2) # State is now a stack of frames

        episode_reward = np.zeros(N_AGENT)

        while True:
            # Get actionss
            agent1_action = agent1.epsilon_greedy_policy(agent1_state, eps)
            agent2_action = agent2.epsilon_greedy_policy(agent2_state, eps)
            agent3_action = agent3.epsilon_greedy_policy(agent3_state, eps)
            agent4_action = agent4.epsilon_greedy_policy(agent4_state, eps)
            agent5_action = agent5.epsilon_greedy_policy(agent5_state, eps)
            actions = [agent1_action, agent2_action, agent3_action, agent4_action, agent5_action]

            print(actions)

            # Take actions, observe next states and rewards
            next_observations, reward_vectors, done, _ = env.step(actions)
            next_agent1_state = next_observations['agent-0']
            next_agent2_state = next_observations['agent-1']
            next_agent3_state = next_observations['agent-2']
            next_agent4_state = next_observations['agent-3']
            next_agent5_state = next_observations['agent-4']

            agent1_rewards = reward_vectors['agent-0']
            agent2_rewards = reward_vectors['agent-1']
            agent3_rewards = reward_vectors['agent-2']
            agent4_rewards = reward_vectors['agent-3']
            agent5_rewards = reward_vectors['agent-4']

            # A dict now
            done = done['__all__']

            rewards = [agent1_rewards, agent2_rewards, agent3_rewards, agent4_rewards, agent5_rewards]

            # Store in replay buffers
            # Agent 1
            agent1_frame_stack.append(next_agent1_state)
            next_agent1_state = np.concatenate(agent1_frame_stack, axis=2)
            # Agent 2
            agent2_frame_stack.append(next_agent2_state)
            next_agent2_state = np.concatenate(agent2_frame_stack, axis=2)
            # Agent 3
            agent3_frame_stack.append(next_agent3_state)
            next_agent3_state = np.concatenate(agent3_frame_stack, axis=2)
            # Agent 4
            agent4_frame_stack.append(next_agent4_state)
            next_agent4_state = np.concatenate(agent4_frame_stack, axis=2)
            # Agent 5
            agent5_frame_stack.append(next_agent5_state)
            next_agent5_state = np.concatenate(agent5_frame_stack, axis=2)

            # Assign next state to current state !!
            agent1_state = next_agent1_state
            agent2_state = next_agent2_state
            agent3_state = next_agent3_state
            agent4_state = next_agent4_state
            agent5_state = next_agent5_state

            steps += 1

            episode_reward += np.array(rewards)

            if done:
                break

        agent1.reward_tracker.append(episode_reward[agent1.agent_id-1])
        agent2.reward_tracker.append(episode_reward[agent2.agent_id-1])
        agent3.reward_tracker.append(episode_reward[agent3.agent_id-1])
        agent4.reward_tracker.append(episode_reward[agent4.agent_id-1])
        agent5.reward_tracker.append(episode_reward[agent5.agent_id-1])

        ep_rewards = [np.round(episode_reward[agent1.agent_id-1], 2),
                      np.round(episode_reward[agent2.agent_id-1], 2),
                      np.round(episode_reward[agent3.agent_id-1], 2),
                      np.round(episode_reward[agent4.agent_id-1], 2),
                      np.round(episode_reward[agent5.agent_id-1], 2)]
        av_rewards = [np.round(agent1.reward_tracker.mean(), 2),
                      np.round(agent2.reward_tracker.mean(), 2),
                      np.round(agent3.reward_tracker.mean(), 2),
                      np.round(agent4.reward_tracker.mean(), 2),
                      np.round(agent5.reward_tracker.mean(), 2)]

        print("\rEpisode: {}, Reward1: {}, Reward2: {}, Reward3: {}, Reward4: {}, Reward5: {}".format(
            episode, ep_rewards[0], ep_rewards[1],  ep_rewards[2], ep_rewards[3], ep_rewards[4]), end="", flush=True)

    agent1.plot_learning_curve(image_path=f'./plots/cleanup_plot_dqn1_single_{date_and_time}.png',
                               csv_path=f'./plots/cleanup_rewards_dqn1_{date_and_time}.csv')
    agent2.plot_learning_curve(image_path=f'./plots/cleanup_plot_dqn2_{date_and_time}.png',
                               csv_path=f'./plots/cleanup_rewards_dqn2_{date_and_time}.csv')
    agent3.plot_learning_curve(image_path=f'./plots/cleanup_plot_dqn3_{date_and_time}.png',
                               csv_path=f'./plots/cleanup_rewards_dqn3_{date_and_time}.csv')
    agent4.plot_learning_curve(image_path=f'./plots/cleanup_plot_dqn4_{date_and_time}.png',
                               csv_path=f'./plots/cleanup_rewards_dqn4_{date_and_time}.csv')
    agent5.plot_learning_curve(image_path=f'./plots/cleanup_plot_dqn5_{date_and_time}.png',
                               csv_path=f'./plots/cleanup_rewards_dqn5_{date_and_time}.csv')
