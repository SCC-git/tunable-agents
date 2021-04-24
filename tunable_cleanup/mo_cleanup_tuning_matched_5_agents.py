# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 09:07:50 2020

Author: David O'Callaghan
"""

import numpy as np
import random
import pandas as pd
import sys
import os
from datetime import datetime # Used for timing script

import tensorflow as tf


from collections import deque # Used for replay buffer and reward tracking

from mo_cleanup import DQNAgent, env, FRAME_STACK_SIZE, N_AGENT

PATH_DIR = "./models/"

MODEL_PATHS = [f'{PATH_DIR}/cleanup_model_agent1_tunable_2021-4-14_11_4.h5',
               f'{PATH_DIR}/cleanup_model_agent2_tunable_2021-4-14_11_4.h5']

now = datetime.now()
date_and_time = f'{now.year}-{now.month}-{now.day}_{now.hour}_{now.minute}'

LOGS_DIR = './logs/' + str(date_and_time) + '/'
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
log_file = open(f'{LOGS_DIR}output.txt', 'w')
sys.stdout = log_file

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

    prefs = np.linspace(0, 0.4, 17)
    results = []
    for pref in prefs:

        steps = 0

        agent1_apples = 0
        agent2_apples = 0
        agent3_apples = 0
        agent4_apples = 0
        agent5_apples = 0
        agent1_cleans = 0
        agent2_cleans = 0
        agent3_cleans = 0
        agent4_cleans = 0
        agent5_cleans = 0

        agent1_total_apples = 0
        agent2_total_apples = 0
        agent3_total_apples = 0
        agent4_total_apples = 0
        agent5_total_apples = 0
        agent1_total_cleans = 0
        agent2_total_cleans = 0
        agent3_total_cleans = 0
        agent4_total_cleans = 0
        agent5_total_cleans = 0

        pref1 = np.array([0.05, 0.55, pref, 0.4 - pref], dtype=np.float32)
        pref2 = pref1.copy()
        pref3 = pref1.copy()
        pref4 = pref1.copy()
        pref5 = pref1.copy()

        weights1 = pref1
        weights2 = pref2
        weights3 = pref3
        weights4 = pref4
        weights5 = pref5

        print(f'\n\n{np.round(pref1, 3)}\n-----------------')

        for episode in range(1, EPISODES+1):

            agent1_apples = 0
            agent2_apples = 0
            agent3_apples = 0
            agent4_apples = 0
            agent5_apples = 0
            agent1_cleans = 0
            agent2_cleans = 0
            agent3_cleans = 0
            agent4_cleans = 0
            agent5_cleans = 0

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

                # Get actions
                agent1_action = agent1.epsilon_greedy_policy(agent1_state, eps, weights1)
                agent2_action = agent2.epsilon_greedy_policy(agent2_state, eps, weights2)
                agent3_action = agent3.epsilon_greedy_policy(agent3_state, eps, weights3)
                agent4_action = agent4.epsilon_greedy_policy(agent4_state, eps, weights4)
                agent5_action = agent5.epsilon_greedy_policy(agent5_state, eps, weights5)
                actions = [agent1_action, agent2_action]


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

                # Linear scalarisation
                agent1_reward = np.dot(agent1_rewards, pref1)
                agent2_reward = np.dot(agent2_rewards, pref2)
                agent3_reward = np.dot(agent3_rewards, pref3)
                agent4_reward = np.dot(agent4_rewards, pref4)
                agent5_reward = np.dot(agent5_rewards, pref5)
                rewards = [agent1_reward, agent2_reward, agent3_reward, agent4_reward, agent5_reward]

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

                if agent1_rewards[2]:
                    agent1_apples += 1
                if agent2_rewards[2]:
                    agent2_apples += 1
                if agent3_rewards[2]:
                    agent3_apples += 1
                if agent4_rewards[2]:
                    agent4_apples += 1
                if agent5_rewards[2]:
                    agent5_apples += 1
                if agent1_rewards[3]:
                    agent1_cleans += 1
                if agent2_rewards[3]:
                    agent2_cleans += 1
                if agent3_rewards[3]:
                    agent3_cleans += 1
                if agent4_rewards[3]:
                    agent4_cleans += 1
                if agent5_rewards[3]:
                    agent5_cleans += 1

                if done:
                    print(f'\rEp {episode}: Agent1_Apples: {agent1_apples},  Agent2_Apples: {agent2_apples}, Agent3_Apples: {agent3_apples}, Agent4_Apples: {agent4_apples}, Agent5_Apples: {agent5_apples},  '
                          f'Agent1_Cleans: {agent1_cleans},  Agent2_Cleans: {agent2_cleans},  Agent3_Cleans: {agent3_cleans},  Agent4_Cleans: {agent4_cleans},  Agent5_Cleans: {agent5_cleans}', end='')
                    break

            agent1_total_apples += agent1_apples
            agent2_total_apples += agent2_apples
            agent3_total_apples += agent3_apples
            agent4_total_apples += agent4_apples
            agent5_total_apples += agent5_apples
            agent1_total_cleans += agent1_cleans
            agent2_total_cleans += agent2_cleans
            agent3_total_cleans += agent3_cleans
            agent4_total_cleans += agent4_cleans
            agent5_total_cleans += agent5_cleans

        results.append([pref1[2], pref1[3], agent1_total_apples, agent2_total_apples, agent3_total_apples, agent4_total_apples, agent5_total_apples,
                        agent1_total_cleans, agent2_total_cleans, agent3_total_cleans, agent4_total_cleans, agent5_total_cleans])

    results = pd.DataFrame(results, columns = ["Competitive (Apple)", "Cooperative (Clean)", "Agent 1 Apple", "Agent 2 Apple", "Agent 3 Apple", "Agent 4 Apple", "Agent 5 Apple",
                                               "Agent 1 Clean", "Agent 2 Clean", "Agent 3 Clean", "Agent 4 Clean", "Agent 5 Clean"])

    results.to_csv(f'./results/cleanup_tuning_matched_prefs_{date_and_time}.csv')
