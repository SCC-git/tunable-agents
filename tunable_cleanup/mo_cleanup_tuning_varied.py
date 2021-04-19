# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 09:07:50 2020

Author: David O'Callaghan
"""

import numpy as np
import random
import pandas as pd

import tensorflow as tf

from collections import deque # Used for replay buffer and reward tracking

from mo_cleanup import DQNAgent, env, FRAME_STACK_SIZE, N_AGENT

PATH_DIR = "./models/"

MODEL_PATHS = [f'{PATH_DIR}/cleanup_model_agent1_tunable_2021-4-14_11_4.h5',
               f'{PATH_DIR}/cleanup_model_agent2_tunable_2021-4-14_11_4.h5']

EPISODES = 250

if __name__ == '__main__':

    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    # Initialise agents
    agent1 = DQNAgent(1)
    agent2 = DQNAgent(2)

    agent1.load_model(MODEL_PATHS[0])
    agent2.load_model(MODEL_PATHS[1])

    steps = 0

    prefs = np.linspace(0, 0.4, 5)
    results = []
    for pref_a in prefs:
        for pref_b in prefs:

            steps = 0

            agent1_apples = 0
            agent2_apples = 0
            agent1_cleans = 0
            agent2_cleans = 0

            pref1 = np.array([0.05, 0.55, pref_a, 0.4 - pref_a], dtype=np.float32)
            pref2 = np.array([0.05, 0.55, pref_b, 0.4 - pref_b], dtype=np.float32)

            weights1 = pref1
            weights2 = pref2

            print(f'\n\n{np.round(pref1, 3)}\n-----------------')

            for episode in range(1, EPISODES+1):

                agent1_apples = 0
                agent2_apples = 0
                agent1_cleans = 0
                agent2_cleans = 0

                # Decay epsilon
                eps = 0.01
                # Reset env
                observations = env.reset()
                #env.render()
                agent1_state = observations['agent-0']
                agent2_state = observations['agent-1']

                # Create deque for storing stack of N frames
                # Agent 1
                agent1_initial_stack = [agent1_state for _ in range(FRAME_STACK_SIZE)]
                agent1_frame_stack = deque(agent1_initial_stack, maxlen=FRAME_STACK_SIZE)
                agent1_state = np.concatenate(agent1_frame_stack, axis=2) # State is now a stack of frames
                # Agent 2
                agent2_initial_stack = [agent2_state for _ in range(FRAME_STACK_SIZE)]
                agent2_frame_stack = deque(agent2_initial_stack, maxlen=FRAME_STACK_SIZE)
                agent2_state = np.concatenate(agent2_frame_stack, axis=2) # State is now a stack of frames

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

                    # Store in replay buffers
                    # Agent 1
                    agent1_frame_stack.append(next_agent1_state)
                    next_agent1_state = np.concatenate(agent1_frame_stack, axis=2)

                    # Agent 2
                    agent2_frame_stack.append(next_agent2_state)
                    next_agent2_state = np.concatenate(agent2_frame_stack, axis=2)

                    # Assign next state to current state !!
                    agent1_state = next_agent1_state
                    agent2_state = next_agent2_state

                    steps += 1
                    episode_reward += np.array(rewards)

                    if agent1_rewards[2]:
                        agent1_apples += 1
                    if agent2_rewards[2]:
                        agent2_apples += 1
                    if agent1_rewards[3]:
                        agent1_cleans += 1
                    if agent2_rewards[3]:
                        agent2_cleans += 1

                    if done:
                        print(f'\rEp {episode}: Agent1_Apples: {agent1_apples},  Agent2_Apples: {agent2_apples},  Agent1_Cleans: {agent1_cleans},  Agent2_Cleans: {agent2_cleans}', end='')
                        break

            results.append([pref1[2], pref2[2], agent1_apples, agent2_apples, agent1_cleans, agent2_cleans])


    results = pd.DataFrame(results, columns = ["Competitive 1", "Competitive 2", "Agent 1 Apple", "Agent 2 Apple", "Agent 1 Clean", "Agent 2 Clean"])

    results.to_csv(f'./results/cleanup_tuning_varied_prefs_190421.csv')