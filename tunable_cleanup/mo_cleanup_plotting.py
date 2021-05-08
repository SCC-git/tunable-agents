# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 09:35:46 2020

Author: David O'Callaghan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib import style
from collections import deque

style.use('ggplot')
colour_palette = get_cmap(name='tab10').colors

data_dir = './plots/'

MEAN_EVERY = 100

fig, ax = plt.subplots()


class MovingAverage(deque):
    def mean(self):
        return sum(self) / len(self)


def plot_reward_data(csv_path, colour_id, legend_label):
    reward_data = np.loadtxt(csv_path, delimiter=',')
    mean_rewards = np.zeros(len(reward_data))
    tracker = MovingAverage(maxlen=MEAN_EVERY)

    for j, (_, reward) in enumerate(reward_data):
        tracker.append(reward)
        mean_rewards[j] = tracker.mean()

    ax.plot(reward_data[MEAN_EVERY//2:80000,0], mean_rewards[MEAN_EVERY//2:80000],
            c=colour_palette[colour_id], label=legend_label, alpha=0.8)

def plot_collective_reward(csv_paths, colour_id, legend_label):
    reward_data = [np.loadtxt(csv_path, delimiter=',') for csv_path in csv_paths]

    mean_rewards = np.zeros(len(reward_data[0]))
    tracker = MovingAverage(maxlen=MEAN_EVERY)

    y = list(map(sum, zip(*[[reward for _, reward in a_reward_data] for a_reward_data in reward_data])))

    summed_rewards = list(map(sum, zip(*y)))

    for j, summed_reward in enumerate(summed_rewards):
        print(summed_reward)
        tracker.append(summed_reward)
        mean_rewards[j] = tracker.mean()

    ax.plot(reward_data[0][MEAN_EVERY//2:1000,0], mean_rewards[MEAN_EVERY//2:1000],
            c=colour_palette[colour_id], label=legend_label, alpha=0.8)


if __name__ == '__main__':
    # PLOT 1 :: TRAINING PROGRESS
    # Wolfpack tunable
    plot_reward_data(f'{data_dir}/cleanup_rewards_dqn1_single_2021-5-7_10_31.csv', 0, 'Agent 1')
    # plot_reward_data(f'{data_dir}/cleanup_rewards_dqn2_2021-4-24_13_38.csv', 1, 'Agent 2')
    # plot_reward_data(f'{data_dir}/cleanup_rewards_dqn3_2021-4-24_13_38.csv', 2, 'Agent 3')
    # plot_reward_data(f'{data_dir}/cleanup_rewards_dqn4_2021-4-24_13_38.csv', 3, 'Agent 4')
    # plot_reward_data(f'{data_dir}/cleanup_rewards_dqn5_2021-4-24_13_38.csv', 4, 'Agent 5')

    # plot_collective_reward([f'{data_dir}/cleanup_rewards_dqn1_2021-4-24_13_38.csv',
    #                        f'{data_dir}/cleanup_rewards_dqn2_2021-4-24_13_38.csv',
    #                        f'{data_dir}/cleanup_rewards_dqn3_2021-4-24_13_38.csv',
    #                        f'{data_dir}/cleanup_rewards_dqn4_2021-4-24_13_38.csv',
    #                        f'{data_dir}/cleanup_rewards_dqn5_2021-4-24_13_38.csv'],
    #                        0, 'Collective Reward')

    # Wolpack fixed
    # plot_reward_data(f'{data_dir}/wolfpack_rewards_fixed_competitive.csv', 0, 'Fixed Competitive')
    # plot_reward_data(f'{data_dir}/wolfpack_rewards_fixed_cooperative.csv', 1, 'Fixed Cooperative')

    # ax.set_ylim([-110,150])
    ax.set_xlabel('Episode')
    ax.set_ylabel(f'Mean {MEAN_EVERY} Episode Reward')
    ax.grid(True, ls=':', c='dimgrey')
    ax.set_facecolor('white')
    ax.legend(facecolor='white')
    ax.set_xticks(np.arange(0, 1001, step=20000))
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    plt.show()


    # # PLOT 2 :: TUNING PERFORMANCE WITH MATCHED PREFERENCES
    # results = pd.read_csv(f'./results/cleanup_tuning_matched_prefs_200421.csv')
    # EPISODES = 250
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    # ax[0].plot(results['Cooperative (Clean)'], results['Agent 1 Clean'] / EPISODES, 'D--',
    #            c=colour_palette[0], label='Agent 1', alpha=1.0)
    # ax[0].plot(results['Cooperative (Clean)'], results['Agent 2 Clean'] / EPISODES, 's--',
    #            c=colour_palette[1], label='Agent 2', alpha=1.0)
    # ax[0].set_xlabel('Cooperativeness')
    # ax[0].set_ylabel('Cleaning Rate')
    # ax[0].grid(True, ls=':', c='dimgrey')
    # ax[0].set_facecolor('white')
    # # a[1].legend(facecolor='white')
    # ax[0].xaxis.set_ticks_position('none')
    # ax[0].yaxis.set_ticks_position('none')
    #
    # ax[1].plot(results['Competitive (Apple)'], results['Agent 1 Apple'] / EPISODES, 'D--',
    #            c=colour_palette[0], label='Agent 1', alpha=1.0)
    # ax[1].plot(results['Competitive (Apple)'], results['Agent 2 Apple'] / EPISODES, 's--',
    #            c=colour_palette[1], label='Agent 2', alpha=1.0)
    # ax[1].set_xlabel('Competitiveness')
    # ax[1].set_ylabel('Apple Collection Rate')
    # ax[1].grid(True, ls=':', c='dimgrey')
    # ax[1].set_facecolor('white')
    # ax[1].legend(facecolor='white')
    # ax[1].xaxis.set_ticks_position('none')
    # ax[1].yaxis.set_ticks_position('none')
    # plt.show()
    #
    #
    # # PLOT 3 :: TUNING PERFORMANCE WITH 3 PREDATORS
    # results = pd.read_csv(f'./results/wolfpack_tuning_3predator.csv')
    #
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    # ax[0].plot(results['Cooperative'], results['Pair Capture'] / EPISODES, 'D--',
    #         c=colour_palette[1], label='2 Predator Team', alpha=1.0)
    # ax[0].plot(results['Cooperative'], results['Team Capture'] / EPISODES, 's--',
    #         c=colour_palette[2], label='3 Predator Team', alpha=1.0)
    # ax[0].set_xlabel('Cooperativeness')
    # ax[0].set_ylabel('Team Capture Rate')
    # ax[0].grid(True, ls=':', c='dimgrey')
    # ax[0].set_facecolor('white')
    # ax[0].legend(facecolor='white')
    # ax[0].xaxis.set_ticks_position('none')
    # ax[0].yaxis.set_ticks_position('none')
    #
    # ax[1].plot(results['Competitive'], results['Lone Capture'] / EPISODES, 'o--',
    #         c=colour_palette[0], label='Single Predator Capture', alpha=1.0)
    # ax[1].set_xlabel('Competitiveness')
    # ax[1].set_ylabel('Lone Capture Rate')
    # ax[1].grid(True, ls=':', c='dimgrey')
    # ax[1].set_facecolor('white')
    # ax[1].legend(facecolor='white')
    # ax[1].xaxis.set_ticks_position('none')
    # ax[1].yaxis.set_ticks_position('none')
    # plt.show()
    #