# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import random
import os
import gym

from gym.spaces import Box
from gym.spaces import Discrete

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

from collections import deque # Used for replay buffer and reward tracking
from datetime import datetime # Used for timing script


from line_profiler import LineProfiler

ACTIONS = {'MOVE_LEFT': [-1, 0],  # Move left
           'MOVE_RIGHT': [1, 0],  # Move right
           'MOVE_UP': [0, -1],  # Move up
           'MOVE_DOWN': [0, 1],  # Move down
           'STAY': [0, 0],  # don't move
           'TURN_CLOCKWISE': [[0, -1], [1, 0]],  # Rotate counter clockwise
           'TURN_COUNTERCLOCKWISE': [[0, 1], [-1, 0]]}  # Move right

ORIENTATIONS = {'LEFT': [-1, 0],
                'RIGHT': [1, 0],
                'UP': [0, -1],
                'DOWN': [0, 1]}

DEFAULT_COLOURS = {' ': [0, 0, 0],  # Black background
                   '0': [0, 0, 0],  # Black background beyond map walls
                   '': [180, 180, 180],  # Grey board walls
                   '@': [180, 180, 180],  # Grey board walls
                   'A': [0, 255, 0],  # Green apples
                   'F': [255, 255, 0],  # Yellow fining beam
                   'P': [159, 67, 255],  # Purple player

                   # Colours for agents. R value is a unique identifier
                   '1': [159, 67, 255],  # Purple
                   '2': [2, 81, 154],  # Blue
                   '3': [204, 0, 204],  # Magenta
                   '4': [216, 30, 54],  # Red
                   '5': [254, 151, 0],  # Orange
                   '6': [100, 255, 255],  # Cyan
                   '7': [99, 99, 255],  # Lavender
                   '8': [250, 204, 255],  # Pink
                   '9': [238, 223, 16]}  # Yellow

# the axes look like
# graphic is here to help me get my head in order
# WARNING: increasing array position in the direction of down
# so for example if you move_left when facing left
# your y position decreases.
#         ^
#         |
#         U
#         P
# <--LEFT*RIGHT---->
#         D
#         O
#         W
#         N
#         |


class MapEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    MAX_STEPS = 1000

    def __init__(self, ascii_map, num_agents=1, render=True, color_map=None):
        """

        Parameters
        ----------
        ascii_map: list of strings
            Specify what the map should look like. Look at constant.py for
            further explanation
        num_agents: int
            Number of agents to have in the system.
        render: bool
            Whether to render the environment
        color_map: dict
            Specifies how to convert between ascii chars and colors
        """
        self.num_agents = num_agents
        self.base_map = self.ascii_to_numpy(ascii_map)
        # map without agents or beams
        self.world_map = np.full((len(self.base_map), len(self.base_map[0])), ' ')
        self.beam_pos = []

        self.agents = {}

        self.n_steps = 0

        # returns the agent at a desired position if there is one
        self.pos_dict = {}
        self.color_map = color_map if color_map is not None else DEFAULT_COLOURS
        self.spawn_points = []  # where agents can appear

        self.wall_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == 'P':
                    self.spawn_points.append([row, col])
                elif self.base_map[row, col] == '@':
                    self.wall_points.append([row, col])
        self.setup_agents()

    def custom_reset(self):
        """Reset custom elements of the map. For example, spawn apples and build walls"""
        pass

    def custom_action(self, agent, action):
        """Execute any custom actions that may be defined, like fire or clean

        Parameters
        ----------
        agent: agent that is taking the action
        action: key of the action to be taken

        Returns
        -------
        updates: list(list(row, col, char))
            List of cells to place onto the map
        """
        pass

    def custom_map_update(self):
        """Custom map updates that don't have to do with agent actions"""
        pass

    def setup_agents(self):
        """Construct all the agents for the environment"""
        raise NotImplementedError

    # FIXME(ev) move this to a utils eventually
    def ascii_to_numpy(self, ascii_list):
        """converts a list of strings into a numpy array


        Parameters
        ----------
        ascii_list: list of strings
            List describing what the map should look like
        Returns
        -------
        arr: np.ndarray
            numpy array describing the map with ' ' indicating an empty space
        """

        arr = np.full((len(ascii_list), len(ascii_list[0])), ' ')
        for row in range(arr.shape[0]):
            for col in range(arr.shape[1]):
                arr[row, col] = ascii_list[row][col]
        return arr

    def step(self, actions):
        """Takes in a dict of actions and converts them to a map update

        Parameters
        ----------
        actions: dict {agent-id: int}
            dict of actions, keyed by agent-id that are passed to the agent. The agent
            interprets the int and converts it to a command

        Returns
        -------
        observations: dict of arrays representing agent observations
        rewards: dict of rewards for each agent
        dones: dict indicating whether each agent is done
        info: dict to pass extra info to gym
        """

        self.beam_pos = []
        agent_actions = {}
        for agent_id, action in zip(self.agents, actions):
            agent_action = self.agents[agent_id].action_map(action)
            agent_actions[agent_id] = agent_action

        # move
        self.update_moves(agent_actions)

        for agent in self.agents.values():
            pos = agent.get_pos()
            new_char = agent.consume(self.world_map[pos[0], pos[1]])
            self.world_map[pos[0], pos[1]] = new_char

        # execute custom moves like firing
        self.update_custom_moves(agent_actions)

        # lp = LineProfiler()
        # lp.add_function(self.update_map)
        # lp.add_function(self.spawn_apples_and_waste)
        # lp_wrapper = lp(self.custom_map_update)
        # lp_wrapper()
        # lp.print_stats()
        # execute spawning events
        # self.custom_map_update()

        map_with_agents = self.get_map_with_agents()

        observations = {}
        rewards = {}
        dones = {}
        info = {}
        for agent in self.agents.values():
            agent.grid = map_with_agents
            rgb_arr = self.map_to_colors(agent.get_state(), self.color_map)
            rgb_arr = self.rotate_view(agent.orientation, rgb_arr)
            observations[agent.agent_id] = rgb_arr
            rewards[agent.agent_id] = agent.compute_reward()
            # dones[agent.agent_id] = agent.get_done()
        # dones["__all__"] = np.any(list(dones.values()))

        self.n_steps += 1
        dones['__all__'] = self.n_steps >= self.MAX_STEPS
        return observations, rewards, dones, info

    def reset(self):
        """Reset the environment.

        This method is performed in between rollouts. It resets the state of
        the environment.

        Returns
        -------
        observation: dict of numpy ndarray
            the initial observation of the space. The initial reward is assumed
            to be zero.
        """
        self.n_steps = 0
        self.beam_pos = []
        self.agents = {}
        self.setup_agents()
        self.reset_map()
        self.custom_map_update()

        map_with_agents = self.get_map_with_agents()

        observations = {}
        for agent in self.agents.values():
            agent.grid = map_with_agents
            # agent.grid = util.return_view(map_with_agents, agent.pos,
            #                               agent.row_size, agent.col_size)
            rgb_arr = self.map_to_colors(agent.get_state(), self.color_map)
            observations[agent.agent_id] = rgb_arr
        return observations

    @property
    def agent_pos(self):
        return [agent.get_pos().tolist() for agent in self.agents.values()]

    # This method is just used for testing
    # FIXME(ev) move into the testing class
    @property
    def test_map(self):
        """Gets a version of the environment map where generic
        'P' characters have been replaced with specific agent IDs.

        Returns:
            2D array of strings representing the map.
        """
        grid = np.copy(self.world_map)

        for agent_id, agent in self.agents.items():
            # If agent is not within map, skip.
            if not (agent.pos[0] >= 0 and agent.pos[0] < grid.shape[0] and
                    agent.pos[1] >= 0 and agent.pos[1] < grid.shape[1]):
                continue

            grid[agent.pos[0], agent.pos[1]] = 'P'

        for beam_pos in self.beam_pos:
            grid[beam_pos[0], beam_pos[1]] = beam_pos[2]

        return grid

    def get_map_with_agents(self):
        """Gets a version of the environment map where generic
        'P' characters have been replaced with specific agent IDs.

        Returns:
            2D array of strings representing the map.
        """
        grid = np.copy(self.world_map)

        for agent_id, agent in self.agents.items():
            char_id = str(int(agent_id[-1]) + 1)

            # If agent is not within map, skip.
            if not(agent.pos[0] >= 0 and agent.pos[0] < grid.shape[0] and
                   agent.pos[1] >= 0 and agent.pos[1] < grid.shape[1]):
                continue

            grid[agent.pos[0], agent.pos[1]] = char_id

        for beam_pos in self.beam_pos:
            grid[beam_pos[0], beam_pos[1]] = beam_pos[2]

        return grid

    def check_agent_map(self, agent_map):
        """Checks the map to make sure agents aren't duplicated"""
        unique, counts = np.unique(agent_map, return_counts=True)
        count_dict = dict(zip(unique, counts))

        # check for multiple agents
        for i in range(self.num_agents):
            if count_dict[str(i+1)] != 1:
                print('Error! Wrong number of agent', i, 'in map!')
                return False
        return True

    def map_to_colors(self, map=None, color_map=None):
        """Converts a map to an array of RGB values.
        Parameters
        ----------
        map: np.ndarray
            map to convert to colors
        color_map: dict
            mapping between array elements and desired colors
        Returns
        -------
        arr: np.ndarray
            3-dim numpy array consisting of color map
        """
        if map is None:
            map = self.get_map_with_agents()
        if color_map is None:
            color_map = self.color_map

        # rgb_arr = np.zeros((map.shape[0], map.shape[1], 3), dtype=int)
        # for row_elem in range(map.shape[0]):
        #     for col_elem in range(map.shape[1]):
        #         rgb_arr[row_elem, col_elem, :] = color_map[map[row_elem, col_elem]]

        test = np.array([[color_map[map[row_elem, col_elem]] for col_elem in range(map.shape[1])] for row_elem in range(map.shape[0])])

        return test

    def render(self, filename=None):
        """ Creates an image of the map to plot or save.

        Args:
            path: If a string is passed, will save the image
                to disk at this location.
        """
        map_with_agents = self.get_map_with_agents()

        rgb_arr = self.map_to_colors(map_with_agents)
        plt.imshow(rgb_arr, interpolation='nearest')
        plt.pause(0.00001)
        if filename is not None:
            # plt.show()
            # else:
            plt.savefig(filename)

    def update_moves(self, agent_actions):
        """Converts agent action tuples into a new map and new agent positions.
        Also resolves conflicts over multiple agents wanting a cell.

        This method works by finding all conflicts over a cell and randomly assigning them
       to one of the agents that desires the slot. It then sets all of the other agents
       that wanted the cell to have a move of staying. For moves that do not directly
       conflict with another agent for a cell, but may not be temporarily resolvable
       due to an agent currently being in the desired cell, we continually loop through
       the actions until all moves have been satisfied or deemed impossible.
       For example, agent 1 may want to move from [1,2] to [2,2] but agent 2 is in [2,2].
       Agent 2, however, is moving into [3,2]. Agent-1's action is first in the order so at the
       first pass it is skipped but agent-2 moves to [3,2]. In the second pass, agent-1 will
       then be able to move into [2,2].

        Parameters
        ----------
        agent_actions: dict
            dict with agent_id as key and action as value
        """

        reserved_slots = []
        for agent_id, action in agent_actions.items():
            agent = self.agents[agent_id]
            selected_action = ACTIONS[action]
            # TODO(ev) these two parts of the actions
            if 'MOVE' in action or 'STAY' in action:
                # rotate the selected action appropriately
                rot_action = self.rotate_action(selected_action, agent.get_orientation())
                new_pos = agent.get_pos() + rot_action
                # allow the agents to confirm what position they can move to
                new_pos = agent.return_valid_pos(new_pos)
                reserved_slots.append((*new_pos, 'P', agent_id))
            elif 'TURN' in action:
                new_rot = self.update_rotation(action, agent.get_orientation())
                agent.update_agent_rot(new_rot)

        # now do the conflict resolution part of the process

        # helpful for finding the agent in the conflicting slot
        agent_by_pos = {tuple(agent.get_pos()): agent.agent_id for agent in self.agents.values()}

        # agent moves keyed by ids
        agent_moves = {}

        # lists of moves and their corresponding agents
        move_slots = []
        agent_to_slot = []

        for slot in reserved_slots:
            row, col = slot[0], slot[1]
            if slot[2] == 'P':
                agent_id = slot[3]
                agent_moves[agent_id] = [row, col]
                move_slots.append([row, col])
                agent_to_slot.append(agent_id)

        # cut short the computation if there are no moves
        if len(agent_to_slot) > 0:

            # first we will resolve all slots over which multiple agents
            # want the slot

            # shuffle so that a random agent has slot priority
            shuffle_list = list(zip(agent_to_slot, move_slots))
            np.random.shuffle(shuffle_list)
            agent_to_slot, move_slots = zip(*shuffle_list)
            unique_move, indices, return_count = np.unique(move_slots, return_index=True,
                                                           return_counts=True, axis=0)
            search_list = np.array(move_slots)

            # first go through and remove moves that can't possible happen. Three types
            # 1. Trying to move into an agent that has been issued a stay command
            # 2. Trying to move into the spot of an agent that doesn't have a move
            # 3. Two agents trying to walk through one another

            # Resolve all conflicts over a space
            if np.any(return_count > 1):
                for move, index, count in zip(unique_move, indices, return_count):
                    if count > 1:
                        # check that the cell you are fighting over doesn't currently
                        # contain an agent that isn't going to move for one of the agents
                        # If it does, all the agents commands should become STAY
                        # since no moving will be possible
                        conflict_indices = np.where((search_list == move).all(axis=1))[0]
                        all_agents_id = [agent_to_slot[i] for i in conflict_indices]
                        # all other agents now stay in place so update their moves
                        # to reflect this
                        conflict_cell_free = True
                        for agent_id in all_agents_id:
                            moves_copy = agent_moves.copy()
                            # TODO(ev) code duplication, simplify
                            if move.tolist() in self.agent_pos:
                                # find the agent that is currently at that spot and make sure
                                # that the move is possible. If it won't be, remove it.
                                conflicting_agent_id = agent_by_pos[tuple(move)]
                                curr_pos = self.agents[agent_id].get_pos().tolist()
                                curr_conflict_pos = self.agents[conflicting_agent_id]. \
                                    get_pos().tolist()
                                conflict_move = agent_moves.get(conflicting_agent_id,
                                                                curr_conflict_pos)
                                # Condition (1):
                                # a STAY command has been issued
                                if agent_id == conflicting_agent_id:
                                    conflict_cell_free = False
                                # Condition (2)
                                # its command is to stay
                                # or you are trying to move into an agent that hasn't
                                # received a command
                                elif conflicting_agent_id not in moves_copy.keys() or \
                                        curr_conflict_pos == conflict_move:
                                    conflict_cell_free = False

                                # Condition (3)
                                # It is trying to move into you and you are moving into it
                                elif conflicting_agent_id in moves_copy.keys():
                                    if agent_moves[conflicting_agent_id] == curr_pos and \
                                            move.tolist() == self.agents[conflicting_agent_id] \
                                            .get_pos().tolist():
                                        conflict_cell_free = False

                        # if the conflict cell is open, let one of the conflicting agents
                        # move into it
                        if conflict_cell_free:
                            self.agents[agent_to_slot[index]].update_agent_pos(move)
                            agent_by_pos = {tuple(agent.get_pos()):
                                                agent.agent_id for agent in self.agents.values()}
                        # ------------------------------------
                        # remove all the other moves that would have conflicted
                        remove_indices = np.where((search_list == move).all(axis=1))[0]
                        all_agents_id = [agent_to_slot[i] for i in remove_indices]
                        # all other agents now stay in place so update their moves
                        # to stay in place
                        for agent_id in all_agents_id:
                            agent_moves[agent_id] = self.agents[agent_id].get_pos().tolist()

            # make the remaining un-conflicted moves
            while len(agent_moves.items()) > 0:
                agent_by_pos = {tuple(agent.get_pos()):
                                    agent.agent_id for agent in self.agents.values()}
                num_moves = len(agent_moves.items())
                moves_copy = agent_moves.copy()
                del_keys = []
                for agent_id, move in moves_copy.items():
                    if agent_id in del_keys:
                        continue
                    if move in self.agent_pos:
                        # find the agent that is currently at that spot and make sure
                        # that the move is possible. If it won't be, remove it.
                        conflicting_agent_id = agent_by_pos[tuple(move)]
                        curr_pos = self.agents[agent_id].get_pos().tolist()
                        curr_conflict_pos = self.agents[conflicting_agent_id].get_pos().tolist()
                        conflict_move = agent_moves.get(conflicting_agent_id, curr_conflict_pos)
                        # Condition (1):
                        # a STAY command has been issued
                        if agent_id == conflicting_agent_id:
                            del agent_moves[agent_id]
                            del_keys.append(agent_id)
                        # Condition (2)
                        # its command is to stay
                        # or you are trying to move into an agent that hasn't received a command
                        elif conflicting_agent_id not in moves_copy.keys() or \
                                curr_conflict_pos == conflict_move:
                            del agent_moves[agent_id]
                            del_keys.append(agent_id)
                        # Condition (3)
                        # It is trying to move into you and you are moving into it
                        elif conflicting_agent_id in moves_copy.keys():
                            if agent_moves[conflicting_agent_id] == curr_pos and \
                                    move == self.agents[conflicting_agent_id].get_pos().tolist():
                                del agent_moves[conflicting_agent_id]
                                del agent_moves[agent_id]
                                del_keys.append(agent_id)
                                del_keys.append(conflicting_agent_id)
                    # this move is unconflicted so go ahead and move
                    else:
                        self.agents[agent_id].update_agent_pos(move)
                        del agent_moves[agent_id]
                        del_keys.append(agent_id)

                # no agent is able to move freely, so just move them all
                # no updates to hidden cells are needed since all the
                # same cells will be covered
                if len(agent_moves) == num_moves:
                    for agent_id, move in agent_moves.items():
                        self.agents[agent_id].update_agent_pos(move)
                    break

    def update_custom_moves(self, agent_actions):
        for agent_id, action in agent_actions.items():
            # check its not a move based action
            if 'MOVE' not in action and 'STAY' not in action and 'TURN' not in action:
                agent = self.agents[agent_id]
                updates = self.custom_action(agent, action)
                if len(updates) > 0:
                    self.update_map(updates)

    def update_map(self, new_points):
        """For points in new_points, place desired char on the map"""
        for i in range(len(new_points)):
            row, col, char = new_points[i]
            # print(f'Old Coords: {row}, {col}\nType: {self.world_map[row, col]}')
            self.world_map[row, col] = char
            # print(f'New Coords: {row}, {col}\nType: {self.world_map[row, col]}')

    def reset_map(self):
        """Resets the map to be empty as well as a custom reset set by subclasses"""
        self.world_map = np.full((len(self.base_map), len(self.base_map[0])), ' ')
        self.build_walls()
        self.custom_reset()

    def update_map_fire(self, firing_pos, firing_orientation, fire_len, fire_char, cell_types=[],
                        update_char=[], blocking_cells='P'):
        """From a firing position, fire a beam that may clean or hit agents

        Notes:
            (1) Beams are blocked by agents
            (2) A beam travels along until it hits a blocking cell at which beam the beam
                covers that cell and stops
            (3) If a beam hits a cell whose character is in cell_types, it replaces it with
                the corresponding index in update_char
            (4) As per the rules, the beams fire from in front of the agent and on its
                sides so the beam that starts in front of the agent travels out one
                cell further than it does along the sides.
            (5) This method updates the beam_pos, an internal representation of how
                which cells need to be rendered with fire_char in the agent view

        Parameters
        ----------
        firing_pos: (list)
            the row, col from which the beam is fired
        firing_orientation: (list)
            the direction the beam is to be fired in
        fire_len: (int)
            the number of cells forward to fire
        fire_char: (str)
            the cell that should be placed where the beam goes
        cell_types: (list of str)
            the cells that are affected by the beam
        update_char: (list of str)
            the character that should replace the affected cells.
        blocking_cells: (list of str)
            cells that block the firing beam
        Returns
        -------
        updates: (tuple (row, col, char))
            the cells that have been hit by the beam and what char will be placed there
        """
        agent_by_pos = {tuple(agent.get_pos()): agent_id for agent_id, agent in self.agents.items()}
        start_pos = np.asarray(firing_pos)
        firing_direction = ORIENTATIONS[firing_orientation]
        # compute the other two starting positions
        right_shift = self.rotate_right(firing_direction)
        firing_pos = [start_pos, start_pos + right_shift - firing_direction,
                      start_pos - right_shift - firing_direction]
        firing_points = []
        updates = []
        for pos in firing_pos:
            next_cell = pos + firing_direction
            for i in range(fire_len):
                if self.test_if_in_bounds(next_cell) and \
                        self.world_map[next_cell[0], next_cell[1]] != '@':

                    # FIXME(ev) code duplication
                    # agents absorb beams
                    # activate the agents hit function if needed
                    if [next_cell[0], next_cell[1]] in self.agent_pos:
                        agent_id = agent_by_pos[(next_cell[0], next_cell[1])]
                        self.agents[agent_id].hit(fire_char)
                        firing_points.append((next_cell[0], next_cell[1], fire_char))
                        if self.world_map[next_cell[0], next_cell[1]] in cell_types:
                            type_index = cell_types.index(self.world_map[next_cell[0],
                                                                         next_cell[1]])
                            updates.append((next_cell[0], next_cell[1], update_char[type_index]))
                        break

                    # update the cell if needed
                    if self.world_map[next_cell[0], next_cell[1]] in cell_types:
                        # print(f'Co-ord: {self.world_map[next_cell[0], next_cell[1]]}, Cell Types: {cell_types}')
                        type_index = cell_types.index(self.world_map[next_cell[0], next_cell[1]])
                        # print(f'Type Index: {type_index}, Update char: {update_char[type_index]}\n')
                        updates.append((next_cell[0], next_cell[1], update_char[type_index]))

                    firing_points.append((next_cell[0], next_cell[1], fire_char))

                    # check if the cell blocks beams. For example, waste blocks beams.
                    if self.world_map[next_cell[0], next_cell[1]] in blocking_cells:
                        break

                    # increment the beam position
                    next_cell += firing_direction

                else:
                    break

        self.beam_pos += firing_points
        return updates

    def spawn_point(self):
        """Returns a randomly selected spawn point."""
        spawn_index = 0
        is_free_cell = False
        curr_agent_pos = [agent.get_pos().tolist() for agent in self.agents.values()]
        random.shuffle(self.spawn_points)
        for i, spawn_point in enumerate(self.spawn_points):
            if [spawn_point[0], spawn_point[1]] not in curr_agent_pos:
                spawn_index = i
                is_free_cell = True
        assert is_free_cell, 'There are not enough spawn points! Check your map?'
        return np.array(self.spawn_points[spawn_index])

    def spawn_rotation(self):
        """Return a randomly selected initial rotation for an agent"""
        rand_int = np.random.randint(len(ORIENTATIONS.keys()))
        return list(ORIENTATIONS.keys())[rand_int]

    def rotate_view(self, orientation, view):
        """Takes a view of the map and rotates it the agent orientation
        Parameters
        ----------
        orientation: str
            str in {'UP', 'LEFT', 'DOWN', 'RIGHT'}
        view: np.ndarray (row, column, channel)
        Returns
        -------
        a rotated view
        """
        if orientation == 'UP':
            return view
        elif orientation == 'LEFT':
            return np.rot90(view, k=1, axes=(0, 1))
        elif orientation == 'DOWN':
            return np.rot90(view, k=2, axes=(0, 1))
        elif orientation == 'RIGHT':
            return np.rot90(view, k=3, axes=(0, 1))
        else:
            raise ValueError('Orientation {} is not valid'.format(orientation))

    def build_walls(self):
        for i in range(len(self.wall_points)):
            row, col = self.wall_points[i]
            self.world_map[row, col] = '@'

    ########################################
    # Utility methods, move these eventually
    ########################################

    # TODO(ev) this can be a general property of map_env or a util
    def rotate_action(self, action_vec, orientation):
        # WARNING: Note, we adopt the physics convention that \theta=0 is in the +y direction
        if orientation == 'UP':
            return action_vec
        elif orientation == 'LEFT':
            return self.rotate_left(action_vec)
        elif orientation == 'RIGHT':
            return self.rotate_right(action_vec)
        else:
            return self.rotate_left(self.rotate_left(action_vec))

    def rotate_left(self, action_vec):
        return np.dot(ACTIONS['TURN_COUNTERCLOCKWISE'], action_vec)

    def rotate_right(self, action_vec):
        return np.dot(ACTIONS['TURN_CLOCKWISE'], action_vec)

    # TODO(ev) this should be an agent property
    def update_rotation(self, action, curr_orientation):
        if action == 'TURN_COUNTERCLOCKWISE':
            if curr_orientation == 'LEFT':
                return 'DOWN'
            elif curr_orientation == 'DOWN':
                return 'RIGHT'
            elif curr_orientation == 'RIGHT':
                return 'UP'
            else:
                return 'LEFT'
        else:
            if curr_orientation == 'LEFT':
                return 'UP'
            elif curr_orientation == 'UP':
                return 'RIGHT'
            elif curr_orientation == 'RIGHT':
                return 'DOWN'
            else:
                return 'LEFT'

    # TODO(ev) this definitely should go into utils or the general agent class
    def test_if_in_bounds(self, pos):
        """Checks if a selected cell is outside the range of the map"""
        if pos[0] < 0 or pos[0] >= self.world_map.shape[0]:
            return False
        elif pos[1] < 0 or pos[1] >= self.world_map.shape[1]:
            return False
        else:
            return True

# basic moves every agent should do
BASE_ACTIONS = {0: 'MOVE_LEFT',  # Move left
                1: 'MOVE_RIGHT',  # Move right
                2: 'MOVE_UP',  # Move up
                3: 'MOVE_DOWN',  # Move down
                4: 'STAY',  # don't move
                5: 'TURN_CLOCKWISE',  # Rotate counter clockwise
                6: 'TURN_COUNTERCLOCKWISE'}  # Rotate clockwise


class Agent(object):

    def __init__(self, agent_id, start_pos, start_orientation, grid, row_size, col_size):
        """Superclass for all agents.

        Parameters
        ----------
        agent_id: (str)
            a unique id allowing the map to identify the agents
        start_pos: (np.ndarray)
            a 2d array indicating the x-y position of the agents
        start_orientation: (np.ndarray)
            a 2d array containing a unit vector indicating the agent direction
        grid: (2d array)
            a reference to this agent's view of the environment
        row_size: (int)
            how many rows up and down the agent can look
        col_size: (int)
            how many columns left and right the agent can look
        """
        self.agent_id = agent_id
        self.pos = np.array(start_pos)
        self.orientation = start_orientation
        # TODO(ev) change grid to env, this name is not very informative
        self.grid = grid
        self.row_size = row_size
        self.col_size = col_size
        # Step, wall, fire, hit, apple
        self.reward_this_turn = 0

    @property
    def action_space(self):
        """Identify the dimensions and bounds of the action space.

        MUST BE implemented in new environments.

        Returns
        -------
        gym Box, Discrete, or Tuple type
            a bounded box depicting the shape and bounds of the action space
        """
        raise NotImplementedError

    @property
    def observation_space(self):
        """Identify the dimensions and bounds of the observation space.

        MUST BE implemented in new environments.

        Returns
        -------
        gym Box, Discrete or Tuple type
            a bounded box depicting the shape and bounds of the observation
            space
        """
        raise NotImplementedError

    def action_map(self, action_number):
        """Maps action_number to a desired action in the map"""
        raise NotImplementedError

    def get_state(self):
        return return_view(self.grid, self.get_pos(),
                           self.row_size, self.col_size)

    def compute_reward(self):
        reward = self.reward_this_turn
        # if (self.reward_this_turn != [-1, 0, 0, 0, 0]):
        #     print(self.reward_this_turn)
        self.reward_this_turn = 0
        return reward

    def set_pos(self, new_pos):
        self.pos = np.array(new_pos)

    def get_pos(self):
        return self.pos

    def translate_pos_to_egocentric_coord(self, pos):
        offset_pos = pos - self.get_pos()
        ego_centre = [self.row_size, self.col_size]
        return ego_centre + offset_pos

    def set_orientation(self, new_orientation):
        self.orientation = new_orientation

    def get_orientation(self):
        return self.orientation

    def get_map(self):
        return self.grid

    def return_valid_pos(self, new_pos):
        """Checks that the next pos is legal, if not return current pos"""
        ego_new_pos = new_pos  # self.translate_pos_to_egocentric_coord(new_pos)
        new_row, new_col = ego_new_pos
        # you can't walk through walls
        temp_pos = new_pos.copy()
        if self.grid[new_row, new_col] == '@':
            temp_pos = self.get_pos()
        return temp_pos

    def update_agent_pos(self, new_pos):
        """Updates the agents internal positions

        Returns
        -------
        old_pos: (np.ndarray)
            2 element array describing where the agent used to be
        new_pos: (np.ndarray)
            2 element array describing the agent positions
        """
        old_pos = self.get_pos()
        ego_new_pos = new_pos  # self.translate_pos_to_egocentric_coord(new_pos)
        new_row, new_col = ego_new_pos
        # you can't walk through walls
        temp_pos = new_pos.copy()
        if self.grid[new_row, new_col] == '@':
            temp_pos = self.get_pos()
        self.set_pos(temp_pos)
        # TODO(ev) list array consistency
        return self.get_pos(), np.array(old_pos)

    def update_agent_rot(self, new_rot):
        self.set_orientation(new_rot)

    def hit(self, char):
        """Defines how an agent responds to being hit by a beam of type char"""
        raise NotImplementedError

    def consume(self, char):
        """Defines how an agent interacts with the char it is standing on"""
        raise NotImplementedError

def return_view(grid, pos, row_size, col_size):
    """Given a map grid, position and view window, returns correct map part

    Note, if the agent asks for a view that exceeds the map bounds,
    it is padded with zeros

    Parameters
    ----------
    grid: 2D array
        map array containing characters representing
    pos: list
        list consisting of row and column at which to search
    row_size: int
        how far the view should look in the row dimension
    col_size: int
        how far the view should look in the col dimension

    Returns
    -------
    view: (np.ndarray) - a slice of the map for the agent to see
    """
    x, y = pos
    left_edge = x - col_size
    right_edge = x + col_size
    top_edge = y - row_size
    bot_edge = y + row_size
    pad_mat, left_pad, top_pad = pad_if_needed(left_edge, right_edge,
                                               top_edge, bot_edge, grid)
    x += left_pad
    y += top_pad
    view = pad_mat[x - col_size: x + col_size + 1,
           y - row_size: y + row_size + 1]
    return view

def pad_if_needed(left_edge, right_edge, top_edge, bot_edge, matrix):
    # FIXME(ev) something is broken here, I think x and y are flipped
    row_dim = matrix.shape[0]
    col_dim = matrix.shape[1]
    left_pad, right_pad, top_pad, bot_pad = 0, 0, 0, 0
    if left_edge < 0:
        left_pad = abs(left_edge)
    if right_edge > row_dim - 1:
        right_pad = right_edge - (row_dim - 1)
    if top_edge < 0:
        top_pad = abs(top_edge)
    if bot_edge > col_dim - 1:
        bot_pad = bot_edge - (col_dim - 1)

    return pad_matrix(left_pad, right_pad, top_pad, bot_pad, matrix, 0), left_pad, top_pad

def pad_matrix(left_pad, right_pad, top_pad, bot_pad, matrix, const_val=1):
    pad_mat = np.pad(matrix, ((left_pad, right_pad), (top_pad, bot_pad)),
                     'constant', constant_values=(const_val, const_val))
    return pad_mat

CLEANUP_ACTIONS = BASE_ACTIONS.copy()
CLEANUP_ACTIONS.update({7: 'FIRE',  # Fire a penalty beam
                        8: 'CLEAN'})  # Fire a cleaning beam

CLEANUP_VIEW_SIZE = 7


class CleanupAgent(Agent):
    def __init__(self, agent_id, start_pos, start_orientation, grid, view_len=CLEANUP_VIEW_SIZE):
        self.view_len = view_len
        super().__init__(agent_id, start_pos, start_orientation, grid, view_len, view_len)
        # remember what you've stepped on
        self.update_agent_pos(start_pos)
        self.update_agent_rot(start_orientation)

    @property
    def action_space(self):
        return Discrete(9)

    @property
    def observation_space(self):
        return Box(low=0.0, high=0.0, shape=(2 * self.view_len + 1,
                                             2 * self.view_len + 1, 3), dtype=np.float32)

    # Ugh, this is gross, this leads to the actions basically being
    # defined in two places
    def action_map(self, action_number):
        """Maps action_number to a desired action in the map"""
        return CLEANUP_ACTIONS[action_number]

    def fire_beam(self, char):
        if char == 'F':
            self.reward_this_turn -= 1

    def get_done(self):
        return False

    def hit(self, char):
        if char == 'F':
            self.reward_this_turn -= 50

    def consume(self, char):
        """Defines how an agent interacts with the char it is standing on"""
        if char == 'A':
            self.reward_this_turn += 1
            return ' '
        else:
            return char

# Add custom actions to the agent
ACTIONS['FIRE'] = 5  # length of firing beam
ACTIONS['CLEAN'] = 5  # length of cleanup beam

# OPPOSITE TO WHAT THEY HAD
# Custom colour dictionary
CLEANUP_COLORS = {'C': [100, 255, 255],  # Cyan cleaning beam
                  'S': [99, 156, 194],  # Light grey-blue stream cell
                  'H': [113, 75, 24],  # brown waste cells
                  'R': [99, 156, 194]}  # Light grey-blue river cell

SPAWN_PROB = [0, 0.005, 0.02, 0.05]

thresholdDepletion = 0.4
thresholdRestoration = 0.0
wasteSpawnProbability = 0.5
appleRespawnProbability = 0.05

CLEANUP_MAP = [
    '@@@@@@@@@@@@@@@@@@',
    '@RRRRRR     BBBBB@',
    '@HHHHHH      BBBB@',
    '@RRRRRR     BBBBB@',
    '@RRRRR  P    BBBB@',
    '@RRRRR    P BBBBB@',
    '@HHHHH       BBBB@',
    '@RRRRR      BBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@RRRRR   P P BBBB@',
    '@HHHHH   P  BBBBB@',
    '@RRRRRR    P BBBB@',
    '@HHHHHH P   BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH    P  BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHHH  P P BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH       BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHHH      BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH       BBBBB@',
    '@@@@@@@@@@@@@@@@@@']

CLEANUP_MAP_SIMPLIFIED = [
    '@@@@@@@@@@@@@',
    '@RRR     BBB@',
    '@HHH      BB@',
    '@RRR     BBB@',
    '@RRR P    BB@',
    '@RRR   P BBB@',
    '@HHH      BB@',
    '@RRR     BBB@',
    '@HHHSSS   BB@',
    '@HHHSSS   BB@',
    '@RRR  P P BB@',
    '@HHH  P  BBB@',
    '@RRR    P BB@',
    '@HHH P   BBB@',
    '@RRR      BB@',
    '@HHH  P  BBB@',
    '@RRR      BB@',
    '@RRR P P BBB@',
    '@@@@@@@@@@@@@']

class CleanupEnv(MapEnv):

    def __init__(self, ascii_map=CLEANUP_MAP, num_agents=1, render=False):
        super().__init__(ascii_map, num_agents, render)

        # compute potential waste area
        unique, counts = np.unique(self.base_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        self.potential_waste_area = counts_dict.get('H', 0) + counts_dict.get('R', 0)
        self.current_apple_spawn_prob = appleRespawnProbability
        self.current_waste_spawn_prob = wasteSpawnProbability
        self.compute_probabilities()

        # make a list of the potential apple and waste spawn points
        self.apple_points = []
        self.waste_start_points = []
        self.waste_points = []
        self.river_points = []
        self.stream_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == 'P':
                    self.spawn_points.append([row, col])
                elif self.base_map[row, col] == 'B':
                    self.apple_points.append([row, col])
                elif self.base_map[row, col] == 'S':
                    self.stream_points.append([row, col])
                if self.base_map[row, col] == 'H':
                    self.waste_start_points.append([row, col])
                if self.base_map[row, col] == 'H' or self.base_map[row, col] == 'R':
                    self.waste_points.append([row, col])
                if self.base_map[row, col] == 'R':
                    self.river_points.append([row, col])

        self.color_map.update(CLEANUP_COLORS)

    @property
    def action_space(self):
        agents = list(self.agents.values())
        return agents[0].action_space

    @property
    def observation_space(self):
        # FIXME(ev) this is an information leak
        agents = list(self.agents.values())
        return agents[0].observation_space

    def custom_reset(self):
        """Initialize the walls and the waste"""
        for waste_start_point in self.waste_start_points:
            self.world_map[waste_start_point[0], waste_start_point[1]] = 'H'
        for river_point in self.river_points:
            self.world_map[river_point[0], river_point[1]] = 'R'
        for stream_point in self.stream_points:
            self.world_map[stream_point[0], stream_point[1]] = 'S'
        self.compute_probabilities()

    def custom_action(self, agent, action):
        """Allows agents to take actions that are not move or turn"""
        updates = []
        if action == 'FIRE':
            agent.fire_beam('F')
            updates = self.update_map_fire(agent.get_pos().tolist(),
                                           agent.get_orientation(), ACTIONS['FIRE'],
                                           fire_char='F')
        elif action == 'CLEAN':
            agent.fire_beam('C')
            updates = self.update_map_fire(agent.get_pos().tolist(),
                                           agent.get_orientation(),
                                           ACTIONS['FIRE'],
                                           fire_char='C',
                                           cell_types=['H'],
                                           update_char=['R'])
            # blocking_cells=['H'])
        return updates

    def custom_map_update(self):
        """"Update the probabilities and then spawn"""
        self.compute_probabilities()
        self.update_map(self.spawn_apples_and_waste())

    def setup_agents(self):
        """Constructs all the agents in self.agent"""
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = 'agent-' + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            # grid = util.return_view(map_with_agents, spawn_point,
            #                         CLEANUP_VIEW_SIZE, CLEANUP_VIEW_SIZE)
            # agent = CleanupAgent(agent_id, spawn_point, rotation, grid)
            agent = CleanupAgent(agent_id, spawn_point, rotation, map_with_agents)
            self.agents[agent_id] = agent

    def spawn_apples_and_waste(self):
        spawn_points = []
        # spawn apples, multiple can spawn per step
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # don't spawn apples where agents already are
            if [row, col] not in self.agent_pos and self.world_map[row, col] != 'A':
                rand_num = np.random.rand(1)[0]
                if rand_num < self.current_apple_spawn_prob:
                    spawn_points.append((row, col, 'A'))

        # spawn one waste point, only one can spawn per step
        if not np.isclose(self.current_waste_spawn_prob, 0):
            random.shuffle(self.waste_points)
            for i in range(len(self.waste_points)):
                row, col = self.waste_points[i]
                # don't spawn waste where it already is
                if self.world_map[row, col] != 'H':
                    rand_num = np.random.rand(1)[0]
                    if rand_num < self.current_waste_spawn_prob:
                        spawn_points.append((row, col, 'H'))
                        break
        return spawn_points

    def compute_probabilities(self):
        waste_density = 0
        if self.potential_waste_area > 0:
            waste_density = 1 - self.compute_permitted_area() / self.potential_waste_area
        if waste_density >= thresholdDepletion:
            self.current_apple_spawn_prob = 0
            self.current_waste_spawn_prob = 0
        else:
            self.current_waste_spawn_prob = wasteSpawnProbability
            if waste_density <= thresholdRestoration:
                self.current_apple_spawn_prob = appleRespawnProbability
            else:
                spawn_prob = (1 - (waste_density - thresholdRestoration)
                              / (thresholdDepletion - thresholdRestoration)) \
                             * appleRespawnProbability
                self.current_apple_spawn_prob = spawn_prob

    def compute_permitted_area(self):
        """How many cells can we spawn waste on?"""
        unique, counts = np.unique(self.world_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        current_area = counts_dict.get('H', 0)
        free_area = self.potential_waste_area - current_area
        return free_area


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
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)]
        return states, actions, rewards, next_states, dones


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
        self.episode_rewards .append(reward)

    def mean(self):
        return sum(self.moving_average) / self.maxlen

    def get_reward_data(self):
        episodes = np.array(
            [i for i in range(len(self.episode_rewards ))]).reshape(-1,1)

        rewards = np.array(self.episode_rewards ).reshape(-1,1)
        return np.concatenate((episodes, rewards), axis=1)

class MovingAverage(deque):
    def mean(self):
        return sum(self) / len(self)


SEED = 42
IMAGE = True

BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 6_000

GAMMA = 0.99
ALPHA = 1e-4

TRAINING_EPISODES = 10

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 1 / (25000 * 0.85)

COPY_TO_TARGET_EVERY = 1000 # Steps
START_TRAINING_AFTER = 50 # Episodes
MEAN_REWARD_EVERY = 300 # Episodes

FRAME_STACK_SIZE = 3

now = datetime.now()
date_and_time = f'{now.year}-{now.month}-{now.day}_{now.hour}_{now.minute}'
PATH_ID = 'single_' + str(date_and_time)
PATH_DIR = './'
VIDEO_DIR = PATH_DIR + 'videos/' + str(date_and_time) + '/'

steps = 0 # Messy but it's basically operating as a static variable anyways

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
        with tf.device('/gpu:0'):
            if IMAGE:
                model = keras.Sequential([
                    Conv2D(256, (3, 3), activation='relu', input_shape=(self.input_size)),
                    Dropout(0.2),
                    Conv2D(256, (3, 3), activation='relu'),
                    Dropout(0.2),
                    Flatten(),
                    Dense(64, activation='relu'),
                    Dense(64, activation='relu'),
                    Dense(env.action_space.n)
                ])
            else:
                model = keras.Sequential([
                    Dense(64, activation='relu', input_shape=(self.input_size)),
                    Dense(64, activation='relu'),
                    Dense(env.action_space.n)
                ])

            self.optimizer = keras.optimizers.Adam(lr=ALPHA)
            self.loss_fn = keras.losses.Huber()
            return model

    def epsilon_greedy_policy(self, state, epsilon):
        """
        Select greedy action from model output based on current state with
        probability epsilon. With probability 1 - epsilon select random action.
        """
        if np.random.rand() < epsilon:
            return random.choice(self.actions)
        else:
            Q_values = self.model.predict(state[np.newaxis])
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
        states, actions, rewards, next_states, dones = experiences

        # Compute target Q values from 'next_states'
        next_Q_values = self.target_model.predict(next_states)

        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards +
                           (1 - dones) * self.gamma * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1) # Make column vector

        # Mask to only consider action taken
        mask = tf.one_hot(actions, self.output_size) # Number of actions
        # Compute loss and gradient for predictions on 'states'
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
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
        agent1_action = agent1.epsilon_greedy_policy(agent1_state, eps)
        agent2_action = agent2.epsilon_greedy_policy(agent2_state, eps)
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

        # # Linear scalarisation
        # agent1_reward = np.dot(agent1_rewards, pref1)
        # agent2_reward = np.dot(agent2_rewards, pref2)
        # rewards = [agent1_reward]
        rewards = [agent1_rewards, agent2_rewards]

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
        agent1.replay_memory.append((agent1_state, agent1_action, agent1_rewards, next_agent1_state, done))

        # Store in replay buffers
        # Agent 2
        if IMAGE:
            agent2_frame_stack.append(next_agent2_state)
            next_agent2_state = np.concatenate(agent2_frame_stack, axis=2)
        else:
            next_agent2_state = next_agent2_state / max_state
        agent2.replay_memory.append((agent2_state, agent2_action, agent2_rewards, next_agent2_state, done))

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
        agent1.model.save(f'{PATH_DIR}/models/cleanup_model_dqn1_{PATH_ID}.h5')
        agent1.plot_learning_curve(image_path=f'{PATH_DIR}/plots/cleanup_plot_dqn1_{PATH_ID}.png',
                                   csv_path=f'{PATH_DIR}/plots/cleanup_rewards_dqn1_{PATH_ID}.csv')
        agent2.model.save(f'{PATH_DIR}/models/cleanup_model_dqn2_{PATH_ID}.h5')
        agent2.plot_learning_curve(image_path=f'{PATH_DIR}/plots/cleanup_plot_dqn2_{PATH_ID}.png',
                                   csv_path=f'{PATH_DIR}/plots/cleanup_rewards_dqn2_{PATH_ID}.csv')


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
    #agent1.load_model(f'/content/models/cleanup_model_dqn1_single_2021-2-1_22_55.h5')
    #agent2.load_model(f'/content/models/cleanup_model_dqn2_single_2021-2-1_22_55.h5')
    if not IMAGE:
        x, y = env.base_gridmap_array.shape[0] - 1, env.base_gridmap_array.shape[1] - 1
        max_state = np.array([x, y, *[z for _ in range(2) for z in [x, y, x+y]]], dtype=np.float32)
    for episode in range(1, TRAINING_EPISODES):
        training_episode() # Don't save images/render
    training_episode(True) # Render last image
    agent1.model.save(f'{PATH_DIR}/models/cleanup_model_dqn1_{PATH_ID}.h5')
    agent1.plot_learning_curve(image_path=f'{PATH_DIR}/plots/cleanup_plot_dqn1_{PATH_ID}.png',
                               csv_path=f'{PATH_DIR}/plots/cleanup_rewards_dqn1_{PATH_ID}.csv')
    agent2.model.save(f'{PATH_DIR}/models/cleanup_model_dqn2_{PATH_ID}.h5')
    agent2.plot_learning_curve(image_path=f'{PATH_DIR}/plots/cleanup_plot_dqn2_{PATH_ID}.png',
                               csv_path=f'{PATH_DIR}/plots/cleanup_rewards_dqn2_{PATH_ID}.csv')
    run_time = datetime.now() - start_time
    print(f'\nRun time: {run_time} s')