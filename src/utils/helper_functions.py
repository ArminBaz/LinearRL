import os

import numpy as np
import gymnasium as gym

import gym_env


available_mazes = ["simple-5x5", "hairpin-14x14", "tolman-9x9-nb", "tolman-9x9-b"]

def load_maze(maze_name):
    """
    Loads specified maze into a gym environment and returns
    """
    if maze_name not in available_mazes:
        raise ValueError(f"{maze_name} not found in available mazes")

    return gym.make(maze_name)

def row_col_to_index(row, col, len):
    """
    Converts (row,col) to an index in array
    """
    return row*len + col

def index_to_row_col(index, len):
    """
    Converts index back to (row,col)
    """
    return (index // len, index % len)

def get_transition_matrix(env):
    """
    Returns the transition matrix, under a uniform policy
    """
    actions = np.arange(env.action_space.n, dtype=int)
    maze = env.unwrapped.maze
    size = maze.size

    # Get the transition matrix T N^2 x N^2
    T = np.zeros(shape=(size, size))

    # loop through the maze
    for row in range(maze.shape[0]):
        for col in range(maze.shape[1]):
            # if we hit a barrier
            if maze[row,col] == '1':
                continue
            # at each location, we want to store the location, keep track of which new states we transition into, and how many states we transition into
            loc = np.array((row,col))
            new_states = []
            for action in actions:     # loop through actions
                env.unwrapped.agent_loc = loc                  # set new agent location based on where we are in maze
                obs, _, _, _, _ = env.step(action)     # take action

                # if dont move because we hit a boundary, do nothing
                if (obs['agent'] == loc).all():
                    continue
                new_states.append(obs['agent'])
            
            idx_cur = row_col_to_index(row, col, maze.shape[0])
            for new_state in new_states:
                idx_new = row_col_to_index(new_state[0], new_state[1], maze.shape[0])
                T[idx_cur, idx_new] = 1/len(new_states)