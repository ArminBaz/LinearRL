import os

import numpy as np
import gymnasium as gym

import gym_env
import config


available_mazes = ["simple-5x5", "simple-15x15", "hairpin-14x14", "tolman-9x9-nb", "tolman-9x9-b"]
available_agents = ["linear-rl", "z-learning"]

def load_maze(maze_name):
    """
    Loads specified maze into a gym environment and returns
    """
    if maze_name not in available_mazes:
        raise ValueError(f"{maze_name} not found in available mazes")

    return gym.make(maze_name)

def get_blocked_states(maze):
    """
    Returns a list of all states that are blocked, represented in the environment with a "1"
    """
    blocked_states = []

    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i,j] == "1":
                blocked_states.append((i,j))
    
    return blocked_states

def create_transition_matrix_mapping(maze):
    """
    Maps (row,col) to an index, taking into account that some states are blocked
    """
    blocked_states = get_blocked_states(maze)
    n = len(maze)  # Size of the maze (N)

    # Create a mapping from maze state indices to transition matrix indices
    mapping = {}
    matrix_idx = 0

    for i in range(n):
        for j in range(n):
            if (i, j) not in blocked_states:
                mapping[(i, j)] = matrix_idx
                matrix_idx += 1

    return mapping

def get_transition_matrix(env, mapping):
    """
    Returns the transition matrix, under a uniform policy
    """
    # mapping, m = create_transition_matrix_mapping(maze)
    # reverse_mapping = {index: (i, j) for (i, j), index in mapping.items()}

    actions = np.arange(env.action_space.n, dtype=int)
    maze = env.unwrapped.maze

    T = np.zeros(shape=(len(mapping), len(mapping)))

    # loop through the maze
    for row in range(maze.shape[0]):
        for col in range(maze.shape[1]):
            # if we hit a barrier
            if maze[row,col] == '1':
                continue
                
            # at each location, we want to store the location, keep track of which new states we transition into, and how many states we transition into
            loc = np.array((row,col))
            idx_cur = mapping[row, col]

            # if we hit a goal
            if maze[row, col] == 'G':
                T[idx_cur, idx_cur] = 1
                continue

            new_states = []
            for action in actions:     # loop through actions
                env.unwrapped.agent_loc = loc                  # set new agent location based on where we are in maze
                obs, reward, term, _, _ = env.step(action)     # take action

                # if dont move because we hit a boundary, do nothing
                if (obs['agent'] == loc).all():
                    continue
                new_states.append(obs['agent'])
            
            for new_state in new_states:
                idx_new =mapping[new_state[0], new_state[1]]
                T[idx_cur, idx_new] = 1/len(new_states)
    
    # Make sure all the rows sum to 1
    assert np.all(np.isclose(np.sum(T, axis=1), 1.0)), "Not all rows sum to one."

    return T

def split_transition_matrix(T, mapping, target_locs):
    """
    Splits our transition matrix into T_nn and T_nt
    T_nn -> transition probability between non-terminal states 
    T_nt -> transition probability from non-terminal to terminal states
    """
    # Make T_nn by excluding the rows and columns associated with the terminal state (also works if we have multiple)
    terminal_indices = [mapping[loc[0], loc[1]] for loc in target_locs]

    T_nn = T.copy()

    for index in terminal_indices:
        T_nn = np.delete(T_nn, index, axis=0)
        T_nn = np.delete(T_nn, index, axis=1)

    # Make T_nt by selecting only the rows corresponding to the terminal states
    all_indices = set(range(T.shape[0]-1))
    nonterminal_indices = all_indices - set(terminal_indices)

    T_nt = np.zeros((len(T)-1, len(terminal_indices)))

    for i, index_term in enumerate(terminal_indices):
        for index in nonterminal_indices:
            T_nt[index, i] = T[index, index_term]
    
    return T_nn, T_nt