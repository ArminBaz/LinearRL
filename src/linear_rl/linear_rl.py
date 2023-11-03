import os

import numpy as np

import gymnasium as gym
import gym_env

from utils import load_maze, create_transition_matrix_mapping, get_transition_matrix

class LinearRL():
    def __init__(self, ENV_NAME, _lambda):
        self._lambda = _lambda
        # load environment and reset it
        self.env = load_maze(ENV_NAME)
        self.env.reset()
        self.maze = self.env.unwrapped.maze
        self.mapping = create_transition_matrix_mapping(self.maze)
        self.T = get_transition_matrix(self.env, self.mapping)

    def get_expv(self):
        """
        Uses equations from the paper to calculate exp(v*)
        """
        terminals = np.diag(self.T) == 1     # find terminal states

        c = np.full(len(self.T), 1)
        c[terminals] = 0
        r = -c

        L = np.diag(np.exp(c / self._lambda)) - self.T

        # Remove rows and columns corresponding to terminals
        L = np.delete(L, terminals, axis=0)
        L = np.delete(L, terminals, axis=1)

        # Calculate the inverse of L
        M = np.linalg.inv(L)

        # Calculate P = T_{NT}
        P = self.T[~terminals][:,terminals]

        # Calculate expr
        expr = np.exp(r[terminals] / self._lambda)

        # Initialize expv as zeros
        expv = np.zeros(r.shape)

        # Calculate expv for non-terminal states
        expv_N = M @ P @ expr
        expv[~terminals] = expv_N

        # Calculate expv for terminal states
        expv[terminals] = np.exp(r[terminals] / self._lambda)

        return expv
    
    def maze_value(self, expv):
        """
        Uses exp(v*) and the mapping to create a value function of the optimal value for each state of the maze.
        Replaces blocked locations with negative infinity.
        """
        v_maze = np.zeros_like(self.maze)
        for row in range(v_maze.shape[0]):
            for col in range(v_maze.shape[1]):
                if self.maze[row, col] == "1":
                    v_maze[row,col] = np.NINF
                    continue
                v_maze[row,col] = round(np.log(expv[self.mapping[(row,col)]]), 2)
        
        return v_maze
    
    def run(self):
        """
        Runs LinearRL agent. Returns expv, which is the output of the linearrl model (z values), also returns the maze_values which uses the expv values
        to assign a value to each block of the maze
        """
        expv = self.get_expv()
        maze_values = self.maze_value(expv)

        return expv, maze_values

# Test
if __name__ == '__main__':
    print("Hello")