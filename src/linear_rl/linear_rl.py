import os

import numpy as np

import gymnasium as gym
import gym_env

import config
from ..utils import helper_functions

ENV_NAME = config.ENV_NAME

class LinearRL():
    def __init__(self, ENV_NAME):
        self.env = helper_functions.load_maze(ENV_NAME)

    def get_expv(self, T, terminals):
        """
        Uses equations from the paper to calculate exp(v*)
        """
        _lambda = config.LAMBDA
        terminals = np.diag(T) == 1     # find terminal states

        c = np.full(len(T), 1)
        c[terminals] = 0
        r = -c

        L = np.diag(np.exp(c / _lambda)) - T

        # Remove rows and columns corresponding to terminals
        L = np.delete(L, terminals, axis=0)
        L = np.delete(L, terminals, axis=1)

        # Calculate the inverse of L
        M = np.linalg.inv(L)

        # Calculate P = T_{NT}
        P = T[~terminals][:,terminals]

        # Calculate expr
        expr = np.exp(r[terminals] / _lambda)

        # Initialize expv as zeros
        expv = np.zeros(r.shape)

        # Calculate expv for non-terminal states
        expv_N = M @ P @ expr
        expv[~terminals] = expv_N

        # Calculate expv for terminal states
        expv[terminals] = np.exp(r[terminals] / _lambda)

        return expv
    
    def maze_value(self, maze, expv, mapping):
        """
        Uses exp(v*) and the mapping to create a value function of the optimal value for each state of the maze.
        Replaces blocked locations with negative infinity.
        """
        v_maze = np.zeros_like(maze)
        for row in range(v_maze.shape[0]):
            for col in range(v_maze.shape[1]):
                if maze[row, col] == "1":
                    v_maze[row,col] = np.NINF
                    continue
                v_maze[row,col] = round(np.log(expv[mapping[(row,col)]]), 2)
        
        return v_maze