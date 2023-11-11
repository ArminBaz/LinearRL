import os

import numpy as np
import gymnasium as gym

import gym_env

from utils import load_maze


class DeepLinearRL():
    """
    Iterative LinearRL with deep neural network to serve as a function approximator
    """
    def __init__(self, env_name, _lambda, epsilon, max_iterations):
        self._lambda = _lambda
        self.env = load_maze(env_name)
        self.env.reset()
    
    def run(self):
        """
        Runs 
        """