import os

import numpy as np
import gymnasium as gym
import torch
import torch.nn.functional as F


class DeepLinearRL:
    """
    Iterative LinearRL with deep neural network to serve as a function approximator
    """
    def __init__(self, 
                 env_name, 
                 _lambda, 
                 epsilon, 
                 max_iterations):
        self._lambda = _lambda
        self.env = gym.make(env_name)
    
    def run(self):
        """
        Runs 
        """