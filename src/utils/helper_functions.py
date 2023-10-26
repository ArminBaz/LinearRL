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

def get_transition_matrix(maze):
    """
    Returns the transition matrix, under a uniform policy
    """
    