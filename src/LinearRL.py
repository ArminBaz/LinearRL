import numpy as np
import gymnasium as gym

import gym_env

from utils import create_transition_matrix_mapping, get_transition_matrix
from replan_utils import add_barrier, change_goal, new_goal


class LinearRL:
    def __init__(self, env_name, _lambda):
        self.env = gym.make(env_name)
        self.start_loc = self.env.unwrapped.start_loc
        self.target_loc = self.env.unwrapped.target_loc
        self.maze = self.env.unwrapped.maze
        self.size = self.maze.size
        self.height, self.width = self.maze.shape
        self.target_locs = [self.target_loc]
        self.mapping = create_transition_matrix_mapping(self.maze)
        self.T = get_transition_matrix(self.env, self.size, self.mapping)
        

        # Get terminal states
        self.terminals = np.diag(self.T) == 1
        # Calculate P = T_{NT}
        self.P = self.T[~self.terminals][:,self.terminals]
        # Calculate reward
        self.r = np.full(len(self.T), -0.1)     # our reward at each non-terminal state to be -0.1
        self.r[self.terminals] = 1              # reward at terminal state is 1
        self.expr = np.exp(self.r[self.terminals] / _lambda)

        # Params
        self._lambda = _lambda
        self.DR = np.eye(self.size)
        self.Z = np.zeros(self.size)
    
    def update_V(self):
        """
        Updates the value of our states according to equations from paper
        """
        self.Z[~self.terminals] = self.DR[~self.terminals][:,~self.terminals] @ self.P @ self.expr
        self.Z[self.terminals] = self.expr
        self.V = np.round(np.log(self.Z), 2)
    
    def select_action(self, state):
        """
        Select action greedily according to Z-values

        Args:
            state (tuple): Current state
        Output:
            action (int): Greedy action 
        """
        action_values = np.full(self.env.action_space.n, -np.inf)
        for action in self.env.unwrapped.get_available_actions(state):
            direction = self.env.unwrapped._action_to_direction[action]
            new_state = state + direction

            if self.maze[new_state[0], new_state[1]] == "1":
                continue
            action_values[action] = round(np.log(self.Z[self.mapping[(new_state[0],new_state[1])]]), 2)

        return np.nanargmax(action_values)
    
    def learn(self):
        """
        Use equations from the paper to solve for the Default Representation
        """
        # Solve for L and take the inverse to get DR
        L = np.diag(np.exp(-self.r / self._lambda)) - self.T
        self.DR = np.linalg.inv(L)

        # Update the value
        self.update_V()

    def replan(self, new_env, experiment, loc=None):
        """
        Function to replan, using equations from paper.

        Note, this probably isn't the best way to do this. In the future I might make the environment directly modifiable so I don't have to load a whole new
        environment in and specificy the changed location, but this is the way I'm doing it for now.

        Args:
            new_env (maze): The updated environment
            experiment (string): The experiment we are running
        """
        # Load the new environment
        new_env = gym.make(new_env)

        # Get the transition matrix of the new environment
        new_mapping = create_transition_matrix_mapping(new_env.unwrapped.maze)
        T = get_transition_matrix(new_env, new_env.unwrapped.maze.size, new_mapping)

        # Check if transition structure changes:
        if experiment == "add_barrier":
            print("Replanning around new barrier...")
            # Border transition using eqn. 17
            add_barrier(self, T)

        # Check if terminal states are updated:
        elif experiment == "update_goal":
            print("Replanning with new terminal state values...")
            raise NotImplementedError

        # Check if we are planning towards a new goal:
        elif experiment == "new_goal":
            print("Replanning towards new goal...")
            new_goal(self, T, loc)
        
        else:
            raise ValueError("Invalid experiment")
        # Update Values
        self.update_V()

        # Set new environment
        self.env = new_env
        self.maze = self.env.unwrapped.maze