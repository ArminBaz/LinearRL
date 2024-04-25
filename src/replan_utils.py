"""
Author: Armin Bazarjani

This file contains three separate functions. One for each experiment I am replicating
1. Addition of a barrier and replanning
2. Changing the value of a terminal state and replanning
3. Addition of a terminal state that we would like to plan towards
"""

import numpy as np

def add_barrier(agent, T):
    """
    The transition structure of the environment has changed because we have included a barrier. We perform a low rank update to the old DR
    based off the states whose transition structure has changed (delta_locs).

    Args:
    agent (LinearRL class): The LinearRL agent 
    new_env (string): The name of the new environment
    """
    # Get locations of change
    differences = agent.T != T
    different_rows, _ = np.where(differences)
    delta_locs = np.unique(different_rows)

    D0 = agent.DR
    L0 = np.diag(np.exp(-agent.r)) - agent.T
    L = np.diag(np.exp(-agent.r)) - T

    # Use locations of change to get different values
    delta = L[delta_locs] - L0[delta_locs]
    D0_j = D0[:,delta_locs]
    I = np.eye(len(delta_locs))

    # Calculate inverse in eqn (17)
    inv = np.linalg.inv(I + np.dot(delta, D0_j))

    # Use everything to get B and update the DR
    B = np.dot( np.dot(D0_j, inv), np.dot(delta, D0) )
    D = D0 - B

    # Set agent's DR to new DR
    agent.DR = D


def change_goal(agent, new_reward):
    """
    New environment is the same as the old one, except we update the value of one of the terminal (goal) states. We can use the old precomputed DR 
    to get a new DR and include the new information as well.

    Args:
    agent (LinearRL class): The LinearRL agent 
    new_reward (list): The name of the new environment
    """
    # Update the reward with the new reward
    r_loc = np.argwhere(agent.terminals)[1]
    agent.r[r_loc] = new_reward
    
    # Get new reward and calculate expr
    r_new = agent.r
    expr_new = np.exp(r_new[agent.terminals] / agent._lambda)

    # Use the new reward and precomputed DR to update Z values
    Z_new = np.zeros(len(r_new))
    Z_new[~agent.terminals] = agent.DR[~agent.terminals][:,~agent.terminals] @ agent.P @ expr_new
    Z_new[agent.terminals] = expr_new
    agent.Z = Z_new

def new_goal(agent, T, loc):
    """
    New Environment is the same as the old one, with the inclusion of a new goal state that we want to use the old DR to
    plan towards
    
    Args:
    agent (LinearRL class): The LinearRL agent 
    T (array): The transition matrix of the new environment
    loc (tuple): Location of the new goal state
    """
    D0 = agent.DR
    L0 = np.diag(np.exp(-agent.r)) - agent.T
    L = np.diag(np.exp(-agent.r)) - T

    idx = agent.mapping[loc]

    d = L[idx, :] - L0[idx, :]
    m0 = D0[:,idx]

    # Convert d to a row vector of size (1, m)
    d = d.reshape(1, -1)

    # Convert m0 to a column vector of size (m, 1)
    m0 = m0.reshape(-1, 1)

    # Get the amount of change to the DR
    alpha = (np.dot(m0,d)) / (1 + (np.dot(d,m0)))
    change = np.dot(alpha,D0)

    # Apply change to DR
    D = np.copy(D0)
    D -= change

    # Set agent's DR to new DR
    agent.DR = D

    # Update terminals
    agent.terminals = np.diag(T) == 1
    # Update P
    agent.P = T[~agent.terminals][:,agent.terminals]
    # Update reward
    agent.r = np.full(len(T), -0.1)     # our reward at each non-terminal state to be -0.1
    agent.r[agent.terminals] = 1         # reward at terminal state is 1
    agent.expr = np.exp(agent.r[agent.terminals] / agent._lambda)