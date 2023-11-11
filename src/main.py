import os
import numpy as np

import config
from linear_rl import LinearRL, LinearRLIterative
from deep_linear_rl import DeepLinearRL

available_agents = ["linear-rl", "iterative-linear-rl", "deep-linear-rl"]

def get_agent(agent_name):
    """
    Returns class of agent that was specified
    """
    if config.AGENT not in available_agents:
        raise ValueError(f"{config.AGENT} not found in available agents")
    
    if agent_name == "linear-rl":
        return LinearRL(config.ENV_NAME, config.LAMBDA)
    # elif agent_name == "iterative-linear-rl":
    #     return LinearRLIterative(config.ENV_NAME, config.LAMBDA, config.EPSILON, config.MAX_ITERATIONS)
    # elif agent_name == "deep-linear-rl":
    #     return DeepLinearRL(config.ENV_NAME, config.LAMBDA, config.EPSILON, config.MAX_ITERATIONS)

def main():
    agent = LinearRL(config.ENV_NAME, config.LAMBDA)
    expv, maze_values = agent.run()
    print(expv.shape, maze_values.shape)
    print(maze_values)

if __name__ == '__main__':
    main()