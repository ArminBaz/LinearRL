import os
import numpy as np

import config
from linear_rl import LinearRL

available_agents = ["linear-rl", "z-learning"]

def get_agent(agent_name):
    """
    Returns class of agent that was specified
    """
    if config.AGENT not in available_agents:
        raise ValueError(f"{config.AGENT} not found in available agents")
    
    if agent_name == "linear-rl":
        return LinearRL(config.ENV_NAME, config.LAMBDA)
    # elif agent_name == "z-learning":
    #     return ZLearning(config.ENV_NAME, config.LAMBDA)

def main():
    agent = LinearRL(config.ENV_NAME, config.LAMBDA)
    expv, maze_values = agent.run()
    print(expv.shape, maze_values.shape)
    print(maze_values)

if __name__ == '__main__':
    main()