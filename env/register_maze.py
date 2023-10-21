from gymnasium.envs.registration import register
from maze_env import MazeEnv5x5

if __name__ == '__main__':
    register(
        id='maze-sample-5x5-v0',
        entry_point='maze_env:  MazeEnv5x5',
        max_episode_steps=2000,
    )