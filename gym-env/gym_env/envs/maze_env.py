import os
import numpy as np

import pygame
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding

class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, maze_file, enable_render):
        # read maze_file
        self.maze = self._read_maze_file(maze_file=maze_file)

        # enable render
        self._enable_render = enable_render

        # get important positions
        self._start_loc = np.where(self.maze == 'S')
        self._target_loc = np.where(self.maze == 'G')
        self._agent_loc = self._start_loc

        self.num_rows, self.num_cols = self.maze.shape

        # 4 possible actions: 0=up, 1=down, 2=right, 3=left
        self.action_space = spaces.Discrete(4)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=np.array(0,0), high=np.array(self.num_rows-1, self.num_cols-1), shape=(2,), dtype=int),
                "target": spaces.Box(low=np.array(0,0), high=np.array(self.num_rows-1, self.num_cols-1), shape=(2,), dtype=int),
            }
        )

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        # Initialize Pygame
        pygame.init()
        self.cell_size = 125

        # setting display size
        if self._enable_render is True:
            self.screen = pygame.display.set_mode((self.num_cols * self.cell_size, self.num_rows * self.cell_size))

    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._agent_loc = self._start_loc

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Get direction
        direction = self._action_to_direction[action]

        new_loc = np.copy(self._agent_loc)
        new_loc += direction

        # Check if the new position is valid
        if self._is_valid_position(new_loc):
            self._agent_loc = new_loc

        # Reward function
        if np.array_equal(self._agent_loc, self._target_loc):
            reward = 1.0
            done = True
        else:
            reward = 0.0
            done = False

        return self._agent_loc, reward, done, {}
    
    def _read_maze_file(self, maze_file):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        rel_path = os.path.join(dir_path, "maze_files", maze_file)

        return np.load(file=rel_path)

    def _get_obs(self):
        return {"agent": self._agent_loc, "target":self._target_loc}
    
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_loc - self._target_loc, ord=1
            )
        }

    def _is_valid_position(self, pos):
        row, col = pos
   
        # If agent goes out of the grid
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols:
            return False

        # If the agent hits an obstacle
        if self.maze[row, col] == '1':
            return False
        return True

    def render(self):
        if self._enable_render is False:
            return
        # Clear the screen
        self.screen.fill((255, 255, 255))  

        # Draw env elements one cell at a time
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size
            
                try:
                    print(np.array(self._agent_loc)==np.array([row,col]).reshape(-1,1))
                except Exception as e:
                    print('Initial state')

                if self.maze[row, col] == '1':  # Obstacle
                    pygame.draw.rect(self.screen, (0, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.maze[row, col] == 'S':  # Starting position
                    pygame.draw.rect(self.screen, (0, 255, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.maze[row, col] == 'G':  # Target position
                    pygame.draw.rect(self.screen, (255, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))

                if np.array_equal(np.array(self.current_pos), np.array([row, col]).reshape(-1,1)):  # Agent position
                    pygame.draw.rect(self.screen, (0, 0, 255), (cell_left, cell_top, self.cell_size, self.cell_size))

        pygame.display.update()  # Update the display
    

class MazeEnv5x5(MazeEnv):
    def __init__(self):
        super(MazeEnv5x5, self).__init__(maze_file="maze2d_5x5.npy")

class MazeEnvHairpin(MazeEnv):
    def __init__(self):
        super(MazeEnvHairpin).__init__(maze_file="hairpin_14x14.npy")

if __name__ == '__main__':
    # Test it out
    env = MazeEnv(maze_file="hairpin_14x14.npy")
    print(env)
    # obs = env.reset()
    # done = True
    # while done:
    #     env.render()
    #     pygame.time.wait(5000)
    #     done = False