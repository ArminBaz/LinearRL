import os
import numpy as np

import pygame
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding

class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, maze_file, render_mode=None):
        # Read maze_file
        self.maze = self._read_maze_file(maze_file=maze_file)

        # Check if render mode is valid and set render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Get important positions
        start_loc = np.where(self.maze == 'S')
        target_loc = np.where(self.maze == 'G')

        self._start_loc = np.array([start_loc[0][0], start_loc[1][0]])
        self._target_loc = np.array([target_loc[0][0], target_loc[1][0]])
        self._agent_loc = self._start_loc

        # Size of maze and pygame window
        self.num_rows, self.num_cols = self.maze.shape
        self.window_size = 512

        # 4 possible actions: 0=up, 1=down, 2=right, 3=left
        self.action_space = spaces.Discrete(4)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=np.array([0,0]), high=np.array([self.num_rows-1, self.num_cols-1]), shape=(2,), dtype=int),
                "target": spaces.Box(low=np.array([0,0]), high=np.array([self.num_rows-1, self.num_cols-1]), shape=(2,), dtype=int),
            }
        )

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        # Only used for human-rendering
        self.window = None
        self.clock = None

    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._agent_loc = self._start_loc

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """
        One step in our environment given the action
        """
        # Get direction
        direction = self._action_to_direction[action]

        new_loc = np.copy(self._agent_loc)
        new_loc += direction

        # Check if the new position is valid
        if self._is_valid_position(new_loc):
            self._agent_loc = new_loc

        # Check if terminated
        terminated = np.array_equal(self._agent_loc, self._target_loc)
        reward = 1 if terminated else 0  # Binary sparse rewards
        
        if self.render_mode == "human":
            self._render_frame()
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info
    
    def _read_maze_file(self, maze_file):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        rel_path = os.path.join(dir_path, "maze_files", maze_file)

        return np.load(file=rel_path)

    def _get_obs(self):
        """
        Observation, returns the agent and target positions
        """
        return {"agent": self._agent_loc, "target":self._target_loc}
    
    def _get_info(self):
        """
        Information, returns the manhattan (L1) distance between agent and target.
        """
        return {
            "distance": np.linalg.norm(
                self._agent_loc - self._target_loc, ord=1
            )
        }

    def _is_valid_position(self, pos):
        """
        Checks if position is in bounds or if obstacle is hit
        """
        row, col = pos
   
        # If agent goes out of the grid
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols:
            return False

        # If the agent hits an obstacle
        if self.maze[row, col] == '1':
            return False
        
        return True

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        """
        Renders a frame in pygame
        """
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.num_cols
        )  # The size of a single grid square in pixels

        # Draw target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_loc,
                (pix_square_size, pix_square_size),
            ),
        )
        # Draw start
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * self._start_loc,
                (pix_square_size, pix_square_size),
            ),
        )
        # Draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_loc + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        # Draw horizontal lines
        for y in range(self.num_rows + 1):
            pygame.draw.line(canvas, 0, (0, y * pix_square_size),
                             (self.window_size, y * pix_square_size))

        # Draw vertical lines
        for x in range(self.num_cols + 1):
            pygame.draw.line(canvas, 0, (x * pix_square_size, 0),
                             (x * pix_square_size, self.window_size))
        
        # Draw obstacles
        obs = np.where(self.maze == '1')
        for i in range(obs[0].size):
            row = obs[0][i]
            col = obs[1][i]

            cell_left = col * pix_square_size
            cell_top = row * pix_square_size

            # Draw Obstacle
            if self.maze[row, col] == '1':  # Obstacle
                pygame.draw.rect(
                    canvas,
                    (0, 0, 0),
                    pygame.Rect(
                        (cell_left, cell_top),
                        (pix_square_size, pix_square_size),
                    ),
                )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    

class MazeEnv5x5(MazeEnv):
    def __init__(self):
        super(MazeEnv5x5, self).__init__(maze_file="maze2d_5x5.npy")

class MazeEnvHairpin(MazeEnv):
    def __init__(self):
        super(MazeEnvHairpin, self).__init__(maze_file="hairpin_14x14.npy")

class MazeEnvTolmanV0(MazeEnv):
    def __init__(self):
        super(MazeEnvTolmanV0, self).__init__(maze_file="tolman_9x9_v0.npy")

class MazeEnvTolmanV1(MazeEnv):
    def __init__(self):
        super(MazeEnvTolmanV1, self).__init__(maze_file="tolman_9x9_v1.npy")

if __name__ == '__main__':
    # Test it out
    env = MazeEnv(maze_file="maze2d_5x5.npy")
    # env = MazeEnv(maze_file="hairpin_14x14.npy")
    print(f"env: {env}")
    print(f"start loc: {env._start_loc}, target loc: {env._target_loc}")
    obs, info = env.reset()
    print(f"Post reset obs: {obs}, info: {info}")
    rand_action = env.action_space.sample()
    print(f"Random action: {rand_action}")
    obs, reward, term, _, info = env.step(rand_action)
    print(f"Post step obs: {obs}, reward: {reward}, terminated: {term}, info: {info}")
    # obs = env.reset()
    # done = True
    # while done:
    #     env.render()
    #     pygame.time.wait(5000)
    #     done = False