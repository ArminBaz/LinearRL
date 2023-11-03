# Gymnasium Environments
This directory contains the code I used to create custom gymnasium environments. <br> <br>

The environments in here are all either environments that were used in the original linear RL paper to demonstrate its effectiveness or cognitive neuroscience inspired environments. <br> <br>

Note that the mazes *can* be constructed with different values for height and width. However, `render_mode` *must* be set to `None` as I haven't added that functionality for rendering yet.

## Usage
The layout of this directory and for creating my own custom gym environments was constructed by following Gymnasium's [tutorial](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#make-your-own-custom-environment). <br> <br>

It requires to create a pip package-like file structure because we are importing our environments locally in order to create an environment instance without having to construct the environment everytime we want to use it. <br> <br>

Following up on this point, if you would like to use the custom environments then you need to run the command:
```bash
pip install -e gym-env
```

## Adding your own environment
If you would like to add your own environment, make sure it fits within the way I have defined the environments. For example, the maze environment class expects a .npy file which contains 0s to indicate open states, 1s to indicate blocked state, 'S' to indicate the starting state, and 'G' to indicate any terminal state. <br> <br>

After you have added your own environment file, make sure you construct a class instantiaion of it inside of `gym_env/envs/maze_env.py`, import the class in `gym_env/envs/__init__.py`, and register it inside of `gym_env/__intit__.py`. <br> <br>

Ater you've done all this, you need to remake the pip installation of our custom gym env by running the command outlined in the previous section.

## From Paper
### 10x10 Maze
I create a similar maze to that was used in the linear RL paper (fig 2d.)

## Cognitive Neuroscience
### Hairpin Maze
Hairpin maze inspired by [Derdikman et al.](https://www.nature.com/articles/nn.2396)

### Tolman's Detour Task
Maze environment inspired by [Tolman's detour task ](https://psycnet.apa.org/record/1949-00103-001).