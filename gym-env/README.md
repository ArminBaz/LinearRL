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


## From Paper
### 10x10 Maze
I create a similar maze to that was used in the linear RL paper (fig 2d.)

## Cognitive Neuroscience
### Hairpin Maze
Hairpin maze inspired by [Derdikman et al.](https://www.nature.com/articles/nn.2396)

### Tolman's Detour Task
Maze environment inspired by [Tolman's detour task ](https://psycnet.apa.org/record/1949-00103-001).