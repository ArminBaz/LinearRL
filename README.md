# LinearRL - Python Implementation
This repository is a Python implementation of ["Linear reinforcement learning in planning, grid fields, and cognitive control"](https://www.nature.com/articles/s41467-021-25123-3) by Piray et al.

## Introduction
The code converts the LinearRL framework introduced by Piray et al. into a Python framework. Additionally, it is converted to be used with OpenAI's Gym reinforcement learning environments. <br> <br>
Currently, the framework is designed to work with tabular environments. <br> <br>
**Please Note** although the code structure is made to handle gym-like function calls. This code will not be compatible with any environment. It has been speficially constructed for tabular environments that we have created, see `gym-env/` for more details.

## Usage
### Conda Environment
I recommend creating a conda environment for usage with this repo. Especially because we will be installing some custom built environments. You can install the conda environment from the yml file I have provided.
```bash
conda env create -f env.yml
```

### Environments
Because we are using custom gym environments, you need to install them locally in order for gymansium to recognize them when we are constructing our environment variables. To install the environemnts, just run:
```bash
pip install -e gym-env
```

### Configuration
To configure the code with your desired parameters you can edit the `config.yml` file in `src/`. <br>
For example, you can change the `ENV` flag with your desired gym environment.

### Run
Examples of using the model can be found in the notebooks. There are two notebooks, one containing a version of the Tolman detour task, the other containing a reward revaluation task.

### Replanning
For the replanning tasks, you can find examples in the two notebooks `lrl-reward.ipynb` and `lrl-detour.ipynb`. The first considers the Tolman latent learning problem where the goal state is moved after learning. 
The second considers the problem adding a barrier to the environment after learning.