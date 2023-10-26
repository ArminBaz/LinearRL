# LinearRL - Python Implementation
This repository is a Python implementation of ["Linear reinforcement learning in planning, grid fields, and cognitive control"](https://www.nature.com/articles/s41467-021-25123-3) by Piray et al.

## Introduction
The code converts the LinearRL framework introduced by Piray et al. into a Python framework. Additionally, it is converted to be used with OpenAI's Gym reinforcement learning environments. <br> <br>
Currently, the framework is designed to work with tabular environments. <br> <br>
**Please Note** although the code structure is made to handle gym-like function calls. This code will not be compatible with any environment. It has been speficially constructed for tabular environments that we have created, see `gym-env/` for more details.

## Usage
### Conda Environment
I reccomend creating a conda environment for usage with this repo. Espcially because we will installing some custom built environments. You can install the conda environment from the yml file I have provided.
```bash
conda env create -f env.yml
```

### Environments
I made custom gym environments to replicate those typically found in cognitive neuroscience literature. Consequently, a lot of the results in the paper also come from such environments. <br> <br>

Because we are using custom gym environments, you need to install them locally in order for gymansium to recognize them when we are constructing our environment variables. To install the environemnts, just run:
```bash
pip install -e gym-env
```

### Configuration
To configure the code with your desired parameters you can edit the `config.yml` file in `src/`. <br>
For example, you can change the `ENV` flag with your desired gym environment.

### Run
To run the LinearRL model you can run `main.py` which will use LinearRL to solve for the optimal value function and optimal policy on the specified environment.
```bash
python src/main.py
```