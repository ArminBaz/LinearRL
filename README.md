# LinearRL - Python Implementation
This repository is a Python implementation of ["Linear reinforcement learning in planning, grid fields, and cognitive control"](https://www.nature.com/articles/s41467-021-25123-3) by Piray et al. <br> <br>

**Another Note**: Althought I believe this to be a faithful reimplementation of the original paper. It is *not* an official implementation, for the official implementation please visit [this repository](https://github.com/payampiray/LinearRL/).

## Introduction
The code converts the LinearRL framework introduced by Piray et al. into Python. Additionally, it is converted to be used with environments that have a similar structure to OpenAI's Gym reinforcement learning environments. <br> <br>
**Please Note** although the code structure is made to handle gym-like function calls. This code will not be compatible with any environment. It has been specifically designed for the tabular environments that I have created, see `gym-env/` for more details. <br> <br>

## Conda & Gym Environments
### Conda Environment
I recommend creating a conda environment for usage with this repo. Especially because we will be installing some custom built environments. You can install the conda environment from the yml file I have provided.
```bash
conda env create -f env.yml
```

### Gym Environments
Because we are using custom gym environments, you need to install them locally in order for gymansium to recognize them when we are constructing our environment variables. To install the environemnts, just run:
```bash
pip install -e gym-env
```

## Usage
### Notebooks
Examples of using the model can be found in the notebooks. There are two notebooks, one containing a version of the Tolman detour task, the other containing a reward revaluation task.

The notebook with reward revaluation planning can be found in `src/linear-rl-reward.ipynb`. <br>
The notebook with the Tolman detour planning task can be found in `src/linear-rl-det.ipynb`.