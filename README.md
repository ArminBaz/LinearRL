# LinearRL - Python Implementation
This repository is a Python implementation of ["Linear reinforcement learning in planning, grid fields, and cognitive control"](https://www.nature.com/articles/s41467-021-25123-3) by Piray et al. <br> <br>

**Another Note**: Although I believe this to be a faithful reimplementation of the original paper. It is *not* an official implementation, for the official implementation please visit [this repository](https://github.com/payampiray/LinearRL/). <br>

The detour task is the only 1-1 task as presented in the paper, for the other two tasks I used inspiration from the great paper ["Predictive representations can link model- based reinforcement learning to model-free mechanisms"](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005768) by Russek et al. Even though the latent learning and change in goal tasks are not presented like this in the original LinearRL paper, the equations I used are from the original paper and still hold up.

## Introduction
The code converts the LinearRL framework introduced by Piray et al. into Python. Additionally, it is converted to be used with environments that have a similar structure to OpenAI's Gym reinforcement learning environments. <br> <br>
Although the code structure is made to handle gym-like function calls. This code will **not be compatible** with any environment. It has been specifically designed for the tabular environments that I have created, see `gym-env/` for more details. <br> <br>

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
The notebook with the Tolman detour planning task can be found in `src/linear-rl-det.ipynb`. <br>
The notebook with updating the reward of a terminal state can be found in `src/linear-rl-change-goal.ipynb`.