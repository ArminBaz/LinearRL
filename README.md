# LinearRL - Python Implementation
This repository is a Python implementation of ["Linear reinforcement learning in planning, grid fields, and cognitive control"](https://www.nature.com/articles/s41467-021-25123-3) by Piray et al.

## Introduction
The code converts the LinearRL framework introduced by Piray et al. into a Python framework. Additionally, it is converted to be used with OpenAI's Gym reinforcement learning environments. <br> <br>
Currently, the framework is designed to work with tabular environments.

## Usage
### Configuration
To configure the code with your desired parameters you can edit the `config.yaml` file. <br>
For example, you can change the `ENV` flag with your desired gym environment.

### Run
To train the LinearRL model you can run `train.py` which will train a LinearRL agent on the specified gym environment.
```bash
python src/train.py
