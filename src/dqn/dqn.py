import os
import random
from itertools import count
from collections import namedtuple

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.buffer import ReplayBuffer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class QNetwork(nn.Module):
    """
    Deep Q Network, defined in the same way as DQN paper
    """
    def __init__(self, input_size, n_actions):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)
        
    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)


class DQN:
    """
    DQN Algorithm
    """
    def __init__(self, 
                 env_name, 
                 n_episodes, 
                 gamma, 
                 epsilon,
                 learning_rate, 
                 input_size,
                 target_update=1000,
                 initial_memory=1000,
                 num_actions=4,
                 memory_size=100000, 
                 batch_size=32):
        """
        
        """
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = learning_rate
        self.n_channels = input_size
        self.n_actions = num_actions
        self.initial_memory = initial_memory
        self.target_update = target_update
        self.memory_size = memory_size
        self.batch_size = batch_size

        # Make environment
        self.env = gym.make(env_name)

        # Make replay buffer
        self.buffer = ReplayBuffer(memory_size, batch_size)

        # Make target and policy networks
        self.policy_net = QNetwork(input_size, num_actions)
        self.target_net = QNetwork(input_size, num_actions)
        self._update_target_network()
        self.target_net.eval()

        # Make optimizer
        self.opt = optim.Adam(self.policy_net.parameters(), lr=self.lr)
    
    def _update_target_network(self):
        """Update the target Q-Network by copying the policy network's weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _get_state(self, obs):
        """Typical operations when using Images with PyTorch"""
        return torch.from_numpy(np.array(obs)).float()
    
    def _select_action(self, state):
        """Epsilon Greedy action selection"""
        if np.random.uniform(0,1) < self.epsilon:
            # Choose a random action
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)
        else:
            # Choose maximizing action
            with torch.no_grad():
                return self.policy_net(state.to('cuda')).max(1)[1].view(1, 1)
    
    def train(self, render=False):
        for ep in range(self.n_episodes):
            obs, info = self.env.reset()
            state = self._get_state(obs)
            total_reward = 0.0
            steps_done = 0

            for t in count():
                action = self._select_action(state)
                steps_done += 1
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                if render:
                    self.env.render()

                if not done:
                    next_state = self._get_state(obs)
                else:
                    next_state = None
                
                self.buffer.push(state, action, reward, next_state, done)
                state = next_state

                if steps_done > 100:
                    self._optimize_model()

                    if steps_done % self.target_update == 0:
                        self._update_target_network()

                if done:
                    break
        
            if ep % 20 == 0:
                print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(steps_done, ep, t, total_reward))

        self.env.close()
        return