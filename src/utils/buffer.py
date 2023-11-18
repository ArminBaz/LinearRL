import random
from collections import namedtuple

import numpy as np


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    Replay Buffer as described in the DQN paper
    """
    def __init__(self, capacity=10000, batch_size=32):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.batch_size = batch_size

    def push(self, experience):
        """Add an experience to the replay buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        """Sample a batch of experiences from the replay buffer."""
        return random.sample(self.buffer, self.batch_size)

    def __len__(self):
        """Return the current size of the replay buffer."""
        return len(self.buffer)
    

class PrioritizedReplayBuffer:
    """
    Prioritized Replay Buffer as described in "Prioritized Experience Replay" paper
    """
    def __init__(self) -> None:
        pass

import torch

if __name__ == '__main__':
    # Test it out
    # Create a replay buffer with a capacity of 1000 and batch size of 32
    buffer_capacity = 20
    batch_size = 5
    replay_buffer = ReplayBuffer(buffer_capacity, batch_size)

    # Fill the replay buffer with dummy experiences
    for i in range(buffer_capacity):
        state = torch.tensor(tuple(np.random.rand(4))).unsqueeze(0)  # Treating states like I would if this is were being implemented (pytorch tensor with batch dim)
        action = np.random.randint(0, 2)  # Assuming two possible actions
        reward = np.random.rand()
        next_state = tuple(np.random.rand(4))
        done = False

        experience = Experience(state, action, reward, next_state, done)
        replay_buffer.push(experience)

    # Sample a batch from the replay buffer
    batch = replay_buffer.sample()

    # Print the sampled batch
    # for experience in batch:
    #     print(experience)

    batch = Experience(*zip(*batch))
    batch_state = torch.cat(batch.state)


    print(batch_state)
    print(batch_state.shape)