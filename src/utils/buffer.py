import random
from collections import namedtuple


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