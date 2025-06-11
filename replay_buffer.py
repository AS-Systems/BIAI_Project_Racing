import numpy as np
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

        #saving experience
    def push(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done))

        #loading random sample of experience for learning
    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, action, reward, next_states, dones = zip(*samples)
        return np.stack(states), action, reward, np.stack(next_states), dones

    def __len__(self):
        return len(self.buffer)