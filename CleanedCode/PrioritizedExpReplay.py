import numpy as np
import torch

from SumTree import SumTree

# to run distributed training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

# Import classes
from SumTree import SumTree

# The prioritized experience replay buffer is used to store the experiences and sample the experiences based on their priorities.
class PrioritizedExpReplay():    
    def __init__(self, num_features, window_size, device, buffer_size=1000000, alpha=0.6, beta=0.4): # initialize the prioritized experience replay buffer
        self.tree = SumTree(buffer_size) # initialize the sumtree
        self.alpha = alpha # initialize the alpha value which is used to control the amount of prioritization
        self.beta = beta    # initialize the beta value which is used to control the amount of importance sampling
        self.device = device    # initialize the device (GPU)
        self.buffer_size = buffer_size # initialize the capacity of the sumtree
        self.window_size = window_size # initialize the window size
        self.num_features = num_features    # initialize the number of features

    def add_exp(self, state, action, reward, next_state, done):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = 1
        exp = (state, action, reward, next_state, done)
        self.tree.add(exp, max_priority ** self.alpha)

    def sample_exp(self, batch_size=64):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, exp = self.tree.get(s)
            priorities.append(p)
            batch.append(exp)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= np.max(is_weights)

        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        is_weights = torch.tensor(is_weights, dtype=torch.float32).to(self.device)
        
        return states, actions, rewards, next_states, dones, idxs, is_weights

    def update_priorities(self, idxs, td_errors):
        priorities = np.abs(td_errors) + 1e-5
        for idx, priority in zip(idxs, priorities):
            self.tree.update(idx, priority ** self.alpha)
