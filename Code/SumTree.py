import numpy as np

# to run distributed training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


# The sumtree is used to store the priorities of the experiences it receives and to sample the experiences based on their priorities.
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity   # capacity of the sumtree
        self.tree = np.zeros(2 * capacity - 1) # stores the priorities
        self.data = np.zeros(capacity, dtype=object)  # stores the experiences
        self.write = 0  # pointer to the next experience to be stored
        self.n_entries = 0  # number of experiences stored

    def _propagate(self, idx, change): # propagate the change in the priority value up the tree
        parent = (idx - 1) // 2  # parent node
        self.tree[parent] += change  # update the priority value
        if parent != 0:  # if the parent node is not the root node
            self._propagate(parent, change)  # propagate the change in the priority value up the tree

    def _retrieve(self, idx, s):  # retrieve the experience based on the priority value
        left = 2 * idx + 1  # left node
        right = left + 1  # right node
        if left >= len(self.tree):  # if the left node is a leaf node
            return idx  # return the index of the leaf node
        if s <= self.tree[left]:  # if the priority value is less than or equal to the priority value of the left node
            return self._retrieve(left, s)  # retrieve the experience from the left node
        else: # if the priority value is greater than the priority value of the left node
            return self._retrieve(right, s - self.tree[left])  # retrieve the experience from the right node

    def total(self):  # return the total priority value of the sumtree
        return self.tree[0]  # return the priority value of the root node

    def add(self, data, p):  # add the experience and its priority value to the sumtree
        idx = self.write + self.capacity - 1  # index of the leaf node
        self.data[self.write] = data  # store the experience
        self.update(idx, p)  # update the priority value of the leaf node
        self.write += 1  # increment the pointer to the next experience to be stored
        if self.write >= self.capacity:  # if the pointer to the next experience to be stored exceeds the capacity of the sumtree
            self.write = 0  # reset the pointer to the next experience to be stored
        if self.n_entries < self.capacity:   # if the number of experiences stored is less than the capacity of the sumtree
            self.n_entries += 1  # increment the number of experiences stored

    def update(self, idx, p): # update the priority value of the leaf node
        change = p - self.tree[idx] # change in the priority value
        self.tree[idx] = p # update the priority value
        self._propagate(idx, change) # propagate the change in the priority value up the tree

    def get(self, s): # retrieve the experience based on the priority value
        idx = self._retrieve(0, s)  # retrieve the experience
        dataIdx = idx - self.capacity + 1 # index of the experience
        return idx, self.tree[idx], self.data[dataIdx]  # return the index of the experience, the priority value of the experience, and the experience

