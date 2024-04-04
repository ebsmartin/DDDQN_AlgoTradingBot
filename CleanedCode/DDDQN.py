import os

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

# to run distributed training
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, get_rank, get_world_size, destroy_process_group


def ddp_setup(rank, world_size):
    """ Setup the distributed training environment 
    
    Args:
    rank (int): rank of the current process
    world_size (int): total number of processes
    
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl", rank=rank, world_size=world_size)

class DDDQN(torch.nn.Module):
    def __init__(self, input_features, window_size):
        super().__init__()
        self.input_size = input_features * window_size  # input size
        # The use of LeakyReLU activation functions can help alleviate the vanishing gradient problem
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.1) 
        
        # Dueling DQN architecture
        self.d1 = torch.nn.Linear(self.input_size, 256)  # input layer
        self.bn1 = torch.nn.BatchNorm1d(256)  # batch normalization  used to normalize the input layer by adjusting and scaling the activations
        self.d2 = torch.nn.Linear(256, 512)  # hidden layer
        self.bn2 = torch.nn.BatchNorm1d(512)  # batch normalization used to normalize the input layer by adjusting and scaling the activations
        self.drop1 = torch.nn.Dropout(0.5)  # dropout layer used to prevent overfitting
        self.d3 = torch.nn.Linear(512, 512)  # hidden layer
        self.bn3 = torch.nn.BatchNorm1d(512)  # batch normalization used to normalize the input layer by adjusting and scaling the activations
        self.drop2 = torch.nn.Dropout(0.5)  # dropout layer used to prevent overfitting
        self.d4 = torch.nn.Linear(512, 256)  # hidden layer
        self.bn4 = torch.nn.BatchNorm1d(256)    # batch normalization used to normalize the input layer by adjusting and scaling the activations
        self.drop3 = torch.nn.Dropout(0.3)  # dropout layer used to prevent overfitting
        self.dv1 = torch.nn.Linear(256, 128)  # value hidden layer
        self.da1 = torch.nn.Linear(256, 128)  # actions hidden layer
        self.dv2 = torch.nn.Linear(128, 1)  # value output
        self.da2 = torch.nn.Linear(128, 9)  # actions output

    # forward propagation
    def forward(self, input_data):
        input_data = input_data.reshape(input_data.size(0), -1)  # Flatten the input tensor
        x = self.leaky_relu(self.d1(input_data))  # activation function
        x = self.bn1(x)  # batch normalization
        x = x.view(x.size(0), -1)  # equivalent to Flatten()
        x = self.leaky_relu(self.d2(x))  # activation function
        x = self.bn2(x)  # batch normalization
        x = self.drop1(x)  # dropout
        x = self.leaky_relu(self.d3(x))  # activation function
        x = self.bn3(x)  # batch normalization
        x = self.drop2(x)  # dropout
        x = self.leaky_relu(self.d4(x))   # activation function
        x = self.bn4(x)  # batch normalization
        x = self.drop3(x)  # dropout
        v = self.leaky_relu(self.dv1(x))  # value function
        a = self.leaky_relu(self.da1(x))  # actions function
        v = self.dv2(v)     # value output
        a = self.da2(a)   # actions output
        Q = v + (a - torch.mean(a, dim=1, keepdim=True)) # dueling Q values
        return Q   # return Q values
    
    # advantage function is used to calculate the advantage of taking each action at each state
    # processes the input state through the common layers and then the advantage stream layers.
    # used for debugging or analysis purposes.
    def advantage(self, state):
        # Common processing
        x = self.leaky_relu(self.d1(state))
        x = self.bn1(x)
        x = self.leaky_relu(self.d2(x))
        x = self.bn2(x)
        x = self.drop1(x)
        x = self.leaky_relu(self.d3(x))
        x = self.bn3(x)
        x = self.drop2(x)
        x = self.leaky_relu(self.d4(x))
        x = self.bn4(x)
        x = self.drop3(x)

        # Advantage stream processing
        a = self.leaky_relu(self.da1(x))  # actions function 
        a = self.da2(a) # actions output

        return a  # return advantage values
