import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from time import time
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from Historic_Crypto import HistoricalData
from Historic_Crypto import Cryptocurrencies
from Historic_Crypto import LiveCryptoData
import yfinance as yf
import os

# to run distributed training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from DDDQN import DDDQN
from PrioritizedExpReplay import PrioritizedExpReplay

class Agent():

    # Initializes the agent with various parameters such as window_size, gamma, number of episodes, learning rate, minimum epsilon, and update interval. 
    # It also initializes the device, portfolio, epsilon, training step counter, memory, batch size, online and target networks, optimizer, scheduler, 
    # and the holding time.
    def __init__(self, data_shape, num_episodes, window_size=30, gamma=0.99, update_interval=30, lr=0.001, min_epsilon=0.02):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window_size = window_size  # number of previous prices to use as state
        self.data_shape = data_shape  # [num_features, window_size]
        self.portfolio = [0, 0, 0, 0]  # [total eth, cash_held, total_portfolio_value (eth value + cash held - initial investment), max ether held so far]
        self.inventory = [] 
        self.gamma = gamma  # discount factor for future rewards
        self.num_episodes = num_episodes    # number of episodes
        self.epsilon = 1.0  # exploration rate
        self.min_epsilon = min_epsilon   # minimum exploration rate
        self.update_interval = update_interval  # number of steps between target network updates
        self.trainstep = 0
        # self.memory = ExpReplay(self.window_size, data_shape[1], self.device)        
        self.memory = PrioritizedExpReplay(self.window_size, data_shape[1], self.device)
        self.batch_size = 64
        self.online_net = DDDQN(self.data_shape[1], window_size).to(self.device)
        self.target_net = DDDQN(self.data_shape[1], window_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.holding_time = 0 # Number of timesteps for which the agent has held the position
        initial_learning_rate = lr
        decay_steps = self.num_episodes * self.data_shape[0] // 10  # You can adjust the divisor to control the decay rate
        decay_rate = 0.9  # You can adjust this value to control the decay rate
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=initial_learning_rate)
        self.scheduler = ExponentialLR(self.optimizer, gamma=decay_rate)
        self.criterion = nn.MSELoss()

    # CURRICULUM TRAINING PART 1
    def get_action(self, state):
            state_numpy = np.array(state)
            state_tensor = torch.FloatTensor(state_numpy).to(self.device).unsqueeze(0)
            # create a random number between 0 and 1 and compare it to epsilon for our epsilon-greedy algorithm
            if np.random.rand() <= self.epsilon:
                # Select a random action out of the list of available actions
                if len(self.inventory) > 0:  # if we have inventory to sell
                    return np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8])
                else:
                    return np.random.choice([4, 5, 6, 7, 8])  # we can only buy or hold
            else:
                with torch.no_grad():
                    self.online_net.eval()
                    actions = self.online_net(state_tensor)
                    # actions = self.online_net.advantage(state_tensor)
                    self.online_net.train()

                    if len(self.inventory) > 0:
                        action = torch.argmax(actions[0, :]).item()
                    else: # if we don't have inventory to sell, we remove that action option
                        action = torch.argmax(actions[0, 4:]).item() + 4
                    return action

    #  This method softly updates the target network by blending the weights of the online network and target network using the parameter tau.
    def update_target(self, tau=0.001):
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    #  Linearly decays the epsilon value from its initial value to the minimum epsilon value over the course of the total training steps.
    def update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            # b = self.min_epsilon**(1/(self.num_episodes*self.data_shape[0]))
            # self.epsilon = b**self.trainstep # exponential decay
            self.epsilon -= (1.0 - self.min_epsilon) / (self.num_episodes * self.data_shape[0])   # linear decay

    # Trains the online network using the samples from the prioritized experience replay memory. 
    # It updates the target network periodically based on the update_interval. 
    # It also calculates the loss, applies the importance sampling weights, and updates the priorities in the buffer.
    def train(self):
        if self.memory.tree.n_entries < self.batch_size:
            return

        if self.trainstep % self.update_interval == 0:
            self.update_target()

        states, actions, rewards, next_states, dones, idxs, is_weights = self.memory.sample_exp(self.batch_size)

        q_next_state_online_net = self.online_net(next_states)
        q_next_state_target_net = self.target_net(next_states)

        max_action = torch.argmax(q_next_state_online_net, dim=1).to(self.device)

        batch_index = torch.arange(self.batch_size, dtype=torch.int64).to(self.device)

        q_predicted = self.online_net(states)
        q_target = q_predicted.clone().detach()

        q_target[batch_index, actions] = rewards + self.gamma * q_next_state_target_net[batch_index, max_action] * dones

        td_errors = q_target - q_predicted
        td_errors = td_errors.detach().cpu().numpy()
        td_errors = td_errors[np.arange(self.batch_size), actions.cpu().numpy()]

        # Update priorities in the buffer
        self.memory.update_priorities(idxs, td_errors)

        # loss = self.criterion(q_predicted, q_target)
        # loss *= is_weights  # Apply importance sampling weights
        # loss = loss.mean()

        loss = self.criterion(q_predicted, q_target).view(-1, 1)
        loss = (loss * is_weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_epsilon()
        self.trainstep += 1
        return loss.item()

    # This method is called to save the model parameters of the online and target networks in the Output folder.
    def save_model(self):
        import os

        output_directory = "Output"
        online_model_directory = os.path.join(output_directory, "online_model")
        target_model_directory = os.path.join(output_directory, "target_model")

        os.makedirs(online_model_directory, exist_ok=True)
        os.makedirs(target_model_directory, exist_ok=True)

        torch.save(self.online_net.state_dict(), os.path.join(online_model_directory, 'model.pt'))
        torch.save(self.target_net.state_dict(), os.path.join(target_model_directory, 'model.pt'))

    # This method is called to load the model parameters of the online and target networks from the Output folder.
    def load_model(self):
        import os

        output_directory = "Output"
        online_model_directory = os.path.join(output_directory, "online_model")
        print("loaded model from: ", online_model_directory)
        target_model_directory = os.path.join(output_directory, "target_model")
        print("loaded model from: ", target_model_directory)


        self.online_net.load_state_dict(torch.load(os.path.join(online_model_directory, 'model.pt')))
        self.target_net.load_state_dict(torch.load(os.path.join(target_model_directory, 'model.pt')))

    
    def trade(self, t, action, eth_df, eth_df_unscaled, initial_investment, trading_fee_rate):
        eth_held, balance, previous_portfolio_value, max_ether_held_so_far = self.portfolio
        reward = 0
        if action >= 0 and action <= 3:
            # reward = eth_df_unscaled["Close"].iloc[t] - self.inventory.pop(0) # this sells the oldest bought eth
            reward = eth_df_unscaled["Close"].iloc[t] - self.inventory.pop() # this sells the most recently bought eth (worth a shot)
        elif action == 4:
            reward = -0.1
            pass
        elif action >= 5 and balance >= eth_df_unscaled["Close"].iloc[t]:
            self.inventory.append(eth_df_unscaled["Close"].iloc[t])
        return reward

    def get_state(self, t, eth_df):
        num_rows = t - self.window_size + 1
        if num_rows >= 0:
            window = eth_df.iloc[num_rows : t + 1]
        else:
            repeated_first_row = pd.concat([pd.DataFrame(np.repeat(eth_df.iloc[[0]].values, -num_rows, axis=0), columns=eth_df.columns)])
            new_data = eth_df.iloc[0 : t + 1]
            window = pd.concat([repeated_first_row, new_data], ignore_index=True)  # prevents us from sampling data that doesn't exist at the start.
        return window.values