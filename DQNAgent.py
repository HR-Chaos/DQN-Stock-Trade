from collections import deque
import random
from DQN import DQN
import torch
import torch.nn as nn
import numpy as np

class DQNAgent:
    def __init__(self, state_size):
        self.state_size = state_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.memory = deque(maxlen = 2000)
        self.gamma = 0.95   # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.001)
        
    def remember(self, state, action, reward, next_state, done):
        # store data into memory
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        state = torch.FloatTensor(state).to(self.device) # Move to device
        if random.random() <= self.epsilon:     # explore
            return np.random.uniform(-1, 1) # Random action
        action = self.model(state).item()
        return action
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device) # Move to device
            # next_state = torch.FloatTensor(next_state).to(self.device) # Move to device
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).to(self.device) # Move to the same device as the model
                target += self.gamma * torch.max(self.model(next_state_tensor)).item()
            target = torch.tensor([target], dtype=torch.float32).to(self.device) # Convert target to tensor
            current_q = self.model(state)
            
            # backprop and update model parameters
            loss = nn.MSELoss()(current_q, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        