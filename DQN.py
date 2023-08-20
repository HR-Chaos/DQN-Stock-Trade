import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_dim):
        super(DQN, self).__init__()
        # CURRENT: Simple architecture
        # TODO: make more complex architecture
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.fc(state)