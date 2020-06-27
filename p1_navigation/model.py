import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.hidden = 128
        self.fc1 = nn.Linear(state_size, self.hidden)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.relu2 = torch.nn.ReLU()        
        self.fc_final = nn.Linear(self.hidden, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        output = self.fc_final(x)
        return output        
