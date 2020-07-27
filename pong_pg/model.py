import torch
import torch.nn as nn


class PongModel(nn.Module):
    """Pong Model"""

    def __init__(self, state_size, action_size=1, hidden_size=128, seed=0.):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(PongModel, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.hidden = hidden_size
        self.fc1 = nn.Linear(state_size, self.hidden, bias=False)
        # self.bn1 = nn.BatchNorm1d(num_features=self.hidden)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = nn.Linear(self.hidden, self.hidden, bias=False)
        # # self.bn2 = nn.BatchNorm1d(num_features=self.hidden)
        self.relu2 = torch.nn.ReLU()
        self.fc_final = nn.Linear(self.hidden, action_size, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc_final.weight)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        # x = self.bn1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        # x = self.bn2(x)
        x = self.relu2(x)

        x = self.fc_final(x)
        output = torch.sigmoid(x)
        return output
