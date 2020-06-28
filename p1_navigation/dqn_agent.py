import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
from SumTree import SumTree
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
ALPHA = 0.6
BETA = 0.4
PER = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, use_ddqn=False):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.loss = F.mse_loss
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        if PER:
            self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, ALPHA, seed)
            self.learn_step = 0
            self.beta = BETA
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # DDQN
        self.ddqn = use_ddqn
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        if PER:
            self.memory.add(state, action, reward, next_state,10, done)
        else:
            self.memory.add(state, action, reward, next_state, done)
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()

                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        if PER:
            # This is for controlling how sampling experiences is done.
            # b controls if experiences are sampled unfoirmly or by priority.
            b = min(1.0, self.beta + self.learn_step * (1.0 - self.beta) / 25000)
            self.learn_step += 1

            states, actions, rewards, next_states, probabilities, dones, indices = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        self.qnetwork_local.train()
        self.optimizer.zero_grad()
        # Forward pass
        y_pred = self.qnetwork_local(states).gather(1, actions)
        if self.ddqn:
            q_vals = self.qnetwork_local(next_states).detach()
            # Find action of network
            best_act_idx = np.argmax(q_vals, axis=1)
            # Use target, and index with previous
            target_Q = self.qnetwork_target(next_states).detach().gather(1, best_act_idx.view(-1,1))
        else:
            target_Q = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        with torch.no_grad():
            target = rewards + gamma*target_Q*(1.-dones)

        if PER:
            # Compute and update new priorities
            new_priorities = (abs(y_pred - target) + 0.2).detach()
            self.memory.update_priority(new_priorities, indices)

            # Compute and apply importance sampling weights to TD Errors
            ISweights = (((1 / len(self.memory)) * (1 / probabilities)) ** b)
            max_ISweight = torch.max(ISweights)
            ISweights /= max_ISweight
            target *= ISweights
            y_pred *= ISweights

        # Compute Loss
        loss = self.loss(target, y_pred)
        # Backward pass
        loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples and sample from by TD target."""

    def __init__(self, action_size, buffer_size, batch_size, alpha, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            alpha (float):  reliance of sampling on prioritization (0= uniform).
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = SumTree(buffer_size)
        self.batch_size = batch_size
        self.alpha = alpha
        self.max_priority = 0.
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "priority", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, priority, done):
        """Add a new experience to memory."""
        # Assign priority of new experiences to max priority to insure they are played at least once
        if len(self.memory) > self.batch_size + 5:
            e = self.experience(state, action, reward, next_state,  self.max_priority, done)
        else:
            e = self.experience(state, action, reward, next_state, int(priority) ** self.alpha, done)
        self.memory.add(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = []
        indices = []
        sub_array_size = self.memory.get_sum() / self.batch_size
        for i in range(self.batch_size):
            choice = np.random.uniform(sub_array_size * i, sub_array_size * (i + 1))
            e, index = self.memory.retrieve(1, choice)
            experiences.append(e)
            indices.append(index)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        probabilities = torch.from_numpy(
            np.vstack([e.priority / self.memory.get_sum() for e in experiences])).float().to(device)
        indices = torch.from_numpy(np.vstack([i for i in indices])).int().to(device)

        return (states, actions, rewards, next_states, probabilities,dones, indices)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def update_priority(self, new_priorities, indices):
        """Updates priority of experience after learning."""
        for new_priority, index in zip(new_priorities, indices):
            old_e = self.memory[index]
            new_p = new_priority.item() ** self.alpha
            new_e = self.experience(old_e.state, old_e.action, old_e.reward, old_e.next_state,new_p, old_e.done)
            self.memory.update(index.squeeze(-1).numpy(), new_e)
            if new_p > self.max_priority:
                self.max_priority = new_p
