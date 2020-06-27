## Summary/Introduction
This document is a summary/report of the model and framework that applies 
q-learning to train an agent that can solve the navigation/banana task. 

The agent trains a non-linear function approximator (e.g. neural network)
to learn an approximation to the Q-function. There are 3 components to this 
algorithm which help it 'work':

1) Target Network
    - Since we need to evaluate an estimate of the Q-values of a next-state when
    training, using the parameters of a second network, identical in architecture to 
     our network which estimates Q-values online helps to reduce the correlation 
     between the current state and the target when calculating the temporal 
     difference target. The target network parameters are updated every $C$ iterations 
     of learning.
    
2) Experience Replay
    - This technique aids in training, where state, action, reward, next_state tuples
    are stored in a memory bank, and mini-batches are sampled uniformly at random during 
    training. In this way, the temporal correlations between state-action-reward tuples 
    can be broken, and experiences the agent had previously can be 'revisited', which helps
    with sample efficiency.
    
3) Double Deep Q-Networks
    - This technique is a small modification to the calculation of the temporal
    difference target used for training. By choosing an action according to our online
    estimates of Q, but evaluating the value of that action with the target network,
    we can reduce the overestimation of our Q-values for each state-action pair.
    
## Learning Algorithm
 We apply a DQN learning algorithm . Specifically, we train and learn the parameters
  and weights of the neural network approximating Q via stochastic gradient descent 
  using the Adam optimizer, with learning of 5e-4. The algorithmic 'tweaks' to help get
  DQN to work are described above.
  
  
  Other hyper-parameter choices for learning can be found in the `dqn_agent.py` file, 
  but are included here for reference as well:
  
  BUFFER_SIZE = int(1e5)  # replay buffer size (n_tuples)
  BATCH_SIZE = 64         # minibatch size
  GAMMA = 0.99            # discount factor when calculating TD target.
  TAU = 1e-3              # for soft update of target parameters
  LR = 5e-4               # learning rate for optimizer.
  UPDATE_EVERY = 4        # how often to update the network by applying SGD.
  
  When choosing actions, the agent follows an epsilon greedy strategy, with epsilon
  starting at 1, decaying at 0.01, and decaying by 0.5% each episode.
  
  The parameter TAU above is used for the 'soft-update' of the target network.
  Specifically, the parameters of the target network are a convex combination of 
  the current parameter values and the current param values of the online network,
  The weight of the online network weights is given by Tau, so that the network weights
  of the target network do not change dramatically during an individual update.
  
### Q-Network Architecture
The Q-network which I used is a relatively simple one, as it's got 2 fully-connected
dense layers of 128 units, each with relu activations, followed by a final output
layer that's 4 units. This is very similar (e.g. the same) as used during the dqn 
exercise, albeit with larger layer size/number of units since the state and action 
space is larger and more complex for this task. 

## Reward Progress

Included below is a plot of the average score over recent 100 episodes. We can observe that
after 1300 episodes, the agent has an avg score of 15.94!

[Score Evolution](Score_Evolution.png)

There were several points during training where the agent was able to achieve an avg score 
per episode > 15, these are included in `checkpoint_files`. Each file name follows the convention
checkpoint_{episode_number}_score-{avg score}.pth.


## Ideas for Future Work
There are several ways that this agent can be improved, some of which were included and detailed
in the course. I implemented an option to improve stability of training using Double
DQN, but modifying to use Prioritized Experience Replay and Dueling Q-networks would be 
the next things I'd like to try. 


Outside of algorithmic improvements, performance can be improved with better tuning. 
I didn't modify any of the hyper-parameters from the Q-Learning exercise related to 
training (e.g. gamma, epsilon etc). I *did* make my network have a few more fully-connected 
layers to my model architecture, and increased the size of each of them to account for the larger 
state space, but didn't do much tuning there. A principled hyper-parameter tuning/search, 
perhaps leveraging a framework like `ray`, would also be valuable as a step towards improving 
performance.