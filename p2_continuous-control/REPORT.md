## Summary/Introduction
This document is a summary/report of the agent, models and framework that applies 
Deep Deterministic Policy Gradients to train an agent that can solve the continuous control/reach task. 

    
## Learning Algorithm
 To train the agents, I used the DDPG learning algorithm. An agent trained using this 
 algorithm actually trains two networks, an "actor" network, which encodes a deterministic
 policy, and a "critic" network, which helps evaluate the value of taking an action according
 to this critic network/policy in a given state, and following the deterministic policy into the future.
 
 The parameters of the Q-network can be learned similar to DQN and Q-learning. The parameters of the 
 actor/policy-network are learned by using the (local) Q-network effectively to define the loss. We want
 to find the parameters of of the policy that maximize predicted future returns (Q-values!).
 
 We train and learn the parameters and weights of the neural networks approximating 
 a actor (policy network) and critic (value network) via stochastic gradient descent. We use the 
 Adam optimizer, with the learning rates for each network differing below. 
  
  
  Other hyper-parameter choices for learning can be found in the `ddpg_agent.py` file, 
  but are included here for reference as well:
  ```
    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 128        # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR_ACTOR = 1e-4         # learning rate of the actor 
    LR_CRITIC = 1e-3        # learning rate of the critic
    WEIGHT_DECAY = 0     # No L2 weight decay
  ```
  
  These parameters *largely* follow the hyper-parameters used in the original DDPG paper, but differ
  in a few spots. Because we're solving a relatively small problem and was training locally, I decreased
  the buffer size by a factor of 10. I also was able to get away with a larger batch size (128 vs 64).
  Also, I didn't apply any L2 weight decay to the weights of the critic.
  
  The parameter TAU above is used for applying soft-updates to the actor and critic networks.
  Specifically, the parameters of the target network are a convex combination of 
  the current parameter values and the current param values of the online network,
  The weight of the online network weights is given by Tau, so that the network weights
  of the target network do not change dramatically during an individual update. This helps
  stabilize the training process, and was also used in project 1. It's neat to see that 
  the same trick for stabilizing training works in both DQN and DDPG. 
  
  Similar to the original paper, this agent adds noise within action space, to help 
  encourage exploration. Specifically, the action has Ornstein-Uhlenbeck noise added, but 
  the noise which is ideal for a problem will likely depend on that problem's settings.
  
### Network Architectures
a) Actor Network: This network has 3 layers, with 2 hidden fully connected layers (with `relu` activations).
For each layer, batch norm is applied after the layer but before applying it's activation. As mentioned
in the DDPG paper, when applied to the inputs, batch norm is useful for controlling the different scales of 
states across tasks. The final output layer has a `tanh` activation function. Each hidden layer 
has 128 units, the output layer has dimension 4 (the dimension of the action space.
b) Critic Network: This network also has 3 layers (2 fully connected hidden layers, each with 128 neurons). 
After the first hidden, the actions are concatenated to the representation of the network. Batchnormalization
is applied in the critic, but only on layers before the actions are added, following the original implementation.
The output of the critic is a single number, the expected reward/value.


Overall, shrinking the layers to have 128 units really helped speed training on the CPU. Initially I experiemented
with much larger layers, but training was quite slow. This was a nice balance of making progress without 
giving up too much performance.

## Reward Progress

Included below is a plot of the score, averaged over the most recent 100 episodes. 
We can observe that after about episode 100, the agent has an avg score > 30.

[Score Evolution](ddpg_training_history.png)

[Score Evolution]: ddpg_training_history.png " Score Training"

Included in this repo are check-pointed weight files of both the actor and the critic at episode 200.
Each file name follows the convention `checkpoint_<actor/critic>_ep-no_<episode_number>.pth`.


## Ideas for Future Work
There are several ways that this agent can be improved, some of which were included and detailed
in the course. To reach this initial solving, I implemented some of the changes that were mentioned
 in the getting started of the project description. Specifically, I clipped clipped the norm of the
 gradient of the critic network, and reduced the frequency of updates, to 10 updates every 20 steps.
 This turned out to be enough to successfully solve it.  
 
 Algorithmically, I'd be interested in implementing something like PPO, to benchmark against the results
 with DDPG. I'm also interested by the simplicity of parameter noise (rather than action noise) to help 
 exploration. The results presented [here](https://openai.com/blog/better-exploration-with-parameter-noise/) are quite compelling.
