""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym.
Heavily adapted from https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
"""

import numpy as np
import gym
import torch
from model import PongModel

# hyperparameters
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
resume = True  # resume from previous checkpoint?
resume_episode_number = 28700
render = False
# model initialization

D = 80 * 80  # input dimensionality: 80x80 grid
hiddens = 200
model = PongModel(state_size=D, hidden_size=hiddens).double()
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
loss = torch.nn.functional.binary_cross_entropy
if resume:
    model = PongModel(state_size=D,hidden_size=hiddens).double()
    model.load_state_dict(torch.load(f'checkpoint_files/checkpoint_{resume_episode_number}.pth'))


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None  # used in computing the difference frame
xs, hs, dlogps, drs, ys, aprobs = [], [], [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
while True:
    if render:
        env.render()

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    aprob = model(torch.from_numpy(x))
    action_prob = aprob.detach().numpy()
    action = 2 if np.random.uniform() < action_prob else 3  # roll the dice!
    rand = np.random.uniform()

    # record various intermediates (needed later for backprop)
    xs.append(x)  # observation
    y = 1 if action == 2 else 0  # a "fake label"
    aprobs.append(aprob)
    ys.append(y)
    dlogps.append(y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    if done:  # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        # epdlogp = torch.from_numpy(np.vstack(dlogps))
        epr = np.vstack(drs)
        y_array = torch.from_numpy(np.vstack(ys).astype('float64'))
        aprob_array = torch.cat(aprobs).unsqueeze(-1)
        xs, hs, dlogps, drs, ys, aprobs = [], [], [], [], [], []  # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)
        discounted_epr = torch.from_numpy(discounted_epr)
        # epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
        # y_array *= discounted_epr
        # aprob_array *= discounted_epr

        if episode_number % batch_size == 0:
            #Update params by apply grads.
            model.train()
            optimizer.zero_grad()
            # Forward pass
            # Compute Loss
            lv = loss(aprob_array, y_array, discounted_epr)
            # Backward pass
            lv.backward()
            optimizer.step()
        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        # print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        if episode_number % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode_number, reward_sum))
            # Every 100 episodes, save weights if task was solved.
            if resume:
                torch.save(model.state_dict(), f'checkpoint_files/checkpoint_{episode_number+resume_episode_number}.pth')
            else:
                torch.save(model.state_dict(), f'checkpoint_files/checkpoint_{episode_number}.pth')
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None

    # if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
    #     print('ep %d: game finished, reward: %f' % (episode_number, reward) + ('' if reward == -1 else ' !!!!!!!!'))
