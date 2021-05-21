import math
import numpy as np

import torch

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

np.random.seed(1)


class RandomAgent:
    """This class implements the random walking agent using the network model"""

    def __init__(self, model, categorical_actions, spatial_actions, id_from_actions, action_from_id):
        self.states = []
        self.rewards = []
        self.actions = []
        self.points = []
        self.score = []
        # self.policy_predictions=[]
        # self.spatial_predictions=[]
        self.gamma = 0.95  # discount rate
        self.categorical_actions = categorical_actions
        self.spatial_actions = spatial_actions
        self.model = model
        # self.epsilon = 0.5
        self.id_from_actions = id_from_actions
        self.action_from_id = action_from_id

    def update_epsilon(self):
        if self.epsilon > 0.1:
            self.epsilon = 0.999 * self.epsilon

    def append_sample(self, state, action, point, reward, score):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)
        self.points.append(point)
        self.score.append(score)
        return [state, action, point, reward, score]

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def act(self, state, init=False, epsilon=0.5):
        ep = np.random.random()
        if not init or ep < epsilon:
            # return self.action_from_id[np.random.choice(len(self.action_from_id), 1)[0]], np.random.randint(4096)
            return np.random.choice(list(self.action_from_id.values()), 1), np.random.randint(4096)
        else:
            state.append(np.zeros((1, 1)))
            preds = self.model(state)
            # return self.action_from_id[np.random.choice(len(self.action_from_id), 1, p=preds[1][0])[0]], np.random.choice(len(self.action_from_id), 1, p=preds[2][0])[0]
            # return self.action_from_id[np.random.choice(len(self.action_from_id), 1, p=preds[0].cpu().detach().numpy())[0]], np.random.choice(4096, 1, p=preds[1].cpu().detach().numpy())[0]
            return np.random.choice(list(self.action_from_id.values()), 1, p=preds[0].cpu().detach().numpy())[0], np.random.choice(4096, 1, p=preds[1].cpu().detach().numpy())[0]
            
    def act_randomly(self):
        return self.action_from_id[np.random.choice(
            len(self.action_from_id), 1
        )[0]], np.random.randint(4096)

    def train(self):
        episode_length = len(self.states)
        discounted_rewards = self.discount_rewards(self.rewards)
        # Standardized discounted rewards
        """discounted_rewards -= np.mean(discounted_rewards) 
        if np.std(discounted_rewards):
            discounted_rewards /= np.std(discounted_rewards)
        else:
            self.states, self.actions, self.rewards = [], [], []
            #print ('std = 0!')
            return 0"""

        update_inputs = [np.zeros((episode_length, 27, 64, 64)),
                         np.zeros((episode_length, 11, 64, 64)),
                         np.zeros((episode_length, 11, ))
                         # np.zeros((episode_length, 1))
                         ]  # Episode_lengthx64x64x4

        # Episode length is like the minibatch size in DQN
        for i in range(episode_length):
            update_inputs[0][i, :, :, :] = self.states[i][0][0, :, :, :]
            update_inputs[1][i, :, :, :] = self.states[i][1][0, :, :, :]
            update_inputs[2][i, :] = self.states[i][2][0, :]

        r = np.vstack(self.rewards)
        update_inputs.append(np.zeros((episode_length, 1)))

        values = self.model.predict(update_inputs)[0]
        r = r + self.gamma * values
        update_inputs[3] = r
        advantages_actions = np.zeros((episode_length, len(self.id_from_actions)))
        advantages_space = np.zeros((episode_length, 4096))

        for i in range(episode_length):
            advantages_actions[i][self.actions[i]] = discounted_rewards[i] - float(values[i])
            advantages_space[i][self.points[i]] = discounted_rewards[i] - float(values[i])
        self.model.fit(update_inputs, [discounted_rewards, advantages_actions, advantages_space], epochs=3, verbose=2)

        self.states, self.actions, self.rewards = [], [], []

        self.update_epsilon()

    def load(self, name):
        if self.model:
            self.model.load_state_dict(torch.load(name))

    def save(self, name):
        if self.model:
            torch.save(self.model.state_dict(), name)
