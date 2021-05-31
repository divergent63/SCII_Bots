# This runner is based on actions.RAW_FUNCTIONS (.\anaconda\a3_64\envs\py37_clone_v8\Lib\site-packages\pysc2\lib\actions.py)
import random
import numpy as np
import pandas as pd
import os, sys
from absl import app, logging
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop

import torch
from torch.autograd import Variable

sys.path.append('../')
import models.models as models


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9):
        self.actions = actions
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation, e_greedy=0.9):
        self.check_state_exist(observation)
        if np.random.uniform() < e_greedy:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(
                state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.reward_decay * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.learning_rate * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions),
                                                         index=self.q_table.columns,
                                                         name=state))


class DeepQLearning:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9):
        self.actions = actions
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

        self.feature_screen = 27
        self.feature_minimap = 11
        self.out_action = 12
        self.out_point = 4096

        model = models.SimpleConvNet_prob(input_size=[self.feature_screen, self.feature_minimap], output_size=[self.out_action, self.out_point])
        self.model = model.cuda() if torch.cuda.is_available() else model

    def choose_action(self, state, init=False, e_greedy=0.2):
        ep = np.random.random()
        if not init and ep < e_greedy:  # TODO: can't understand the probability matrix from uniform distribution --> SOLVED
            # return self.action_from_id[np.random.choice(len(self.action_from_id), 1)[0]], np.random.randint(4096)
            p1 = Variable(torch.Tensor(
                np.random.dirichlet(np.ones(self.out_action), size=1)).cuda() if torch.cuda.is_available() else torch.Tensor(
                np.random.dirichlet(np.ones(self.out_action), size=1)))
            p2 = Variable(torch.Tensor(
                np.random.dirichlet(np.ones(self.out_point), size=1)).cuda() if torch.cuda.is_available() else torch.Tensor(
                np.random.dirichlet(np.ones(self.out_point), size=1)))
            print("explored actions")
            return [p1, p2]
        else:
            # state.append(np.zeros((1, 1)))
            preds = self.model(state)
            print("learned actions")
            # return self.action_from_id[np.random.choice(len(self.action_from_id), 1, p=preds[1][0])[0]], np.random.choice(len(self.action_from_id), 1, p=preds[2][0])[0]
            # return self.action_from_id[np.random.choice(len(self.action_from_id), 1, p=preds[0].cpu().detach().numpy())[0]], np.random.choice(4096, 1, p=preds[1].cpu().detach().numpy())[0]
            # return np.random.choice(list(self.action_from_id.values()), 1, p=preds[0].cpu().detach().numpy())[0], \
            #        np.random.choice(4096, 1, p=preds[1].cpu().detach().numpy())[0]
            return preds

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.reward_decay * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.learning_rate * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions),
                                                         index=self.q_table.columns,
                                                         name=state))
