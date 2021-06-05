import random
import numpy as np
import pickle
import pandas as pd
import os, sys
from absl import app, logging
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop

from pathlib import Path
from matplotlib import pyplot as plt

import torch
from torch.autograd import Variable
from torch import nn

import models.models as models

sys.path.append('../')


class SupervisedDeepLearning:
    def __init__(self, model_path, learning_rate=0.01, reward_decay=0.9):
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay

        self.feature_screen = 27
        self.feature_minimap = 11
        self.out_action = 12
        self.out_point = 4096

        # model_p = models.SimpleConvNet_prob(input_size=[self.feature_screen, self.feature_minimap], output_size=[self.out_action, self.out_point])
        model_v = models.SimpleConvNet_val(input_size=[self.feature_screen, self.feature_minimap], output_size=[self.out_action, self.out_point])
        self.model = model_v.cuda() if torch.cuda.is_available() else model_v
        print('model:  \n', self.model)

        self.criterion = nn.MSELoss()

        # model_path = Path(Path(os.getcwd()) / 'save' / 'dqn' / 'Simple64-dqn-best.pt')
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path))

        self. critic_optim = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def learn(self, history_raw, id_from_actions, epochs=3):
        batch_size = len(history_raw )//40
        critic_network_loss_lst = []
        # history_raw: [e, time, state_model, state_model_next, action, actual_action, last_action, point, reward, score, done]
        for epoch in range(epochs):
            history = random.sample(history_raw, batch_size)
            e, time, state_model, state_model_next, action, actual_action, last_action, point, reward, score, done = zip \
                (*history)
            # for _ in range(len(history)):
            #     idx = random.randint(0, len(history) - 1)

            state_tensor_lst = [[Variable
                (torch.Tensor(state_model[b][i]).float()).cuda() if torch.cuda.is_available() else Variable
                (torch.Tensor(state_model[b][i]).float()) for i in range(3)] for b in range(len(history))]
            state_tensor_batch_lst = [torch.cat([state_tensor_lst[b][i].unsqueeze(0) for b in range(len(history))]) for i in range(3)]

            next_state_tensor_lst = [[Variable
                (torch.Tensor(state_model_next[b][i]).float()).cuda() if torch.cuda.is_available() else Variable
                (torch.Tensor(state_model_next[b][i]).float()) for i in range(3)] for b in range(len(history))]
            next_state_tensor_batch_lst = \
                [torch.cat([next_state_tensor_lst[b][i].unsqueeze(0) for b in range(len(history))]) for i in range(3)]

            q_predict = self.model(state_tensor_batch_lst)
            q_predict_lst = [self.model(state_tensor_batch_lst)[i].detach().cpu().numpy() for i in range(2)]  # state
            q_predict_nxt = self.model(next_state_tensor_batch_lst)
            q_predict_nxt_lst = [self.model(next_state_tensor_batch_lst)[i].detach().cpu().numpy() for i in
                                 range(2)]  # next_state

            r_with_a = np.zeros((batch_size, len(id_from_actions)))
            r_with_ap = np.zeros((batch_size, 4096))
            for b in range(len(history)):
                r_with_a[b][id_from_actions[actual_action[b]]] = reward[b][0]
                r_with_ap[b][point[b]] = reward[b][1]
            r = r_with_a  # reward
            rp = r_with_ap
            if done is not True:  # if done is not True:
                # q_target = r + self.reward_decay * max(q_predict_nxt[0].detach().cpu().numpy()[0])
                # q_target = r + self.reward_decay * q_predict_nxt[0] * (1-done)
                q_target = r + self.reward_decay * q_predict_nxt[0].detach().cpu().numpy()  # * (1 - np.array(done))
                qp_target = rp + self.reward_decay * q_predict_nxt[1].detach().cpu().numpy()
            else:
                q_target = r
                qp_target = rp

            # train value network
            # self.critic_optim.zero_grad()
            target_values = Variable(torch.Tensor(q_target).float()).cuda() if torch.cuda.is_available() else Variable(
                torch.Tensor(q_target).float())
            target_values_p = Variable(
                torch.Tensor(qp_target).float()).cuda() if torch.cuda.is_available() else Variable(
                torch.Tensor(qp_target).float())

            # values = model_critic([states_var_screen, states_var_minimap, states_var_player])
            critic_network_loss = self.criterion(q_predict[0], target_values) + self.criterion(q_predict[1],
                                                                                               target_values_p)  # + criterion(q_predict[1], target_values)
            print('epoch:  ', epoch, '  critic_network_loss:  \n', critic_network_loss, '\n')
            critic_network_loss_lst.append(float(critic_network_loss.detach().cpu().numpy()))

            self.critic_optim.zero_grad()
            critic_network_loss.backward()
            # torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.5)
            self.critic_optim.step()
        return critic_network_loss_lst

    def save(self, name):
        if self.model:
            torch.save(self.model.state_dict(), name)

