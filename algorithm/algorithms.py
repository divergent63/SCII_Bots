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

import model.models as models
from model.models import SimpleConvNet_prob, SimpleConvNet_val

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


class DeepQLearning:
    """
    Natural-DQN with self.copy() after 'done == True'
    """

    def __init__(self, model_path, learning_rate=0.01, reward_decay=0.9):
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay

        self.feature_screen = 27
        self.feature_minimap = 11
        self.out_action = 12
        self.out_point = 4096

        # model_p = models.SimpleConvNet_prob(input_size=[self.feature_screen, self.feature_minimap], output_size=[self.out_action, self.out_point])
        model_v = models.SimpleConvNet_val(input_size=[self.feature_screen, self.feature_minimap],
                                           output_size=[self.out_action, self.out_point])
        model_v_process = models.SimpleConvNet_val(input_size=[self.feature_screen, self.feature_minimap],
                                                   output_size=[self.out_action, self.out_point])
        self.model = model_v.cuda() if torch.cuda.is_available() else model_v
        self.model_process = model_v_process.cuda() if torch.cuda.is_available() else model_v_process

        print('model:  \n', self.model)

        self.criterion = nn.MSELoss()

        # model_path = Path(Path(os.getcwd()) / 'save' / 'dqn' / 'Simple64-dqn-best.pt')
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path))

        self.critic_optim = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def choose_action_p(self, state, init=False, e_greedy=0.2):
        ep = np.random.random()
        if not init and ep < e_greedy:  # TODO: can't understand the probability matrix from uniform distribution --> SOLVED
            # return self.action_from_id[np.random.choice(len(self.action_from_id), 1)[0]], np.random.randint(4096)
            p1 = Variable(torch.Tensor(
                np.random.dirichlet(np.ones(self.out_action),
                                    size=1)).cuda() if torch.cuda.is_available() else torch.Tensor(
                np.random.dirichlet(np.ones(self.out_action), size=1)))
            p2 = Variable(torch.Tensor(
                np.random.dirichlet(np.ones(self.out_point),
                                    size=1)).cuda() if torch.cuda.is_available() else torch.Tensor(
                np.random.dirichlet(np.ones(self.out_point), size=1)))
            # print("explored actions")
            return [p1, p2]
        else:
            # state.append(np.zeros((1, 1)))
            preds = self.model_process(state)
            # print("learned actions")
            # return self.action_from_id[np.random.choice(len(self.action_from_id), 1, p=preds[1][0])[0]], np.random.choice(len(self.action_from_id), 1, p=preds[2][0])[0]
            # return self.action_from_id[np.random.choice(len(self.action_from_id), 1, p=preds[0].cpu().detach().numpy())[0]], np.random.choice(4096, 1, p=preds[1].cpu().detach().numpy())[0]
            # return np.random.choice(list(self.action_from_id.values()), 1, p=preds[0].cpu().detach().numpy())[0], \
            #        np.random.choice(4096, 1, p=preds[1].cpu().detach().numpy())[0]
            return preds

    def choose_action_v(self, state, init=False, e_greedy=0.2):
        ep = np.random.random()
        if not init and ep < e_greedy:  # TODO: can't understand the probability matrix from uniform distribution --> SOLVED
            # return self.action_from_id[np.random.choice(len(self.action_from_id), 1)[0]], np.random.randint(4096)
            v1, v2 = np.reshape(np.random.rand(self.out_action), (1, self.out_action)), np.reshape(
                np.random.rand(self.out_point), (1, self.out_point))
            # print("explored actions")
            return [
                Variable(torch.Tensor(v1).float()).cuda() if torch.cuda.is_available() else Variable(
                    torch.Tensor(v1).float()),
                Variable(torch.Tensor(v2).float()).cuda() if torch.cuda.is_available() else Variable(
                    torch.Tensor(v2).float())
            ]
        else:
            # state.append(np.zeros((1, 1)))
            v = self.model_process(state)
            # print("learned actions")
            return v

    def learn(self, history_raw, id_from_actions, epochs=3):
        batch_size = len(history_raw) // 16
        critic_network_loss_lst = []
        # history_raw: [e, time, state_model, state_model_next, action, actual_action, last_action, point, reward, score, done]
        for epoch in range(epochs):
            history = random.sample(history_raw, batch_size)
            e, time, state_model, state_model_next, action, actual_action, last_action, point, reward, score, done = zip(
                *history)
            # for _ in range(len(history)):
            #     idx = random.randint(0, len(history) - 1)

            state_tensor_lst = [[Variable(
                torch.Tensor(state_model[b][i]).float()).cuda() if torch.cuda.is_available() else Variable(
                torch.Tensor(state_model[b][i]).float()) for i in range(3)] for b in range(len(history))]
            state_tensor_batch_lst = [torch.cat([state_tensor_lst[b][i].unsqueeze(0) for b in range(len(history))]) for
                                      i in range(3)]

            next_state_tensor_lst = [[Variable(
                torch.Tensor(state_model_next[b][i]).float()).cuda() if torch.cuda.is_available() else Variable(
                torch.Tensor(state_model_next[b][i]).float()) for i in range(3)] for b in range(len(history))]
            next_state_tensor_batch_lst = [
                torch.cat([next_state_tensor_lst[b][i].unsqueeze(0) for b in range(len(history))]) for i in range(3)]

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
            # print('epoch:  ', epoch, '  critic_network_loss:  \n', critic_network_loss, '\n')
            critic_network_loss_lst.append(float(critic_network_loss.detach().cpu().numpy()))

            self.critic_optim.zero_grad()
            critic_network_loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.5)
            self.critic_optim.step()
        return critic_network_loss_lst

    def save(self, name):
        if self.model:
            torch.save(self.model.state_dict(), name)

    def copy(self):
        for target_param, param in zip(self.model.parameters(), self.model_process.parameters()):
            param.data.copy_(target_param.data)


class AdvantageActorCritic:
    """
    Advantage Actor-Critic
    """

    def __init__(self, model_path, learning_rate=0.01, reward_decay=0.9):
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay

        self.feature_screen = 27
        self.feature_minimap = 11
        self.out_action = 12
        self.out_point = 4096

        # <obs[0].observation.feature_screen.shape(1) = 27> + <obs[0].observation.feature_screen.shape(1) = 11> = 38
        model_actor = models.SimpleConvNet_prob(input_size=[27, 11], output_size=[len(categorical_actions), 4096])
        model_actor = model_actor.cuda() if torch.cuda.is_available() else model_actor

        model_critic = models.SimpleConvNet_val(input_size=[27, 11], output_size=1)
        model_critic = model_critic.cuda() if torch.cuda.is_available() else model_critic

        model = [model_actor, model_critic]
        # model = None
        print(model[0], model[1])

        self.criterion = nn.MSELoss()

        # model_path = Path(Path(os.getcwd()) / 'save' / 'dqn' / 'Simple64-dqn-best.pt')
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path))

        self.critic_optim = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def choose_action_p(self, state, init=False, e_greedy=0.2):
        ep = np.random.random()
        if not init and ep < e_greedy:  # TODO: can't understand the probability matrix from uniform distribution --> SOLVED
            # return self.action_from_id[np.random.choice(len(self.action_from_id), 1)[0]], np.random.randint(4096)
            p1 = Variable(torch.Tensor(
                np.random.dirichlet(np.ones(self.out_action),
                                    size=1)).cuda() if torch.cuda.is_available() else torch.Tensor(
                np.random.dirichlet(np.ones(self.out_action), size=1)))
            p2 = Variable(torch.Tensor(
                np.random.dirichlet(np.ones(self.out_point),
                                    size=1)).cuda() if torch.cuda.is_available() else torch.Tensor(
                np.random.dirichlet(np.ones(self.out_point), size=1)))
            # print("explored actions")
            return [p1, p2]
        else:
            # state.append(np.zeros((1, 1)))
            preds = self.model_process(state)
            return preds

    def choose_action_v(self, state, init=False, e_greedy=0.2):
        ep = np.random.random()
        if not init and ep < e_greedy:  # TODO: can't understand the probability matrix from uniform distribution --> SOLVED
            # return self.action_from_id[np.random.choice(len(self.action_from_id), 1)[0]], np.random.randint(4096)
            v1, v2 = np.reshape(np.random.rand(self.out_action), (1, self.out_action)), np.reshape(
                np.random.rand(self.out_point), (1, self.out_point))
            # print("explored actions")
            return [
                Variable(torch.Tensor(v1).float()).cuda() if torch.cuda.is_available() else Variable(
                    torch.Tensor(v1).float()),
                Variable(torch.Tensor(v2).float()).cuda() if torch.cuda.is_available() else Variable(
                    torch.Tensor(v2).float())
            ]
        else:
            # state.append(np.zeros((1, 1)))
            v = self.model_process(state)
            # print("learned actions")
            return v

    def learn(self, history_raw, id_from_actions, epochs=3):
        batch_size = len(history_raw) // 16
        critic_network_loss_lst = []
        # history_raw: [e, time, state_model, state_model_next, action, actual_action, last_action, point, reward, score, done]
        for epoch in range(epochs):
            history = random.sample(history_raw, batch_size)
            e, time, state_model, state_model_next, action, actual_action, last_action, point, reward, score, done = zip(
                *history)
            # for _ in range(len(history)):
            #     idx = random.randint(0, len(history) - 1)

            state_tensor_lst = [[Variable(
                torch.Tensor(state_model[b][i]).float()).cuda() if torch.cuda.is_available() else Variable(
                torch.Tensor(state_model[b][i]).float()) for i in range(3)] for b in range(len(history))]
            state_tensor_batch_lst = [torch.cat([state_tensor_lst[b][i].unsqueeze(0) for b in range(len(history))]) for
                                      i in range(3)]

            next_state_tensor_lst = [[Variable(
                torch.Tensor(state_model_next[b][i]).float()).cuda() if torch.cuda.is_available() else Variable(
                torch.Tensor(state_model_next[b][i]).float()) for i in range(3)] for b in range(len(history))]
            next_state_tensor_batch_lst = [
                torch.cat([next_state_tensor_lst[b][i].unsqueeze(0) for b in range(len(history))]) for i in range(3)]

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
            # print('epoch:  ', epoch, '  critic_network_loss:  \n', critic_network_loss, '\n')
            critic_network_loss_lst.append(float(critic_network_loss.detach().cpu().numpy()))

            self.critic_optim.zero_grad()
            critic_network_loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.5)
            self.critic_optim.step()
        return critic_network_loss_lst

    def save(self, name):
        if self.model:
            torch.save(self.model.state_dict(), name)

    def copy(self):
        for target_param, param in zip(self.model.parameters(), self.model_process.parameters()):
            param.data.copy_(target_param.data)


class AdvantageActorCritic_bak:
    """This class implements the random walking agent using the network model"""

    def __init__(self, model, categorical_actions, spatial_actions, id_from_actions, action_from_id):
        self.states = []
        self.next_states = []
        self.rewards = []
        self.actions = []
        self.points = []
        self.score = []
        # self.policy_predictions=[]
        # self.spatial_predictions=[]
        self.gamma = 0.95  # discount rate
        self.categorical_actions = categorical_actions
        self.spatial_actions = spatial_actions
        self.model = model[0]           # Actor
        self.value_model = model[1]         # Critic
        # self.epsilon = 0.5
        self.id_from_actions = id_from_actions
        self.action_from_id = action_from_id

    def update_epsilon(self):
        if self.epsilon > 0.1:
            self.epsilon = 0.999 * self.epsilon

    def append_sample(self, states,  last_actions, actions, rewards, scores):
        self.states.append(states)
        self.next_states.append(last_actions)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.score.append(scores)
        return [states, last_actions, actions, rewards, scores]

    # def discount_rewards(self, rewards):
    #     discounted_rewards = np.zeros_like(rewards)
    #     running_add = 0
    #     for t in reversed(range(0, len(rewards))):
    #         running_add = running_add * self.gamma + rewards[t]
    #         discounted_rewards[t] = running_add
    #     return discounted_rewards

    def discount_rewards(self, rewards, final_r):
        discounted_r = np.zeros_like(rewards)
        running_add = final_r
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    def act(self, state, init=False, epsilon=0.2):
        ep = np.random.random()
        if not init and ep < epsilon:           # TODO: can't understand the probability matrix from uniform distribution --> SOLVED
            # return self.action_from_id[np.random.choice(len(self.action_from_id), 1)[0]], np.random.randint(4096)
            p1 = Variable(torch.Tensor(np.random.dirichlet(np.ones(11), size=1)).cuda() if torch.cuda.is_available() else torch.Tensor(np.random.dirichlet(np.ones(11), size=1)))
            p2 = Variable(torch.Tensor(np.random.dirichlet(np.ones(4096), size=1)).cuda() if torch.cuda.is_available() else torch.Tensor(np.random.dirichlet(np.ones(4096), size=1)))
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
                         np.zeros((episode_length, 11,))
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

    def learn(self):

        actor_network_losses = []
        critic_network_losses = []

        # for n in range(len(agent.states)):

        # # train
        # # out_spatial, out_non_spatial = model(state_model)
        # if score > score_pre:
        #     history_arr = np.array(history)
        #     np.savez_compressed('./save/history_random.npz', history)
        #     agent.save("./save/Simple64-rand.pt")
        #     score_pre = score

        # init_state = agent.states[n]
        actions_var_action = Variable(
            torch.Tensor([agent.actions[i][0] for i in range(len(agent.actions))]).view(-1, len(categorical_actions)))
        actions_var_point = Variable(
            torch.Tensor([agent.actions[i][1] for i in range(len(agent.actions))]).view(-1, 4096))
        states_var_screen = Variable(
            torch.Tensor([agent.states[i][0] for i in range(len(agent.states))]).view(-1, 27, 64, 64))
        states_var_minimap = Variable(
            torch.Tensor([agent.states[i][1] for i in range(len(agent.states))], ).view(-1, 11, 64, 64))
        states_var_player = Variable(
            torch.Tensor([agent.states[i][2] for i in range(len(agent.states))], ).view(-1, 11))

        batch_size = 33
        train_dataloader = DataLoader([
            states_var_screen, states_var_minimap, states_var_player, actions_var_action, actions_var_point
        ], batch_size=batch_size, shuffle=False)

        train_dataloader_screen = DataLoader(states_var_screen, batch_size=batch_size, shuffle=False)
        train_dataloader_map = DataLoader(states_var_minimap, batch_size=batch_size, shuffle=False)
        train_dataloader_play = DataLoader(states_var_player, batch_size=batch_size, shuffle=False)
        train_dataloader_a = DataLoader(actions_var_point, batch_size=batch_size, shuffle=False)
        train_dataloader_p = DataLoader(actions_var_point, batch_size=batch_size, shuffle=False)

        qs = DataLoader(Variable(torch.Tensor(
            agent.discount_rewards(agent.rewards, reward))).cuda() if torch.cuda.is_available() else Variable(
            torch.Tensor(agent.discount_rewards(agent.rewards))), batch_size=batch_size, shuffle=False)

        for i in range(train_dataloader.dataset[0].data.shape[0] // batch_size):
            # Display image and label.
            train_dataloader_screen_var = next(iter(train_dataloader_screen))
            train_dataloader_map_var = next(iter(train_dataloader_map))
            train_dataloader_player_var = next(iter(train_dataloader_play))
            train_dataloader_a_var = next(iter(train_dataloader_a))
            train_dataloader_p_var = next(iter(train_dataloader_p))

            # img = states_var_screen_batch[0].squeeze()
            # plt.imshow(img, cmap="gray")
            # plt.show()
            # states_var = Variable(torch.Tensor(next_state_model).view(-1, len(38*64*64)))

            # train actor network
            model_actor.zero_grad()
            log_softmax_actions_1, log_softmax_actions_2 = model_actor(
                [train_dataloader_screen_var, train_dataloader_map_var, train_dataloader_player_var])
            vs = model_critic([train_dataloader_screen_var, train_dataloader_map_var, train_dataloader_player_var])
            # calculate qs

            qs_var = next(iter(qs))
            advantages = qs_var - vs.detach().squeeze(1)
            actor_network_loss = - torch.mean(
                torch.sum(log_softmax_actions_1.cpu() * train_dataloader_a_var, 1) * advantages.cpu()) - torch.mean(
                torch.sum(log_softmax_actions_2.cpu() * train_dataloader_p_var, 1) * advantages.cpu())
            actor_network_loss.backward()
            torch.nn.utils.clip_grad_norm(model_actor.parameters(), 0.5)
            actor_optim.step()

            # train value network
            critic_optim.zero_grad()
            target_values = qs_var
            # values = model_critic([states_var_screen, states_var_minimap, states_var_player])
            criterion = nn.MSELoss()
            critic_network_loss = criterion(vs, target_values)
            critic_network_loss.backward()
            torch.nn.utils.clip_grad_norm(model_critic.parameters(), 0.5)
            critic_optim.step()

        actor_network_losses.append(float(actor_network_loss.detach().numpy()))
        critic_network_losses.append(float(critic_network_loss.cpu().detach().numpy()))
        return [actor_network_losses, critic_network_losses]

    def load(self, name):
        if self.model and self.value_model:
            self.model.load_state_dict(torch.load(name[0]))
            self.value_model.load_state_dict(torch.load(name[1]))

    def save(self, name):
        if self.model and self.value_model:
            torch.save(self.model.state_dict(), name[0])
            torch.save(self.value_model.state_dict(), name[1])

