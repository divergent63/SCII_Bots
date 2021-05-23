from pysc2.lib import actions, features, units

from pysc2.env import sc2_env, run_loop, available_actions_printer
from pysc2 import maps
from absl import flags

# import sc2
# from sc2 import run_game, maps, Race, Difficulty, position
# from sc2.player import Bot, Computer, Human
# from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
#     CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY, OBSERVER, ROBOTICSFACILITY
# # from terran_agent import TerranAgent

import random
from pathlib import Path

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from random_agent import RandomAgent
from a2c_agent import A2CAgent
# from network import FullyConv
from network import SimpleConvNet_prob, SimpleConvNet_val

from utils import get_state, get_action_v2
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

seed = 500
np.random.seed(seed)

_MOVE_RAND = 1000
_MOVE_MIDDLE = 2000
_BACKGROUND = 0
_AI_SELF = 1
_AI_ALLIES = 2
_AI_NEUTRAL = 3
_AI_HOSTILE = 4
_SELECT_ALL = [0]
_NOT_QUEUED = [0]
QUEUED = [1]
EPS_START = 0.9
EPS_END = 0.025
EPS_DECAY = 2500

# define our actions
# it can choose to move to
# the beacon or to do nothing
# it can select the marine or deselect
# the marine, it can move to a random point

_AI_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_AI_SELECTED = features.SCREEN_FEATURES.selected.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE = actions.FUNCTIONS.Scan_Move_screen.id

_MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id
# _MOVE_SCREEN = actions.FUNCTIONS.Attack_unit.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
# _ATTACK_MINIMAP = actions.FUNCTIONS.Attack_Attack_minimap.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

army_selected = False
army_rallied = False

_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id

_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id

_COLLECT_RESOURCES = actions.FUNCTIONS.Harvest_Gather_SCV_screen.id
_BUILD_MISSLE_TURRENT = actions.FUNCTIONS.Build_MissileTurret_screen.id
_BUILD_ENG_BAY = actions.FUNCTIONS.Build_EngineeringBay_screen.id

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_MissileTurret = 23

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_SCV = 'selectscv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_SELECT_BARRACKS = 'selectbarracks'
ACTION_TRAIN_MARINE = 'buildmarine'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'
ACTION_COLLECT_RESOUCES = 'collect'
ACTION_BUILD_ENGBAY = 'buildengbay'
ACTION_BUILD_MISSLE_TURRENT = 'buildmissleturrent'
ACTION_TRAIN_SCV = 'trainscv'

categorical_actions = [
    ACTION_BUILD_ENGBAY,
    ACTION_SELECT_SCV,
    ACTION_ATTACK,
    ACTION_DO_NOTHING,
    ACTION_BUILD_MISSLE_TURRENT,
    ACTION_TRAIN_MARINE,
    ACTION_SELECT_ARMY,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_SELECT_BARRACKS,
    ACTION_TRAIN_SCV
    # ACTION_COLLECT_RESOUCES
]
#     ACTION_BUILD_SUPPLY_DEPOT,

categorical_actions_id = [
    _NO_OP,
    _SELECT_IDLE_WORKER,
    _BUILD_SUPPLY_DEPOT,
    _BUILD_BARRACKS,
    _SELECT_POINT,
    _TRAIN_MARINE,
    _SELECT_ARMY
]

# spatial_actions = [ACTION_ATTACK]
spatial_actions = [_MOVE_SCREEN]

id_from_actions = {}
action_from_id = {}
# for ix, k in enumerate(spatial_actions):
#     id_from_actions[k] = ix
#     action_from_id[ix] = k
for ix, k in enumerate(categorical_actions):
    id_from_actions[k] = ix+len(categorical_actions)
    action_from_id[ix+len(categorical_actions)] = k

# initialize NN model hyperparameters
eta = 0.1
expl_rate = 0.2

# initialize model object
# model = FullyConv(eta, expl_rate, categorical_actions,spatial_actions)

# <obs[0].observation.feature_screen.shape(1) = 27> + <obs[0].observation.feature_screen.shape(1) = 11> = 38
model_actor = SimpleConvNet_prob(input_size=[27, 11], output_size=[len(categorical_actions), 4096])
model_actor = model_actor.cuda() if torch.cuda.is_available() else model_actor

model_critic = SimpleConvNet_val(input_size=[27, 11], output_size=1)
model_critic = model_critic.cuda() if torch.cuda.is_available() else model_critic

model = [model_actor, model_critic]
# model = None
print(model[0], model[1])

# initalize Agent
agent = A2CAgent(model, categorical_actions, spatial_actions, id_from_actions, action_from_id)

FLAGS = flags.FLAGS
FLAGS(['run_sc2'])

viz = True
save_replay = False
real_time = False
ensure_available_actions = True
disable_fog = True

steps_per_episode = 0   # 0 actually means unlimited
MAX_EPISODES = 1000
MAX_STEPS = 1000            # 运行500个step耗时约为2：57
steps = 0
num_samples = 64


def tensor2array(T):
    lst = []
    for tensor in T:
        lst.append(tensor.cpu().detach().numpy())
    return lst


# run trajectories and train
with sc2_env.SC2Env(
        map_name="Simple64",
        players=[sc2_env.Agent(sc2_env.Race.terran),
                 sc2_env.Bot(sc2_env.Race.protoss, sc2_env.Difficulty.cheat_vision)
                 # sc2_env.Bot(sc2_env.Race.protoss, sc2_env.Difficulty.easy)
                 ],
        visualize=viz, agent_interface_format=sc2_env.AgentInterfaceFormat(
        feature_dimensions=sc2_env.Dimensions(
                screen=64,
                minimap=64)),
        random_seed=seed,
        # discount=0.3,
        realtime=real_time,
        ensure_available_actions=ensure_available_actions,
        disable_fog=disable_fog
) as env:
    max_episode_in_last_play = 0

    path_lst = os.listdir('./save/a2c')
    if len(path_lst) != 0:
        max_episode_in_last_play = max([int(p.split('.')[0].split('i')[-1]) for p in path_lst])
        load_path = [Path(Path(os.getcwd()) / 'save' / 'a2c' / 'Simple64-a2c_actor-epi{}.pt'.format(max_episode_in_last_play)), Path(Path(os.getcwd()) / 'save' / 'a2c' / 'Simple64-a2c_critic-epi{}.pt'.format(max_episode_in_last_play))]
        if model and load_path[0].is_file() and load_path[1].is_file():
            # agent.load(load_path)
            model_actor.load_state_dict(torch.load(load_path[0]))
            model_critic.load_state_dict(torch.load(load_path[1]))

    done = False
    history = []

    critic_optim = torch.optim.Adam(model_critic.parameters(), lr=0.01)
    actor_optim = torch.optim.Adam(model_critic.parameters(), lr=0.01)

    for e in range(max_episode_in_last_play, max_episode_in_last_play+MAX_EPISODES-1):
        obs = env.reset()

        score = 0
        score_pre = 0
        state = get_state(obs[0])
        time = 0

        agent.states = []
        agent.next_states = []
        agent.rewards = []
        agent.actions = []
        agent.points = []
        agent.score = []

        for time in range(MAX_STEPS-1):

            init = False
            if e == 0 and time == 0:
                init = True

            state_model = [np.array(obs[0].observation.feature_screen), np.array(obs[0].observation.feature_minimap), np.array(obs[0].observation.player)]
            preds = agent.act(state_model, init)
            action, point = np.random.choice(list(action_from_id.values()), 1, p=preds[0].squeeze(0).cpu().detach().numpy())[0], np.random.choice(4096, 1, p=preds[1].squeeze(0).cpu().detach().numpy())[0]

            func, act_a = get_action_v2(action, point, obs=obs[0])
            next_obs = env.step([func])
            print(act_a, point)

            next_state = get_state(next_obs[0])
            next_state_model = [np.array(next_obs[0].observation.feature_screen), np.array(next_obs[0].observation.feature_minimap), np.array(next_obs[0].observation.player)]

            reward = float(next_obs[0].reward) + float(np.sum([
                next_obs[0].observation.score_cumulative[0],
                next_obs[0].observation.score_cumulative[3],
                next_obs[0].observation.score_cumulative[4],
                10*next_obs[0].observation.score_cumulative[5],
                10*next_obs[0].observation.score_cumulative[6],
                next_obs[0].observation.score_cumulative[7],
                next_obs[0].observation.score_cumulative[9],
                5*next_obs[0].observation.score_cumulative[11]
            ])-8*next_obs[0].observation.score_cumulative[2]) * 10e-6

            if env._controllers and env._controllers[0].status.value != 3:          # env._controllers[0].status.value = 3 --> game running; env._controllers[0].status.value = 5 --> defeat;
                done = True
                if env._controllers[0].status.value == 5:           # 战败
                    reward = reward / 1000
                    # break
                    
            if time == MAX_STEPS-2:
                done = True
            # if next_obs[0].last():
            #     done = True
            # if not next_obs[0].last():              #
            #     done = True

            # history.append(agent.append_sample(state_model, next_state_model, act_a, point, reward, score))
            # agent.append_sample(tensor2array(state_model), tensor2array(next_state_model), tensor2array(preds), reward, score)
            agent.append_sample(state_model, next_state_model, tensor2array(preds), reward, score)

            state = next_state
            obs = next_obs
            # if done:
            #     print("episode: {}/{}, score: {}".format(e, MAX_EPISODES, score))
            #     # if score_pre < score:
            #     #     score_pre = score
            #     done = False

            score += reward
            # time += 1

            # history.append(model)

            # if len(agent.states) > 2*num_samples:
            if done:

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
                actions_var_action = Variable(torch.Tensor([agent.actions[i][0] for i in range(len(agent.actions))]).view(-1, len(categorical_actions)))
                actions_var_point = Variable(torch.Tensor([agent.actions[i][1] for i in range(len(agent.actions))]).view(-1, 4096))
                states_var_screen = Variable(torch.Tensor([agent.states[i][0] for i in range(len(agent.states))]).view(-1, 27, 64, 64))
                states_var_minimap = Variable(torch.Tensor([agent.states[i][1] for i in range(len(agent.states))],).view(-1, 11, 64, 64))
                states_var_player = Variable(torch.Tensor([agent.states[i][2] for i in range(len(agent.states))],).view(-1, 11))

                # states_var = Variable(torch.Tensor(next_state_model).view(-1, len(38*64*64)))

                # train actor network
                model_actor.zero_grad()
                log_softmax_actions_1, log_softmax_actions_2 = model_actor([states_var_screen, states_var_minimap, states_var_player])
                vs = model_critic([states_var_screen, states_var_minimap, states_var_player])
                # calculate qs
                qs = Variable(torch.Tensor(agent.discount_rewards(agent.rewards))).cuda() if torch.cuda.is_available() else Variable(torch.Tensor(agent.discount_rewards(agent.rewards)))

                advantages = qs - vs.detach().squeeze(1)
                actor_network_loss = - torch.mean(torch.sum(log_softmax_actions_1.cpu()*actions_var_action, 1) * advantages.cpu()) - torch.mean(torch.sum(log_softmax_actions_2.cpu()*actions_var_point, 1) * advantages.cpu())
                actor_network_loss.backward()
                torch.nn.utils.clip_grad_norm(model_actor.parameters(), 0.5)
                actor_optim.step()

                # train value network
                critic_optim.zero_grad()
                target_values = qs
                # values = model_critic([states_var_screen, states_var_minimap, states_var_player])
                criterion = nn.MSELoss()
                critic_network_loss = criterion(vs, target_values)
                critic_network_loss.backward()
                torch.nn.utils.clip_grad_norm(model_critic.parameters(), 0.5)
                critic_optim.step()

                actor_network_losses.append(float(actor_network_loss.detach().numpy()))
                critic_network_losses.append(float(critic_network_loss.cpu().detach().numpy()))

                print('episode: {},   loss of actor: {},   loss of critic: {},   score: {}'.format(e, np.sum(actor_network_losses), np.sum(critic_network_losses), score))

            else:
                continue

            condition1 = e <= 995 and e % 100 == 0
            condition2 = e > 995
            if condition1 or condition2:
                save_path = ['./save/a2c/Simple64-a2c_actor-epi{}.pt'.format(e), './save/a2c/Simple64-a2c_critic-epi{}.pt'.format(e)]
                agent.save(save_path)
                # torch.save(model_actor, './save/Simple64-a2c_actor-epi{}.pt'.format(e))
                # torch.save(model_critic, './save/Simple64-a2c_critic-epi{}.pt'.format(e))
                history.append([e, agent.states, agent.actions, agent.rewards])
                np.savez_compressed("./logs/a2c_e{}.npz".format(e))
            done = False

    print('train complete')


