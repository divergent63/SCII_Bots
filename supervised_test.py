# This runner is based on actions.FUNCTIONS (.\anaconda\a3_64\envs\py37_clone_v8\Lib\site-packages\pysc2\lib\actions.py)
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
import algorithms.algorithms as algorithms

sys.path.append('../')


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
_QUEUED = [1]
EPS_START = 0.9
EPS_END = 0.025
EPS_DECAY = 2500

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4

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

# _COLLECT_RESOURCES = actions.FUNCTIONS.Harvest_Gather_SCV_screen.id
_COLLECT_RESOURCES = actions.FUNCTIONS.Harvest_Gather_screen.id
_BUILD_MISSLE_TURRENT = actions.FUNCTIONS.Build_MissileTurret_screen.id
_BUILD_ENG_BAY = actions.FUNCTIONS.Build_EngineeringBay_screen.id

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_MARINE = 48
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_TERRAN_ENGINEERINGBAY = 22
_MissileTurret = 23
_NEUTRAL_BATTLESTATIONMINERALFIELD = 886,
_NEUTRAL_BATTLESTATIONMINERALFIELD750 = 887,

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
    ACTION_SELECT_SCV,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_SELECT_BARRACKS,
    ACTION_TRAIN_MARINE,
    ACTION_BUILD_ENGBAY,
    ACTION_BUILD_MISSLE_TURRENT,
    ACTION_SELECT_ARMY,
    ACTION_ATTACK,
    ACTION_TRAIN_SCV,
    ACTION_DO_NOTHING,
    ACTION_COLLECT_RESOUCES
]
id_from_actions = {}
for ix, k in enumerate(categorical_actions):
    id_from_actions[k] = ix


if __name__ == '__main__':
    epochs = 3000
    path_lst = os.listdir('./logs')
    # history_data = np.load('./logs/history_dqn_sequence_bp1.npz')           # 'arr_0.npy'
    history_data_all = []
    for i in range(len(path_lst)):
        with open('./logs/{}'.format(path_lst[i]), 'rb') as f:
            history_data = pickle.load(f)
            history_data_all.append(history_data)
    print()

    SDL_Algo = algorithms.SupervisedDeepLearning(model_path='none')
    critic_network_loss_lst = SDL_Algo.learn(list(np.concatenate(history_data_all)), id_from_actions, epochs=epochs)
    plt.plot(critic_network_loss_lst)
    plt.show()

    save_path = './save/sdl/Simple64-sdl-epi{}.pt'.format(epochs)
    SDL_Algo.save(save_path)
    pass

