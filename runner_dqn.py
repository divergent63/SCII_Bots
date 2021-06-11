# -*- coding: UTF-8 -*-#
# This runner is based on actions.FUNCTIONS (.\anaconda\a3_64\envs\py37_clone_v8\Lib\site-packages\pysc2\lib\actions.py)

from pysc2.lib import actions, features, units

from pysc2.env import sc2_env, run_loop, available_actions_printer
from pysc2 import maps

import models.models as models
import algorithms.q_learning as q_learning

from pathlib import Path
from absl import app, logging, flags
from matplotlib import pyplot as plt

import random
import math
import pickle
import pandas as pd
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np

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

categorical_actions_id = [
    _SELECT_IDLE_WORKER,
    _BUILD_SUPPLY_DEPOT,
    _BUILD_BARRACKS,
    _SELECT_POINT,
    _TRAIN_MARINE,
    _SELECT_ARMY,
    _NO_OP,
]

# spatial_actions = [ACTION_ATTACK]
spatial_actions = [_MOVE_SCREEN]

id_from_actions = {}
action_from_id = {}

for ix, k in enumerate(categorical_actions):
    id_from_actions[k] = ix
    action_from_id[ix] = k

FLAGS = flags.FLAGS
FLAGS(['run_sc2'])


def get_state(obs):
    return [np.array(obs.observation['feature_screen']).reshape(1, 27, 64, 64),
            np.array(obs.observation['feature_minimap']).reshape(1, 11, 64, 64),
            np.array(obs.observation['player']).reshape(1, 11)
            ]


def to_yx(point):
    """transform a scalar from [0;4095] to a (y,x) coordinate in [0:63,0:63]"""
    return int(point % 64), int((point - (point % 64)) / 64)


def transformLocation(obs, x, y):
    player_y, player_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()

    base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
    if not base_top_left:
        return [64 - x, 64 - y]
    else:
        return [x, y]


def get_action_v3(id_action, point, obs, num_dict=None):
    # obs = obs[0]
    unit_type = obs.observation['feature_screen'][_UNIT_TYPE]

    depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
    supply_depot_exist = 1 if depot_y.any() else 0
    if not supply_depot_exist:
        num_dict['supply_deports'] = 0

    barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
    barracks_exist = 1 if barracks_y.any() else 0
    if not barracks_exist:
        num_dict['barracks'] = 0

    engbays_y, engbays_x = (unit_type == _TERRAN_ENGINEERINGBAY).nonzero()
    engbays_exist = 1 if engbays_y.any() else 0

    supply_limit = obs.observation['player'][4]
    army_supply = obs.observation['player'][5]
    food_workers = obs.observation['player'][6]
    idle_workers_cnt = obs.observation['player'][7]
    army_cnt = obs.observation['player'][8]

    killed_unit_score = obs.observation['score_cumulative'][5]
    killed_building_score = obs.observation['score_cumulative'][6]

    current_state = np.zeros(20)
    current_state[0] = supply_depot_exist
    current_state[1] = barracks_exist
    current_state[2] = supply_limit
    current_state[3] = army_supply

    hot_squares = np.zeros(16)

    if (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero():
        enemy_y, enemy_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
        for i in range(0, len(enemy_y)):
            y = int(math.ceil((enemy_y[i] + 1) / 16))
            x = int(math.ceil((enemy_x[i] + 1) / 16))

            hot_squares[((y - 1) * 4) + (x - 1)] = 1

        for i in range(0, 16):
            current_state[i + 4] = hot_squares[i]

    smart_action = id_action

    # if '_' in smart_action:
    #     smart_action, x, y = smart_action.split('_')

    # (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
    if smart_action == ACTION_SELECT_SCV:
        unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

        if unit_y.any():
            i = random.randint(0, len(unit_y) - 1)
            target = [unit_x[i], unit_y[i]]
            if _SELECT_IDLE_WORKER in obs.observation["available_actions"]:
                func = actions.FunctionCall(_SELECT_IDLE_WORKER, [_NOT_QUEUED])
            else:
                func = actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

    elif smart_action == ACTION_TRAIN_SCV:
        # worker_cnt = num_dict["workers"]
        # if _TRAIN_SCV in obs.observation['available_actions'] and worker_cnt < 16:
        if _TRAIN_SCV in obs.observation['available_actions']:
            func = actions.FunctionCall(_TRAIN_SCV, [_QUEUED])
            # num_dict["workers"] += 1

    elif smart_action == ACTION_COLLECT_RESOUCES:
        # TODO: Warning about "必须以资源为目标"
        unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
        scv_y, scv_x = (unit_type == units.Terran.SCV).nonzero()
        mineral_y, mineral_x = (unit_type == units.Neutral.MineralField).nonzero()
        # mineral_y, mineral_x = (unit_type == _NEUTRAL_BATTLESTATIONMINERALFIELD).nonzero()

        if _COLLECT_RESOURCES in obs.observation['available_actions'] and idle_workers_cnt > 0:
            if mineral_y.any() and scv_y.any():
                i = random.randint(0, len(scv_y) - 1)
                # target = (mineral_y[i], mineral_y[i])
                # target = (mineral_y.mean(), mineral_y.mean())
                # target = (scv_y.mean(), scv_x.mean())
                target = (scv_y[i], scv_x[i])
                # target = (11, 16)
                func = actions.FunctionCall(_COLLECT_RESOURCES, [_NOT_QUEUED, target])

    elif smart_action == ACTION_BUILD_SUPPLY_DEPOT:
        if _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
            unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

            if unit_y.any():
                if supply_limit < 23:
                    target = (31, 8)
                elif supply_limit < 31:
                    target = (26, 8)
                elif supply_limit < 39:
                    target = (21, 8)
                elif supply_limit < 47:
                    target = (16, 8)
                # else:
                #     target = to_yx(point)

                try:
                    func = actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])
                    return func, smart_action, num_dict
                except UnboundLocalError:
                    # print(str(smart_action) + " " + str(point) + " is not an available action")
                    return get_action_v3(action_from_id[0], point, obs, num_dict)  # 'selectscv'

    elif smart_action == ACTION_BUILD_ENGBAY:
        engbays_cnt = num_dict["engbays"]

        if _BUILD_ENG_BAY in obs.observation['available_actions'] and not engbays_exist:
            unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

            if unit_y.any():
                # target = to_yx(point)
                target = (38, 44)
                func = actions.FunctionCall(_BUILD_ENG_BAY, [_NOT_QUEUED, target])
        try:
            # num_dict["engbays"] += 1
            return func, smart_action, num_dict
        except UnboundLocalError:
            # num_dict["engbays"] -= 1
            # print(str(smart_action) + " " + str(point) + " is not an available action")
            return get_action_v3(action_from_id[0], point, obs, num_dict)  # 'selectscv'

    elif smart_action == ACTION_BUILD_MISSLE_TURRENT:
        missile_turrets_cnt = num_dict["missile_turrets"]

        if _BUILD_MISSLE_TURRENT in obs.observation['available_actions'] and missile_turrets_cnt < 16:
            unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()

            if unit_y.any():
                target = [(60, 16), (60, 26), (60, 36)]
                if num_dict['missile_turrets'] == 0:
                    func = actions.FunctionCall(_BUILD_MISSLE_TURRENT, [_NOT_QUEUED, target[0]])
                elif num_dict['missile_turrets'] == 1:
                    func = actions.FunctionCall(_BUILD_MISSLE_TURRENT, [_NOT_QUEUED, target[1]])
                elif num_dict['missile_turrets'] == 2:
                    func = actions.FunctionCall(_BUILD_MISSLE_TURRENT, [_NOT_QUEUED, target[2]])
                # else:
                #     target = to_yx(point)
                #     func = actions.FunctionCall(_BUILD_MISSLE_TURRENT, [_NOT_QUEUED, target])

        try:
            num_dict['missile_turrets'] += 1
            return func, smart_action, num_dict
        except UnboundLocalError:
            num_dict['missile_turrets'] -= 1
            # print(str(smart_action) + " " + str(point) + " is not an available action")
            if engbays_exist:
                return get_action_v3(action_from_id[0], point, obs, num_dict)  # 'selectscv'
            else:
                return get_action_v3(action_from_id[5], point, obs,
                                     num_dict)  # 'buildengbay'         # TODO: 无法建造导弹塔的原因不一定是因未建造工程港而未解锁，还有可能是前置动作未选择农民        # SOLVED #

    elif smart_action == ACTION_BUILD_BARRACKS:
        if _BUILD_BARRACKS in obs.observation['available_actions']:
            unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

            if unit_y.any() and num_dict["barracks"] < 3:
                # target = to_yx(point)
                if num_dict["barracks"] == 0:
                    target = (52, 18)
                # elif num_dict["barracks"] == 1:
                #     target = (52, 28)
                # elif num_dict["barracks"] == 2:
                #     target = (52, 38)
                # else:
                #     target = to_yx(point)

                try:
                    func = actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

                    num_dict["barracks"] += 1
                    return func, smart_action, num_dict
                except UnboundLocalError:
                    num_dict["barracks"] -= 1
                    # print(str(smart_action) + " " + str(point) + " is not an available action")
                    if num_dict['supply_deports'] == 0:
                        return get_action_v3(action_from_id[1], point, obs, num_dict)  # 'buildsupplydepot'
                    else:
                        return get_action_v3(action_from_id[0], point, obs, num_dict)  # 'selectscv'

    elif smart_action == ACTION_SELECT_BARRACKS:
        unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()

        if unit_y.any():
            # target = [int(unit_x.mean()), int(unit_y.mean())]
            # target = (np.random.([(unit_x[i], unit_y[i]) for i in range(len(unit_x))]))
            a_list = [(unit_x[i], unit_y[i]) for i in range(len(unit_x))]
            target = list(map(lambda x: random.choice(a_list), range(1)))[0]
            func = actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

            try:
                return func, smart_action, num_dict
            except UnboundLocalError:
                # print(str(smart_action) + " " + str(point) + " is not an available action")
                return get_action_v3(action_from_id[2], point, obs, num_dict)  # 'buildbarracks'

    elif smart_action == ACTION_TRAIN_MARINE:
        unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()

        if _TRAIN_MARINE in obs.observation['available_actions'] and unit_y.any():
            func = actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
        try:
            # num_dict["marines"] += 1
            return func, smart_action, num_dict
        except UnboundLocalError:
            # num_dict["marines"] -= 1
            # print(str(smart_action) + " " + str(point) + " is not an available action")
            return get_action_v3(action_from_id[3], point, obs, num_dict)  # 'selectbarracks'

    elif smart_action == ACTION_SELECT_ARMY:
        if _SELECT_ARMY in obs.observation['available_actions']:
            func = actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
        try:
            return func, smart_action, num_dict
        except UnboundLocalError:
            # print(str(smart_action) + " " + str(point) + " is not an available action")
            return get_action_v3(action_from_id[4], point, obs, num_dict)  # 'buildmarine'

    elif smart_action == ACTION_ATTACK:
        enemy_y, enemy_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()

        if (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero():  # 攻击已知敌人

            if len(obs.observation['multi_select']) and army_cnt > 8:
                if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                    # if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                    if enemy_y.any():
                        # target = [int(np.random.choice(enemy_x)), int(np.random.choice(enemy_y))]
                        target = to_yx(point)           # TODO:
                        func = actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, target])
                        # num_dict['marines'] = 0
        else:  # 攻击任意位置（未找到敌人时，类似巡逻）
            if len(obs.observation['multi_select']):
                if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                    target = to_yx(point)
                    func = actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, target])
        try:
            num_dict['attack_cnt'] += 1
            return func, smart_action, num_dict
        except UnboundLocalError:
            num_dict['attack_cnt'] -= 1
            # print(str(smart_action) + " " + str(point) + " is not an available action")
            if army_cnt < 8:
                return get_action_v3(action_from_id[4], point, obs, num_dict)  # 'buildmarine'
            else:
                return get_action_v3(action_from_id[7], point, obs, num_dict)   # 'select_army'

    elif smart_action == ACTION_DO_NOTHING:
        func = actions.FunctionCall(_NO_OP, [])

    try:
        return func, smart_action, num_dict

    except UnboundLocalError:
        # print(str(smart_action) + " " + str(point) + " is not an available action")
        return actions.FunctionCall(_NO_OP, []), ACTION_DO_NOTHING, num_dict


def main(unused_argv):
    # TODO: save replay not working ...

    viz = True
    replay_prefix = 'D:/software/python_prj/SCII/SCII_Bots/replays/deterministic_sequence'
    replay_dir = '/replays'
    real_time = False
    ensure_available_actions = True
    disable_fog = True
    steps_per_episode = 0  # 0 actually means unlimited
    MAX_EPISODES = 1
    MAX_STEPS = 300000
    train_mode = True           # True  False
    if train_mode == True:
        MAX_EPISODES = 1000
    else:
        MAX_EPISODES = 1
    try:
        # run trajectories and train
        with sc2_env.SC2Env(
                map_name="Simple64",
                players=[sc2_env.Agent(sc2_env.Race.terran),
                         # sc2_env.Bot(sc2_env.Race.protoss, sc2_env.Difficulty.easy)
                         sc2_env.Bot(sc2_env.Race.protoss, sc2_env.Difficulty.very_easy)
                         ],
                visualize=viz, agent_interface_format=sc2_env.AgentInterfaceFormat(
                    use_raw_units=True,
                    feature_dimensions=sc2_env.Dimensions(
                        screen=64,
                        minimap=64)),
                random_seed=seed,
                # discount=0.3,
                realtime=real_time,
                ensure_available_actions=ensure_available_actions,
                disable_fog=disable_fog,
                # save_replay_episodes=1,
                # replay_prefix=replay_prefix,
                # replay_dir=replay_dir,
        ) as env:

            done = False
            dataset = []

            max_batch_pool_in_last_play = 0
            max_episode_in_last_play = 0

            path_lst = os.listdir('./save/dqn')
            if len(path_lst) != 0:
                max_episode_in_last_play = max([int(p.split('.')[0].split('i')[-1].split('-')[0]) for p in path_lst])
                load_path = Path(Path(os.getcwd()) / 'save' / 'dqn' / 'Simple64-dqn-epi{}.pt'.format(max_episode_in_last_play))
            else:
                load_path = 'none'
            algo = q_learning.DeepQLearning(load_path)

            logs_path_lst = os.listdir('./logs/history_data')
            if len(logs_path_lst) != 0:
                max_batch_pool_in_last_play = max([int(os.listdir('./logs/history_data')[p].split('p')[-2].split('.')[0].split('_')[0]) for p in range(len(logs_path_lst))])
            batch_pool_idx = max_batch_pool_in_last_play + 1
            losses_lst = []
            for e in range(max_episode_in_last_play+1, max_episode_in_last_play+MAX_EPISODES+1):
                if e > 0:
                    obs = env.reset()

                score = 0

                num_dict = {"workers": 0, "idle_workers": 0, "barracks": 0, "engbays": 0,
                            # "marines": 0,
                            "missile_turrets": 0, 'attack_cnt': 0}
                for time in range(MAX_STEPS):
                    init = False
                    if e == 0 and time == 0:
                        init = True

                    state_model = [np.array(obs[0].observation.feature_screen),
                                   np.array(obs[0].observation.feature_minimap), np.array(obs[0].observation.player)]
                    # TODO: state_model = [np.array(obs[0].observation.feature_screen), np.array(obs[0].observation.feature_minimap), np.array(obs[0].observation.player), np.array(obs[0].observation.last_actions)]

                    preds = algo.choose_action_v(state_model, init)

                    action, point = action_from_id[np.argmax(preds[0].detach().cpu().numpy())], \
                                    [i for i in range(4096)][np.argmax(preds[1].detach().cpu().numpy())]
                    func, actual_action, new_num_dict = get_action_v3(action, point, obs=obs[0], num_dict=num_dict)

                    next_obs = env.step([func])

                    next_state = get_state(next_obs[0])
                    num_dict = new_num_dict
                    state_model_next = [np.array(obs[0].observation.feature_screen),
                                        np.array(obs[0].observation.feature_minimap),
                                        np.array(obs[0].observation.player)]

                    # reward_a = float(next_obs[0].observation.score_cumulative[11]) * 10e-2          # spent_minerals
                    reward_a = float(next_obs[0].observation.score_cumulative[3])        # total_value_units
                    reward_p = float(next_obs[0].observation.score_cumulative[5] + next_obs[0].observation.score_cumulative[6])  # next_obs[0].observation.score_cumulative[5], [6]: 'killed_value_units' (2745642291968)，'killed_value_structures' (2745642292040)

                    if actual_action == action:
                        reward_a = reward_a * 10
                    reward = [reward_a, reward_p]

                    last_action = obs[0].observation.last_actions

                    if env._controllers and env._controllers[0].status.value != 3:
                        done = True
                        print("episode: {}/{}, score: {}".format(e, max_episode_in_last_play+MAX_EPISODES, score))

                        if env._obs[0].player_result[0].result == 1:  # player0(unknown)胜利
                            reward = list(np.array(reward) + 10000)
                            # for k in range(len(dataset)):
                            #     dataset[k][8] = list(np.array([np.array(dataset[i][8]) for i in range(len(dataset))]) * 100)[k]
                        elif env._obs[0].player_result[0].result == 2:  # player0(unknown)战败
                            reward = list(np.array(reward) - 10000)
                            # for k in range(len(dataset)):
                            #     dataset[k][8] = list(np.array([np.array(dataset[i][8]) for i in range(len(dataset))]) / 100)[k]

                    if time == MAX_STEPS - 1:
                        done = True
                    score += reward_a

                    if done:
                        reward = list(np.array(reward) - 5000)
                        dataset.append(
                            [e, time, state_model, state_model_next, action, actual_action, last_action, point, reward,
                             score, done]
                        )
                        num_dict["barracks"] = 0

                        done = False

                        if train_mode:
                            algo.copy()
                        break

                    dataset.append(
                        [e, time, state_model, state_model_next, action, actual_action, last_action, point, reward,
                         score, done]
                    )
                    if len(dataset) >= 256:  # TODO: HOW TO LEARN ??
                        if train_mode:
                            loss_per_batch = algo.learn(dataset, id_from_actions)
                            losses_lst.append(np.mean(loss_per_batch))
                            print('episode:   ', e, '    critic_network_loss', loss_per_batch)
                        if e % 10 == 0:
                            # 保存为pickle文件
                            with open('./logs/history_data/history_dqn_sequence_bp{}_score{}.pkl'.format(str(batch_pool_idx), str(score)), "wb") as f:
                                pickle.dump(np.array(dataset), f)
                        dataset = []
                        batch_pool_idx += 1

                    state = next_state
                    obs = next_obs

                if train_mode:
                    save_path = './save/dqn/Simple64-dqn-epi{}-score{}.pt'.format(e, score)
                    algo.save(save_path)

            if train_mode:
                plt.plot(losses_lst)
                plt.savefig('./logs/train_process/train_process_dqn_{}_to_{}'.format(max_episode_in_last_play+1, max_episode_in_last_play+MAX_EPISODES+1))
                plt.show()

                pd.DataFrame(losses_lst).to_csv('./logs/train_process/losses_lst.csv')
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app.run(main)
    pass
