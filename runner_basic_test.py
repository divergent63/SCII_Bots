#!/usr/bin/env python
# coding=utf-8
'''
Author: ZZ_Guo
Email: 634184805@qq.com
Date: 2020-09-11 23:03:00
LastEditor: ZZ_Guo
LastEditTime: 2021-05-06 17:04:38
Discription:
Environment:
'''
from pysc2.lib import actions, features, units

from pysc2.env import sc2_env, run_loop, available_actions_printer
from pysc2 import maps

import sc2
from sc2 import run_game, maps, Race, Difficulty, position
from sc2.player import Bot, Computer, Human
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
    CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY, OBSERVER, ROBOTICSFACILITY
# from terran_agent import TerranAgent

import models.models as models
import algorithms.q_learning as q_learning

from pathlib import Path
from absl import app, logging, flags

import random
import math
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
# for ix, k in enumerate(spatial_actions):
#     id_from_actions[k] = ix
#     action_from_id[ix] = k
# for ix, k in enumerate(categorical_actions):
#     id_from_actions[k] = ix+len(spatial_actions)
#     action_from_id[ix+len(spatial_actions)] = k
for ix, k in enumerate(categorical_actions):
    id_from_actions[k] = ix
    action_from_id[ix] = k


FLAGS = flags.FLAGS
FLAGS(['run_sc2'])

# def get_action_v3(state):
#
#     pass


def get_state(obs):

    return [np.array(obs.observation['feature_screen']).reshape(1, 27, 64, 64),
            np.array(obs.observation['feature_minimap']).reshape(1, 11, 64, 64),
            np.array(obs.observation['player']).reshape(1, 11)
            ]


def to_yx(point):
    """transform a scalar from [0;4095] to a (y,x) coordinate in [0:63,0:63]"""
    return point % 64, (point - (point % 64)) / 64


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
    supply_depot_count = supply_depot_count = 1 if depot_y.any() else 0

    barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
    barracks_count = 1 if barracks_y.any() else 0

    supply_limit = obs.observation['player'][4]
    army_supply = obs.observation['player'][5]
    food_workers = obs.observation['player'][6]
    idle_workers_cnt = obs.observation['player'][7]
    army_cnt = obs.observation['player'][8]

    killed_unit_score = obs.observation['score_cumulative'][5]
    killed_building_score = obs.observation['score_cumulative'][6]

    current_state = np.zeros(20)
    current_state[0] = supply_depot_count
    current_state[1] = barracks_count
    current_state[2] = supply_limit
    current_state[3] = army_supply

    hot_squares = np.zeros(16)

    army_selected = False
    army_rallied = False

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
            if mineral_y.any():
                i = random.randint(0, len(scv_y) - 1)
                # target = (mineral_y[i], mineral_y[i])
                # target = (mineral_y.mean(), mineral_y.mean())
                # target = (scv_y.mean(), scv_x.mean())
                target = (scv_y[i], scv_x[i])
                # target = (11, 16)
                func = actions.FunctionCall(_COLLECT_RESOURCES, [_NOT_QUEUED, target])

    elif smart_action == ACTION_BUILD_SUPPLY_DEPOT:
        deports_cnt = num_dict["supply_deports"]
        if _BUILD_SUPPLY_DEPOT in obs.observation['available_actions'] and deports_cnt < 4:
            unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

            if unit_y.any():
                if num_dict["supply_deports"] == 0:
                    target = (31, 8)
                elif num_dict["supply_deports"] == 1:
                    target = (26, 8)
                elif num_dict["supply_deports"] == 2:
                    target = (21, 8)
                elif num_dict["supply_deports"] == 3:
                    target = (16, 8)
                else:
                    target = to_yx(point)

                func = actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])

                try:
                    num_dict["supply_deports"] += 1
                    return func, smart_action, num_dict
                except UnboundLocalError:
                    num_dict["supply_deports"] -= 1
                    print(str(smart_action) + " " + str(point) + " is not an available action")
                    return get_action_v3(action_from_id[0], point, obs, num_dict)

    elif smart_action == ACTION_BUILD_BARRACKS:
        if _BUILD_BARRACKS in obs.observation['available_actions']:
            unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

            if unit_y.any() and num_dict['barracks'] < 3:
                # target = to_yx(point)
                if num_dict["barracks"] == 0:
                    target = (56, 18)
                elif num_dict["barracks"] == 1:
                    target = (56, 28)
                elif num_dict["barracks"] == 2:
                    target = (56, 38)
                else:
                    target = to_yx(point)

                func = actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])
                try:
                    num_dict["barracks"] += 1
                    return func, smart_action, num_dict
                except UnboundLocalError:
                    num_dict["barracks"] -= 1
                    print(str(smart_action) + " " + str(point) + " is not an available action")
                    if num_dict['supply_deports'] == 0:
                        return get_action_v3(action_from_id[1], point, obs, num_dict)
                    else:
                        return get_action_v3(action_from_id[0], point, obs, num_dict)

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
                print(str(smart_action) + " " + str(point) + " is not an available action")
                return get_action_v3(action_from_id[2], point, obs, num_dict)

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
            print(str(smart_action) + " " + str(point) + " is not an available action")
            return get_action_v3(action_from_id[3], point, obs, num_dict)

    elif smart_action == ACTION_SELECT_ARMY:
        if _SELECT_ARMY in obs.observation['available_actions']:
            func = actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
        try:
            return func, smart_action, num_dict
        except UnboundLocalError:
            print(str(smart_action) + " " + str(point) + " is not an available action")
            return get_action_v3(action_from_id[4], point, obs, num_dict)

    elif smart_action == ACTION_ATTACK:
        enemy_y, enemy_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()

        if (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero():  # 攻击已知敌人
            # for i in range(0, len(enemy_y)):
            # marines_cnt = num_dict["marines"]

            if len(obs.observation['multi_select']) and army_cnt > 12 and num_dict['attack_cnt'] < 2:
                # if obs.observation['multi_select'][0][0] != _TERRAN_SCV and _ATTACK_MINIMAP in obs.observation["available_actions"]:
                if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                    # if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                    if enemy_y.any():
                        target = [int(np.random.choice(enemy_x)), int(np.random.choice(enemy_y))]
                        # target = to_yx(point)           # TODO:
                        func = actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, target])
                        # num_dict['marines'] = 0
                        num_dict['attack_cnt'] += 1
            elif num_dict['attack_cnt'] >= 2 and len(obs.observation['multi_select']) and army_cnt >= 3:
                    # if obs.observation['multi_select'][0][0] != _TERRAN_SCV and _ATTACK_MINIMAP in obs.observation["available_actions"]:
                    if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                        # if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                        if enemy_y.any():
                            target = [int(np.random.choice(enemy_x)), int(np.random.choice(enemy_y))]
                            # target = to_yx(point)           # TODO:
                            func = actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, target])
                            # num_dict['marines'] = 0
                            num_dict['attack_cnt'] += 1
            # else:
            #     if len(obs.observation['multi_select']):
            #         # if obs.observation['multi_select'][0][0] != _TERRAN_SCV and _ATTACK_MINIMAP in obs.observation["available_actions"]:
            #         if _ATTACK_MINIMAP in obs.observation["available_actions"]:
            #             # if _ATTACK_MINIMAP in obs.observation["available_actions"]:
            #             if enemy_y.any():
            #                 target = [int(np.random.choice(enemy_x)), int(np.random.choice(enemy_y))]
            #                 # target = to_yx(point)           # TODO:
            #                 func = actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, target])
            #                 # num_dict['marines'] = 0
            #                 num_dict['attack_cnt'] += 1
        else:  # 攻击任意位置（未找到敌人时，类似巡逻）
            if len(obs.observation['multi_select']):
                # if obs.observation['multi_select'][0][0] != _TERRAN_SCV and _ATTACK_MINIMAP in obs.observation["available_actions"]:
                if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                    target = to_yx(point)
                    func = actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, target])
        try:
            return func, smart_action, num_dict
        except UnboundLocalError:
            num_dict['attack_cnt'] -= 1
            print(str(smart_action) + " " + str(point) + " is not an available action")
            return get_action_v3(action_from_id[4], point, obs, num_dict)

    elif smart_action == ACTION_BUILD_ENGBAY:
        engbays_cnt = num_dict["engbays"]

        if _BUILD_ENG_BAY in obs.observation['available_actions'] and engbays_cnt == 0:
            unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

            if unit_y.any():
                # target = to_yx(point)
                target = (38, 44)
                func = actions.FunctionCall(_BUILD_ENG_BAY, [_NOT_QUEUED, target])
        try:
            num_dict["engbays"] += 1
            return func, smart_action, num_dict
        except UnboundLocalError:
            num_dict["engbays"] -= 1
            print(str(smart_action) + " " + str(point) + " is not an available action")
            return get_action_v3(action_from_id[0], point, obs, num_dict)

    elif smart_action == ACTION_BUILD_MISSLE_TURRENT:
        missile_turrets_cnt = num_dict["missile_turrets"]

        if _BUILD_MISSLE_TURRENT in obs.observation['available_actions'] and missile_turrets_cnt < 16:
            unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()

            if unit_y.any():
                target = [(62, 16), (62, 26), (62, 36)]
                if num_dict['missile_turrets'] == 0:
                    func = actions.FunctionCall(_BUILD_MISSLE_TURRENT, [_NOT_QUEUED, target[0]])
                elif num_dict['missile_turrets'] == 1:
                    func = actions.FunctionCall(_BUILD_MISSLE_TURRENT, [_NOT_QUEUED, target[1]])
                elif num_dict['missile_turrets'] == 2:
                    func = actions.FunctionCall(_BUILD_MISSLE_TURRENT, [_NOT_QUEUED, target[2]])
                else:
                    target = to_yx(point)
                    func = actions.FunctionCall(_BUILD_MISSLE_TURRENT, [_NOT_QUEUED, target])

        try:
            num_dict['missile_turrets'] += 1
            return func, smart_action, num_dict
        except UnboundLocalError:
            num_dict['missile_turrets'] -= 1
            print(str(smart_action) + " " + str(point) + " is not an available action")
            return get_action_v3(action_from_id[5], point, obs, num_dict)

    elif smart_action == ACTION_DO_NOTHING:
        func = actions.FunctionCall(_NO_OP, [])

    try:
        return func, smart_action, num_dict

    except UnboundLocalError:
        print(str(smart_action) + " " + str(point) + " is not an available action")
        return actions.FunctionCall(_NO_OP, []), ACTION_DO_NOTHING, num_dict


def main(unused_argv):

    viz = True
    replay_prefix = 'D:/software/python_prj/SCII/SCII_Bots/replays/deterministic_sequence'
    replay_dir = '/replays'
    real_time = False
    ensure_available_actions = True
    disable_fog = True
    steps_per_episode = 0  # 0 actually means unlimited
    MAX_EPISODES = 1
    MAX_STEPS = 5000
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
                save_replay_episodes=1,
                replay_prefix=replay_prefix,
                replay_dir=replay_dir,
        ) as env:

            done = False
            history = []

            for e in range(MAX_EPISODES):
                obs = env.reset()

                score = 0
                score_pre = 0
                state = get_state(obs[0])

                control_seq = []
                control_idx = 1

                num_dict = {"workers": 0, "idle_workers": 0, "supply_deports": 0, "barracks": 0, "engbays": 0,
                            # "marines": 0,
                            "missile_turrets": 0, 'attack_cnt': 0}
                for time in range(MAX_STEPS):
                    init = False
                    if e == 0 and time == 0:
                        init = True

                    a, point = action_from_id[np.random.choice(len(action_from_id), 1)[0]], np.random.randint(4096)
                    # # TODO: build supply deport with deterministic time sequence
                    # point = np.random.randint(4096)
                    # a = action_from_id[control_idx]
                    # # control_seq.append(action_from_id[control_idx])
                    # control_seq.append(action_from_id[control_idx+1])
                    # if len(control_seq) > 0:
                    #     a = control_seq[-1]
                    func, act_a, new_num_dict = get_action_v3(a, point, obs=obs[0], num_dict=num_dict)
                    # if act_a == a:
                    #     control_seq.pop()
                    #     control_idx += 1

                    next_obs = env.step([func])
                    print(act_a, point)

                    next_state = get_state(next_obs[0])
                    num_dict = new_num_dict

                    reward = float(next_obs[0].reward) + float(np.sum(next_obs[0].observation.score_cumulative)) * 10e-8

                    if env._controllers and env._controllers[0].status.value != 3:
                        done = True
                    if time == MAX_STEPS - 1:
                        done = True

                    state = next_state
                    obs = next_obs
                    if done:
                        print("episode: {}/{}, score: {}".format(e, MAX_EPISODES, score))
                        if score_pre < score:
                            score_pre = score

                        done = False

                    if time % 10 == 0:
                        if score_pre < score:
                            # save agent model
                            history.append(
                                [e, time, state, next_state, a, act_a, point, reward, score, done]
                            )

                    score += reward

                    time += 1
                history_arr = np.array(history)
                np.savez_compressed('./logs/history_deterministic_sequence_{}.npz'.format(str(e)), history_arr)
    except KeyboardInterrupt:
        pass
    # finally:
    #     elapsed_time = time.time() - start_time
    #     print("Took %.3f seconds for %s steps: %.3f fps" % (
    #         elapsed_time, total_frames, total_frames / elapsed_time))


if __name__ == '__main__':
    app.run(main)
