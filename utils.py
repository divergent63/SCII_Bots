import numpy as np
import random
import math

from pysc2.agents import base_agent
from pysc2.lib import actions, features, units

from pysc2.env import sc2_env, run_loop, available_actions_printer
from pysc2 import maps

np.random.seed(1)

_AI_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_AI_SELECTED = features.SCREEN_FEATURES.selected.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE = actions.FUNCTIONS.Scan_Move_screen.id

_MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id

_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
# _ATTACK_MINIMAP = actions.FUNCTIONS.Attack_Attack_minimap.id

_COLLECT_RESOUCES = actions.FUNCTIONS.Harvest_Gather_SCV_screen.id
_BUILD_MISSLE_TURRENT = actions.FUNCTIONS.Build_MissileTurret_screen.id
_BUILD_ENG_BAY = actions.FUNCTIONS.Build_EngineeringBay_screen.id

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_TERRAN_ENGBAY = 22

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

KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5


def get_state(obs):

    return [np.array(obs.observation['feature_screen']).reshape(1, 27, 64, 64),
            np.array(obs.observation['feature_minimap']).reshape(1, 11, 64, 64),
            np.array(obs.observation['player']).reshape(1, 11)
            ]


def to_yx(point):
    """transform a scalar from [0;4096] to a (y,x) coordinate in 64,64"""
    return point % 64, (point - (point % 64)) / 64


def transformLocation(obs, x, y):
    player_y, player_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()

    base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
    if not base_top_left:
        return [64 - x, 64 - y]
    else:
        return [x, y]


def get_action_v2(id_action, point, obs):
    # obs = obs[0]
    unit_type = obs.observation['feature_screen'][_UNIT_TYPE]

    depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
    supply_depot_count = supply_depot_count = 1 if depot_y.any() else 0

    barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
    barracks_count = 1 if barracks_y.any() else 0

    supply_limit = obs.observation['player'][4]
    army_supply = obs.observation['player'][5]

    killed_unit_score = obs.observation['score_cumulative'][5]
    killed_building_score = obs.observation['score_cumulative'][6]

    current_state = np.zeros(20)
    current_state[0] = supply_depot_count
    current_state[1] = barracks_count
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
        worker_cnt = 0
        if _TRAIN_SCV in obs.observation['available_actions'] and worker_cnt < 16:
            func = actions.FunctionCall(_TRAIN_SCV, [_QUEUED])

    # elif smart_action == ACTION_COLLECT_RESOUCES:
    #     unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
    #     unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
    #
    #     if unit_y.any():
    #         i = random.randint(0, len(unit_y) - 1)
    #         target = [unit_x[i], unit_y[i]]
    #
    #         func = actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

    elif smart_action == ACTION_BUILD_SUPPLY_DEPOT:
        if _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
            unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

            if unit_y.any():
                target = to_yx(point)

                func = actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])

    elif smart_action == ACTION_BUILD_BARRACKS:
        if _BUILD_BARRACKS in obs.observation['available_actions']:
            unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

            if unit_y.any():
                target = to_yx(point)
                func = actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

    elif smart_action == ACTION_SELECT_BARRACKS:
        unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()

        if unit_y.any():
            target = [int(unit_x.mean()), int(unit_y.mean())]
            func = actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

    elif smart_action == ACTION_TRAIN_MARINE:
        if _TRAIN_MARINE in obs.observation['available_actions']:
            func = actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

    elif smart_action == ACTION_SELECT_ARMY:
        if _SELECT_ARMY in obs.observation['available_actions']:
            func = actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

    elif smart_action == ACTION_ATTACK:
        enemy_y, enemy_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()

        if (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero():
            for i in range(0, len(enemy_y)):
                # y = int(math.ceil((enemy_y[i] + 1) / 16))
                # x = int(math.ceil((enemy_x[i] + 1) / 16))
                # x_offset = random.randint(-4, 4)
                # y_offset = random.randint(-4, 4)
                if len(obs.observation['multi_select']):
                    if obs.observation['multi_select'][0][0] != _TERRAN_SCV and _ATTACK_MINIMAP in obs.observation["available_actions"]:
                        # if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                        if enemy_y.any():
                            target = [int(enemy_x.mean()), int(enemy_y.mean())]
                            func = actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, target])
                            # func = actions.FunctionCall(_ATTACK_MINIMAP, [_QUEUED, transformLocation(obs, int(x_offset) + int(x), int(y_offset) + int(y))])
                            # func = actions.FUNCTIONS.Attack_minimap('now', obs, transformLocation(int(x) + (x_offset * 8), int(y) + (y_offset * 8)))
            # if _SELECT_ARMY in obs.observation['available_actions']:
            #     func = actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
        else:
            if len(obs.observation['multi_select']):
                if obs.observation['multi_select'][0][0] != _TERRAN_SCV and _ATTACK_MINIMAP in obs.observation[
                    "available_actions"]:
                    target = to_yx(point)
                    func = actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, target])

    elif smart_action == ACTION_BUILD_ENGBAY:
        if _BUILD_ENG_BAY in obs.observation['available_actions']:
            unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

            if unit_y.any():
                target = to_yx(point)
                func = actions.FunctionCall(_BUILD_ENG_BAY, [_NOT_QUEUED, target])

    elif smart_action == ACTION_BUILD_MISSLE_TURRENT:
        if _BUILD_MISSLE_TURRENT in obs.observation['available_actions']:
            unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()

            if unit_y.any():
                target = to_yx(point)
                func = actions.FunctionCall(_BUILD_MISSLE_TURRENT, [_NOT_QUEUED, target])

    elif smart_action == ACTION_DO_NOTHING:
        func = actions.FunctionCall(_NO_OP, [])

    try:
        return func, smart_action

    except UnboundLocalError:
        print(str(smart_action) + " " + str(point) + " is not an available action")
        smart_action = ACTION_SELECT_SCV
        # return actions.FunctionCall(_NO_OP, []), smart_action
        target = to_yx(point)
        return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target]), smart_action
