# This runner is based on actions.FUNCTIONS (.\anaconda\a3_64\envs\py37_clone_v8\Lib\site-packages\pysc2\lib\actions.py)

from pysc2.lib import actions, features, units

from pysc2.env import sc2_env
from absl import app, logging, flags

from pathlib import Path

import os, sys
import gc
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import algorithm.algorithms.AdvantageActorCritic as A2CAgent
from model.models import SimpleConvNet_prob, SimpleConvNet_val

# from utils import get_state, get_action_v3
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

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

_AI_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_AI_SELECTED = features.SCREEN_FEATURES.selected.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE = actions.FUNCTIONS.Scan_Move_screen.id

_MOVE_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
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

# all units ids: https://github.com/Blizzard/s2client-api/blob/master/include/sc2api/sc2_typeenums.h
_TERRAN_SCV = 45
_TERRAN_MARINE = 48

_TERRAN_COMMANDCENTER = 18
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_MissileTurret = 23
TERRAN_ENGINEERINGBAY = 22

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

    turrent_y, turrent_x = (unit_type == _MissileTurret).nonzero()
    missile_turrets_exist = 1 if turrent_y.any() else 0
    if not missile_turrets_exist:
        num_dict['missile_turrets'] = 0

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

        if _BUILD_MISSLE_TURRENT in obs.observation['available_actions'] and missile_turrets_cnt < 4:
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
                # elif num_dict["barracks"] == 2:
                #     target = (52, 28)
                elif num_dict["barracks"] == 1:
                    target = (52, 38)
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


def tensor2array(T):
    lst = []
    for tensor in T:
        lst.append(tensor.cpu().detach().numpy())
    return lst


def main():
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

    try:
        # run trajectories and train
        with sc2_env.SC2Env(
                map_name="Simple64",
                players=[sc2_env.Agent(sc2_env.Race.terran),
                         # sc2_env.Bot(sc2_env.Race.protoss, sc2_env.Difficulty.cheat_vision)
                         sc2_env.Bot(sc2_env.Race.protoss, sc2_env.Difficulty.very_easy)
                         ],
                visualize=viz,
                agent_interface_format=sc2_env.AgentInterfaceFormat(
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

            for e in range(max_episode_in_last_play+1, max_episode_in_last_play+MAX_EPISODES):
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
                num_dict = {"workers": 0, "idle_workers": 0, "supply_deports": 0, "barracks": 0, "engbays": 0, "marines": 0, "missile_turrets": 0}
                for time in range(MAX_STEPS-1):

                    init = False
                    if e == 0 and time == 0:
                        init = True

                    state_model = [np.array(obs[0].observation.feature_screen), np.array(obs[0].observation.feature_minimap), np.array(obs[0].observation.player)]
                    # TODO: state_model = [np.array(obs[0].observation.feature_screen), np.array(obs[0].observation.feature_minimap), np.array(obs[0].observation.player), np.array(obs[0].observation.last_actions)]

                    preds = agent.act(state_model, init)
                    action, point = np.random.choice(list(action_from_id.values()), 1, p=preds[0].squeeze(0).cpu().detach().numpy())[0], np.random.choice(4096, 1, p=preds[1].squeeze(0).cpu().detach().numpy())[0]

                    func, act_a, new_num_dict = get_action_v3(action, point, obs=obs[0], num_dict=num_dict)
                    num_dict = new_num_dict
                    next_obs = env.step([func])
                    print(act_a, point)

                    next_state = get_state(next_obs[0])
                    next_state_model = [np.array(next_obs[0].observation.feature_screen), np.array(next_obs[0].observation.feature_minimap), np.array(next_obs[0].observation.player)]

                    # reward = float(next_obs[0].reward) + float(np.sum([
                    #     next_obs[0].observation.score_cumulative[0],        # score
                    #     next_obs[0].observation.score_cumulative[3],        # total\_value\_units
                    #     next_obs[0].observation.score_cumulative[4],        # total\_value\_structures
                    #     10*next_obs[0].observation.score_cumulative[5],     # killed\_value\_units
                    #     10*next_obs[0].observation.score_cumulative[6],     # killed\_value\_structures
                    #     next_obs[0].observation.score_cumulative[7],        # collected\_minerals
                    #     next_obs[0].observation.score_cumulative[9],        # collected\_rate\_minerals
                    #     5*next_obs[0].observation.score_cumulative[11]      # spent\_minerals
                    # ])-8*next_obs[0].observation.score_cumulative[2]        # idle\_work\_time
                    #                                            ) * 10e-5           # TODO: add (collected mins - spent mins) as penalty

                    # reward = float(np.sum([
                    #     next_obs[0].observation.score_cumulative[3],        # total\_value\_units
                    #     10*next_obs[0].observation.score_cumulative[5],     # killed\_value\_units
                    #     10*next_obs[0].observation.score_cumulative[6],     # killed\_value\_structures
                    #     next_obs[0].observation.score_cumulative[7],        # collected\_minerals
                    #     next_obs[0].observation.score_cumulative[9],        # collected\_rate\_minerals
                    #     5*next_obs[0].observation.score_cumulative[11]      # spent\_minerals
                    # ])
                    #                -8*next_obs[0].observation.score_cumulative[2]        # idle\_work\_time
                    #                - (
                    #     next_obs[0].observation.score_cumulative[7] - next_obs[0].observation.score_cumulative[11],        # collected\_minerals -   # spent\_minerals
                    #                )
                    #                                            ) * 10e-5           # TODO: add (collected mins - spent mins) as penalty
                    reward = float(np.sum([
                        next_obs[0].observation.score_cumulative[3],        # total\_value\_units
                    ])) * 10e-5           # TODO: add (collected mins - spent mins) as penalty

                    if env._controllers and env._controllers[0].status.value != 3:          # env._controllers[0].status.value = 3 --> game running; env._controllers[0].status.value = 5 --> defeat;
                        done = True
                        # if env._controllers[0].status.value == 5:           # 战败
                        #     reward = reward / 10
                        #     break

                    if time == MAX_STEPS-2:
                        done = True
                        # reward = reward / 5

                    # if next_obs[0].last():
                    #     done = True
                    # if not next_obs[0].last():              #
                    #     done = True

                    last_action = obs[0].observation.last_actions            # TODO
                    # history.append(agent.append_sample(state_model, next_state_model, act_a, point, reward, score))
                    # agent.append_sample(tensor2array(state_model), tensor2array(next_state_model), tensor2array(preds), reward, score)
                    agent.append_sample(state_model, last_action, tensor2array(preds), reward, score)

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
                            log_softmax_actions_1, log_softmax_actions_2 = model_actor([train_dataloader_screen_var, train_dataloader_map_var, train_dataloader_player_var])
                            vs = model_critic([train_dataloader_screen_var, train_dataloader_map_var, train_dataloader_player_var])
                            # calculate qs

                            qs_var = next(iter(qs))
                            advantages = qs_var - vs.detach().squeeze(1)
                            actor_network_loss = - torch.mean(torch.sum(log_softmax_actions_1.cpu()*train_dataloader_a_var, 1) * advantages.cpu()) - torch.mean(torch.sum(log_softmax_actions_2.cpu()*train_dataloader_p_var, 1) * advantages.cpu())
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

                        print('episode: {},   loss of actor: {},   loss of critic: {},   score: {}'.format(e, np.sum(actor_network_losses), np.sum(critic_network_losses), score))
                        gc.collect()

                    else:
                        continue

                    condition1 = e <= 995 and e % 50 == 0
                    condition2 = e > 995
                    if condition1 or condition2:
                        save_path = ['./save/a2c/Simple64-a2c_actor-epi{}.pt'.format(e), './save/a2c/Simple64-a2c_critic-epi{}.pt'.format(e)]
                        agent.save(save_path)
                        # torch.save(model_actor, './save/Simple64-a2c_actor-epi{}.pt'.format(e))
                        # torch.save(model_critic, './save/Simple64-a2c_critic-epi{}.pt'.format(e))
                        history.append([e, agent.states, agent.actions, agent.rewards])
                        # np.savez_compressed("./logs/a2c_e{}.npz".format(e))
                    done = False
                print('time:  ', time)
            env.close()
            print('train complete')
    except KeyboardInterrupt:
        pass
    # finally:
    #     elapsed_time = time.time() - start_time
    #     print("Took %.3f seconds for %s steps: %.3f fps" % (
    #         elapsed_time, total_frames, total_frames / elapsed_time))


if __name__ == '__main__':
    app.run(main)
    pass
