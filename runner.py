from pysc2.lib import actions, features, units

from pysc2.env import sc2_env, run_loop, available_actions_printer
from pysc2 import maps
from absl import flags

import sc2
from sc2 import run_game, maps, Race, Difficulty, position
from sc2.player import Bot, Computer, Human
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
    CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY, OBSERVER, ROBOTICSFACILITY
# from terran_agent import TerranAgent

import random
from pathlib import Path

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from random_agent import RandomAgent
# from network import FullyConv
from utils import get_state, get_action_v2
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
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_SELECT_ARMY,
    ACTION_TRAIN_MARINE,
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
for ix, k in enumerate(spatial_actions):
    id_from_actions[k] = ix
    action_from_id[ix] = k
for ix, k in enumerate(categorical_actions):
    id_from_actions[k] = ix+len(spatial_actions)
    action_from_id[ix+len(spatial_actions)] = k

# initialize NN model hyperparameters
eta = 0.1
expl_rate = 0.2

# initialize model object
# model = FullyConv(eta, expl_rate, categorical_actions,spatial_actions)
model = None

# initalize Agent
agent = RandomAgent(model, categorical_actions, spatial_actions, id_from_actions, action_from_id)

FLAGS = flags.FLAGS
FLAGS(['run_sc2'])

viz = True
save_replay = False
real_time = False
ensure_available_actions = True
disable_fog = True

steps_per_episode = 0   # 0 actually means unlimited
MAX_EPISODES = 5
MAX_STEPS = 50
steps = 0

# run trajectories and train
with sc2_env.SC2Env(
        map_name="Simple64",
        players=[sc2_env.Agent(sc2_env.Race.terran),
                 # sc2_env.Bot(sc2_env.Race.protoss, sc2_env.Difficulty.very_easy)
                 sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.easy)
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
    if model and Path(Path(os.getcwd()) / 'save' / 'Simple64-a2c.h5').is_file():
        agent.load("./save/Simple64-a2c.h5")

    done = False
    history = []

    for e in range(MAX_EPISODES):
        obs = env.reset()

        score = 0
        score_pre = 0
        state = get_state(obs[0])
        time = 0

        for time in range(MAX_STEPS):

            init = False
            if e == 0 and time == 0:
                init = True

            a, point = agent.act_randomly()

            func, act_a = get_action_v2(a, point, obs=obs[0])
            next_obs = env.step([func])
            print(act_a, point)

            next_state = get_state(next_obs[0])

            reward = float(next_obs[0].reward) + float(np.sum(next_obs[0].observation.score_cumulative)) * 10e-8

            if env._controllers and env._controllers[0].status.value != 3:
                done = True
            if time == MAX_STEPS-1:
                done = True
            # done = next_obs[0].last()
            
            agent.append_sample(state, act_a, point, reward, score)
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
                        [e, time, state, next_state, act_a, reward, score, done]
                    )

            score += reward

            time += 1
    history_arr = np.array(history)
    # np.savez_compressed('history_random.npz', history_arr)
