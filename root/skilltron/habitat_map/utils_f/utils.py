import contextlib
import os
import random
import time
import habitat
from collections import OrderedDict, defaultdict, deque

import numpy as np
import torch
import torch.distributed as distrib
import torch.nn as nn
from gym import spaces
from gym.spaces.dict_space import Dict as SpaceDict
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
from typing import Any, Optional, List, Union
from habitat.core.simulator import Observations
from habitat.core.dataset import Dataset
from habitat.core.env import Env
from habitat.utils.visualizations import maps
from habitat import get_config as get_task_config
from habitat.config import Config as CN
import cv2
import transformations as tf
import matplotlib.pyplot as plt
import gym

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.BASE_TASK_CONFIG_PATH = "/habitat-api/configs/tasks/pointnav.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "ppo"
_C.ENV_NAME = "NavRLEnv"
_C.SIMULATOR_GPU_ID = 0
_C.TORCH_GPU_ID = 0
_C.VIDEO_OPTION = ["disk", "tensorboard"]
_C.TENSORBOARD_DIR = "tb"
_C.VIDEO_DIR = "video_dir"
_C.TEST_EPISODE_COUNT = 2
_C.EVAL_CKPT_PATH_DIR = "data/checkpoints"  # path to ckpt or path to ckpts dir
_C.NUM_PROCESSES = 16
_C.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
_C.CHECKPOINT_FOLDER = "data/checkpoints"
_C.NUM_UPDATES = 10000
_C.LOG_INTERVAL = 10
_C.LOG_FILE = "train.log"
_C.CHECKPOINT_INTERVAL = 50
# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# The split to evaluate on
_C.EVAL.SPLIT = "val"
_C.EVAL.USE_CKPT_CONFIG = True
# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (RL) ENVIRONMENT CONFIG
# -----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.REWARD_MEASURE = "distance_to_goal"
_C.RL.SUCCESS_MEASURE = "spl"
_C.RL.SUCCESS_REWARD = 10.0
_C.RL.SLACK_REWARD = -0.01
# -----------------------------------------------------------------------------
# PROXIMAL POLICY OPTIMIZATION (PPO)
# -----------------------------------------------------------------------------
_C.RL.PPO = CN()
_C.RL.PPO.clip_param = 0.2
_C.RL.PPO.ppo_epoch = 4
_C.RL.PPO.num_mini_batch = 16
_C.RL.PPO.value_loss_coef = 0.5
_C.RL.PPO.entropy_coef = 0.01
_C.RL.PPO.lr = 7e-4
_C.RL.PPO.eps = 1e-5
_C.RL.PPO.max_grad_norm = 0.5
_C.RL.PPO.num_steps = 5
_C.RL.PPO.use_gae = True
_C.RL.PPO.use_linear_lr_decay = False
_C.RL.PPO.use_linear_clip_decay = False
_C.RL.PPO.gamma = 0.99
_C.RL.PPO.tau = 0.95
_C.RL.PPO.reward_window_size = 50
_C.RL.PPO.use_normalized_advantage = True
_C.RL.PPO.hidden_size = 512
# -----------------------------------------------------------------------------
# DECENTRALIZED DISTRIBUTED PROXIMAL POLICY OPTIMIZATION (DD-PPO)
# -----------------------------------------------------------------------------
_C.RL.DDPPO = CN()
_C.RL.DDPPO.sync_frac = 0.6
_C.RL.DDPPO.distrib_backend = "GLOO"
_C.RL.DDPPO.rnn_type = "LSTM"
_C.RL.DDPPO.num_recurrent_layers = 2
_C.RL.DDPPO.backbone = "resnet50"
_C.RL.DDPPO.pretrained_weights = "data/ddppo-models/gibson-2plus-resnet50.pth"
# Loads pretrained weights
_C.RL.DDPPO.pretrained = False
# Loads just the visual encoder backbone weights
_C.RL.DDPPO.pretrained_encoder = False
# Whether or not the visual encoder backbone will be trained
_C.RL.DDPPO.train_encoder = True
# Whether or not to reset the critic linear layer
_C.RL.DDPPO.reset_critic = True
# -----------------------------------------------------------------------------
# ORBSLAM2 BASELINE
# -----------------------------------------------------------------------------
_C.ORBSLAM2 = CN()
_C.ORBSLAM2.SLAM_VOCAB_PATH = "habitat_baselines/slambased/data/ORBvoc.txt"
_C.ORBSLAM2.SLAM_SETTINGS_PATH = (
    "habitat_baselines/slambased/data/mp3d3_small1k.yaml"
)
_C.ORBSLAM2.MAP_CELL_SIZE = 0.1
_C.ORBSLAM2.MAP_SIZE = 40
_C.ORBSLAM2.CAMERA_HEIGHT = get_task_config().SIMULATOR.DEPTH_SENSOR.POSITION[
    1
]
_C.ORBSLAM2.BETA = 100
_C.ORBSLAM2.H_OBSTACLE_MIN = 0.3 * _C.ORBSLAM2.CAMERA_HEIGHT
_C.ORBSLAM2.H_OBSTACLE_MAX = 1.0 * _C.ORBSLAM2.CAMERA_HEIGHT
_C.ORBSLAM2.D_OBSTACLE_MIN = 0.1
_C.ORBSLAM2.D_OBSTACLE_MAX = 4.0
_C.ORBSLAM2.PREPROCESS_MAP = True
_C.ORBSLAM2.MIN_PTS_IN_OBSTACLE = (
    get_task_config().SIMULATOR.DEPTH_SENSOR.WIDTH / 2.0
)
_C.ORBSLAM2.ANGLE_TH = float(np.deg2rad(15))
_C.ORBSLAM2.DIST_REACHED_TH = 0.15
_C.ORBSLAM2.NEXT_WAYPOINT_TH = 0.5
_C.ORBSLAM2.NUM_ACTIONS = 3
_C.ORBSLAM2.DIST_TO_STOP = 0.05
_C.ORBSLAM2.PLANNER_MAX_STEPS = 500
_C.ORBSLAM2.DEPTH_DENORM = get_task_config().SIMULATOR.DEPTH_SENSOR.MAX_DEPTH


def get_rl_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()
    print(config.BASE_TASK_CONFIG_PATH)
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)
        
    config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)
    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    config.freeze()
    return config


def init_config():
    W = 640
    H = 360
    config_paths="/data_config/challenge_pointnav2020.local.rgbd.yaml"
    config = habitat.get_config(config_paths=config_paths)
    config.defrost()
    config.SIMULATOR.RGB_SENSOR.HEIGHT = H
    config.SIMULATOR.RGB_SENSOR.WIDTH = W
    config.SIMULATOR.DEPTH_SENSOR.HEIGHT = H
    config.SIMULATOR.DEPTH_SENSOR.WIDTH = W
    config.DATASET.DATA_PATH = '/data/datasets/pointnav/gibson/v2/{split}/{split}.json.gz'
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.SENSORS = ["HEADING_SENSOR", "COMPASS_SENSOR", "GPS_SENSOR", "POINTGOAL_SENSOR", "POINTGOAL_WITH_GPS_COMPASS_SENSOR"]
    config.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.DIMENSIONALITY = 3
    config.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.GOAL_FORMAT = "CARTESIAN"
    config.TASK.POINTGOAL_SENSOR.DIMENSIONALITY = 3
    config.TASK.POINTGOAL_SENSOR.GOAL_FORMAT = "CARTESIAN"
    config.TASK.GPS_SENSOR.DIMENSIONALITY = 3
    config.TASK.GPS_SENSOR.GOAL_FORMAT = "CARTESIAN"
    config.TASK.AGENT_POSITION_SENSOR = habitat.Config()
    config.TASK.AGENT_POSITION_SENSOR.TYPE = "position_sensor"
    config.TASK.AGENT_POSITION_SENSOR.ANSWER_TO_LIFE = 42
    config.TASK.SENSORS.append("AGENT_POSITION_SENSOR")
    config.SIMULATOR.TURN_ANGLE = 10
    config.SIMULATOR.TILT_ANGLE = 10
    config.SIMULATOR.FORWARD_STEP_SIZE = 0.25
    config.ENVIRONMENT.MAX_EPISODE_STEPS = 100000
    config.TASK.TOP_DOWN_MAP.MAX_EPISODE_STEPS = 100000
    config.DATASET.SCENES_DIR = '/data/scene_datasets'
    config.DATASET.SPLIT = 'train'
    config.SIMULATOR.SCENE = '/data/scene_datasets/gibson/Aldrich.glb'
    config.freeze()
    return config

class FrameStack(gym.Wrapper):
    def __init__(self, env, k, channel_order='hwc'):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        super().__init__(env)
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames_rgb = deque([], maxlen=k)
        self.frames_depth = deque([], maxlen=k)
        self.frames_pose = deque([], maxlen=k)
        

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames_rgb.append(ob['rgb'])
            self.frames_depth.append(ob['depth'])
            self.frames_pose.append(ob['pos'])
            
        return {'rgb':torch.cat(list(self.frames_rgb), dim=1), 'depth':torch.cat(list(self.frames_depth), dim=1), 'pos':torch.cat(list(self.frames_pose), dim=0).unsqueeze(0)}

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames_rgb.append(ob['rgb'])
        self.frames_depth.append(ob['depth'])
        self.frames_pose.append(ob['pos'])
        return {'rgb':torch.cat(list(self.frames_rgb), dim=1), 'depth':torch.cat(list(self.frames_depth), dim=1), 'pos':torch.cat(list(self.frames_pose), dim=0).unsqueeze(0)}, reward, done, info

class FrameSkip(gym.Wrapper):
    """Return every `skip`-th frame and repeat given action during skip.
    Note that this wrapper does not "maximize" over the skipped frames.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)

        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break      
        return obs, total_reward, done, info        

@habitat.registry.register_sensor(name="position_sensor")
class AgentPositionSensor(habitat.Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)

        self._sim = sim

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "agent_position"

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.POSITION

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    # This is called whenver reset is called or an action is taken
    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        sensor_states = self._sim.get_agent_state().sensor_states
        return (sensor_states['rgb'].position, sensor_states['rgb'].rotation)

def draw_top_down_map(info, heading, output_size):
    top_down_map = maps.colorize_topdown_map(
        info["top_down_map"]["map"], info["top_down_map"]["fog_of_war_mask"]
    )
    original_map_size = top_down_map.shape[:2]
    map_scale = np.array(
        (1, original_map_size[1] * 1.0 / original_map_size[0])
    )
    new_map_size = np.round(output_size * map_scale).astype(np.int32)
    # OpenCV expects w, h but map size is in h, w
    top_down_map = cv2.resize(top_down_map, (new_map_size[1], new_map_size[0]))

    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    map_agent_pos = np.round(
        map_agent_pos * new_map_size / original_map_size
    ).astype(np.int32)
    top_down_map = maps.draw_agent(
        top_down_map,
        map_agent_pos,
        heading - np.pi / 2,
        agent_radius_px=top_down_map.shape[0] / 40,
    )
    return top_down_map 

def inverse_transform(x, y, start_x, start_y, start_angle):
    new_x = (x - start_x) * np.cos(start_angle) + (y - start_y) * np.sin(start_angle)
    new_y = -(x - start_x) * np.sin(start_angle) + (y - start_y) * np.cos(start_angle)
    return new_x, new_y


def make_train_data(reward, done, value, gamma, num_step, num_worker, use_gae, lam):
    discounted_return = np.empty([num_worker, num_step])
    # Discounted Return
    if use_gae:
        gae = np.zeros_like([num_worker, ])
        for t in range(num_step - 1, -1, -1):

            delta = reward[:, t] + gamma * value[:, t + 1] * (1 - done[:, t]) - value[:, t]

            gae = delta + gamma * lam * (1 - done[:, t]) * gae

            discounted_return[:, t] = gae + value[:, t]

            # For Actor
        adv = discounted_return - value[:, :-1]
    else:
        running_add = value[:, -1]
        for t in range(num_step - 1, -1, -1):
            running_add = reward[:, t] + gamma * running_add * (1 - done[:, t])
            discounted_return[:, t] = running_add
        # For Actor
        adv = discounted_return - value[:, :-1]
    return discounted_return.reshape([-1]), adv.reshape([-1])

class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count          


def global_grad_norm_(parameters, norm_type=2):
    
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == np.inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

    return total_norm

def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np


    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap

def plot_colortable(colors, title, sort_colors=False, emptycols=0):
    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12
    topmargin = 40

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                         name)
                        for name, color in colors.items())
        names = [name for hsv, name in by_hsv]
    else:
        names = list(colors)

    n = len(names)
    ncols = 4 - emptycols
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + margin + topmargin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-topmargin)/height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title(title, fontsize=24, loc="left")

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        swatch_end_x = cell_width * col + swatch_width
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.hlines(y, swatch_start_x, swatch_end_x,
                  color=colors[name], linewidth=18)
        
    plt.show()


