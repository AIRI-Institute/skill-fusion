import random
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast

import gym
import numba
import numpy as np
from gym import spaces

from habitat.config import read_write
from habitat.config import DictConfig as Config
from habitat.core.dataset import Dataset, Episode, EpisodeIterator
from habitat.core.embodied_task import EmbodiedTask, Metrics
from habitat.core.simulator import Observations, Simulator
from habitat.datasets import make_dataset
from habitat.sims import make_sim
from habitat.tasks.registration import make_task
from habitat_sim.agent.agent import AgentState
from habitat_sim.utils.common import quat_from_angle_axis

import math
import quaternion
import skimage
import skimage.morphology

import torch
import copy
import cv2
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from gym.spaces.box import Box
import pandas as pd
from .utils_f import pose as pu

def resize2d(img, size):
    return (F.adaptive_avg_pool2d(Variable(img,volatile=True), size)).data

def _preprocess_depth(depth):
    depth = depth[:, :, 0]*1
    mask2 = depth > 0.99
    depth[mask2] = 0.

    for i in range(depth.shape[1]):
        depth[:,i][depth[:,i] == 0.] = depth[:,i].max()

    mask1 = depth == 0
    depth[mask1] = np.NaN
    depth = depth*450. + 50.
    return depth

def get_grid(pose, grid_size, device):
    """
    Input:
        `pose` FloatTensor(bs, 3)
        `grid_size` 4-tuple (bs, _, grid_h, grid_w)
        `device` torch.device (cpu or gpu)
    Output:
        `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
        `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)

    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    bs = x.size(0)
    t = t * np.pi / 180.
    cos_t = t.cos()
    sin_t = t.sin()

    theta11 = torch.stack([cos_t, -sin_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta12 = torch.stack([sin_t, cos_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta1 = torch.stack([theta11, theta12], 1)

    theta21 = torch.stack([torch.ones(x.shape).to(device),
                           -torch.zeros(x.shape).to(device), x], 1)
    theta22 = torch.stack([torch.zeros(x.shape).to(device),
                           torch.ones(x.shape).to(device), y], 1)
    theta2 = torch.stack([theta21, theta22], 1)

    rot_grid = F.affine_grid(theta1, torch.Size(grid_size))
    trans_grid = F.affine_grid(theta2, torch.Size(grid_size))

    return rot_grid, trans_grid

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__  


class Env:
    r"""Fundamental environment class for :ref:`habitat`.
    :data observation_space: ``SpaceDict`` object corresponding to sensor in
        sim and task.
    :data action_space: ``gym.space`` object corresponding to valid actions.
    All the information  needed for working on embodied tasks with simulator
    is abstracted inside :ref:`Env`. Acts as a base for other derived
    environment classes. :ref:`Env` consists of three major components:
    ``dataset`` (`episodes`), ``simulator`` (:ref:`sim`) and :ref:`task` and
    connects all the three components together.
    """

    observation_space: spaces.Dict
    action_space: spaces.Dict
    _config: "DictConfig"
    _dataset: Optional[Dataset]
    number_of_episodes: Optional[int]
    _episodes: List[Episode]
    _current_episode_index: Optional[int]
    _current_episode: Optional[Episode]
    _episode_iterator: Optional[Iterator]
    _sim: Simulator
    _task: EmbodiedTask
    _max_episode_seconds: int
    _max_episode_steps: int
    _elapsed_steps: int
    _episode_start_time: Optional[float]
    _episode_over: bool

    def __init__(
        self, config: "DictConfig", dataset: Optional[Dataset[Episode]] = None
    ) -> None:
        """Constructor
        :param config: config for the environment. Should contain id for
            simulator and ``task_name`` which are passed into ``make_sim`` and
            ``make_task``.
        :param dataset: reference to dataset for task instance level
            information. Can be defined as :py:`None` in which case
            ``_episodes`` should be populated from outside.
        """


        if "habitat" in config:
            config = config.habitat
        self._config = config
        
        self.dt = config.simulator.turn_angle
        self._dataset = dataset
        self._current_episode_index = None
        if self._dataset is None and self._config.dataset.type:
            self._dataset = make_dataset(
                id_dataset=self._config.dataset.type, config=self._config.dataset
            )
        self._episodes = (
            self._dataset.episodes
            if self._dataset
            else cast(List[Episode], [])
        )
        self._current_episode = None
        iter_option_dict = {
            k.lower(): v
            for k, v in self._config.environment.iterator_options.items()
        }
        #iter_option_dict["seed"] = config.SEED
        self._episode_iterator = self._dataset.get_episode_iterator(
            **iter_option_dict
        )

        # load the first scene if dataset is present
        if self._dataset:
            assert (len(self._dataset.episodes) > 0), "dataset should have non-empty episodes list"
            self._setup_episode_iterator()
            self.current_episode = next(self.episode_iterator)
            with read_write(self._config):
                self._config.simulator.scene_dataset = (self.current_episode.scene_dataset_config)
                self._config.simulator.scene = self.current_episode.scene_id
            self.number_of_episodes = len(self.episodes)
        else:
            self.number_of_episodes = None

        self._sim = make_sim(id_sim=self._config.simulator.type, config=self._config.simulator)
        self._task = make_task(
            self._config.task.type,
            config=self._config.task,
            sim=self._sim,
            dataset=self._dataset)
        
        self.observation_space = spaces.Dict(
            {
                **self._sim.sensor_suite.observation_spaces.spaces,
                **self._task.sensor_suite.observation_spaces.spaces,
            }
        )
        self.action_space = self._task.action_space
        self._max_episode_seconds = (self._config.environment.max_episode_seconds)
        self._max_episode_steps = self._config.environment.max_episode_steps
        self._elapsed_steps = 0
        self._episode_start_time: Optional[float] = None
        self._episode_over = False

        self.eps = list(range(len(self._episode_iterator.episodes)))

        
        self.scenes_eps = []        
        a = ''
        for ii,i in enumerate(np.array(self.eps)):
            b = self._episode_iterator.episodes[i].scene_id
            if a!=b:
                print(ii,b)
                a = b
                self.scenes_eps.append(ii)
        print('NUMBER OF SCENES ',len(self.scenes_eps))        
        print('NUMBER OF EPS ',len(self.eps))  
        
        del self.observation_space.spaces['objectgoal']
        #del self.observation_space.spaces['heading']
        del self.observation_space.spaces['gps']
        del self.observation_space.spaces['compass']

        
        self.observation_space.spaces['depth'] = Box(low=-1000, high=1000, shape=(240,320,1), dtype=np.float32)
        self.observation_space.spaces['rgb'] = Box(low=-1000, high=1000, shape=(240,320,3), dtype=np.float32)
        #self.observation_space.spaces['pos'] = Box(low=-1000, high=1000, shape=(2,), dtype=np.float32)
        #self.observation_space.spaces['semantic'] = Box(low=-1000, high=1000, shape=(256,256,1), dtype=np.float32)
        
        
        self.action_space = self._task.action_space
        self.observation_space.spaces['semantic'] = Box(low=-1000, high=1000, shape=(240,320,1), dtype=np.float32)

        self.cord_goal = None

        
    def _setup_episode_iterator(self):
        assert self._dataset is not None
        iter_option_dict = {
            k.lower(): v
            for k, v in self._config.environment.iterator_options.items()
        }
        iter_option_dict["seed"] = self._config.seed
        self._episode_iterator = self._dataset.get_episode_iterator(
            **iter_option_dict
        )    
        
    @property
    def current_episode(self) -> Episode:
        assert self._current_episode is not None
        return self._current_episode

    @current_episode.setter
    def current_episode(self, episode: Episode) -> None:
        self._current_episode = episode

    @property
    def episode_iterator(self) -> Iterator:
        return self._episode_iterator

    @episode_iterator.setter
    def episode_iterator(self, new_iter: Iterator) -> None:
        self._episode_iterator = new_iter

    @property
    def episodes(self) -> List[Episode]:
        return self._episodes

    @episodes.setter
    def episodes(self, episodes: List[Episode]) -> None:
        assert (
            len(episodes) > 0
        ), "Environment doesn't accept empty episodes list."
        self._episodes = episodes

    @property
    def sim(self) -> Simulator:
        return self._sim

    @property
    def episode_start_time(self) -> Optional[float]:
        return self._episode_start_time

    @property
    def episode_over(self) -> bool:
        return self._episode_over

    @property
    def task(self) -> EmbodiedTask:
        return self._task

    @property
    def _elapsed_seconds(self) -> float:
        assert (
            self._episode_start_time
        ), "Elapsed seconds requested before episode was started."
        return time.time() - self._episode_start_time

    def get_metrics(self) -> Metrics:
        return self._task.measurements.get_metrics()

    def _past_limit(self) -> bool:
        if (
            self._max_episode_steps != 0
            and self._max_episode_steps <= self._elapsed_steps
        ):
            return True
        elif (
            self._max_episode_seconds != 0
            and self._max_episode_seconds <= self._elapsed_seconds
        ):
            return True
        return False

    def dist(self, p1, p2):
        (x1, y1), (x2, y2) = p1, p2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def _reset_stats(self) -> None:
        self._episode_start_time = time.time()
        self._elapsed_steps = 0
        self._episode_over = False
    
    def globalmap_toagentcord1(self,x):
        x2 = x-24
        x3 = ((np.dot((x2),np.array(self.matr_pov))+self.start_pos)*(-1))[::-1]
        return x3

    def reset(self, i=None, goal_reacher=True) -> Observations:
        r"""Resets the environments and returns the initial observations.
        :return: initial observations from the environment.
        """
        
        if i is None:
            nn = np.random.choice(self.eps)
        else:
            nn = self.eps[i]
        
        self.steps = 0
        self.closest_goal_coordinates = None
        self.closest_goal_on_map_coordinates = None
        self.prev_goal_distance = None
        self.valid_goal = False
        self._reset_stats()
        self.cord_goal = None

        assert len(self.episodes) > 0, "Episodes list is empty"
        # Delete the shortest path cache of the current episode
        # Caching it for the next time we see this episode isn't really worth it
        if self._current_episode is not None:
            self._current_episode._shortest_path_cache = None
          
        self._current_episode = self._episode_iterator.episodes[nn]
        self.reconfigure(self._config)

        observations = self.task.reset(episode=self._current_episode)
        self._task.measurements.reset_measures(
            episode=self._current_episode, task=self.task, observations=observations,
        )
        #self.obs = observations

        self.info = self.get_metrics()
        #self.info['sensor_pose'] = [0., 0., 0.]
        
        
        data = pd.read_csv('hm3dsem_category_mappings.tsv', encoding='utf-8', sep="\t",engine='python')
        data = data.set_index('raw_category')
        self.category_mapping = data['mpcat40']
        self.index_to_title_map = {obj.category.index(): self.category_mapping.get(obj.category.name(), 'unknown') for obj in self.sim.semantic_annotations().objects}
        self.instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in self.sim.semantic_annotations().objects}


        self.objectgoal_name = {v: k for k, v in self.task._dataset.category_to_task_category_id.items()}[observations['objectgoal'][0]]
        self.objectgoal = [i for i in self.index_to_title_map if self.index_to_title_map[i] == self.objectgoal_name]
        #self.sem_to_model = np.vectorize(self.instance_id_to_label_id.get)(observations['semantic']) 
        #self.sem_to_model_goal = np.zeros_like(self.sem_to_model)
        #for i in self.objectgoal:
        #    self.sem_to_model_goal += (self.sem_to_model)==i
        #self.sem_to_model_goal = np.float32(self.sem_to_model_goal)[:,:,np.newaxis]
        

        self.start_pos = self.get_sim_location()[:2]
        self.start_angle = np.rad2deg(self.get_sim_location()[2])
        alpha = -self.start_angle
        self.matr_pov = [[np.cos(np.deg2rad(alpha)),-np.sin(np.deg2rad(alpha))],[np.sin(np.deg2rad(alpha)),np.cos(np.deg2rad(alpha))]]
        
        self.agent_pose = self._sim.get_agent_state(0).position
        
        

        ds = 2
        return {'depth':observations['depth'],#[ds // 2::ds, ds // 2::ds], 
                'rgb':observations['rgb'], 
                #'semantic':self.sem_to_model_goal, 
                #'heading':observations['heading'],
                'objectgoal':observations['objectgoal'],
                'gps':observations['gps'],
                'compass':observations['compass'],
                }#observations
  
        

    def _update_step_stats(self) -> None:
        self._elapsed_steps += 1
        self._episode_over = not self._task.is_episode_active
        if self._past_limit():
            self._episode_over = True

        if self.episode_iterator is not None and isinstance(
            self.episode_iterator, EpisodeIterator
        ):
            self.episode_iterator.step_taken()

    def step(
        self, action: Union[int, str, Dict[str, Any]],in_reset=False, **kwargs
    ) -> Observations:
        r"""Perform an action in the environment and return observations.
        :param action: action (belonging to :ref:`action_space`) to be
            performed inside the environment. Action is a name or index of
            allowed task's action and action arguments (belonging to action's
            :ref:`action_space`) to support parametrized and continuous
            actions.
        :return: observations after taking action in environment.
        """
        assert (self._episode_start_time is not None), "Cannot call step before calling reset"
        assert (self._episode_over is False), "Episode over, call reset before calling step"

        if isinstance(action, (str, int, np.integer)):
            action = {"action": action}
            
        self._previous_action = action['action']   

        observations = self.task.step(action=action, episode=self._current_episode)
        self.steps+=1
        #self.obs = observations
        self._task.measurements.update_measures(episode=self._current_episode, action=action, task=self.task,observations=observations)
        self._update_step_stats()
        self.info = self.get_metrics()

        #self.sem_to_model = np.vectorize(self.instance_id_to_label_id.get)(observations['semantic'])
        #self.sem_to_model_goal = np.zeros_like(self.sem_to_model)
        #for i in self.objectgoal:
        #    self.sem_to_model_goal += (self.sem_to_model)==i
        #self.sem_to_model_goal = np.float32(self.sem_to_model_goal)[:,:,np.newaxis]  
        
        self.agent_pose = self._sim.get_agent_state(0).position
            
        reward = 0.
        
        return {'depth':observations['depth'], 
                    'rgb':observations['rgb'], 
                    #'semantic':self.sem_to_model_goal, 
                    #'heading':observations['heading'],
                    'objectgoal':observations['objectgoal'],
                    'gps':observations['gps'],
                    'compass':observations['compass'],
                    #'gps':observations['gps']
                   }, reward, self._episode_over, self.info

 

    def something(self, point):
        x, y = point
        r, c = x * 100 / self.args.map_resolution, \
                   y * 100 / self.args.map_resolution
        r, c = int(r), int(c)
        
        return np.array([r, c])    
        
    def get_projection(self, pos):
        pos = np.array(pos)
        pos = self.to_agent_cord(pos)+np.array([24])
        return self.cord_to_global_map(pos)     
    
    def cord_to_global_map(self,x):
        return (np.array(list(x))*100.0/self.map_size_cm*(self.map_size_cm//self.args.map_resolution)).astype(int)
        
    def to_agent_cord(self,a):
        return np.dot((a[::-1]*(-1)-self.start_pos),np.array(self.matr_pov).T)
        
    def get_sim_location(self):
        agent_state = self.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2*np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_gt_pose_change(self):
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do

    @staticmethod
    @numba.njit
    def _seed_numba(seed: int):
        random.seed(seed)
        np.random.seed(seed)

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self._seed_numba(seed)
        self._sim.seed(seed)
        self._task.seed(seed)

    def reconfigure(self, config: "DictConfig") -> None:
        self._config = self._task.overwrite_sim_config(config, self.current_episode)

        self._sim.reconfigure(self._config.simulator, self.current_episode)

    def render(self, mode="rgb") -> np.ndarray:
        return self._sim.render(mode)

    def close(self) -> None:
        self._sim.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()