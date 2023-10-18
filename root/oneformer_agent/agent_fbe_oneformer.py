import sys
sys.path.append('/root/exploration_ros_free/habitat_map')
sys.path.append('/root/oneformer_agent')
sys.path.append('/root')

import argparse
import os
import random
from collections import OrderedDict
import imageio

import numba
import numpy as np
import PIL
import torch
from gym.spaces import Box, Dict, Discrete

import habitat
from habitat.core.agent import Agent
from habitat import make_dataset
from exploration_ros_free.semexp.arguments import get_args as get_args_env
from habitat.utils.visualizations import maps
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from torchvision import transforms

import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
from torch.cuda.amp import autocast
from scipy import ndimage

import torch.distributed as dist
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import glob
import time
import matplotlib.patches as patches

import argparse

from EXPLORE_policy import ResNetPolicy as EX_Policy
from GR_policy import PointNavResNetPolicy as GR_Policy

from hab_base_utils_common import batch_obs, batch_obs1

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

#from mmseg.apis import multi_gpu_test, single_gpu_test, init_segmentor, inference_segmentor, show_result_pyplot
#from mmseg.core.evaluation import get_palette
#from mmseg.datasets import build_dataloader, build_dataset
#from mmseg.models import build_segmentor

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from train_net import Trainer
from detectron2.projects.deeplab import add_deeplab_config
from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from exploration.frontier.frontier_exploration import FrontierExploration
from planners.astar.astar_planner import AStarPlanner
from planners.theta_star.theta_star_planner import ThetaStarPlanner
from mapper import Mapper
import yaml

import time
from PIL import Image


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__  
    
def normalize(angle):
    while angle < -np.pi:
        angle += 2 * np.pi
    while angle > np.pi:
        angle -= 2 * np.pi
    return angle    

meta_palete = [(106, 247, 32),
                  (52, 193, 234),
                  (76, 217, 2),
                  (208, 93, 134),
                  (232, 117, 158),
                  (178, 63, 104),
                  (0, 141, 182),
                  (238, 123, 164),
                  (24, 165, 206),
                  (226, 111, 152),
                  (6, 147, 188),
                  (112, 253, 38),
                  (250, 135, 176),
                  (196, 81, 122),
                  (136, 21, 62),
                  (82, 223, 8),
                  (220, 105, 146),
                  (160, 45, 86),
                  (244, 129, 170),
                  (142, 27, 68),
                  (184, 69, 110),
                  (130, 15, 56),
                  (12, 153, 194),
                  (166, 51, 92),
                  (154, 39, 80),
                  (100, 241, 26),
                  (190, 75, 116),
                  (40, 181, 222),
                  (124, 9, 50),
                  (214, 99, 140),
                  (64, 205, 246),
                  (202, 87, 128),
                  (148, 33, 74),
                  (46, 187, 228),
                  (88, 229, 14),
                  (172, 57, 98),
                  (70, 211, 252),
                  (58, 199, 240),
                  (94, 235, 20),
                  (18, 159, 200),
                  (118, 3, 44)]


class Agent_hlpo(Agent):
    def __init__(self, task_config):
        
        self.p240 = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((240,320)),
                                transforms.ToTensor()])
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obs_space2 = dotdict()
        obs_space2.spaces = {}
        obs_space2.spaces['rgb'] = Box(low=-1000, high=1000, shape=(240,320,3), dtype=np.float32)
        obs_space2.spaces['depth'] = Box(low=-1000, high=1000, shape=(240,320,1), dtype=np.float32)
        obs_space2.spaces['semantic'] = Box(low=-1000, high=1000, shape=(240,320,1), dtype=np.float32)
        #act_space2 = dotdict()
        #act_space2.n = 4
        #act_space2.shape = [1]
        act_space2 = Discrete(5)
        self.actor_critic_gr = GR_Policy(
            observation_space = obs_space2,
            action_space = act_space2,
            hidden_size = 512,
            rnn_type = 'GRU',
            num_recurrent_layers = 1,
            backbone = 'resnet18',
            normalize_visual_inputs=True)
        
        pretrained_state = torch.load('/home/AI/yudin.da/zemskova_ts/skill-fusion/root/exploration_ros_free/weights/goalreacher_model_4.pth', map_location="cpu")
        #pretrained_state = torch.load('/root/exploration_ros_free/weights/goalreacher_may21_65.pth', map_location="cpu")
        #pretrained_state = torch.load('/root/weights/8901_gr_OP_train_aug29.pth', map_location="cpu")
        #pretrained_state = torch.load('/root/weights/8902_gr_CLIP_train_aug29.pth', map_location="cpu")
        #pretrained_state = torch.load('/root/weights/8903_ACTION5_CLIP_aug30.pth', map_location="cpu")
        #pretrained_state = torch.load('/root/weights/8906_CLIP_RGB_train_aug30.pth', map_location="cpu")
        #pretrained_state = {k[len("actor_critic.") :]: v for k, v in pretrained_state["state_dict"].items()}          
        self.actor_critic_gr.load_state_dict(pretrained_state)
        self.actor_critic_gr.to(self.device)
        self.actor_critic_gr.eval()
        
        obs_space1 = dotdict()
        obs_space1.spaces = {}
        obs_space1.spaces['rgb'] = Box(low=0, high=0, shape=(3,320,240), dtype=np.uint8)
        obs_space1.spaces['depth'] = Box(low=0, high=0, shape=(1,320,240), dtype=np.uint8)
        obs_space1.spaces['pointgoal_with_gps_compass'] = Box(low=0, high=0, shape=(2,), dtype=np.uint8)
        obs_space1.spaces['task_id'] = Box(low=0, high=1, shape=(1,), dtype=np.uint8)
        act_space = dotdict()
        act_space.n = 4
        act_space.shape = [1]  
        self.actor_critic_3fusion = EX_Policy(
                    observation_space=obs_space1,
                    action_space=act_space,
                    hidden_size=512,
                    resnet_baseplanes=64,
                    rnn_type='LSTM',
                    num_recurrent_layers=1,
                    backbone='resnet18',
                )
        pretrained_state = torch.load('/root/exploration_ros_free/weights/explore_model_2.pth', map_location="cpu")
        #pretrained_state = {k[len("actor_critic.") :]: v for k, v in pretrained_state["state_dict"].items()} 
        self.actor_critic_3fusion.load_state_dict(pretrained_state)
        self.actor_critic_3fusion.to(self.device)
        self.actor_critic_3fusion.eval()
        
        self.objgoal_to_cat = {0: 'chair',     1: 'bed',     2: 'plant',           3: 'toilet',           4: 'tv_monitor',   
                               5: 'sofa'}
        
        self.crossover = {'chair':['chair', 'armchair', 'swivel chair'], 'bed':['bed'], 'plant':['tree','plant','flower'], 'toilet':['toilet, can, commode, crapper, pot, potty, stool, throne'], 'tv_monitor':['computer','monitor','tv','crt screen','screen', 'tv_monitor'], 'sofa':['sofa']}   
 
        self.object_extention = {'chair':['cabinet'],
                                 'bed':['chest of drawers','cushion'],
                                 'plant':['curtain','cabinet'],
                                 'toilet':['bathtub','sink','mirror'],
                                 'tv_monitor':['bed'],
                                 'sofa':['television receiver','crt screen','screen'],
                                }
        
        config_file = '/root/exploration_ros_free/weights/config_oneformer_ade20k.yaml'
        checkpoint_file = '/root/exploration_ros_free/weights/model_0299999.pth'
        #self.model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
        # Build a model
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_common_config(cfg)
        add_swin_config(cfg)
        add_dinat_config(cfg)
        add_convnext_config(cfg)
        add_oneformer_config(cfg)
        cfg.merge_from_file(config_file)

        self.model = Trainer.build_model(cfg)

        DetectionCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(checkpoint_file, resume=False)
        
        self.model.eval() 
        self.clss = self.model.metadata.stuff_classes
        #self.clss = ['appliances', 'bathtub', 'beam', 'bed', 'blinds', 'board_panel', 'cabinet', 'ceiling', 'chair', 'chest_of_drawers', 'clothes', 'column', 'counter', 'curtain', 'cushion', 'door', 'fireplace', 'floor', 'furniture', 'gym_equipment', 'lighting', 'mirror', 'misc', 'objects', 'picture', 'plant', 'railing', 'seating', 'shelving', 'shower', 'sink', 'sofa', 'stairs', 'stool', 'table', 'toilet', 'towel', 'tv_monitor', 'wall', 'window', 'unlabeled']
        self.confidence_threshold = 0.3 # threshold value for semantic masks
        """
        self.clss = {
            'floor': 3,
            'wall': -1,
            'ceiling': 5,
            'window' : 8,
            'door': 14,
            'picture': 22,
            'curtain': 18,
            'table': 15,
            'cushion': 39,
            'stairs': 53,
            'cabinet': 10,
            'counter': 45,
            'shower': 145,
            'chest_of_drawers': 44,
            'lighting' : 82,
            
            'chair': 19,
            'bed': 7,
            'plant': 17,
            'toilet': 65,
            'tv_monitor': 143,
            'sofa': 23,
        }
        """
        
        #checkpoint_file = '/root/weights/segformer.b4.512x512.ade.160k.pth'
        #config_file = '/SegFormer/local_configs/segformer/B5/segformer.b5.640x640.ade.160k.py'
        #checkpoint_file = '/root/weights/SegFormerB5_216000.pth'
        #self.model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
        
        
        # FBE PART
        fin = open('/root/exploration_ros_free/config.yaml', 'r')
        config = yaml.safe_load(fin)
        fin.close()

        # Initialize visualization
        visualization_config = config['visualization']
        self.visualize = visualization_config['visualize']
        
        follower_config = config['path_follower']
        self.goal_radius = follower_config['goal_radius']
        self.max_d_angle = follower_config['max_d_angle']
        self.finish_radius = config['task']['finish_radius']

        # Initialize mapper
        mapper_config = config['mapper']
        self.mapper = Mapper(obstacle_inflation=mapper_config['obstacle_inflation'],
                             semantic_inflation=mapper_config['semantic_inflation'],
                             vision_range=mapper_config['vision_range'],
                             semantic_vision_range=mapper_config['semantic_vision_range'],
                             map_size_cm= mapper_config['map_size_cm'], ################################
                             map_resolution=mapper_config['map_resolution_cm'], 
                             semantic_threshold=mapper_config['semantic_threshold'],
                             semantic_decay=mapper_config['semantic_decay'])

        # Initialize semantic predictor
        semantic_config = config['semantic']

        # Initialize exploration
        exploration_config = config['exploration']
        if exploration_config['type'] == 'frontier':
            self.exploration = FrontierExploration(self.mapper,
                                           potential_scale=exploration_config['potential_scale'],
                                           orientation_scale=exploration_config['orientation_scale'],
                                           gain_scale=exploration_config['gain_scale'],
                                           min_frontier_size=exploration_config['min_frontier_size'])
        else:
            print('UNKNOWN EXPLORATION TYPE!!!')
            self.exploration = None
        self.timeout = exploration_config['timeout']

        # Initialize path planner
        planner_config = config['path_planner']
        self.planner_frequency = planner_config['frequency']
        planner_type = planner_config['type']
        if planner_type == 'theta_star':
            self.path_planner = ThetaStarPlanner(self.mapper.mapper,
                                                 agent_radius=planner_config['agent_radius'],
                                                 reach_radius=planner_config['reach_radius'],
                                                 allow_diagonal=planner_config['allow_diagonal'])
        elif planner_type == 'a_star':
            self.path_planner = AStarPlanner(self.mapper,
                                             reach_radius=planner_config['reach_radius'],
                                             allow_diagonal=planner_config['allow_diagonal'])
        else:
            print('PATH PLANNER TYPE {} IS NOT DEFINED!!!'.format(planner_type))
            self.path_planner = None

        self.step = 0
        self.stuck = False
        self.steps_after_stuck = 1000
        self.robot_pose_track = []
        self.action_track = []
        self.goal = None
        self.path = None
        self.objgoal_found = False
        self.step_to_goal = 0
        
        self.robot_pose_track = []
        self.goal_coords = []
        self.maps = []
        self.rgbs = []
        self.depths = []
        self.action_track = []
        self.obs_maps = []
        self.agent_positions = []
        self.goal_coords_ij = []
        self.paths = []
        
        
    def goal_reached(self, observations):
        x, y = observations['gps']
        y_cell, x_cell = self.mapper.world_to_map(x, -y)
        semantic_map = self.mapper.semantic_map
        i, j = (semantic_map > 0).nonzero()
        #print('I, J:', y_cell, x_cell)
        if len(i) > 0:
            dst = np.sqrt((y_cell - i) ** 2 + (x_cell - j) ** 2)
            print('Min distance:', dst.min())
            if dst.min() <= self.finish_radius / self.mapper.args.map_resolution * 100:
                return True
        return False


    def interm_goal_reached(self, observations, goal):
        x, y = observations['gps']
        y *= -1
        #print('Robot position:', x, y)
        goal_x, goal_y = goal
        #print('Goal position:', goal_x, goal_y)
        dst = np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)
        #print('Dst:', dst)
        return (dst < self.goal_radius + self.path_planner.reach_radius)
    
    def choose_goal_fbe(self, observations):
        robot_x, robot_y = observations['gps']
        robot_y = -robot_y
        # If we found objectgoal, cancel current intermediate goal to move to the objectgoal immediately
        if not self.objgoal_found and self.mapper.semantic_map.max() > 0:
            self.objgoal_found = True
            self.exploration.reject_goal()
            self.goal = None
            self.step_to_goal = 0
        if self.objgoal_found and self.mapper.semantic_map.max() == 0:
            self.objgoal_found = False
            self.exploration.reject_goal()
        if self.objgoal_found:
            semantic_map = self.mapper.semantic_map
            i, j = (semantic_map > 0).nonzero()
            if len(i) > 0:
                y_cell, x_cell = self.mapper.world_to_map(robot_x, robot_y)
                dst = np.sqrt((y_cell - i) ** 2 + (x_cell - j) ** 2)
                min_id = dst.argmin()
                self.goal = self.mapper.map_to_world(i[min_id], j[min_id])
                return
        # If we have no goal, let's find one
        if self.goal is None:
            self.step_to_goal = 0
            self.goal = self.exploration.choose_goal(observations)
   
        # If goal search failed, clear additional obstacles and try again

        if self.goal is None:
            #print('Goal not found. Clear obstacles')
            self.mapper.clear_obstacles()
            self.goal = self.exploration.choose_goal(observations)
        if not self.objgoal_found and self.goal is None:
            self.mapper.reset()
            self.step = 0


    def create_path_by_planner(self, observations):
        robot_x, robot_y = observations['gps']
        robot_y = -robot_y
        # Create path to goal. If path planning failed, create path along both free and unknown map cells
        if self.goal is not None and (self.path is None or self.step % self.planner_frequency == 0 or self.stuck):
            self.path = self.path_planner.create_path(observations, self.goal, unknown_is_obstacle=True)
            old_reach_radius = self.path_planner.reach_radius
            if self.objgoal_found and self.path is None:
                print('Path to goal with radius 0.5 not found. Try expand reach radius to 0.7')
                self.path_planner.reach_radius = 0.7
                self.path = self.path_planner.create_path(observations, self.goal, unknown_is_obstacle=True)
            if self.path is None:
                #print('Path by free cells not found. Trying with unknown cells')
                self.path = self.path_planner.create_path(observations, self.goal, unknown_is_obstacle=False)
            self.path_planner.reach_radius = old_reach_radius
            # If path to goal not found, reject this goal
            if self.path is None:
                #print('Path not found. Rejecting goal')
                self.exploration.reject_goal()
                self.goal = None
                self.step_to_goal = 0
                if self.objgoal_found:
                    self.mapper.reset()
                    self.step = 0       

        self.step_to_goal += 1
        if self.goal is not None and self.interm_goal_reached(observations, self.goal):
            self.exploration.accept_goal()
            self.goal = None
            self.step_to_goal = 0
        if self.step_to_goal >= self.timeout:
            self.exploration.reject_goal()
            self.goal = None
            self.step_to_goal = 0


    def act_fbe(self, observations):
        robot_x, robot_y = observations['gps']
        robot_y = -robot_y
        robot_angle = observations['compass'][0]
        #print('Robot x, y, angle:', robot_x, robot_y, robot_angle)
        #print('Goal x, y:', self.goal)

        # at start, let's rotate 360 degrees to look around
        if self.step < 12:
            return HabitatSimActions.turn_left

        # If there is no valid path, return random action
        if self.path is None or len(self.path) < 2:
            action = np.random.choice([HabitatSimActions.turn_left, HabitatSimActions.move_forward])
            return action

        # Find the closest point of the path to the robot
        nearest_id = 0
        best_distance = np.inf
        for i in range(len(self.path)):
            x, y = self.path[i]
            dst = np.sqrt((robot_x - x) ** 2 + (robot_y - y) ** 2)
            if dst < best_distance:
                best_distance = dst
                nearest_id = i

        # If nearest point is too close, and we are not stuck, take the next point of the path
        x, y = self.path[min(nearest_id + 1, len(self.path) - 1)]
        x_prev, y_prev = self.path[min(nearest_id, len(self.path) - 1)]
        segment = np.sqrt((x_prev - x) ** 2 + (y_prev - y) ** 2)
        dst = np.sqrt((robot_x - x) ** 2 + (robot_y - y) ** 2)
        if (segment < 0.2 or dst < 0.2) and self.steps_after_stuck > 2 and nearest_id + 2 < len(self.path):
            #print('TAKE NEXT POINT OF PATH')
            nearest_id += 1
            x, y = self.path[min(nearest_id + 1, len(self.path) - 1)]

        # Compute distance and angle to next point of path
        dst = np.sqrt((robot_x - x) ** 2 + (robot_y - y) ** 2)
        angle_to_goal = np.arctan2(y - robot_y, x - robot_x)
        turn_angle = normalize(angle_to_goal - robot_angle)

        # if next path point is too close to us, move forward to avoid "swing"
        if dst < 0.3 and abs(turn_angle) < np.pi / 2 and not self.stuck:
            return HabitatSimActions.move_forward

        # if our direction is close to direction to goal, move forward
        if abs(turn_angle) < self.max_d_angle and not self.stuck:
            return HabitatSimActions.move_forward

        if self.steps_after_stuck == 1:
            return HabitatSimActions.move_forward

        if self.stuck and np.random.random() < 0.3:
            action = np.random.choice([HabitatSimActions.turn_right, HabitatSimActions.turn_left])
            return action

        # if direction isn't close, turn left or right
        if turn_angle < 0:
            return HabitatSimActions.turn_right
        return HabitatSimActions.turn_left
    
    
    def reset(self):
        self.test_recurrent_hidden_states_gr = torch.zeros(1, self.actor_critic_gr.net.num_recurrent_layers, 512, device=self.device)
        self.not_done_masks = torch.zeros(1, 1, device=self.device)
        self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        
        self.recurrent_hidden_states = torch.zeros(1, 2, 512, device=self.device)
        self.not_done_masks0 = torch.zeros(1, 1, device=self.device)
        self.prev_actions0 = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        
        self.skil_goalreacher = False
        self.skil_goalreacher_rl = False
        self.subgoal_rl = False
        self.dstance_to_object = 10.
        
        self.object_extention = {'chair':[],
                                 'bed':[],
                                 'plant': [],#['chest of drawers','cabinet'],
                                 'toilet':['bathtub','sink','mirror'],
                                 'tv_monitor':[],#['sofa','cushion'],
                                 'sofa':[],#['cushion'],
                                }
        
        self.subgoal_step = 0
        self.stuck_again = 0
        self.forward_distances = []
        self.rl_stop_distances = [100.,100.]
        self.rl_stuck = [False,False]
        self.explore_areas = [100 for i in range(20)]
        self.area_threshold = 40.
        self.current_explorearea = 0
        self.past_explorearea = 0
        
        self.found_semantic = [False]
        self.found_semantic_subgoal = [False]
        
        self.zero_rls = [False, False]
        self.lastACTION5 = [False, False]
        
        
        # FBE PART
        self.step_switcher = 0
        self.step = 0
        self.robot_pose_track = []
        self.action_track = []
        self.exploration.reset()
        self.path_planner.reset()
        self.mapper.reset()
        self.goal = None
        self.path = None
        self.objgoal_found = False
        self.step_to_goal = 0
        
        self.critic_values = [0.]
        self.critic_gr_values_start = [0.]
        self.map_expand_values = [0.]
        
        self.robot_pose_track = []
        self.goal_coords = []
        self.maps = []
        self.rgbs = []
        self.depths = []
        self.action_track = []
        self.obs_maps = []
        self.agent_positions = []
        self.goal_coords_ij = []
        self.paths = []
        
        
    def act(self, obs, depth = True, semantic = False):
        dpth = obs['depth']
        
        robot_x, robot_y = obs['gps']
        robot_y = -robot_y
        print('Robot x, y:', robot_x, robot_y)

        after_crossover = self.crossover[self.objgoal_to_cat[obs['objectgoal'][0]]]
        self.subgoal_names = self.object_extention[self.objgoal_to_cat[obs['objectgoal'][0]]]
        self.segformer_index = [i for i,x in enumerate(self.clss) if x in self.subgoal_names]
        #print(self.subgoal_names)
        #print(self.segformer_index)
        if semantic:
            obs_semantic = np.ones((640,480,1))
            sem = obs['semantic'][:,:,0]
            segformer_index = [i for i,x in enumerate(self.clss) if x in after_crossover]
            #segformer_index = [self.clss[x]+1 for i,x in enumerate(self.clss) if x in after_crossover]
            mask = np.zeros((640,480))
            for seg_index in segformer_index:
                #print("Seg index", seg_index)
                mask += (sem == seg_index).astype(float)
                #mask += (result == seg_index).astype(float)
            mask.astype(bool).astype(float)
            obs_semantic *= cv2.erode(mask,np.ones((3,3),np.uint8),iterations = 1)[:,:,np.newaxis]
            #print("Unique obs, semantic", np.unique(sem))
            #sem = [sem.astype(bool).astype(float)
            #print(sem.shape)
            #obs_semantic = cv2.erode(sem,np.ones((3,3),np.uint8),iterations = 1)[:,:,np.newaxis]
            masks = np.zeros((len(self.segformer_index),640,480))
            for ii,seg_index in enumerate(self.segformer_index):
                #print("Seg index subgoals", seg_index)
                masks[ii] *= (sem == seg_index).astype(bool).astype(float)
                #masks[ii] *= (result == seg_index).astype(bool).astype(float)
                masks[ii] = cv2.erode(masks[ii],np.ones((3,3),np.uint8),iterations = 1)#[:,:,np.newaxis]  
        else:
            
            obs_semantic = np.ones((640,480,1))
            masks = np.zeros((len(self.segformer_index),640,480))
            for _ in range(1):
                item = [{
                    "image": torch.from_numpy(np.moveaxis(obs['rgb'], -1, 0)),
                    "height": 640,
                    "width": 480,
                    "task": "semantic"
                }]
                with torch.no_grad():
                    result = self.model(item)
                    result = result[0]["sem_seg"].cpu().numpy()
                    result = np.argmax(result, axis=0)
                    #result = (result - np.min(result)) / (np.max(result) - np.min(result))

                ################################## FOR SEGFORMER B5  second lane
                segformer_index = [i for i,x in enumerate(self.clss) if x in after_crossover]
                #segformer_index = [self.clss[x]+1 for i,x in enumerate(self.clss) if x in after_crossover]
                mask = np.zeros((640,480))
                for seg_index in segformer_index:
                    print("Seg index", seg_index)
                    #mask += (result[seg_index] > self.confidence_threshold).astype(float)
                    mask += (result == seg_index).astype(float)
                    print("Max confidence:", np.max(result))
                mask.astype(bool).astype(float)
                obs_semantic *= cv2.erode(mask,np.ones((3,3),np.uint8),iterations = 1)[:,:,np.newaxis]
                

                # Маски подцелей
                for ii,seg_index in enumerate(self.segformer_index):
                    print("Seg index subgoals", seg_index)
                    #masks[ii] *= (result[seg_index] > self.confidence_threshold).astype(bool).astype(float)
                    masks[ii] *= (result == seg_index).astype(bool).astype(float)
                    masks[ii] = cv2.erode(masks[ii],np.ones((3,3),np.uint8),iterations = 1)#[:,:,np.newaxis]
                    
        self.robot_pose_track.append((robot_x, robot_y))
        self.maps.append(self.mapper.mapper.map[:, :, 1])
        rgb_with_semantic = obs['rgb'].copy()
        rgb_with_semantic[obs_semantic[:, :, 0] > 0] = [255, 0, 0]
        self.rgbs.append(rgb_with_semantic)
        self.depths.append(obs['depth'])
          
        if np.sum(obs_semantic)>300.:
            self.found_semantic.append(True)
        else:
            self.found_semantic.append(False) 
            #obs_semantic*=0   
        ######################################################## 2nd_start     
        #if sum(self.found_semantic[-3:])<2:
        #    obs_semantic*=0  
        ########################################################    
            
            
        #robot_x, robot_y = obs['gps']
        #self.robot_pose_track.append((robot_x, robot_y))
        self.step_switcher+=1
        self.step += 1
        if obs['objectgoal'][0] == 4:
            self.mapper.erosion_size = 2
        else:
            self.mapper.erosion_size = 3
        
        
        ######################################################## SEM_MAP100
        """
        if self.mapper.semantic_map is None:
            self.mapper.step(obs, obs_semantic[:,:,0])
        elif self.mapper.semantic_map.sum()>100:
            self.mapper.step(obs, obs_semantic[:,:,0]*0)  
        else:
        """
        self.mapper.step(obs, obs_semantic[:,:,0])  
        ######################################################## 
            
            
        self.current_explorearea = np.sum(self.mapper.explored_map)
        self.explore_areas.append(max(self.current_explorearea-self.past_explorearea,0.))
        self.past_explorearea = self.current_explorearea
        

        #if sum(self.zero_rls[-7:])==0:  ############################### ZERO_RLS
        if np.sum(obs_semantic)>300.:
            self.skil_goalreacher_rl = True
            self.subgoal_rl = False
            print('FOUND GOAL CLASS')
        elif self.mapper.semantic_map.max()==0 and not self.subgoal_rl and not self.skil_goalreacher_rl:
            for ii,msk in enumerate(masks):
                if np.sum(msk)>300.:
                    self.found_semantic_subgoal.append(True)
                    self.index_pop_subgoal = ii
                    print('FOUND SUBGOAL ',self.segformer_index[self.index_pop_subgoal],[x for i,x in enumerate(self.clss) if x in self.subgoal_names][self.index_pop_subgoal])
                    self.skil_goalreacher_rl = True
                    self.subgoal_rl = True
                    break
               
                    
        ########################################################## DISABLE  
        #"""
        if self.skil_goalreacher_rl and sum(self.found_semantic[-7:])<1:
            self.skil_goalreacher_rl = False
            self.test_recurrent_hidden_states_gr = torch.zeros(1, self.actor_critic_gr.net.num_recurrent_layers, 512, device=self.device)
            self.not_done_masks0 = torch.zeros(1, 1, device=self.device)
            self.prev_actions0 = torch.zeros(1, 1, dtype=torch.long, device=self.device)
            print('DISABLE GOALREACHER')       
        #"""    
        ##########################################################
                    
        if self.subgoal_rl and np.sum(masks[self.index_pop_subgoal])>300.:
            print('existing subgoal: ',self.segformer_index[self.index_pop_subgoal],[x for i,x in enumerate(self.clss) if x in self.subgoal_names][self.index_pop_subgoal])
            msk = masks[self.index_pop_subgoal]
            obs_semantic = msk[:,:,np.newaxis] 

        if self.mapper.semantic_map.max() > 0:
            self.skil_goalreacher = True
        else:
            self.skil_goalreacher = False
            
        # Detect stuck
        self.stuck = False
        if len(self.action_track) > 0:
            print('Last action:', self.action_track[-1])
        if len(self.action_track) > 0 and \
           int(self.action_track[-1][0]) == HabitatSimActions.move_forward and \
           len(self.robot_pose_track) > 1 \
           and np.sqrt((self.robot_pose_track[-1][0] - self.robot_pose_track[-2][0]) ** 2 + \
                       (self.robot_pose_track[-1][1] - self.robot_pose_track[-2][1]) ** 2) < 0.01:
            self.stuck = True
        self.steps_after_stuck += 1
        if self.stuck:
            self.steps_after_stuck = 0
            print('WE stuck')
            self.mapper.draw_obstacle_ahead(obs)
        self.rl_stuck.append(self.stuck)    
        
        #if self.stuck:
        if self.skil_goalreacher_rl and np.sum(self.rl_stuck[-5:])>=4:
            self.skil_goalreacher_rl = False
            print('RL disable STUCK')
            self.rl_stuck = [False,False]
            if self.subgoal_rl:
                print('POP stuck',self.index_pop_subgoal,[x for i,x in enumerate(self.clss) if x in self.subgoal_names][self.index_pop_subgoal])
                self.object_extention[self.objgoal_to_cat[obs['objectgoal'][0]]].remove([x for i,x in enumerate(self.clss) if x in self.subgoal_names][self.index_pop_subgoal])
                self.subgoal_rl = False
                
        if self.goal is not None:
            self.goal_coords.append(self.goal)
        else:
            self.goal_coords.append((-1000, -1000))
 
        #if self.stuck and self.skil_goalreacher_rl: #############################
        #    self.skil_goalreacher_rl = False    

        # If we reached objectgoal, stop

        if self.goal_reached(obs):
            print('GOAL REACHED. FINISH EPISODE!')
            self.action_track.append(('0', 'FINISH'))
            return {'action': HabitatSimActions.stop}   
        
        
        rgb_trans = (self.p240(obs['rgb']).permute(1,2,0)*255)
        depth_trans = self.p240(dpth).permute(1,2,0)
        sem_trans = self.p240(torch.tensor(obs_semantic)[:,:,0]).permute(1,2,0).int().float()
        
        ###########################################################  TO UPDATE VALUES
        batch = batch_obs1([{'rgb':rgb_trans,
                            'depth':depth_trans,
                            'semantic':sem_trans}], device=self.device)
        with torch.no_grad():
            gr_values, action, _, self.no_update_rnn = self.actor_critic_gr.act(
                batch,
                self.test_recurrent_hidden_states_gr,
                self.prev_actions0,
                self.not_done_masks0.byte(),
                deterministic=False)
            action = action.item()
        self.choose_goal_fbe(obs)  
        self.create_path_by_planner(obs)
        ########################################################
                    
        # GOALREACHER RL
        if (self.skil_goalreacher_rl) and (not self.skil_goalreacher):
            print('GoalReacher RL')
            batch = batch_obs1([{'rgb':rgb_trans,
                                'depth':depth_trans,
                                'semantic':sem_trans}], device=self.device)
            with torch.no_grad():
                gr_values, action, _, self.test_recurrent_hidden_states_gr = self.actor_critic_gr.act(
                    batch,
                    self.test_recurrent_hidden_states_gr,
                    self.prev_actions0,
                    self.not_done_masks0.byte(),
                    deterministic=False)
                self.not_done_masks0.fill_(1.0)
                self.prev_actions0.copy_(action) 
                action = action.item() 
                if action==4:
                    action=0
                    print('ACTION5 IS DONE:')
                    self.lastACTION5.append(True)
                else:
                    self.lastACTION5.append(False)
                
            if not self.subgoal_rl:    
                mask_depth = (depth_trans*sem_trans).cpu().numpy()
                mask_depth[mask_depth==0] = 100.
                mmin, mmax, xymin,xymax = cv2.minMaxLoc(mask_depth)
                self.rl_stop_distances.append(mmin*4.5+0.5)
                self.dstance_to_object = min(self.dstance_to_object,mmin*4.5+0.5)   
                print('DIST TO GLOBAL GOAL: ',self.dstance_to_object,(mmin*4.5+0.5))
            
            if action==0:
                if (self.lastACTION5[-1]==True or np.min(self.rl_stop_distances[-7:])>0.8) and not self.subgoal_rl: # or np.min(self.rl_stop_distances[-7:])>0.8
                    print('RL05_skip ',np.min(self.rl_stop_distances[-6:]))
                    self.skil_goalreacher_rl = False                      ######################
                    self.zero_rls.append(True)
                    self.test_recurrent_hidden_states_gr = torch.zeros(1, self.actor_critic_gr.net.num_recurrent_layers, 512, device=self.device)
                    self.not_done_masks0 = torch.zeros(1, 1, device=self.device)
                    self.prev_actions0 = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                    
                if self.subgoal_rl:
                    self.subgoal_step = 0
                    self.skil_goalreacher_rl = False
                    print('POP ACT0 ',self.index_pop_subgoal,[x for i,x in enumerate(self.clss) if x in self.subgoal_names][self.index_pop_subgoal])
                    self.object_extention[self.objgoal_to_cat[obs['objectgoal'][0]]].remove([x for i,x in enumerate(self.clss) if x in self.subgoal_names][self.index_pop_subgoal])  
                    self.subgoal_rl = False
                    self.test_recurrent_hidden_states_gr = torch.zeros(1, self.actor_critic_gr.net.num_recurrent_layers, 512, device=self.device)
                    self.not_done_masks0 = torch.zeros(1, 1, device=self.device)
                    self.prev_actions0 = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                
                batch = batch_obs([{'task_id':np.ones((1,))*1,
                                    'pointgoal_with_gps_compass':np.zeros((2,)),
                                    'depth':depth_trans,
                                    'rgb':rgb_trans}], device=self.device)
                (fusion_values, actions, self.recurrent_hidden_states) = self.actor_critic_3fusion.act(
                            batch,
                            self.recurrent_hidden_states,
                            self.prev_actions.long(),
                            self.not_done_masks.byte(),
                            deterministic=False)
                actions = actions['actions']
                assert actions != 0
                self.action_track.append((str(action), 'RL'))
            else:
                self.zero_rls.append(False)
                # UPDATE EXPLORE RL STATE
                batch = batch_obs([{'task_id':np.ones((1,))*1,
                                    'pointgoal_with_gps_compass':np.zeros((2,)),
                                    'depth':depth_trans,
                                    'rgb':rgb_trans}], device=self.device)
                with torch.no_grad():
                    (fusion_values, actions, self.recurrent_hidden_states) = self.actor_critic_3fusion.act(
                            batch,
                            self.recurrent_hidden_states,
                            self.prev_actions.long(),
                            self.not_done_masks.byte(),
                            deterministic=False)    
                    self.not_done_masks.fill_(1.0) 
                    if actions!=0:
                        actions = actions['actions']
                    action = actions.item()
                    self.prev_actions.copy_(torch.tensor([action]))  # ACTION FROM GR_RL     
                self.action_track.append((str(action), 'RL GoalReacher'))
        else:
            #self.step_switcher>=250 np.array(self.explore_areas[-20:]).mean()<self.area_threshold)
            print('FBE')
            if self.path is not None:
                self.paths.append(self.path)
            else:
                self.paths.append([])
            self.zero_rls.append(False)
            self.lastACTION5.append(False)
            # UPDATE EXPLORE RL STATE
            rgb_trans = (self.p240(obs['rgb']).permute(1,2,0)*255)
            depth_trans = self.p240(dpth).permute(1,2,0)
            batch = batch_obs([{'task_id':np.ones((1,))*1,
                                'pointgoal_with_gps_compass':np.zeros((2,)),
                                'depth':depth_trans,
                                'rgb':rgb_trans}], device=self.device)
            with torch.no_grad():
                (fusion_values, actions, self.recurrent_hidden_states) = self.actor_critic_3fusion.act(
                        batch,
                        self.recurrent_hidden_states,
                        self.prev_actions.long(),
                        self.not_done_masks.byte(),
                        deterministic=False)    
                self.not_done_masks.fill_(1.0) 
                if actions!=0:
                    actions = actions['actions']
                action = actions.item()

                
            self.choose_goal_fbe(obs)
            self.create_path_by_planner(obs)
            if self.path is not None:
                action = self.act_fbe(obs)                               ######  action_fbe
                self.prev_actions.copy_(torch.tensor(action))
                self.action_track.append((str(action), 'FBE'))
                #return action_fbe, obs_semantic, dpth
            else:
                self.prev_actions.copy_(actions)
                self.action_track.append((str(action), 'RL'))
                #return action, obs_semantic, dpth
        
        #self.critic_values.append(fusion_values.item())
        self.critic_gr_values_start.append(gr_values.item())
        #self.fbe_cost_values.append((15 - self.exploration.goal_cost) / 10)
        self.map_expand_values.append(np.array(self.explore_areas[-20:]).mean())
        
        
            
        #if self.dstance_to_object<.8:
        #    action = 0
        #    print('Act0: ',self.dstance_to_object)

        print('Action:', action)
        #return action, obs_semantic, dpth
        return {"action": action}
    
    
    # EXPLORE RL
        """
        #if (not self.skil_goalreacher) and (not self.skil_goalreacher_rl) and self.critic_values[-1]>=self.fbe_cost_values[-1]:
        if (not self.skil_goalreacher) and (not self.skil_goalreacher_rl) and np.array(self.explore_areas[-20:]).mean()>=self.area_threshold:  
        
        #self.step_switcher<250  np.array(self.explore_areas[-20:]).mean()>=self.area_threshold
            self.zero_rls.append(False)
            self.lastACTION5.append(False)
            print('EXplore RL')
            rgb_trans = (self.p240(obs['rgb']).permute(1,2,0)*255)
            depth_trans = self.p240(dpth).permute(1,2,0)
            batch = batch_obs([{'task_id':np.ones((1,))*1,
                                'pointgoal_with_gps_compass':np.zeros((2,)),
                                'depth':depth_trans,
                                'rgb':rgb_trans}], device=self.device)
            with torch.no_grad():
                (fusion_values, actions, self.recurrent_hidden_states) = self.actor_critic_3fusion.act(
                        batch,
                        self.recurrent_hidden_states,
                        self.prev_actions.long(),
                        self.not_done_masks.byte(),
                        deterministic=False)    
                self.not_done_masks.fill_(1.0) 
                if actions!=0:
                    actions = actions['actions']
                self.prev_actions.copy_(actions)
            action = actions.item()
            self.action_track.append((str(action), 'RL'))
        """
        
        # FBE
        
        #if (self.skil_goalreacher or self.critic_values[-1]<self.fbe_cost_values[-1]) and not ((self.skil_goalreacher_rl) and (not self.skil_goalreacher)):
        #if (self.skil_goalreacher or np.array(self.explore_areas[-20:]).mean()<self.area_threshold) and not ((self.skil_goalreacher_rl) and (not self.skil_goalreacher)):
        #if not self.skil_goalreacher_rl or self.skil_goalreacher: