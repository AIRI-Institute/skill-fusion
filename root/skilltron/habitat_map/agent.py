import argparse
import os
import sys
import random
from collections import OrderedDict
import imageio

import numba
import numpy as np
import PIL
import torch
from gym.spaces import Box, Dict, Discrete

import habitat
from habitat import Config
from habitat.core.agent import Agent
from habitat import make_dataset
from hab_base_utils_common import batch_obs
from habitat_baselines.config.default import get_config
from arguments import get_args as get_args_env
from habitat.utils.visualizations import maps
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from torchvision import transforms

from env_orb import Env

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
from utils import FrameSkip, FrameStack, draw_top_down_map, plot_colortable, rand_cmap
import utils_for_vectorenv
import matplotlib.patches as patches

import argparse
from adet.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.engine.defaults import DefaultPredictor
from adet.utils.visualizer import TextVisualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
#from rednet import RedNet, load_rednet, RedNetResizeWrapper

#from bps_nav_test.rl.ddppo.policy.resnet_policy_fusion import ResNetPolicy as ResNetPolicy_fusion
#from bps_nav_test.rl.ddppo.policy import ResNetPolicy
from resnet_policy import PointNavResNetPolicy
from resnet_policy_gr import PointNavResNetPolicy as Policy_gr
from resnet_policy_3fusion import ResNetPolicy as Policy_3fusion

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def to_gt_map(one_env,obs,info,x):
    original_map_size = maps.colorize_topdown_map(info["top_down_map"]["map"], info["top_down_map"]["fog_of_war_mask"]).shape[:2]
    map_scale = np.array((1, original_map_size[1] * 1.0 / original_map_size[0]))
    new_map_size = np.round(obs['rgb'][0].shape[0] * map_scale).astype(np.int32)
    cord_to_map1 = x
    cord_map1 = np.round(maps.to_grid(cord_to_map1[2],cord_to_map1[0],one_env.task .measurements.__dict__['measures']['top_down_map']._top_down_map.shape[0:2],sim=one_env._sim)* new_map_size / original_map_size).astype(np.int32)[::-1]
    return cord_map1

def pointInRect(point,rect):
    x1, y1, z1, w, h, l = rect
    x2, y2, z2 = x1+w, y1+h, z1+l
    x, y, z = point
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            if (z1 < z and z < z2):
                return True
    return False

def compute_pointgoal(source_position,source_rotation,goal_position):
    direction_vector = goal_position - source_position
    direction_vector_agent = quaternion_rotate_vector(source_rotation.inverse(), direction_vector)
    rho, phi = cartesian_to_polar(-direction_vector_agent[2], direction_vector_agent[0])
    return np.array([rho, -phi], dtype=np.float32)   

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg  

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file")
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument("--output",
                help="A file or directory to save output visualizations. "
                "If not given, will show output in an OpenCV window.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown")
    parser.add_argument("--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER)
    return parser 

def resize2d(img, size):
    return (F.adaptive_avg_pool2d(Variable(img,volatile=True), size)).data

class Agent_hlpo:
    def __init__(self):
        
        self.p448 = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((448,448)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225)),
                                ])
        self.p = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((120,160)),
                                transforms.ToTensor()])
        self.p240 = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((240,320)),
                                transforms.ToTensor()])
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        obs_space1 = dotdict()
        obs_space1.spaces = {}
        obs_space1.spaces['rgb'] = Box(low=0, high=0, shape=(3,120,160), dtype=np.uint8)
        obs_space1.spaces['task_id'] = Box(low=0, high=1, shape=(1,), dtype=np.uint8)
        act_space = dotdict()
        act_space.n = 4
        act_space.shape = [1] 
        self.actor_critic_bex = ResNetPolicy_fusion(
                    observation_space=obs_space1,
                    action_space=act_space,
                    hidden_size=512,
                    resnet_baseplanes=64,
                    rnn_type='LSTM',
                    num_recurrent_layers=1,
                    backbone='resnet18',
                )
        pretrained_state = torch.load('/root/weights/rgb_fusion_1447.pth', map_location="cpu")
        
        pretrained_state = {k[len("actor_critic.") :]: v for k, v in pretrained_state["state_dict"].items()} 
        self.actor_critic_bex.load_state_dict(pretrained_state)
        self.actor_critic_bex.to(self.device)

        obs_space1 = dotdict()
        obs_space1.spaces = {}
        obs_space1.spaces['rgb'] = Box(low=0, high=0, shape=(3,120,160), dtype=np.uint8)
        obs_space1.spaces['depth'] = Box(low=-3.4028234663852886e+38, high=3.4028234663852886e+38, shape=(1,120,160), dtype=np.float32)
        obs_space1.spaces['pointgoal_with_gps_compass'] = Box(low=0., high=1.0, shape=(2,), dtype=np.float32)
        act_space = dotdict()
        act_space.n = 4
        act_space.shape = [1] 
        self.actor_critic_pnav = ResNetPolicy(
                    observation_space=obs_space1,
                    action_space=act_space,
                    hidden_size=512,
                    resnet_baseplanes=64,
                    rnn_type='LSTM',
                    num_recurrent_layers=1,
                    backbone='resnet18',
                )
        pretrained_state = torch.load('/root/weights/PNAV_rgbd_75.pth', map_location="cpu")
        pretrained_state = {k[len("actor_critic.") :]: v for k, v in pretrained_state["state_dict"].items()} 
        self.actor_critic_pnav.load_state_dict(pretrained_state)
        self.actor_critic_pnav.to(self.device)
        
        obs_space2 = dotdict()
        obs_space2.spaces = {}
        obs_space2.spaces['rgb'] = Box(low=-1000, high=1000, shape=(120,160, 3), dtype=np.float32)
        obs_space2.spaces['depth'] = Box(low=-1000, high=1000, shape=(120,160, 1), dtype=np.float32)
        obs_space2.spaces['semantic'] = Box(low=-1000, high=1000, shape=(120,160, 1), dtype=np.float32)
        act_space2 = dotdict()
        act_space2.n = 4
        act_space2.shape = [1] 
        self.actor_critic_newgr = Policy_gr(observation_space=obs_space2,
                                            action_space=act_space2,
                                            hidden_size=512,
                                            num_recurrent_layers=1,
                                            rnn_type="GRU",
                                            resnet_baseplanes=32,
                                            backbone="resnet18",
                                            normalize_visual_inputs=True)
        pretrained_state = pretrained_state = torch.load('/root/weights/newgr_218.pth', map_location="cpu")
        #pretrained_state = pretrained_state = torch.load('/root/weights/newgr_655.pth', map_location="cpu")
        self.actor_critic_newgr.load_state_dict({k[len("actor_critic.") :]: v for k, v in pretrained_state["state_dict"].items()})
        self.actor_critic_newgr.to(self.device)

        obs_space2 = dotdict()
        obs_space2.spaces = {}
        obs_space2.spaces['rgb'] = Box(low=-1000, high=1000, shape=(240,320,3), dtype=np.float32)
        obs_space2.spaces['depth'] = Box(low=-1000, high=1000, shape=(240,320,1), dtype=np.float32)
        obs_space2.spaces['semantic'] = Box(low=-1000, high=1000, shape=(240,320,1), dtype=np.float32)
        act_space2 = dotdict()
        act_space2.n = 4
        act_space2.shape = [1] 
        self.actor_critic_gr = PointNavResNetPolicy(
            observation_space = obs_space2,#self.obs_space,#
            action_space = act_space2,
            hidden_size = 512,
            rnn_type = 'GRU',
            num_recurrent_layers = 1,
            backbone = 'resnet18',
            normalize_visual_inputs=True)
        pretrained_state = torch.load('/root/weights/goalreacher_rgbd_230.pth', map_location="cpu")
        pretrained_state = {k[len("actor_critic.") :]: v for k, v in pretrained_state["state_dict"].items()}          
        self.actor_critic_gr.load_state_dict(pretrained_state)
        self.actor_critic_gr.to(self.device)
        
        obs_space1 = dotdict()
        obs_space1.spaces = {}
        obs_space1.spaces['rgb'] = Box(low=0, high=0, shape=(3,120,160), dtype=np.uint8)
        obs_space1.spaces['depth'] = Box(low=0, high=0, shape=(1,120,160), dtype=np.uint8)
        obs_space1.spaces['pointgoal_with_gps_compass'] = Box(low=0, high=0, shape=(2,), dtype=np.uint8)
        obs_space1.spaces['task_id'] = Box(low=0, high=1, shape=(1,), dtype=np.uint8)
        act_space = dotdict()
        act_space.n = 4
        act_space.shape = [1] 
        self.actor_critic_3fusion = Policy_3fusion(
                    observation_space=obs_space1,
                    action_space=act_space,
                    hidden_size=512,
                    resnet_baseplanes=64,
                    rnn_type='LSTM',
                    num_recurrent_layers=1,
                    backbone='resnet18',
                )
        pretrained_state = torch.load('/root/3_FUSION_mp3d/checkpoint/ckpt.175.pth', map_location="cpu")
        #pretrained_state = torch.load('/root/weights/3fusion_873.pth', map_location="cpu")
        pretrained_state = {k[len("actor_critic.") :]: v for k, v in pretrained_state["state_dict"].items()} 
        self.actor_critic_3fusion.load_state_dict(pretrained_state)
        self.actor_critic_3fusion.to(self.device)
        
        self.dataset_meta = {1: ('wall'),       2: ('floor'),      3: ('chair'),         4: ('door'),            5: ('table'),       6: ('picture'),
                7: ('cabinet'),            8: ('cushion'),    9: ('window'),        10: ('sofa'),           11: ('bed'),        12: ('curtain'),
                13: ('chest_of_drawers'),  14: ('plant'),     15: ('sink'),         16: ('stairs'),         17: ('ceiling'),
                18: ('toilet'),            19: ('stool'),     20: ('towel'),        21: ('mirror'),         22: ('tv_monitor'), 23: ('shower'),
                24: ('column'),            25: ('bathtub'),   26: ('counter'),      27: ('fireplace'),      28: ('lighting'),   29: ('beam'),
                30: ('railing'),           31: ('shelving'),  32: ('blinds'),       33: ('gym_equipment'),  34: ('seating'),
                35: ('board_panel'),       36: ('furniture'), 37: ('appliances'),   38: ('clothes'),        39: ('objects'),    40: ('misc')}
        
        self.objgoal_to_cat = {0: 'chair',     1: 'table',     2: 'picture',           3: 'cabinet',           4: 'cushion',   
                          5: 'sofa',      6: 'bed',       7: 'chest_of_drawers',  8: 'plant',             9: 'sink',
                          10: 'toilet',   11: 'stool',    12: 'towel',            13: 'tv_monitor',       14: 'shower',
                          15: 'bathtub',  16: 'counter',  17: 'fireplace',        18: 'gym_equipment',    19: 'seating', 20: 'clothes'}
        
        self.dataset_crossover = {'chair':'chair',  'table':'dining_table',  'picture':'_',         'cabinet':'_',
                     'cushion':'_',                'sofa':'couch',           'bed':'bed',           'chest_of_drawers':'_',
                     'plant':'potted_plant',       'sink':'sink',            'toilet':'toilet',     'stool':'chair',
                     'towel':'_',                  'tv_monitor':'tv',        'shower':'_',          'bathtub':'_',
                     'counter':'_',                'fireplace':'oven',       'gym_equipment':'_',   'seating':'chair',        'clothes':'_'}
        
        self.dataset_meta2 = {1: ('person'),       2: ('bicycle'),                3: ('car'),                    4: ('motorcycle'),
                    5: ('airplane'),                6: ('bus'),                    7: ('train'),                  8: ('truck'),
                    9: ('boat'),                    10: ('traffic_light'),         11: ('fire_hydrant'),          12: ('stop_sign'),
                    13: ('parking_meter'),          14: ('bench'),                 15: ('bird'),                  16: ('cat'),
                    17: ('dog'),                    18: ('horse'),                 19: ('sheep'),                 20: ('cow'),
                    21: ('elephant'),               22: ('bear'),                  23: ('zebra'),                 24: ('giraffe'),
                    25: ('backpack'),               26: ('umbrella'),              27: ('handbag'),               28: ('tie'),
                    29: ('suitcase'),               30: ('frisbee'),               31: ('skis'),                  32: ('snowboard'),
                    33: ('sports_ball'),            34: ('kite'),                  35: ('baseball_bat'),          36: ('baseball_glove'),
                    37: ('skateboard'),             38: ('surfboard'),             39: ('tennis_racket'),         40: ('bottle'),
                    41: ('wine_glass'),             42: ('cup'),                   43: ('fork'),                  44: ('knife'),
                    45: ('spoon'),                  46: ('bowl'),                  47: ('banana'),                48: ('apple'),
                    49: ('sandwich'),               50: ('orange'),                51: ('broccoli'),              52: ('carrot'),
                    53: ('hot_dog'),                54: ('pizza'),                 55: ('donut'),                 56: ('cake'),
                    57: ('chair'),                  58: ('couch'),                 59: ('potted_plant'),          60: ('bed'),
                    61: ('dining_table'),           62: ('toilet'),                63: ('tv'),                    64: ('laptop'),
                    65: ('mouse'),                  66: ('remote'),                67: ('keyboard'),              68: ('cell_phone'),
                    69: ('microwave'),              70: ('oven'),                  71: ('toaster'),               72: ('sink'),
                    73: ('refrigerator'),           74: ('book'),                  75: ('clock'),                 76: ('vase'),
                    77: ('scissors'),               78: ('teddy_bear'),            79: ('hair_drier'),            80: ('toothbrush')}

        self.cat_to_objectgoal = {v:k for k,v in self.objgoal_to_cat.items()}
        self.cat40_to_cat20 = {k:self.cat_to_objectgoal[v] if v in list(self.objgoal_to_cat.values()) else -1 for k,v in self.dataset_meta.items()}
        self.cat40_to_cat20[0] = -1
        self.cat40_to_cat20[-1] = -1
        
        a2 = "--config-file /AdelaiDet/configs/SOLOv2/R50_3x.yaml \
            --input input1.jpg input2.jpg \
            --opts MODEL.WEIGHTS /root/weights/SOLOv2_R50_3x.pth".split()
        args = get_parser().parse_args(a2)
        cfg = setup_cfg(args)
        self.predictor = DefaultPredictor(cfg) 
        self.rednet_model = RedNetResizeWrapper(self.device,num_classes=40,resize=True)
        pretrained_state = torch.load('/root/weights/rednet_semmap_mp3d_40.pth', map_location="cpu")
        pretrained_state = {k[len("module.") :]: v for k, v in pretrained_state['model_state'].items()}          
        self.rednet_model.rednet.load_state_dict(pretrained_state)
        self.rednet_model.to(self.device)

    def reset(self, one_env):
        
        self.recurrent_hidden_states_pnav = torch.zeros(1,2,512,device=self.device)
        self.not_done_masks0 = torch.zeros(1, 1, device=self.device)
        self.prev_actions0 = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self.recurrent_hidden_states = torch.zeros(1, 2, 512, device=self.device)
        self.test_recurrent_hidden_states_gr = torch.zeros(1, self.actor_critic_gr.net.num_recurrent_layers, 512, device=self.device)
        self.not_done_masks = torch.zeros(1, 1, device=self.device)
        self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self.not_done_masks1 = torch.zeros(1, 1, device=self.device)
        self.prev_actions1 = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self.skil_goalreacher = False
        
        self.in_room_count = 0
        self.dstance_to_object = 10.
        self.pnav = True
        self.change_room = True
        self.room_visited = 0
        self.forever_explore = False
        
        rooms_goal = []
        rooms_dist = []
        for ii in one_env.current_episode.goals:
            state = ii.position
            for i in range(len(one_env.sizes)):
                if pointInRect(state,list(one_env.regions[i]-one_env.sizes[i]/2)+list(one_env.sizes[i])):
                    rooms_goal.append(one_env.names[i])
        self.rooms_goal = list(set(rooms_goal)) 
        if len(set(rooms_goal))>1:
            rooms_goal = [i for i in rooms_goal if i!='hallway']
        ind_rooms = []
        rooms_dist = []
        for ii,name in enumerate(one_env.names):
            if name in rooms_goal:
                ind_rooms.append(ii)            
        for i in range(len(one_env.sizes)):            
            rooms_dist.append(one_env.dist(one_env.regions[i][[0,2]],one_env.sim.get_agent_state().position[[0,2]]))
        rooms_cords = [one_env.regions[i] for i in ind_rooms]
        rooms_sizes = [one_env.sizes[i] for i in ind_rooms]
        self.rooms_cords1 = list(np.copy(np.array(rooms_cords)[np.argsort(np.array(rooms_dist)[ind_rooms])[::-1]])) 
        self.rooms_sizes1 = list(np.copy(np.array(rooms_sizes)[np.argsort(np.array(rooms_dist)[ind_rooms])[::-1]])) 
        
        
    def step_3skill(self,one_env,obs, depth = True, semantic = True):
        
        if depth:
            if len(obs['depth'].shape)==2:
                dpth = obs['depth']
            else:
                dpth = obs['depth'][:,:,0]
        else:
            dpth = obs['depth']
        
        if semantic:
            sem = obs['semantic']
            obs_semantic = cv2.erode(sem,np.ones((4,4),np.uint8),iterations = 3)[:,:,np.newaxis].astype(bool).astype(float)
        else:
            after_crossover = self.dataset_crossover[self.objgoal_to_cat[obs['objectgoal'][0]]]
            if after_crossover != '_':
                image = np.transpose(obs['rgb'],(0,1,2))
                predictions = self.predictor(image)
                classes = [i for ii,i in enumerate(predictions["instances"].pred_classes.cpu().numpy()) if predictions["instances"].scores.cpu().numpy()[ii]>0.25]
                class_meta2 = {v:k for k,v in self.dataset_meta2.items()}[after_crossover]
                mask = np.zeros((480,640))
                if class_meta2 in [i+1 for i in classes]:
                    mask_ids = np.where(predictions["instances"].pred_classes.cpu().numpy() == class_meta2-1)[0]
                    mask = np.sum(np.array([predictions["instances"].pred_masks[mask_id].cpu().float().numpy() for mask_id in mask_ids]), axis=0).astype(bool).astype(float)  
            else:
                rgb_trans = (self.p240(obs['rgb']).permute(1,2,0)*255).unsqueeze(0).cuda()
                depth_trans = self.p240(dpth).permute(1,2,0).unsqueeze(0).cuda()
                obs_sem = np.vectorize({k:k-1 for k,v in self.dataset_meta.items()}.get)(self.rednet_model(rgb_trans,depth_trans).cpu()[0])   # k:k-1 for mp3d_40
                obs_sem = np.vectorize(self.cat40_to_cat20.get)(obs_sem)
                obs_sem = resize2d(torch.tensor(obs_sem).float().unsqueeze(0), (480,640))[0].numpy()
                mask = np.float32((obs_sem)==int(obs['objectgoal']))[:,:,np.newaxis]
            obs_semantic = cv2.erode(mask,np.ones((4,4),np.uint8),iterations = 4)[:,:,np.newaxis] 
    
    
        rooms = [one_env.names[i] for i in range(len(one_env.sizes)) if pointInRect(one_env.sim.get_agent_state().position,list(one_env.regions[i]-one_env.sizes[i]/2)+list(one_env.sizes[i]))]+['empty']
        
        if self.pnav and (len(self.rooms_cords1)>0):
            if self.change_room:
                self.change_room = False
                self.gl_rm = self.rooms_cords1.pop()
                self.gl_sz = self.rooms_sizes1.pop()
                print('Change room')
            goal_pos = np.array(self.gl_rm)
            agent_pos = one_env.sim.get_agent_state().position
            agent_rot = one_env.sim.get_agent_state().rotation
            pointgoal_with_gps_compass = compute_pointgoal(agent_pos,agent_rot,goal_pos) 
            
            rgb_trans = (self.p(obs['rgb'])*255).int()
            depth_trans = self.p(dpth)
            batch = batch_obs([{'pointgoal_with_gps_compass':pointgoal_with_gps_compass,
                                'depth':depth_trans,
                                'rgb':rgb_trans}], device=self.device)
            
            with torch.no_grad():
                (values, actions, self.recurrent_hidden_states_pnav) = self.actor_critic_pnav.act(
                        batch,
                        self.recurrent_hidden_states_pnav,
                        self.prev_actions0.long(),
                        self.not_done_masks0.byte(),
                        deterministic=False)    
                self.not_done_masks0.fill_(1.0) 
                if actions!=0:
                    actions = actions['actions']
                self.prev_actions0.copy_(actions) 
            action = actions.item() 
            
            if action==0:
                self.change_room = True
                self.pnav = False
                print('Act0 in room')
                
            if pointInRect(one_env.sim.get_agent_state().position,list(self.gl_rm-self.gl_sz/2)+list(self.gl_sz)):
                self.room_visited += 1
                
            if self.room_visited>=3:
                print('In Room')
                self.change_room = True
                self.pnav = False
                self.room_visited = 0.
                
            if np.sum(obs_semantic)>600. and (rooms[0] in self.rooms_goal):
                print('Goal seen')
                self.pnav = False

        if (not self.pnav) or (len(self.rooms_cords1)==0):
            action, obs_semantic, dpth = self.step_2skill(obs, depth = depth, semantic = semantic)
            
            if not (rooms[0] in self.rooms_goal) and (not self.skil_goalreacher):
                self.change_room = True
                self.pnav = True
                self.room_visited = 0
                self.recurrent_hidden_states_pnav = torch.zeros(1,2,512,device=self.device)
                self.not_done_masks0 = torch.zeros(1, 1, device=self.device)
                self.prev_actions0 = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                self.recurrent_hidden_states = torch.zeros(1, 2, 512, device=self.device)
                self.test_recurrent_hidden_states_gr = torch.zeros(1, self.actor_critic_gr.net.num_recurrent_layers, 512, device=self.device)
                self.not_done_masks = torch.zeros(1, 1, device=self.device)
                self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                self.not_done_masks1 = torch.zeros(1, 1, device=self.device)
                self.prev_actions1 = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                self.skil_goalreacher = False
                
            
        return action, obs_semantic, dpth
        
    def step_2skill(self,obs, depth = True, semantic = True):
        
        if depth:
            if len(obs['depth'].shape)==2:
                dpth = obs['depth']
            else:
                dpth = obs['depth'][:,:,0]
        else:
            dpth = obs['depth']
        
        if semantic:
            sem = obs['semantic']
            obs_semantic = cv2.erode(sem,np.ones((4,4),np.uint8),iterations = 4)[:,:,np.newaxis].astype(bool).astype(float)
        else:
            after_crossover = self.dataset_crossover[self.objgoal_to_cat[obs['objectgoal'][0]]]
            if after_crossover != '_':
                image = np.transpose(obs['rgb'],(0,1,2))
                predictions = self.predictor(image)
                classes = [i for ii,i in enumerate(predictions["instances"].pred_classes.cpu().numpy()) if predictions["instances"].scores.cpu().numpy()[ii]>0.3]
                class_meta2 = {v:k for k,v in self.dataset_meta2.items()}[after_crossover]
                mask = np.zeros((480,640))
                if class_meta2 in [i+1 for i in classes]:
                    mask_ids = np.where(predictions["instances"].pred_classes.cpu().numpy() == class_meta2-1)[0]
                    mask = np.sum(np.array([predictions["instances"].pred_masks[mask_id].cpu().float().numpy() for mask_id in mask_ids]), axis=0).astype(bool).astype(float)  
            else:
                rgb_trans = (self.p240(obs['rgb']).permute(1,2,0)*255).unsqueeze(0).cuda()
                depth_trans = self.p240(dpth).permute(1,2,0).unsqueeze(0).cuda()
                obs_sem = np.vectorize({k:k-1 for k,v in self.dataset_meta.items()}.get)(self.rednet_model(rgb_trans,depth_trans).cpu()[0])   # k:k-1 for mp3d_40
                obs_sem = np.vectorize(self.cat40_to_cat20.get)(obs_sem)
                obs_sem = resize2d(torch.tensor(obs_sem).float().unsqueeze(0), (480,640))[0].numpy()
                mask = np.float32((obs_sem)==int(obs['objectgoal']))[:,:,np.newaxis]
            obs_semantic = cv2.erode(mask,np.ones((4,4),np.uint8),iterations = 4)[:,:,np.newaxis]   
        
        if np.sum(obs_semantic)>600.:
            self.skil_goalreacher = True
            
        if (not self.skil_goalreacher):
            rgb_trans = (self.p(obs['rgb'])*255)
            batch = batch_obs([{'task_id':np.ones((1,)),
                                'rgb':rgb_trans}], device=self.device)
            with torch.no_grad():
                (values, actions, self.recurrent_hidden_states) = self.actor_critic_bex.act(
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
            
        else:
            rgb_trans = (self.p240(obs['rgb']).permute(1,2,0)*255)
            depth_trans = self.p240(dpth).permute(1,2,0)
            sem_trans = self.p240(torch.tensor(obs_semantic)[:,:,0]).permute(1,2,0)
            ds=2
            batch = batch_obs([{'rgb':rgb_trans,
                                'depth':depth_trans,
                                'semantic':sem_trans}], device=self.device)
            with torch.no_grad():
                _, action, _, self.test_recurrent_hidden_states_gr = self.actor_critic_gr.act(
                    batch,
                    self.test_recurrent_hidden_states_gr,
                    self.prev_actions1,
                    self.not_done_masks1.byte(),
                    deterministic=False)
                self.not_done_masks1.fill_(1.0)
                self.prev_actions1.copy_(action) 
                action = action.item() 
                
            mask_depth = dpth*obs_semantic[:,:,0]
            mask_depth[mask_depth==0] = 100.
            mmin, mmax, xymin,xymax = cv2.minMaxLoc(mask_depth)
            self.dstance_to_object = min(self.dstance_to_object,mmin*4.5+0.5)     
                
        if action==0 and self.dstance_to_object>.8: 
            print('Act0: ',self.dstance_to_object)
            action = np.random.choice([1,2,3])
            self.skil_goalreacher = False 
            self.recurrent_hidden_states_pnav = torch.zeros(1,2,512,device=self.device)
            self.not_done_masks0 = torch.zeros(1, 1, device=self.device)
            self.recurrent_hidden_states = torch.zeros(1, 2, 512, device=self.device)
            self.test_recurrent_hidden_states_gr = torch.zeros(1, self.actor_critic_gr.net.num_recurrent_layers, 512, device=self.device)
            #self.not_done_masks = torch.zeros(1, 1, device=self.device)
            #self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)
            #self.not_done_masks1 = torch.zeros(1, 1, device=self.device)
            #self.prev_actions1 = torch.zeros(1, 1, dtype=torch.long, device=self.device)        
                
        return action, obs_semantic, dpth    
    
    
    
    def step_3skill_newgr(self,one_env,obs, depth = True, semantic = True):
        
        if depth:
            if len(obs['depth'].shape)==2:
                dpth = obs['depth']
            else:
                dpth = obs['depth'][:,:,0]
        else:
            dpth = obs['depth']
        
        if semantic:
            sem = obs['semantic']
            obs_semantic = cv2.erode(sem,np.ones((4,4),np.uint8),iterations = 4)[:,:,np.newaxis].astype(bool).astype(float)
            depth_mask = obs['depth']!=0.
            obs_semantic*=depth_mask
            
        else:
            after_crossover = self.dataset_crossover[self.objgoal_to_cat[obs['objectgoal'][0]]]
            if after_crossover != '_':
                image = np.transpose(obs['rgb'],(0,1,2))
                predictions = self.predictor(image)
                classes = [i for ii,i in enumerate(predictions["instances"].pred_classes.cpu().numpy()) if predictions["instances"].scores.cpu().numpy()[ii]>0.25]
                class_meta2 = {v:k for k,v in self.dataset_meta2.items()}[after_crossover]
                mask = np.zeros((480,640))
                if class_meta2 in [i+1 for i in classes]:
                    mask_ids = np.where(predictions["instances"].pred_classes.cpu().numpy() == class_meta2-1)[0]
                    mask = np.sum(np.array([predictions["instances"].pred_masks[mask_id].cpu().float().numpy() for mask_id in mask_ids]), axis=0).astype(bool).astype(float)  
            else:
                rgb_trans = (self.p240(obs['rgb']).permute(1,2,0)*255).unsqueeze(0).cuda()
                depth_trans = self.p240(dpth).permute(1,2,0).unsqueeze(0).cuda()
                obs_sem = np.vectorize({k:k-1 for k,v in self.dataset_meta.items()}.get)(self.rednet_model(rgb_trans,depth_trans).cpu()[0])   # k:k-1 for mp3d_40
                obs_sem = np.vectorize(self.cat40_to_cat20.get)(obs_sem)
                obs_sem = resize2d(torch.tensor(obs_sem).float().unsqueeze(0), (480,640))[0].numpy()
                mask = np.float32((obs_sem)==int(obs['objectgoal']))[:,:,np.newaxis]
            obs_semantic = cv2.erode(mask,np.ones((4,4),np.uint8),iterations = 4)[:,:,np.newaxis] 
    
    
        rooms = [one_env.names[i] for i in range(len(one_env.sizes)) if pointInRect(one_env.sim.get_agent_state().position,list(one_env.regions[i]-one_env.sizes[i]/2)+list(one_env.sizes[i]))]+['empty']
        
        if self.pnav and (not self.forever_explore):
            if self.change_room:
                self.recurrent_hidden_states_pnav = torch.zeros(1,2,512,device=self.device)
                self.not_done_masks0 = torch.zeros(1, 1, device=self.device)
                self.prev_actions0 = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                self.change_room = False
                if (len(self.rooms_cords1)>0):
                    self.gl_rm = self.rooms_cords1.pop()
                    self.gl_sz = self.rooms_sizes1.pop()
                else:
                    print('FOREVER EXPLORE')
                    self.forever_explore = True
                print('Change room',len(self.rooms_cords1))
                
            goal_pos = np.array(self.gl_rm)
            agent_pos = one_env.sim.get_agent_state().position
            agent_rot = one_env.sim.get_agent_state().rotation
            pointgoal_with_gps_compass = compute_pointgoal(agent_pos,agent_rot,goal_pos) 
            
            rgb_trans = (self.p(obs['rgb'])*255).int()
            depth_trans = self.p(dpth)
            batch = batch_obs([{'task_id':np.ones((1,))*2,
                                'pointgoal_with_gps_compass':pointgoal_with_gps_compass,
                                'depth':depth_trans,
                                'rgb':rgb_trans}], device=self.device)
            
            with torch.no_grad():
                (values, actions, self.recurrent_hidden_states_pnav) = self.actor_critic_3fusion.act(
                        batch,
                        self.recurrent_hidden_states_pnav,
                        self.prev_actions0.long(),
                        self.not_done_masks0.byte(),
                        deterministic=False)    
                self.not_done_masks0.fill_(1.0) 
                if actions!=0:
                    actions = actions['actions']
                self.prev_actions0.copy_(actions) 
            action = actions.item()
            
            if action==0:
                self.change_room = True
                self.pnav = False
                print('Act0 in room')
                
            if pointInRect(one_env.sim.get_agent_state().position,list(self.gl_rm-self.gl_sz/2)+list(self.gl_sz)):
                self.room_visited += 1
                
            if self.room_visited>=3:
                print('In Room')
                self.change_room = True
                self.pnav = False
                self.room_visited = 0.
                
            if np.sum(obs_semantic)>200. and (rooms[0] in self.rooms_goal):
                print('Goal seen')
                self.pnav = False

        if (not self.pnav) or self.forever_explore:
            action, obs_semantic, dpth = self.step_2skill_newgr(obs, depth = depth, semantic = semantic)
            
            if not (rooms[0] in self.rooms_goal) and (not self.skil_goalreacher):
                self.change_room = True
                self.pnav = True
                self.room_visited = 0
                self.skil_goalreacher = False

        return action, obs_semantic, dpth
    
    def step_2skill_newgr(self,obs, depth = True, semantic = True):
        
        if depth:
            if len(obs['depth'].shape)==2:
                dpth = obs['depth']
            else:
                dpth = obs['depth'][:,:,0]
        else:
            dpth = obs['depth']
        
        if semantic:
            sem = obs['semantic']
            obs_semantic = cv2.erode(sem,np.ones((4,4),np.uint8),iterations = 4)[:,:,np.newaxis].astype(bool).astype(float)
        else:
            after_crossover = self.dataset_crossover[self.objgoal_to_cat[obs['objectgoal'][0]]]
            if after_crossover != '_':
                image = np.transpose(obs['rgb'],(0,1,2))
                predictions = self.predictor(image)
                classes = [i for ii,i in enumerate(predictions["instances"].pred_classes.cpu().numpy()) if predictions["instances"].scores.cpu().numpy()[ii]>0.3]
                class_meta2 = {v:k for k,v in self.dataset_meta2.items()}[after_crossover]
                mask = np.zeros((480,640))
                if class_meta2 in [i+1 for i in classes]:
                    mask_ids = np.where(predictions["instances"].pred_classes.cpu().numpy() == class_meta2-1)[0]
                    mask = np.sum(np.array([predictions["instances"].pred_masks[mask_id].cpu().float().numpy() for mask_id in mask_ids]), axis=0).astype(bool).astype(float)  
            else:
                rgb_trans = (self.p240(obs['rgb']).permute(1,2,0)*255).unsqueeze(0).cuda()
                depth_trans = self.p240(dpth).permute(1,2,0).unsqueeze(0).cuda()
                obs_sem = np.vectorize({k:k-1 for k,v in self.dataset_meta.items()}.get)(self.rednet_model(rgb_trans,depth_trans).cpu()[0])   # k:k-1 for mp3d_40
                obs_sem = np.vectorize(self.cat40_to_cat20.get)(obs_sem)
                obs_sem = resize2d(torch.tensor(obs_sem).float().unsqueeze(0), (480,640))[0].numpy()
                mask = np.float32((obs_sem)==int(obs['objectgoal']))[:,:,np.newaxis]
            obs_semantic = cv2.erode(mask,np.ones((4,4),np.uint8),iterations = 4)[:,:,np.newaxis]
            
        obs_semantic[dpth==0]*=0. ##################################################################################################
        
        if np.sum(obs_semantic)>400.:
            self.skil_goalreacher = True
            
        if (not self.skil_goalreacher):
            rgb_trans = (self.p(obs['rgb'])*255)
            depth_trans = self.p(dpth).permute(1,2,0)
            batch = batch_obs([{'task_id':np.ones((1,)),
                                'pointgoal_with_gps_compass':np.zeros((2,)),
                                'depth':depth_trans,
                                'rgb':rgb_trans}], device=self.device)
            with torch.no_grad():
                (values, actions, self.recurrent_hidden_states) = self.actor_critic_3fusion.act(
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
            
        else:
            rgb_trans = (self.p(obs['rgb'])*255).permute(1,2,0)
            depth_trans = self.p(dpth).permute(1,2,0)
            sem_trans = self.p(torch.tensor(obs_semantic)[:,:,0]).permute(1,2,0)
            batch = batch_obs([{'rgb':rgb_trans,
                                'depth':depth_trans,
                                'semantic':sem_trans}], device=self.device)
            with torch.no_grad():
                _, action, _, self.test_recurrent_hidden_states_gr = self.actor_critic_newgr.act(
                    batch,
                    self.test_recurrent_hidden_states_gr,
                    self.prev_actions1,
                    self.not_done_masks1.byte(),
                    deterministic=False)
                self.not_done_masks1.fill_(1.0)
                self.prev_actions1.copy_(action)  
                action = action.item() 
                
            mask_depth = dpth*obs_semantic[:,:,0]
            mask_depth[mask_depth==0] = 100.
            mmin, mmax, xymin,xymax = cv2.minMaxLoc(mask_depth)
            self.dstance_to_object = min(self.dstance_to_object,mmin*4.5+0.5)  
        self.dstance_to_object +=0.05
                
        if action==0 and self.dstance_to_object>0.8: 
            print('Act0: ',self.dstance_to_object)
            action = 1#np.random.choice([1,2,3])
            self.skil_goalreacher = False 
            self.recurrent_hidden_states = torch.zeros(1, 2, 512, device=self.device)
            self.test_recurrent_hidden_states_gr = torch.zeros(1, self.actor_critic_gr.net.num_recurrent_layers, 512, device=self.device)
            self.not_done_masks = torch.zeros(1, 1, device=self.device)
            self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)
            self.not_done_masks1 = torch.zeros(1, 1, device=self.device)
            self.prev_actions1 = torch.zeros(1, 1, dtype=torch.long, device=self.device)        
                
        return action, obs_semantic, dpth   
