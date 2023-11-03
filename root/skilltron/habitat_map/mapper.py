import sys
import os
sys.path.append('/'.join(os.path.realpath(__file__).split('/')[:-1]))
sys.path.append(os.path.realpath(__file__))
from arguments import get_args as get_args_env
from utils_f.map_builder_objnav import MapBuilder
import skimage
import numpy as np
import utils_f.pose as pu
from time import time
import cv2

class Mapper:
    def __init__(self, 
                 obstacle_inflation=0,
                 semantic_inflation=3,
                 map_size_cm=2400, 
                 map_resolution=5, 
                 semantic_threshold=2, 
                 vision_range=3.2,
                 semantic_vision_range=3.2,
                 semantic_decay=0.9):
        arguments = "--split train \
            --auto_gpu_config 0 \
            -n 1 \
            --num_processes_on_first_gpu 5 \
                --num_processes_per_gpu 16 \
            --train_global 0 --train_local 0 --train_slam 0 \
            --slam_memory_size 150000 \
                --exp_name zero-noise \
            --num_mini_batch 9 \
            --total_num_scenes 1 \
            --split train \
            --task_config my_challenge_mp3d_objectnav2020.local.rgbd.yaml \
            --load_global pretrained_models/model_best.global \
            --load_local pretrained_models/model_best.local \
            --load_slam pretrained_models/model_best.slam \
            --max_episode_length 500".split()
        args_env = get_args_env(arguments)
        args_env.hfov = 42
        args_env.env_frame_height = 640
        args_env.env_frame_width = 480
        args_env.env_frame_semantic_height = 640
        args_env.env_frame_semantic_width = 480
        args_env.camera_height = 1.31
        args_env.du_scale = 1
        args_env.map_resolution = map_resolution
        args_env.map_size_cm = map_size_cm
        args_env.vision_range = int(vision_range / args_env.map_resolution * 100.)
        args_env.semantic_vision_range = int(semantic_vision_range / args_env.map_resolution * 100.)
        self.args = args_env 

        self.semantic_threshold = semantic_threshold
        self.semantic_decay = semantic_decay
        self.obstacle_inflation = obstacle_inflation
        self.semantic_inflation = semantic_inflation
        self.mapper = self.build_mapper()
        full_map_size = args_env.map_size_cm//args_env.map_resolution
        self.map_cells = full_map_size
        self.mapper.set_class_number(7)
        self.map_size_cm = args_env.map_size_cm
        self.mapper.reset_map(self.map_size_cm)

        self.map = None
        self.semantic_map = None
        self.obstacles = []
        self.erosion_size = 3


    def clear_obstacles(self):
        for i, j in self.obstacles:
            self.mapper.map[i, j, 1] = 0
        self.obstacles = []


    def reset(self):
        self.mapper.reset_map(self.map_size_cm)
        
        self.collision_map = np.zeros((self.map_cells, self.map_cells))
        self.curr_loc = [self.map_size_cm/100.0/2.0, self.map_size_cm/100.0/2.0]
        self.last_loc = [self.map_size_cm/100.0/2.0, self.map_size_cm/100.0/2.0]
        self.curr_loc_map = [self.map_size_cm/100.0/2.0, self.map_size_cm/100.0/2.0]
        self.last_loc_map = [self.map_size_cm/100.0/2.0, self.map_size_cm/100.0/2.0, 0]
        self.col_width = 1
        self.fat_map = 6
        self.goal_x, self.goal_y = 10,10
        self.poses = [[479,479],[480,480]]
        self.after_reset = 0
        self.last_sim_location = [0.,0.,0.]#self.get_sim_location()
        self.curr_loc_gt = [self.map_size_cm/100.0/2.0,
                         self.map_size_cm/100.0/2.0, 0.]
        self.clear_obstacles()

    
    def step(self, observations, semantic_mask):
        
        dx_gt, dy_gt, do_gt = self.get_gt_pose_change(list(observations['gps']*np.array([1,-1]))+list(observations['compass']))
        self.curr_loc_gt = pu.get_new_pose(self.curr_loc_gt,(dx_gt, dy_gt, do_gt))
        self.curr_loc_map = [int((observations['gps'][0] + self.map_size_cm / 200.) * 100.0/self.args.map_resolution),
                int((-observations['gps'][1] + self.map_size_cm / 200.) * 100.0/self.args.map_resolution)]

        if semantic_mask.max() > 0:
            print('Goal is observed')
        
        depth = self._preprocess_depth(observations['depth'])
        mapper_gt_pose = (self.curr_loc_gt[0]*100.0,self.curr_loc_gt[1]*100.0,np.deg2rad(self.curr_loc_gt[2]))
        fp_proj, self.map, fp_explored, self.explored_map, fp_semantic, self.semantic_map = \
            self.mapper.update_map(depth, mapper_gt_pose, semantic_mask, 0) #observations['semantic'][:,:,0]
        self.semantic_map = self.semantic_map.sum(axis=0)
        self.semantic_map[self.semantic_map > 0] = 1
        kernel = np.ones((self.erosion_size, self.erosion_size), dtype=np.uint8)
        self.semantic_map = cv2.erode(self.semantic_map, kernel)
        i, j = (self.semantic_map > 0).nonzero()
        for di in range(-self.semantic_inflation, self.semantic_inflation + 1):
            for dj in range(-self.semantic_inflation, self.semantic_inflation + 1):
                self.semantic_map[np.clip(i + di, 0, self.semantic_map.shape[0] - 1), np.clip(j + dj, 0, self.semantic_map.shape[1] - 1)] = 1
        self.poses.append(self.curr_loc_map)


    def world_to_map(self, x, y):
        i = int(round(y / self.args.map_resolution / 0.01 - 0.9999)) + self.map_cells // 2
        j = int(round(x / self.args.map_resolution / 0.01)) + self.map_cells // 2
        i = np.clip(i, 0, self.map_cells - 1)
        j = np.clip(j, 0, self.map_cells - 1)
        return i, j


    def map_to_world(self, i, j):
        x = (j - self.map_cells // 2) * self.args.map_resolution * 0.01
        y = (i - self.map_cells // 2) * self.args.map_resolution * 0.01
        x += self.args.map_resolution * 0.005
        y += self.args.map_resolution * 0.005
        #y *= (-1)
        return x, y


    def draw_obstacle_ahead(self, observations):
        #print('Draw obstacle ahead')
        x, y = observations['gps']
        #y *= -1
        angle = observations['compass'][0]
        obst_x = x + np.cos(angle) * 2 * self.args.map_resolution * 0.01
        obst_y = y + np.sin(angle) * 2 * self.args.map_resolution * 0.01
        #print('Obstacle x and y:', obst_x, obst_y)
        obst_i, obst_j = self.world_to_map(obst_x, obst_y)
        #print('Obstacle i and j:', obst_i, obst_j)
        if obst_i >= 0 and obst_i < self.map_cells and obst_j >= 0 and obst_j < self.map_cells:
            self.obstacles.append((obst_i, obst_j))
            self.mapper.map[obst_i, obst_j, 1] = 1
        
        
    def build_mapper(self):
        params = {}
        params['frame_width'] = self.args.env_frame_width
        params['frame_height'] = self.args.env_frame_height
        params['frame_semantic_width'] = self.args.env_frame_semantic_width
        params['frame_semantic_height'] = self.args.env_frame_semantic_height
        params['fov'] =  self.args.hfov
        params['resolution'] = self.args.map_resolution
        params['map_size_cm'] = self.args.map_size_cm
        params['agent_min_z'] = 25
        params['agent_max_z'] = 150
        params['agent_height'] = self.args.camera_height * 100
        params['agent_view_angle'] = 0
        params['du_scale'] = self.args.du_scale
        params['vision_range'] = self.args.vision_range
        params['semantic_vision_range'] = self.args.semantic_vision_range
        params['visualize'] = self.args.visualize
        params['obs_threshold'] = self.args.obs_threshold
        params['classes_number'] = 42
        params['goal_threshold'] = self.semantic_threshold
        params['goal_decay'] = self.semantic_decay
        params['obstacle_inflation'] = self.obstacle_inflation
        params['semantic_inflation'] = self.semantic_inflation

        self.selem = skimage.morphology.disk(self.args.obstacle_boundary /
                                             self.args.map_resolution)
        mapper = MapBuilder(params)
        return mapper 
    
    def get_gt_pose_change(self,loc):
        curr_sim_pose = loc
        dx, dy, do = pu.get_rel_pose_change(curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do   
    
    def _preprocess_depth(self,depth):
        depth = depth[:, :, 0]*1
        #mask2 = depth > 0.99
        #depth[mask2] = 0.

        for i in range(depth.shape[1]):
            depth[:,i][depth[:,i] == 0.] = depth[:,i].max()

        mask1 = depth == 0
        depth[mask1] = np.NaN
        depth = depth*450. + 50.
        #depth = depth * 100
        return depth 