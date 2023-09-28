import numpy as np
import math
import torch
import torch.nn as nn
import gym
import skimage
import cv2
from torchvision import transforms
from .frontier_search import FrontierSearch
from semexp.model import Semantic_Mapping
from semexp.model_pf import RL_Policy
from semexp.utils.storage import GlobalRolloutStorage
from semexp.envs.utils.fmm_planner import FMMPlanner
import semexp.envs.utils.pose as pu
from copy import deepcopy
from PIL import Image

class PoniExploration:
    def __init__(self, args, min_frontier_size=0.25):
        self.args = args
        self.min_frontier_size = min_frontier_size
        self.frontier_blacklist = []
        self.reached_list = []
        self.current_goal = None
        self.replan = False
        self.device = torch.device("cuda:0" if args.cuda else "cpu")
        
        self.g_masks = torch.ones(1).float().to(self.device)
        
        # Semantic Mapping
        self.sem_map_module = Semantic_Mapping(self.args).to(self.device)
        self.sem_map_module.eval()

        # initializations for planning:
        self.selem = skimage.morphology.disk(3)
        self.stg_selem = skimage.morphology.disk(10)
        
        # Initializing full and local map
        nc = self.args.num_sem_categories + 4  # num channels
        map_size = self.args.map_size_cm // self.args.map_resolution
        self.full_w, self.full_h = map_size, map_size
        self.local_w = int(self.full_w / self.args.global_downscaling)
        self.local_h = int(self.full_h / self.args.global_downscaling)
        self.full_map = torch.zeros(1, nc, self.full_w, self.full_h).float().to(self.device)
        self.local_map = torch.zeros(1, nc, self.local_w, self.local_h).float().to(self.device)
        
        # Episode initializations
        map_shape = (
            args.map_size_cm // args.map_resolution,
            args.map_size_cm // args.map_resolution,
        )
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.col_width = 1
        self.count_forward_actions = 0
        self.curr_loc = [
            args.map_size_cm / 100.0 / 2.0,
            args.map_size_cm / 100.0 / 2.0,
            0.0,
        ]
        self.last_loc = None
        self.last_action = None
        #self.prev_goal_ix = self.active_goal_ix
        self.num_conseq_fwd = None
        if args.seg_interval > 0:
            self.num_conseq_fwd = 0
        
        # Origin of local map
        self.origins = np.zeros((1, 3))

        # Local Map Boundaries
        self.lmb = np.zeros((1, 4)).astype(int)
        
        # Planner pose inputs has 7 dimensions
        # 1-3 store continuous global agent location
        # 4-7 store local map boundaries
        self.planner_pose_inputs = np.zeros((1, 7))

        # Initial full and local pose
        self.full_pose = torch.zeros(1, 3).float().to(self.device)
        self.local_pose = torch.zeros(1, 3).float().to(self.device)
        self.local_pose_old = torch.zeros(1, 3).float().to(self.device)
        self.init_map_and_pose()
        
        self.res = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    (args.frame_height, args.frame_width), interpolation=Image.NEAREST
                ),
            ]
        )
        
        # Global policy
        self.g_policy = RL_Policy(args, args.pf_model_path).to(self.device)
        if self.args.eval:
            self.g_policy.eval()
        self.needs_egocentric_transform = self.g_policy.needs_egocentric_transform
        if self.needs_egocentric_transform:
            print("\n\n=======> Needs egocentric transformation!")
        self.needs_dist_maps = self.args.add_agent2loc_distance or self.args.add_agent2loc_distance_v2
        
        # Global policy observation space
        ngc = 8 + args.num_sem_categories
        es = 2
        g_observation_space = gym.spaces.Box(0, 1, (ngc, self.local_w, self.local_h), dtype="uint8")

        # Global policy action space
        g_action_space = gym.spaces.Box(low=0.0, high=0.99, shape=(2,), dtype=np.float32)
        
        locs = self.local_pose.cpu().numpy()
        self.global_input = torch.zeros(1, ngc, self.local_w, self.local_h)
        self.global_orientation = torch.zeros(1, 1).long()
        self.extras = torch.zeros(1, es)
        
        r, c = locs[0, 1], locs[0, 0]
        loc_r, loc_c = [
            int(r * 100.0 / args.map_resolution),
            int(c * 100.0 / args.map_resolution),
        ]

        self.local_map[0, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2] = 1.0
        self.global_orientation[0] = int((locs[0, 2] + 180.0) / 5.0)

        self.global_input[:, 0:4, :, :] = self.local_map[:, 0:4, :, :].detach()
        self.global_input[:, 4:8, :, :] = nn.MaxPool2d(self.args.global_downscaling)(
            self.full_map[:, 0:4, :, :]
        )
        self.global_input[:, 8:, :, :] = self.local_map[:, 4:, :, :].detach()

        self.extras = torch.zeros(1, es)
        self.extras[:, 0] = self.global_orientation[:, 0]
        
        self.g_rollouts = GlobalRolloutStorage(
            self.args.num_global_steps,
            1,
            g_observation_space.shape,
            g_action_space,
            self.g_policy.rec_state_size,
            es,
        ).to(self.device)
        
        # Get fmm distance from agent in predicted map
        self.planner_inputs = {}
        obs_map = self.local_map[0, 0, :, :].cpu().numpy()
        exp_map = self.local_map[0, 1, :, :].cpu().numpy()
        # set unexplored to navigable by default
        self.planner_inputs["map_pred"] = obs_map * np.rint(exp_map)
        self.planner_inputs["pose_pred"] = self.planner_pose_inputs[0]
        _, self.fmm_dists = self.get_reachability_map(self.planner_inputs)
        
        self.g_rollouts.obs[0].copy_(self.global_input)
        self.g_rollouts.extras[0].copy_(self.extras)

        self.agent_locations = []
        pose_pred = self.planner_pose_inputs[0]
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        map_r, map_c = start_y, start_x
        map_loc = [
            int(map_r * 100.0 / self.args.map_resolution - gx1),
            int(map_c * 100.0 / self.args.map_resolution - gy1),
        ]
        map_loc = pu.threshold_poses(map_loc, self.global_input[0].shape[1:])
        self.agent_locations.append(map_loc)
        
        self.global_goals = None
        self.prev_pfs = None
        self.goal_pf = 0
        
        self.prev_pose = np.array([0., 0.])
        self.prev_angle = np.array([0.])
        self.prev_pose_old = np.array([0., 0.])
        self.prev_angle_old = np.array([0.])
        self.steps_in_collision = 0
        
        self.goal_id_to_cat = {
            0: 'chair',
            1: 'bed',
            2: 'plant',
            3: 'toilet',
            4: 'tv_monitor',
            5: 'sofa'
        }
        self.cat_to_coco = {
            'chair': 0,
            'sofa': 1,
            'plant': 2,
            'bed': 3,
            'toilet': 4,
            'tv_monitor': 5
        }
        self.habitat_to_coco = {x: self.cat_to_coco[self.goal_id_to_cat[x]] for x in self.goal_id_to_cat}
        
        self.agent_positions = []
        self.goal_coords = []
        self.obs_maps = []
        
        
    def reset(self):
        self.frontier_blacklist = []
        self.reached_list = []
        self.current_goal = None
        self.replan = False
        
        args = self.args
        # Initializing full and local map
        nc = self.args.num_sem_categories + 4  # num channels
        map_size = self.args.map_size_cm // self.args.map_resolution
        self.full_w, self.full_h = map_size, map_size
        self.local_w = int(self.full_w / self.args.global_downscaling)
        self.local_h = int(self.full_h / self.args.global_downscaling)
        self.full_map = torch.zeros(1, nc, self.full_w, self.full_h).float().to(self.device)
        self.local_map = torch.zeros(1, nc, self.local_w, self.local_h).float().to(self.device)
        
        # Episode initializations
        map_shape = (
            args.map_size_cm // args.map_resolution,
            args.map_size_cm // args.map_resolution,
        )
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.col_width = 1
        self.count_forward_actions = 0
        self.curr_loc = [
            args.map_size_cm / 100.0 / 2.0,
            args.map_size_cm / 100.0 / 2.0,
            0.0,
        ]
        self.last_loc = None
        self.last_action = None
        #self.prev_goal_ix = self.active_goal_ix
        self.num_conseq_fwd = None
        if args.seg_interval > 0:
            self.num_conseq_fwd = 0
        
        # Origin of local map
        self.origins = np.zeros((1, 3))

        # Local Map Boundaries
        self.lmb = np.zeros((1, 4)).astype(int)
        
        # Planner pose inputs has 7 dimensions
        # 1-3 store continuous global agent location
        # 4-7 store local map boundaries
        self.planner_pose_inputs = np.zeros((1, 7))

        # Initial full and local pose
        self.full_pose = torch.zeros(1, 3).float().to(self.device)
        self.local_pose = torch.zeros(1, 3).float().to(self.device)
        self.local_pose_old = torch.zeros(1, 3).float().to(self.device)
        self.init_map_and_pose()
        
        # Global policy observation space
        ngc = 8 + args.num_sem_categories
        es = 2
        g_observation_space = gym.spaces.Box(0, 1, (ngc, self.local_w, self.local_h), dtype="uint8")

        # Global policy action space
        g_action_space = gym.spaces.Box(low=0.0, high=0.99, shape=(2,), dtype=np.float32)
        
        locs = self.local_pose.cpu().numpy()
        self.global_input = torch.zeros(1, ngc, self.local_w, self.local_h)
        self.global_orientation = torch.zeros(1, 1).long()
        self.extras = torch.zeros(1, es)
        
        r, c = locs[0, 1], locs[0, 0]
        loc_r, loc_c = [
            int(r * 100.0 / args.map_resolution),
            int(c * 100.0 / args.map_resolution),
        ]

        self.local_map[0, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2] = 1.0
        self.global_orientation[0] = int((locs[0, 2] + 180.0) / 5.0)

        self.global_input[:, 0:4, :, :] = self.local_map[:, 0:4, :, :].detach()
        self.global_input[:, 4:8, :, :] = nn.MaxPool2d(self.args.global_downscaling)(
            self.full_map[:, 0:4, :, :]
        )
        self.global_input[:, 8:, :, :] = self.local_map[:, 4:, :, :].detach()

        self.extras = torch.zeros(1, es)
        self.extras[:, 0] = self.global_orientation[:, 0]
        
        self.g_rollouts = GlobalRolloutStorage(
            self.args.num_global_steps,
            1,
            g_observation_space.shape,
            g_action_space,
            self.g_policy.rec_state_size,
            es,
        ).to(self.device)
        self.goal_pf = 0
        
        # Get fmm distance from agent in predicted map
        self.planner_inputs = {}
        obs_map = self.local_map[0, 0, :, :].cpu().numpy()
        exp_map = self.local_map[0, 1, :, :].cpu().numpy()
        # set unexplored to navigable by default
        self.planner_inputs["map_pred"] = obs_map * np.rint(exp_map)
        self.planner_inputs["pose_pred"] = self.planner_pose_inputs[0]
        _, self.fmm_dists = self.get_reachability_map(self.planner_inputs)
        
        self.g_rollouts.obs[0].copy_(self.global_input)
        self.g_rollouts.extras[0].copy_(self.extras)

        self.agent_locations = []
        pose_pred = self.planner_pose_inputs[0]
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        map_r, map_c = start_y, start_x
        map_loc = [
            int(map_r * 100.0 / self.args.map_resolution - gx1),
            int(map_c * 100.0 / self.args.map_resolution - gy1),
        ]
        map_loc = pu.threshold_poses(map_loc, self.global_input[0].shape[1:])
        self.agent_locations.append(map_loc)
        
        self.global_goals = None
        self.prev_pfs = None
        
        self.prev_pose = np.array([0., 0.])
        self.prev_angle = np.array([0.])
        self.prev_pose_old = np.array([0., 0.])
        self.prev_angle_old = np.array([0.])
        self.steps_in_collision = 0
        
        self.agent_positions = []
        self.goal_coords = []
        self.obs_maps = []
        
        self.sem_map_module.agent_views = []
        self.sem_map_module.st_poses = []
        self.sem_map_module.pose_shifts = []
        self.sem_map_module.depths = []
        
        
    def init_map_and_pose(self):
        self.full_map.fill_(0.0)
        self.full_pose.fill_(0.0)
        self.full_pose[:, :2] = self.args.map_size_cm / 100.0 / 2.0

        locs = self.full_pose.cpu().numpy()
        #planner_pose_inputs[:, :3] = locs
        r, c = locs[0, 1], locs[0, 0]
        loc_r, loc_c = [
            int(r * 100.0 / self.args.map_resolution),
            int(c * 100.0 / self.args.map_resolution),
        ]

        self.full_map[0, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2] = 1.0

        self.lmb[0] = self.get_local_map_boundaries(
            (loc_r, loc_c), (self.local_w, self.local_h), (self.full_w, self.full_h)
        )

        self.planner_pose_inputs[0, 3:] = self.lmb[0]
        self.origins[0] = [
            self.lmb[0][2] * self.args.map_resolution / 100.0,
            self.lmb[0][0] * self.args.map_resolution / 100.0,
            0.0,
        ]

        self.local_map[0] = self.full_map[0, :, self.lmb[0, 0] : self.lmb[0, 1], self.lmb[0, 2] : self.lmb[0, 3]]
        self.local_pose[0] = (
            self.full_pose[0] - torch.from_numpy(self.origins[0]).to(self.device).float()
        )
        self.local_pose_old[0] = (
            self.full_pose[0] - torch.from_numpy(self.origins[0]).to(self.device).float()
        )
        
        
    def reset_map(self):
        args = self.args
        self.full_map.fill_(0.0)
        self.local_map.fill_(0.0)
        
        locs = self.full_pose.cpu().numpy()
        #planner_pose_inputs[:, :3] = locs
        r, c = locs[0, 1], locs[0, 0]
        loc_r, loc_c = [
            int(r * 100.0 / self.args.map_resolution),
            int(c * 100.0 / self.args.map_resolution),
        ]

        self.full_map[0, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2] = 1.0
        
        ngc = 8 + args.num_sem_categories
        es = 2
        
        locs = self.local_pose.cpu().numpy()
        self.global_input = torch.zeros(1, ngc, self.local_w, self.local_h)
        self.global_orientation = torch.zeros(1, 1).long()
        self.extras = torch.zeros(1, es)
        
        r, c = locs[0, 1], locs[0, 0]
        loc_r, loc_c = [
            int(r * 100.0 / args.map_resolution),
            int(c * 100.0 / args.map_resolution),
        ]

        self.local_map[0, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2] = 1.0
        self.global_orientation[0] = int((locs[0, 2] + 180.0) / 5.0)

        self.global_input[:, 0:4, :, :] = self.local_map[:, 0:4, :, :].detach()
        self.global_input[:, 4:8, :, :] = nn.MaxPool2d(self.args.global_downscaling)(
            self.full_map[:, 0:4, :, :]
        )
        self.global_input[:, 8:, :, :] = self.local_map[:, 4:, :, :].detach()
        
        
    def get_local_map_boundaries(self, agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        self.local_w, self.local_h = local_sizes
        self.full_w, self.full_h = full_sizes

        if self.args.global_downscaling > 1:
            gx1, gy1 = loc_r - self.local_w // 2, loc_c - self.local_h // 2
            gx2, gy2 = gx1 + self.local_w, gy1 + self.local_h
            if gx1 < 0:
                gx1, gx2 = 0, self.local_w
            if gx2 > self.full_w:
                gx1, gx2 = self.full_w - self.local_w, self.full_w

            if gy1 < 0:
                gy1, gy2 = 0, self.local_h
            if gy2 > self.full_h:
                gy1, gy2 = self.full_h - self.local_h, self.full_h
        else:
            gx1, gx2, gy1, gy2 = 0, self.full_w, 0, self.full_h

        return [gx1, gx2, gy1, gy2]
    
    
    def _get_reachability(self, grid, goal, planning_window):

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = (
            0,
            0,
        )
        x2, y2 = grid.shape

        traversible = 1.0 - cv2.dilate(grid[x1:x2, y1:y2], self.selem)
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        # Note: Unlike _get_stg, no boundary is added here since we only want
        # to determine reachability.
        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(3)
        goal = cv2.dilate(goal, selem)
        planner.set_multi_goal(goal)
        fmm_dist = planner.fmm_dist * self.args.map_resolution / 100.0

        reachability = fmm_dist < fmm_dist.max()

        return reachability.astype(np.float32), fmm_dist.astype(np.float32)
    
    
    def get_reachability_map(self, planner_inputs):
        """Function responsible for planning, and identifying reachable locations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)

        Returns:
            reachability_map (ndarray): (M, M) map of reachable locations
            fmm_dist (ndarray): (M, M) map of geodesic distance
        """
        args = self.args

        # Get agent position + local boundaries
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_inputs["pose_pred"]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]
        # Get map
        map_pred = np.rint(planner_inputs["map_pred"])
        # Convert current location to map coordinates
        r, c = start_y, start_x
        start = [
            int(r * 100.0 / args.map_resolution - gx1),
            int(c * 100.0 / args.map_resolution - gy1),
        ]
        start = pu.threshold_poses(start, map_pred.shape)
        # Create a goal map (start is goal)
        goal_map = np.zeros(map_pred.shape)
        goal_map[start[0] - 0 : start[0] + 1, start[1] - 0 : start[1] + 1] = 1
        # Figure out reachable locations
        reachability, fmm_dist = self._get_reachability(
            map_pred, goal_map, planning_window
        )

        return reachability, fmm_dist
    
    
    def _get_stg(self, grid, start, goal, planning_window):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = (
            0,
            0,
        )
        x2, y2 = grid.shape
        
        selem = skimage.morphology.disk(7)
        traversible = 1.0 - cv2.dilate(grid[x1:x2, y1:y2], selem)
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        traversible[
            int(start[0] - x1) - 1 : int(start[0] - x1) + 2,
            int(start[1] - y1) - 1 : int(start[1] - y1) + 2,
        ] = 1

        traversible = self.add_boundary(traversible)
        goal = self.add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(10)
        goal = cv2.dilate(goal, self.stg_selem)
        planner.set_multi_goal(goal)
        
        #print('Start x and y:', start[0], start[1])
        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        stg_x, stg_y, self.replan, stop = planner.get_short_term_goal(state)
        if self.replan:
            print('Change dilation radius from 7 to 3')
            selem = skimage.morphology.disk(3)
            traversible = 1.0 - cv2.dilate(grid[x1:x2, y1:y2], selem)
            traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
            traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1
            traversible[
                int(start[0] - x1) - 1 : int(start[0] - x1) + 2,
                int(start[1] - y1) - 1 : int(start[1] - y1) + 2,
            ] = 1
            traversible = self.add_boundary(traversible)
            planner = FMMPlanner(traversible)
            planner.set_multi_goal(goal)
            stg_x, stg_y, self.replan, stop = planner.get_short_term_goal(state)
        if self.replan:
            print('FMM PLANNER FAILED!')
        #print('Short-term goal:', stg_x, stg_y)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), stop
    
    
    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        #for i in range(depth.shape[1]):
        #    depth[:, i][depth[:, i] == 0.0] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.0

        mask1 = depth == 0
        depth[mask1] = 1.0
        depth = min_d * 100.0 + depth * (max_d - min_d) * 100.0
        return depth
    
    
    def _preprocess_obs(self, observations, semantic_prediction):
        args = self.args
        rgb = observations['rgb']
        depth = observations['depth']
        depth = depth * (args.max_depth - args.min_depth) + args.min_depth
        depth = (depth - args.min_depth) / (args.max_depth - args.min_depth)
        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)
        depth_old = observations['depth_old']
        depth_old = depth_old * (args.max_depth - args.min_depth) + args.min_depth
        depth_old = (depth_old - args.min_depth) / (args.max_depth - args.min_depth)
        depth_old = self._preprocess_depth(depth_old, args.min_depth, args.max_depth)
        
        ds = args.env_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2 :: ds, ds // 2 :: ds]
            depth_old = depth_old[ds // 2 :: ds, ds // 2 :: ds]
            semantic_prediction = semantic_prediction[ds // 2 :: ds, ds // 2 :: ds]
        semantic_prediction = np.concatenate([semantic_prediction, np.zeros_like(semantic_prediction[:, :, :1])], axis=2)

        depth = np.expand_dims(depth, axis=2)
        depth_old = np.expand_dims(depth_old, axis=2)
        state = np.concatenate((rgb, depth, depth_old, semantic_prediction), axis=2).transpose(2, 0, 1)

        return state
    
    
    def add_boundary(self, mat, value=1):
        h, w = mat.shape
        new_mat = np.zeros((h + 2, w + 2)) + value
        new_mat[1 : h + 1, 1 : w + 1] = mat
        return new_mat

    
    def get_pose_shift(self, pose, prev_pose, angle, prev_angle):
        x_old, y_old = prev_pose
        x_new, y_new = pose
        d_theta = angle[0] - prev_angle[0]
        dx = (x_new - x_old) * np.cos(prev_angle[0]) + (y_new - y_old) * np.sin(prev_angle[0])
        dy = -(x_new - x_old) * np.sin(prev_angle[0]) + (y_new - y_old) * np.cos(prev_angle[0])
        return np.array([dx, dy, d_theta])
    
    
    def update(self, observations, semantic_prediction):
        rgb = observations['rgb']
        pose = observations['gps']#.copy()
        pose[1] *= -1
        pose_old = observations['gps_old']#.copy()
        pose_old[1] *= -1
        angle = observations['compass']
        #print('Pose and angle:', pose, angle)
        angle_old = observations['compass_old']
        objgoal = self.habitat_to_coco[observations['objectgoal'][0]]
        pose_shift = torch.from_numpy(self.get_pose_shift(pose, self.prev_pose, angle, self.prev_angle)[np.newaxis, :]).float().to(self.device)
        #print('Pose shift:', pose_shift)
        old_pose_shift = torch.from_numpy(self.get_pose_shift(pose_old, self.prev_pose_old, angle_old, self.prev_angle_old)[np.newaxis, :]).float().to(self.device)
        self.prev_pose = pose
        self.prev_angle = angle
        self.prev_pose_old = pose_old
        self.prev_angle_old = angle_old
        rgbs = torch.from_numpy(rgb[np.newaxis, ...]).float().to(self.device)
        obs = self._preprocess_obs(observations, semantic_prediction)
        obs = torch.from_numpy(obs[np.newaxis, ...]).float().to(self.device)
        _, self.local_map, _, self.local_pose, self.local_pose_old = self.sem_map_module(obs, pose_shift, old_pose_shift, self.local_map, self.local_pose, self.local_pose_old, objgoal)
        
        locs = self.local_pose.cpu().numpy()
        #print('Local pose:', locs)
        self.planner_pose_inputs[:, :3] = locs + self.origins
        self.local_map[:, 2, :, :].fill_(0.0)  # Resetting current location channel
        r, c = locs[0, 1], locs[0, 0]
        loc_r, loc_c = [
            int(r * 100.0 / self.args.map_resolution),
            int(c * 100.0 / self.args.map_resolution),
        ]
        self.local_map[0, 2:4, loc_r - 2 : loc_r + 3, loc_c - 2 : loc_c + 3] = 1.0
        
        # Take action and get next observation
        self.planner_inputs = {}
        pf_visualizations = None
        if self.args.visualize or self.args.print_images:
            pf_visualizations = g_policy.visualizations
        self.planner_inputs["map_pred"] = self.local_map[0, 0, :, :].cpu().numpy()
        self.planner_inputs["exp_pred"] = self.local_map[0, 1, :, :].cpu().numpy()
        self.planner_inputs["pose_pred"] = self.planner_pose_inputs[0]
        #self.planner_inputs["goal"] = goal_maps[0]  # global_goals[e]
                
        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = self.planner_inputs["pose_pred"]
        self.last_loc = self.curr_loc
        self.curr_loc = [start_x, start_y, start_o]
        map_pred = self.planner_inputs["map_pred"]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]
        r, c = start_y, start_x
        start = [
            int(r * 100.0 / self.args.map_resolution - gx1),
            int(c * 100.0 / self.args.map_resolution - gy1),
        ]
        start = pu.threshold_poses(start, map_pred.shape)
        self.agent_positions.append(start)
        
        if self.global_goals is None:
            self.choose_goal(observations, semantic_prediction)
        
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Update long-term goal if target object is found
        goal_maps = [np.zeros((self.local_w, self.local_h))]

        if not self.g_policy.has_action_output:
            # Ignore goal and use nearest frontier baseline if requested
            if not self.args.use_nearest_frontier:
                if self.global_goals is not None:
                    goal_maps[0][self.global_goals[0][0], self.global_goals[0][1]] = 1
                else:
                    goal_maps[0][:, :] = self.cpu_actions[0]
            else:
                fmap = self.frontier_maps[0].cpu().numpy()
                goal_maps[0][fmap] = 1
        
        goal = (goal_maps[0].nonzero()[0][0], goal_maps[0].nonzero()[1][0])
        #print('Goal:', goal)
        self.goal_coords.append(goal)
        
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Take action and get next observation
        self.planner_inputs = {}
        pf_visualizations = None
        if self.args.visualize or self.args.print_images:
            pf_visualizations = g_policy.visualizations
        self.planner_inputs["map_pred"] = self.local_map[0, 0, :, :].cpu().numpy()
        self.planner_inputs["exp_pred"] = self.local_map[0, 1, :, :].cpu().numpy()
        self.planner_inputs["pose_pred"] = self.planner_pose_inputs[0]
        self.planner_inputs["goal"] = goal_maps[0]  # global_goals[e]
        
        
    def check_and_draw_collisions(self):
        if self.last_loc is None:
            self.last_loc = self.curr_loc
            return
        args = self.args
        x1, y1, t1 = self.last_loc
        x2, y2, _ = self.curr_loc
        #print('x1 y1 x2 y2:', x1, y1, x2, y2)
        buf = 4
        length = 2
        
        if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
            self.col_width += 2
            if self.col_width == 7:
                length = 4
                buf = 3
            self.col_width = min(self.col_width, 3)
        else:
            self.col_width = 1
        #self.col_width = 2

        dist = pu.get_l2_distance(x1, x2, y1, y2)
        if dist < args.collision_threshold:  # Collision
            self.steps_in_collision += 1
            width = self.col_width
            for i in range(length):
                for j in range(width):
                    wx = x1 + 0.05 * (
                        (i + buf) * np.cos(np.deg2rad(t1))
                        + (j - width // 2) * np.sin(np.deg2rad(t1))
                    )
                    wy = y1 + 0.05 * (
                        (i + buf) * np.sin(np.deg2rad(t1))
                        - (j - width // 2) * np.cos(np.deg2rad(t1))
                    )
                    r, c = wy, wx
                    r, c = int(r * 100 / args.map_resolution), int(
                        c * 100 / args.map_resolution
                    )
                    [r, c] = pu.threshold_poses([r, c], self.collision_map.shape)
                    self.collision_map[r, c] = 1
                    #print('Collision_map[{}, {}] = 1'.format(r, c))
        else:
            self.steps_in_collision = 0
        #self.last_loc = self.curr_loc
    
    
    def _plan(self):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        args = self.args
        
        # Get Map prediction
        map_pred = np.rint(self.planner_inputs["map_pred"])
        goal = self.planner_inputs["goal"]

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = self.planner_inputs["pose_pred"]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [
            int(r * 100.0 / args.map_resolution - gx1),
            int(c * 100.0 / args.map_resolution - gy1),
        ]
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][
            start[0] - 0 : start[0] + 1, start[1] - 0 : start[1] + 1
        ] = 1

        # Collision check
        if self.last_action == 1:
            self.check_and_draw_collisions()
        else:
            self.steps_in_collision = 0

        stg, stop = self._get_stg(map_pred, start, np.copy(goal), planning_window)
        self.agent_positions.append(start)

        # Deterministic Local Policy
        (stg_x, stg_y) = stg
        angle_st_goal = math.degrees(math.atan2(stg_x - start[0], stg_y - start[1]))
        angle_agent = (start_o) % 360.0
        if angle_agent > 180:
            angle_agent -= 360

        relative_angle = (angle_agent - angle_st_goal) % 360.0
        if relative_angle > 180:
            relative_angle -= 360

        if relative_angle > self.args.turn_angle / 2.0:
            action = 3  # Right
        elif relative_angle < -self.args.turn_angle / 2.0:
            action = 2  # Left
        else:
            action = 1  # Forward
        self.last_action = action
        
        return action
    
    
    def choose_goal(self, observations, semantic_prediction):
        ########################################################################################
        # Transform to egocentric coordinates if needed
        # Note: The agent needs to be at the center of the map facing right.
        # Conventions: start_x, start_y, start_o are as follows.
        # X -> downward, Y -> rightward, origin (top-left corner of map)
        # O -> measured from Y to X clockwise.
        ########################################################################################
        objgoal = self.habitat_to_coco[observations['objectgoal'][0]]
        print('Choose goal by PONI')
        self.full_map[
            0, :, self.lmb[0, 0] : self.lmb[0, 1], self.lmb[0, 2] : self.lmb[0, 3]
        ] = self.local_map[0]
        self.full_pose[0] = (
            self.local_pose[0] + torch.from_numpy(self.origins[0]).to(self.device).float()
        )

        locs = self.full_pose[0].cpu().numpy()
        r, c = locs[1], locs[0]
        loc_r, loc_c = [
            int(r * 100.0 / self.args.map_resolution),
            int(c * 100.0 / self.args.map_resolution),
        ]

        self.lmb[0] = self.get_local_map_boundaries(
            (loc_r, loc_c), (self.local_w, self.local_h), (self.full_w, self.full_h)
        )

        self.planner_pose_inputs[0, 3:] = self.lmb[0]
        self.origins[0] = [
            self.lmb[0][2] * self.args.map_resolution / 100.0,
            self.lmb[0][0] * self.args.map_resolution / 100.0,
            0.0,
        ]

        self.local_map[0] = self.full_map[
            0, :, self.lmb[0, 0] : self.lmb[0, 1], self.lmb[0, 2] : self.lmb[0, 3]
        ]
        self.local_pose[0] = (
            self.full_pose[0] - torch.from_numpy(self.origins[0]).to(self.device).float()
        )

        locs = self.local_pose.cpu().numpy()
        self.global_orientation[0] = int((locs[0, 2] + 180.0) / 5.0)
        self.global_input[:, 0:4, :, :] = self.local_map[:, 0:4, :, :]
        self.global_input[:, 4:8, :, :] = nn.MaxPool2d(self.args.global_downscaling)(
            self.full_map[:, 0:4, :, :]
        )
        self.global_input[:, 8:, :, :] = self.local_map[:, 4:, :, :].detach()
        self.extras[:, 0] = self.global_orientation[:, 0]
        self.extras[:, 1] = objgoal

        # Get fmm_dists from agent in predicted map
        self.fmm_dists = None
        if self.needs_dist_maps:
            self.planner_inputs = {}
            obs_map = self.local_map[0, 0, :, :].cpu().numpy()
            exp_map = self.local_map[0, 1, :, :].cpu().numpy()
            # set unexplored to navigable by default
            self.planner_inputs["map_pred"] = obs_map * np.rint(exp_map)
            self.planner_inputs["pose_pred"] = self.planner_pose_inputs[0]
            _, self.fmm_dists = self.get_reachability_map(self.planner_inputs)
        
        g_obs = self.global_input.to(self.local_map.device)  # g_rollouts.obs[g_step]
        unk_map = 1.0 - self.local_map[:, 1, :, :]
        ego_agent_poses = None
        if self.needs_egocentric_transform:
            ego_agent_poses = []
            map_loc = self.agent_locations[0]
            # Crop map about a center
            ego_agent_poses.append(
                [map_loc[0], map_loc[1], math.radians(start_o)]
            )
            ego_agent_poses = torch.Tensor(ego_agent_poses).to(g_obs.device)

        # Sample long-term goal from global policy
        g_value, g_action, g_action_log_prob, g_rec_states, self.prev_pfs = self.g_policy.act(
            g_obs,
            None,  # g_rollouts.rec_states[g_step],
            self.g_masks.to(g_obs.device),  # g_rollouts.masks[g_step],
            extras=self.extras.to(g_obs.device),  # g_rollouts.extras[g_step],
            extra_maps={
                "dmap": self.fmm_dists,
                "umap": unk_map,
                "pfs": self.prev_pfs,
                "agent_locations": self.agent_locations,
                "ego_agent_poses": ego_agent_poses,
            },
            deterministic=False,
        )
        
        if not self.g_policy.has_action_output:
            self.cpu_actions = g_action.cpu().numpy()
            if len(self.cpu_actions.shape) == 2:  # (B, 2) XY locations
                self.global_goals = [
                    [int(action[0] * self.local_w), int(action[1] * self.local_h)]
                    for action in self.cpu_actions
                ]
                self.global_goals = [
                    [min(x, int(self.local_w - 1)), min(y, int(self.local_h - 1))]
                    for x, y in self.global_goals
                ]
            else:
                assert len(self.cpu_actions.shape) == 3  # (B, H, W) action maps
                self.global_goals = None
                
        goal_i, goal_j = self.global_goals[0]
        self.goal_pf = self.prev_pfs['area_pfs'][0, 0, goal_i, goal_j]
        goal_x = (goal_j / 100 * self.args.map_resolution - 12)
        goal_y = (goal_i / 100 * self.args.map_resolution - 12)
        self.current_goal = (goal_x, goal_y)
        
        return self.current_goal
    
    
    def accept_goal(self):
        if self.current_goal is not None:
            self.reached_list.append(self.current_goal)
        self.current_goal = None
    
    
    def reject_goal(self):
        if self.current_goal is not None:
            self.frontier_blacklist.append(self.current_goal)
        self.current_goal = None
        
        
    def map_to_world(self, i, j):
        x = j / 100 * self.args.map_resolution - 12
        y = i / 100 * self.args.map_resolution - 12
        return x, y
        
        
    def world_to_map(self, x, y):
        i = (y + 12) / self.args.map_resolution * 100
        j = (x + 12) / self.args.map_resolution * 100
        i, j = int(round(i)), int(round(j))
        return i, j