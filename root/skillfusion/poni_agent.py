import numpy as np
import habitat
import math
import torch
import torch.nn as nn
import sys
import yaml
import gym
import time
import cv2
import skimage
import semexp.envs.utils.pose as pu
from PIL import Image
from parse_args import parse_args_from_config
from habitat.sims.habitat_simulator.actions import HabitatSimActions
sys.path.append('/home/kirill/habitat-lab/exploration_ros_free')
from semexp.agents.utils.semantic_prediction import SemanticPredMaskRCNN
from semexp.model import Semantic_Mapping
from semexp.model_pf import RL_Policy
from semexp.utils.storage import GlobalRolloutStorage
from semexp.envs.utils.fmm_planner import FMMPlanner
from torchvision import transforms


class PoniAgent(habitat.Agent):
    def __init__(self, task_config: habitat.config.DictConfig):
        fin = open('config_poni.yaml', 'r')
        config = yaml.safe_load(fin)
        fin.close()
        print('Config:', config)

        args = parse_args_from_config(config['args'])
        print('ARGS:', args)
        self.args = args
        self.device = torch.device("cuda:0" if args.cuda else "cpu")
        
        self.g_masks = torch.ones(1).float().to(self.device)
        
        self.sem_pred = SemanticPredMaskRCNN(self.args)
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
        
        self.prev_pose = np.array([0., 0.])
        self.prev_angle = np.array([0.])
        
        self.step = 0
        
        self.agent_positions = []
        self.goal_coords = []
        self.obs_maps = []
        
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
        
        
    def reset(self):
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
        
        self.step = 0
        
        self.agent_positions = []
        self.goal_coords = []
        self.obs_maps = []
        
        
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
        
        
    def reset_map(self):
        self.full_map = torch.zeros(1, nc, self.full_w, self.full_h).float().to(self.device)
        self.local_map = torch.zeros(1, nc, self.local_w, self.local_h).float().to(self.device)
        
            
            
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

        start_time = time.time()
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
        planning_time = time.time() - start_time

        return reachability, fmm_dist
    
    
    def get_frontier_map(self, planner_inputs):
        """Function responsible for computing frontiers in the input map

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'obs_map' (ndarray): (M, M) map of obstacle locations
                    'exp_map' (ndarray): (M, M) map of explored locations

        Returns:
            frontier_map (ndarray): (M, M) binary map of frontier locations
        """
        args = self.args

        obs_map = np.rint(planner_inputs["obs_map"])
        exp_map = np.rint(planner_inputs["exp_map"])
        # compute free and unexplored maps
        free_map = (1 - obs_map) * exp_map
        unk_map = 1 - exp_map
        # Clean maps
        kernel = np.ones((5, 5), dtype=np.uint8)
        free_map = cv2.morphologyEx(free_map, cv2.MORPH_CLOSE, kernel)
        unk_map[free_map == 1] = 0
        # https://github.com/facebookresearch/exploring_exploration/blob/09d3f9b8703162fcc0974989e60f8cd5b47d4d39/exploring_exploration/models/frontier_agent.py#L132
        unk_map_shiftup = np.pad(
            unk_map, ((0, 1), (0, 0)), mode="constant", constant_values=0
        )[1:, :]
        unk_map_shiftdown = np.pad(
            unk_map, ((1, 0), (0, 0)), mode="constant", constant_values=0
        )[:-1, :]
        unk_map_shiftleft = np.pad(
            unk_map, ((0, 0), (0, 1)), mode="constant", constant_values=0
        )[:, 1:]
        unk_map_shiftright = np.pad(
            unk_map, ((0, 0), (1, 0)), mode="constant", constant_values=0
        )[:, :-1]
        frontiers = (
            (free_map == unk_map_shiftup)
            | (free_map == unk_map_shiftdown)
            | (free_map == unk_map_shiftleft)
            | (free_map == unk_map_shiftright)
        ) & (
            free_map == 1
        )  # (H, W)
        frontiers = frontiers.astype(np.uint8)
        # Select only large-enough frontiers
        contours, _ = cv2.findContours(
            frontiers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if len(contours) > 0:
            contours = [c[:, 0].tolist() for c in contours]  # Clean format
            new_frontiers = np.zeros_like(frontiers)
            # Only pick largest 5 frontiers
            contours = sorted(contours, key=lambda x: len(x), reverse=True)
            for contour in contours[:5]:
                contour = np.array(contour)
                # Select only the central point of the contour
                lc = len(contour)
                if lc > 0:
                    new_frontiers[contour[lc // 2, 1], contour[lc // 2, 0]] = 1
            frontiers = new_frontiers
        frontiers = frontiers > 0
        # Mask out frontiers very close to the agent
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_inputs["pose_pred"]
        ## Convert current location to map coordinates
        r, c = start_y, start_x
        start = [
            int(r * 100.0 / args.map_resolution - gx1),
            int(c * 100.0 / args.map_resolution - gy1),
        ]
        start = pu.threshold_poses(start, frontiers.shape)
        ## Mask out a 100.0 x 100.0 cm region center on the agent
        ncells = int(100.0 / args.map_resolution)
        frontiers[
            (start[0] - ncells) : (start[0] + ncells + 1),
            (start[1] - ncells) : (start[1] + ncells + 1),
        ] = False
        # Handle edge case where frontier becomes zero
        if not np.any(frontiers):
            # Set a random location to True
            rand_y = np.random.randint(start[0] - ncells, start[0] + ncells + 1)
            rand_x = np.random.randint(start[1] - ncells, start[1] + ncells + 1)
            frontiers[rand_y, rand_x] = True

        return frontiers
    

    def _get_sem_pred(self, rgb, use_seg=True):
        if use_seg:
            semantic_pred, self.rgb_vis = self.sem_pred.get_prediction(rgb)
            semantic_pred = semantic_pred.astype(np.float32)
        else:
            if self.zero_sem_seg is None:
                self.zero_sem_seg = np.zeros((rgb.shape[0], rgb.shape[1], 16))
            semantic_pred = self.zero_sem_seg
            self.rgb_vis = rgb[:, :, ::-1]
        return semantic_pred
    
    
    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.0] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.0

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_d * 100.0 + depth * max_d * 100.0
        return depth
    
    
    def add_boundary(self, mat, value=1):
        h, w = mat.shape
        new_mat = np.zeros((h + 2, w + 2)) + value
        new_mat[1 : h + 1, 1 : w + 1] = mat
        return new_mat
    
    
    def _get_stg(self, grid, start, goal, planning_window):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = (
            0,
            0,
        )
        x2, y2 = grid.shape

        traversible = 1.0 - cv2.dilate(grid[x1:x2, y1:y2], self.selem)
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        traversible[
            int(start[0] - x1) - 1 : int(start[0] - x1) + 2,
            int(start[1] - y1) - 1 : int(start[1] - y1) + 2,
        ] = 1

        traversible = self.add_boundary(traversible)
        goal = self.add_boundary(goal, value=0)

        # goal = cv2.dilate(goal, self.stg_selem)

        # step_size = 5
        # stg_x, stg_y = None, None
        # obstacles = (1 - traversible).astype(np.float32)
        # astar_goal = goal.astype(np.float32)
        # astar_start = [int(start[1] - y1 + 1), int(start[0] - x1 + 1)]
        # path_y, path_x = pyastar.multi_goal_astar_planner(
        #     obstacles, astar_start, astar_goal, True
        # )
        # if path_x is not None:
        #     # The paths are in reversed order
        #     stg_x = path_x[-min(step_size, len(path_x))]
        #     stg_y = path_y[-min(step_size, len(path_y))]
        #     stop = False
        #     if len(path_x) < step_size:
        #         # Measure distance along the shortest path
        #         path_xy = np.stack([path_x, path_y], axis=1)
        #         d2g = np.linalg.norm(path_xy[1:] - path_xy[:-1], axis=1)
        #         d2g = d2g.sum() * self.args.map_resolution / 100.0 # In meters
        #         if d2g <= 0.25:
        #             stop = True
        #             print(f'=======> Estimated DTS: {d2g:.2f}')

        # if stg_x is None:
        #     # Pick some arbitrary location as the short-term goal
        #     random_theta = np.random.uniform(-np.pi, np.pi, (1, ))[0].item()
        #     stg_x = int(step_size * np.cos(random_theta))
        #     stg_y = int(step_size * np.sin(random_theta))
        #     stop = False

        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(10)
        goal = cv2.dilate(goal, self.stg_selem)
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        stg_x, stg_y, _, stop = planner.get_short_term_goal(state)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), stop
    
    
    def _plan(self, planner_inputs):
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

        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(planner_inputs["map_pred"])
        goal = planner_inputs["goal"]

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_inputs["pose_pred"]
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

        if args.visualize or args.print_images:
            # Get last loc
            last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
            r, c = last_start_y, last_start_x
            last_start = [
                int(r * 100.0 / args.map_resolution - gx1),
                int(c * 100.0 / args.map_resolution - gy1),
            ]
            last_start = pu.threshold_poses(last_start, map_pred.shape)
            self.visited_vis[gx1:gx2, gy1:gy2] = vu.draw_line(
                last_start, start, self.visited_vis[gx1:gx2, gy1:gy2]
            )

        # Collision check
        if self.last_action == 1:
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold:  # Collision
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

        stg, stop = self._get_stg(map_pred, start, np.copy(goal), planning_window)
        self.agent_positions.append(start)

        # Deterministic Local Policy
        if stop and planner_inputs["found_goal"] == 1:
            action = 0  # Stop
        else:
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

        return action
    
    
    def _preprocess_obs(self, obs, use_seg=True):
        args = self.args
        #obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]
        depth = (depth - args.min_depth) / (args.max_depth - args.min_depth)
        
        if args.use_gt_segmentation:
            semantic_category = obs[:, :, 4]
            sem_seg_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
            kernel = np.ones((10, 10))
            for i in range(0, sem_seg_pred.shape[2]):
                cat_img = (semantic_category == i).astype(np.float32)
                cat_img = cv2.erode(cat_img, kernel)
                # Fixes a bug in rendering where it semantics are vertically flipped
                sem_seg_pred[..., i] = cat_img
            self.rgb_vis = rgb[:, :, ::-1]
        else:
            sem_seg_pred = self._get_sem_pred(rgb.astype(np.uint8), use_seg=use_seg)
        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

        ds = args.env_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2 :: ds, ds // 2 :: ds]
            sem_seg_pred = sem_seg_pred[ds // 2 :: ds, ds // 2 :: ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred), axis=2).transpose(2, 0, 1)

        return state
    
    
    def get_pose_shift(self, pose, prev_pose, angle, prev_angle):
        x_old, y_old = prev_pose
        x_new, y_new = pose
        d_theta = angle[0] - prev_angle[0]
        dx = (x_new - x_old) * np.cos(prev_angle[0]) + (y_new - y_old) * np.sin(prev_angle[0])
        dy = -(x_new - x_old) * np.sin(prev_angle[0]) + (y_new - y_old) * np.cos(prev_angle[0])
        return np.array([dx, dy, d_theta])


    def act(self, observations, use_seg=True):
        rgb = observations['rgb']
        depth = observations['depth']
        pose = observations['gps']
        pose[1] *= -1
        angle = observations['compass']
        objgoal = self.habitat_to_coco[observations['objectgoal'][0]]
        pose_shift = torch.from_numpy(self.get_pose_shift(pose, self.prev_pose, angle, self.prev_angle)[np.newaxis, :]).float().to(self.device)
        #print('GLOBAL POSE:', pose)
        #print('GLOBAL ANGLE:', angle)
        #print('POSE SHIFT:', pose_shift)
        self.prev_pose = pose
        self.prev_angle = angle
        rgbs = torch.from_numpy(rgb[np.newaxis, ...]).float().to(self.device)
        obs = np.concatenate((rgb, depth), axis=2)
        if self.args.use_gt_segmentation:
            obs = np.concatenate((obs, observations['semantic']), axis=2)
        obs = self._preprocess_obs(obs)
        obs = torch.from_numpy(obs[np.newaxis, ...]).float().to(self.device)
        semantic_mask = obs.cpu().numpy().max(axis=-1).min(axis=-1)[0, objgoal + 4]
        if semantic_mask.max() > 0:
            print('OBJECTGOAL IS OBSERVED!')
        _, self.local_map, _, self.local_pose = self.sem_map_module(obs, pose_shift, self.local_map, self.local_pose)
        semantic_map = self.local_map[0, objgoal + 4, :, :].cpu().numpy()
        #skimage.io.imsave('local_maps/local_map.png', semantic_map)
        
        # ------------------------------------------------------------------
        # Reinitialize variables when episode ends
        #l_masks = torch.FloatTensor([0 if x else 1 for x in done]).to(self.device)
        self.g_masks *= 1#l_masks
        
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

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Global Policy
        if self.step % self.args.num_local_steps == 0:
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
            
            ########################################################################################
            # Transform to egocentric coordinates if needed
            # Note: The agent needs to be at the center of the map facing right.
            # Conventions: start_x, start_y, start_o are as follows.
            # X -> downward, Y -> rightward, origin (top-left corner of map)
            # O -> measured from Y to X clockwise.
            ########################################################################################
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
                    
            self.g_masks = torch.ones(1).float().to(self.device)
                    
            # Compute frontiers if needed
            if self.args.use_nearest_frontier:
                print('UPDATE FRONTIER MAP')
                self.planner_inputs = {}
                obs_map = self.local_map[0, 0, :, :].cpu().numpy()
                exp_map = self.local_map[0, 1, :, :].cpu().numpy()
                self.planner_inputs["obs_map"] = obs_map
                self.planner_inputs["exp_map"] = exp_map
                self.planner_inputs["pose_pred"] = self.planner_pose_inputs[0]
                self.frontier_maps = [self.get_frontier_map(self.planner_inputs)]                
            
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Update long-term goal if target object is found
        found_goal = [0]
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
                # #================ Visualize for debugging ======================
                # kernel = np.ones((5, 5), dtype=np.uint8)
                # fmap = cv2.morphologyEx((fmap * 255.0).astype(np.uint8), cv2.MORPH_DILATE, kernel)
                # obs_map = np.rint(planner_inputs[e]['obs_map'])
                # exp_map = np.rint(planner_inputs[e]['exp_map'])
                # vis_map = np.zeros((*obs_map.shape, 3), dtype=np.uint8)
                # # Green is free
                # vis_map[:, :, 1] = (((1 - obs_map) * exp_map) * 255.0).astype(np.uint8)
                # # Blue is obstacles
                # vis_map[:, :, 0] = (obs_map * exp_map * 255.0).astype(np.uint8)
                # # Red is frontier
                # vis_map[:, :, 2] = fmap
                # vis_map[fmap > 0, 1]  = 0
                # cv2.imshow("Frontier map", vis_map)
                # cv2.waitKey(0)
        
        goal = (goal_maps[0].nonzero()[0][0], goal_maps[0].nonzero()[1][0])
        self.goal_coords.append(goal)

        cn = objgoal + 4
        cat_semantic_map = self.local_map[0, cn, :, :]

        if cat_semantic_map.sum() != 0.0:
            cat_semantic_map = cat_semantic_map.cpu().numpy()
            cat_semantic_scores = cat_semantic_map
            cat_semantic_scores[cat_semantic_scores > 0] = 1.0
            goal_maps[0] = cat_semantic_scores
            found_goal[0] = 1
            print('FOUND GOAL!')
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Take action and get next observation
        self.planner_inputs = {}
        pf_visualizations = None
        if self.args.visualize or self.args.print_images:
            pf_visualizations = g_policy.visualizations
        self.planner_inputs["map_pred"] = self.local_map[0, 0, :, :].cpu().numpy()
        self.planner_inputs["exp_pred"] = self.local_map[0, 1, :, :].cpu().numpy()
        self.obs_maps.append(self.planner_inputs["map_pred"])
        self.planner_inputs["pose_pred"] = self.planner_pose_inputs[0]
        self.planner_inputs["goal"] = goal_maps[0]  # global_goals[e]
        self.planner_inputs["new_goal"] = (self.step % self.args.num_local_steps == 0)
        self.planner_inputs["found_goal"] = found_goal[0]
        if self.g_policy.has_action_output:
            self.planner_inputs["atomic_action"] = g_action[0]
        if self.args.visualize or self.args.print_images:
            self.local_map[0, -1, :, :] = 1e-5
            self.planner_inputs["sem_map_pred"] = self.local_map[0, 4:, :, :].argmax(0).cpu().numpy()
            self.planner_inputs["pf_pred"] = pf_visualizations[0]
            obs[0, -1, :, :] = 1e-5
            self.planner_inputs["sem_seg"] = obs[0, 4:].argmax(0).cpu().numpy()
            
        self.step += 1
                
        if "atomic_action" in self.planner_inputs and not self.planner_inputs["found_goal"]:
            action = self.planner_inputs["atomic_action"]
            print('Atomic action', action)
        else:
            action = self._plan(self.planner_inputs)
            print('Planner action', action)
        if action >= 0:
            if self.args.seg_interval > 0:
                use_seg = True if self.num_conseq_fwd == 0 else False
                self.num_conseq_fwd = (self.num_conseq_fwd + 1) % self.args.seg_interval
                if action != 1:  # not forward
                    use_seg = True
                    self.num_conseq_fwd = 0
            else:
                use_seg = True
            self.last_action = action
            return action

        else:
            self.last_action = None
            assert(False)
            return 0