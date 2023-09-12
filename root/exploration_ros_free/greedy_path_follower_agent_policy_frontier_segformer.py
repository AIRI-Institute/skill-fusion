import numpy as np
import habitat
import keyboard
from habitat.config import DictConfig
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from exploration.frontier.frontier_exploration import FrontierExploration
from exploration.frontier.poni_exploration import PoniExploration
from planners.astar.astar_planner import AStarPlanner
from planners.theta_star.theta_star_planner import ThetaStarPlanner
#from publishers import HabitatObservationPublisher
from semantic_predictor_segformer_multicat import SemanticPredictor
from habitat_map.mapper import Mapper
import yaml

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
from gym.spaces import Box, Dict, Discrete

from EXPLORE_policy import ResNetPolicy as EX_Policy
from GR_policy import PointNavResNetPolicy as GR_Policy

from resnet_policy import PointNavResNetPolicy
from hab_base_utils_common import batch_obs
from resnet_policy_fusion import ResNetPolicy as Policy_3fusion

import gc
from parse_args import parse_args_from_config

DEFAULT_TIMEOUT_VALUE = 2


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


class GreedyPathFollowerAgent(habitat.Agent):

    def __init__(self, task_config: DictConfig):
        fin = open('config_poni_exploration.yaml', 'r')
        config = yaml.safe_load(fin)
        fin.close()
        self.config = config

        # Initialize visualization
        visualization_config = config['visualization']
        self.visualize = visualization_config['visualize']
        if self.visualize:
            rgb_topic = visualization_config['rgb_topic']
            depth_topic = visualization_config['depth_topic']
            true_pose_topic = visualization_config['true_pose_topic']
            map_topic = visualization_config['map_topic']
            semantic_map_topic = visualization_config['semantic_map_topic']
            goal_topic = visualization_config['goal_topic']
            path_topic = visualization_config['path_topic']
            #self.publisher = HabitatObservationPublisher(rgb_topic=rgb_topic, 
            #                                            depth_topic=depth_topic, 
            #                                            true_pose_topic=true_pose_topic,
            #                                            map_topic=map_topic,
            #                                            semantic_map_topic=semantic_map_topic,
            #                                            goal_topic=goal_topic,
            #                                            path_topic=path_topic)

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
                             map_size_cm=mapper_config['map_size_cm'],
                             map_resolution=mapper_config['map_resolution_cm'],
                             semantic_threshold=mapper_config['semantic_threshold'],
                             semantic_decay=mapper_config['semantic_decay'])

        # Initialize semantic predictor
        semantic_config = config['semantic']
        self.semantic_predictor = SemanticPredictor()#model=semantic_config['model'],
                                                    #threshold=semantic_config['confidence_threshold'])

        # Initialize exploration
        exploration_config = config['exploration']
        if exploration_config['type'] == 'frontier':
            self.exploration = FrontierExploration(self.mapper,
                                           potential_scale=exploration_config['potential_scale'],
                                           orientation_scale=exploration_config['orientation_scale'],
                                           gain_scale=exploration_config['gain_scale'],
                                           min_frontier_size=exploration_config['min_frontier_size'])
        elif exploration_config['type'] == 'poni':
            args = parse_args_from_config(config['args'])
            self.exploration = PoniExploration(args)
        else:
            print('UNKNOWN EXPLORATION TYPE!!!')
            self.exploration = None
        self.timeout = exploration_config['timeout']
        self.gr_timeout = exploration_config['gr_timeout']

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

        # Initialize RL params
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

        # Initialize RL agent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obs_space1 = dotdict()
        obs_space1.spaces = {}
        obs_space1.spaces['rgb'] = Box(low=0, high=0, shape=(3,240,320), dtype=np.uint8)
        obs_space1.spaces['depth'] = Box(low=0, high=0, shape=(1,240,320), dtype=np.uint8)
        obs_space1.spaces['pointgoal_with_gps_compass'] = Box(low=0, high=0, shape=(2,), dtype=np.uint8)
        obs_space1.spaces['task_id'] = Box(low=0, high=1, shape=(1,), dtype=np.uint8)
        act_space = dotdict()
        act_space.n = 4
        act_space.shape = [1]
        """
        self.actor_critic_3fusion = Policy_3fusion(
                    observation_space=obs_space1,
                    action_space=act_space,
                    hidden_size=512,
                    resnet_baseplanes=64,
                    rnn_type='LSTM',
                    num_recurrent_layers=1,
                    backbone='resnet18',
                )
        pretrained_state = torch.load('weights/explore_may7_4_state_dict_only.pth', map_location="cpu")
        pretrained_state = {k[len("actor_critic.") :]: v for k, v in pretrained_state["state_dict"].items()}
        """
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
        self.actor_critic_3fusion.load_state_dict(pretrained_state)
        self.actor_critic_3fusion.to(self.device)
        self.actor_critic_3fusion.eval()

        obs_space2 = dotdict()
        obs_space2.spaces = {}
        obs_space2.spaces['rgb'] = Box(low=-1000, high=1000, shape=(240,320,3), dtype=np.float32)
        obs_space2.spaces['depth'] = Box(low=-1000, high=1000, shape=(240,320,1), dtype=np.float32)
        obs_space2.spaces['semantic'] = Box(low=-1000, high=1000, shape=(240,320,1), dtype=np.float32)
        #act_space2 = dotdict()
        #act_space2.n = 4
        #act_space2.shape = [1]
        """
        act_space2 = Discrete(4)
        self.actor_critic_gr = PointNavResNetPolicy(
            observation_space = obs_space2,
            action_space = act_space2,
            hidden_size = 512,
            rnn_type = 'GRU',
            num_recurrent_layers = 1,
            backbone = 'resnet18',
            normalize_visual_inputs=True)
        pretrained_state = torch.load('weights/goalreacher_190_state_dict_only.pth', map_location="cpu")
        pretrained_state = {k[len("actor_critic.") :]: v for k, v in pretrained_state["state_dict"].items()}     
        """
        act_space2 = Discrete(5)
        self.actor_critic_gr = GR_Policy(
            observation_space = obs_space2,
            action_space = act_space2,
            hidden_size = 512,
            rnn_type = 'GRU',
            num_recurrent_layers = 1,
            backbone = 'resnet18',
            normalize_visual_inputs=True)
        pretrained_state = torch.load('/root/exploration_ros_free/weights/goalreacher_model_4.pth', map_location="cpu")
        self.actor_critic_gr.load_state_dict(pretrained_state)
        self.actor_critic_gr.to(self.device)
        self.actor_critic_gr.eval()

        # Initialize common params
        self.step = 0
        self.stuck = False
        self.steps_after_stuck = 1000
        self.escape_from_stuck = False
        self.rl_escape_steps = 0
        self.steps_in_stuck = 0
        self.robot_pose_track = []
        self.action_track = []
        self.maps = []
        self.rgbs = []
        self.goal_coords = []
        self.depths = []
        self.obs_maps = []
        self.agent_positions = []
        self.goal_coords_ij = []
        self.paths = []
        self.goal = None
        self.path = None
        self.objgoal_found = False
        self.step_to_goal = 0
        self.critic_values = []

        self.goal_cost_threshold = 15
        self.goal_cost_scale = 0.1
        self.g_value = 0


    def goal_reached(self, observations):
        x, y = observations['gps']
        y_cell, x_cell = self.mapper.world_to_map(x, y)
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


    def reset(self):
        # RL params
        self.test_recurrent_hidden_states_gr = torch.zeros(1, self.actor_critic_gr.net.num_recurrent_layers, 512, device=self.device)
        self.not_done_masks = torch.zeros(1, 1, device=self.device)
        self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)     
        self.recurrent_hidden_states = torch.zeros(1, 2, 512, device=self.device)
        self.not_done_masks0 = torch.zeros(1, 1, device=self.device)
        self.prev_actions0 = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        
        self.skil_goalreacher = False
        self.dstance_to_object = 10.

        # Common params
        self.step = 0
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
        self.exploration.reset()
        self.path_planner.reset()
        self.mapper.reset()
        self.goal = None
        self.path = None
        self.objgoal_found = False
        self.stuck = False
        self.steps_after_stuck = 1000
        self.steps_in_stuck = 0
        self.escape_from_stuck = False
        self.rl_escape_steps = 0
        self.step_to_goal = 0
        self.step_gr = 1000
        self.steps_wo_goal = 0
        self.critic_values = []
        self.g_value = 0
        
        gc.collect()


    def create_path_by_planner(self, observations):
        robot_x, robot_y = observations['gps']
        #robot_y = -robot_y
        # Create path to goal. If path planning failed, create path along both free and unknown map cells
        if self.goal is not None and (self.path is None or self.step % self.planner_frequency == 0 or self.stuck):
            self.path = self.path_planner.create_path(observations, self.goal, unknown_is_obstacle=True)
            old_reach_radius = self.path_planner.reach_radius
            if self.objgoal_found and self.path is None:
                print('Path to goal with radius 0.5 not found. Try expand reach radius to 0.7')
                self.path_planner.reach_radius = 0.7
                self.path = self.path_planner.create_path(observations, self.goal, unknown_is_obstacle=True)
            if self.path is None:
                print('Path by free cells not found. Trying with unknown cells')
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
            
            
    def get_path_following_action(self, observations):
        robot_x, robot_y = observations['gps']
        #robot_y = -robot_y
        robot_angle = observations['compass'][0]
        self.create_path_by_planner(observations)
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


    def act_fbe(self, observations, semantic_prediction):
        robot_x, robot_y = observations['gps']
        #robot_y = -robot_y
        print('Robot x and y:', robot_x, robot_y)
        robot_angle = observations['compass'][0]

        # at start, let's rotate 360 degrees to look around
        if self.step < 12:
            return HabitatSimActions.turn_left
        
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
            print('Objectgoal found!')
            semantic_map = self.mapper.semantic_map
            i, j = (semantic_map > 0).nonzero()
            if len(i) > 0:
                y_cell, x_cell = self.mapper.world_to_map(robot_x, robot_y)
                dst = np.sqrt((y_cell - i) ** 2 + (x_cell - j) ** 2)
                min_id = dst.argmin()
                self.goal = self.mapper.map_to_world(i[min_id], j[min_id])
                self.create_path_by_planner(observations)
                return self.get_path_following_action(observations)                
            
        # If we have no goal, let's find one
        if self.goal is None:
            self.step_to_goal = 0
            self.goal = self.exploration.choose_goal(observations, semantic_prediction)
            
        self.step_to_goal += 1
        if self.goal is not None and self.interm_goal_reached(observations, self.goal):
            print('Goal reached!')
            self.exploration.accept_goal()
            self.goal = None
            self.step_to_goal = 0
        if self.step_to_goal >= self.timeout:
            print('Goal timed out!')
            self.exploration.reject_goal()
            self.goal = None
            self.step_to_goal = 0
        return self.exploration._plan()
        

    def act_rl(self, observations):
        dpth = observations['depth']
        dpth = (dpth - 0.5) / 4.5
        dpth[dpth == 0] = 1.
        rgb_trans = (self.p240(observations['rgb']).permute(1,2,0)*255)
        depth_trans = self.p240(dpth).permute(1,2,0)
        batch = batch_obs([{'task_id':np.ones((1,))*1,
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
            self.critic_values.append(values.item())    
            self.not_done_masks.fill_(1.0) 
            if actions!=0:
                actions = actions['actions']
            self.prev_actions.copy_(actions)
        action = actions.item()
        return action


    def act_rl_goalreacher(self, observations, obs_semantic):
        dpth = observations['depth']
        dpth = (dpth - 0.5) / 4.5
        dpth[dpth == 0] = 1.
        rgb_trans = (self.p240(observations['rgb']).permute(1,2,0)*255)
        depth_trans = self.p240(dpth).permute(1,2,0)
        sem_trans = self.p240(torch.tensor(obs_semantic)[:,:,0]).permute(1,2,0).int()
        batch = batch_obs([{'rgb':rgb_trans,
                            'depth':depth_trans,
                            'semantic':sem_trans}], device=self.device)
        with torch.no_grad():
            #_, action, _, self.test_recurrent_hidden_states_gr = self.actor_critic_gr.act(
            print(self.prev_actions0.shape)
            policy_action_data = self.actor_critic_gr.act(
                batch,
                self.test_recurrent_hidden_states_gr,
                self.prev_actions0,
                self.not_done_masks0.byte(),
                deterministic=False)
            action = policy_action_data.actions
            print('Action:', action)
            self.test_recurrent_hidden_states_gr = policy_action_data.rnn_hidden_states
            self.not_done_masks0.fill_(1.0)
            self.prev_actions0.copy_(action) 
            action = action.item() 
            
        return action
        

    def act(self, observations):
        robot_x, robot_y = observations['gps']
        self.robot_pose_track.append((robot_x, robot_y))
        self.maps.append(self.mapper.mapper.map[:, :, 1])
        action_rl = self.act_rl(observations)
        if observations['objectgoal'][0] in [2, 4]:
            self.mapper.erosion_size = 2
        else:
            self.mapper.erosion_size = 3
        #if observations['objectgoal'][0] in [2, 3]:
        #    self.mapper.semantic_threshold = 1
        #else:
        #    self.mapper.semantic_threshold = 2

        # Update obstacle map and semantic map
        semantic_prediction, semantic_mask = self.semantic_predictor(observations['rgb'], observations['objectgoal'][0])
        self.mapper.step(observations, semantic_mask)
        self.exploration.update(observations, semantic_prediction)
        if np.sum(semantic_mask) > 300 or self.mapper.semantic_map.max() > 0:
            self.skil_goalreacher = True
            if self.step_gr >= self.gr_timeout or self.mapper.semantic_map.max() > 0:
                self.step_gr = 0
        else:
            self.steps_wo_goal += 1
            if self.step_gr >= self.gr_timeout or self.steps_wo_goal > 5:
                self.skil_goalreacher = False
                self.steps_wo_goal = 0
        if self.skil_goalreacher:
            action_rl_gr = self.act_rl_goalreacher(observations, semantic_mask[np.newaxis, ...])

        observations['rgb'][semantic_mask > 0] = [255, 0, 0]
        self.rgbs.append(observations['rgb'].astype(np.uint8))
        self.depths.append(observations['depth'])

        # Detect stuck
        self.stuck = False
        if len(self.action_track) > 0 and \
           int(self.action_track[-1][0]) == HabitatSimActions.move_forward and \
           len(self.robot_pose_track) > 1 \
           and np.sqrt((self.robot_pose_track[-1][0] - self.robot_pose_track[-2][0]) ** 2 + \
                       (self.robot_pose_track[-1][1] - self.robot_pose_track[-2][1]) ** 2) < 0.01:
            self.stuck = True
        self.steps_after_stuck += 1
        if len(self.robot_pose_track) > 1 and np.sqrt((self.robot_pose_track[-1][0] - self.robot_pose_track[-2][0]) ** 2 + \
                       (self.robot_pose_track[-1][1] - self.robot_pose_track[-2][1]) ** 2) < 0.01:
            self.steps_after_stuck = 0
            self.steps_in_stuck += 1
        else:
            self.steps_in_stuck = 0
        print('Steps in stuck:', self.steps_in_stuck)
        if self.steps_in_stuck > 5:
            self.escape_from_stuck = True
            self.rl_escape_steps = 0
        self.rl_escape_steps += 1
        if self.rl_escape_steps > 10 or (self.skil_goalreacher and self.rl_escape_steps > 5):
            self.escape_from_stuck = False

        # If we stuck, draw obstacle
        if self.stuck:
            self.mapper.draw_obstacle_ahead(observations)

        # Visualize map, goal, and path
        #if self.visualize:
        #    self.publisher.publish(observations, self.mapper, self.goal, self.path)
        
        if self.goal is not None:
            self.goal_coords.append(self.goal)
        else:
            self.goal_coords.append((-1000, -1000))
        if self.exploration.global_goals is not None:
            self.goal_coords_ij.append(self.exploration.global_goals[0])
        else:
            self.goal_coords_ij.append((240, 240))
        if len(self.exploration.obs_maps) == 0:
            self.obs_maps.append(np.zeros((480, 480)))
        else:
            self.obs_maps.append(self.exploration.obs_maps[-1])
        if len(self.exploration.agent_positions) == 0:
            self.agent_positions.append([240, 240])
        else:
            self.agent_positions.append(self.exploration.agent_positions[-1])

        # If we reached objectgoal, stop
        if self.goal_reached(observations):
            print('GOAL REACHED. FINISH EPISODE!')
            self.action_track.append(('0', 'FINISH'))
            return HabitatSimActions.stop

        # if arrow keys pressed, give control to keyboard
        """
        if keyboard.is_pressed('left'):
            self.action_track.append(HabitatSimActions.TURN_LEFT)
            return HabitatSimActions.TURN_LEFT
        if keyboard.is_pressed('right'):
            self.action_track.append(HabitatSimActions.TURN_RIGHT)
            return HabitatSimActions.TURN_RIGHT
        if keyboard.is_pressed('up'):
            self.action_track.append(HabitatSimActions.MOVE_FORWARD)
            return HabitatSimActions.MOVE_FORWARD
        """

        self.step += 1
        #print('Critic value:', self.critic_values[-1])
        #print('Poni value:', self.exploration.goal_pf)
        #print('Exploration cost:', self.exploration.goal_cost)
        #print('Values:', (self.goal_cost_threshold - self.exploration.goal_cost) * self.goal_cost_scale, self.critic_values[-1])
        #if self.exploration.goal_pf < self.critic_values[-1] / 3 and not self.skil_goalreacher:
        if self.critic_values[-1] > 3 and not self.skil_goalreacher:
        #if self.step < 150 and not self.skil_goalreacher:
            self.action_track.append((str(action_rl), 'RL'))
            print('Action RL:', action_rl)
            return action_rl
        if self.escape_from_stuck:
            if self.skil_goalreacher and action_rl_gr != HabitatSimActions.stop:
                print('Step with RL goalreacher to escape')
                self.action_track.append((str(action_rl_gr), 'RL Goalreacher'))
                self.step_gr += 1
                print('Action RL Goalreacher:', action_rl_gr)
                return action_rl_gr
            else:
                self.action_track.append((str(action_rl), 'RL'))
                print('Action RL escape:', action_rl)
                return action_rl
        if self.path is not None:
            self.paths.append(self.path)
        else:
            self.paths.append([])
        if self.skil_goalreacher and \
           self.step_gr < self.gr_timeout and \
           self.mapper.semantic_map.max() == 0 and \
           self.steps_after_stuck > 2 and \
           action_rl_gr != HabitatSimActions.stop:
            print('Step with RL goalreacher')
            self.action_track.append((str(action_rl_gr), 'RL Goalreacher'))
            self.step_gr += 1
            print('Action RL Goalreacher:', action_rl_gr)
            return action_rl_gr
        action_fbe = self.act_fbe(observations, semantic_prediction)
        if self.mapper.semantic_map.max() == 0:
            action_type = 'PONI'
        else:
            action_type = 'FBE Goalreacher'
        self.action_track.append((str(action_fbe), action_type))
        print('Action FBE:', action_fbe)
        return action_fbe
        if action_rl != HabitatSimActions.stop:
            self.action_track.append((str(action_rl), 'RL'))
            print('Action RL after all:', action_rl)
            return action_rl
        action = np.random.choice([HabitatSimActions.move_forward, HabitatSimActions.turn_left])
        self.action_track.append((str(action), 'Random'))
        print('Random action:', action)
        return action