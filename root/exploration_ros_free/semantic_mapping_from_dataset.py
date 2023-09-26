import numpy as np
from PIL import Image
import yaml
import cv2

import os
import sys
sys.path.append('/root/exploration_ros_free/habitat_map')
from tqdm import tqdm
from scipy.signal import convolve2d

from exploration.frontier.poni_exploration import PoniExploration
from parse_args import parse_args_from_config
from habitat_map.mapper import Mapper

meta_clss = {0: 'wall',
 1: 'building',
 2: 'sky',
 3: 'floor',
 4: 'tree',
 5: 'ceiling',
 6: 'road, route',
 7: 'bed',
 8: 'window ',
 9: 'grass',
 10: 'cabinet',
 11: 'sidewalk, pavement',
 12: 'person',
 13: 'earth, ground',
 14: 'door',
 15: 'table',
 16: 'mountain, mount',
 17: 'plant',
 18: 'curtain',
 19: 'chair',
 20: 'car',
 21: 'water',
 22: 'painting, picture',
 23: 'sofa',
 24: 'shelf',
 25: 'house',
 26: 'sea',
 27: 'mirror',
 28: 'rug',
 29: 'field',
 30: 'armchair',
 31: 'seat',
 32: 'fence',
 33: 'desk',
 34: 'rock, stone',
 35: 'wardrobe, closet, press',
 36: 'lamp',
 37: 'tub',
 38: 'rail',
 39: 'cushion',
 40: 'base, pedestal, stand',
 41: 'box',
 42: 'column, pillar',
 43: 'signboard, sign',
 44: 'chest of drawers, chest, bureau, dresser',
 45: 'counter',
 46: 'sand',
 47: 'sink',
 48: 'skyscraper',
 49: 'fireplace',
 50: 'refrigerator, icebox',
 51: 'grandstand, covered stand',
 52: 'path',
 53: 'stairs',
 54: 'runway',
 55: 'case, display case, showcase, vitrine',
 56: 'pool table, billiard table, snooker table',
 57: 'pillow',
 58: 'screen door, screen',
 59: 'stairway, staircase',
 60: 'river',
 61: 'bridge, span',
 62: 'bookcase',
 63: 'blind, screen',
 64: 'coffee table',
 65: 'toilet, can, commode, crapper, pot, potty, stool, throne',
 66: 'flower',
 67: 'book',
 68: 'hill',
 69: 'bench',
 70: 'countertop',
 71: 'stove',
 72: 'palm, palm tree',
 73: 'kitchen island',
 74: 'computer',
 75: 'swivel chair',
 76: 'boat',
 77: 'bar',
 78: 'arcade machine',
 79: 'hovel, hut, hutch, shack, shanty',
 80: 'bus',
 81: 'towel',
 82: 'light',
 83: 'truck',
 84: 'tower',
 85: 'chandelier',
 86: 'awning, sunshade, sunblind',
 87: 'street lamp',
 88: 'booth',
 89: 'tv',
 90: 'plane',
 91: 'dirt track',
 92: 'clothes',
 93: 'pole',
 94: 'land, ground, soil',
 95: 'bannister, banister, balustrade, balusters, handrail',
 96: 'escalator, moving staircase, moving stairway',
 97: 'ottoman, pouf, pouffe, puff, hassock',
 98: 'bottle',
 99: 'buffet, counter, sideboard',
 100: 'poster, posting, placard, notice, bill, card',
 101: 'stage',
 102: 'van',
 103: 'ship',
 104: 'fountain',
 105: 'conveyer belt, conveyor belt, conveyer, conveyor, transporter',
 106: 'canopy',
 107: 'washer, automatic washer, washing machine',
 108: 'plaything, toy',
 109: 'pool',
 110: 'stool',
 111: 'barrel, cask',
 112: 'basket, handbasket',
 113: 'falls',
 114: 'tent',
 115: 'bag',
 116: 'minibike, motorbike',
 117: 'cradle',
 118: 'oven',
 119: 'ball',
 120: 'food, solid food',
 121: 'step, stair',
 122: 'tank, storage tank',
 123: 'trade name',
 124: 'microwave',
 125: 'pot',
 126: 'animal',
 127: 'bicycle',
 128: 'lake',
 129: 'dishwasher',
 130: 'screen',
 131: 'blanket, cover',
 132: 'sculpture',
 133: 'hood, exhaust hood',
 134: 'sconce',
 135: 'vase',
 136: 'traffic light',
 137: 'tray',
 138: 'trash can',
 139: 'fan',
 140: 'pier',
 141: 'crt screen',
 142: 'plate',
 143: 'monitor',
 144: 'bulletin board',
 145: 'shower',
 146: 'radiator',
 147: 'glass, drinking glass',
 148: 'clock',
 149: 'flag'}

class SemanticMapper:
    def __init__(self):
        fin = open('/root/exploration_ros_free/config_poni_exploration.yaml', 'r')
        config = yaml.safe_load(fin)
        fin.close()
        self.config = config
        self.objectgoal = None
        
        self.steps = 1
        self.delay = 1
        self.semantic_threshold = 2

        follower_config = config['path_follower']
        self.goal_radius = follower_config['goal_radius']
        self.max_d_angle = follower_config['max_d_angle']
        self.finish_radius = config['task']['finish_radius']

        # Initialize PONI
        args = parse_args_from_config(config['args'])
        exploration_config = config['exploration']
        self.exploration = PoniExploration(args)
        
        # Initialize mapper
        mapper_config = config['mapper']
        self.gt_mapper = Mapper(obstacle_inflation=0,
                             semantic_inflation=2,
                             vision_range=4.9,
                             semantic_vision_range=4.9,
                             map_size_cm=args.map_size_cm,
                             map_resolution=args.map_resolution,
                             semantic_threshold=0.99,
                             semantic_decay=1.)
        self.gt_mapper.erosion_size = 0
        
        self.objgoals = ['chair', 'bed', 'plant', 'toilet', 'tv_monitor', 'sofa']
        self.objgoal_to_cat = {0: 'chair',     1: 'bed',     2: 'plant',           3: 'toilet',           4: 'tv_monitor',   
                               5: 'sofa'}
        self.crossover = {'chair':['chair', 'armchair', 'swivel chair'], 
                          'bed':['bed'], 
                          'plant':['tree','plant','flower'], 
                          'toilet':['toilet, can, commode, crapper, pot, potty, stool, throne'], 
                          'tv_monitor':['computer','monitor','tv','crt screen','screen', 'tv_monitor'], 
                          'sofa':['sofa']} 
        self.crossover_for_coco = {'chair':['chair', 'armchair', 'swivel chair'], 
                                   'couch':['sofa'],
                                   'potted plant':['tree','plant','flower'], 
                                   'bed':['bed'],
                                   'toilet':['toilet, can, commode, crapper, pot, potty, stool, throne'], 
                                   'tv':['computer','monitor','tv','crt screen','screen', 'tv_monitor'],
                                   'dining-table':['table'],
                                   'oven':['oven', 'stove'],
                                   'sink':['sink'],
                                   'refrigerator':['refrigerator, icebox'],
                                   'book':['book'],
                                   'clock':['clock'],
                                   'vase':['vase'],
                                   'cup':['glass, drinking glass'],
                                   'bottle':['bottle']
                                   }
        self.coco_categories = {
            "chair": 0,
            "couch": 1,
            "potted plant": 2,
            "bed": 3,
            "toilet": 4,
            "tv": 5,
            "dining-table": 6,
            "oven": 7,
            "sink": 8,
            "refrigerator": 9,
            "book": 10,
            "clock": 11,
            "vase": 12,
            "cup": 13,
            "bottle": 14,
        }
        self.cat_to_coco = {
            'chair': 0,
            'sofa': 1,
            'plant': 2,
            'bed': 3,
            'toilet': 4,
            'tv_monitor': 5
        }
        self.object_extention = {'chair':['cabinet'],
                                 'bed':['chest of drawers','cushion'],
                                 'plant':['curtain','cabinet'],
                                 'toilet':['bathtub','sink','mirror'],
                                 'tv_monitor':['bed'],
                                 'sofa':['television receiver','crt screen','screen'],
                                }
        
        self.sem_maps = []
        self.gt_sem_maps = []
        self.obs_maps = []
        self.gt_obs_maps = []
        self.sem_masks = []
        self.gt_sem_masks = []
        self.robot_poses = []
        
        
    def reset(self):
        self.sem_maps = []
        self.gt_sem_maps = []
        self.obs_maps = []
        self.gt_obs_maps = []
        self.sem_masks = []
        self.gt_sem_masks = []
        self.robot_poses = []
        
        
    def get_multiclass_semantic(self, sem_mask_ade, objectgoal):
        multiclass_prediction = np.zeros((sem_mask_ade.shape[0], sem_mask_ade.shape[1], len(self.coco_categories)))
        for cat, i in self.coco_categories.items():
            after_crossover = self.crossover_for_coco[cat]
            segformer_index = [i for i,x in meta_clss.items() if x in after_crossover]
            for seg_index in segformer_index:
                multiclass_prediction[:, :, i] += (sem_mask_ade == seg_index).astype(float)
            if i == self.cat_to_coco[objectgoal]:
                multiclass_prediction[:, :, i].astype(bool).astype(float)
                multiclass_prediction[:, :, i] = cv2.erode(multiclass_prediction[:, :, i], np.ones((4, 4), np.uint8), iterations=1)
        multiclass_prediction.astype(bool).astype(float)
        return multiclass_prediction
    
    
    def get_sem_mask_objgoal(self, sem_mask_ade, objectgoal):
        after_crossover = self.crossover[objectgoal]
        segformer_index = [i for i,x in meta_clss.items() if x in after_crossover]
        mask = np.zeros((640,480))
        for seg_index in segformer_index:
            mask += (sem_mask_ade==seg_index).astype(float)
        mask.astype(bool).astype(float)
        obs_semantic = cv2.erode(mask,np.ones((4,4),np.uint8),iterations = 1)[:,:,np.newaxis]
        semantic_mask = obs_semantic[:, :, 0]
        return semantic_mask
    
    
    def get_obstacle_map(self):
        obstacle_map = self.exploration.local_map[0, 0, :, :].cpu().numpy()
        kernel = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])
        convolution = convolve2d(obstacle_map, kernel, mode='same')
        obstacle_map[convolution == 0] = 0
        i, j = (obstacle_map > 0).nonzero()
        obstacle_map = np.maximum(obstacle_map, self.exploration.collision_map)
        k = 3
        for di in range(-k, k + 1):
            for dj in range(-k, k + 1):
                ii = np.clip(i + di, 0, obstacle_map.shape[0] - 1)
                jj = np.clip(j + dj, 0, obstacle_map.shape[1] - 1)
                obstacle_map[ii, jj] = np.maximum(obstacle_map[ii, jj], 0.5)
        explored_map = self.exploration.local_map[0, 1, :, :].cpu().numpy()
        full_map = np.concatenate([np.expand_dims(explored_map, 2), np.expand_dims(obstacle_map, 2)], axis=2)
        return full_map
    
    
    def calculate_metrics(self):
        gt_map = self.gt_mapper.semantic_map
        cn = self.cat_to_coco[self.objectgoal] + 4
        pred_map = self.exploration.local_map[0, cn, :, :].cpu().numpy()
        pred_map[pred_map < self.semantic_threshold] = 0
        pred_map[pred_map >= self.semantic_threshold] = 1
        gt_i, gt_j = gt_map.nonzero()
        pred_i, pred_j = pred_map.nonzero()
        dists = []
        if len(gt_i) == 0:
            print('GT semantic is empty!')
        for ii, jj in zip(pred_i, pred_j):
            if len(gt_i) == 0:
                dists.append(gt_map.shape[0] + gt_map.shape[1])
            else:
                dst = np.sqrt((gt_i - ii) ** 2 + (gt_j - jj) ** 2)
                dists.append(2. / (1  + np.exp(-dst.min() / 20)) - 1)
        intersection = np.sum(gt_map * pred_map)
        union = np.sum(gt_map.astype(bool) | pred_map.astype(bool))
        iou = intersection / union
        return np.mean(dists), iou
        
        
    def run(self, data_path, trajectory, tilt_angles):
        self.exploration.reset()
        self.gt_mapper.reset()
        filenames = os.listdir(data_path)
        filenames = [x for x in filenames if x.endswith('jpg') or x.endswith('png')]
        filenames.sort(key=lambda x: int(x.split('_')[0]))
        #trajectory = np.loadtxt(os.path.join(data_path, 'poses.txt'))
        #tilt_angles = np.loadtxt(os.path.join(data_path, 'view_angles.txt'))
        tilt_angles = [0] + list(tilt_angles)
        n_steps = len(trajectory)#int(filenames[-1].split('_')[0])
        data_dirname = data_path.split('/')[-1]
        print(data_dirname.split('_'))
        if len(data_dirname.split('_')) == 3:
            objectgoal = data_dirname.split('_')[-1]
        else:
            objectgoal = '_'.join(data_dirname.split('_')[-2:])
        self.objectgoal = objectgoal
        if objectgoal in ['plant', 'tv_monitor']:
            self.exploration.sem_map_module.erosion_size = 3
        else:
            self.exploration.sem_map_module.erosion_size = 5
        for step in tqdm(list(range(n_steps))):
            if self.delay == 0:
                rgb = np.array(Image.open(os.path.join(data_path, '{}_rgb.jpg'.format(step))))
            else:
                rgb = np.array(Image.open(os.path.join(data_path, '{}_rgb.jpg'.format(max(step - self.delay, 0)))))
            depth = np.array(Image.open(os.path.join(data_path, '{}_depth.png'.format(step))))
            depth = depth * 450 / 500 + 50
            depth = (depth.astype(float) - 50) / 450
            if self.delay == 0:
                depth_old = np.array(Image.open(os.path.join(data_path, '{}_depth.png'.format(step))))
            else:
                depth_old = np.array(Image.open(os.path.join(data_path, '{}_depth.png'.format(max(step - self.delay, 0)))))
            depth_old = depth_old * 450 / 500 + 50
            depth_old = (depth_old.astype(float) - 50) / 450
            if self.delay == 0:
                semantic_mask = np.array(Image.open(os.path.join(data_path, '{}_pred_segmatron_{}_steps.png'.format(step, self.steps))))
            else:
                semantic_mask = np.array(Image.open(os.path.join(data_path, '{}_pred_segmatron_{}_steps_delayed.png'.format(step, self.steps))))
            #semantic_mask = np.array(Image.open(os.path.join(data_path, '{}_pred_segmatron.png'.format(step))))
            semantic_mask_gt = np.array(Image.open(os.path.join(data_path, '{}_sem.png'.format(step))))
            #semantic_mask_gt = np.array(Image.open(os.path.join(data_path, '{}_pred_segmatron.png'.format(step))))
            multiclass_prediction = self.get_multiclass_semantic(semantic_mask, objectgoal)
            semantic_mask_objgoal_gt = self.get_sem_mask_objgoal(semantic_mask_gt, objectgoal)
            robot_x, robot_y, robot_angle = trajectory[step]
            if self.delay == 0:
                robot_x_old, robot_y_old, robot_angle_old = trajectory[step]
            else:
                robot_x_old, robot_y_old, robot_angle_old = trajectory[max(step - self.delay, 0)]
            observations = {
                'rgb': rgb, 
                'depth': depth[..., np.newaxis],
                'depth_old': depth_old[..., np.newaxis],
                'gps': [robot_x, robot_y],
                'gps_old': [robot_x_old, robot_y_old],
                'compass': [robot_angle],
                'compass_old': [robot_angle_old],
                'objectgoal': [self.objgoals.index(objectgoal)]
            }
            self.exploration.sem_map_module.agent_view_angle = tilt_angles[step]
            self.gt_mapper.mapper.agent_view_angle = tilt_angles[step]
            self.exploration.update(observations, multiclass_prediction)
            observations = {
                'rgb': rgb, 
                'depth': depth[..., np.newaxis],
                'depth_old': depth_old[..., np.newaxis],
                'gps': [robot_x, robot_y],
                'gps_old': [robot_x_old, robot_y_old],
                'compass': [robot_angle],
                'compass_old': [robot_angle_old],
                'objectgoal': [self.objgoals.index(objectgoal)]
            }
            self.gt_mapper.step(observations, semantic_mask_objgoal_gt)
            
            sem_mask = rgb.copy()
            sem_mask[(sem_mask[:, :, 0] > 250) * (sem_mask[:, :, 1] < 5) * (sem_mask[:, :, 2] < 5)] *= 0
            semantic_mask_objgoal = self.get_sem_mask_objgoal(semantic_mask, objectgoal)
            sem_mask[semantic_mask_objgoal == 1] = [255, 0, 0]
            self.sem_masks.append(sem_mask)
            sem_mask_gt = rgb.copy()
            sem_mask_gt[(sem_mask_gt[:, :, 0] > 250) * (sem_mask_gt[:, :, 1] < 5) * (sem_mask_gt[:, :, 2] < 5)] *= 0
            sem_mask_gt[semantic_mask_objgoal_gt == 1] = [255, 0, 0]
            self.gt_sem_masks.append(sem_mask_gt)
            self.robot_poses.append([robot_x, robot_y, robot_angle])
            cn = self.cat_to_coco[objectgoal] + 4
            semantic_map = self.exploration.local_map[0, cn, :, :].cpu().numpy()
            semantic_map[semantic_map < self.semantic_threshold] = 0
            semantic_map[semantic_map >= self.semantic_threshold] = 1
            self.sem_maps.append(semantic_map)
            full_map = self.get_obstacle_map()
            obst_map = full_map[:, :, 1]
            obst_map[full_map[:, :, 0] == 0] = 0.5
            obst_map[obst_map > 0.5] = 1
            self.obs_maps.append(obst_map)
            gt_obst_map = self.gt_mapper.map
            gt_obst_map[self.gt_mapper.explored_map == 0] = 0.5
            gt_obst_map[gt_obst_map > 0.5] = 1
            self.gt_obs_maps.append(gt_obst_map)
            self.gt_sem_maps.append(self.gt_mapper.semantic_map)