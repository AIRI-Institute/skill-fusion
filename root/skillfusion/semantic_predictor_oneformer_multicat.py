import numpy as np
import torch
import cv2
import sys
sys.path.append('/root/oneformer_agent')
from torch.nn import functional as F
from train_net import Trainer
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

class SemanticPredictor():
    def __init__(self, config):
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
        
        # Initialize semantic predictor
        config_file = config["config_path"]
        checkpoint_file = config["weights"]

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
        print('CLSS:', self.clss)
        
        
    def __call__(self, image, objectgoal=None):
        semantic_mask = None
        after_crossover = self.crossover[self.objgoal_to_cat[objectgoal]]
        #self.subgoal_names = self.object_extention[self.objgoal_to_cat[objectgoal]]
        self.segformer_index = [i for i,x in enumerate(self.clss) if x in after_crossover]
        obs_semantic = np.ones((640,480,1))
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        image_size = (image.shape[-2], image.shape[-1])
        padding_size = [
            0,
            640 - image_size[1],
            0,
            640 - image_size[0],
        ]
        image = F.pad(image, padding_size, value=128).contiguous()
        item = [{
            "image": image,
            "height": 640,
            "width": 640,
            "task": "The task is semantic"
        }]
        with torch.no_grad():
            result = self.model(item)
            result = result[0]["sem_seg"].cpu().numpy()
            result[result < 0.5] = 0
            result = np.argmax(result, axis=0)
            result = result[:, :480]
        ################################## FOR SEGFORMER B5  second lane
        if objectgoal is not None:
            after_crossover = self.crossover[self.objgoal_to_cat[objectgoal]]
            segformer_index = [i for i,x in enumerate(self.clss) if x in after_crossover]
            mask = np.zeros((640,480))
            for seg_index in segformer_index:
                mask += (result==seg_index).astype(float)
            mask.astype(bool).astype(float)
            obs_semantic = cv2.erode(mask,np.ones((4,4),np.uint8),iterations = 1)[:,:,np.newaxis]
            semantic_mask = obs_semantic[:, :, 0]
        multiclass_prediction = np.zeros((result.shape[0], result.shape[1], len(self.coco_categories)))
        for cat, i in self.coco_categories.items():
            after_crossover = self.crossover_for_coco[cat]
            segformer_index = [i for i,x in enumerate(self.clss) if x in after_crossover]
            for seg_index in segformer_index:
                multiclass_prediction[:, :, i] += (result == seg_index).astype(float)
            if i == self.cat_to_coco[self.objgoal_to_cat[objectgoal]]:
                multiclass_prediction[:, :, i].astype(bool).astype(float)
                multiclass_prediction[:, :, i] = cv2.erode(multiclass_prediction[:, :, i], np.ones((4, 4), np.uint8), iterations=1)
            multiclass_prediction.astype(bool).astype(float)
        return multiclass_prediction, semantic_mask
