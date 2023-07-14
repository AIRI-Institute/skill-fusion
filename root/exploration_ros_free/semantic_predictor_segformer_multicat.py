import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test, init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

import numpy as np
import cv2
import time

class SemanticPredictor():
    def __init__(self, threshold=0.25):
        self.objgoal_to_cat = {0: 'chair',     1: 'bed',     2: 'plant',           3: 'toilet',           4: 'tv_monitor',   
                               5: 'sofa'}
        self.crossover = {'chair':['chair', 'armchair', 'swivel chair'], 
                          'bed':['bed '], 
                          'plant':['tree','plant','flower'], 
                          'toilet':['toilet'], 
                          'tv_monitor':['computer','monitor','television receiver','crt screen','screen'], 
                          'sofa':['sofa']} 
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
        self.crossover_for_coco = {
            'chair': ['chair', 'armchair', 'swivel chair'],
            'couch': ['sofa'],
            'potted plant': ['tree', 'plant', 'flower'],
            'bed': ['bed '],
            'toilet': ['toilet'],
            'tv':['computer', 'monitor', 'television receiver', 'crt screen', 'screen'], 
            'dining-table': ['table', 'coffee table'],
            'oven': ['oven', 'stove'],
            'sink': ['sink'],
            'refrigerator': ['refrigerator'],
            'book': ['book'],
            'clock': ['clock'],
            'vase': ['vase'],
            'cup': ['glass'],
            'bottle': ['bottle']            
        }
        
        config_file = '/root/SegFormer/local_configs/segformer/B4/segformer.b4.512x512.ade.160k.py'
        checkpoint_file = '/root/exploration_ros_free/weights/iter_160000.pth'
        self.model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
        self.clss = self.model.CLASSES
        print('CLASSES:', self.clss)


    def __call__(self, image, objectgoal=None):
        result = inference_segmentor(self.model, image)
        semantic_mask = None
        if objectgoal is not None:
            after_crossover = self.crossover[self.objgoal_to_cat[objectgoal]]
            segformer_index = [i for i,x in enumerate(self.clss) if x in after_crossover]
            mask = np.zeros((640,480))
            for seg_index in segformer_index:
                mask += (result[0]==seg_index).astype(float)
            mask.astype(bool).astype(float)
            obs_semantic = cv2.erode(mask,np.ones((4,4),np.uint8),iterations = 1)[:,:,np.newaxis]
            semantic_mask = obs_semantic[:, :, 0]
        multiclass_prediction = np.zeros((result[0].shape[0], result[0].shape[1], len(self.coco_categories)))
        for cat, i in self.coco_categories.items():
            after_crossover = self.crossover_for_coco[cat]
            segformer_index = [i for i,x in enumerate(self.clss) if x in after_crossover]
            for seg_index in segformer_index:
                multiclass_prediction[:, :, i] += (result[0] == seg_index).astype(float)
            multiclass_prediction.astype(bool).astype(float)
        return multiclass_prediction, semantic_mask