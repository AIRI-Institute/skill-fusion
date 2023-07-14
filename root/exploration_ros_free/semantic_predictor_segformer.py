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


class SemanticPredictor():
	def __init__(self):
		self.objgoal_to_cat = {0: 'chair',     1: 'bed',     2: 'plant',           3: 'toilet',           4: 'tv_monitor',   
		                       5: 'sofa'}
		self.crossover = {'chair':['chair', 'armchair', 'swivel chair'], 
		                  'bed':['bed '], 
		                  'plant':['tree','plant','flower'], 
		                  'toilet':['toilet'], 
		                  'tv_monitor':['computer','monitor','television receiver','crt screen','screen'], 
		                  'sofa':['sofa']} 

		config_file = '/root/SegFormer/local_configs/segformer/B4/segformer.b4.512x512.ade.160k.py'
		checkpoint_file = '/root/exploration_ros_free/weights/iter_160000.pth'
		self.model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
		self.clss = self.model.CLASSES
		print('CLASSES:', self.clss)


	def __call__(self, image, objectgoal):
		result = inference_segmentor(self.model, image)
		after_crossover = self.crossover[self.objgoal_to_cat[objectgoal]]
		segformer_index = [i for i,x in enumerate(self.clss) if x in after_crossover]
		mask = np.zeros((480,640))
		for seg_index in segformer_index:
		    mask += (result[0]==seg_index).astype(float)
		mask.astype(bool).astype(float)
		obs_semantic = cv2.erode(mask,np.ones((4,4),np.uint8),iterations = 1)[:,:,np.newaxis]
		return obs_semantic[:, :, 0]