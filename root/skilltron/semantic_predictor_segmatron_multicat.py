import numpy as np
import torch
import cv2
import sys
sys.path.append('/home/AI/yudin.da/zemskova_ts/skill-fusion/root/segmatron_agent')
from torch.nn import functional as F
from config_utils import (
    get_config,
    build_model,
)
from PIL import Image
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
                          'tv_monitor':['monitor','tv','crt screen','screen', 'tv_monitor'], 
                          'sofa':['sofa']} 
        self.crossover_for_coco = {'chair':['chair', 'armchair', 'swivel chair'], 
                                   'couch':['sofa'],
                                   'potted plant':['tree','plant','flower'], 
                                   'bed':['bed'],
                                   'toilet':['toilet, can, commode, crapper, pot, potty, stool, throne'], 
                                   'tv':['monitor','tv','crt screen','screen', 'tv_monitor'],
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
        self.object_extention = {'chair':['cabinet'],
                                 'bed':['chest of drawers','cushion'],
                                 'plant':['curtain','cabinet'],
                                 'toilet':['bathtub','sink','mirror'],
                                 'tv_monitor':['bed'],
                                 'sofa':['television receiver','crt screen','screen'],
                                }

        self.cat_to_coco = {
            'chair': 0,
            'sofa': 1,
            'plant': 2,
            'bed': 3,
            'toilet': 4,
            'tv_monitor': 5
        }        
        # Initialize semantic predictor
        cfg = get_config(config["config_path"])
        self.num_actions = cfg.MODEL.NUM_ACTIONS
        self.model = build_model(cfg)
        self.model.load_state_dict(
                        torch.load(config["weights"], map_location=torch.device('cpu'))['model'], strict=True)
        device = torch.cuda.current_device()
        self.model.to(device)
        self.model.eval()
        self.clss = self.model.segm_model.metadata.stuff_classes
        self.frames = []
        print('CLSS:', self.clss)
        self.count = 0
        self.delayed = config["delayed"] # if True, use image to predict segmentation for [t-(NUM_ACTIONS-1)] frame.
                            # if False, predict segmentation for the image - current frame t.

    def __call__(self, image, objectgoal=None):
        self.count +=1
        semantic_mask = None
        after_crossover = self.crossover[self.objgoal_to_cat[objectgoal]]
        #self.subgoal_names = self.object_extention[self.objgoal_to_cat[objectgoal]]
        self.segformer_index = [i for i,x in enumerate(self.clss) if x in after_crossover]
        obs_semantic = np.ones((640,480,1))

        image = Image.fromarray(image)
        image = np.array(image.resize((240,320)))
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        image_size = (image.shape[-2], image.shape[-1])
        padding_size = [
            0,
            320 - image_size[1],
            0,
            320 - image_size[0],
        ]
        image = F.pad(image, padding_size, value=128).contiguous()

        len_frames = len(self.frames)
        if len_frames < self.num_actions:
            self.frames = [image for i in range(self.num_actions)]
        else:
            if self.delayed:
                self.frames.pop(0)
                self.frames.append(image) # prediction for the frame [t - (NUM_ACTIONS-1)]
            else:
                self.frames.pop(-1)
                self.frames.insert(0, image) # prediction for the current frame t
        item = {
            "frames": torch.stack([torch.stack(self.frames)]),
            "height": [[320]],
            "width": [[320]],
            "task": [["The task is semantic"]]
        }
        # for i in range(2):
        #     image = Image.fromarray(torch.stack([torch.stack(self.frames)])[0,i,:,:,:].cpu().numpy().transpose(1, 2, 0))
        #     image.save(f"/home/AI/yudin.da/zemskova_ts/skill-fusion/root/exploration_ros_free/fbe_poni_exploration_maps_4_steps/attempt2/{self.count}_frame_{i}.jpg")
        #exit()
        print("Frames shape", item["frames"].shape)
        result = self.model.predict(item)
        mask_cls_results = result["pred_logits"][0][0]
        mask_pred_results = result["pred_masks"][0][0].unsqueeze(0)

        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(640, 640),
            mode="bilinear",
            align_corners=False,
        )
        mask_pred_results = mask_pred_results.squeeze(0)
        result = self.model.segm_model.semantic_inference(mask_cls_results, mask_pred_results)
        result = result.detach().cpu().numpy()
        #result[result < 0.5] = 0
        result = np.argmax(result, axis=0)
        result = result[:, :480]
        print(np.unique(result))
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
        print("Multiclass", np.unique(multiclass_prediction))
        print("Semantic mask", np.unique(semantic_mask))
        return multiclass_prediction, semantic_mask, result