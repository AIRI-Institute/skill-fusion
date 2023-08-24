from adet.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.engine.defaults import DefaultPredictor
from adet.utils.visualizer import TextVisualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer

import argparse
import numpy as np

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
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


class SemanticPredictor():
	def __init__(self, model='BlendMask', threshold=0.25):
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
		self.dataset_meta = {1: ('wall'),       2: ('floor'),      3: ('chair'),         4: ('door'),            5: ('table'),       6: ('picture'),
		            7: ('cabinet'),            8: ('cushion'),    9: ('window'),        10: ('sofa'),           11: ('bed'),        12: ('curtain'),
		            13: ('chest_of_drawers'),  14: ('plant'),     15: ('sink'),         16: ('stairs'),         17: ('ceiling'),
		            18: ('toilet'),            19: ('stool'),     20: ('towel'),        21: ('mirror'),         22: ('tv_monitor'), 23: ('shower'),
		            24: ('column'),            25: ('bathtub'),   26: ('counter'),      27: ('fireplace'),      28: ('lighting'),   29: ('beam'),
		            30: ('railing'),           31: ('shelving'),  32: ('blinds'),       33: ('gym_equipment'),  34: ('seating'),
		            35: ('board_panel'),       36: ('furniture'), 37: ('appliances'),   38: ('clothes'),        39: ('objects'),    40: ('misc')}
		self.objgoal_to_cat = {0: 'chair',     1: 'bed',     2: 'plant',           3: 'toilet',           4: 'tv_monitor',   
		                       5: 'sofa'}

		if model == 'BlendMask':
			a2 = "--config-file /home/kirill/AdelaiDet/configs/BlendMask/R_101_dcni3_5x.yaml \
			    --input input1.jpg input2.jpg \
			    --opts MODEL.WEIGHTS /home/kirill/AdelaiDet/weights/R_101_dcni3_5x.pth".split()
		else:
			a2 = "--config-file /home/kirill/AdelaiDet/configs/SOLOv2/R50_3x.yaml \
			    --input input1.jpg input2.jpg \
			    --opts MODEL.WEIGHTS /home/kirill/AdelaiDet/weights/SOLOv2_R50_3x.pth".split()
		args = get_parser().parse_args(a2)
		cfg = setup_cfg(args)
		self.predictor = DefaultPredictor(cfg)
		self.threshold = threshold


	def __call__(self, image, objectgoal):
		prediction = self.predictor(image)
		after_crossover = self.dataset_crossover[self.objgoal_to_cat[objectgoal]]
		classes = [i for ii,i in enumerate(prediction["instances"].pred_classes.cpu().numpy()) \
           if prediction["instances"].scores.cpu().numpy()[ii] > self.threshold]
		class_meta2 = {v:k for k,v in self.dataset_meta2.items()}[after_crossover]
		mask = np.zeros((480,640))
		if class_meta2 in [i+1 for i in classes]:
		    mask_ids = np.where(prediction["instances"].pred_classes.cpu().numpy() == class_meta2-1)[0]
		    mask = np.sum(np.array([prediction["instances"].pred_masks[mask_id].cpu().float().numpy() \
		                            for mask_id in mask_ids]), axis=0).astype(bool).astype(float)
		return mask