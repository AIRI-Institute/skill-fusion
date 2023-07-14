import numpy as np
import torch
import cv2
import sys
import os
sys.path.append('/root/oneformer_agent')
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
from tqdm import tqdm
from skimage.io import imread
from PIL import Image as PIL_Image

class SemanticPredictor():
    def __init__(self):
        # Initialize semantic predictor
        config_file = '/root/exploration_ros_free/weights/config_oneformer_ade20k.yaml'
        checkpoint_file = '/root/exploration_ros_free/weights/model_0299999.pth'
        #self.model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
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
        
    def __call__(self, image):
        item = [{
            "image": torch.from_numpy(np.moveaxis(image[:, :, :3], -1, 0)),
            "height": 720,
            "width": 1280,
            "task": "semantic"
        }]
        with torch.no_grad():
            result = self.model(item)
            result = result[0]["sem_seg"].cpu().numpy()
            result = np.argmax(result, axis=0)
        return result.astype(np.uint8)
    
model = SemanticPredictor()
input_dir = '/root/itlp_campus_data/front_cam'
output_dir = '/root/itlp_campus_data/labels/front_cam'
filenames = list(os.listdir(input_dir))
filenames.sort()
for fn in tqdm(filenames):
    pil_img = PIL_Image.open(os.path.join(input_dir, fn))
    img = np.asarray(pil_img)
    semantic = model(img)
    pil_img = PIL_Image.fromarray(semantic)
    pil_img.save(os.path.join(output_dir, fn))