import torch
import torch.nn as nn
import copy

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog,
)
from detectron2.config import get_cfg

from detectron2.modeling import build_model
from models.oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)
from detectron2.projects.deeplab import add_deeplab_config


class oneformer_single_frame(nn.Module):

    def __init__(
        self,
        config,
    ):
        super().__init__()
        # build OneFormer
        self.cfg = copy.deepcopy(config)  # cfg can be modified by model

        cfg = get_cfg()

        add_deeplab_config(cfg)
        add_common_config(cfg)
        add_swin_config(cfg)
        add_dinat_config(cfg)
        add_convnext_config(cfg)
        add_oneformer_config(cfg)
        cfg.merge_from_file(self.cfg.MODEL.CONFIG)
        self.segm_model = build_model(cfg)
        #checkpointer = DetectionCheckpointer(self.segm_model)
        #checkpointer.load(config.MODEL.WEIGHTS)
        self.criterion = self.segm_model.criterion

        self.logger = None
        self.mode = 'train'
        self.config = config

    def predict(self, data):

        # reformat img data

        b, s, c, h, w = data["frames"].shape
        img = data["frames"].view(s, c, h, w)

        batched_inputs = [
            {
                "image": img[0],
                "height": data["height"][0][0],
                "width": data["width"][0][0],
                "task": data["task"][0][0]
            }
        ]

        out = self.segm_model(batched_inputs)

        out["pred_logits"] = out["pred_logits"].unsqueeze(0)
        out["pred_masks"] = out["pred_masks"].unsqueeze(0)

        return out

    def forward(self, data, train=True):

        b, s, c, h, w = data["frames"].shape
        img = data["frames"].view(b, s, c, h, w)

        labels = []
        for i in range(b):
            labels.append([])
            for j in range(s):
                category_ids = torch.unique(data["masks"][i][j]).type(torch.cuda.LongTensor)
                category_ids = category_ids[category_ids != 255]
                masks = torch.zeros((len(category_ids), data["height"][i][j], data["width"][i][j]))
                for k, cat in enumerate(category_ids):
                    masks[k, :, :] = (data["masks"][i][j] == cat).type(torch.cuda.DoubleTensor)
                labels[i].append({
                    "labels": category_ids,
                    "masks": masks
                })

        segm_losses = []
        out_logits_list = []
        out_masks_list = []

        for task in range(b):
            import random
            ridx = random.randint(0, 4)
            batched_inputs = [
                {
                    "image": img[task][ridx],
                    "height": data["height"][task][ridx],
                    "width": data["width"][task][ridx],
                    "task": data["task"][task][ridx]
                }
            ]

            out = self.segm_model(batched_inputs)
            segm_loss = self.criterion(out, labels[task][ridx:ridx+1])
            segm_losses.append({k: v.detach() for k, v in segm_loss.items()})
            segm_loss = 2*segm_loss["loss_ce"] + 5 * segm_loss["loss_dice"] + 5 * segm_loss["loss_mask"]

            segm_loss.backward()

            out_logits_list.append(out["pred_logits"])
            out_masks_list.append(out["pred_masks"])

        predictions = {
            "pred_logits": torch.stack(out_logits_list, dim=0),
            "pred_masks": torch.stack(out_masks_list, dim=0)
        }
        mean_segm_losses = {
            k.replace("loss", "loss_segm"):
            torch.mean(torch.stack([x[k] for x in segm_losses]))
            for k, v in segm_losses[0].items()
        }

        losses = mean_segm_losses

        return predictions, losses

    def eval(self):
        self.segm_model.eval()
        return self.train(False)

    def train(self, mode=True):
        self.mode = 'train' if mode else 'test'
        self.segm_model.train(mode)
        return self

    def get_optimizer_groups(self, train_config):
        optim_groups = [
            {"params": list(self.segm_model.parameters()), "weight_decay": 0.0},
        ]
        return optim_groups

    def set_logger(self, logger):
        assert self.logger is None, "This model already has a logger!"
        self.logger = logger

