import torch
import torch.nn as nn
import math
import numpy as np

import time

from models.detr_models.detr import MLP
from models.detr_models.transformer import TransformerDecoderLayer, TransformerDecoder

# Positional embeddings
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, n):
    grid = np.arange(n)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class SemanticMultiStepTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.input_feature_resolution = int(config.MODEL.INPUT_RESOLUTION / 32)
        print("INPUT FEATURE RESOLUTION", self.input_feature_resolution)
        self.img_feature_embedding = nn.Linear(config.MODEL.IMG_FEATURE_SIZE, config.MODEL.EMBEDDING_DIM)
        self.prediction_embedding = MLP(config.MODEL.MASK_EMB_SIZE + config.MODEL.NUM_CLASSES + 1, 512, config.MODEL.EMBEDDING_DIM, 3)
        self.mask_decoder = MLP(config.MODEL.OUTPUT_SIZE, 512, 64 * self.input_feature_resolution * self.input_feature_resolution, 3)
        self.logit_decoder = nn.Linear(config.MODEL.OUTPUT_SIZE, config.MODEL.NUM_CLASSES + 1)
        self.loss_decoder = MLP(config.MODEL.OUTPUT_SIZE, 512, 1, 3)

        self.action_decoder = MLP(config.MODEL.OUTPUT_SIZE, 512, 4, 3)
        self.action_tokens = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(1, config.MODEL.NUM_ACTIONS, config.MODEL.EMBEDDING_DIM),
                                                                   a=math.sqrt(config.MODEL.NUM_ACTIONS)))
        # build transformer
        decoder_layer = TransformerDecoderLayer(config.MODEL.EMBEDDING_DIM, config.MODEL.NUM_HEADS, 2048, 0.1, "relu", False)
        decoder_norm = nn.LayerNorm(config.MODEL.EMBEDDING_DIM)
        self.transformer = TransformerDecoder(decoder_layer, config.MODEL.NUM_LAYERS, decoder_norm, return_intermediate=False)

        self.img_len = self.input_feature_resolution * self.input_feature_resolution
        self.num_actions = config.MODEL.NUM_ACTIONS
        self.pos_embed = nn.Parameter(torch.zeros(1, config.MODEL.NUM_ACTIONS*self.img_len, config.MODEL.EMBEDDING_DIM), requires_grad=False)
        self.num_queries = 250
        self.embed_dim = config.MODEL.EMBEDDING_DIM
        self.query_embed = nn.Parameter(torch.zeros(1, self.num_queries*config.MODEL.NUM_ACTIONS + config.MODEL.NUM_ACTIONS, config.MODEL.EMBEDDING_DIM), requires_grad=True)
        self.init_pos_emb()

    def forward(self, x):
        # fold data into sequence
        img_feature_embedding = self.img_feature_embedding(x["embedded_memory_features"].permute(0, 1, 3, 4, 2))
        b, s, p, h, w = x["pred_masks"].shape 
        preds = torch.cat((x["mask_features"], x["pred_logits"]), dim=-1)
        prediction_embeddings = self.prediction_embedding(preds) # B X S X P X N
        b, s, p, n = prediction_embeddings.shape

        memory = torch.zeros((b, self.num_actions * self.input_feature_resolution * self.input_feature_resolution, n), device=prediction_embeddings.device)
        memory[:, :(s * self.input_feature_resolution * self.input_feature_resolution)] = img_feature_embedding.reshape(b, -1, n)
        tgt = torch.zeros((b, self.num_actions * self.num_queries + self.num_actions , n), device=prediction_embeddings.device)
        tgt[:, :(s * self.num_queries)] = prediction_embeddings.reshape(b, -1, n)
        tgt[:, self.num_actions *self.num_queries:(self.num_actions *self.num_queries + self.num_actions )] = self.action_tokens.repeat(b, 1, 1).reshape(b, -1, n)
        mask = torch.zeros((b, self.num_actions  * self.input_feature_resolution * self.input_feature_resolution), dtype=torch.bool, device=x["mask_features"].device)
        # pass sequence through model

        y = self.transformer(tgt.permute(1, 0, 2), memory.permute(1, 0, 2), memory_key_padding_mask=mask,
                             pos=self.pos_embed.permute(1, 0, 2), query_pos=self.query_embed.permute(1, 0, 2))

        # unfold data
        y_preds = y[:, :-self.num_actions].reshape(b, s, p, -1)

        masks = self.mask_decoder(y_preds).sigmoid().reshape(b, s, p, h, w)
        logits = self.logit_decoder(y_preds)
        loss = self.loss_decoder(y_preds)

        actions = self.action_decoder(y[:, -self.num_actions :-1].reshape(b,self.num_actions-1, -1))

        return {"seq": y_preds.squeeze(), "pred_masks": masks.squeeze(), "pred_logits": logits.squeeze(),
                "loss": loss, "actions": actions.squeeze()}

    def init_pos_emb(self):
        img_sin_embed = get_2d_sincos_pos_embed(self.embed_dim // 2, (self.input_feature_resolution, self.input_feature_resolution))
        img_pos_embed = torch.zeros((1, self.img_len, self.embed_dim))

        img_pos_embed[:, :, :self.embed_dim // 2] = torch.from_numpy(img_sin_embed).float()

        seq_sin_embed = get_1d_sincos_pos_embed(self.embed_dim // 2, self.num_actions)
        seq_pos_embed = torch.zeros((1, self.num_actions, self.embed_dim))
        seq_pos_embed[:, :, self.embed_dim // 2:] = torch.from_numpy(seq_sin_embed).float()

        pos_emb = torch.zeros((1, self.num_actions*self.input_feature_resolution*self.input_feature_resolution, self.embed_dim))
        for i in range(self.num_actions):
            pos_emb[:, self.img_len*i:self.img_len*(i+1)] = img_pos_embed + seq_pos_embed[:, i]

        self.pos_embed.data.copy_(pos_emb)
