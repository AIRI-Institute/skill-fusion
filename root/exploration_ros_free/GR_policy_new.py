#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, Tuple

import numpy as np
import torch
from gym import spaces
from torch import nn as nn
from torch.nn import functional as F

from habitat.config import DictConfig as Config
from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
    HeadingSensor,
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
    ProximitySensor,
)
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from rnn_state_encoder import (
    build_rnn_state_encoder,
)

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import clip
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision import transforms


from policy import Net, Policy


#@baseline_registry.register_policy
class PointNavResNetPolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        resnet_baseplanes: int = 32,
        backbone: str = "resnet18",
        normalize_visual_inputs: bool = False,
        force_blind_policy: bool = False,
        policy_config: Config = None,
        **kwargs
    ):
        if policy_config is not None:
            discrete_actions = (
                policy_config.action_distribution_type == "categorical"
            )
            self.action_distribution_type = (
                policy_config.action_distribution_type
            )
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"
        super().__init__(
            PointNavResNetNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                force_blind_policy=force_blind_policy,
                discrete_actions=discrete_actions,
            ),
            dim_actions=action_space.n,  # for action distribution
            policy_config=policy_config,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: spaces.Dict, action_space
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.PPO.hidden_size,
            rnn_type=config.RL.DDPPO.rnn_type,
            num_recurrent_layers=config.RL.DDPPO.num_recurrent_layers,
            backbone=config.RL.DDPPO.backbone,
            normalize_visual_inputs="rgb" in observation_space.spaces,
            force_blind_policy=config.FORCE_BLIND_POLICY,
            policy_config=config.RL.POLICY,
        )


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        baseplanes: int = 32,
        ngroups: int = 32,
        spatial_size: int = 128,
        make_backbone=None,
        normalize_visual_inputs: bool = False,
    ):
        super().__init__()

        #if "rgb" in observation_space.spaces:
        #    self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        #    spatial_size = observation_space.spaces["rgb"].shape[0] // 2
        #else:
        self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
            spatial_size = observation_space.spaces["depth"].shape[0] // 2
        else:
            self._n_input_depth = 0
            
        if "semantic" in observation_space.spaces:
            self._n_input_semantic = observation_space.spaces["semantic"].shape[2]
            spatial_size = observation_space.spaces["semantic"].shape[0] // 2
        else:
            self._n_input_semantic = 0      

        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(
                self._n_input_depth + self._n_input_rgb + self._n_input_semantic
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        if not self.is_blind:
            input_channels = self._n_input_depth + self._n_input_rgb + self._n_input_semantic
            self.backbone = make_backbone(input_channels, baseplanes, ngroups)

            final_spatial = int(
                spatial_size * self.backbone.final_spatial_compress
            )
            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(after_compression_flat_size / (final_spatial**2)) 
            )
            num_compression_channels = 128
            
            print('!input_channels_baseplanes_ngroups ',input_channels,baseplanes,ngroups)
            print('!!NEW CHECK: ',spatial_size, self.backbone.final_spatial_compress, input_channels)
            print('!!num_compression_channels ',num_compression_channels,final_spatial,spatial_size,self.backbone.final_spatial_compress, self.backbone.final_channels)
            
            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.backbone.final_channels,
                    num_compression_channels, #128
                    kernel_size=(5,5), #3, ConvTranspose2d(7,5) Conv2d(3,4) (4,6)
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels), #128
                nn.ReLU(True),
            )

            self.output_shape = (
                num_compression_channels,
                6,
                8,
            )
            
            self.model, self.preprocess = clip.load("RN50")
            self.model.eval()
            
            #for param in self.model.parameters():
            #    param.requires_grad = False
            
            print('LOADED CLIP')
            
            self.tr = transforms.Compose([
                    Resize(size=(self.model.visual.input_resolution,self.model.visual.input_resolution), interpolation=InterpolationMode.BICUBIC),
                        #CenterCrop(self.model.visual.input_resolution),
                        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth + self._n_input_semantic == 0
    
    def forward_clip(self,x):

        def stem(x):
            x = self.model.visual.relu1(self.model.visual.bn1(self.model.visual.conv1(x)))
            x = self.model.visual.relu2(self.model.visual.bn2(self.model.visual.conv2(x)))
            x = self.model.visual.relu3(self.model.visual.bn3(self.model.visual.conv3(x)))
            x = self.model.visual.avgpool(x)
            return x

        x = x.type(self.model.visual.conv1.weight.dtype)
        x = stem(x)
        x = self.model.visual.layer1(x)
        x = self.model.visual.layer2(x)
        x = self.model.visual.layer3(x)
        x = self.model.visual.layer4(x)
        x = self.model.visual.attnpool(x)
        return x

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        if self.is_blind:
            return None

        cnn_input = []
        #if self._n_input_rgb > 0:
            #rgb_observations = observations["rgb"]
            #rgb_observations = torch.clamp(rgb_observations+torch.randn_like(rgb_observations).uniform_(-1, 1)*255*0.2,min=0.,max=255.)  
            
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            #rgb_observations = rgb_observations.permute(0, 3, 1, 2).float()
            #rgb_observations = (rgb_observations / 255.0)  # normalize RGB
            #cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations =  torch.nn.functional.dropout(observations["depth"], p=0.2, training=True, inplace=False)
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input.append(depth_observations)
            
        if self._n_input_semantic > 0:
            semantic_observations = torch.nn.functional.dropout(observations["semantic"], p=0.3, training=True, inplace=False)
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            semantic_observations = semantic_observations.permute(0, 3, 1, 2)
            cnn_input.append(semantic_observations)      

        x = torch.cat(cnn_input, dim=1)
        #x = F.avg_pool2d(x, 2)

        #print('0 X_SHAPE: ',x.shape,x.max(),x.min())
        #x = self.running_mean_and_var(x)
        #print('1 X_SHAPE: ',x.shape,x.max(),x.min())
        x = self.backbone(x)
        #print('2 X_SHAPE: ',x.shape,x.max(),x.min())
        x = self.compression(x)
        #print('3 X_SHAPE: ',x.shape,x.max(),x.min())
        
        
        return x


class PointNavResNetNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    prev_action_embedding: nn.Module

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs: bool,
        force_blind_policy: bool = False,
        discrete_actions: bool = True,
    ):
        super().__init__()
        self.prev_action_embedding: nn.Module
        self.discrete_actions = discrete_actions
        if discrete_actions:
            self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        else:
            self.prev_action_embedding = nn.Linear(action_space.n, 32)

        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action

        self._hidden_size = hidden_size

        self.visual_encoder = ResNetEncoder(
            observation_space if not force_blind_policy else spaces.Dict({}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )
        
        #print(self.visual_encoder.model)

        if not self.visual_encoder.is_blind:
            self.visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    np.prod(self.visual_encoder.output_shape), 1024 #hidden_size
                ),
                nn.ReLU(True),
            )

        self.state_encoder = build_rnn_state_encoder(
            2080, #(0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = []
        
        with torch.no_grad():
            rgb_observations = observations["rgb"].permute(0,3,1,2)/255.
            #print('0 RGB SHAPE: ',rgb_observations.shape,rgb_observations.max(),rgb_observations.min())
            rgb_observations = self.visual_encoder.tr(rgb_observations)
            #print('1 RGB SHAPE: ',rgb_observations.shape,rgb_observations.max(),rgb_observations.min())
            rgb_observations = self.visual_encoder.forward_clip(rgb_observations)
            #print('2 RGB SHAPE: ',rgb_observations.shape,rgb_observations.max(),rgb_observations.min())
            x.append(rgb_observations)
        
        if not self.is_blind:
            visual_feats = observations.get(
                "visual_features", self.visual_encoder(observations)
            )
            visual_feats = self.visual_fc(visual_feats)
            #print('44 RGB SHAPE: ',visual_feats.shape)
            x.append(visual_feats)

        if self.discrete_actions:
            prev_actions = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions)
            prev_actions = self.prev_action_embedding(
                torch.where(masks.view(-1), prev_actions + 1, start_token)
            )
        else:
            prev_actions = self.prev_action_embedding(
                masks * prev_actions.float()
            )

        x.append(prev_actions)

        out = torch.cat(x, dim=1)
        #print('55 RGB SHAPE: ',out.shape)
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks
        )

        return out, rnn_hidden_states
