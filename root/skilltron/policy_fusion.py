#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Tuple, Optional
import abc
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from bps_nav_test.common.utils import CustomCategorical
from bps_nav_test.common.running_mean_and_var import RunningMeanAndVar
from bps_nav_test.rl.models.rnn_state_encoder import build_rnn_state_encoder
from bps_nav_test.rl.models.simple_cnn import SimpleCNN


@torch.jit.script
def _process_depth(
    observations: Dict[str, torch.Tensor], n_discrete_depth: int = 10
) -> Dict[str, torch.Tensor]:
    if "depth" in observations:
        depth_observations = observations["depth"]
        if depth_observations.shape[1] != 1:
            depth_observations = depth_observations.permute(0, 3, 1, 2)

        depth_observations.clamp_(0.5, 5.0).add_(-0.5).mul_(1.0 / 4.5)  #!!!!!!!!!
        
        observations["depth"] = depth_observations

    return observations

@torch.jit.script
def _process_depth1(
    observations: Dict[str, torch.Tensor], n_discrete_depth: int = 10
) -> Dict[str, torch.Tensor]:
    if "depth" in observations:
        depth_observations = observations["depth"]
        if depth_observations.shape[1] != 1:
            depth_observations = depth_observations.permute(0, 3, 1, 2)

        observations["depth"] = depth_observations

    return observations


class SNIBottleneck(nn.Module):
    active: bool
    __constants__ = ["active"]

    def __init__(self, input_size, output_size, active=False):
        super().__init__()
        self.active: bool = active

        if active:
            self.output_size = output_size
            self.bottleneck = nn.Sequential(nn.Linear(input_size, 2 * output_size))
        else:
            self.output_size = input_size
            self.bottleneck = nn.Sequential()

    def forward(
        self, x
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.active:
            return x, None, None
        else:
            x = self.bottleneck(x)
            mu, sigma = torch.chunk(x, 2, x.dim() - 1)

            if self.training:
                sigma = F.softplus(sigma)
                sample = torch.addcmul(mu, sigma, torch.randn_like(sigma), value=1.0)

                # This is KL with standard normal for only
                # the parts that influence the gradient!
                kl = torch.addcmul(-torch.log(sigma), mu, mu, value=0.5)
                kl = torch.addcmul(kl, sigma, sigma, value=0.5)
            else:
                sample = None
                kl = None

            return mu, sample, kl


class ScriptableAC(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net

        self.out_size = int(self.net.output_size)
        self.action_distribution = CategoricalNet(self.out_size, dim_actions)
        self.action_distribution1 = CategoricalNet(self.out_size, dim_actions)
        self.action_distribution2 = CategoricalNet_obs(self.out_size, dim_actions)
        
        self.critic = CriticHead(self.out_size)
        self.critic1 = CriticHead(self.out_size)
        self.critic2 = CriticHead_obs(self.out_size)

    def post_net(
        self, observations: Dict[str, torch.Tensor], features, rnn_hidden_states, deterministic: bool
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        
        indicies0 = (observations['task_id']==0.)[:,0]
        indicies1 = (observations['task_id']==1.)[:,0]
        indicies2 = (observations['task_id']==2.)[:,0]

        #features0 = features[:,:256]
        #features1 = features[:,256:]
        
        c = torch.tensor([i for i in range(features.shape[0])])
        d = torch.cat((c[indicies0],c[indicies1],c[indicies2]),0)
        
        
        #logits = self.action_distribution(features)
        logits0 = self.action_distribution(features[indicies0])#[torch.argsort(d)]
        logits1 = self.action_distribution1(features[indicies1])#[torch.argsort(d)]
        logits2 = self.action_distribution2(features[indicies2],observations['pointgoal_with_gps_compass'][indicies2])#[torch.argsort(d)]
        
        #value = self.critic(features)["value"]
        value0 = self.critic(features[indicies0])["value"]#[torch.argsort(d)]
        value1 = self.critic1(features[indicies1])["value"]#[torch.argsort(d)]
        value2 = self.critic2(features[indicies2],observations['pointgoal_with_gps_compass'][indicies2])["value"]#[torch.argsort(d)]

        vl = torch.cat((value0, value1, value2),0)[torch.argsort(d)]
        
        
        if logits0.shape[0]>0:
            dist_result0 = self.action_distribution.dist.act(logits0, sample=not deterministic)
            return (
                vl, #value,
                {'actions': dist_result0['actions'],
                 'action_log_probs': dist_result0['action_log_probs']}, #dist_result,
                rnn_hidden_states)
        if logits1.shape[0]>0:    
            dist_result1 = self.action_distribution1.dist.act(logits1, sample=not deterministic)
            return (
                vl, #value,
                {'actions': dist_result1['actions'],
                 'action_log_probs': dist_result1['action_log_probs']}, #dist_result,
                rnn_hidden_states)
        if logits2.shape[0]>0:
            dist_result2 = self.action_distribution2.dist.act(logits2, sample=not deterministic)
            return (
                vl, #value,
                {'actions': dist_result2['actions'],
                 'action_log_probs': dist_result2['action_log_probs']}, #dist_result,
                rnn_hidden_states)
        
        actn = torch.cat((dist_result0['actions'], dist_result1['actions'], dist_result2['actions']),0)[torch.argsort(d)]
        actn_lg_prb = torch.cat((dist_result0['action_log_probs'], dist_result1['action_log_probs'], dist_result2['action_log_probs']),0)[torch.argsort(d)]
        
        return (
            vl, #value,
            {'actions': actn,
             'action_log_probs': actn_lg_prb}, #dist_result,
            rnn_hidden_states)
    
        #return (value0, dist_result0, rnn_hidden_states[(observations==0.)[:,0]]),
        #        (value1, dist_result1, rnn_hidden_states[(observations==1.)[:,0]])
    
    ###########################################################################################

    @torch.jit.export
    def act(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )

        return self.post_net(observations, features, rnn_hidden_states, deterministic)

    """
    @torch.jit.export
    def act_post_visual(
        self,
        visual_out,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        features, rnn_hidden_states = self.net.rnn_forward(
            visual_out,
            observations["pointgoal_with_gps_compass"],
            observations["task_id"],
            rnn_hidden_states,
            prev_actions,
            masks,
        )

        return self.post_net(observations['task_id'], features, rnn_hidden_states, deterministic)
    """
    
    @torch.jit.export
    def get_value(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
    ):
        features, _ = self.net(observations, rnn_hidden_states, prev_actions, masks)
        
        indicies0 = (observations['task_id']==0.)[:,0]#.bool()
        indicies1 = (observations['task_id']==1.)[:,0]#.bool()
        indicies2 = (observations['task_id']==2.)[:,0]#.bool()
        
        c = torch.tensor([i for i in range(features.shape[0])])
        d = torch.cat((c[indicies0],c[indicies1], c[indicies2]),0)
        
        #features0 = features[:,:256]
        #features1 = features[:,256:]
        
        #return self.critic(features)["value"]
        
        return torch.cat((self.critic(features[indicies0])["value"],
                          self.critic1(features[indicies1])["value"],
                          self.critic2(features[indicies2],observations['pointgoal_with_gps_compass'][indicies2])["value"]),0)[torch.argsort(d)]
    
        #return self.critic(features[(observations['task_id']==0.)[:,0]])["value"], 
        #        self.critic1(features[(observations['task_id']==1.)[:,0]])["value"]

    @torch.jit.export
    def evaluate_actions(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        #Dict[str, torch.Tensor]
        #Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
         
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks)  
        
        indicies0 = (observations['task_id']==0.)[:,0]
        indicies1 = (observations['task_id']==1.)[:,0]
        indicies2 = (observations['task_id']==2.)[:,0]
        
        c = torch.tensor([i for i in range(features.shape[0])])
        d = torch.cat((c[indicies0],c[indicies1],c[indicies2]),0)
        
        #features0 = features[:,:256]
        #features1 = features[:,256:]

        #result: Dict[str, torch.Tensor] = {}

        #logits = self.action_distribution(features)
        logits0 = self.action_distribution(features[indicies0])
        logits1 = self.action_distribution1(features[indicies1])
        logits2 = self.action_distribution2(features[indicies2],observations['pointgoal_with_gps_compass'][indicies2])

        #result.update(self.action_distribution.dist.evaluate_actions(logits, action))
        #result.update(self.critic(features))
        
        dict_log = self.action_distribution.dist.evaluate_actions(logits0, action[indicies0])
        dict_log1 = self.action_distribution1.dist.evaluate_actions(logits1, action[indicies1])
        dict_log2 = self.action_distribution2.dist.evaluate_actions(logits2, action[indicies2])
        
        #print('DICTLOG:', dict_log['log_probs'].shape)
        #print('DICTLOG1:', dict_log1['log_probs'].shape)
        
        """
        result.update({'probs': torch.cat((         dict_log['probs'],                dict_log1['probs']),0)[torch.argsort(d)],
                     'log_probs': torch.cat((       dict_log['log_probs'],            dict_log1['log_probs']),0)[torch.argsort(d)],
                     'action_log_probs': torch.cat((dict_log['action_log_probs'],     dict_log1['action_log_probs']),0)[torch.argsort(d)],
                     'entropy': torch.cat((         dict_log['entropy'],              dict_log1['entropy']),0)[torch.argsort(d)]})
                                
        result.update({'value':torch.cat((self.critic(features[(observations['task_id']==0.)[:,0]])["value"],
                                          self.critic1(features[(observations['task_id']==1.)[:,0]])["value"]),0)[torch.argsort(d)]})

        return result
    
        """
        result0: Dict[str, torch.Tensor] = {}
        result1: Dict[str, torch.Tensor] = {}
        result2: Dict[str, torch.Tensor] = {}
        
        result0.update(dict_log)
        result0.update(self.critic(features[indicies0]))
        
        result1.update(dict_log1)
        result1.update(self.critic1(features[indicies1]))
        
        result2.update(dict_log2)
        result2.update(self.critic2(features[indicies2],observations['pointgoal_with_gps_compass'][indicies2]))
        
        return result0, result1, result2
        
        #"""


class Policy(nn.Module):
    def __init__(self, net, observation_space, dim_actions):
        super().__init__()
        self.dim_actions = dim_actions

        self.num_recurrent_layers = net.num_recurrent_layers
        self.is_blind = net.is_blind

        self.ac = ScriptableAC(net, self.dim_actions)
        self.accelerated_net = None
        self.accel_out = None
        
        self.depth_max = False


        if "rgb" in observation_space.spaces:
            self.running_mean_and_var = RunningMeanAndVar(
                observation_space.spaces["rgb"].shape[0]
                + (
                    observation_space.spaces["depth"].shape[0]
                    if "depth" in observation_space.spaces
                    else 0
                ),
                initial_count=1e4,
            )
        else:
            self.running_mean_and_var = None
   
        #self.running_mean_and_var = None

    def script_net(self):
        self.ac = torch.jit.script(self.ac)

    def init_trt(self):
        raise NotImplementedError

    def update_trt_weights(self):
        raise NotImplementedError

    def trt_enabled(self):
        return self.accelerated_net != None

    def forward(self, *x):
        raise NotImplementedError

    def _preprocess_obs(self, observations):
        dtype = next(self.parameters()).dtype
        observations = {
            k: v.to(dtype=dtype, copy=True) for k, v in observations.items()
        }

        if not self.depth_max:
            if observations['depth'].shape[0]>0:
                if observations['depth'].max()>1.2:
                    self.depth_max = True
         
        if self.depth_max:
            observations = _process_depth(observations)
        else:
            observations = _process_depth1(observations)

        if "rgb" in observations:
            rgb = observations["rgb"]
            if rgb.shape[1] != 3:
                rgb = rgb.permute(0, 3, 1, 2)

            rgb = torch.clamp(rgb+torch.randn_like(rgb)*0.1,min=0.,max=255.)  
            rgb.mul_(1.0 / 255.0)
            x = [rgb]
            if "depth" in observations:
                depth = observations["depth"]
                depth = torch.nn.functional.dropout(torch.clamp(depth+torch.randn_like(depth)*0.2,min=0.,max=1.), p=0.2, training=True, inplace=False)
                x.append(depth)
   
            #x = torch.nn.functional.dropout(self.running_mean_and_var(torch.cat(x, 1)), p=0.2, training=True, inplace=False)
            x = self.running_mean_and_var(torch.cat(x, 1))
            #x = torch.cat(x, 1)

            observations["rgb"] = x[:, 0:3]
            if "depth" in observations:
                observations["depth"] = x[:, 3:]

        return observations

    def act(
        self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False,
    ):
        
        observations = self._preprocess_obs(observations)
        
        return self.ac.act(
            observations, rnn_hidden_states, prev_actions, masks, deterministic
        )

    def act_fast(
        self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False,
    ):
        observations = self._preprocess_obs(observations)
        
        if self.accelerated_net == None:
            return self.ac.act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        observations = self._preprocess_obs(observations)
        return self.ac.get_value(observations, rnn_hidden_states, prev_actions, masks)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action,
    ):
        observations = self._preprocess_obs(observations)
        return self.ac.evaluate_actions(
            observations, rnn_hidden_states, prev_actions, masks, action
        )

class CategoricalNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.linear = nn.Linear(num_inputs, 256)
        self.linear1 = nn.Linear(256, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

        self.dist = CustomCategorical()

    def forward(self, x):
        x = self.linear(x)
        x = self.linear1(x)
        return x    
    
class CategoricalNet_obs(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.linear = nn.Linear(num_inputs+32, 256)
        self.linear1 = nn.Linear(256, num_outputs)
        
        self.linear1_pnav = nn.Linear(2, 32)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)
        
        nn.init.orthogonal_(self.linear1.weight, gain=0.01)
        nn.init.constant_(self.linear1.bias, 0)

        self.dist = CustomCategorical()

    def forward(self, x, obs):
        
        x1 = self.linear1_pnav(obs)
        x2 = torch.cat((x,x1),1)
        x2 = self.linear(x2)
        x2 = self.linear1(x2)
        return x2
    
class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        self.fc = nn.Linear(input_size, 256)
        self.fc1 = nn.Linear(256, 1)
        #self.fc2 = nn.Linear(128, 1)

        self.layer_init()

    def layer_init(self):
        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

            if isinstance(m, nn.Linear):
                m.weight.data *= 0.1 / torch.norm(
                    m.weight.data, p=2, dim=1, keepdim=True
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        x = self.fc(x)
        x = self.fc1(x)
        #x = self.fc2(x)
        
        return {"value": x}
    
class CriticHead_obs(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        self.fc_pnav = nn.Linear(2, 32)
        self.fc = nn.Linear(input_size+32, 256)
        self.fc1 = nn.Linear(256, 1)
        #self.fc2 = nn.Linear(128, 1)

        self.layer_init()

    def layer_init(self):
        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

            if isinstance(m, nn.Linear):
                m.weight.data *= 0.1 / torch.norm(
                    m.weight.data, p=2, dim=1, keepdim=True
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, obs) -> Dict[str, torch.Tensor]:
        x1 = self.fc_pnav(obs)
        x2 = torch.cat((x,x1),1)
        x2 = self.fc(x2)
        x2 = self.fc1(x2)
        #x = self.fc2(x)
        
        return {"value": x2}    

class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass
