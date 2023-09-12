#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.optim.lr_scheduler import LambdaLR

from bps_nav_test.common.base_trainer import BaseRLTrainer
from bps_nav_test.common.env_utils import construct_envs, construct_envs_habitat
from bps_nav_test.common.rollout_storage import RolloutStorage
from bps_nav_test.common.tensorboard_utils import TensorboardWriter
from bps_nav_test.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
)
from bps_nav_test.common.logger import logger
from bps_nav_test.rl.ppo.ppo import PPO
from bps_nav_test.common.tree_utils import (
    tree_append_in_place,
    tree_clone_shallow,
    tree_map,
    tree_select,
    tree_clone_structure,
    tree_copy_in_place,
)

from bps_nav_test.rl.ddppo.policy import ResNetPolicy
from gym import spaces
from gym.spaces import Dict as SpaceDict

@torch.jit.script
def so3_to_matrix(q, m):
    m[..., 0, 0] = 1.0 - 2.0 * (q[..., 2] ** 2 + q[..., 3] ** 2)
    m[..., 0, 1] = 2.0 * (q[..., 1] * q[..., 2] - q[..., 3] * q[..., 0])
    m[..., 0, 2] = 2.0 * (q[..., 1] * q[..., 3] + q[..., 2] * q[..., 0])
    m[..., 1, 0] = 2.0 * (q[..., 1] * q[..., 2] + q[..., 3] * q[..., 0])
    m[..., 1, 1] = 1.0 - 2.0 * (q[..., 1] ** 2 + q[..., 3] ** 2)
    m[..., 1, 2] = 2.0 * (q[..., 2] * q[..., 3] - q[..., 1] * q[..., 0])
    m[..., 2, 0] = 2.0 * (q[..., 1] * q[..., 3] - q[..., 2] * q[..., 0])
    m[..., 2, 1] = 2.0 * (q[..., 2] * q[..., 3] + q[..., 1] * q[..., 0])
    m[..., 2, 2] = 1.0 - 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2)


@torch.jit.script
def se3_to_4x4(se3_states):
    n = se3_states.size(0)

    mat = torch.zeros((n, 4, 4), dtype=torch.float32, device=se3_states.device)
    mat[:, 3, 3] = 1

    so3 = se3_states[:, 0:4]
    so3_to_matrix(so3, mat[:, 0:3, 0:3])

    mat[:, 0:3, 3] = se3_states[:, 4:]

    return mat


class PPOTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None, resume_from=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        #  if config is not None:
        #  logger.info(f"config: {config}")

        self._static_encoder = False
        self._encoder = None

    def _setup_actor_critic_agent(self, ppo_cfg) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        self.actor_critic = ResNetPolicy(
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
            rnn_type=self.config.RL.DDPPO.rnn_type,
            num_recurrent_layers=self.config.RL.DDPPO.num_recurrent_layers,
            backbone=self.config.RL.DDPPO.backbone,
        )
        self.actor_critic.to(self.device)

        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """

        def _cast(param):
            if "Half" in param.type():
                param = param.to(dtype=torch.float32)

            return param

        checkpoint = {
            "state_dict": {k: _cast(v) for k, v in self.agent.state_dict().items()},
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name))

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(cls, info: Dict[str, Any]) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(v).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif v is None:
                result[k] = None
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _inference(self, rollouts, rollouts1, rollouts2, idx):
        with torch.no_grad(), self.timing.add_time("Rollout-Step"):
            with self.timing.add_time("Inference"):
                step_input = tree_select(rollouts[idx].step, rollouts[idx].storage_buffers)
                step_input1 = tree_select(rollouts1[idx].step, rollouts1[idx].storage_buffers)
                step_input2 = tree_select(rollouts2[idx].step, rollouts2[idx].storage_buffers)

                (values0, dist_result0, recurrent_hidden_states0) = self.actor_critic.act(
                    step_input["observations"],
                    step_input["recurrent_hidden_states"],
                    step_input["prev_actions"],
                    step_input["masks"])

                (values1, dist_result1, recurrent_hidden_states1) = self.actor_critic.act(
                    step_input1["observations"],
                    step_input1["recurrent_hidden_states"],
                    step_input1["prev_actions"],
                    step_input1["masks"])
                
                (values2, dist_result2, recurrent_hidden_states2) = self.actor_critic.act(
                    step_input2["observations"],
                    step_input2["recurrent_hidden_states"],
                    step_input2["prev_actions"],
                    step_input2["masks"])

            with self.timing.add_time("Rollouts-Insert"):
                #print('Roll0: ',dist_result0["actions"].shape)
                rollouts[idx].insert(
                    recurrent_hidden_states=recurrent_hidden_states0,
                    action_log_probs=dist_result0["action_log_probs"],
                    value_preds=values0,
                    actions=dist_result0["actions"],
                    non_blocking=False)
                
                #print('Roll1: ',dist_result1["actions"].shape)
                rollouts1[idx].insert(
                    recurrent_hidden_states=recurrent_hidden_states1,
                    action_log_probs=dist_result1["action_log_probs"],
                    value_preds=values1,
                    actions=dist_result1["actions"],
                    non_blocking=False)
                
                rollouts2[idx].insert(
                    recurrent_hidden_states=recurrent_hidden_states2,
                    action_log_probs=dist_result2["action_log_probs"],
                    value_preds=values2,
                    actions=dist_result2["actions"],
                    non_blocking=False)

            with self.timing.add_time("Inference"):
                cpu_actions0 = dist_result0["actions"].squeeze(-1).to(device="cpu")
                cpu_actions1 = dist_result1["actions"].squeeze(-1).to(device="cpu")
                cpu_actions2 = dist_result2["actions"].squeeze(-1).to(device="cpu")

        return cpu_actions0, cpu_actions1, cpu_actions2

    def _step_simulation(self, cpu_actions, idx):
        with self.timing.add_time("Rollout-Step"), self.timing.add_time(
            "Habitat-Step-Start"
        ):
            """
            self.envs.step(idx, cpu_actions.numpy())
            obs = self._observations[idx]
            rewards = self._rewards[idx]
            masks = self._masks[idx]
            infos = self._rollout_infos[idx]
            """
            
            obs, obs1, obs2, rewards, rewards1, rewards2, masks, masks1, masks2, infos, infos1, infos2 = self.envs.step(cpu_actions.numpy())

            return (obs, rewards, masks, infos), (obs1, rewards1, masks1, infos1), (obs2, rewards2, masks2, infos2)


    def _sync_renderer_and_insert(self, rollouts, rollouts1, rollouts2, sim_step_res, sim_step_res1, sim_step_res2, idx):
        with self.timing.add_time("Rollout-Step"):
            batch, rewards, masks, infos = sim_step_res
            batch1, rewards1, masks1, infos1 = sim_step_res1
            batch2, rewards2, masks2, infos2 = sim_step_res2
            
            with self.timing.add_time("Renderer-Wait"):
                self._syncs[idx].wait()
                torch.cuda.current_stream().synchronize()
                
            with self.timing.add_time("Rollouts-Insert"):
                rollouts[idx].insert(batch, rewards=rewards, masks=masks, non_blocking=False)
                rollouts1[idx].insert(batch1, rewards=rewards1, masks=masks1, non_blocking=False)
                rollouts2[idx].insert(batch2, rewards=rewards2, masks=masks2, non_blocking=False)

            rollouts[idx].advance()
            rollouts1[idx].advance()
            rollouts2[idx].advance()

            return masks.size(0), masks1.size(0), masks2.size(0)

    def _update_stats(
        self,
        rollouts,
        rollouts1,
        rollouts2,
        current_episode_reward,
        current_episode_reward1,
        current_episode_reward2,
        running_episode_stats,
        running_episode_stats1,
        running_episode_stats2,
        sim_step_res,
        sim_step_res1,
        sim_step_res2,
        stats_inds,
        stats_inds1,
        stats_inds2,
        idx,
    ):
        with self.timing.add_time("Rollout-Step"):
            batch, rewards, masks, infos = sim_step_res
            batch1, rewards1, masks1, infos1 = sim_step_res1
            batch2, rewards2, masks2, infos2 = sim_step_res2


            with self.timing.add_time("Update-Stats"):
                dones = masks == 0
                dones1 = masks1 == 0
                dones2 = masks2 == 0

                def _masked(v):
                    return torch.where(dones, v, v.new_zeros(()))
                def _masked1(v):
                    return torch.where(dones1, v, v.new_zeros(()))
                def _masked2(v):
                    return torch.where(dones2, v, v.new_zeros(()))

                current_episode_reward[stats_inds] += rewards
                current_episode_reward1[stats_inds1] += rewards1
                current_episode_reward2[stats_inds2] += rewards2
                
                running_episode_stats["reward"][stats_inds] += _masked(
                    current_episode_reward[stats_inds])
                running_episode_stats1["reward"][stats_inds1] += _masked1(
                    current_episode_reward1[stats_inds1])
                running_episode_stats2["reward"][stats_inds2] += _masked2(
                    current_episode_reward2[stats_inds2])
                
                running_episode_stats["count"][stats_inds] += dones.type_as(
                    running_episode_stats["count"])
                running_episode_stats1["count"][stats_inds1] += dones1.type_as(
                    running_episode_stats1["count"])
                running_episode_stats2["count"][stats_inds2] += dones2.type_as(
                    running_episode_stats2["count"])
                
                
                for k, v in infos.items():
                    if k not in running_episode_stats:
                        running_episode_stats[k] = torch.zeros(v.shape[0],1) #zeros_like running_episode_stats["count"])
                        print('INFO ',k,' ',running_episode_stats[k].shape,v.shape) 
                    #running_episode_stats[k][stats_inds] += _masked(v)
                for k, v in infos1.items():
                    if k not in running_episode_stats1:
                        running_episode_stats1[k] = torch.zeros(v.shape[0],1) #zeros_like running_episode_stats["count"])
                        print('INFO1 ',k,' ',running_episode_stats1[k].shape,v.shape)
                for k, v in infos2.items():
                    if k not in running_episode_stats2:
                        running_episode_stats2[k] = torch.zeros(v.shape[0],1) #zeros_like running_episode_stats["count"])
                        print('INFO2 ',k,' ',running_episode_stats2[k].shape,v.shape)         
                        
                
                #print(infos.keys())
                #print(running_episode_stats['distanceFromStart'].view(-1,).max(),dones.view(-1,).max())
                running_episode_stats['distanceFromStart'][stats_inds] += torch.where(dones, infos['distanceFromStart'], infos['distanceFromStart'].new_zeros(()))
                running_episode_stats['numVisited'][stats_inds] += torch.where(dones, infos['numVisited'], infos['numVisited'].new_zeros(()))
                running_episode_stats['exploredSpace'][stats_inds] += torch.where(dones, infos['exploredSpace'], infos['exploredSpace'].new_zeros(()))
                
                running_episode_stats1['distanceFromStart_'][stats_inds1] += torch.where(dones1, infos1['distanceFromStart_'], infos1['distanceFromStart_'].new_zeros(()))
                running_episode_stats1['numVisited_'][stats_inds1] += torch.where(dones1, infos1['numVisited_'], infos1['numVisited_'].new_zeros(()))
                running_episode_stats1['exploredSpace_'][stats_inds1] += torch.where(dones1, infos1['exploredSpace_'], infos1['exploredSpace_'].new_zeros(()))
                
                running_episode_stats2['distanceFromStart__'][stats_inds2] += torch.where(dones2, infos2['distanceFromStart__'], infos2['distanceFromStart__'].new_zeros(()))
                running_episode_stats2['numVisited__'][stats_inds2] += torch.where(dones2, infos2['numVisited__'], infos2['numVisited__'].new_zeros(()))
                running_episode_stats2['exploredSpace__'][stats_inds2] += torch.where(dones2, infos2['exploredSpace__'], infos2['exploredSpace__'].new_zeros(()))
                running_episode_stats2['success'][stats_inds2] += torch.where(dones2, infos2['success'], infos2['success'].new_zeros(()))
                running_episode_stats2['spl'][stats_inds2] += torch.where(dones2, infos2['spl'], infos2['spl'].new_zeros(()))
                running_episode_stats2['distanceToGoal'][stats_inds2] += torch.where(dones2, infos2['distanceToGoal'], infos2['distanceToGoal'].new_zeros(()))
                
                
      
                current_episode_reward[stats_inds].masked_fill_(dones, 0)
                current_episode_reward1[stats_inds1].masked_fill_(dones1, 0)
                current_episode_reward2[stats_inds2].masked_fill_(dones2, 0)

    def _collect_rollout_step(self, rollouts, current_episode_reward, running_episode_stats):
        with self.timing.add_time("Rollout-Step"):
            with torch.no_grad(), self.timing.add_time("Inference"):
                with torch.no_grad():
                    step_observation = {
                        k: v[rollouts.step] for k, v in rollouts.observations.items()
                    }

                    (
                        values,
                        dist_result,
                        recurrent_hidden_states,
                    ) = self.actor_critic.act(
                        step_observation,
                        rollouts.recurrent_hidden_states[rollouts.step],
                        rollouts.prev_actions[rollouts.step],
                        rollouts.masks[rollouts.step],
                    )

                    cpu_actions = actions.squeeze(-1).to(device="cpu")

            with self.timing.add_time("Habitat-Step-Start"):
                self.envs.async_step(cpu_actions)

            with self.timing.add_time("Habitat-Step-Wait"):
                batch, rewards, masks, infos = self.envs.wait_step()

            with self.timing.add_time("Renderer-Render"):
                sync = self._draw_batch(batch)

            with self.timing.add_time("Update-Stats"):
                current_episode_reward += rewards
                running_episode_stats["reward"] += (1 - masks) * current_episode_reward
                running_episode_stats["count"] += 1 - masks
                for k, v in infos.items():
                    if k not in running_episode_stats:
                        running_episode_stats[k] = torch.zeros_like(
                            running_episode_stats["count"]
                        )

                    running_episode_stats[k] += (1 - masks) * v

                current_episode_reward *= masks

            with self.timing.add_time("Rollouts-Insert"):
                rollouts.insert(
                    rewards=rewards, masks=masks,
                )

            with self.timing.add_time("Renderer-Wait"):
                batch = self._fill_batch_result(batch, sync)

            with self.timing.add_time("Rollouts-Insert"):
                rollouts.insert(batch)

            rollouts.advance()

        return self.envs.num_envs

    @staticmethod
    def _update_agent_internal_fn(
        rollouts, rollouts1, rollouts2, agent, actor_critic, _static_encoder, timing, warmup=False
    ):
        actor_critic.train()
        if _static_encoder:
            _encoder.eval()

        with timing.add_time("PPO"):
            value_loss, action_loss, dist_entropy, value_loss1, action_loss1, dist_entropy1, value_loss2, action_loss2, dist_entropy2 = agent.update(
                rollouts, rollouts1, rollouts2, timing, warmup=warmup
            )

        rollouts.after_update()
        rollouts1.after_update()
        rollouts2.after_update()

        return (value_loss, action_loss, dist_entropy, value_loss1, action_loss1, dist_entropy1, value_loss2, action_loss2, dist_entropy2)

    def _compute_returns(self, ppo_cfg, rollouts, rollouts1, rollouts2):
        with self.timing.add_time("Learning"), torch.no_grad(), self.timing.add_time(
            "Inference"
        ):
            for idx in range(len(rollouts)):
                last_input = tree_select(rollouts[idx].step, rollouts[idx].storage_buffers)
                last_input1 = tree_select(rollouts1[idx].step, rollouts1[idx].storage_buffers)
                last_input2 = tree_select(rollouts2[idx].step, rollouts2[idx].storage_buffers)

                next_value = self.actor_critic.get_value(
                    last_input["observations"],
                    last_input["recurrent_hidden_states"],
                    last_input["prev_actions"],
                    last_input["masks"])
                
                next_value1 = self.actor_critic.get_value(
                    last_input1["observations"],
                    last_input1["recurrent_hidden_states"],
                    last_input1["prev_actions"],
                    last_input1["masks"])
                
                next_value2 = self.actor_critic.get_value(
                    last_input2["observations"],
                    last_input2["recurrent_hidden_states"],
                    last_input2["prev_actions"],
                    last_input2["masks"])
                


                with self.timing.add_time("Compute-Returns"):
                    rollouts[idx].compute_returns(next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau)
                    rollouts1[idx].compute_returns(next_value1, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau)
                    rollouts2[idx].compute_returns(next_value2, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau)

    def _update_agent(self, rollouts, rollouts1, rollouts2, warmup=False):
        with self.timing.add_time("Learning"):
            losses = self._update_agent_internal_fn(
                rollouts,
                rollouts1,
                rollouts2,
                self.agent,
                self.actor_critic,
                self._static_encoder,
                self.timing,
                warmup=warmup,
            )

            return losses