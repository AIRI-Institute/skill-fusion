#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.distributed as distrib

from bps_nav_test.common.rollout_storage import RolloutStorage
from bps_nav_test.rl.ppo import PPO

EPS_PPO = 1e-5


def distributed_mean_and_var(
    values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Computes the mean and variances of a tensor over multiple workers.

    This method is equivalent to first collecting all versions of values and
    then computing the mean and variance locally over that

    :param values: (*,) shaped tensors to compute mean and variance over.  Assumed
                        to be solely the workers local copy of this tensor,
                        the resultant mean and variance will be computed
                        over _all_ workers version of this tensor.
    """
    assert distrib.is_initialized(), "Distributed must be initialized"

    world_size = distrib.get_world_size()
    mean = values.mean()
    distrib.all_reduce(mean)
    mean /= world_size

    sq_diff = (values - mean).pow(2).mean()
    distrib.all_reduce(sq_diff)
    var = sq_diff / world_size

    return mean, var


class DecentralizedDistributedMixin:
    def _get_advantages_distributed(self, rollouts: RolloutStorage) -> torch.Tensor:
        advantages = []
        for idx in range(len(rollouts.buffers)):
            advantages.append(
                rollouts[idx].storage_buffers["returns"][:-1]
                - rollouts[idx].storage_buffers["value_preds"][:-1]
            )

        advantages = torch.cat(advantages, 1)

        if not self.use_normalized_advantage:
            return advantages

        mean, var = distributed_mean_and_var(advantages)
        beta = 0.8
        self.adv_mean_biased.mul_(beta).add_(mean, alpha=(1 - beta))
        self.adv_mean_unbias.mul_(beta).add_(1 - beta)

        return advantages - (self.adv_mean_biased / self.adv_mean_unbias)

    def init_distributed(self, find_unused_params: bool = True) -> None:
        r"""Initializes distributed training for the model

        1. Broadcasts the model weights from world_rank 0 to all other workers
        2. Adds gradient hooks to the model

        :param find_unused_params: Whether or not to filter out unused parameters
                                   before gradient reduction.  This *must* be True if
                                   there are any parameters in the model that where unused in the
                                   forward pass, otherwise the gradient reduction
                                   will not work correctly.
        """
        # NB: Used to hide the hooks from the nn.Module,
        # so they don't show up in the state_dict
        class Guard:
            def __init__(self, model, device):
                if torch.cuda.is_available():
                    self.ddp = torch.nn.parallel.DistributedDataParallel(
                        model, device_ids=[device], output_device=device
                    )
                else:
                    self.ddp = torch.nn.parallel.DistributedDataParallel(model)

        self._ddp_hooks = Guard(self.actor_critic, self.device)
        self.get_advantages = self._get_advantages_distributed

        self.reducer = self._ddp_hooks.ddp.reducer
        self.find_unused_params = find_unused_params

    def before_backward(self, loss, will_step_optim=False):
        super().before_backward(loss, will_step_optim)

        if not will_step_optim:
            return

        if self.find_unused_params:
            self.reducer.prepare_for_backward([loss])
        else:
            self.reducer.prepare_for_backward([])


class DDPPO(DecentralizedDistributedMixin, PPO):
    pass
