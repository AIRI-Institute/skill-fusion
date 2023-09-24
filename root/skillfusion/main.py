#! /usr/bin/env python

import sys
sys.path.append('/home/AI/yudin.da/zemskova_ts/skill-fusion/root')

import warnings
warnings.filterwarnings("ignore")

#import rospy
import habitat
import numpy as np
from habitat_map.env_orb import Env
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from habitat.config.default_structured_configs import register_hydra_plugin, HeadingSensorConfig, TopDownMapMeasurementConfig
from skillfusion_agent import SkillFusionAgent
from habitat.config.default import get_config, patch_config
from omegaconf import OmegaConf
import os
import yaml
import argparse

DEFAULT_RATE = 30
DEFAULT_AGENT_TYPE = 'keyboard'
DEFAULT_GOAL_RADIUS = 0.25
DEFAULT_MAX_ANGLE = 0.1


class HabitatChallengeConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="habitat_challenge",
            path="file:/configs",
        )


class HabitatRunner():
    def __init__(self):
        fin = open('config_poni_exploration.yaml', 'r')
        config = yaml.safe_load(fin)
        fin.close()
        
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--input-type",
            default="rgbd",
            choices=["blind", "rgb", "depth", "rgbd"],
        )
        parser.add_argument(
            "--evaluation", type=str, required=True, choices=["local", "remote"]
        )
        parser.add_argument("--model-path", default="", type=str)
        parser.add_argument(
            "--task", required=True, type=str, choices=["objectnav", "imagenav"]
        )
        parser.add_argument("--task-config", type=str, required=True)
        parser.add_argument(
            "--action_space",
            type=str,
            default="velocity_controller",
            choices=[
                "velocity_controller",
                "waypoint_controller",
                "discrete_waypoint_controller",
            ],
            help="Action space to use for the agent",
        )
        parser.add_argument(
            "overrides",
            default=None,
            nargs=argparse.REMAINDER,
            help="Modify config options from command line",
        )

        benchmark_config_path =  "/home/AI/yudin.da/zemskova_ts/skill-fusion/root/configs/benchmark/nav/objectnav/objectnav_v2_hm3d_stretch_challenge.yaml"
        args = parser.parse_args('--evaluation local --model-path objectnav_baseline_habitat_navigation_challenge_2023.pth --input-type rgbd --task objectnav --action_space discrete_waypoint_controller --task-config /home/AI/yudin.da/zemskova_ts/skill-fusion/root/configs/ddppo_objectnav_v2_hm3d_stretch.yaml'.split())
        
        register_hydra_plugin(HabitatChallengeConfigPlugin)
        overrides = args.overrides + [
                '+habitat_baselines.rl.policy.action_distribution_type=categorical', 
                '+habitat_baselines.load_resume_state_config=True',
                "habitat/task/actions=" + args.action_space,
                "habitat.dataset.split=val",
                "habitat.dataset.data_path=/data/datasets/objectnav/hm3d/v2/val/val.json.gz",
                "+pth_gpu_id=0",
                "+input_type=" + args.input_type,
                "+model_path=" + args.model_path,
                "+random_seed=7",
            ]
        print('OVERRIDES:', overrides)
        os.environ["CHALLENGE_CONFIG_FILE"] = "/home/AI/yudin.da/zemskova_ts/skill-fusion/root/configs/benchmark/nav/objectnav/objectnav_v2_hm3d_stretch_challenge.yaml"

        # Now define the config for the sensor
        habitat_path = '/home/kirill/habitat-lab/data'
        task_config = get_config(config['task']['config'], overrides)
        OmegaConf.set_readonly(task_config, False)

        # Initialize environment
        self.env = Env(config=task_config)

        # Initialize metric counters
        self.eval_episodes = config['task']['eval_episodes']
        self.successes = []
        self.spls = []
        self.softspls = []
        self.fake_finishes = 0

        # initialize agent
        self.agent = SkillFusionAgent(task_config)

        self.top_down_map_save_path = config['task']['top_down_map_save_path']
        self.last_pic_save_path = config['task']['last_pic_save_path']


    def run_episode(self, ii):
        print('Run episode')
        observations = self.env.reset(i=ii)
        print('Episode number', ii)
        objectgoal = observations['objectgoal'][0]
        objectgoal_name = {v: k for k, v in self.env.task._dataset.category_to_task_category_id.items()}[objectgoal]
        print('Objectgoal:', objectgoal_name)

        self.agent.reset()

        # Run the simulator with agent
        observations = self.env.step(HabitatSimActions.move_forward)

        finished = False
        step = 0
        while not self.env.episode_over:
            step += 1
            action = self.agent.act(observations)
            action = action['action']
            observations = self.env.step(action)
            if action == HabitatSimActions.stop:
                finished = True
                break

        # Calculate and show metrics
        metrics = self.env.task.measurements.get_metrics()
        print('METRICS:', metrics)

        if finished and metrics['success'] == 0:
            self.fake_finishes += 1
        print('Success:', metrics['success'])
        print('SPL:', metrics['spl'])
        print('SoftSPL:', metrics['soft_spl'])
        self.successes.append(metrics['success'])
        self.spls.append(metrics['spl'])
        self.softspls.append(metrics['soft_spl'])
        print('Average success:', np.mean(self.successes))
        print('Average SPL:', np.mean(self.spls))
        print('Average softSPL:', np.mean(self.softspls))
        print('Number of false goal detections:', self.fake_finishes)
                

def main():
    runner = HabitatRunner()
    for i in runner.eval_episodes:
        runner.run_episode(i)
    fout = open('fbe_maps/results.txt', 'w')
    print('Success: {}'.format(np.mean(runner.successes)), file=fout)
    print('SPL: {}'.format(np.mean(runner.spls)), file=fout)
    print('SoftSPL: {}'.format(np.mean(runner.softspls)), file=fout)
    fout.close()


if __name__ == '__main__':
    main()