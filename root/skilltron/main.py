#! /usr/bin/env python

import sys
sys.path.append('/root')

import warnings
warnings.filterwarnings("ignore")

import argparse
import datetime
#import rospy
import habitat
import numpy as np
from habitat_map.env_orb import Env
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from habitat.config.default_structured_configs import register_hydra_plugin, HeadingSensorConfig, TopDownMapMeasurementConfig
from skilltron_agent import SkillTronAgent
#from poni_agent import PoniAgent
from habitat.config.default import get_config, patch_config
from omegaconf import OmegaConf
from habitat_map.utils import draw_top_down_map
from skimage.io import imsave
from PIL import Image
from time import time
import cv2
import os
import yaml
import subprocess
import argparse
from utils import draw_top_down_map
import random
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
        #rospy.init_node('habitat_runner')
        fin = open('config_skilltron.yaml', 'r')
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

        benchmark_config_path =  "/root/configs/benchmark/nav/objectnav/objectnav_v2_hm3d_stretch_challenge.yaml"
        args = parser.parse_args('--evaluation local --model-path objectnav_baseline_habitat_navigation_challenge_2023.pth --input-type rgbd --task objectnav --action_space discrete_waypoint_controller --task-config /home/AI/yudin.da/zemskova_ts/skill-fusion/root/configs/ddppo_objectnav_v2_hm3d_stretch.yaml'.split())
        
        register_hydra_plugin(HabitatChallengeConfigPlugin)
        overrides = args.overrides + [
                '+habitat_baselines.rl.policy.action_distribution_type=categorical', 
                '+habitat_baselines.load_resume_state_config=True',
                #"+benchmark/nav/" + args.task + "=" + os.path.basename(benchmark_config_path),
                "habitat.dataset.scenes_dir=/data/scene_datasets",
                "habitat/task/actions=" + args.action_space,
                "habitat.dataset.split=val",
                "habitat.simulator.scene_dataset=/data/scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json",
                "habitat.dataset.data_path=/data/datasets/objectnav/hm3d/v2/val/val.json.gz",
                "+pth_gpu_id=0",
                "+input_type=" + args.input_type,
                "+model_path=" + args.model_path,
                "+random_seed=7",
            ]
        print('OVERRIDES:', overrides)
        os.environ["CHALLENGE_CONFIG_FILE"] = "/root/configs/benchmark/nav/objectnav/objectnav_v2_hm3d_stretch_challenge.yaml"

        # Now define the config for the sensor
        habitat_path = '/home/kirill/habitat-lab/data'
        task_config = get_config(config['task']['config'], overrides)
        OmegaConf.set_readonly(task_config, False)
        

        with habitat.config.read_write(task_config):    
            task_config.habitat.task.lab_sensors.update(
                {"heading_sensor": HeadingSensorConfig()})
        with habitat.config.read_write(task_config):
            task_config.habitat.task.measurements.update(
                {"top_down_map": TopDownMapMeasurementConfig()})
        with habitat.config.read_write(task_config):
            task_config.habitat.task.actions.turn_left_waypoint.turn_angle = 15 * 2 * np.pi / 360 # 15 degrees
            task_config.habitat.task.actions.turn_right_waypoint.turn_angle = 15 * 2 * np.pi / 360 # 15 degrees
        # Initialize environment
        self.env = Env(config=task_config)

        # Initialize metric counters
        self.eval_episodes = config['task']['eval_episodes']
        #self.eval_episodes = [297, 299, 301]
        self.successes = []
        self.spls = []
        self.softspls = []
        self.fake_finishes = 0

        # initialize agent
        #path_follower_config = config['path_follower']
        #agent_type = path_follower_config['type']
        agent_type = 'greedy'
        if agent_type == 'greedy':
            self.agent = SkillFusionAgent(task_config)
        elif agent_type == 'poni':
            self.agent = PoniAgent(task_config)
        elif agent_type == 'oneformer':
            self.agent = Agent_hlpo(task_config)
        else:
            print('AGENT TYPE {} IS NOT DEFINED!!!'.format(agent_type))
            self.agent = None

        self.top_down_map_save_path = config['task']['top_down_map_save_path']
        self.last_pic_save_path = config['task']['last_pic_save_path']


    def run_episode(self, ii):
        print('Run episode')
        observations = self.env.reset(i=ii)
        print('Episode number', ii)
        objectgoal = observations['objectgoal'][0]
        objectgoal_name = {v: k for k, v in self.env.task._dataset.category_to_task_category_id.items()}[objectgoal]
        print('Objectgoal:', objectgoal_name)
        
        save = False
        self.agent.reset()

        # Run the simulator with agent
        #step_start_time = rospy.Time.now()
        observations = self.env.step(HabitatSimActions.move_forward)

        finished = False
        cost_values = []
        agent_trajectory = []
        #while not rospy.is_shutdown() and not self.env.episode_over:
        step = 0
        rgbs = []
        depths = []
        gt_sems = []
        pred_sems = []
        view_angles = []
        while not self.env.episode_over:
            step += 1
            #cost_values.append(self.agent.exploration.goal_cost)
            robot_x, robot_y = observations['gps']
            robot_y = -robot_y
            agent_trajectory.append([robot_x, robot_y])
            #action = self.agent.act(observations)
            pred = self.agent.act(observations)
            action = pred['action']
        
            if save:
                rgbs.append(observations['rgb'])
                depths.append(observations['depth'])
                gt_sems.append(observations['semantic'][:,:,0])
                pred_sems.append(pred['pred'])
                img_sem = Image.fromarray(observations['semantic'][:,:,0].astype(np.uint8)).convert('P')
                print(observations['semantic'][:,:,0].astype(np.uint8).shape)
                print(np.unique(observations['semantic'][:,:,0].astype(np.uint8)))
                print(pred['pred'].astype(np.uint8).shape)
                print(np.unique(pred['pred'].astype(np.uint8)))
                view_angles.append(self.agent.exploration.sem_map_module.agent_view_angle)

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

        # Save top_down_map

        if self.top_down_map_save_path is not None:
            top_down_map = draw_top_down_map(metrics,
                                             observations['compass'][0],
                                             observations['rgb'][0].shape[0])
            im = Image.fromarray(top_down_map)
            im.save(os.path.join(self.top_down_map_save_path, 'episode_{}_{}_{}_{}.png'.format(ii, objectgoal_name, metrics['success'], metrics['spl'])))
        if self.last_pic_save_path is not None:
            imsave(os.path.join(self.last_pic_save_path, 'episode_{}_{}.png'.format(ii, objectgoal_name)), observations['rgb'])
        """
        np.savetxt(os.path.join(self.top_down_map_save_path, 'cost_values', 'episode_{}_{}.txt'.format(ii, objectgoal_name)), np.array(cost_values))
        np.savetxt(os.path.join(self.top_down_map_save_path, 'critic_values', 'episode_{}_{}.txt'.format(ii, objectgoal_name)), np.array(critic_values))
        np.savetxt(os.path.join(self.top_down_map_save_path, 'trajectories', 'episode_{}_{}.txt'.format(ii, objectgoal_name)), np.array(agent_trajectory))
        """
        if save:
            np.savetxt(os.path.join('observations_dataset_15/trajectories', 'episode_{}_{}.txt'.format(ii, objectgoal_name)), np.array(agent_trajectory))
            np.savetxt(os.path.join('observations_dataset_15/poses', 'episode_{}_{}.txt'.format(ii, objectgoal_name)), np.array(self.agent.robot_pose_track))
            np.savetxt(os.path.join('observations_dataset_15/view_angles', 'episode_{}_{}.txt'.format(ii, objectgoal_name)), np.array(view_angles))    
        save_dir = 'skillfusion_maps/episode_{}_{}_{}_{}'.format(ii, objectgoal_name, metrics['success'], round(metrics['spl'], 3))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        #top_down_map = draw_top_down_map(self.env.info, observations['heading'][0], observations['rgb'][0].shape[0])
        #imsave(os.path.join(save_dir, 'top_down_map.png'), top_down_map)
        #np.savez(os.path.join(save_dir, 'maps.npz'), self.agent.maps)
        np.savez(os.path.join(save_dir, 'semantic_maps.npz'), self.agent.semantic_maps)
        np.savez(os.path.join(save_dir, 'rgbs.npz'), self.agent.rgbs)
        np.savez(os.path.join(save_dir, 'depths.npz'), self.agent.depths)
        np.savez(os.path.join(save_dir, 'depths_for_map.npz'), self.agent.depths_for_map)
        np.savetxt(os.path.join(save_dir, 'poses.txt'), self.agent.robot_pose_track)
        np.savetxt(os.path.join(save_dir, 'goal_coords.txt'), self.agent.goal_coords)
        np.savez(os.path.join(save_dir, 'actions.txt'), self.agent.action_track)
        np.savez(os.path.join(save_dir, 'obs_maps.npz'), self.agent.obs_maps)
        np.savetxt(os.path.join(save_dir, 'agent_positions.txt'), self.agent.agent_positions)
        np.savetxt(os.path.join(save_dir, 'goal_coords_ij.txt'), self.agent.goal_coords_ij)
        np.savez(os.path.join(save_dir, 'agent_views.txt'), self.agent.agent_views)
        np.savetxt(os.path.join(save_dir, 'st_poses.txt'), self.agent.st_poses)
        np.savetxt(os.path.join(save_dir, 'pose_shifts.txt'), self.agent.pose_shifts)
        fout = open(os.path.join(save_dir, 'results.txt'), 'w')
        print('Success: {}'.format(metrics['success']), file=fout)
        print('SPL: {}'.format(metrics['spl']), file=fout)
        print('SoftSPL: {}'.format(metrics['soft_spl']), file=fout)
        fout.close()
        fout = open(os.path.join(save_dir, 'path_to_goal.txt'), 'w')
        for path in self.agent.paths:
            for x, y in path:
                print(x, y, end=' ', file=fout)
            print('', file=fout)
        fout.close()
        
        #subprocess.run(['python', 'create_gif.py', save_dir, 'skillfusion_results'])
        for file in os.listdir(save_dir):
            if file != 'results.txt':
                try:
                    os.remove(os.path.join(save_dir, file))
                except IsADirectoryError:
                    pass
                

def main():

    runner = HabitatRunner()
    for i in runner.eval_episodes:
        runner.run_episode(i)
    fout = open(f"results_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{datetime.datetime.now()}.txt", 'w')
    print('Config: {}'.format('config_poni_exploration.yaml'), file=fout)
    print('Success: {}'.format(np.mean(runner.successes)), file=fout)
    print('SPL: {}'.format(np.mean(runner.spls)), file=fout)
    print('SoftSPL: {}'.format(np.mean(runner.softspls)), file=fout)
    print('Number of false goal detections: {}'.format(runner.fake_finishes), file=fout)
    print('Number of false goal detections: {}'.format(runner.fake_finishes), file=fout)
    fout.close()


if __name__ == '__main__':
    main()