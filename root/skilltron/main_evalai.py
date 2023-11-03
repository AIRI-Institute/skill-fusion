import argparse
import os
import habitat
from config import HabitatChallengeConfigPlugin
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.config.default import get_config

import sys
from greedy_path_follower_agent_policy_frontier import GreedyPathFollowerAgent

def main():
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
    args = parser.parse_args('--evaluation local --model-path objectnav_baseline_habitat_navigation_challenge_2023.pth --input-type rgbd --task objectnav --action_space discrete_waypoint_controller --task-config /root/configs/ddppo_objectnav_v2_hm3d_stretch.yaml'.split())
    overrides = args.overrides + [
                '+habitat_baselines.rl.policy.action_distribution_type=categorical', 
                '+habitat_baselines.load_resume_state_config=True',
                #"+benchmark/nav/" + args.task + "=" + os.path.basename(benchmark_config_path),
                #"+data/scene_datasets/hm3d_v0.2/val=/data/scene_datasets/hm3d_v0.2/val",
                "habitat/task/actions=" + args.action_space,
                "habitat.dataset.split=val",
                "habitat.dataset.data_path=/data/datasets/objectnav/hm3d/v2/val/val.json.gz",
                "+pth_gpu_id=0",
                "+input_type=" + args.input_type,
                "+model_path=" + args.model_path,
                "+random_seed=7",
            ]
    
    register_hydra_plugin(HabitatChallengeConfigPlugin)

    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    print('Config paths:', config_paths)
    config = get_config(config_paths, overrides)
    print('Dataset config:', config.habitat.dataset)
    agent = GreedyPathFollowerAgent(task_config=config)

    if args.evaluation == "local":
        challenge = habitat.Challenge(eval_remote=False, action_space="discrete_waypoint_controller")
    else:
        challenge = habitat.Challenge(eval_remote=True, action_space="discrete_waypoint_controller")

    challenge.submit(agent)


if __name__ == "__main__":
    main()
