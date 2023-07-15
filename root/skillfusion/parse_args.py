import argparse
import torch
from habitat.config import DictConfig

def add_argument(parser, value, name, type=int, nargs=None, action=None, default=None, help=None):
    if action is not None:
        if value is None:
            parser.add_argument(name, action=action, default=default, help=help)
        else:
            parser.add_argument(name, action=action, default=value, help=help)
    elif nargs is not None:
        if value is None:
            parser.add_argument(name, type=type, nargs=nargs, default=default, help=help)
        else:
            parser.add_argument(name, type=type, nargs=nargs, default=value, help=help)
    else:
        if value is None:
            parser.add_argument(name, type=type, default=default, help=help)
        else:
            parser.add_argument(name, type=type, default=value, help=help)

def parse_args_from_config(config: DictConfig):
    parser = argparse.ArgumentParser(description="Goal-Oriented-Semantic-Exploration")

    # General Arguments
    add_argument(parser, config.get('seed', None), "--seed", type=int, default=1, help="random seed (default: 1)")
    add_argument(parser, config.get('auto_gpu_config', None), "--auto_gpu_config", type=int, default=1)
    add_argument(parser, config.get('total_num_scenes', None), "--total_num_scenes", type=str, default="auto")
    add_argument(parser, config.get('num_processes', None),
        "--num_processes",
        type=int,
        default=5,
        help="""how many training processes to use (default:5)
                                Overridden when auto_gpu_config=1
                                and training on gpus""",
    )
    add_argument(parser, config.get('num_processes_per_gpu', None), "--num_processes_per_gpu", type=int, default=6)
    add_argument(parser, config.get('num_processes_on_first_gpu', None), "--num_processes_on_first_gpu", type=int, default=1)
    add_argument(parser, config.get('eval', None),
        "--eval", type=int, default=0, help="0: Train, 1: Evaluate (default: 0)"
    )
    add_argument(parser, config.get('num_training_frames', None),
        "--num_training_frames",
        type=int,
        default=10000000,
        help="total number of training frames",
    )
    add_argument(parser, config.get('num_eval_episodes', None),
        "--num_eval_episodes",
        type=int,
        default=200,
        help="number of test episodes per scene",
    )
    add_argument(parser, config.get('num_train_episodes', None),
        "--num_train_episodes",
        type=int,
        default=10000,
        help="""number of train episodes per scene
                                before loading the next scene""",
    )
    add_argument(parser, config.get('no_cuda', None),
        "--no_cuda", action="store_true", default=False, help="disables CUDA training"
    )
    add_argument(parser, config.get('sim_gpu_id', None),
        "--sim_gpu_id", type=int, default=0, help="gpu id on which scenes are loaded"
    )
    add_argument(parser, config.get('sem_gpu_id', None),
        "--sem_gpu_id",
        type=int,
        default=-1,
        help="""gpu id for semantic model,
                                -1: same as sim gpu, -2: cpu""",
    )

    # Logging, loading models, visualization
    add_argument(parser, config.get('log_interval', None),
        "--log_interval",
        type=int,
        default=10,
        help="""log interval, one log per n updates
                                (default: 10) """,
    )
    add_argument(parser, config.get('save_interval', None),
        "--save_interval", type=int, default=1, help="""save interval"""
    )
    add_argument(parser, config.get('dump_location', None),
        "--dump_location",
        type=str,
        default="./tmp/",
        help="path to dump models and log (default: ./tmp/)",
    )
    add_argument(parser, config.get('exp_name', None),
        "--exp_name", type=str, default="exp1", help="experiment name (default: exp1)"
    )
    add_argument(parser, config.get('load', None),
        "--load",
        type=str,
        default="0",
        help="""model path to load,
                                0 to not reload (default: 0)""",
    )
    add_argument(parser, config.get('visualize', None),
        "--visualize",
        type=int,
        default=0,
        help="""1: Render the observation and
                                   the predicted semantic map,
                                2: Render the observation with semantic
                                   predictions and the predicted semantic map
                                (default: 0)""",
    )
    add_argument(parser, config.get('print_images', None),
        "--print_images", type=int, default=0, help="1: save visualization as images"
    )

    # Environment, dataset and episode specifications
    add_argument(parser, config.get('object_cat_offset', None),
        "--object_cat_offset",
        type=int,
        default=1,
        help="Category ID offset to get objects in SemMap",
    )
    add_argument(parser, config.get('env_frame_width', None),
        "--env_frame_width",
        type=int,
        default=640,
        help="Frame width (default:640)",
    )
    add_argument(parser, config.get('env_frame_height', None),
        "--env_frame_height",
        type=int,
        default=480,
        help="Frame height (default:480)",
    )
    add_argument(parser, config.get('frame_width', None),
        "--frame_width", type=int, default=160, help="Frame width (default:160)"
    )
    add_argument(parser, config.get('frame_height', None),
        "--frame_height",
        type=int,
        default=120,
        help="Frame height (default:120)",
    )
    add_argument(parser, config.get('max_episode_length', None),
        "--max_episode_length",
        type=int,
        default=500,
        help="""Maximum episode length""",
    )
    add_argument(parser, config.get('task_config', None),
        "--task_config",
        type=str,
        default="tasks/objectnav_gibson.yaml",
        help="path to config yaml containing task information",
    )
    add_argument(parser, config.get('split', None),
        "--split",
        type=str,
        default="train",
        help="dataset split (train | val | val_mini) ",
    )
    add_argument(parser, config.get('camera_height', None),
        "--camera_height",
        type=float,
        default=0.88,
        help="agent camera height in metres",
    )
    add_argument(parser, config.get('hfov', None),
        "--hfov", type=float, default=79.0, help="horizontal field of view in degrees"
    )
    add_argument(parser, config.get('turn_angle', None),
        "--turn_angle", type=float, default=30, help="Agent turn angle in degrees"
    )
    add_argument(parser, config.get('min_depth', None),
        "--min_depth",
        type=float,
        default=0.5,
        help="Minimum depth for depth sensor in meters",
    )
    add_argument(parser, config.get('max_depth', None),
        "--max_depth",
        type=float,
        default=5.0,
        help="Maximum depth for depth sensor in meters",
    )
    add_argument(parser, config.get('success_dist', None),
        "--success_dist",
        type=float,
        default=1.0,
        help="success distance threshold in meters",
    )
    add_argument(parser, config.get('floor_thr', None),
        "--floor_thr", type=int, default=50, help="floor threshold in cm"
    )
    add_argument(parser, config.get('min_d', None),
        "--min_d",
        type=float,
        default=1.5,
        help="min distance to goal during training in meters",
    )
    add_argument(parser, config.get('max_d', None),
        "--max_d",
        type=float,
        default=100.0,
        help="max distance to goal during training in meters",
    )
    add_argument(parser, config.get('version', None), "--version", type=str, default="v1.1", help="dataset version")
    add_argument(parser, config.get('num_goals', None),
        "--num_goals", type=int, default=1, help="number of goals to reach in task"
    )
    add_argument(parser, config.get('success_distance', None),
        "--success_distance",
        type=float,
        default=0.1,
        help="radius around goal locations for success",
    )

    # Model Hyperparameters
    add_argument(parser, config.get('agent', None), "--agent", type=str, default="sem_exp")
    add_argument(parser, config.get('main_model', None), "--main_model", type=str, default="simple_cnn")
    add_argument(parser, config.get('lr', None),
        "--lr", type=float, default=2.5e-5, help="learning rate (default: 2.5e-5)"
    )
    add_argument(parser, config.get('lr_schedule', None),
        "--lr_schedule",
        type=float,
        nargs="+",
        default=[],
        help="lr decay schedule for MultiStepLR (in steps)",
    )
    add_argument(parser, config.get('global_hidden_size', None),
        "--global_hidden_size", type=int, default=256, help="global_hidden_size"
    )
    add_argument(parser, config.get('eps', None),
        "--eps", type=float, default=1e-5, help="RL Optimizer epsilon (default: 1e-5)"
    )
    add_argument(parser, config.get('alpha', None),
        "--alpha", type=float, default=0.99, help="RL Optimizer alpha (default: 0.99)"
    )
    add_argument(parser, config.get('gamma', None),
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )
    add_argument(parser, config.get('use_gae', None),
        "--use_gae",
        action="store_true",
        default=False,
        help="use generalized advantage estimation",
    )
    add_argument(parser, config.get('tau', None),
        "--tau", type=float, default=0.95, help="gae parameter (default: 0.95)"
    )
    add_argument(parser, config.get('entropy_coef', None),
        "--entropy_coef",
        type=float,
        default=0.001,
        help="entropy term coefficient (default: 0.01)",
    )
    add_argument(parser, config.get('value_loss_coef', None),
        "--value_loss_coef",
        type=float,
        default=0.5,
        help="value loss coefficient (default: 0.5)",
    )
    add_argument(parser, config.get('max_grad_norm', None),
        "--max_grad_norm",
        type=float,
        default=0.5,
        help="max norm of gradients (default: 0.5)",
    )
    add_argument(parser, config.get('num_global_steps', None),
        "--num_global_steps",
        type=int,
        default=20,
        help="number of forward steps in A2C (default: 5)",
    )
    add_argument(parser, config.get('ppo_epoch', None),
        "--ppo_epoch", type=int, default=4, help="number of ppo epochs (default: 4)"
    )
    add_argument(parser, config.get('num_mini_batch', None),
        "--num_mini_batch",
        type=str,
        default="auto",
        help="number of batches for ppo (default: 32)",
    )
    add_argument(parser, config.get('clip_param', None),
        "--clip_param",
        type=float,
        default=0.2,
        help="ppo clip parameter (default: 0.2)",
    )
    add_argument(parser, config.get('use_recurrent_global', None),
        "--use_recurrent_global",
        type=int,
        default=0,
        help="use a recurrent global policy",
    )
    add_argument(parser, config.get('num_local_steps', None),
        "--num_local_steps",
        type=int,
        default=25,
        help="""Number of steps the local policy
                                between each global step""",
    )
    add_argument(parser, config.get('reward_coeff', None),
        "--reward_coeff", type=float, default=0.1, help="Object goal reward coefficient"
    )
    add_argument(parser, config.get('intrinsic_rew_coeff', None),
        "--intrinsic_rew_coeff",
        type=float,
        default=0.02,
        help="intrinsic exploration reward coefficient",
    )
    add_argument(parser, config.get('num_sem_categories', None), "--num_sem_categories", type=float, default=16)
    add_argument(parser, config.get('sem_pred_prob_thr', None),
        "--sem_pred_prob_thr",
        type=float,
        default=0.9,
        help="Semantic prediction confidence threshold",
    )
    add_argument(parser, config.get('sem_pred_weights', None),
        "--sem_pred_weights",
        type=str,
        default="../pretrained_models/maskrcnn_gibson.pth",
        help="Weights for semantic prediction",
    )

    # Mapping
    add_argument(parser, config.get('global_downscaling', None), "--global_downscaling", type=int, default=2)
    add_argument(parser, config.get('vision_range', None), "--vision_range", type=int, default=100)
    add_argument(parser, config.get('map_resolution', None), "--map_resolution", type=int, default=5)
    add_argument(parser, config.get('du_scale', None), "--du_scale", type=int, default=1)
    add_argument(parser, config.get('map_size_cm', None), "--map_size_cm", type=int, default=2400)
    add_argument(parser, config.get('cat_pred_threshold', None), "--cat_pred_threshold", type=float, default=5.0)
    add_argument(parser, config.get('map_pred_threshold', None), "--map_pred_threshold", type=float, default=1.0)
    add_argument(parser, config.get('exp_pred_threshold', None), "--exp_pred_threshold", type=float, default=1.0)
    add_argument(parser, config.get('collision_threshold', None), "--collision_threshold", type=float, default=0.20)
    add_argument(parser, config.get('seg_interval', None), "--seg_interval", type=int, default=3)
    add_argument(parser, config.get('semantic_threshold', None), "--semantic-threshold", type=int, default=3)

    # Potential-fields arguments
    add_argument(parser, config.get('pf_model_path', None),
        "--pf_model_path",
        type=str,
        default="./model.ckpt",
        help="path to PF model weights",
    )
    add_argument(parser, config.get('pf_masking_opt', None),
        "--pf_masking_opt",
        type=str,
        default="none",
        #choices=["unexplored", "none"],
    )
    add_argument(parser, config.get('use_nearest_frontier', None),
        "--use_nearest_frontier",
        action="store_true",
        default=False,
        help="uses the nearest frontier instead of argmax PF",
    )
    add_argument(parser, config.get('add_agent2loc_distance', None), "--add_agent2loc_distance", action="store_true", default=False)
    add_argument(parser, config.get('add_agent2loc_distance_v2', None),
        "--add_agent2loc_distance_v2", action="store_true", default=False
    )
    add_argument(parser, config.get('mask_nearest_locations', None), "--mask_nearest_locations", action="store_true", default=False)
    add_argument(parser, config.get('mask_size', None),
        "--mask_size",
        type=float,
        default=1.0,
        help="mask size (meters) for mask_nearest_locations option",
    )
    add_argument(parser, config.get('area_weight_coef', None), "--area_weight_coef", type=float, default=0.5)
    add_argument(parser, config.get('dist_weight_coef', None), "--dist_weight_coef", type=float, default=0.3)
    add_argument(parser, config.get('num_pf_maps', None), "--num_pf_maps", type=int, default=0)
    add_argument(parser, config.get('use_gt_segmentation', None), "--use_gt_segmentation", action="store_true", default=False)
    
    add_argument(parser, config.get('erosion_size', None), "--erosion_size", type=int, default=2)
    add_argument(parser, config.get('goal_decay', None), "--goal_decay", type=int, default=0.9)

    add_argument(parser, config.get('use_egocentric_transform', None),
        "--use_egocentric_transform", action="store_true", default=False
    )
    
    add_argument(parser, config.get('device', None), "--device", type=str, default='cuda:0')

    # parse arguments
    args = parser.parse_args("")

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        if args.auto_gpu_config:
            num_gpus = torch.cuda.device_count()
            if args.total_num_scenes != "auto":
                args.total_num_scenes = int(args.total_num_scenes)
            elif "objectnav_gibson" in args.task_config and "train" in args.split:
                args.total_num_scenes = 25
            elif "objectnav_gibson" in args.task_config and "val" in args.split:
                args.total_num_scenes = 5
            else:
                assert False, (
                    "Unknown task config, please specify" + " total_num_scenes"
                )

            # GPU Memory required for the SemExp model:
            #       0.8 + 0.4 * args.total_num_scenes (GB)
            # GPU Memory required per thread: 3.0 (GB)
            min_memory_required = max(0.8 + 0.4 * args.total_num_scenes, 3.0)
            # Automatically configure number of training threads based on
            # number of GPUs available and GPU memory size
            gpu_memory = 1000
            for i in range(num_gpus):
                gpu_memory = min(
                    gpu_memory,
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024,
                )
                assert (
                    gpu_memory > min_memory_required
                ), """Insufficient GPU memory for GPU {}, gpu memory ({}GB)
                    needs to be greater than {}GB""".format(
                    i, gpu_memory, min_memory_required
                )

            num_processes_per_gpu = int(gpu_memory / 3.0)
            num_processes_on_first_gpu = int(gpu_memory / 3.0)#int((gpu_memory - min_memory_required) / 3.0)

            if args.eval:
                max_threads = (
                    num_processes_per_gpu * (num_gpus - 1) + num_processes_on_first_gpu
                )
                #assert (
                #    max_threads >= args.total_num_scenes
                #), """Insufficient GPU memory for evaluation"""

            print("Auto GPU config:")
            print("Number of processes: {}".format(args.num_processes))
            print(
                "Number of processes on GPU 0: {}".format(
                    args.num_processes_on_first_gpu
                )
            )
            print("Number of processes per GPU: {}".format(args.num_processes_per_gpu))

            if num_gpus == 1:
                args.num_processes_on_first_gpu = min(
                    num_processes_on_first_gpu, args.total_num_scenes
                )
                args.num_processes_per_gpu = 0
                args.num_processes = min(
                    num_processes_on_first_gpu, args.total_num_scenes
                )
                assert args.num_processes > 0, "Insufficient GPU memory"
            else:
                num_threads = (
                    num_processes_per_gpu * (num_gpus - 1) + num_processes_on_first_gpu
                )
                num_threads = min(num_threads, args.total_num_scenes)
                args.num_processes_per_gpu = num_processes_per_gpu
                args.num_processes_on_first_gpu = max(
                    0, num_threads - args.num_processes_per_gpu * (num_gpus - 1)
                )
                args.num_processes = num_threads

            args.sim_gpu_id = 1

            print("Auto GPU config:")
            print("Number of processes: {}".format(args.num_processes))
            print(
                "Number of processes on GPU 0: {}".format(
                    args.num_processes_on_first_gpu
                )
            )
            print("Number of processes per GPU: {}".format(args.num_processes_per_gpu))
    else:
        args.sem_gpu_id = -2

    if args.num_mini_batch == "auto":
        args.num_mini_batch = max(args.num_processes // 2, 1)
    else:
        args.num_mini_batch = int(args.num_mini_batch)

    return args