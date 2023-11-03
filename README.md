# Skill Fusion in Hybrid Robotic Framework for Visual Object Goal Navigation 
This is a PyTorch implementation of our work:

Aleksei Staroverov, Kirill Muravyev, Tatiana Zemskova, Dmitry Yudin and Aleksandr I. Panov<br />
AIRI, MIPT, FRCCSC

SkillFusion is the winner of the [CVPR 2023 Habitat ObjectNav Challenge](https://aihabitat.org/challenge/2023/).

### Overview:
In recent years, Embodied AI has become one of the main topics in robotics. For the agent to operate in human-centric environments, it needs the ability to explore previously unseen areas and to navigate to objects that humans want the agent to interact with. This task, which can be formulated as ObjectGoal Navigation (ObjectNav), is the main focus of this work. To solve this challenging problem, we suggest a hybrid framework consisting of both not-learnable and learnable modules and a switcher between them -- SkillFusion. The former are more accurate while the latter are more robust to sensors' noise. To mitigate the sim-to-real gap which often arises with learnable methods we suggest training them in a way that they are less environment-dependent. As a result, our method showed both the top results in the Habitat simulator and during the evaluations on a real robot.

![overview](./docs/objectnav_spec_2023.gif)

## SkillTron: Interactive Semantic Map Representation for Skill-based Visual Object Navigation

Additionaly, we provide a PyTorch implementation of SkillTron -- a method that further refines visual object navigation performance of SkillFusion approach by using a two-level interactive semantic map representation.

## Installation

### Choose one of provided containers
For convenience, we provide different containers that can be used to evaluate SkillFusion and SkillTron:
- A docker container for interactive work with our SkillFusion and SkillTron agents via JupyterLab
- Eval-AI submission-like docker containers that will launch a script for evaluation without access to docker container system.
- A Singularity container for evaluation on platforms that do not have Docker.

#### JupyterLab Docker Container (Interactive)

All necessary files to build and launch an interactive docker container with Jupyter Lab can be found in [docker_interactive](docker_interactive/) 

To build the correspoding docker image run:

```bash
bash build.sh
```

This will create "alstar_docker" docker image.


To start a docker-container from this image run: 

```bash
bash run.sh <PORT>
```

This will launch a Jupyter Lab server at ```localhost:<PORT>```

You may need to change paths to mounted volumes before starting the container:
```bash
-v <path_to_Habitat_sim_data>/data:/data \
-v <path_to_SkillFusion>/root/:/root alstar_docker
```

#### Evalutation Docker Containers (Eval-AI submission-like)

We provide minimal docker images to run evaluation of SkillFusion and SkillTron methods in Eval-AI submission-like mode.

**Note**: you need to download and put to expected folders pretrained weights before building evaluation containers.

- SkillFusion

  To build the minimal SkillFusion docker image, go to [docker_skillfusion](docker_skillfusion/):

  ```bash
  bash build.sh
  ```

  This will create "skillfusion_docker" docker image. Here, the code will not be mounted to docker container, but rather copied to the docker image during its built. The image will contain only the code corresponding to SkillFusion agent in the folder /root/exploration_ros_free/. 

  To evaluate SkillFusion agent run:
  ```bash
  bash test_local.sh
  ```
  You may need to change paths to mounted volumes before starting the container:
  ```bash
    -v $(pwd)/habitat-challenge-data:/data \
  ```
- SkillTron

  To build the minimal SkillTron docker image, go to [docker_skilltron](docker_skilltron/):

  ```bash
  bash build.sh
  ```

  This will create "skilltron_docker" docker image. Here, the code will not be mounted to docker container, but rather copied to the docker image during its built. The image will contain only the code corresponding to SkillTron agent in the folder /root/exploration_ros_free/. 

  To evaluate SkillTron agent run:
  ```bash
  bash test_local.sh
  ```
  You may need to change paths to mounted volumes before starting the container:
  ```bash
    -v $(pwd)/habitat-challenge-data:/data \
  ```

#### Singularity Container

We provide a .def file to build a singularity container with necessary dependencies to run SkillFusion and SkillTron agents evaluation.

Go to the folder [singularty](singularity/).

To build the singularity container:

```bash
sudo singularity build skill_fusion.sif skill_fusion.def 
```

Additionally, we provide the [skilltron_eval.sh](singularity/skilltron_eval.sh) script that can be used to launch SkillTron evaluation on slurm based clusters. To use it you may need to change paths to home and data directories as well as change partition name.


### Download Habitat Sim Dataset

Follow official instruction to download [Habitat-Matterport 3D Research Dataset (HM3D)](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md). For evaluation you will need only it's validation part.

### Download pretrained weights

- [PONI Exploration](https://disk.yandex.ru/d/6v5SQ57ACvNpRg)
- [RL Exploration](https://disk.yandex.ru/d/zIb37VHjhh17tg)
- [RL Goal Reacher](https://disk.yandex.ru/d/Q-w2nZVNDu8mDA)
- [OneFormer Semantic Predictor for SkillFusion](https://disk.yandex.ru/d/BIAb3Jyh5G-_rQ)
- [SegmATRon Single Frame Baseline Semantic Predictor for SkillTron](https://disk.yandex.ru/d/p2GKLZ_Z49IWdA)
- [SegmATRon 1 step Semantic Predictor for SkillTron](https://disk.yandex.ru/d/DwRCff70Ij3ang)
- [SegmATRon 2 steps Semantic Predictor for SkillTron](https://disk.yandex.ru/d/a-LWtayiJf54VA)
- [SegmATRon 4 steps Semantic Predictor for SkillTron](https://disk.yandex.ru/d/goW34Me8jbme9A)

**Expected Data Structure**:
```bash
skill-fusion
└── root
    ├── PONI
    │   └── pretrained_models
    │       └── gibson_models
    │           └── poni_seed_234.ckpt
    ├── skillfusion
    │   └── weights
    │       ├── config_oneformer_ade20k.yaml
    │       ├── model_oneformer_ade20k.pth
    │       ├── ex_tilt_4.pth
    │       └── grTILT_june1.pth
    └── skilltron
        └── weights
            ├── config_oneformer_ade20k.yaml
            ├── config_single_frame.yaml
            ├── segmatron_1_step.yaml
            ├── segmatron_2_step.yaml
            ├── segmatron_4_steps.yaml
            ├── single_frame_baseline.pt
            ├── segmatron_1_step.pt
            ├── segmatron_2_step.pt
            ├── segmatron_4_steps.pt
            ├── ex_tilt_4.pth
            └── grTILT_june1.pth
```


### Choose from available SkillTron options
To change versions of SegmATRon go to [config_skilltron.yaml](root/skilltron/config_skilltron.yaml) and change paths to config file and weights in SemanticPredictor initialization.
Available options:

- **Single Frame Baseline** weights/config_single_frame.yaml, [weights](https://disk.yandex.ru/d/p2GKLZ_Z49IWdA)
- **SegmATRon 1 step** weights/segmatron_1_step.yaml, [weights](https://disk.yandex.ru/d/DwRCff70Ij3ang)
- **SegmATRon 2 steps** weights/segmatron_2_step.yaml, [weights](https://disk.yandex.ru/d/a-LWtayiJf54VA)
- **SegmATRon 4 steps** weights/segmatron_4_steps.yaml, [weights](https://disk.yandex.ru/d/goW34Me8jbme9A)

To control real-time or delayed semantic predictions change parameters of semantic predictor in configuration file[config_skilltron.yaml](root/skilltron/config_skilltron.yaml):

```bash
semantic_predictor:
  config_path: 'weights/segmatron_1_step.yaml'
  weights: 'weights/segmatron_1_steps_15.pt'
  delayed: False
```

## Evaluation

If you are inside an interactive shell of the docker/singularity containers run:

### SkillFusion
```bash
  cd /root/skillfusion/;
  export PYTHONPATH=/root/PONI/:$PYTHONPATH;
  python main.py
```
### SkillTron
```bash
  cd /root/skilltron/;
  export PYTHONPATH=/root/PONI/:$PYTHONPATH;
  python main.py
```

If you have built submission-like docker images, you may use provided scripts ```test_local.sh``` (see [Evalutation Docker Containers (Eval-AI submission-like)](#evalutation-docker-containers-eval-ai-submission-like))