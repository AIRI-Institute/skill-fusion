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
For convenience, we provide two dockerfiles:
- The Dockerfile for interactive work with our SkillFusion agent via JupyterLab

#### JupyterLab Docker Container (Interactive)

#### SkillFusion Evalutation Docker Container (Eval-AI submission-like)

#### SkillTron Evalutation Docker Container (Eval-AI submission-like)

#### Singularity Container

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

### SkillFusion
```bash
  cd ~/skill-fusion/root/skillfusion/;
  export PYTHONPATH=/root/PONI/:$PYTHONPATH;
  python main.py
```
### SkillTron
```bash
  cd ~/skill-fusion/root/skilltron/;
  export PYTHONPATH=/root/PONI/:$PYTHONPATH;
  python main.py
```