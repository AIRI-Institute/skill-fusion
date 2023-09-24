# Skill Fusion in Hybrid Robotic Framework for Visual Object Goal Navigation
This is a PyTorch implementation of our work:

Aleksei Staroverov, Kirill Muravyev, Tatiana Zemskova, Dmitry Yudin and Aleksandr I. Panov<br />
AIRI, MIPT, FRCCSC

Winner of the [CVPR 2023 Habitat ObjectNav Challenge](https://aihabitat.org/challenge/2023/).

### Overview:
In recent years, Embodied AI has become one of the main topics in robotics. For the agent to operate in human-centric environments, it needs the ability to explore previously unseen areas and to navigate to objects that humans want the agent to interact with. This task, which can be formulated as ObjectGoal Navigation (ObjectNav), is the main focus of this work. To solve this challenging problem, we suggest a hybrid framework consisting of both not-learnable and learnable modules and a switcher between them -- SkillFusion. The former are more accurate while the latter are more robust to sensors' noise. To mitigate the sim-to-real gap which often arises with learnable methods we suggest training them in a way that they are less environment-dependent. As a result, our method showed both the top results in the Habitat simulator and during the evaluations on a real robot.

![overview](./docs/objectnav_spec_2023.gif)

### SegmATRon
To change versions of SegmATRon go to [semantic_predictor_segmatron_multicat.py](./root/skillfusion/semantic_predictor_segmatron_multicat.py) and change paths to config file and weights in SemanticPredictor initialization.
Available options:

- **Single Frame Baseline** weights/config_single_frame.yaml, weights: https://drive.google.com/file/d/1N4DeNte-nr7Yn8DBef52nV6zGHA19WW0/view?usp=drive_link
- **SegmATRon 1 step** weights/segmatron_1_step.yaml, weights: https://drive.google.com/file/d/1oqSR7CThQvE9eJdib3nG93OAMuibACxG/view?usp=drive_link
- **SegmATRon 2 steps** weights/segmatron_2_step.yaml, weights: https://disk.yandex.ru/d/a-LWtayiJf54VA
- **SegmATRon 4 steps** weights/segmatron_4_steps.yaml, weights: https://disk.yandex.ru/d/goW34Me8jbme9A
To control real-time or delayed semantic predictions use config_poni_exploration.yaml:

```bash
semantic_predictor:
  config_path: 'weights/segmatron_1_step.yaml'
  weights: 'weights/segmatron_1_steps_15.pt'
  delayed: False
```