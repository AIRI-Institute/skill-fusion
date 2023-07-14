# Skill Fusion in Hybrid Robotic Framework for Visual Object Goal Navigation
This is a PyTorch implementation of our work:

Aleksei Staroverov, Kirill Muravyev, Tatiana Zemskova, Dmitry Yudin and Aleksandr I. Panov<br />
AIRI, MIPT, FRCCSC

Winner of the [CVPR 2023 Habitat ObjectNav Challenge](https://aihabitat.org/challenge/2023/).

### Overview:
In recent years, Embodied AI has become one of the main topics in robotics. For the agent to operate in human-centric environments, it needs the ability to explore previously unseen areas and to navigate to objects that humans want the agent to interact with. This task, which can be formulated as ObjectGoal Navigation (ObjectNav), is the main focus of this work. To solve this challenging problem, we suggest a hybrid framework consisting of both not-learnable and learnable modules and a switcher between them -- SkillFusion. The former are more accurate while the latter are more robust to sensors' noise. To mitigate the sim-to-real gap which often arises with learnable methods we suggest training them in a way that they are less environment-dependent. As a result, our method showed both the top results in the Habitat simulator and during the evaluations on a real robot.

![overview](./docs/objectnav_spec_2023.gif)
