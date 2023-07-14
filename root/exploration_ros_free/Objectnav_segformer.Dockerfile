FROM fairembodied/habitat-challenge:testing_2022_habitat_base_docker

ENV PYTHONPATH=/opt/conda/envs/habitat/bin/python3

RUN /bin/bash -c ". activate habitat; pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html" 
RUN /bin/bash -c ". activate habitat; pip install torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html"

RUN rm -rf /etc/apt/sources.list.d/nvidia-ml.list && \
    rm -rf /etc/apt/sources.list.d/cuda.list
RUN sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list
RUN apt-key del 7fa2af80
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update && \
    apt-get install -y software-properties-common
    
RUN apt-get update && rm -rf /var/lib/apt/lists/* && \
    add-apt-repository multiverse && \
    add-apt-repository ppa:graphics-drivers && \
    apt-get update && \
    apt-get install -y nvidia-cuda-toolkit

RUN apt-get update && apt-get install -y apt-utils

RUN apt-get update
RUN apt-get update && apt-get install -y cuda-toolkit-11.1

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin 

RUN mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
RUN apt-get update


RUN apt-get install libcudnn8=8.0.5.39-1+cuda11.1
RUN apt-get install libcudnn8-dev=8.0.5.39-1+cuda11.1

# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN /bin/bash -c ". activate habitat; pip install timm==0.3.2"
RUN /bin/bash -c ". activate habitat; pip install mmcv-full==1.2.7"

RUN git clone https://github.com/NVlabs/SegFormer.git 
RUN ls
WORKDIR /SegFormer
RUN /bin/bash -c ". activate habitat; pip install -e . --user"


RUN /bin/bash -c ". activate habitat; pip install transformations"
RUN /bin/bash -c ". activate habitat; pip install h5py"
RUN /bin/bash -c ". activate habitat; pip install psutil"
RUN /bin/bash -c ". activate habitat; pip install ipython"
RUN /bin/bash -c ". activate habitat; pip install pyyaml"
RUN /bin/bash -c ". activate habitat; pip install scikit-image"
RUN /bin/bash -c ". activate habitat; pip install tf"
RUN /bin/bash -c ". activate habitat; pip install opencv-python"

WORKDIR /

ADD main_evalai.py main.py
COPY ./habitat_map habitat_map
COPY ./weights weights
COPY ./habitat-challenge-data habitat-challenge-data
COPY ./bps_nav_test bps_nav_test
ADD planners planners
ADD exploration exploration
ADD greedy_path_follower_agent.py greedy_path_follower_agent.py
ADD semantic_predictor_segformer.py semantic_predictor_segformer.py
ADD policy_fusion.py policy_fusion.py
ADD resnet_policy.py resnet_policy.py
ADD resnet_policy_fusion.py resnet_policy_fusion.py

ADD submission.sh submission.sh
ADD config.yaml config.yaml
ADD ./challenge_objectnav2022.local.rgbd.yaml /challenge_objectnav2022.local.rgbd.yaml
ENV AGENT_EVALUATION_TYPE remote

RUN mkdir -p SegFormer/weights
ADD weights/iter_160000.pth SegFormer/weights/iter_160000.pth


ENV TRACK_CONFIG_FILE "/challenge_objectnav2022.local.rgbd.yaml"

RUN apt-get --purge remove -y "*nvidia*"

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash submission.sh"]
