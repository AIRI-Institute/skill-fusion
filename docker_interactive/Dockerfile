FROM fairembodied/habitat-challenge:habitat_navigation_2023_base_docker


ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/envs/habitat/bin:$PATH  
ENV PYTHONPATH=/opt/conda/envs/habitat/bin/python3 

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git python&& \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates curl wget less sudo lsof git net-tools nano psmisc xz-utils nemo vim net-tools iputils-ping traceroute htop \
    chromium-browser xterm terminator zenity make cmake gcc libc6-dev \
    x11-xkb-utils xauth xfonts-base xkb-data \
    mesa-utils xvfb libgl1-mesa-dri libgl1-mesa-glx libglib2.0-0 libxext6 libsm6 libxrender1 \
    libglu1 libglu1:i386 libxv1 libxv1:i386 \
    libsuitesparse-dev libgtest-dev \
    libeigen3-dev libsdl1.2-dev libarmadillo-dev libsdl-image1.2-dev libsdl-dev \
    software-properties-common supervisor vim-tiny dbus-x11 x11-utils alsa-utils \
    lxde x11vnc gtk2-engines-murrine gnome-themes-standard gtk2-engines-pixbuf gtk2-engines-murrine\
    firefox libxmu-dev \
    libssl-dev:i386 libxext-dev x11proto-gl-dev \
    ninja-build meson autoconf libtool \
    zlib1g-dev libjpeg-dev ffmpeg xorg-dev python-opengl python3-opengl libsdl2-dev swig \
    libglew-dev libboost-dev libboost-thread-dev libboost-filesystem-dev libpython2.7-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt update
RUN apt install -y gcc-9 g++-9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9  

RUN dpkg --add-architecture i386
RUN apt-get update   
RUN apt install -y libprotobuf-dev protobuf-compiler build-essential libssl-dev  

RUN /bin/bash -c '. cd /; wget https://github.com/Kitware/CMake/releases/download/v3.21.3/cmake-3.21.3.tar.gz; tar -zxvf cmake-3.21.3.tar.gz; \
cd cmake-3.21.3; ./bootstrap; make; sudo make install'

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
        curl wget \
        supervisor \
        sudo \
        vim-tiny \
        net-tools \ 
        xz-utils \
        dbus-x11 x11-utils alsa-utils \
        mesa-utils libgl1-mesa-dri \
        lxde x11vnc xvfb \
        gtk2-engines-murrine gnome-themes-standard gtk2-engines-pixbuf gtk2-engines-murrine \
        firefox \
    && apt-get autoclean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/*

# tini for subreap                                   
ARG TINI_VERSION=v0.9.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /bin/tini
RUN chmod +x /bin/tini

# set default screen to 1 (this is crucial for gym's rendering)
ENV DISPLAY=:1
RUN apt-get update && apt-get install -y \
        git vim \
        python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb ffmpeg xorg-dev python-opengl libboost-all-dev libsdl2-dev swig \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /
RUN apt-get update

RUN /bin/bash -c ". activate habitat; pip install numpy ipython jupyterlab prompt-toolkit"
 
WORKDIR /root

RUN apt-get install -y \
         libqt4-dev \
         qt4-dev-tools \ 
         libglew-dev \ 
         glew-utils \ 
         libgstreamer1.0-dev \ 
         libgstreamer-plugins-base1.0-dev \ 
         libglib2.0-dev

#Fix locale (UTF8) issue https://askubuntu.com/questions/162391/how-do-i-fix-my-locale-issue
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y locales
RUN locale-gen "en_US.UTF-8"

RUN pip install imageio

WORKDIR /
# Conda environment

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        cuda-cudart-11-3 \
        cuda-compat-11-3 \
        cuda-visual-tools-11-3 #\
        cuda-toolkit-11-3 cuda-tools-11-3 cuda-samples-11-3 cuda-documentation-11-3 cuda-libraries-dev-11-3\
        cuda-11-3

RUN apt-get -y upgrade

COPY install_nvidia.sh /app/
RUN chmod +x /app/install_nvidia.sh
RUN NVIDIA_VERSION=$NVIDIA_VERSION /app/install_nvidia.sh

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
RUN apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-11.3
RUN nvcc -V

# libcublas-dev=10.2.1.243-1

RUN pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install numpy==1.20.0

RUN pip install 'git+https://github.com/facebookresearch/fvcore'
RUN python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

WORKDIR /
RUN git clone https://github.com/SHI-Labs/OneFormer.git oneformer

# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN pip install git+https://github.com/cocodataset/panopticapi.git
RUN pip install git+https://github.com/mcordts/cityscapesScripts.git
WORKDIR /oneformer
RUN pip install -r requirements.txt

WORKDIR /oneformer/oneformer/modeling/pixel_decoder/ops
RUN sh make.sh

WORKDIR /

#RUN pip install 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
#RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
#WORKDIR /detectron2_repo
#RUN git reset --hard 9eb4831f742ae6a13b8edb61d07b619392fb6543
WORKDIR /


#RUN pip install -e detectron2_repo


RUN /bin/bash -c 'wget -qO - http://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add -'
#RUN /bin/bash -c 'wget -qO /etc/apt/sources.list.d/lunarg-vulkan-bionic.list http://packages.lunarg.com/vulkan/lunarg-vulkan-bionic.list'

RUN apt install -y libxcb-dri3-0 libxcb-present0 libpciaccess0 \
libpng-dev libxcb-keysyms1-dev libxcb-dri3-dev libx11-dev g++-multilib \
libmirclient-dev libwayland-dev libxrandr-dev libxcb-randr0-dev libxcb-ewmh-dev \
bison libx11-xcb-dev liblz4-dev libzstd-dev libdwarf-dev

RUN /bin/bash -c 'apt update'

RUN git clone https://github.com/KhronosGroup/SPIRV-Tools.git; cd SPIRV-Tools; \
    git clone https://github.com/KhronosGroup/SPIRV-Headers.git external/spirv-headers; \
    git clone https://github.com/google/effcee.git external/effcee; \
    git clone https://github.com/google/re2.git external/re2; \
    git clone https://github.com/google/googletest.git external/googletest; \
    mkdir build && cd build; \
    cmake /SPIRV-Tools;

RUN apt list -a lunarg-vktrace

COPY nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json

RUN wget -qO - http://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add - && \
    wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.2.170-bionic.list http://packages.lunarg.com/vulkan/1.2.170/lunarg-vulkan-1.2.170-bionic.list && \
    apt update && apt install -y vulkan-sdk && apt upgrade -y && apt autoremove -y   

RUN apt-get update
RUN apt-get upgrade     
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9


#RUN /bin/bash -c 'git clone --recursive https://github.com/shacklettbp/bps3D; \
#cd bps3D; \
#mkdir build; \
#cd build; \
#cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..; \
#make' 

#add_definitions(-D GLM_ENABLE_EXPERIMENTAL)

#RUN /bin/bash -c 'git clone --recursive https://github.com/shacklettbp/bps-nav.git; \
#cd bps-nav; \
#cd simulator/python/bps_sim; \
#pip install -e . # Build simulator; \
#cd ../bps_pytorch; \
#pip install -e . # Build pytorch integration; \
#cd ../../../; \
#pip install -e .'

RUN apt-get update


RUN pip install matplotlib
RUN pip install tqdm
RUN pip install tabulate
RUN pip install pandas
RUN pip install scikit-fmm
RUN pip install scikit-image
RUN pip install --no-cache-dir Cython
RUN pip install keyboard
RUN pip install seaborn
RUN pip install ifcfg
RUN pip install imgaug
RUN pip install pycocotools
RUN pip install keras
RUN pip install easydict
RUN pip install pyquaternion
RUN pip install ipywidgets
RUN pip install wandb
RUN pip install lmdb
RUN pip install transformations
RUN pip install scikit-learn
RUN pip install h5py
RUN pip install --upgrade numba
RUN pip install git+https://github.com/openai/CLIP.git


# jupyterlab port
EXPOSE 8888
# tensorboard (if any)
EXPOSE 6006
# startup
COPY image /
#COPY habitat-challenge-data /data_config
ENV HOME /
ENV SHELL /bin/bash

# no password and token for jupyter
ENV JUPYTER_PASSWORD "longpasswordforjupytertostopviruses"
ENV JUPYTER_TOKEN "longpasswordforjupytertostopviruses_jupytertokin"

RUN chmod 777 /startup.sh
RUN chmod 777 /usr/local/bin/jupyter.sh
RUN chmod 777 /usr/local/bin/xvfb.sh

WORKDIR /
# services like lxde, xvfb, x11vnc, jupyterlab will be started

ENTRYPOINT ["/startup.sh"]