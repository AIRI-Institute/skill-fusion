xhost +local:docker || true
docker run --runtime=nvidia --rm --name bps_docker \
--env="DISPLAY=$DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
-env="XAUTHORITY=$XAUTH" \
--volume="$XAUTH:$XAUTH" \
--privileged \
-p $2:8888 -e jup_port=$2 \
-v /home/kirill/habitat-lab/data/:/data \
-v ${HOME}/.Xauthority:/root/.Xauthority:rw \
-v /home/kirill/habitat-lab/docker/vulkan_habitat_cds1/root/:/root alstar_docker
