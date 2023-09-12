docker run --runtime=nvidia --rm --name bps_docker1 \
--env="DISPLAY=$DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
-env="XAUTHORITY=$XAUTH" \
--volume="$XAUTH:$XAUTH" \
--privileged \
-p $2:8888 -e jup_port=$2  \
-v /data_hdd1/alex_star/habitat_data/data/:/data \
-v /home/minerl/Desktop/vulkan_rt_docker/root1/:/root alstar_docker

#-p $1:5900 -e vnc_port=$1