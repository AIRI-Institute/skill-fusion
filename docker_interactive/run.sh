docker run --runtime=nvidia --rm --name bps_docker \
--env="DISPLAY=$DISPLAY" \
--env="QT_X11_NO_MITSHM=1" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
-env="XAUTHORITY=$XAUTH" \
--volume="$XAUTH:$XAUTH" \
--privileged \
-p $1:8888 -e jup_port=$1  \
-v <path_to_Habitat_sim_data>/data:/data \
-v <path_to_SkillFusion>/root/:/root alstar_docker