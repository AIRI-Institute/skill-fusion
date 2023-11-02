#!/usr/bin/env bash

DOCKER_NAME="skilltron_docker"

while [[ $# -gt 0 ]]
do
key="${1}"

case $key in
      --docker-name)
      shift
      DOCKER_NAME="${1}"
	  shift
      ;;
    *) # unknown arg
      echo unkown arg ${1}
      exit
      ;;
esac
done

docker run \
    -v $(pwd)/habitat-challenge-data:/data \
    --runtime=nvidia \
    -e "AGENT_EVALUATION_TYPE=local" \
    -e "TRACK_CONFIG_FILE=/configs/benchmark/nav/objectnav/objectnav_v2_hm3d_stretch_challenge.yaml" \
    ${DOCKER_NAME}\
