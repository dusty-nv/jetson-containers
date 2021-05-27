#!/usr/bin/env bash

set -e

source scripts/docker_base.sh

SUPPORTED_ROS_DISTROS=("melodic" "noetic" "eloquent" "foxy")
ROS_DISTRO=${1:-"all"}

echo "building containers for $ROS_DISTRO..."

if [[ "$ROS_DISTRO" == "all" ]]; then
	TO_BUILD=${SUPPORTED_ROS_DISTROS[@]}
else
	TO_BUILD=($ROS_DISTRO)
fi

for DISTRO in ${TO_BUILD[@]}; do
	sh ./scripts/docker_build.sh ros:$DISTRO-ros-base-l4t-r$L4T_VERSION Dockerfile.ros.$DISTRO --build-arg BASE_IMAGE=$BASE_IMAGE
done
