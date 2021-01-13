#!/usr/bin/env bash

set -e

BASE_IMAGE="nvcr.io/nvidia/l4t-base"
L4T_VERSION="r32.4.4"
SUPPORTED_ROS_DISTROS=("melodic" "noetic" "eloquent" "foxy")
ROS_DISTRO=${1:-"all"}

echo "building containers for $ROS_DISTRO..."

if [[ "$ROS_DISTRO" == "all" ]]; then
	TO_BUILD=${SUPPORTED_ROS_DISTROS[@]}
else
	TO_BUILD=($ROS_DISTRO)
fi

for DISTRO in ${TO_BUILD[@]}; do
	sh ./scripts/docker_build.sh ros:$DISTRO-ros-base-l4t-$L4T_VERSION Dockerfile.ros.$DISTRO --build-arg BASE_IMAGE=$BASE_IMAGE:$L4T_VERSION
done
