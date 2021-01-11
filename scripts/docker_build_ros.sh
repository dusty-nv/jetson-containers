#!/usr/bin/env bash

set -e

BASE_IMAGE="nvcr.io/nvidia/l4t-base"
L4T_VERSION="r32.4.4"
ROS_DISTRO=${1:-"all"}

echo "building containers for $ROS_DISTRO..."

# ROS Melodic
if [[ "$ROS_DISTRO" == "melodic" || "$ROS_DISTRO" == "all" ]]; then
	sh ./scripts/docker_build.sh ros:melodic-ros-base-l4t-$L4T_VERSION Dockerfile.ros.melodic --build-arg BASE_IMAGE=$BASE_IMAGE:$L4T_VERSION
fi

# ROS Noetic
if [[ "$ROS_DISTRO" == "noetic" || "$ROS_DISTRO" == "all" ]]; then
	sh ./scripts/docker_build.sh ros:noetic-ros-base-l4t-$L4T_VERSION Dockerfile.ros.noetic --build-arg BASE_IMAGE=$BASE_IMAGE:$L4T_VERSION
fi

# ROS2 Eloquent
if [[ "$ROS_DISTRO" == "eloquent" || "$ROS_DISTRO" == "all" ]]; then
	sh ./scripts/docker_build.sh ros:eloquent-ros-base-l4t-$L4T_VERSION Dockerfile.ros.eloquent --build-arg BASE_IMAGE=$BASE_IMAGE:$L4T_VERSION
fi

# ROS2 Foxy
if [[ "$ROS_DISTRO" == "foxy" || "$ROS_DISTRO" == "all" ]]; then
	sh ./scripts/docker_build.sh ros:foxy-ros-base-l4t-$L4T_VERSION Dockerfile.ros.foxy --build-arg BASE_IMAGE=$BASE_IMAGE:$L4T_VERSION
fi
