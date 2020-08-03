#!/usr/bin/env bash

set -e

BASE_IMAGE="nvcr.io/nvidia/l4t-base:r32.4.3"

# ROS2 Eloquent
sh ./scripts/docker_build.sh ros:eloquent-ros-base-l4t-r32.4.3 Dockerfile.ros.eloquent --build-arg BASE_IMAGE=$BASE_IMAGE

# ROS2 Foxy
sh ./scripts/docker_build.sh ros:foxy-ros-base-l4t-r32.4.3 Dockerfile.ros.foxy --build-arg BASE_IMAGE=$BASE_IMAGE

