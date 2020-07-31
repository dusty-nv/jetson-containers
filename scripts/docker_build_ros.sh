#!/usr/bin/env bash

set -e

BASE_IMAGE="nvcr.io/nvidian/nvidia-l4t-base:r32.4.3"

sh ./scripts/docker_build.sh ros:foxy-ros-base-l4t-r32.4.3 Dockerfile.ros.foxy \
		--build-arg BASE_IMAGE=$BASE_IMAGE

