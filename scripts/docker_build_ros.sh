#!/usr/bin/env bash

set -e

source scripts/docker_base.sh

SUPPORTED_ROS_DISTROS=("melodic" "noetic" "eloquent" "foxy")
SUPPORTED_ROS_PACKAGES=("ros_base" "ros_core" "desktop")

ROS_DISTRO=${1:-"all"}
ROS_PACKAGE=${2:-"base"}

echo "Building containers for $ROS_DISTRO..."

if [[ "$ROS_DISTRO" == "all" ]]; then
	BUILD_DISTRO=${SUPPORTED_ROS_DISTROS[@]}
else
	BUILD_DISTRO=($ROS_DISTRO)
fi

if [[ "$ROS_PACKAGE" == "all" ]]; then
	BUILD_PACKAGES=${SUPPORTED_ROS_PACKAGES[@]}
else
	BUILD_PACKAGES=($ROS_PACKAGE)
	
	if [[ ! " ${SUPPORTED_ROS_PACKAGES[@]} " =~ " ${ROS_PACKAGE} " ]]; then
		echo "error -- '$ROS_PACKAGE' isn't one of the supported ROS packages:"
		echo "              ${SUPPORTED_ROS_PACKAGES[@]}"
		exit 1
     fi
fi

build_ros()
{
	local distro=$1
	local package=$2
	local package_name=`echo $package | tr '_' '-'`
	local container_tag="ros:$distro-$package_name-l4t-r$L4T_VERSION"
	
	sh ./scripts/docker_build.sh $container_tag Dockerfile.ros.$distro \
			--build-arg ROS_PKG=$package \
			--build-arg BASE_IMAGE=$BASE_IMAGE
}

for DISTRO in ${BUILD_DISTRO[@]}; do
	for PACKAGE in ${BUILD_PACKAGES[@]}; do
		build_ros $DISTRO $PACKAGE
	done
done
