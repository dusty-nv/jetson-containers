#!/usr/bin/env bash
#
# Builds ROS container(s) by installing packages or from source (when needed)
#
# Arguments:
#
#    scripts/docker_build_ros.sh <DISTRO> <PACKAGE> <PYTORCH>
#
#    DISTRO - 'melodic', 'noetic', 'eloquent', 'foxy', 'galactic', or 'all' (default is 'all')
#    PACKAGE - 'ros_base', 'ros_core', 'desktop' (default is 'ros_base' - 'desktop' may have issues on some distros)
#    PYTORCH - 'yes' to build the version of the containers with PyTorch support, otherwise 'no' (default is 'yes')
#
set -e

source scripts/docker_base.sh

SUPPORTED_ROS_DISTROS=("melodic" "noetic" "eloquent" "foxy" "galactic")
SUPPORTED_ROS_PACKAGES=("ros_base" "ros_core" "desktop")

ROS_DISTRO=${1:-"all"}
ROS_PACKAGE=${2:-"ros_base"}
ROS_PYTORCH=${3:-"yes"}

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
	local base_image=$3
	local extra_tag=$4
	local package_name=`echo $package | tr '_' '-'`
	local container_tag="ros:${distro}-${package_name}-${extra_tag}l4t-r${L4T_VERSION}"
	
	# opencv.csv mounts files that preclude us installing different version of opencv
	# temporarily disable the opencv.csv mounts while we build the container
	CV_CSV="/etc/nvidia-container-runtime/host-files-for-container.d/opencv.csv"

	if [ -f "$CV_CSV" ]; then
		sudo mv $CV_CSV $CV_CSV.backup
	fi
	
	echo ""
	echo "Building container $container_tag"
	echo "BASE_IMAGE=$base_image"
	echo ""
	
	sh ./scripts/docker_build.sh $container_tag Dockerfile.ros.$distro \
			--build-arg ROS_PKG=$package \
			--build-arg BASE_IMAGE=$base_image
			
	# restore opencv.csv mounts
	if [ -f "$CV_CSV.backup" ]; then
		sudo mv $CV_CSV.backup $CV_CSV
	fi
}

for DISTRO in ${BUILD_DISTRO[@]}; do
	for PACKAGE in ${BUILD_PACKAGES[@]}; do
		build_ros $DISTRO $PACKAGE $BASE_IMAGE
		
		if [[ "$ROS_PYTORCH" == "yes" ]] && [[ "$DISTRO" != "melodic" ]] ; then
			build_ros $DISTRO $PACKAGE "dustynv/jetson-inference:r$L4T_VERSION" "pytorch-"
		fi
	done
done
