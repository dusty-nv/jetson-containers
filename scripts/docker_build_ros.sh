#!/usr/bin/env bash
#
# Builds ROS container(s) by installing packages or from source (when needed)
# See help text below or run './scripts/docker_build_ros.sh --help' for options
#

set -e

show_help() {
    echo " "
    echo "usage: Builds various ROS Docker containers for Jetson / JetPack-L4T"
    echo " "
    echo "   ./scripts/docker_build_ros.sh --distro DISTRO"
    echo "                                 --package PACKAGE"
    echo "                                 --with-pytorch"
    echo " "
    echo "args:"
    echo " "
    echo "   --help                       Show this help text and quit"
    echo " "
    echo "   --distro DISTRO Specifies the ROS distro to build, one of:"
    echo "                   'melodic', 'noetic', 'eloquent', 'foxy', 'galactic'"
    echo "                   Or the default of 'all' will build all distros."
    echo " "
    echo "   --package PKG   Specifies the ROS meta-package to install, one of:"
    echo "                   'ros_base', 'ros_core', 'desktop', 'all'"
    echo "                   The default is 'ros_base'.  Note that 'desktop' may"
    echo "                   have issues on some distros that are built from source."
    echo " "
    echo "   --with-pytorch  Builds additional container with PyTorch support."
    echo "                   This only applies to noetic, foxy, and galactic."
    echo " "
}

die() {
    printf '%s\n' "$1"
    show_help
    exit 1
}

# determine the L4T version
source scripts/docker_base.sh
source scripts/opencv_version.sh

# define default options
if [[ $L4T_RELEASE -eq 34 || $L4T_RELEASE -eq 35 ]]; then   # JetPack 5.x / Ubuntu 20.04
	SUPPORTED_ROS_DISTROS=("noetic" "foxy" "galactic" "humble")
else
	SUPPORTED_ROS_DISTROS=("melodic" "noetic" "eloquent" "foxy" "galactic" "humble")
fi

SUPPORTED_ROS_PACKAGES=("ros_base" "ros_core" "desktop")

ROS_DISTRO="all"
ROS_PACKAGE="ros_base"
WITH_PYTORCH="off"

# parse options
while :; do
    case $1 in
        -h|-\?|--help)
            show_help
            exit
            ;;
        --distro)    # Takes an option argument; ensure it has been specified.
            if [ "$2" ]; then
                ROS_DISTRO=$2
                shift
            else
                die 'ERROR: "--distro" requires a non-empty option argument.'
            fi
            ;;
        --distro=?*)
            ROS_DISTRO=${1#*=} # Delete everything up to "=" and assign the remainder.
            ;;
        --distror=)         # Handle the case of an empty --distro=
            die 'ERROR: "--distro" requires a non-empty option argument.'
            ;;
	   --package)
            if [ "$2" ]; then
                ROS_PACKAGE=$2
                shift
            else
                die 'ERROR: "--package" requires a non-empty option argument.'
            fi
            ;;
        --package=?*)
            ROS_PACKAGE=${1#*=}
            ;;
        --package=)         # Handle the case of an empty --distro=
            die 'ERROR: "--package" requires a non-empty option argument.'
            ;;
	   --with-pytorch)
            WITH_PYTORCH="on"
            ;;
        --)              # End of all options.
            shift
            break
            ;;
        -?*)
            printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
            ;;
        *)               # Default case: No more options, so break out of the loop.
            break
    esac

    shift
done

echo "ROS_DISTRO:   $ROS_DISTRO"
echo "ROS_PACKAGE:  $ROS_PACKAGE"
echo "WITH_PYTORCH: $WITH_PYTORCH"

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


# check for local version of PyTorch base container
BASE_IMAGE_PYTORCH="jetson-inference:r$L4T_VERSION"

if [[ "$(sudo docker images -q $BASE_IMAGE_PYTORCH 2> /dev/null)" == "" ]]; then
	BASE_IMAGE_PYTORCH="dustynv/$BASE_IMAGE_PYTORCH"
fi


build_ros()
{
	local distro=$1
	local package=$2
	local base_image=$3
	local extra_tag=$4
	local dockerfile=${5:-"Dockerfile.ros.$distro"}
	local container_tag="ros:${distro}-${extra_tag}l4t-r${L4T_VERSION}"
	
	echo ""
	echo "Building container $container_tag"
	echo "BASE_IMAGE=$base_image"
	echo ""
	
	sh ./scripts/docker_build.sh $container_tag $dockerfile \
			--build-arg ROS_PKG=$package \
			--build-arg BASE_IMAGE=$base_image \
			--build-arg OPENCV_URL=$OPENCV_URL \
			--build-arg OPENCV_DEB=$OPENCV_DEB
			
	# restore opencv.csv mounts
	if [ -f "$CV_CSV.backup" ]; then
		sudo mv $CV_CSV.backup $CV_CSV
	fi
}


for DISTRO in ${BUILD_DISTRO[@]}; do
	for PACKAGE in ${BUILD_PACKAGES[@]}; do
		build_ros $DISTRO $PACKAGE $BASE_IMAGE "`echo $PACKAGE | tr '_' '-'`-"
		
		if [[ "$WITH_PYTORCH" == "on" && "$DISTRO" != "melodic" && "$DISTRO" != "eloquent" ]]; then
			build_ros $DISTRO $PACKAGE $BASE_IMAGE_PYTORCH "pytorch-"
		fi
	done
done
