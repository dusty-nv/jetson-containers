#!/usr/bin/env bash

set -e
source scripts/docker_base.sh

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
TEST_MOUNT="$ROOT/../test:/test"

if [[ $L4T_RELEASE -eq 34 || $L4T_RELEASE -eq 35 ]]; then   # JetPack 5.x / Ubuntu 20.04
	SUPPORTED_ROS_DISTROS=("noetic" "foxy" "galactic" "humble")
else
	SUPPORTED_ROS_DISTROS=("melodic" "noetic" "eloquent" "foxy" "galactic" "humble")
fi

ROS_DISTRO=${1:-"all"}
ROS_PYTORCH=${2:-"yes"}

test_ros_version()
{
	echo "testing container $1 => ros_version"
	local DISTRO_TO_TEST=$(echo "$1" | cut -d ":" -f 2 | cut -d "-" -f 1)
	if [[ "$DISTRO_TO_TEST" == "melodic" || "$DISTRO_TO_TEST" == "noetic" ]]; then
		sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r test/test_ros_version.sh
	else
		sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r test/test_ros2_version.sh
	fi
	echo -e "done testing container $1 => ros_version\n"
}

test_opencv()
{
	echo "testing container $1 => OpenCV"
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_opencv.py
	echo -e "done testing container $1 => OpenCV\n"
}

test_pytorch()
{
	echo "testing container $1 => PyTorch"
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_pytorch.py
	echo -e "done testing container $1 => PyTorch\n"
}

test_numpy()
{
	echo "testing container $1 => numpy"
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_numpy.py
	echo -e "done testing container $1 => numpy\n"
}

if [[ "$ROS_DISTRO" == "all" ]]; then
	TO_TEST=${SUPPORTED_ROS_DISTROS[@]}
else
	TO_TEST=($ROS_DISTRO)
fi

for DISTRO in ${TO_TEST[@]}; do
	container_tag="ros:$DISTRO-ros-base-l4t-r$L4T_VERSION"
	test_ros_version $container_tag
	
	if [[ "$DISTRO" != "melodic" ]] && [[ "$DISTRO" != "noetic" ]] && [[ "$DISTRO" != "eloquent" ]]; then
		test_opencv $container_tag
		test_numpy $container_tag
	fi
	
	if [[ "$ROS_PYTORCH" == "yes" ]] && [[ "$DISTRO" != "melodic" ]]; then
		container_tag="ros:$DISTRO-pytorch-l4t-r$L4T_VERSION"
		test_ros_version $container_tag
		test_pytorch $container_tag
		test_numpy $container_tag
		
		if [[ "$DISTRO" != "eloquent" ]]; then
			test_opencv $container_tag
		fi
	fi
done
