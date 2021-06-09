#!/usr/bin/env bash

set -e
source scripts/docker_base.sh

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
TEST_MOUNT="$ROOT/../test:/test"
SUPPORTED_ROS_DISTROS=("melodic" "noetic" "eloquent" "foxy" "galactic")
ROS_DISTRO=${1:-"all"}

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

if [[ "$ROS_DISTRO" == "all" ]]; then
	TO_TEST=${SUPPORTED_ROS_DISTROS[@]}
else
	TO_TEST=($ROS_DISTRO)
fi

for DISTRO in ${TO_TEST[@]}; do
	test_ros_version "ros:$DISTRO-ros-base-l4t-r$L4T_VERSION"
done
