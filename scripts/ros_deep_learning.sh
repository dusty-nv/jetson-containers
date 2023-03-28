#!/usr/bin/env bash
# this script installs the ros_deep_learning nodes for jetson-inference

set -e
#set -x

if [ ! -d "/jetson-inference" ]; then
	echo "jetson-inference not found, skipping installation of ros_deep_learning package"
	exit 0
fi

echo "ROS_DISTRO = $ROS_DISTRO"

WORKSPACE="/ros_deep_learning"

mkdir -p $WORKSPACE/src
source /ros_entrypoint.sh

if [[ "$ROS_DISTRO" == "melodic" || "$ROS_DISTRO" == "noetic" ]]; then
	echo "building ros_deep_learning for ROS1 $ROS_DISTRO"
	cd $WORKSPACE
	catkin_make
	source devel/setup.bash
	cd $WORKSPACE/src
	git clone --depth=1 https://github.com/dusty-nv/ros_deep_learning
	cd $WORKSPACE
	catkin_make
	source devel/setup.bash
	echo "testing that ROS $ROS_DISTO can find ros_deep_learning package:"
	rospack find ros_deep_learning
else
	cd $WORKSPACE/src
	git clone --depth=1 https://github.com/dusty-nv/ros_deep_learning
	echo "Building ros_deep_learning package for ROS2 $ROS_DISTRO"
	cd $WORKSPACE
	colcon build
	source /ros_entrypoint.sh
	echo "testing that ROS2 $ROS_DISTO can find ros_deep_learning package:"
	ros2 pkg prefix ros_deep_learning
fi