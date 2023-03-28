#!/bin/bash
#set -e

function ros_source_env() 
{
	if [ -f "$1" ]; then
		echo "sourcing   $1"
		source "$1"
	else
		echo "notfound   $1"
	fi	
}

if [[ "$ROS_DISTRO" == "melodic" || "$ROS_DISTRO" == "noetic" ]]; then
	ros_source_env "/opt/ros/$ROS_DISTRO/setup.bash"
	ros_source_env "/ros_deep_learning/devel/setup.bash"
else
	ros_source_env "$ROS_ROOT/install/setup.bash"
	ros_source_env "/ros_deep_learning/install/setup.bash"

	if [ -d "/ros_deep_learning/install/ros_deep_learning" ]; then
		export AMENT_PREFIX_PATH="/ros_deep_learning/install/ros_deep_learning:$AMENT_PREFIX_PATH"
	fi

	#echo "ROS_PACKAGE_PATH   $ROS_PACKAGE_PATH"
	#echo "COLCON_PREFIX_PATH $COLCON_PREFIX_PATH"
	#echo "AMENT_PREFIX_PATH  $AMENT_PREFIX_PATH"
	#echo "CMAKE_PREFIX_PATH  $CMAKE_PREFIX_PATH"
fi

echo "ROS_DISTRO $ROS_DISTRO"
echo "ROS_ROOT   $ROS_ROOT"

exec "$@"