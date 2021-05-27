#!/bin/bash
set -e

echo "sourcing /opt/ros/$ROS_DISTRO/setup.bash"

# setup ROS environment
source "/opt/ros/$ROS_DISTRO/setup.bash"

echo "ROS_DISTRO       $ROS_DISTRO"
echo "ROS_ROOT         $ROS_ROOT"
echo "ROS_PACKAGE_PATH $ROS_PACKAGE_PATH"
echo "ROS_MASTER_URI   $ROS_MASTER_URI"

exec "$@"
