#!/bin/bash
set -e

echo "sourcing /opt/ros/$ROS_DISTRO/setup.bash"

# setup ROS environment
source "/opt/ros/$ROS_DISTRO/setup.bash"

echo "ROS_DISTRO $ROS_DISTRO"
echo "ROS_ROOT   $ROS_ROOT"

exec "$@"
