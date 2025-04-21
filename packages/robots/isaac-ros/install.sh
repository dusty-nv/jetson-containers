#!/usr/bin/env bash
# isaac_ros_common compatability setup
source /ros_environment.sh
set -ex

# assure environment setup
export ROS_PACKAGE_PATH=${AMENT_PREFIX_PATH}
mkdir -p $ROS_WORKSPACE/src || true;

# install foxglove_msgs
git clone https://github.com/foxglove/foxglove-sdk /tmp/foxglove-sdk
cp -r /tmp/foxglove-sdk/ros/foxglove_msgs $ROS_WORKSPACE/src
rm -rf /tmp/foxglove-sdk

# distro-specific workarounds
if [ $ROS_DISTRO == "jazzy" ]; then
  #git clone --branch=$ROS_DISTRO https://github.com/ament/ament_index $ROS_WORKSPACE/src/ament_index
  ament_index_root="$ROS_ROOT/install/include/ament_index_cpp"
  cp $ament_index_root/ament_index_cpp/* $ament_index_root/
fi

# build various sources
cd $ROS_WORKSPACE
colcon build --symlink-install --base-paths src --event-handlers console_direct+

# build other source dependencies
ROS_BRANCH=master /ros2_install.sh "https://github.com/osrf/negotiated"
ROS_BRANCH=v0.9.3 /ros2_install.sh "https://github.com/Neargye/magic_enum"

# install isaac_ros_common
/ros2_install.sh "${ROS_PACKAGE}"
