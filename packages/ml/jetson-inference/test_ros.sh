#!/usr/bin/env bash
set -e

ros_packages=$(ros2 pkg list)

echo "ROS2 packages installed:"
echo "$ros_packages"

if echo "$ros_packages" | grep ros_deep_learning; then
    echo "ros_deep_learning found"
else
    echo "ros_deep_learning not found"
    exit 1
fi   

ros_nodes=$(ros2 pkg executables ros_deep_learning)

echo "ros_deep_learning nodes found:"
echo "$ros_nodes"

if echo "$ros_nodes" | grep detectnet; then
    echo "ros_deep_learning nodes found"
else
    echo "ros_deep_learning nodes not found"
    exit 1
fi   
