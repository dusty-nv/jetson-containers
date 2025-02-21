#!/usr/bin/env bash
set -e

ros_packages=$(ros2 pkg list)

echo "ROS2 packages installed:"
echo "$ros_packages"

if echo "$ros_packages" | grep ros2_nanollm; then
    echo "ros2_nanollm found"
else
    echo "ros2_nanollm not found"
    exit 1
fi   

ros_nodes=$(ros2 pkg executables ros2_nanollm)

echo "ros2_nanollm nodes found:"
echo "$ros_nodes"

if echo "$ros_nodes" | grep nano_llm_py; then
    echo "ros2_nanollm nodes found"
else
    echo "ros2_nanollm nodes not found"
    exit 1
fi   
