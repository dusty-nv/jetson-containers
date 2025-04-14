# isaac_ros_common
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG ROS_PACKAGE=https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common \
    ROS_WORKSPACE=/opt/isaac-ros \
    ROS_BRANCH=main

COPY rosdeps.yml /etc/ros/rosdep/sources.list.d/extra-rosdeps.yml

RUN echo "yaml file:///etc/ros/rosdep/sources.list.d/extra-rosdeps.yml" | \
    tee /etc/ros/rosdep/sources.list.d/00-extras.list && \
    ROS_BRANCH=master /ros2_install.sh "https://github.com/osrf/negotiated" && \
    ROS_BRANCH=v0.9.3 /ros2_install.sh "https://github.com/Neargye/magic_enum"

RUN /ros2_install.sh "${ROS_PACKAGE}"