# isaac_ros_common
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG ROS_PACKAGE=https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common \
    ROS_WORKSPACE=/opt/isaac-ros \
    ROS_BRANCH=main \
    TMP_DIR=/tmp/isaac-ros

COPY install.sh ${TMP_DIR}/
RUN ${TMP_DIR}/install.sh