#!/usr/bin/env bash
# downloads, builds, and installs ROS2 packages from source in $ROS_WORKSPACE directory
# for example:  ros2_install.sh xacro teleop_twist_joy
set -ex

source $ROS_ROOT/install/setup.bash
export ROS_PACKAGE_PATH=${AMENT_PREFIX_PATH}

: "${ROS_WORKSPACE:=/ros2_workspace}"
: "${ROSDEP_SKIP_KEYS:=gazebo11 libgazebo11-dev libopencv-dev libopencv-contrib-dev libopencv-imgproc-dev python-opencv python3-opencv}"

echo "ROS2 building packages in $ROS_WORKSPACE => $@"

mkdir -p $ROS_WORKSPACE/src
cd $ROS_WORKSPACE

if [[ $1 == http* ]]; then
    cd src
    git clone $1
    cd ../
else
    rosinstall_generator --deps --exclude RPP --rosdistro ${ROS_DISTRO} $@ > ros2.${ROS_DISTRO}.rosinstall
    cat ros2.${ROS_DISTRO}.rosinstall
    vcs import src/ < ros2.${ROS_DISTRO}.rosinstall

    apt-get update

    rosdep install -y \
        --ignore-src \
        --from-paths src \
        --rosdistro ${ROS_DISTRO} \
        --skip-keys "${ROSDEP_SKIP_KEYS}"

    rm -rf /var/lib/apt/lists/*
    apt-get clean
fi

colcon build --symlink-install --base-paths src --event-handlers console_direct+ 

#rm -rf ${ROS_WORKSPACE}/src
#rm -rf ${ROS_WORKSPACE}/logs
#rm -rf ${ROS_WORKSPACE}/build 

#rm ${ROS_WORKSPACE}/*.rosinstall

if grep $ROS_WORKSPACE /bin/ros_entrypoint.sh; then
    echo "workspace $ROS_WORKSPACE was already set to be sourced on startup:"
else
    tac /bin/ros_entrypoint.sh | sed -e "3iros_source_env $ROS_WORKSPACE/install/setup.bash" | tac | tee /bin/ros_entrypoint.sh
    echo "added $ROS_WORKSPACE to be sourced on startup:"
fi

echo ""
cat /bin/ros_entrypoint.sh
