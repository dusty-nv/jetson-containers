#!/usr/bin/env bash
# downloads, builds, and installs ROS2 packages from source in $ROS_WORKSPACE directory
# for example:  ros2_install.sh xacro teleop_twist_joy
set -ex

source $ROS_ROOT/install/setup.bash
export ROS_PACKAGE_PATH=${AMENT_PREFIX_PATH}

: "${ROS_WORKSPACE:=${ROS_ROOT}}"
: "${ROSDEP_SKIP_KEYS:=gazebo11 libgazebo11-dev libopencv-dev libopencv-contrib-dev libopencv-imgproc-dev python-opencv python3-opencv}"

echo "ROS2 building packages in $ROS_WORKSPACE => $@"

mkdir -p $ROS_WORKSPACE/src
cd $ROS_WORKSPACE

if [[ $1 == http* ]]; then
    SOURCE="git_clone"
    cd src
    git clone $1
    cd ../
else
    SOURCE="rosinstall_generator"
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

if [ "${ROS_WORKSPACE}" = "${ROS_ROOT}" ]; then
    COLCON_FLAGS="--merge-install"
else
    COLCON_FLAGS="--symlink-install"
fi

colcon build ${COLCON_FLAGS} --base-paths src --event-handlers console_direct+ 

#rm -rf ${ROS_WORKSPACE}/src
#rm -rf ${ROS_WORKSPACE}/logs
#rm -rf ${ROS_WORKSPACE}/build 

#rm ${ROS_WORKSPACE}/*.rosinstall

if grep $ROS_WORKSPACE /ros_entrypoint.sh; then
    echo "workspace $ROS_WORKSPACE was already set to be sourced on startup:"
else
    if [ -f /ros_entrypoint.sh ] && grep -q 'function ros_source_env' /ros_entrypoint.sh; then
    	if ! grep -q "$ROS_WORKSPACE/install/setup.bash" /ros_entrypoint.sh; then
        	echo "Adding $ROS_WORKSPACE to ros_entrypoint.sh"
        	tac /ros_entrypoint.sh | sed -e "3iros_source_env $ROS_WORKSPACE/install/setup.bash" | tac > /tmp/ros_entrypoint.sh \
            		&& mv /tmp/ros_entrypoint.sh /ros_entrypoint.sh
    	fi
    	chmod +x /ros_entrypoint.sh
    else
    	echo "WARNING: /ros_entrypoint.sh is missing or invalid. Skipping insertion."
    fi  
    echo "added $ROS_WORKSPACE to be sourced on startup:"
fi

echo ""
cat /ros_entrypoint.sh

if [ "$SOURCE" = "rosinstall_generator" ]; then
    source $ROS_WORKSPACE/install/setup.bash
    ros_packages=$(ros2 pkg list)
    echo "ROS2 packages installed:"
    echo "$ros_packages"
    if echo "$ros_packages" | grep "$@"; then
        echo "$@ found"
    else
        echo "$@ not found"
        exit 1
    fi   
fi
