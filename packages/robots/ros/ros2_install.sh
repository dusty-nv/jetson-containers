#!/usr/bin/env bash
# downloads, builds, and installs ROS2 packages from source in $ROS_WORKSPACE directory
# for example:  ros2_install.sh xacro teleop_twist_joy
source /ros_environment.sh

export ROS_PACKAGE_PATH=${AMENT_PREFIX_PATH}
export MAKEFLAGS="-j $(nproc)"

ROS_WORKSPACE="${ROS_WORKSPACE:=${ROS_ROOT}}"
ROSDEP_SKIP_KEYS="$ROSDEP_SKIP_KEYS gazebo11 libgazebo11-dev libopencv-dev libopencv-contrib-dev libopencv-imgproc-dev python-opencv python3-opencv"
ROS_INSTALL_FLAGS="--deps --exclude RPP --rosdistro ${ROS_DISTRO} $ROS_INSTALL_FLAGS"
COLCON_FLAGS="--base-paths src --event-handlers console_direct+ $COLCON_FLAGS"

if [ "${ROS_WORKSPACE}" = "${ROS_ROOT}" ]; then
    COLCON_FLAGS="--merge-install $COLCON_FLAGS"
else
    COLCON_FLAGS="--symlink-install $COLCON_FLAGS"
fi

echo "ROS2 building packages in $ROS_WORKSPACE => $@"
mkdir -p $ROS_WORKSPACE/src
set -ex

if [[ $1 == http* ]]; then
    # direct clone of ROS repo/package from git
    SOURCE="git_clone"

    if [ ! -z "${ROS_BRANCH}" ]; then
        ROS_BRANCH="--branch ${ROS_BRANCH}"
    fi

    cd $ROS_WORKSPACE/src
    git clone --recursive --depth=1 $ROS_BRANCH $1
    
    # INSTALL_PREFIX is a marker for install(EXPORT) only
    if [[ $1 == *isaac_ros* ]]; then
        cd $(basename $1)
        find . -type f -name "CMakeLists.txt" -print0 | \
        xargs -0 sed -i'' -e 's|<INSTALL_PREFIX>|{CMAKE_INSTALL_PREFIX}|g'
    fi

    cd $ROS_WORKSPACE

    COLCON_FLAGS="$COLCON_FLAGS" #--packages-up-to $(basename $1)"
    rosinstall_list="$(basename $1).rosinstall"

    rosinstall_generator ${ROS_INSTALL_FLAGS} --from-path src > $rosinstall_list || \
    rosinstall_generator ${ROS_INSTALL_FLAGS} --from-path src --upstream > $rosinstall_list || true
else
    # pull sources from ROS by their package names
    SOURCE="rosinstall_generator"
    cd $ROS_WORKSPACE

    rosinstall_list="ros2.${ROS_DISTRO}.rosinstall"
    rosinstall_generator ${ROS_INSTALL_FLAGS} $@ > $rosinstall_list
fi

if [ -s $rosinstall_list ]; then
    cat $rosinstall_list
    vcs import --skip-existing src/ < $rosinstall_list
fi

cd $ROS_WORKSPACE
apt-get update

rosdep init || true;
rosdep update --rosdistro ${ROS_DISTRO}
rosdep install -y \
    --ignore-src \
    --from-paths src \
    --rosdistro ${ROS_DISTRO} \
    --skip-keys "${ROSDEP_SKIP_KEYS}"

rm -rf /var/lib/apt/lists/*
apt-get clean

colcon build ${COLCON_FLAGS} --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF 

#rm -rf ${ROS_WORKSPACE}/src
#rm -rf ${ROS_WORKSPACE}/logs
#rm -rf ${ROS_WORKSPACE}/build 

#rm ${ROS_WORKSPACE}/*.rosinstall
set +x

if grep $ROS_WORKSPACE /ros_environment.sh; then
    echo "workspace $ROS_WORKSPACE was already set to be sourced on startup:"
else
    if [ -f /ros_environment.sh ] && grep -q 'function ros_source_env' /ros_environment.sh; then
    	if ! grep -q "$ROS_WORKSPACE/install/setup.bash" /ros_environment.sh; then
        	echo "Adding $ROS_WORKSPACE to ros_environment.sh"
        	tac /ros_environment.sh | sed -e "5iros_source_env $ROS_WORKSPACE/install/setup.bash" | tac > /tmp/ros_environment.sh \
            		&& mv /tmp/ros_environment.sh /ros_environment.sh
    	fi
    	chmod +x /ros_environment.sh
    else
    	echo "WARNING: /ros_environment.sh is missing or invalid. Skipping insertion."
    fi  
    echo "added $ROS_WORKSPACE to be sourced on startup:"
fi

echo ""
cat /ros_environment.sh

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
