#!/usr/bin/env bash
# This script builds a ROS2 distribution from source, or installs
# a cached build from jetson-ai-lab (unless FORCE_BUILD=on)
#
# ROS_DISTRO, ROS_ROOT, ROS_PACKAGE environment variables are set
# by the dockerfile or config.py (ex. ROS_ROOT=/opt/ros/humble)
set -e

FORCE_BUILD="${FORCE_BUILD:=off}"
BUILD_CACHE="ros-$ROS_DISTRO-$ROS_PACKAGE"

SEPARATOR="********************************************************"

function print_log() {
	printf "\n$SEPARATOR\n$1\n$SEPARATOR\n\n"
}

print_log " ROS2 $ROS_DISTRO installer ($(uname -m))

   ROS_DISTRO=$ROS_DISTRO
   ROS_PACKAGE=$ROS_PACKAGE
   ROS_ROOT=$ROS_ROOT
   FORCE_BUILD=$FORCE_BUILD
   BUILD_CACHE=$TAR_INDEX_URL/$BUILD_CACHE.tar.gz"

# add the ROS deb repo to the apt sources list
apt-get update
apt-get install -y --no-install-recommends \
		curl \
		wget \
		gnupg2 \
		lsb-release \
		ca-certificates

curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
apt-get update

# install development packages
apt-get install -y --no-install-recommends \
		libeigen3-dev \
		libbullet-dev \
		libpython3-dev \
		python3-colcon-common-extensions \
		python3-flake8 \
		python3-pip \
		python3-pytest-cov \
		python3-rosdep \
		python3-setuptools \
		python3-vcstool \
		python3-rosinstall-generator \
		libasio-dev \
		libtinyxml2-dev \
		libcunit1-dev \
		libacl1-dev \
		libssl-dev \
		libxaw7-dev \
		libfreetype-dev

# install some pip packages needed for testing
pip3 install --upgrade \
		argcomplete \
		flake8-blind-except \
		flake8-builtins \
		flake8-class-newline \
		flake8-comprehensions \
		flake8-deprecated \
		flake8-docstrings \
		flake8-import-order \
		flake8-quotes \
		pytest-repeat \
		pytest-rerunfailures \
		pytest \
		lark \
		scikit-build

# upgrade pybind11
apt-get purge -y pybind11-dev
pip3 install --upgrade pybind11-global

# restore cmake and numpy versions
bash /tmp/cmake/install.sh
bash /tmp/numpy/install.sh

# remove other versions of Python3
# workaround for 'Could NOT find Python3 (missing: Python3_NumPy_INCLUDE_DIRS Development'
apt purge -y python3.9 libpython3.9* || echo "python3.9 not found, skipping removal"
ls -ll /usr/bin/python*
    
# create the ROS_ROOT directory
mkdir -p ${ROS_ROOT}/src
cd ${ROS_ROOT}

# install additional rosdep entries
ROSDEP_DIR="/etc/ros/rosdep/sources.list.d"
mkdir -p $ROSDEP_DIR || true;

cp $TMP/rosdeps.yml $ROSDEP_DIR/extra-rosdeps.yml
echo "yaml file://$ROSDEP_DIR/extra-rosdeps.yml" | \
tee $ROSDEP_DIR/00-extras.list

# skip installation of some conflicting packages (these now get handled in rosdeps.yml)
#SKIP_KEYS="libopencv-dev libopencv-contrib-dev libopencv-imgproc-dev python-opencv python3-opencv"
SKIP_KEYS=""

# patches for building Humble on 18.04
if [ "$ROS_DISTRO" = "humble" ] || [ "$ROS_DISTRO" = "iron" ] && [ $(lsb_release --codename --short) = "bionic" ]; then
	# rti_connext_dds_cmake_module: No definition of [rti-connext-dds-6.0.1] for OS version [bionic]
	SKIP_KEYS="$SKIP_KEYS rti-connext-dds-6.0.1 ignition-cmake2 ignition-math6"

	# the default gcc-7 is too old to build humble
	apt-get install -y --no-install-recommends gcc-8 g++-8
	export CC="/usr/bin/gcc-8"
	export CXX="/usr/bin/g++-8"
	echo "CC=$CC CXX=$CXX"

	# upgrade pybind11
	apt-get purge -y pybind11-dev
	pip3 install --upgrade pybind11-global
   
	# https://github.com/dusty-nv/jetson-containers/issues/160#issuecomment-1429572145
	git -C /tmp clone -b yaml-cpp-0.6.0 https://github.com/jbeder/yaml-cpp.git
	cmake -S /tmp/yaml-cpp -B /tmp/yaml-cpp/BUILD -DBUILD_SHARED_LIBS=ON
	cmake --build /tmp/yaml-cpp/BUILD --parallel $(nproc --ignore=1)
	cmake --install /tmp/yaml-cpp/BUILD
	rm -rf /tmp/yaml-cpp
fi

# attempt to install from network build cache
function cached_install() {
	TARPACK_PREFIX=$ROS_ROOT tarpack install $BUILD_CACHE
	local result=$?
	if [ "$result" != 0 ]; then
		return $result
	fi
	rosdep_install
	result=$?
	if [ "$result" != 0 ]; then
		return $result
	fi
	CACHED_INSTALL="yes"
}

function rosdep_install() {
	cat $ROS_ROOT/rosdeps.txt | xargs apt-get install -y --no-install-suggests --no-install-recommends
}

if [ "$FORCE_BUILD" != "on" ]; then
	cached_install || true;

	if [ "$CACHED_INSTALL" == "yes" ]; then
		print_log "INSTALLED ROS2 $ROS_DISTRO from cache:\n  $TAR_INDEX_URL/$BUILD_CACHE.tar.gz"
		exit 0
	fi
fi

print_log " BUILDING ROS2 $ROS_DISTRO from source ($ROS_PACKAGE)"
set -x

# download ROS sources
# https://answers.ros.org/question/325245/minimal-ros2-installation/?answer=325249#post-id-325249
rosinstall_generator --deps --rosdistro ${ROS_DISTRO} ${ROS_PACKAGE} \
	launch_xml \
	launch_yaml \
	launch_testing \
	launch_testing_ament_cmake \
	demo_nodes_cpp \
	demo_nodes_py \
	example_interfaces \
	camera_calibration_parsers \
	camera_info_manager \
	cv_bridge \
	v4l2_camera \
	vision_opencv \
	vision_msgs \
	image_geometry \
	image_pipeline \
	image_transport \
	compressed_image_transport \
	compressed_depth_image_transport \
 	rosbag2_storage_mcap \
 	rmw_fastrtps \
> ros2.${ROS_DISTRO}.${ROS_PACKAGE}.rosinstall

cat ros2.${ROS_DISTRO}.${ROS_PACKAGE}.rosinstall
vcs import src < ros2.${ROS_DISTRO}.${ROS_PACKAGE}.rosinstall
    
# https://github.com/dusty-nv/jetson-containers/issues/181
rm -r ${ROS_ROOT}/src/ament_cmake
git -C ${ROS_ROOT}/src/ clone https://github.com/ament/ament_cmake -b ${ROS_DISTRO}

# install dependencies using rosdep 
rosdep init
rosdep update
rosdep keys \
  --from-paths src \
	--ignore-src src \
	--rosdistro $ROS_DISTRO \
	| xargs rosdep resolve \
	| grep -v \# \
	| grep -v opencv \
	| grep -v pybind11 \
	> rosdeps.txt

rosdep_install

# restore cmake and numpy versions
bash /tmp/cmake/install.sh
bash /tmp/numpy/install.sh

# build it all - for verbose, see https://answers.ros.org/question/363112/how-to-see-compiler-invocation-in-colcon-build
colcon build \
	--merge-install \
	--cmake-args \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_WARN_DEPRECATED=OFF
    
# remove build files
rm -rf ${ROS_ROOT}/src
rm -rf ${ROS_ROOT}/log
rm -rf ${ROS_ROOT}/build
rm ${ROS_ROOT}/*.rosinstall
    
# cleanup apt   
rm -rf /var/lib/apt/lists/*
apt-get clean

# attempt to upload build
tarpack upload $BUILD_CACHE $ROS_ROOT || echo "failed to upload tarball"