#!/usr/bin/env bash
#
# Builds the self-contained JetPack Docker container, including development 
# headers/libraries/samples for CUDA Toolkit, cuDNN, TensorRT, VPI, and OpenCV.
#
# Run this script as follows:
#
#   $ cd jetson-containers
#   $ scripts/docker_build_jetpack.sh
#
set -e

source scripts/docker_base.sh
source scripts/opencv_version.sh

# 
# Notes for running local pre-release apt server:
#
# copy .deb's from SDK Manager into jetson-containers/packages/apt/<JETPACK_VERSION>
#
# extract debs from CUDA local repo:
#   mkdir -p cuda/tmp
#   cd cuda/tmp
#   ar -x ../../cuda-repo-l4t-11-4-local_11.4.14-1_arm64.deb
#   tar -xvf data.tar.xz
#   mv var/cuda-repo-l4t-11-4-local/*.deb ../
#   cd ../../
#   rm -rf cuda/tmp
#
# extract debs from cuDNN local repo:
#   mkdir -p cudnn/tmp
#   cd cudnn/tmp
#   ar -x ../../cudnn-local-repo-ubuntu2004-8.3.2.49_1.0-1_arm64.deb
#   tar -xvf data.tar.gz
#   mv var/cudnn-local-repo-ubuntu2004-8.3.2.49/*.deb ../
#   cd ../../
#   rm -rf cudnn/tmp
# 
# https://wiki.debian.org/DebianRepository/Setup#apt-ftparchive
#   sudo apt-get install apt-utils
#   apt-ftparchive packages . > Packages
#   apt-ftparchive release . > Release
#
# start webserver:
#   python3 -m http.server
#
# add to apt sources:
#   deb [trusted=yes] http://127.0.0.1:8000 ./
#

build_jetpack()
{
	local base_image=$1
	local container_tag="jetpack:r$L4T_VERSION"

	# print available packages
	grep -h -P -o "^Package: \K.*" /var/lib/apt/lists/repo.download.nvidia.com_jetson_common_dists_*_main_binary-arm64_Packages | sort -u
	#grep -h -P -o "^Package: \K.*" /var/lib/apt/lists/repo.download.nvidia.com_jetson_t194_dists_*_main_binary-arm64_Packages | sort -u
    
	# configure apt sources
	local l4t_apt_source="deb https://repo.download.nvidia.com/jetson/common r$L4T_RELEASE.$L4T_REVISION_MAJOR main"

	if [ $L4T_RELEASE -eq 34 ]; then
		if [ $L4T_REVISION_MAJOR -eq 0 ]; then
			# JetPack 5.0 EA
			l4t_apt_source="deb https://repo.download.nvidia.com/jetson/jetson-50/common r34.0 main"
		elif [ $L4T_REVISION_MAJOR -ge 2 ]; then
			# JetPack 5.x pre-releases (see above for setting up local apt server)
			l4t_apt_source="deb [trusted=yes] http://127.0.0.1:8000 ./"
			l4t_apt_source_clean=""
		else
			# l4t-base in JetPack 5 already includes the nvidia apt repo
			l4t_apt_source=""
		fi
	fi
	
	if ! [ -v l4t_apt_source_clean ]; then
		l4t_apt_source_clean=$l4t_apt_source
	fi
	
	echo $l4t_apt_source > packages/nvidia-l4t-apt-source.list
	echo $l4t_apt_source_clean > packages/nvidia-l4t-apt-source.clean.list
	
	# build container
	echo ""
	echo "Building container $container_tag"
	echo "BASE_IMAGE=$base_image"
	echo "L4T_APT_SOURCE=$l4t_apt_source"
	echo "L4T_APT_SOURCE_CLEAN=$l4t_apt_source_clean"
	echo ""
	
	sh ./scripts/docker_build.sh $container_tag Dockerfile.jetpack \
			--build-arg BASE_IMAGE=$base_image \
			--build-arg OPENCV_URL=$OPENCV_URL \
			--build-arg OPENCV_DEB=$OPENCV_DEB

}
	
build_jetpack $BASE_IMAGE_L4T


