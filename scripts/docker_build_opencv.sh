#!/usr/bin/env bash

set -e
source scripts/docker_base.sh

OPENCV_VERSION=${1:-"4.5.0"}

build_opencv()
{
	local opencv_version=$1
	local container_tag="opencv-builder:r$L4T_VERSION-cv$opencv_version"

	# opencv.csv mounts files that preclude us installing different version of opencv
	# temporarily disable the opencv.csv mounts while we build the container
	CV_CSV="/etc/nvidia-container-runtime/host-files-for-container.d/opencv.csv"
	
	if [ -f "$CV_CSV" ]; then
		sudo mv $CV_CSV $CV_CSV.backup
	fi
	
	echo "building OpenCV $opencv_version deb packages"

	sh ./scripts/docker_build.sh $container_tag Dockerfile.opencv \
			--build-arg BASE_IMAGE=$BASE_IMAGE \
			--build-arg OPENCV_VERSION=$OPENCV_VERSION

	echo "done building OpenCV $opencv_version deb packages"
	
	if [ -f "$CV_CSV.backup" ]; then
		sudo mv $CV_CSV.backup $CV_CSV
	fi
	
	# copy deb packages to jetson-containers/packages directory
	sudo docker run --rm \
			--volume $PWD/packages:/mount \
			$container_tag \
			cp opencv/build/OpenCV-${opencv_version}-aarch64.tar.gz /mount
			
	echo "packages are at $PWD/packages/OpenCV-${opencv_version}-aarch64.tar.gz"
}
	
build_opencv $OPENCV_VERSION
