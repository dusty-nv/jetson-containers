#!/usr/bin/env bash

set -e
source scripts/docker_base.sh


build_jetpack()
{
	local base_image=$1
	local container_tag="jetpack:r$L4T_VERSION"

	# print available packages
	grep -h -P -o "^Package: \K.*" /var/lib/apt/lists/repo.download.nvidia.com_jetson_common_dists_*_main_binary-arm64_Packages | sort -u
	#grep -h -P -o "^Package: \K.*" /var/lib/apt/lists/repo.download.nvidia.com_jetson_t194_dists_*_main_binary-arm64_Packages | sort -u
    
	# configure apt sources
	local l4t_apt_source="deb https://repo.download.nvidia.com/jetson/common r$L4T_RELEASE.$L4T_REVISION_MAJOR main"
	
	echo "L4T_APT_SOURCE=$l4t_apt_source"
	
	if [ $L4T_RELEASE -eq 34 ]; then
		if [ $L4T_REVISION_MAJOR -lt 2 ]; then
			l4t_apt_source="deb https://repo.download.nvidia.com/jetson/jetson-50/common r34.0 main"
		fi
	fi
	
	echo $l4t_apt_source > packages/nvidia-l4t-apt-source.list
	
	# build container
	echo ""
	echo "Building container $container_tag"
	echo "BASE_IMAGE=$base_image"
	echo "L4T_APT_SOURCE=$l4t_apt_source"
	echo ""
	
	sh ./scripts/docker_build.sh $container_tag Dockerfile.jetpack \
			--build-arg BASE_IMAGE=$base_image

}
	
build_jetpack $BASE_IMAGE_L4T


