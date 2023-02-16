#!/usr/bin/env bash

source scripts/l4t_version.sh

NGC_GROUP="nvcr.io/ea-linux4tegra"
DOCKERHUB="dustynv"

CONTAINERS=${1:-"all"}

push_retag() 
{
	local registry=$1
	local src_tag=$2
	local dst_tag=$3
	
	sudo docker rmi $registry/$dst_tag
	sudo docker tag $src_tag $registry/$dst_tag
	
	echo "pushing container $src_tag => $registry/$dst_tag"
	sudo docker push $registry/$dst_tag
	echo "done pushing $registry/$dst_tag"
}

push() 
{
	push_retag $1 $2 $2
}


#
# JetPack container
#
#if [[ "$CONTAINERS" == "jetpack" || "$CONTAINERS" == "all" ]]; then
#	jetpack_container="jetpack:r$L4T_VERSION"
#	
#	if [[ "$(sudo docker images -q $jetpack_container 2> /dev/null)" != "" ]]; then
#		push $NGC_GROUP $jetpack_container
#	else
#		echo "did not find $jetpack_container locally, skipping..."
#	fi
#fi


#
# ML containers
#
if [[ "$CONTAINERS" == "pytorch" || "$CONTAINERS" == "all" ]]; then
	#push $NGC_GROUP "l4t-pytorch:r$L4T_VERSION-pth1.11-py3"
	#push $NGC_GROUP "l4t-pytorch:r$L4T_VERSION-pth1.12-py3"
	#push $NGC_GROUP "l4t-pytorch:r$L4T_VERSION-pth1.13-py3"
	push $NGC_GROUP "l4t-pytorch:r$L4T_VERSION-pth2.0-py3"
fi

if [[ "$CONTAINERS" == "tensorflow" || "$CONTAINERS" == "all" ]]; then
	push $NGC_GROUP "l4t-tensorflow:r$L4T_VERSION-tf1.15-py3"
	push $NGC_GROUP "l4t-tensorflow:r$L4T_VERSION-tf2.11-py3"
fi

if [[ "$CONTAINERS" == "ml" || "$CONTAINERS" == "all" ]]; then
	push $NGC_GROUP "l4t-ml:r$L4T_VERSION-py3"
fi


#
# ROS containers
#
if [[ $L4T_RELEASE -eq 34 || $L4T_RELEASE -eq 35 ]]; then   # JetPack 5.x / Ubuntu 20.04
	SUPPORTED_ROS_DISTROS=("noetic" "foxy" "galactic" "humble")
else
	SUPPORTED_ROS_DISTROS=("melodic" "noetic" "eloquent" "foxy" "galactic" "humble")
fi

if [[ "$CONTAINERS" == "ros" ]]; then
	ROS_CONTAINERS=${SUPPORTED_ROS_DISTROS[@]}
elif (echo "${SUPPORTED_ROS_DISTROS[@]}" | fgrep -q "$CONTAINERS"); then
	ROS_CONTAINERS=($CONTAINERS)
fi

for ROS_DISTRO in ${ROS_CONTAINERS[@]}; do
	ros_image="ros:$ROS_DISTRO-ros-base-l4t-r$L4T_VERSION"
	ros_pytorch_image="ros:$ROS_DISTRO-pytorch-l4t-r$L4T_VERSION"
	ros_desktop_image="ros:$ROS_DISTRO-desktop-l4t-r$L4T_VERSION"
	
	push $DOCKERHUB $ros_image
	
	if [[ "$(sudo docker images -q $ros_pytorch_image 2> /dev/null)" != "" ]]; then
		push $DOCKERHUB $ros_pytorch_image
	fi
	
	if [[ "$(sudo docker images -q $ros_desktop_image 2> /dev/null)" != "" ]]; then
		push $DOCKERHUB $ros_desktop_image
	fi
done
