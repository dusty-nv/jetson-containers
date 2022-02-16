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
# ML containers
#
if [[ "$CONTAINERS" == "pytorch" || "$CONTAINERS" == "all" ]]; then
	#push "l4t-pytorch:r$L4T_VERSION-pth1.2-py3"
	#push "l4t-pytorch:r$L4T_VERSION-pth1.3-py3"
	#push "l4t-pytorch:r$L4T_VERSION-pth1.4-py3"
	#push "l4t-pytorch:r$L4T_VERSION-pth1.5-py3"
	#push $NGC_GROUP "l4t-pytorch:r$L4T_VERSION-pth1.6-py3"
	#push $NGC_GROUP "l4t-pytorch:r$L4T_VERSION-pth1.7-py3"
	#push $NGC_GROUP "l4t-pytorch:r$L4T_VERSION-pth1.8-py3"
	push $NGC_GROUP "l4t-pytorch:r$L4T_VERSION-pth1.9-py3"
	push $NGC_GROUP "l4t-pytorch:r$L4T_VERSION-pth1.10-py3"
fi

if [[ "$CONTAINERS" == "tensorflow" || "$CONTAINERS" == "all" ]]; then
	push $NGC_GROUP "l4t-tensorflow:r$L4T_VERSION-tf1.15-py3"
	push $NGC_GROUP "l4t-tensorflow:r$L4T_VERSION-tf2.7-py3"
fi

if [[ "$CONTAINERS" == "ml" || "$CONTAINERS" == "all" ]]; then
	push $NGC_GROUP "l4t-ml:r$L4T_VERSION-py3"
fi


#
# ROS containers
#
SUPPORTED_ROS_DISTROS=("melodic" "noetic" "eloquent" "foxy" "galactic")

if [[ "$CONTAINERS" == "ros" ]]; then
	ROS_CONTAINERS=${SUPPORTED_ROS_DISTROS[@]}
elif (echo "${SUPPORTED_ROS_DISTROS[@]}" | fgrep -q "$CONTAINERS"); then
	ROS_CONTAINERS=($CONTAINERS)
fi

for ROS_DISTRO in ${ROS_CONTAINERS[@]}; do
	ros_image="ros:$ROS_DISTRO-ros-base-l4t-r$L4T_VERSION"
	ros_pytorch_image="ros:$ROS_DISTRO-pytorch-l4t-r$L4T_VERSION"
	ros_slam_image="ros:$ROS_DISTRO-slam-l4t-r$L4T_VERSION"
	
	push $DOCKERHUB $ros_image
	
	if [[ "$(sudo docker images -q $ros_pytorch_image 2> /dev/null)" != "" ]]; then
		push $DOCKERHUB $ros_pytorch_image
	fi
	
	if [[ "$(sudo docker images -q $ros_slam_image 2> /dev/null)" != "" ]]; then
		push $DOCKERHUB $ros_slam_image
	fi
done
