#!/usr/bin/env bash

source scripts/l4t_version.sh

REPOSITORY=${2:-"docker.io/edgyr"}
CONTAINERS=${1:-"all"}

push_retag() 
{
	local src_tag=$1
	local dst_tag=$2
	
	sudo docker rmi $REPOSITORY/$dst_tag
	sudo docker tag $src_tag $REPOSITORY/$dst_tag
	
	echo "pushing container $src_tag => $REPOSITORY/$dst_tag"
	sudo docker push $REPOSITORY/$dst_tag
	echo "done pushing $REPOSITORY/$dst_tag"
}

push() 
{
	push_retag $1 $1
}

if [[ "$CONTAINERS" == "pytorch" || "$CONTAINERS" == "all" ]]; then
	#push "l4t-pytorch:r$L4T_VERSION-pth1.2-py3"
	#push "l4t-pytorch:r$L4T_VERSION-pth1.3-py3"
	#push "l4t-pytorch:r$L4T_VERSION-pth1.4-py3"
	#push "l4t-pytorch:r$L4T_VERSION-pth1.5-py3"
	push "l4t-pytorch:r$L4T_VERSION-pth1.6-py3"
fi

if [[ "$CONTAINERS" == "tensorflow" || "$CONTAINERS" == "all" ]]; then
	push "l4t-tensorflow:r$L4T_VERSION-tf1.15-py3"
	push "l4t-tensorflow:r$L4T_VERSION-tf2.3-py3"
fi

if [[ "$CONTAINERS" == "ml" || "$CONTAINERS" == "all" ]]; then
	push "l4t-ml:r$L4T_VERSION-py3"
	push "l4t-ml:r$L4T_VERSION-tf1.15-py3"
	push "l4t-ml:r$L4T_VERSION-tf2.3-py3"
fi
