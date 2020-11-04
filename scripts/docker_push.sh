#!/usr/bin/env bash

set -e

# check what to push
if [ "$1" == "" ]
then
  echo "You need to specify two arguments: the first argument is what"
  echo "to push and the second argument is the destination image"
  echo "registry where you are authenticated for pushing."
  echo ""
  echo "Valid values for what to push are 'pytorch', 'tensorflow'"
  echo "'ml', or 'all'."
  exit -1
fi

# check the destination
if [ "$2" == "" ]
then
  echo "You need to specify two arguments: the first argument is what"
  echo "to push and the second argument is the destination image"
  echo "registry where you are authenticated for pushing."
  echo ""
  echo "Valid values for what to push are 'pytorch', 'tensorflow'"
  echo "'ml', or 'all'."
  exit -2
fi

source scripts/l4t_version.sh

REGISTRY=${2:-"docker.io/edgyr"}
CONTAINERS=${1:-"all"}

push_retag() 
{
	local src_tag=$1
	local dst_tag=$2
	
	sudo docker rmi $REGISTRY/$dst_tag
	sudo docker tag $src_tag $REGISTRY/$dst_tag
	
	echo "pushing container $src_tag => $REGISTRY/$dst_tag"
	sudo docker push $REGISTRY/$dst_tag
	echo "done pushing $REGISTRY/$dst_tag"
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
