#!/usr/bin/env bash

CONTAINER=$1
DOCKERFILE=$2

shift 
shift

#sudo cp cuda-devel.csv /etc/nvidia-container-runtime/host-files-for-container.d/

echo "Building $CONTAINER container..."

sudo docker build -t $CONTAINER -f $DOCKERFILE "$@" .

