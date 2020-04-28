#!/usr/bin/env bash

NGC_GROUP="nvcr.io/ea-linux4tegra"

push_retag() 
{
	local src_tag=$1
	local dst_tag=$2
	
	sudo docker rmi $NGC_GROUP/$dst_tag
	sudo docker tag $src_tag $NGC_GROUP/$dst_tag
	sudo docker push $NGC_GROUP/$dst_tag
}

push() 
{
	push_retag $1 $1
}

push "l4t-pytorch:r32.4.2-pth1.2-py3"
push "l4t-pytorch:r32.4.2-pth1.3-py3"
push "l4t-pytorch:r32.4.2-pth1.4-py3"
push "l4t-pytorch:r32.4.2-pth1.5-py3"

push "l4t-tensorflow:r32.4.2-tf1.15-py3"

push "l4t-ml:r32.4.2-py3"



