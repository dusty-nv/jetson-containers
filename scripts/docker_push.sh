#!/usr/bin/env bash

NGC_GROUP="nvcr.io/ea-linux4tegra"

push() 
{
	local container=$1

	sudo docker rmi $NGC_GROUP/$container
	sudo docker tag $container $NGC_GROUP/$container
	sudo docker push $NGC_GROUP/$container
}

push "l4t-pytorch:r32.4-pth1.2-py3"


