#!/usr/bin/env bash

set -e

# dev packages
#sh ./scripts/stage_dev.sh

#
# PyTorch (for JetPack 4.4 DP)
#
#  PyTorch v1.2.0 https://nvidia.box.com/shared/static/lufbgr3xu2uha40cs9ryq1zn4kxsnogl.whl (torch-1.2.0-cp36-cp36m-linux_aarch64.whl)
#  PyTorch v1.3.0 https://nvidia.box.com/shared/static/017sci9z4a0xhtwrb4ps52frdfti9iw0.whl (torch-1.3.0-cp36-cp36m-linux_aarch64.whl)
#  PyTorch v1.4.0 https://nvidia.box.com/shared/static/c3d7vm4gcs9m728j6o5vjay2jdedqb55.whl (torch-1.4.0-cp36-cp36m-linux_aarch64.whl)
#  PyTorch v1.5.0 https://nvidia.box.com/shared/static/3ibazbiwtkl181n95n9em3wtrca7tdzp.whl (torch-1.5.0-cp36-cp36m-linux_aarch64.whl)
#
build_pytorch()
{
	local pytorch_url=$1
	local pytorch_whl=$2
	local pytorch_tag=$3
	
	local vision_version=$4
	local pillow_version=$5
	
	echo "building PyTorch $pytorch_whl, torchvision $vision_version ($pillow_version)"
	sh ./scripts/docker_build.sh $pytorch_tag Dockerfile.pytorch \
			--build-arg PYTORCH_URL=$pytorch_url \
			--build-arg PYTORCH_WHL=$pytorch_whl \
			--build-arg TORCHVISION_VERSION=$vision_version \
			--build-arg PILLOW_VERSION=$pillow_version
	echo "done building PyTorch $pytorch_whl, torchvision $vision_version ($pillow_version)"
}

# PyTorch v1.2.0
build_pytorch "https://nvidia.box.com/shared/static/lufbgr3xu2uha40cs9ryq1zn4kxsnogl.whl" \
			  "torch-1.2.0-cp36-cp36m-linux_aarch64.whl" \
			  "l4t-pytorch:r32.4.2-pth1.2-py3" \
			  "v0.4.0" \
			  "pillow<7"

# PyTorch v1.3.0
build_pytorch "https://nvidia.box.com/shared/static/017sci9z4a0xhtwrb4ps52frdfti9iw0.whl" \
			  "torch-1.3.0-cp36-cp36m-linux_aarch64.whl" \
			  "l4t-pytorch:r32.4.2-pth1.3-py3" \
			  "v0.4.2" \
			  "pillow<7"  

# PyTorch v1.4.0
build_pytorch "https://nvidia.box.com/shared/static/c3d7vm4gcs9m728j6o5vjay2jdedqb55.whl" \
			  "torch-1.4.0-cp36-cp36m-linux_aarch64.whl" \
			  "l4t-pytorch:r32.4.2-pth1.4-py3" \
			  "v0.5.0" \
			  "pillow" 

# PyTorch v1.5.0
build_pytorch "https://nvidia.box.com/shared/static/3ibazbiwtkl181n95n9em3wtrca7tdzp.whl" \
			  "torch-1.5.0-cp36-cp36m-linux_aarch64.whl" \
			  "l4t-pytorch:r32.4.2-pth1.5-py3" \
			  "v0.6.0" \
			  "pillow" 
			  
# TensorFlow
sh ./scripts/docker_build.sh l4t-tensorflow:r32.4.2-tf1.15-py3 Dockerfile.tensorflow

# TensorRT
##sh ./scripts/docker_build.sh l4t-tensorrt:r32.4-py3 Dockerfile.tensorrt

# Machine Learning
sh ./scripts/docker_build.sh l4t-ml:r32.4.2-py3 Dockerfile.ml \
	   --build-arg PYTORCH_IMAGE=l4t-pytorch:r32.4.2-pth1.5-py3 \
       --build-arg TENSORFLOW_IMAGE=l4t-tensorflow:r32.4.2-tf1.15-py3

