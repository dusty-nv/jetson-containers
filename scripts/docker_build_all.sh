#!/usr/bin/env bash

set -e

BASE_IMAGE="nvcr.io/nvidia/l4t-base:r32.4.3"

#
# PyTorch (for JetPack 4.4)
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
	local audio_version=$6

	echo "building PyTorch $pytorch_whl, torchvision $vision_version ($pillow_version), torchaudio $audio_version"

	sh ./scripts/docker_build.sh $pytorch_tag Dockerfile.pytorch \
			--build-arg BASE_IMAGE=$BASE_IMAGE \
			--build-arg PYTORCH_URL=$pytorch_url \
			--build-arg PYTORCH_WHL=$pytorch_whl \
			--build-arg TORCHVISION_VERSION=$vision_version \
			--build-arg PILLOW_VERSION=$pillow_version \
			--build-arg TORCHAUDIO_VERSION=$audio_version

	echo "done building PyTorch $pytorch_whl, torchvision $vision_version ($pillow_version), torchaudio $audio_version"
}

# PyTorch v1.2.0
#build_pytorch "https://nvidia.box.com/shared/static/lufbgr3xu2uha40cs9ryq1zn4kxsnogl.whl" \
#			  "torch-1.2.0-cp36-cp36m-linux_aarch64.whl" \
#			  "l4t-pytorch:r32.4.3-pth1.2-py3" \
#			  "v0.4.0" \
#			  "pillow<7"

# PyTorch v1.3.0
#build_pytorch "https://nvidia.box.com/shared/static/017sci9z4a0xhtwrb4ps52frdfti9iw0.whl" \
#			  "torch-1.3.0-cp36-cp36m-linux_aarch64.whl" \
#			  "l4t-pytorch:r32.4.3-pth1.3-py3" \
#			  "v0.4.2" \
#			  "pillow<7"  

# PyTorch v1.4.0
#build_pytorch "https://nvidia.box.com/shared/static/c3d7vm4gcs9m728j6o5vjay2jdedqb55.whl" \
#			  "torch-1.4.0-cp36-cp36m-linux_aarch64.whl" \
#			  "l4t-pytorch:r32.4.3-pth1.4-py3" \
#			  "v0.5.0" \
#			  "pillow" 

# PyTorch v1.5.0
#build_pytorch "https://nvidia.box.com/shared/static/3ibazbiwtkl181n95n9em3wtrca7tdzp.whl" \
#			  "torch-1.5.0-cp36-cp36m-linux_aarch64.whl" \
#			  "l4t-pytorch:r32.4.3-pth1.5-py3" \
#			  "v0.6.0" \
#			  "pillow" 

# PyTorch v1.6.0
build_pytorch "https://nvidia.box.com/shared/static/9eptse6jyly1ggt9axbja2yrmj6pbarc.whl" \
			  "torch-1.6.0-cp36-cp36m-linux_aarch64.whl" \
			  "l4t-pytorch:r32.4.3-pth1.6-py3" \
			  "v0.7.0" \
			  "pillow" \
			  "v0.6.0"

#			  
# TensorFlow (for JetPack 4.4)
#
#  TensorFlow 1.15.2 https://nvidia.box.com/shared/static/8a3q3dz6juk0xg2e2kwwng9teosyohad.whl (tensorflow-1.15.2+nv20.6-cp36-cp36m-linux_aarch64.whl)
#  TensorFlow 2.2.0  https://nvidia.box.com/shared/static/l5lzgqh6cm5kw1b1nzdzuwcpf70xndak.whl (tensorflow-2.2.0+nv20.6-cp36-cp36m-linux_aarch64.whl)
#
build_tensorflow()
{
	local tensorflow_url=$1
	local tensorflow_whl=$2
	local tensorflow_tag=$3
	
	echo "building TensorFlow $tensorflow_whl, $tensorflow_tag"

	sh ./scripts/docker_build.sh $tensorflow_tag Dockerfile.tensorflow \
		--build-arg BASE_IMAGE=$BASE_IMAGE \
		--build-arg TENSORFLOW_URL=$tensorflow_url \
		--build-arg TENSORFLOW_WHL=$tensorflow_whl

	echo "done building TensorFlow $tensorflow_whl, $tensorflow_tag"
}

# TensorFlow 1.15.2
build_tensorflow "https://nvidia.box.com/shared/static/8a3q3dz6juk0xg2e2kwwng9teosyohad.whl" \
			  "tensorflow-1.15.2+nv20.6-cp36-cp36m-linux_aarch64.whl" \
			  "l4t-tensorflow:r32.4.3-tf1.15-py3"

# TensorFlow 2.2.0
build_tensorflow "https://nvidia.box.com/shared/static/l5lzgqh6cm5kw1b1nzdzuwcpf70xndak.whl" \
			  "tensorflow-2.2.0+nv20.6-cp36-cp36m-linux_aarch64.whl" \
			  "l4t-tensorflow:r32.4.3-tf2.2-py3"

#
# Machine Learning
#
sh ./scripts/docker_build.sh l4t-ml:r32.4.3-py3 Dockerfile.ml \
		--build-arg BASE_IMAGE=$BASE_IMAGE \
		--build-arg PYTORCH_IMAGE=l4t-pytorch:r32.4.3-pth1.6-py3 \
		--build-arg TENSORFLOW_IMAGE=l4t-tensorflow:r32.4.3-tf1.15-py3

