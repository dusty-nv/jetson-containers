#!/usr/bin/env bash

set -e
source scripts/docker_base.sh

CONTAINERS=${1:-"all"}

#
# PyTorch 
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

if [[ "$CONTAINERS" == "pytorch" || "$CONTAINERS" == "all" ]]; then

	# PyTorch v1.2.0
	#build_pytorch "https://nvidia.box.com/shared/static/lufbgr3xu2uha40cs9ryq1zn4kxsnogl.whl" \
	#			  "torch-1.2.0-cp36-cp36m-linux_aarch64.whl" \
	#			  "l4t-pytorch:r$L4T_VERSION-pth1.2-py3" \
	#			  "v0.4.0" \
	#			  "pillow<7"

	# PyTorch v1.3.0
	#build_pytorch "https://nvidia.box.com/shared/static/017sci9z4a0xhtwrb4ps52frdfti9iw0.whl" \
	#			  "torch-1.3.0-cp36-cp36m-linux_aarch64.whl" \
	#			  "l4t-pytorch:r$L4T_VERSION-pth1.3-py3" \
	#			  "v0.4.2" \
	#			  "pillow<7"  

	# PyTorch v1.4.0
	#build_pytorch "https://nvidia.box.com/shared/static/c3d7vm4gcs9m728j6o5vjay2jdedqb55.whl" \
	#			  "torch-1.4.0-cp36-cp36m-linux_aarch64.whl" \
	#			  "l4t-pytorch:r$L4T_VERSION-pth1.4-py3" \
	#			  "v0.5.0" \
	#			  "pillow" 

	# PyTorch v1.5.0
	#build_pytorch "https://nvidia.box.com/shared/static/3ibazbiwtkl181n95n9em3wtrca7tdzp.whl" \
	#			  "torch-1.5.0-cp36-cp36m-linux_aarch64.whl" \
	#			  "l4t-pytorch:r$L4T_VERSION-pth1.5-py3" \
	#			  "v0.6.0" \
	#			  "pillow" 

	# PyTorch v1.6.0
	build_pytorch "https://nvidia.box.com/shared/static/9eptse6jyly1ggt9axbja2yrmj6pbarc.whl" \
				  "torch-1.6.0-cp36-cp36m-linux_aarch64.whl" \
				  "l4t-pytorch:r$L4T_VERSION-pth1.6-py3" \
				  "v0.7.0" \
				  "pillow" \
				  "v0.6.0"
				  
	# PyTorch v1.7.0
	build_pytorch "https://nvidia.box.com/shared/static/cs3xn3td6sfgtene6jdvsxlr366m2dhq.whl" \
				  "torch-1.7.0-cp36-cp36m-linux_aarch64.whl" \
				  "l4t-pytorch:r$L4T_VERSION-pth1.7-py3" \
				  "v0.8.1" \
				  "pillow" \
				  "v0.7.0"
fi

#			  
# TensorFlow
#
build_tensorflow()
{
	local tensorflow_url=$1
	local tensorflow_whl=$2
	local tensorflow_tag=$3
	
	echo "building TensorFlow $tensorflow_whl, $tensorflow_tag"

	sh ./scripts/docker_build.sh $tensorflow_tag Dockerfile_custom.tensorflow \
		--build-arg BASE_IMAGE=$BASE_IMAGE \
		--build-arg TENSORFLOW_URL=$tensorflow_url \
		--build-arg TENSORFLOW_WHL=$tensorflow_whl

	echo "done building TensorFlow $tensorflow_whl, $tensorflow_tag"
}

if [[ "$CONTAINERS" == "tensorflow" || "$CONTAINERS" == "all" ]]; then

	# TensorFlow 1.15.4
	build_tensorflow "https://developer.download.nvidia.com/compute/redist/jp/v44/tensorflow/tensorflow-1.15.4+nv20.11-cp36-cp36m-linux_aarch64.whl" \
				  "tensorflow-1.15.4+nv20.11-cp36-cp36m-linux_aarch64.whl" \
				  "l4t-tensorflow:r$L4T_VERSION-tf1.15-py3"

	# TensorFlow 2.3.1
	build_tensorflow "https://developer.download.nvidia.com/compute/redist/jp/v44/tensorflow/tensorflow-2.3.1+nv20.11-cp36-cp36m-linux_aarch64.whl" \
				  "tensorflow-2.3.1+nv20.11-cp36-cp36m-linux_aarch64.whl" \
				  "l4t-tensorflow:r$L4T_VERSION-tf2.3-py3"
fi

#
# Machine Learning
#
if [[ "$CONTAINERS" == "all" ]]; then

	# alternate source:  http://repo.download.nvidia.com/jetson/jetson-ota-public.asc
	cp /etc/apt/trusted.gpg.d/jetson-ota-public.asc .
	
	sh ./scripts/docker_build.sh l4t-ml:r$L4T_VERSION-py3 Dockerfile.ml \
			--build-arg BASE_IMAGE=$BASE_IMAGE \
			--build-arg PYTORCH_IMAGE=l4t-pytorch:r$L4T_VERSION-pth1.7-py3 \
			--build-arg TENSORFLOW_IMAGE=l4t-tensorflow:r$L4T_VERSION-tf1.15-py3 \
			--build-arg L4T_APT_SOURCE="deb https://repo.download.nvidia.com/jetson/common r32.4 main"

			#--build-arg L4T_APT_KEY=$L4T_APT_KEY \
			#--build-arg L4T_APT_SOURCE="$(head -1 /etc/apt/sources.list.d/nvidia-l4t-apt-source.list | sed 's/'"$L4T_APT_SERVER_PUBLIC"'/'"$L4T_APT_SERVER"'/g')"
fi


