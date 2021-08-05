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
	#build_pytorch "https://nvidia.box.com/shared/static/9eptse6jyly1ggt9axbja2yrmj6pbarc.whl" \
	#			  "torch-1.6.0-cp36-cp36m-linux_aarch64.whl" \
	#			  "l4t-pytorch:r$L4T_VERSION-pth1.6-py3" \
	#			  "v0.7.0" \
	#			  "pillow" \
	#			  "v0.6.0"
				  
	# PyTorch v1.7.0
	#build_pytorch "https://nvidia.box.com/shared/static/cs3xn3td6sfgtene6jdvsxlr366m2dhq.whl" \
	#			  "torch-1.7.0-cp36-cp36m-linux_aarch64.whl" \
	#			  "l4t-pytorch:r$L4T_VERSION-pth1.7-py3" \
	#			  "v0.8.1" \
	#			  "pillow" \
	#			  "v0.7.0"
	
	# PyTorch v1.8.0
	build_pytorch "https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl" \
				"torch-1.8.0-cp36-cp36m-linux_aarch64.whl" \
				"l4t-pytorch:r$L4T_VERSION-pth1.8-py3" \
				"v0.9.0" \
				"pillow" \
				"v0.8.0"
		
	# PyTorch v1.9.0
	build_pytorch "https://nvidia.box.com/shared/static/h1z9sw4bb1ybi0rm3tu8qdj8hs05ljbm.whl" \
				"torch-1.9.0-cp36-cp36m-linux_aarch64.whl" \
				"l4t-pytorch:r$L4T_VERSION-pth1.9-py3" \
				"v0.10.0" \
				"pillow" \
				"v0.9.0"
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
	
	sh ./scripts/docker_build.sh $tensorflow_tag Dockerfile.tensorflow \
		--build-arg BASE_IMAGE=$BASE_IMAGE \
		--build-arg TENSORFLOW_URL=$tensorflow_url \
		--build-arg TENSORFLOW_WHL=$tensorflow_whl

	echo "done building TensorFlow $tensorflow_whl, $tensorflow_tag"
}

if [[ "$CONTAINERS" == "tensorflow" || "$CONTAINERS" == "all" ]]; then

	if [[ $L4T_RELEASE -eq 32 ]] && [[ $L4T_REVISION_MAJOR -ge 6 ]]; then
		# TensorFlow 1.15.5 for JetPack 4.6
		build_tensorflow "https://nvidia.box.com/shared/static/0e4otnp1pvbo7exwrkermahfrlfe9exo.whl" \
					  "tensorflow-1.15.5+nv21.7-cp36-cp36m-linux_aarch64.whl" \
					  "l4t-tensorflow:r$L4T_VERSION-tf1.15-py3"

		# TensorFlow 2.5.0 for JetPack 4.6
		build_tensorflow "https://nvidia.box.com/shared/static/jfbpcioxcb3d3d3wrm1dbtom5aqq5azq.whl" \
					  "tensorflow-2.5.0+nv21.7-cp36-cp36m-linux_aarch64.whl" \
					  "l4t-tensorflow:r$L4T_VERSION-tf2.5-py3"
	else
		# TensorFlow 1.15.5 for JetPack 4.4/4.5
		build_tensorflow "https://developer.download.nvidia.com/compute/redist/jp/v45/tensorflow/tensorflow-1.15.5+nv21.6-cp36-cp36m-linux_aarch64.whl" \
					  "tensorflow-2.5.0+nv21.6-cp36-cp36m-linux_aarch64.whl" \
					  "l4t-tensorflow:r$L4T_VERSION-tf1.15-py3"

		# TensorFlow 2.5.0 for JetPack 4.4/4.5
		build_tensorflow "https://developer.download.nvidia.com/compute/redist/jp/v45/tensorflow/tensorflow-2.5.0+nv21.6-cp36-cp36m-linux_aarch64.whl" \
					  "tensorflow-2.5.0+nv21.6-cp36-cp36m-linux_aarch64.whl" \
					  "l4t-tensorflow:r$L4T_VERSION-tf2.5-py3"
	fi
fi


#
# Machine Learning
#
if [[ "$CONTAINERS" == "all" ]]; then

	# opencv.csv mounts files that preclude us installing different version of opencv
	# temporarily disable the opencv.csv mounts while we build the container
	CV_CSV="/etc/nvidia-container-runtime/host-files-for-container.d/opencv.csv"
	
	if [ -f "$CV_CSV" ]; then
		sudo mv $CV_CSV $CV_CSV.backup
	fi
	
	sh ./scripts/docker_build.sh l4t-ml:r$L4T_VERSION-py3 Dockerfile.ml \
			--build-arg BASE_IMAGE=$BASE_IMAGE \
			--build-arg PYTORCH_IMAGE=l4t-pytorch:r$L4T_VERSION-pth1.9-py3 \
			--build-arg TENSORFLOW_IMAGE=l4t-tensorflow:r$L4T_VERSION-tf1.15-py3 #\

	if [ -f "$CV_CSV.backup" ]; then
		sudo mv $CV_CSV.backup $CV_CSV
	fi
fi


