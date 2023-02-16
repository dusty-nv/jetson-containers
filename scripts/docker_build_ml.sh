#!/usr/bin/env bash

set -e

source scripts/docker_base.sh
source scripts/opencv_version.sh
source scripts/python_version.sh

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
	local audio_version=$5
	local cuda_arch_list="5.3;6.2;7.2"
	
	if [[ $L4T_RELEASE -ge 34 ]]; then  
		cuda_arch_list="7.2;8.7" # JetPack 5.x (Xavier/Orin)
	fi
	
	echo "building PyTorch $pytorch_whl, torchvision $vision_version, torchaudio $audio_version, cuda arch $cuda_arch_list"

	sh ./scripts/docker_build.sh $pytorch_tag Dockerfile.pytorch \
			--build-arg BASE_IMAGE=$BASE_IMAGE \
			--build-arg PYTORCH_URL=$pytorch_url \
			--build-arg PYTORCH_WHL=$pytorch_whl \
			--build-arg TORCHVISION_VERSION=$vision_version \
			--build-arg TORCHAUDIO_VERSION=$audio_version \
			--build-arg TORCH_CUDA_ARCH_LIST=$cuda_arch_list \
			--build-arg OPENCV_URL=$OPENCV_URL \
			--build-arg OPENCV_DEB=$OPENCV_DEB 

	echo "done building PyTorch $pytorch_whl, torchvision $vision_version, torchaudio $audio_version, cuda arch $cuda_arch_list"
}

if [[ "$CONTAINERS" == "pytorch" || "$CONTAINERS" == "all" ]]; then

	if [[ $L4T_RELEASE -eq 32 ]]; then   # JetPack 4.x

		# PyTorch v1.2.0
		#build_pytorch "https://nvidia.box.com/shared/static/lufbgr3xu2uha40cs9ryq1zn4kxsnogl.whl" \
		#			  "torch-1.2.0-cp36-cp36m-linux_aarch64.whl" \
		#			  "l4t-pytorch:r$L4T_VERSION-pth1.2-py3" \
		#			  "v0.4.0"

		# PyTorch v1.3.0
		#build_pytorch "https://nvidia.box.com/shared/static/017sci9z4a0xhtwrb4ps52frdfti9iw0.whl" \
		#			  "torch-1.3.0-cp36-cp36m-linux_aarch64.whl" \
		#			  "l4t-pytorch:r$L4T_VERSION-pth1.3-py3" \
		#			  "v0.4.2"

		# PyTorch v1.4.0
		#build_pytorch "https://nvidia.box.com/shared/static/c3d7vm4gcs9m728j6o5vjay2jdedqb55.whl" \
		#			  "torch-1.4.0-cp36-cp36m-linux_aarch64.whl" \
		#			  "l4t-pytorch:r$L4T_VERSION-pth1.4-py3" \
		#			  "v0.5.0"

		# PyTorch v1.5.0
		#build_pytorch "https://nvidia.box.com/shared/static/3ibazbiwtkl181n95n9em3wtrca7tdzp.whl" \
		#			  "torch-1.5.0-cp36-cp36m-linux_aarch64.whl" \
		#			  "l4t-pytorch:r$L4T_VERSION-pth1.5-py3" \
		#			  "v0.6.0"

		# PyTorch v1.6.0
		#build_pytorch "https://nvidia.box.com/shared/static/9eptse6jyly1ggt9axbja2yrmj6pbarc.whl" \
		#			  "torch-1.6.0-cp36-cp36m-linux_aarch64.whl" \
		#			  "l4t-pytorch:r$L4T_VERSION-pth1.6-py3" \
		#			  "v0.7.0" \
		#			  "v0.6.0"
					  
		# PyTorch v1.7.0
		#build_pytorch "https://nvidia.box.com/shared/static/cs3xn3td6sfgtene6jdvsxlr366m2dhq.whl" \
		#			  "torch-1.7.0-cp36-cp36m-linux_aarch64.whl" \
		#			  "l4t-pytorch:r$L4T_VERSION-pth1.7-py3" \
		#			  "v0.8.1" \
		#			  "v0.7.0"
		
		# PyTorch v1.8.0
		#build_pytorch "https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl" \
		#			"torch-1.8.0-cp36-cp36m-linux_aarch64.whl" \
		#			"l4t-pytorch:r$L4T_VERSION-pth1.8-py3" \
		#			"v0.9.0" \
		#			"v0.8.0"
			
		# PyTorch v1.9.0
		build_pytorch "https://nvidia.box.com/shared/static/h1z9sw4bb1ybi0rm3tu8qdj8hs05ljbm.whl" \
					"torch-1.9.0-cp36-cp36m-linux_aarch64.whl" \
					"l4t-pytorch:r$L4T_VERSION-pth1.9-py3" \
					"v0.10.0" \
					"v0.9.0"
					
		# PyTorch v1.10.0
		build_pytorch "https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl" \
					"torch-1.10.0-cp36-cp36m-linux_aarch64.whl" \
					"l4t-pytorch:r$L4T_VERSION-pth1.10-py3" \
					"v0.11.1" \
					"v0.10.0"
					
		# PyTorch v1.11.0
		build_pytorch "https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.11.0a0+17540c5-cp36-cp36m-linux_aarch64.whl" \
					"torch-1.11.0a0+17540c5-cp36-cp36m-linux_aarch64.whl" \
					"l4t-pytorch:r$L4T_VERSION-pth1.11-py3" \
					"v0.11.3" \
					"v0.10.2"
					
	elif [[ $L4T_RELEASE -eq 34 ]]; then   # JetPack 5.0.0 (DP) / 5.0.1 (DP2)

		# PyTorch v1.11.0
		build_pytorch "https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl" \
					"torch-1.11.0-cp38-cp38-linux_aarch64.whl" \
					"l4t-pytorch:r$L4T_VERSION-pth1.11-py3" \
					"v0.12.0" \
					"v0.11.0"
					
		# PyTorch v1.12.0
		build_pytorch "https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl" \
					"torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl" \
					"l4t-pytorch:r$L4T_VERSION-pth1.12-py3" \
					"v0.12.0" \
					"v0.11.0"
					
	elif [[ $L4T_RELEASE -eq 35 ]] && [[ $L4T_REVISION_MAJOR -le 1 ]]; then   # JetPack 5.0.2 (GA)

		# PyTorch v1.11.0
		build_pytorch "https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl" \
					"torch-1.11.0-cp38-cp38-linux_aarch64.whl" \
					"l4t-pytorch:r$L4T_VERSION-pth1.11-py3" \
					"v0.12.0" \
					"v0.11.0"
					
		# PyTorch v1.12.0
		build_pytorch "https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.12.0a0+8a1a93a9.nv22.5-cp38-cp38-linux_aarch64.whl" \
					"torch-1.12.0a0+8a1a93a9.nv22.5-cp38-cp38-linux_aarch64.whl" \
					"l4t-pytorch:r$L4T_VERSION-pth1.12-py3" \
					"v0.13.0" \
					"v0.12.0"
					
		# PyTorch v1.13.0
		build_pytorch "https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.13.0a0+340c4120.nv22.06-cp38-cp38-linux_aarch64.whl" \
					"torch-1.13.0a0+340c4120.nv22.06-cp38-cp38-linux_aarch64.whl" \
					"l4t-pytorch:r$L4T_VERSION-pth1.13-py3" \
					"v0.13.0" \
					"v0.12.0"
					
	elif [[ $L4T_RELEASE -eq 35 ]]; then   # JetPack 5.1
	
		# PyTorch v2.0
		build_pytorch "https://nvidia.box.com/shared/static/rehpfc4dwsxuhpv4jgqv8u6rzpgb64bq.whl" \
					"torch-2.0.0a0+ec3941ad.nv23.2-cp38-cp38-linux_aarch64.whl" \
					"l4t-pytorch:r$L4T_VERSION-pth2.0-py3" \
					"v0.14.1" \
					"v0.13.1"
					
	else
		echo "warning -- unsupported L4T R$L4T_VERSION, skipping PyTorch..."
	fi
fi

#			  
# TensorFlow
#
build_tensorflow()
{
	local tensorflow_url=$1
	local tensorflow_whl=$2
	local tensorflow_tag=$3
	local protobuf_version=$4
	
	echo "building TensorFlow $tensorflow_whl, $tensorflow_tag"
	
	sh ./scripts/docker_build.sh $tensorflow_tag Dockerfile.tensorflow \
		--build-arg BASE_IMAGE=$BASE_IMAGE \
		--build-arg TENSORFLOW_URL=$tensorflow_url \
		--build-arg TENSORFLOW_WHL=$tensorflow_whl \
		--build-arg PROTOBUF_VERSION=$protobuf_version \
		--build-arg OPENCV_URL=$OPENCV_URL \
		--build-arg OPENCV_DEB=$OPENCV_DEB 
		
	echo "done building TensorFlow $tensorflow_whl, $tensorflow_tag"
}

if [[ "$CONTAINERS" == "tensorflow" || "$CONTAINERS" == "all" ]]; then

	if [[ $L4T_RELEASE -eq 32 ]] && [[ $L4T_REVISION_MAJOR -eq 7 ]]; then
	
		# TensorFlow 1.15.5 for JetPack 4.6.1
		build_tensorflow "https://developer.download.nvidia.com/compute/redist/jp/v461/tensorflow/tensorflow-1.15.5+nv22.1-cp36-cp36m-linux_aarch64.whl" \
					  "tensorflow-1.15.5+nv22.1-cp36-cp36m-linux_aarch64.whl" \
					  "l4t-tensorflow:r$L4T_VERSION-tf1.15-py3" \
					  "3.19.4"

		# TensorFlow 2.7.0 for JetPack 4.6.1
		build_tensorflow "https://developer.download.nvidia.com/compute/redist/jp/v461/tensorflow/tensorflow-2.7.0+nv22.1-cp36-cp36m-linux_aarch64.whl" \
					  "tensorflow-2.7.0+nv22.1-cp36-cp36m-linux_aarch64.whl" \
					  "l4t-tensorflow:r$L4T_VERSION-tf2.7-py3" \
					  "3.19.4"
					  
	elif [[ $L4T_RELEASE -eq 32 ]] && [[ $L4T_REVISION_MAJOR -eq 6 ]]; then
	
		# TensorFlow 1.15.5 for JetPack 4.6
		build_tensorflow "https://nvidia.box.com/shared/static/0e4otnp1pvbo7exwrkermahfrlfe9exo.whl" \
					  "tensorflow-1.15.5+nv21.7-cp36-cp36m-linux_aarch64.whl" \
					  "l4t-tensorflow:r$L4T_VERSION-tf1.15-py3" \
					  "3.19.4"

		# TensorFlow 2.5.0 for JetPack 4.6
		build_tensorflow "https://nvidia.box.com/shared/static/jfbpcioxcb3d3d3wrm1dbtom5aqq5azq.whl" \
					  "tensorflow-2.5.0+nv21.7-cp36-cp36m-linux_aarch64.whl" \
					  "l4t-tensorflow:r$L4T_VERSION-tf2.5-py3" \
					  "3.19.4"
					  
	elif [[ $L4T_RELEASE -eq 32 ]] && [[ $L4T_REVISION_MAJOR -lt 6 ]]; then
	
		# TensorFlow 1.15.5 for JetPack 4.4/4.5
		build_tensorflow "https://developer.download.nvidia.com/compute/redist/jp/v45/tensorflow/tensorflow-1.15.5+nv21.6-cp36-cp36m-linux_aarch64.whl" \
					  "tensorflow-1.15.5+nv21.6-cp36-cp36m-linux_aarch64.whl" \
					  "l4t-tensorflow:r$L4T_VERSION-tf1.15-py3" \
					  "3.19.4"

		# TensorFlow 2.5.0 for JetPack 4.4/4.5
		build_tensorflow "https://developer.download.nvidia.com/compute/redist/jp/v45/tensorflow/tensorflow-2.5.0+nv21.6-cp36-cp36m-linux_aarch64.whl" \
					  "tensorflow-2.5.0+nv21.6-cp36-cp36m-linux_aarch64.whl" \
					  "l4t-tensorflow:r$L4T_VERSION-tf2.5-py3" \
					  "3.19.4"
	
	elif [[ $L4T_RELEASE -eq 34 ]] && [[ $L4T_REVISION_MAJOR -le 1 ]]; then
	
		# TensorFlow 1.15.5 for JetPack 5.0.0 / 5.0.1
		build_tensorflow "https://developer.download.nvidia.com/compute/redist/jp/v50/tensorflow/tensorflow-1.15.5+nv22.4-cp38-cp38-linux_aarch64.whl" \
					  "tensorflow-1.15.5+nv22.4-cp38-cp38-linux_aarch64.whl" \
					  "l4t-tensorflow:r$L4T_VERSION-tf1.15-py3" \
					  "3.20.1"

		# TensorFlow 2.8 for JetPack 5.0.0 / 5.0.1
		build_tensorflow "https://developer.download.nvidia.com/compute/redist/jp/v50/tensorflow/tensorflow-2.8.0+nv22.4-cp38-cp38-linux_aarch64.whl" \
					  "tensorflow-2.8.0+nv22.4-cp38-cp38-linux_aarch64.whl" \
					  "l4t-tensorflow:r$L4T_VERSION-tf2.8-py3" \
					  "3.20.1"
	
	elif [[ $L4T_RELEASE -eq 35 ]] && [[ $L4T_REVISION_MAJOR -le 1 ]]; then
	
		# TensorFlow 1.15.5 for JetPack 5.0.2
		build_tensorflow "https://developer.download.nvidia.com/compute/redist/jp/v50/tensorflow/tensorflow-1.15.5+nv22.5-cp38-cp38-linux_aarch64.whl" \
					  "tensorflow-1.15.5+nv22.5-cp38-cp38-linux_aarch64.whl" \
					  "l4t-tensorflow:r$L4T_VERSION-tf1.15-py3" \
					  "3.20.1"

		# TensorFlow 2.9 for JetPack 5.0.2
		build_tensorflow "https://developer.download.nvidia.com/compute/redist/jp/v50/tensorflow/tensorflow-2.9.1+nv22.06-cp38-cp38-linux_aarch64.whl" \
					  "tensorflow-2.9.1+nv22.06-cp38-cp38-linux_aarch64.whl" \
					  "l4t-tensorflow:r$L4T_VERSION-tf2.9-py3" \
					  "3.20.1"
		
	elif [[ $L4T_RELEASE -eq 35 ]]; then
	
		# TensorFlow 1.15.5 for JetPack 5.1
		build_tensorflow "https://nvidia.box.com/shared/static/28np6obvzx6hwrh6ufhfsusg616t575c.whl" \
					  "tensorflow-1.15.5+nv23.01-cp38-cp38-linux_aarch64.whl" \
					  "l4t-tensorflow:r$L4T_VERSION-tf1.15-py3" \
					  "3.20.3"

		# TensorFlow 2.11 for JetPack 5.1
		build_tensorflow "https://nvidia.box.com/shared/static/9vcmax1wmlqlw2e2r1adh1c045k2ju21.whl" \
					  "tensorflow-2.11.0+nv23.01-cp38-cp38-linux_aarch64.whl" \
					  "l4t-tensorflow:r$L4T_VERSION-tf2.11-py3" \
					  "3.20.3"
					  
	else
		echo "warning -- unsupported L4T R$L4T_VERSION, skipping TensorFlow..."
	fi
fi


#
# Machine Learning
#
if [[ "$CONTAINERS" == "all" ]]; then

	sh ./scripts/docker_build.sh l4t-ml:r$L4T_VERSION-py3 Dockerfile.ml \
			--build-arg BASE_IMAGE=$BASE_IMAGE \
			--build-arg PYTORCH_IMAGE=l4t-pytorch:r$L4T_VERSION-pth2.0-py3 \
			--build-arg TENSORFLOW_IMAGE=l4t-tensorflow:r$L4T_VERSION-tf2.11-py3 \
			--build-arg PYTHON3_VERSION=$PYTHON3_VERSION \
			--build-arg OPENCV_URL=$OPENCV_URL \
			--build-arg OPENCV_DEB=$OPENCV_DEB 
fi


