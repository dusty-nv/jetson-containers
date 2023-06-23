#!/usr/bin/env bash

set -e
source scripts/l4t_version.sh

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
TEST_MOUNT="$ROOT/../test:/test"
CONTAINERS=${1:-"all"}

# cuda tests
test_cuda()
{
	echo "testing container $1 => PyCUDA"
	bash ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_cuda.py
	echo -e "done testing container $1 => PyCUDA\n"
}

# cupy tests
test_cupy()
{
	echo "testing container $1 => CuPy"
	bash ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_cupy.py
	echo -e "done testing container $1 => CuPy\n"
}

# nemo tests
test_nemo()
{
	echo "testing container $1 => nemo"
	bash ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_nemo.py
	echo -e "done testing container $1 => nemo\n"
}

# numpy tests
test_numpy()
{
	echo "testing container $1 => numpy"
	bash ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_numpy.py
	echo -e "done testing container $1 => numpy\n"
}

# numba tests
test_numba()
{
	echo "testing container $1 => numba"
	bash ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_numba.py
	echo -e "done testing container $1 => numba\n"
}

# onnx tests
test_onnx()
{
	echo "testing container $1 => onnx"
	bash ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_onnx.py
	echo -e "done testing container $1 => onnx\n"
}

# onnxruntime tests
test_onnxruntime()
{
	echo "testing container $1 => onnxruntime"

	# download test model
	local MODEL_URL="https://nvidia.box.com/shared/static/zlvb4y43djygotpjn6azjhwu0r3j0yxc.gz"
	local MODEL_NAME="cat_dog_epoch_100"
	local MODEL_PATH="test/data/$MODEL_NAME/resnet18.onnx"

	if [ ! -f "$MODEL_PATH" ]; then
		echo 'downloading model for testing onnxruntime...'
		wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate $MODEL_URL -O test/data/$MODEL_NAME.tar.gz
		tar -xzf test/data/$MODEL_NAME.tar.gz -C test/data/
	fi

	bash ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_onnxruntime.py --model=$MODEL_PATH

	echo -e "done testing container $1 => onnxruntime\n"
}

# opencv tests
test_opencv()
{
	echo "testing container $1 => OpenCV"
	bash ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_opencv.py
	echo -e "done testing container $1 => OpenCV\n"
}

# pandas tests
test_pandas()
{
	echo "testing container $1 => pandas"
	bash ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_pandas.py
	echo -e "done testing container $1 => pandas\n"
}

# PyTorch tests
test_pytorch()
{
	echo "testing container $1 => PyTorch"

	bash ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_pytorch.py
	
	# download data for testing torchvision models
	DATA_URL="https://nvidia.box.com/shared/static/y1ygiahv8h75yiyh0pt50jqdqt7pohgx.gz"
	DATA_NAME="ILSVRC2012_img_val_subset_5k"
	DATA_PATH="test/data/$DATA_NAME"

	if [ ! -d "$DATA_PATH" ]; then
		echo 'downloading data for testing torchvision...'
		wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate $DATA_URL -O test/data/$DATA_NAME.tar.gz
		tar -xzf test/data/$DATA_NAME.tar.gz -C test/data/
	fi

	bash ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_torchvision.py --data=$DATA_PATH --use-cuda
	bash ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_torchaudio.py

	echo -e "done testing container $1 => PyTorch\n"
}

# protobuf tests
test_protobuf()
{
	echo "testing container $1 => protobuf"
	bash ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r bash test/test_protobuf.sh
	bash ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_protobuf.py
	echo -e "done testing container $1 => protobuf\n"
}

# TensorFlow tests
test_tensorflow()
{
	echo "testing container $1 => TensorFlow"
	bash ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_tensorflow.py
	echo -e "done testing container $1 => TensorFlow\n"
}

# TensorRT tests
test_tensorrt()
{
	echo "testing container $1 => TensorRT"
	bash ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_tensorrt.py
	echo -e "done testing container $1 => TensorRT\n"
}

# transformers tests
test_transformers()
{
	echo "testing container $1 => transformers"
	bash ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_transformers.py --model=distilgpt2 --provider=cuda
	bash ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_transformers.py --model=distilgpt2 --provider=tensorrt --fp16
	echo -e "done testing container $1 => transformers\n"
}

# scipy tests
test_scipy()
{
	echo "testing container $1 => scipy"
	bash ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_scipy.py
	echo -e "done testing container $1 => scipy\n"
}

# sklearn tests
test_sklearn()
{
	echo "testing container $1 => sklearn"
	bash ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_sklearn.py
	echo -e "done testing container $1 => sklearn\n"
}

# vpi tests
test_vpi()
{
	echo "testing container $1 => VPI"
	bash ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_vpi.py
	bash ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r test/test_vpi.sh
	echo -e "done testing container $1 => VPI\n"
}

# PyTorch tests (all)
test_pytorch_all()
{
	test_pytorch $1
	test_tensorrt $1
	test_cuda $1
	test_numpy $1
	#test_vpi $1 
	
	if [[ $L4T_RELEASE -ge 34 ]]; then
		test_opencv $1
	fi
}

# TensorFlow tests (all)
test_tensorflow_all()
{
	test_protobuf $1
	test_tensorflow $1
	test_tensorrt $1
	test_cuda $1
	test_numpy $1
	#test_vpi $1
	
	if [[ $L4T_RELEASE -ge 34 ]]; then
		test_opencv $1
	fi
}

# ML tests (all)
test_all()
{
	test_pytorch $1
	test_protobuf $1
	test_tensorflow $1
	test_tensorrt $1
	test_cuda $1
	test_numpy $1
	test_cupy $1
	test_numba $1
	test_nemo $1
	test_onnx $1
	test_onnxruntime $1
	test_opencv $1
	test_pandas $1
	test_scipy $1
	test_sklearn $1
	test_transformers $1
	#test_vpi $1
}


#
# PyTorch container
#
if [[ "$CONTAINERS" == "pytorch" || "$CONTAINERS" == "all" ]]; then
	#test_pytorch_all "l4t-pytorch:r$L4T_VERSION-pth1.10-py3"
	#test_pytorch_all "l4t-pytorch:r$L4T_VERSION-pth1.11-py3"
	#test_pytorch_all "l4t-pytorch:r$L4T_VERSION-pth1.12-py3"
	#test_pytorch_all "l4t-pytorch:r$L4T_VERSION-pth1.13-py3"
	test_pytorch_all "l4t-pytorch:r$L4T_VERSION-pth2.0-py3"
fi

#
# TensorFlow container
#
if [[ "$CONTAINERS" == "tensorflow" || "$CONTAINERS" == "all" ]]; then
	#test_tensorflow_all "l4t-tensorflow:r$L4T_VERSION-tf1.15-py3"
	test_tensorflow_all "l4t-tensorflow:r$L4T_VERSION-tf2.11-py3"
fi

#
# ML container
#
if [[ "$CONTAINERS" == "ml" || "$CONTAINERS" == "all" ]]; then
	test_all "l4t-ml:r$L4T_VERSION-py3"
fi


