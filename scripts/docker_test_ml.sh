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
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_cuda.py
	echo -e "done testing container $1 => PyCUDA\n"
}

# cupy tests
test_cupy()
{
	echo "testing container $1 => CuPy"
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_cupy.py
	echo -e "done testing container $1 => CuPy\n"
}

# numpy tests
test_numpy()
{
	echo "testing container $1 => numpy"
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_numpy.py
	echo -e "done testing container $1 => numpy\n"
}

# numpa tests
test_numba()
{
	echo "testing container $1 => numba"
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_numba.py
	echo -e "done testing container $1 => numba\n"
}

# onnx tests
test_onnx()
{
	echo "testing container $1 => onnx"
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_onnx.py
	echo -e "done testing container $1 => onnx\n"
}

# opencv tests
test_opencv()
{
	echo "testing container $1 => OpenCV"
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_opencv.py
	echo -e "done testing container $1 => OpenCV\n"
}

# pandas tests
test_pandas()
{
	echo "testing container $1 => pandas"
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_pandas.py
	echo -e "done testing container $1 => pandas\n"
}

# PyTorch tests
test_pytorch()
{
	echo "testing container $1 => PyTorch"

	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_pytorch.py
	
	# download data for testing torchvision models
	DATA_URL="https://nvidia.box.com/shared/static/y1ygiahv8h75yiyh0pt50jqdqt7pohgx.gz"
	DATA_NAME="ILSVRC2012_img_val_subset_5k"
	DATA_PATH="test/data/$DATA_NAME"

	if [ ! -d "$DATA_PATH" ]; then
		echo 'downloading data for testing torchvision...'
		if [ ! -d "test/data" ]; then
			mkdir test/data
		fi
		wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate $DATA_URL -O test/data/$DATA_NAME.tar.gz
		tar -xzf test/data/$DATA_NAME.tar.gz -C test/data/
	fi

	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_torchvision.py --data=$DATA_PATH --use-cuda
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_torchaudio.py

	echo -e "done testing container $1 => PyTorch\n"
}

# TensorFlow tests
test_tensorflow()
{
	echo "testing container $1 => TensorFlow"
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_tensorflow.py
	echo -e "done testing container $1 => TensorFlow\n"
}

# TensorRT tests
test_tensorrt()
{
	echo "testing container $1 => TensorRT"
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_tensorrt.py
	echo -e "done testing container $1 => TensorRT\n"
}

# scipy tests
test_scipy()
{
	echo "testing container $1 => scipy"
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_scipy.py
	echo -e "done testing container $1 => scipy\n"
}

# sklearn tests
test_sklearn()
{
	echo "testing container $1 => sklearn"
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_sklearn.py
	echo -e "done testing container $1 => sklearn\n"
}

# PyTorch tests (all)
test_pytorch_all()
{
	test_pytorch $1
	test_tensorrt $1
	test_cuda $1
	test_numpy $1
}

# TensorFlow tests (all)
test_tensorflow_all()
{
	test_tensorflow $1
	test_tensorrt $1
	test_cuda $1
	test_numpy $1
}

# ML tests (all)
test_all()
{
	test_pytorch $1
	test_tensorflow $1
	test_tensorrt $1
	test_cuda $1
	test_numpy $1
	test_cupy $1
	test_numba $1
	test_onnx $1
	test_opencv $1
	test_pandas $1
	test_scipy $1
	test_sklearn $1
}


#
# PyTorch container
#
if [[ "$CONTAINERS" == "pytorch" || "$CONTAINERS" == "all" ]]; then
	#test_pytorch_all "l4t-pytorch:r$L4T_VERSION-pth1.2-py3"
	#test_pytorch_all "l4t-pytorch:r$L4T_VERSION-pth1.3-py3"
	#test_pytorch_all "l4t-pytorch:r$L4T_VERSION-pth1.4-py3"
	#test_pytorch_all "l4t-pytorch:r$L4T_VERSION-pth1.5-py3"
	test_pytorch_all "l4t-pytorch:r$L4T_VERSION-pth1.6-py3"
	test_pytorch_all "l4t-pytorch:r$L4T_VERSION-pth1.7-py3"
fi

#
# TensorFlow container
#
if [[ "$CONTAINERS" == "tensorflow" || "$CONTAINERS" == "all" ]]; then
	test_tensorflow_all "l4t-tensorflow:r$L4T_VERSION-tf1.15-py3"
	test_tensorflow_all "l4t-tensorflow:r$L4T_VERSION-tf2.3-py3"
fi

#
# ML container
#
if [[ "$CONTAINERS" == "ml" || "$CONTAINERS" == "all" ]]; then
	test_all "l4t-ml:r$L4T_VERSION-py3"
fi


