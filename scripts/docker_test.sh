#!/usr/bin/env bash

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
TEST_MOUNT="$ROOT/../test:/test"


# numpy tests
test_numpy()
{
	echo "testing container $1 => numpy"
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_numpy.py
	echo -e "done testing container $1 => numpy\n"
}

# onnx tests
test_onnx()
{
	echo "testing container $1 => onnx"
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_onnx.py
	echo -e "done testing container $1 => onnx\n"
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
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_torchvision.py
	
	echo -e "done testing container $1 => PyTorch\n"
}

# TensorFlow tests
test_tensorflow()
{
	echo "testing container $1 => TensorFlow"
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_tensorflow.py
	echo -e "done testing container $1 => TensorFlow\n"
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

	
#
# PyTorch container
#
PYTORCH_CONTAINER="l4t-pytorch:r32.4-pth1.2-py3"

test_pytorch $PYTORCH_CONTAINER
test_numpy $PYTORCH_CONTAINER

#
# TensorFlow container
#
TENSORFLOW_CONTAINER="l4t-tensorflow:r32.4-tf1.15-py3"

test_tensorflow $TENSORFLOW_CONTAINER
test_numpy $TENSORFLOW_CONTAINER

#
# ML container
#
ML_CONTAINER="l4t-ml:r32.4-py3"

test_pytorch $ML_CONTAINER
test_tensorflow $ML_CONTAINER
test_numpy $ML_CONTAINER
test_onnx $ML_CONTAINER
test_pandas $ML_CONTAINER
test_scipy $ML_CONTAINER
test_sklearn $ML_CONTAINER

