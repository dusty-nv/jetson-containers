#!/usr/bin/env bash

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
TEST_MOUNT="$ROOT/../test:/test"


# numpy tests
test_numpy()
{
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_numpy.py
}

# PyTorch tests
test_pytorch()
{
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_pytorch.py
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_torchvision.py
}

# TensorFlow tests
test_tensorflow()
{
	sh ./scripts/docker_run.sh -c $1 -v $TEST_MOUNT -r python3 test/test_tensorflow.py
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


